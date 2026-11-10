"""Subgroup gradients from independent physical equations and event bins."""
import copy
import os
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer
from test_native_training import RUNNER
from test_training_linked import cython_cache
from test_training_integer_ir import engine
from test_training_stochastic import normal
from test_training_event_delay_gradients import routing
from test_training_subgroup import model


def oracle(bundle,x,layout,weights=None,initial=None,anchors=None,sequence=11,batch=0,start_tick=0):
    p=bundle.plan;d=p['dynamic'];weights=bundle.weights if weights is None else weights
    z=np.asarray(bundle.initial_state if initial is None else initial,float).copy()
    if initial is None:
        for k,ref in enumerate(d['initial_parameters']):
            if ref is not None:z[k]=weights[ref[0]][ref[1]]
    nl=bundle.provenance['neuron_state_layout'];v=[nl[f'view_g{k}']['v'] for k in range(2)];q=[nl[f'view_g{k}']['q'] for k in range(2)]
    sl=[bundle.provenance['dynamic_state_layout'][f'view_s{k}'] for k in range(2)]
    gain=[np.asarray(weights[next(e['bank'] for e in bundle.provenance['bindings'] if e['object']==f'view_g{k}' and e['variables']==['gain'])]) for k in range(2)]
    restricted_source=layout in ('source','both');restricted_target=layout in ('target','both')
    source_start=1 if restricted_source else 0;target_start=[2,1] if restricted_target else [0,0]
    ni=[3,4] if restricted_source else [5,6];nj=3 if restricted_target else 6
    ii=[np.array([2,0,1,2]),np.array([3,1,0,2])];jj=np.array([1,2,0,2])
    source=[row+source_start for row in ii];target=[jj+start for start in target_start]
    path_order=[(0,True),(1,True),(0,False),(1,False)]
    paths=d['delay_layout']['paths']
    assert [path['name'] for path in paths]==[f'view_s{k}_{"pre" if pre else "post"}' for k,pre in path_order]
    latched,histories=routing(p,z) if anchors is None else (anchors['latched'],anchors['histories']);bins=[{} for _ in paths]
    for path,edge,slots in histories:
        for offset,slot in enumerate(slots):
            hard=z[slot]!=0 if anchors is None else anchors['initial'][slot]!=0
            bins[path].setdefault(offset,[]).append((edge,z[slot],hard))
    delay=[[e['delay_state'] for e in path['edges']] for path in paths]
    initial_z=z.copy();before=[];margins=[];hard=[];spikes=[];dt=.0002
    def summed(k):
        z[q[k][target_start[k]:target_start[k]+nj]]=0
        for edge,j in enumerate(target[k]):z[q[k][j]]+=z[sl[k]['w'][edge]]+.1*z[sl[k]['z'][edge]]+.01*gain[k][j]+.002*ii[k][edge]+.003*jj[edge]
    for tick,external in enumerate(x):
        if anchors is not None and p['tbptt_window'] and tick and tick%p['tbptt_window']==0:z,bins=copy.deepcopy(anchors['before'][tick])
        before.append(copy.deepcopy((z,bins)));time=(round(p['clock']['origin']/dt)+start_tick+tick)*dt
        if not restricted_target:
            for k in range(2):summed(k)
        for k in range(2):
            old=z[v[k]].copy();z[v[k]]=old+.2*(.7-old+.1*z[q[k]])
            if p.get('noise_streams'):
                dw=np.sqrt(.2)*np.array([normal(p['seed'],sequence,batch,k,j,start_tick+tick,0) for j in range(6)])
                g0=.03*(old+1);z[v[k]]+=.5*dw*(g0+.03*(old+g0*dw+1))
        for k in range(2):
            indices=sl[k]['z'];old=z[indices].copy();z[indices]=old+.2*(-old+.1*z[sl[k]['w']]+.01*ii[k])
            if p.get('noise_streams'):
                dw=np.sqrt(.2)*np.array([normal(p['seed'],sequence,batch,k+2,j,start_tick+tick,0) for j in range(4)])
                g0=.02*(old+1);z[indices]+=.5*dw*(g0+.02*(old+g0*dw+1))
            # Subgroup.order == parent.order+1, so its summed runner has
            # order 0 rather than -1. In this named model, it follows the
            # neuron runners and its own synaptic state updater.
            if restricted_target:summed(k)
        margin=np.array([z[slots]-.6 for slots in v]);hard_event=(margin>0).astype(float);soft=hard_event.copy()
        if anchors is not None:
            hard_event=anchors['hard'][tick];old=anchors['margins'][tick]
            soft=hard_event+p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(old))**2*(margin-old)
        margins.append(margin.copy());hard.append(hard_event.copy());spikes.append(soft.copy())
        for path,(k,pre) in enumerate(path_order):
            index=source[k] if pre else target[k]
            emitted=(external if k==0 else soft[0]) if pre else soft[k]
            actual=(external if k==0 else hard_event[0]) if pre else hard_event[k]
            for edge in sorted(range(4),key=lambda e:(index[e],e)):
                j=index[edge];bins[path].setdefault(tick+latched[path][edge],[]).append((edge,emitted[j],actual[j]!=0))
            for edge,amplitude,event in bins[path].pop(tick,[]):
                s=sl[k];a=z[s['a'][edge]];value=a*np.exp(-(time-z[s['lastupdate'][edge]])/.002);weight=z[s['w'][edge]];j=target[k][edge]
                if pre:
                    z[v[k][j]]+=amplitude*(weight+.01*z[s['z'][edge]]+.02*gain[k][j]+.001*ii[k][edge]+.002*jj[edge]+.003*ni[k]+.004*nj)
                    value+=.1;weight+=.01*value
                    if k:z[v[0][source[k][edge]]]+=amplitude*.01*gain[0][source[k][edge]]
                else:value+=.05;weight-=.002*gain[k][j]
                slot=delay[path][edge];z[slot]+=amplitude*((-.4 if pre else -.3)*z[slot]+(.0001 if pre else .00004)*weight)
                z[s['a'][edge]]+=amplitude*(value-a);z[s['w'][edge]]+=amplitude*(weight-z[s['w'][edge]])
                if event:z[s['lastupdate'][edge]]=time
        for k in range(2):z[v[k]]-=.4*(hard_event[k] if p['detach_reset'] else soft[k])
    spikes=np.asarray(spikes);logits=spikes[:,1].mean(0)*p['logit_scale'];loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
    return loss,z,spikes.reshape(len(x),12),dict(initial=initial_z,before=before,margins=margins,hard=hard,latched=latched,histories=histories)


@pytest.mark.parametrize('layout',['source','target','both'])
@pytest.mark.parametrize('noisy',[False,True])
@pytest.mark.parametrize('window',[None,3])
def test_subgroup_every_floating_derivative(engine,layout,noisy,window):
    *_,x,bundle=model(layout,noisy=noisy,delayed=True,backend=engine,detach_reset=False,tbptt_window=window)
    out=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights).gradients(x[None],[0],**(dict(noise_sequence=11) if noisy else {}))
    loss,z,spikes,anchors=oracle(bundle,x,layout)
    tol=2e-5 if engine=='cpu' else 2e-3;absolute=3e-7 if engine=='cpu' else 2e-5
    assert out['loss']==pytest.approx(loss,abs=absolute);np.testing.assert_array_equal(out['spikes'][0],spikes)
    queue={k for path in bundle.plan['dynamic']['delay_layout']['paths'] for e in [*path['edges'],*path['pending']] for k in e['states']}
    physical=[k for k in range(len(z)) if k not in queue];np.testing.assert_allclose(np.asarray(out['final_state'])[0,physical],z[physical],rtol=tol,atol=absolute)
    for bank,row in enumerate(bundle.weights):
        for k,value in enumerate(row):
            eps=1e-6;hi=copy.deepcopy(bundle.weights);lo=copy.deepcopy(bundle.weights);hi[bank][k]+=eps;lo[bank][k]-=eps
            fd=(oracle(bundle,x,layout,weights=hi,anchors=anchors)[0]-oracle(bundle,x,layout,weights=lo,anchors=anchors)[0])/(2*eps)
            assert out['gradients'][bank][k]==pytest.approx(fd,rel=tol,abs=absolute),(bank,k,fd)
    for binding in bundle.provenance['bindings']:
        if binding['variables']==['gain']:assert max(abs(g) for g in out['gradients'][binding['bank']])>1e-8
    temporal={k for row in bundle.provenance['pathway_state_layout'].values() for k in row['delay']}
    assert len(out['initial_state_gradients'][0])==len(bundle.initial_state)
    for k,value in enumerate(bundle.initial_state):
        if bundle.plan['dynamic']['detached'][k]:assert out['initial_state_gradients'][0][k]==0.;continue
        eps=1e-9 if k in temporal else 1e-6;hi=np.asarray(bundle.initial_state).copy();lo=hi.copy();hi[k]+=eps;lo[k]-=eps
        fd=(oracle(bundle,x,layout,initial=hi,anchors=anchors)[0]-oracle(bundle,x,layout,initial=lo,anchors=anchors)[0])/(2*eps)
        assert out['initial_state_gradients'][0][k]==pytest.approx(fd,rel=tol,abs=absolute),(k,fd,out['initial_state_gradients'][0][k])


@pytest.mark.parametrize('layout',['source','target','both'])
@pytest.mark.parametrize('ranks',[None,2,8])
def test_subgroup_optimizer_carry_mpi(engine,layout,ranks,tmp_path):
    if ranks is not None and os.environ.get('B2_TEST_MPI')!='1':pytest.skip('local MPI required')
    *_,x,bundle=model(layout,noisy=True,delayed=True,refractory=True,backend=engine,mpi_ranks=ranks,detach_reset=False,tbptt_window=2,learning_rate=1e-8)
    p=bundle.plan;q=copy.deepcopy(p);q.update(backend='cpu',mpi_ranks=None)
    trainer=NativeLIFTrainer(p,runner=RUNNER,weights=bundle.weights);cpu=NativeLIFTrainer(q,runner=RUNNER,weights=bundle.weights)
    xx=np.stack([x,x[:,::-1]]);cursor=0
    for length in (3,2,3):
        kw=dict(noise_sequence=11) if cursor==0 else dict(initial='carry');chunk=xx[:,cursor:cursor+length]
        actual=trainer.step(chunk,[0,1],**kw);expected=cpu.step(chunk,[0,1],**kw);cursor+=length
        for key in ('final_state','spikes','initial_state_gradients','logits'):np.testing.assert_allclose(actual[key],expected[key],rtol=2e-3,atol=2e-5)
        for a,c in zip(actual['gradients'],expected['gradients']):np.testing.assert_allclose(a,c,rtol=2e-3,atol=2e-5)
        if engine!='cpu':assert actual['gpu_dispatches']>0
        if ranks is not None:assert 'mpi-dynamic' in actual['numeric_profile']
        saved=tmp_path/'subgroup.json';trainer.store(saved);restored=NativeLIFTrainer(trainer.plan,runner=RUNNER);restored.restore(saved);trainer=restored
    assert any(a!=c for a,c in zip(trainer.state['weights'],bundle.weights))
