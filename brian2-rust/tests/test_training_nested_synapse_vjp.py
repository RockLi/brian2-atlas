"""Independent frozen-routing VJPs for three-level synaptic bank access."""
import copy
import os
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer
from test_native_training import RUNNER
from test_training_linked import cython_cache
from test_training_integer_ir import engine
from test_training_stochastic import normal
from test_training_event_delay_gradients import routing as mutable_routing
from test_training_nested_synapse import model,SOURCES,TARGETS,BANK_MAP
from test_training_delay_update import snapshot


def routing(plan,initial):
    pre=copy.deepcopy(plan);pre['dynamic']['delay_layout']['paths']=pre['dynamic']['delay_layout']['paths'][:1]
    delays,histories=mutable_routing(pre,initial)
    post=plan['dynamic']['delay_layout']['paths'][1]
    delays.append([len(e['states']) for e in post['edges']])
    histories.extend((1,edge,e['states']) for edge,e in enumerate(post['edges']))
    histories.extend((1,e['edge'],e['states']) for e in post['pending'])
    return delays,histories


def oracle(bundle,x,mutation,weights=None,initial=None,anchors=None,sequence=9,start_tick=0,batch=0):
    p=bundle.plan;d=p['dynamic'];weights=bundle.weights if weights is None else weights
    z=np.asarray(bundle.initial_state if initial is None else initial,float).copy()
    if initial is None:
        for k,ref in enumerate(d['initial_parameters']):
            if ref is not None:z[k]=weights[ref[0]][ref[1]]
    def bank(name):return next(e['bank'] for e in bundle.provenance['bindings'] if e['object']=='nested_pool' and e['variables']==[name])
    gain=np.asarray(weights[bank('gain')]);h=weights[bank('h')][0]
    nl=bundle.provenance['neuron_state_layout'];pv=nl['nested_pool']['v'];route=nl['nested_pool']['route'];v=nl['nested_output']['v'];q=nl['nested_output']['q']
    sl=bundle.provenance['dynamic_state_layout']['nested_syn'];pick=sl['pick'];w=sl['w']
    paths=d['delay_layout']['paths'];latched,histories=routing(p,z) if anchors is None else (anchors['latched'],anchors['histories'])
    bins=[{},{}];delay=[e['delay_state'] for e in paths[0]['edges']]
    for path,edge,slots in histories:
        for offset,slot in enumerate(slots):
            actual=z[slot]!=0 if anchors is None else anchors['initial'][slot]!=0
            bins[path].setdefault(offset,[]).append((edge,z[slot],actual))
    initial_z=z.copy();before=[];margins=[];hard=[];spikes=[]
    def coefficient(edge):return gain[BANK_MAP[int(z[route[int(z[pick[edge]])]])]]
    for tick,external in enumerate(x):
        if anchors is not None and p['tbptt_window'] and tick and tick%p['tbptt_window']==0:z,bins=copy.deepcopy(anchors['before'][tick])
        before.append(copy.deepcopy((z,bins)));z[q]=0
        for edge,j in enumerate(TARGETS):z[q[j]]+=z[w[edge]]+.1*coefficient(edge)
        z[pv]*=.8;z[v]=.8*z[v]+.14+.04*z[q]
        if p.get('noise_streams'):z[v]+=.03*(z[q]+1)*np.sqrt(.2)*np.array([normal(p['seed'],sequence,batch,1,j,start_tick+tick,0) for j in range(2)])
        for edge in range(4):
            coef=coefficient(edge);z[w[edge]]=.8*z[w[edge]]+.01*coef
            if p.get('noise_streams'):z[w[edge]]+=.02*(coef+1)*np.sqrt(.2)*normal(p['seed'],sequence,batch,2,edge,start_tick+tick,0)
        margin=z[v]-.65;hard_event=(margin>0).astype(float);soft=hard_event.copy()
        if anchors is not None:
            hard_event=anchors['hard'][tick];old=anchors['margins'][tick]
            soft=hard_event+p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(old))**2*(margin-old)
        margins.append(margin.copy());hard.append(hard_event.copy());spikes.append(soft.copy())
        for path in range(2):
            indices=SOURCES if path==0 else TARGETS
            # Earlier emissions already occupy each bin. Brian appends new events
            # by source neuron then original edge order.
            for edge in sorted(range(4),key=lambda e:(indices[e],e)):
                source=indices[edge];amplitude=external[source] if path==0 else soft[source];actual=external[source] if path==0 else hard_event[source]
                bins[path].setdefault(tick+latched[path][edge],[]).append((edge,amplitude,actual!=0))
            for edge,amplitude,actual in bins[path].pop(tick,[]):
                coef=coefficient(edge)
                if path==0:
                    old=int(z[pick[edge]]);old_route=int(z[route[old]])
                    dest=(old+1)%5 if mutation in ('pick','both') else old
                    if actual:
                        z[pick[edge]]=dest
                        if mutation in ('route','both'):z[route[dest]]=(old_route+1)%3
                    z[v[TARGETS[edge]]]+=amplitude*(.12*coef+.1*h+z[w[edge]])
                    z[w[edge]]+=amplitude*.01*coef
                    z[delay[edge]]+=amplitude*(-.4*z[delay[edge]]+.0001*coef)
                else:z[w[edge]]-=amplitude*.005*coef
        z[v]-=.4*(hard_event if p['detach_reset'] else soft)
    logits=np.asarray(spikes).mean(0)*p['logit_scale'];loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
    return loss,z,np.asarray(spikes),dict(initial=initial_z,before=before,margins=margins,hard=hard,latched=latched,histories=histories)


@pytest.mark.parametrize('mutation',['pick','route','both'])
@pytest.mark.parametrize('noisy',[False,True])
@pytest.mark.parametrize('window',[None,3])
def test_nested_synapse_all_float_and_integer_derivatives(engine,mutation,noisy,window):
    *_,x,bundle=model(mutation,noisy=noisy,delayed=True,backend=engine,detach_reset=False,tbptt_window=window)
    out=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights).gradients(x[None],[0],**(dict(noise_sequence=9) if noisy else {}))
    loss,z,spikes,anchors=oracle(bundle,x,mutation)
    tol=2e-5 if engine=='cpu' else 2e-3;absolute=3e-7 if engine=='cpu' else 2e-5
    assert out['loss']==pytest.approx(loss,abs=absolute)
    np.testing.assert_array_equal(np.asarray(out['spikes'])[0,:,5:],spikes)
    queue={k for path in bundle.plan['dynamic']['delay_layout']['paths'] for e in [*path['edges'],*path['pending']] for k in e['states']}
    physical=[k for k in range(len(z)) if k not in queue]
    np.testing.assert_allclose(np.asarray(out['final_state'])[0,physical],z[physical],rtol=tol,atol=absolute)
    integers={bank for bank,index in bundle.plan['dynamic']['integer_parameters']}
    assert any(e['kind']=='parameter_index_table' and bundle.weights[e['bank']]==BANK_MAP.tolist() for e in bundle.provenance['bindings'])
    for bank,row in enumerate(bundle.weights):
        if bank in integers:np.testing.assert_array_equal(out['gradients'][bank],0);continue
        for k,value in enumerate(row):
            eps=1e-6;hi=copy.deepcopy(bundle.weights);lo=copy.deepcopy(bundle.weights);hi[bank][k]+=eps;lo[bank][k]-=eps
            fd=(oracle(bundle,x,mutation,weights=hi,anchors=anchors)[0]-oracle(bundle,x,mutation,weights=lo,anchors=anchors)[0])/(2*eps)
            assert out['gradients'][bank][k]==pytest.approx(fd,rel=tol,abs=absolute),(bank,k,fd)
    time_slots={k for row in bundle.provenance['pathway_state_layout'].values() for k in row['delay']}
    assert len(out['initial_state_gradients'][0])==len(bundle.initial_state)
    for k,value in enumerate(bundle.initial_state):
        if bundle.plan['dynamic']['detached'][k]:assert out['initial_state_gradients'][0][k]==0.;continue
        eps=1e-9 if k in time_slots else 1e-6;hi=np.asarray(bundle.initial_state).copy();lo=hi.copy();hi[k]+=eps;lo[k]-=eps
        fd=(oracle(bundle,x,mutation,initial=hi,anchors=anchors)[0]-oracle(bundle,x,mutation,initial=lo,anchors=anchors)[0])/(2*eps)
        assert out['initial_state_gradients'][0][k]==pytest.approx(fd,rel=tol,abs=absolute),(k,fd,out['initial_state_gradients'][0][k])


@pytest.mark.parametrize('ranks',[None,2,8])
def test_nested_routes_optimizer_carry_mpi_rollback(engine,ranks,tmp_path):
    if ranks is not None and os.environ.get('B2_TEST_MPI')!='1':pytest.skip('local MPI required')
    *_,x,bundle=model('both',noisy=True,delayed=True,backend=engine,mpi_ranks=ranks,detach_reset=False,tbptt_window=2,learning_rate=1e-8)
    p=bundle.plan;q=copy.deepcopy(p);q.update(backend='cpu',mpi_ranks=None)
    trainer=NativeLIFTrainer(p,runner=RUNNER,weights=bundle.weights);cpu=NativeLIFTrainer(q,runner=RUNNER,weights=bundle.weights)
    xx=np.stack([x,x[:,::-1]]);cursor=0;route=bundle.provenance['neuron_state_layout']['nested_pool']['route'];pick=bundle.provenance['dynamic_state_layout']['nested_syn']['pick']
    for length in (3,2,3):
        kw=dict(noise_sequence=9) if cursor==0 else dict(initial='carry');chunk=xx[:,cursor:cursor+length]
        actual=trainer.step(chunk,[0,1],**kw);expected=cpu.step(chunk,[0,1],**kw);cursor+=length
        for key in ('final_state','spikes','initial_state_gradients','logits'):np.testing.assert_allclose(actual[key],expected[key],rtol=2e-3,atol=2e-5)
        np.testing.assert_array_equal(np.asarray(actual['final_state'])[:,route+pick],np.asarray(expected['final_state'])[:,route+pick])
        for a,c in zip(actual['gradients'],expected['gradients']):np.testing.assert_allclose(a,c,rtol=2e-3,atol=2e-5)
        if engine!='cpu':assert actual['gpu_dispatches']>0
        if ranks is not None:assert 'mpi-dynamic' in actual['numeric_profile']
        saved=tmp_path/'nested.json';trainer.store(saved);restored=NativeLIFTrainer(trainer.plan,runner=RUNNER);restored.restore(saved);trainer=restored
    assert any(a!=c for a,c in zip(trainer.state['weights'],bundle.weights))
    before=snapshot(trainer);bad=np.asarray(trainer.neuron_state).copy();bad[1,route]=3
    with pytest.raises(ValueError,match='runtime index|parameter gather|nonfinite|domain'):
        trainer.step(xx,[0,1],initial=bad,start_tick=cursor,noise_sequence=9)
    assert snapshot(trainer)==before
