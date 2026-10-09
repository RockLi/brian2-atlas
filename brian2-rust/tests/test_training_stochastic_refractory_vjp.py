"""Independent SDE, event-bin and refractory recurrences; no native SSA evaluation."""
import copy
import os
import struct
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer
from test_native_training import RUNNER
from test_training_integer_ir import engine
from test_training_linked import cython_cache
from test_training_stochastic import normal
from test_training_stochastic_refractory_combined import model, VARIANTS
from test_training_delay_update import snapshot
from test_training_event_delay_gradients import routing


def reference(bundle,x,variant,kind,weights=None,initial=None,anchors=None,sequence=17,start_tick=0,plan=None,batch=0):
    p=bundle.plan if plan is None else plan;d=p['dynamic'];weights=bundle.weights if weights is None else weights
    z=np.asarray(bundle.initial_state if initial is None else initial,float).copy()
    if initial is None:
        for k,ref in enumerate(d['initial_parameters']):
            if ref is not None:z[k]=weights[ref[0]][ref[1]]
    def param(name):return np.asarray(weights[next(e['bank'] for e in bundle.provenance['bindings'] if e['object']=='combo_group' and e['variables']==[name])])
    drive=param('drive');sigma=param('sigma');control=param('limit' if kind=='boolean' else 'shift')
    layout=bundle.provenance['neuron_state_layout']['combo_group'];v=layout['v'];u=layout['u'];r=layout['r'];q=layout['q']
    hv=bundle.provenance['neuron_state_layout']['combo_hidden']['v'];syn=bundle.provenance['dynamic_state_layout']['combo_syn'];w=syn['w'];ap=syn['ap'];last=syn['lastupdate'];edge_z=syn['z'];rho=np.asarray(weights[next(e['bank'] for e in bundle.provenance['bindings'] if e['object']=='combo_syn' and e['variables']==['rho'])])
    flag=bundle.provenance['refractory_activity_layout']['combo_group'];age=bundle.provenance['refractory_age_layout']['combo_group'];lastspike=bundle.provenance['refractory_lastspike_layout']['combo_group'];words=bundle.provenance['refractory_timestamp_words']['combo_group']
    paths=d.get('delay_layout',{}).get('paths',[]);bins=[{},{}]
    if paths:
        latched,histories=routing(p,z) if anchors is None else (anchors['latched'],anchors['histories'])
        delay=[[e['delay_state'] for e in path['edges']] for path in paths]
        for path,edge,slots in histories:
            for offset,slot in enumerate(slots):
                hard=(z[slot]!=0) if anchors is None else anchors['initial'][slot]!=0
                bins[path].setdefault(offset,[]).append((edge,z[slot],hard))
    else:latched=[[0,0],[0,0]];histories=[];delay=None
    initial_z=z.copy();before=[];gates=[];margins=[];hard_spikes=[];spikes=[]
    dt=.0002;streams=bundle.provenance['noise_names'][1]
    for tick,external in enumerate(x):
        if anchors is not None and p['tbptt_window'] and tick and tick%p['tbptt_window']==0:z,bins=copy.deepcopy(anchors['before'][tick])
        before.append(copy.deepcopy((z,bins)));time=(round(p['clock']['origin']/dt)+start_tick+tick)*dt
        z[q]=z[w]+.2*z[w]**2+.1*z[edge_z]
        free=z[flag].astype(bool)|(z[u]<=control) if kind=='boolean' else z[age]>=np.trunc((z[r]+control+.001*dt)/dt)
        if anchors is not None:free=anchors['gates'][tick]
        gates.append(free.copy());z[hv]*=.8
        state=np.array([z[v],z[u]]);drift=np.array([(drive-state[0]+.15*state[1]+.1*z[q])/ .001,-state[1]/.001]);drift[0]*=free
        updated=state+dt*drift
        for stream,name in enumerate(streams):
            dw=np.sqrt(dt)*np.array([normal(p['seed'],sequence,batch,1,j,start_tick+tick,stream) for j in range(2)])
            def diffusion(value):
                if name=='xi_extra':g=sigma/np.sqrt(.001)*np.array([.2*(value[1]+1),.1*(value[0]+1)])
                else:
                    mask=np.array([name in ('xi_v','xi_common'),name in ('xi_u','xi_common')])[:,None]
                    g=mask*sigma/np.sqrt(.001)*np.array([value[0]+1,.5*(value[1]+1)])
                g[0]*=free;return g
            base=diffusion(state)
            if variant.startswith('heun'):updated+=.5*dw*(base+diffusion(state+base*dw))
            else:updated+=base*dw+(diffusion(state+dt*drift+np.sqrt(dt)*base)-base)*dw**2/(2*np.sqrt(dt))
        z[v]=np.where(free,updated[0],state[0]);z[u]=updated[1];z[r]*=.9;z[age]=np.minimum(z[age],2**31-2)+1
        old_z=z[edge_z].copy();f=(-old_z+.2*z[w])/.001;g0=rho*(1+old_z)/np.sqrt(.001)
        domain=bundle.provenance['synaptic_noise_domains']['combo_syn']
        dw=np.sqrt(dt)*np.array([normal(p['seed'],sequence,batch,domain,j,start_tick+tick,0) for j in range(2)])
        def edge_diffusion(value):return rho*(1+value)/np.sqrt(.001)
        if variant.startswith('heun'):z[edge_z]=old_z+dt*f+.5*dw*(g0+edge_diffusion(old_z+g0*dw))
        else:z[edge_z]=old_z+dt*f+g0*dw+(edge_diffusion(old_z+dt*f+np.sqrt(dt)*g0)-g0)*dw**2/(2*np.sqrt(dt))
        margin=z[v]-.6;hard=((margin>0)&free).astype(float);soft=hard.copy()
        if anchors is not None:
            hard=anchors['hard'][tick];old=anchors['margins'][tick]
            soft=hard+free*p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(old))**2*(margin-old)
        margins.append(margin.copy());hard_spikes.append(hard.copy());spikes.append(soft.copy());z[flag]=free*(1-hard)
        for j in range(2):
            if hard[j]:
                z[age[j]]=1;z[lastspike[j]]=time
                low,high=struct.unpack('<ii',struct.pack('<d',time));z[words[0][j]]=low;z[words[1][j]]=high
        for path in range(2):
            emitted=external if path==0 else soft;actual=external if path==0 else hard
            for j in range(2):bins[path].setdefault(tick+latched[path][j],[]).append((j,emitted[j],actual[j]!=0))
            for j,amplitude,event in bins[path].pop(tick,[]):
                a=z[ap[j]];value=a*np.exp(-(time-z[last[j]])/.002);weight=z[w[j]]
                if path==0:
                    if z[flag[j]]:z[v[j]]+=amplitude*(weight+.03*value+.02*z[edge_z[j]])
                    z[r[j]]+=amplitude*.0001;value+=.2;weight+=.01*value+.002*z[edge_z[j]]
                else:value+=.1;weight-=.002*z[u[j]]+.001*z[edge_z[j]]
                if delay is not None:
                    slot=delay[path][j];factor=.6 if path==0 else .7;gain=.00012 if path==0 else .00006
                    z[slot]+=amplitude*((factor-1)*z[slot]+gain*weight)
                z[ap[j]]+=amplitude*(value-a);z[w[j]]+=amplitude*(weight-z[w[j]])
                if event:z[last[j]]=time
        reset=hard if p['detach_reset'] else soft
        z[v]-=.4*reset;z[u]+=.8*reset;z[r]+=.0004*reset
    # Persist each current-route FIFO. The forward model starts with one route per edge.
    if paths:
        for path,pth in enumerate(paths):
            for j,e in enumerate(pth['edges']):
                for offset,slot in enumerate(e['states']):z[slot]=sum(value for edge,value,_ in bins[path].get(len(x)+offset,[]) if edge==j)
    logits=np.asarray(spikes).mean(0)*p['logit_scale'];loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
    return loss,z,np.asarray(spikes),dict(initial=initial_z,before=before,gates=gates,margins=margins,hard=hard_spikes,latched=latched,histories=histories)


@pytest.mark.parametrize('variant',VARIANTS)
@pytest.mark.parametrize('kind',['boolean','duration'])
@pytest.mark.parametrize('window',[None,3])
def test_combined_all_floating_pathwise_derivatives(engine,variant,kind,window):
    *_,x,bundle=model(variant,kind,delayed=True,backend=engine,detach_reset=False,tbptt_window=window)
    out=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights).gradients(x[None],[0],noise_sequence=17)
    loss,z,spikes,anchors=reference(bundle,x,variant,kind)
    tol=2e-6 if engine=='cpu' else 2e-3;absolute=2e-7 if engine=='cpu' else 2e-5
    assert out['loss']==pytest.approx(loss,abs=absolute)
    np.testing.assert_array_equal(np.asarray(out['spikes'])[0,:,2:],spikes)
    histories={k for path in bundle.plan['dynamic']['delay_layout']['paths'] for edge in [*path['pending'],*path['edges']] for k in edge['states']}
    physical=[k for k in range(len(z)) if k not in histories]
    np.testing.assert_allclose(np.asarray(out['final_state'])[0,physical],z[physical],rtol=tol,atol=absolute)
    assert len(out['initial_state_gradients'][0])==len(bundle.initial_state)
    for binding in bundle.provenance['bindings']:
        if binding['variables'] in (['sigma'],['rho']):
            assert max(abs(g) for g in out['gradients'][binding['bank']])>1e-8
    for bank,row in enumerate(bundle.weights):
        for k,value in enumerate(row):
            eps=1e-6*max(abs(value),.001);hi=copy.deepcopy(bundle.weights);lo=copy.deepcopy(bundle.weights);hi[bank][k]+=eps;lo[bank][k]-=eps
            fd=(reference(bundle,x,variant,kind,weights=hi,anchors=anchors)[0]-reference(bundle,x,variant,kind,weights=lo,anchors=anchors)[0])/(2*eps)
            assert out['gradients'][bank][k]==pytest.approx(fd,rel=tol,abs=absolute),(bank,k)
    temporal=set(bundle.provenance['neuron_state_layout']['combo_group']['r'])|{k for row in bundle.provenance['pathway_state_layout'].values() for k in row['delay']}
    for k,value in enumerate(bundle.initial_state):
        if bundle.plan['dynamic']['detached'][k]:assert out['initial_state_gradients'][0][k]==0.;continue
        eps=1e-9 if k in temporal else 1e-6;hi=np.asarray(bundle.initial_state).copy();lo=hi.copy();hi[k]+=eps;lo[k]-=eps
        fd=(reference(bundle,x,variant,kind,initial=hi,anchors=anchors)[0]-reference(bundle,x,variant,kind,initial=lo,anchors=anchors)[0])/(2*eps)
        assert out['initial_state_gradients'][0][k]==pytest.approx(fd,rel=tol,abs=absolute),(k,fd,out['initial_state_gradients'][0][k])


@pytest.mark.parametrize('variant,kind',[('heun_mixed','boolean'),('milstein','duration')])
@pytest.mark.parametrize('ranks',[None,2,8])
def test_combined_optimizer_queues_mpi_and_atomic_failure(engine,variant,kind,ranks,tmp_path):
    if ranks is not None and os.environ.get('B2_TEST_MPI')!='1':pytest.skip('local MPI required')
    *_,x,bundle=model(variant,kind,delayed=True,backend=engine,mpi_ranks=ranks,detach_reset=False,tbptt_window=2,learning_rate=1e-8)
    p=bundle.plan;q=copy.deepcopy(p);q.update(backend='cpu',mpi_ranks=None)
    trainer=NativeLIFTrainer(p,runner=RUNNER,weights=bundle.weights);cpu=NativeLIFTrainer(q,runner=RUNNER,weights=bundle.weights)
    xx=np.stack([x,x[:,::-1]]);cursor=0;grew=False
    for length in (3,2,5):
        chunk=xx[:,cursor:cursor+length];kw=dict(noise_sequence=17) if cursor==0 else dict(initial='carry')
        actual=trainer.step(chunk,[0,1],**kw);expected=cpu.step(chunk,[0,1],**kw);cursor+=length
        for key in ('final_state','spikes','initial_state_gradients','logits'):
            np.testing.assert_allclose(actual[key],expected[key],rtol=2e-3,atol=2e-5,err_msg=key)
        for a,c in zip(actual['gradients'],expected['gradients']):np.testing.assert_allclose(a,c,rtol=2e-3,atol=2e-5)
        if engine!='cpu':assert actual['gpu_dispatches']>0
        if ranks is not None:assert 'mpi-dynamic' in actual['numeric_profile']
        grew|=len(actual['final_state'][0])>len(bundle.initial_state)
        saved=tmp_path/'combined-optimizer.json';trainer.store(saved)
        restored=NativeLIFTrainer(trainer.plan,runner=RUNNER);restored.restore(saved);trainer=restored
        assert trainer.clock_tick==cpu.clock_tick==cursor and trainer.noise_sequence==17
    assert grew and any(a!=c for a,c in zip(trainer.state['weights'],bundle.weights))
    before=snapshot(trainer);bad=np.asarray(trainer.neuron_state).copy()
    bad[1,bundle.provenance['dynamic_state_layout']['combo_syn']['lastupdate'][1]]=1e6
    with pytest.raises(ValueError,match='nonfinite|domain'):trainer.step(np.ones((2,4,2)),[0,1],initial=bad,start_tick=cursor,noise_sequence=17)
    assert snapshot(trainer)==before
