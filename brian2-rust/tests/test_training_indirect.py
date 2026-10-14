"""Native runtime addressing: independent recurrence, VJP and transaction checks."""
import copy
import os
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer
from brian2_rust.training import lif_training_plan
from brian2_rust.training_dynamic import compile_dynamic_transform, dynamic_action
from brian2_rust.training_equations import neuron_parameter_bank, NormalNoise
from test_native_training import RUNNER
from test_training_delay_update import snapshot


def model(detach=True, window=None, ranks=None, nested=False, noisy=False):
    identity=[[[dict(op='state',index=0)]]]*2
    weights=[[.12,.16],[.2]]+([[.07]] if noisy else [])
    p=lif_training_plan([2,2,2],projections=[neuron_parameter_bank(len(w)) for w in weights],
        state_equations=identity,state_resets=identity,clock=dict(origin=0.,dt=.0002),
        threshold=[10.,.5],detach_reset=detach,tbptt_window=window,mpi_ranks=ranks,noise_streams=[0,1] if noisy else None)
    programs=[];actions=[]
    def add(code,names,reads,owner,params=None,**kw):
        transform=compile_dynamic_transform(code,states=names,parameters=params,
            state_types={slot:'integer' if name=='pick' else 'float' for name,slot in names.items()})
        ps=len(programs);programs.append(transform['programs'])
        action=dynamic_action(transform,reads,owner=owner,program_set=ps,**kw);actions.append(action)
        return action
    for j in range(2):add('v=.8*v',dict(v=0),[j],j)
    for j in range(2):
        params=dict(alpha=(1,0));noise={}
        if noisy:params.update(sigma=(2,0),eta=NormalNoise(0));noise=dict(noise_streams=1,noise_domain=1,noise_entity=j)
        a=add('v=.8*v+alpha*.2*peer'+('+sigma*eta' if noisy else ''),dict(v=0,peer=1,pick=2),[2+j,6+j,4+j],2+j,params,**noise)
        a['indirect']=dict(reads={'1':dict(index=8+j if nested else 4+j,tables=[[4,5],[0,1]] if nested else [[0,1]])})
    for j in range(4):actions.append(dict(owner=j,reads=[j],writes=[],program_set=None,threshold=j,trigger=None))
    for j in range(2):add('v+=w',dict(v=0),[j],j,dict(w=(0,j)),trigger=dict(external=True,index=j))
    for j in range(2):add('v=0.',dict(v=0),[j],j,trigger=dict(external=False,index=j),detach_trigger=True)
    for j in range(2):
        a=add('pick=1-pick\npeer+=.1\nv-=.3',dict(v=0,peer=1,pick=2),[2+j,6+j,4+j],2+j,
              trigger=dict(external=False,index=2+j),detach_trigger=detach)
        a['indirect']=dict(reads={'1':dict(index=4+j,tables=[[0,1]])},
                           writes={'1':dict(index=dict(kind='output',slot=2),tables=[[0,1]])})
    initial=[.2,.4,1.4,1.7,0,1,0.,0.]+([0,1] if nested else [])
    integers=[4,5]+([8,9] if nested else [])
    p.update(schema='b2-dynamic-training-plan-v5',dynamic=dict(initial=initial,initial_parameters=[None]*len(initial),
        integer_states=integers,detached=[j in integers for j in range(len(initial))],voltage=[0,1,2,3],program_sets=programs,actions=actions))
    x=np.array([[1,0],[0,1],[1,0],[0,1],[1,1],[0,0]],float)
    return p,weights,x


def oracle(p,weights,x,initial=None,anchors=None):
    z=np.array(p['dynamic']['initial'] if initial is None else initial,float)
    saved=[];spikes=[];before=[];states=[]
    for t,inp in enumerate(x):
        if anchors is not None and p['tbptt_window'] and t and t%p['tbptt_window']==0:z=anchors['before'][t].copy()
        before.append(z.copy());z[:2]*=.8
        pick=z[4:6].astype(int)
        z[2:4]=.8*z[2:4]+weights[1][0]*.2*z[pick]
        if p.get('noise_streams'):
            from test_training_stochastic import normal
            z[2:4]+=weights[2][0]*np.array([normal(p['seed'],0,0,1,j,t,0) for j in range(2)])
        threshold=np.array([10.,10.,.5,.5]);v=z[:4].copy();hard=(v>threshold).astype(float);s=hard.copy()
        if anchors is not None:
            hard=anchors['hard'][t]
            phi=p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(anchors['v'][t]-threshold))**2
            s=hard+phi*(v-anchors['v'][t])
        z[:2]+=inp*weights[0]
        z[:2]*=1-hard[:2]
        for j in range(2):
            old=int(z[4+j]);new=1-old;g=hard[2+j] if p['detach_reset'] else s[2+j]
            value=z[old]+.1
            z[new]+=g*(value-z[new]);z[2+j]-=.3*g
            if hard[2+j]:z[4+j]=new
        saved.append(v);spikes.append(s);states.append(z.copy())
    logits=np.array(spikes)[:,2:].mean(axis=0)*p['logit_scale']
    loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
    return loss,z,np.array(spikes),dict(v=np.array(saved),hard=(np.array(saved)>np.array([10.,10.,.5,.5])).astype(float),before=np.array(before),states=np.array(states))


@pytest.mark.parametrize('detach',[False,True])
@pytest.mark.parametrize('window',[None,2])
@pytest.mark.parametrize('nested',[False,True])
@pytest.mark.parametrize('noisy',[False,True])
def test_runtime_index_all_float_state_and_parameter_finite_difference(detach,window,nested,noisy):
    p,w,x=model(detach,window,nested=nested,noisy=noisy)
    out=NativeLIFTrainer(p,runner=RUNNER,weights=w).gradients(x[None],[0])
    assert out['gradient_scope']==('full-bptt-detached-index-routing' if window is None else 'tbptt-detach-boundaries-and-index-routing')
    loss,z,spikes,anchors=oracle(p,w,x)
    assert out['loss']==pytest.approx(loss,abs=1e-14)
    np.testing.assert_allclose(out['final_state'][0],z,atol=2e-15,rtol=2e-15)
    np.testing.assert_array_equal(out['spikes'][0],spikes)
    eps=1e-6
    for bank,row in enumerate(w):
        for k in range(len(row)):
            upper=copy.deepcopy(w);lower=copy.deepcopy(w);upper[bank][k]+=eps;lower[bank][k]-=eps
            fd=(oracle(p,upper,x,anchors=anchors)[0]-oracle(p,lower,x,anchors=anchors)[0])/(2*eps)
            assert out['gradients'][bank][k]==pytest.approx(fd,rel=3e-6,abs=2e-8)
    for k in range(len(z)):
        if k in p['dynamic']['integer_states']:
            assert out['initial_state_gradients'][0][k]==0;continue
        upper=np.array(p['dynamic']['initial']);lower=upper.copy();upper[k]+=eps;lower[k]-=eps
        fd=(oracle(p,w,x,upper,anchors)[0]-oracle(p,w,x,lower,anchors)[0])/(2*eps)
        assert out['initial_state_gradients'][0][k]==pytest.approx(fd,rel=3e-6,abs=2e-8)


@pytest.mark.parametrize('ranks',[None,2,8])
@pytest.mark.parametrize('detach',[False,True])
def test_batch_indices_carry_checkpoint_and_mpi(ranks,detach,tmp_path):
    if ranks and os.environ.get('B2_TEST_MPI')!='1':pytest.skip('local MPI required')
    p,w,x=model(detach,ranks=ranks)
    trainer=NativeLIFTrainer(p,runner=RUNNER,weights=w)
    init=np.tile(p['dynamic']['initial'],(2,1));init[1,4:6]=[1,0];init[1,2:4]=[.8,1.1]
    xx=np.stack([x,x[:,::-1]])
    out=trainer.gradients(xx,[0,1],initial=init)
    base=copy.deepcopy(p);base['mpi_ranks']=None
    refs=[NativeLIFTrainer(base,runner=RUNNER,weights=w).gradients(xx[i:i+1],[i],initial=init[i:i+1]) for i in range(2)]
    for i,r in enumerate(refs):
        np.testing.assert_allclose(out['final_state'][i],r['final_state'][0],atol=1e-14)
        np.testing.assert_allclose(np.array(out['initial_state_gradients'][i])*2,r['initial_state_gradients'][0],atol=1e-13)
    for bank,row in enumerate(out['gradients']):np.testing.assert_allclose(row,np.mean([r['gradients'][bank] for r in refs],axis=0),atol=1e-13)
    p['trainable']=[False]*len(w);trainer=NativeLIFTrainer(p,runner=RUNNER,weights=w)
    trainer.step(xx[:,:2],[0,1],initial=init)
    path=tmp_path/'indexed.json';trainer.store(path);restored=NativeLIFTrainer(p,runner=RUNNER);restored.restore(path)
    actual=restored.step(xx[:,2:],[0,1],initial='carry')
    np.testing.assert_allclose(actual['final_state'],out['final_state'],atol=1e-14)
    before=snapshot(restored);restored.gradients(xx,[0,1],initial='carry');assert snapshot(restored)[:-1]==before[:-1]


@pytest.mark.parametrize('field',['negative','large','typed','nested_type','write_slot','output_type','empty_table','budget'])
def test_invalid_addressing_rejects_atomically(field):
    p,w,x=model();a=p['dynamic']['actions'][2]
    if field in ('negative','large'):p['dynamic']['initial'][4]=-1 if field=='negative' else 2
    elif field=='typed':a['indirect']['reads']['1']['index']=0
    elif field=='nested_type':a['indirect']['reads']['1']['tables']=[[0,1],[0,1]]
    elif field=='write_slot':p['dynamic']['actions'][-1]['indirect']['writes']['99']=p['dynamic']['actions'][-1]['indirect']['writes'].pop('1')
    elif field=='output_type':p['dynamic']['actions'][-1]['indirect']['writes']['1']['index']['slot']=0
    elif field=='empty_table':a['indirect']['reads']['1']['tables']=[[]]
    elif field=='budget':p['max_tape_bytes']=800
    trainer=NativeLIFTrainer(p,runner=RUNNER,weights=w);before=snapshot(trainer)
    with pytest.raises(ValueError):trainer.step(x[None],[0])
    assert snapshot(trainer)==before


def test_inactive_detached_event_does_not_access_invalid_index():
    p,w,x=model();p['dynamic']['actions']=[a for a in p['dynamic']['actions'] if a not in p['dynamic']['actions'][2:4]]
    # Remove unreferenced programs and renumber after removing the indexed ODE.
    ids=sorted({a['program_set'] for a in p['dynamic']['actions'] if a['program_set'] is not None})
    p['dynamic']['program_sets']=[p['dynamic']['program_sets'][i] for i in ids]
    for a in p['dynamic']['actions']:
        if a['program_set'] is not None:a['program_set']=ids.index(a['program_set'])
    p['dynamic']['initial'][2:6]=[.1,.2,-1,9]
    out=NativeLIFTrainer(p,runner=RUNNER,weights=w).gradients(np.zeros((1,3,2)),[0])
    assert out['final_state'][0][4:6]==[-1,9]


def test_dynamic_write_collisions_keep_only_last_output_vjp():
    p,w,x=model()
    # Both outputs resolve to A[0]. The first output's parameter path must vanish.
    p['projections'].append(neuron_parameter_bank(1));p['masks'].append([1.]);p['trainable'].append(True);w.append([.37])
    transform=compile_dynamic_transform('a=peer+unused\nb=2*peer',states=dict(a=0,b=1,peer=2,pick=3),
        parameters=dict(unused=(2,0)),state_types={0:'float',1:'float',2:'float',3:'integer'})
    ps=len(p['dynamic']['program_sets']);p['dynamic']['program_sets'].append(transform['programs'])
    action=dynamic_action(transform,[6,7,0,4],owner=0,program_set=ps)
    action['indirect']=dict(writes={str(s):dict(index=dict(kind='read',slot=3),tables=[[0,1]]) for s in [0,1]})
    p['dynamic']['actions'].insert(0,action)
    xx=x[None,:1]
    out=NativeLIFTrainer(p,runner=RUNNER,weights=w).gradients(xx,[0])
    assert out['gradients'][2]==[0.]
    # A fixed-address equivalent isolates resolution and last-writer VJP.
    q=copy.deepcopy(p);q['dynamic']['program_sets'][-1]=[transform['programs'][1]]
    q['dynamic']['actions'][0]=dict(action,indirect=None,writes=[0])
    ref=NativeLIFTrainer(q,runner=RUNNER,weights=w).gradients(xx,[0])
    for key in ('final_state','initial_state_gradients','spikes'):
        np.testing.assert_allclose(out[key],ref[key],atol=1e-14)
    for a,b in zip(out['gradients'],ref['gradients']):np.testing.assert_allclose(a,b,atol=1e-14)


@pytest.mark.parametrize('ranks',[None,2,8])
def test_nonroot_runtime_write_index_error_preserves_trainer(ranks):
    if ranks and os.environ.get('B2_TEST_MPI')!='1':pytest.skip('local MPI required')
    p,w,x=model(ranks=ranks)
    last=p['dynamic']['actions'][-1]
    # Owner 3 computes an invalid new integer index during its reset.
    ps=last['program_set'];p['dynamic']['program_sets'][ps][2]=[dict(op='integer_constant',value=7)]
    trainer=NativeLIFTrainer(p,runner=RUNNER,weights=w);before=snapshot(trainer)
    with pytest.raises(ValueError,match='runtime index outside'):trainer.step(x[None],[0])
    assert snapshot(trainer)==before


def test_index_core_matches_actual_compiled_brian(tmp_path):
    import brian2 as b
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='cython'
    old=b.prefs.codegen.runtime.cython.cache_dir;b.prefs.codegen.runtime.cython.cache_dir=str(tmp_path/'cython')
    try:
        dt=.2*b.ms
        inp=b.SpikeGeneratorGroup(2,[0,1,0,1],np.arange(4)*dt,dt=dt)
        a=b.NeuronGroup(2,'dv/dt=-v/ms:1',threshold='v>10',reset='v=0',method='euler',dt=dt,name='indirect_a')
        a.v=[.2,.4]
        c=b.NeuronGroup(2,'dv/dt=(-v+.2*peer)/ms:1\npeer:1 (linked)\npick:integer',threshold='v>.5',
            reset='pick=1-pick\npeer+=.1\nv-=.3',method='euler',dt=dt,name='indirect_c')
        c.pick=[0,1];c.peer=b.linked_var(a,'v',index='pick');c.v=[1.4,1.7]
        syn=b.Synapses(inp,a,'w:1',on_pre='v_post+=w',dt=dt);syn.connect(j='i');syn.w=[.12,.16]
        ma=b.StateMonitor(a,'v',record=True,when='end');mc=b.StateMonitor(c,['v','pick'],record=True,when='end')
        net=b.Network(inp,a,c,syn,ma,mc);net.run(4*dt,namespace={})
        assert c.resetter['spike'].codeobj.compiled_code['run'] is not None
        p,w,x=model();p['trainable']=[False]*len(w);trainer=NativeLIFTrainer(p,runner=RUNNER,weights=w)
        actual=[]
        for tick in range(4):actual.append(trainer.step(x[None,tick:tick+1],[0],initial='carry' if tick else None)['final_state'][0])
        actual=np.array(actual)
        np.testing.assert_allclose(actual[:,:2].T,ma.v[:],atol=1e-14)
        np.testing.assert_allclose(actual[:,2:4].T,mc.v[:],atol=1e-14)
        np.testing.assert_array_equal(actual[:,4:6].T,mc.pick[:])
    finally:b.prefs.codegen.runtime.cython.cache_dir=old


def test_indirect_access_cannot_alias_managed_delay_queue():
    from test_training_event_delay import model as delayed_model
    *_,x,bundle=delayed_model()
    p=copy.deepcopy(bundle.plan);spec=p['dynamic'];index=len(spec['initial'])
    spec['initial'].append(0.);spec['initial_parameters'].append(None);spec['detached'].append(True)
    spec.setdefault('integer_states',[]).append(index)
    histories={k for path in spec['delay_layout']['paths'] for edge in path['edges'] for k in edge['states']}
    placeholder=len(spec['initial']);spec['initial'].append(0.);spec['initial_parameters'].append(None)
    spec['detached'].append(True);spec['binary_states'].append(placeholder)
    blocks={k for path in spec['delay_layout']['paths'] for k in range(path['start'],path['end'])}
    action=next(a for i,a in enumerate(spec['actions']) if i not in blocks and a.get('threshold') is None)
    slot=len(action['reads']);action['reads'].append(placeholder)
    action['indirect']=dict(reads={str(slot):dict(index=index,tables=[[min(histories)]])})
    with pytest.raises(ValueError,match='index mappings cannot access delay queue'):
        NativeLIFTrainer(p,runner=RUNNER,weights=bundle.weights).evaluate(x[None],[0])


@pytest.mark.parametrize('backend',['metal','cuda'])
def test_gpu_indirect_actions_match_cpu(backend):
    flag='B2_TEST_GPU' if backend=='metal' else 'B2_TEST_CUDA_TRAIN'
    if os.environ.get(flag)!='1':pytest.skip('actual '+backend+' hardware required')
    p,w,x=model(detach=False,nested=True,noisy=True)
    reference=NativeLIFTrainer(p,runner=RUNNER,weights=w).gradients(x[None],[0])
    p['backend']=backend;trainer=NativeLIFTrainer(p,runner=RUNNER,weights=w);before=snapshot(trainer)
    result=trainer.gradients(x[None],[0])
    assert result['backend']==backend and result['gpu_dispatches']>0
    assert snapshot(trainer)[:-1]==before[:-1]
    for key in ('final_state','initial_state_gradients','spikes','logits'):
        np.testing.assert_allclose(result[key],reference[key],rtol=5e-4,atol=2e-6)
    for a,b in zip(result['gradients'],reference['gradients']):np.testing.assert_allclose(a,b,rtol=5e-4,atol=2e-6)
