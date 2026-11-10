"""Automatic mutable-link lowering against compiled Brian and independent VJPs."""
import copy
import os
import brian2 as b
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer, lower_brian_dynamic_training
from test_native_training import RUNNER
from test_training_linked import cython_cache
from test_training_delay_update import snapshot


def model(*,noisy=False,method='euler',threshold=False,index_alias=False,**options):
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='cython';dt=.2*b.ms
    x=np.array([[1,0],[0,1],[1,0],[0,1],[1,1],[0,0]],float);ticks,ids=np.nonzero(x)
    inp=b.SpikeGeneratorGroup(2,ids,ticks*dt,dt=dt,name='runtime_input')
    a=b.NeuronGroup(2,'dv/dt=-v/ms:1',threshold='v>10',reset='v=0',method=method,dt=dt,name='runtime_a');a.v=[.2,.4]
    eq='dv/dt=(-v+alpha*peer)/ms'+('+sigma*xi/sqrt(ms)' if noisy else '')+':1\npeer:1 (linked)\npick:integer\nalpha:1 (constant)'+ ('\nsigma:1 (constant)' if noisy else '')
    if index_alias:eq+='\nzpick:integer (linked)'
    c=b.NeuronGroup(2,eq,threshold='v+.1*peer>.5' if threshold else 'v>.5',reset='pick=1-pick\npeer+=.1\nv-=.3'+('\nzpick=zpick' if index_alias else ''),method=method,dt=dt,name='runtime_c')
    c.pick=[0,1];c.peer=b.linked_var(a,'v',index='pick');c.v=[1.4,1.7];c.alpha=.2
    if index_alias:c.zpick=b.linked_var(c,'pick')
    if noisy:c.sigma=.07
    syn=b.Synapses(inp,a,'w:1',on_pre='v_post+=w',dt=dt,name='runtime_drive');syn.connect(j='i');syn.w=[.12,.16]
    net=b.Network(inp,a,c,syn)
    bundle=lower_brian_dynamic_training(net,input_group=inp,layers=[a,c],trainable_neuron_parameters={c.name:['alpha']+(['sigma'] if noisy else [])},**options)
    return net,inp,a,c,syn,x,bundle


@pytest.mark.parametrize('index_alias',[False,True])
@pytest.mark.parametrize('method',['euler','rk2','rk4'])
@pytest.mark.parametrize('threshold',[False,True])
def test_automatic_runtime_links_match_compiled_brian(method,threshold,index_alias):
    net,inp,a,c,syn,x,bundle=model(method=method,threshold=threshold,index_alias=index_alias)
    p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights)
    assert any(action.get('indirect') for action in p['dynamic']['actions'])
    trainer=NativeLIFTrainer(p,runner=RUNNER,weights=bundle.weights)
    states=[];spikes=[]
    for t in range(len(x)):
        r=trainer.step(x[None,t:t+1],[0],initial='carry' if t else None);states.append(r['final_state'][0]);spikes.append(r['spikes'][0][0])
    ma=b.StateMonitor(a,'v',record=True,when='end');mc=b.StateMonitor(c,['v','pick'],record=True,when='end');sm=[b.SpikeMonitor(g) for g in (a,c)]
    net.add(ma,mc,*sm);net.run(len(x)*.2*b.ms,namespace={})
    assert c.resetter['spike'].codeobj.compiled_code['run'] is not None
    layout=bundle.provenance['neuron_state_layout'];states=np.array(states)
    np.testing.assert_allclose(states[:,layout[a.name]['v']].T,ma.v[:],rtol=1e-13,atol=1e-13)
    np.testing.assert_allclose(states[:,layout[c.name]['v']].T,mc.v[:],rtol=1e-13,atol=1e-13)
    np.testing.assert_array_equal(states[:,layout[c.name]['pick']].T,mc.pick[:])
    expected=np.zeros((len(x),4))
    for l,m in enumerate(sm):expected[np.rint(np.asarray(m.t/b.second)/.0002).astype(int),2*l+np.asarray(m.i)]=1
    np.testing.assert_array_equal(spikes,expected)


def reference(bundle,x,initial=None,weights=None,anchors=None):
    from test_training_stochastic import normal
    p=bundle.plan;layout=bundle.provenance['neuron_state_layout'];weights=bundle.weights if weights is None else weights
    z=np.array(bundle.initial_state if initial is None else initial,float)
    av=layout['runtime_a']['v'];cv=layout['runtime_c']['v'];ix=layout['runtime_c']['pick']
    def bank(obj,name):return np.asarray(weights[next(b['bank'] for b in bundle.provenance['bindings'] if b['object']==obj and name in b['variables'])])
    alpha=bank('runtime_c','alpha');drive=bank('runtime_drive','w')
    noisy=any(b['object']=='runtime_c' and 'sigma' in b['variables'] for b in bundle.provenance['bindings'])
    sigma=bank('runtime_c','sigma') if noisy else None
    before=[];voltages=[];hard_spikes=[];spikes=[]
    for t,inp in enumerate(x):
        if anchors is not None and p['tbptt_window'] and t and t%p['tbptt_window']==0:z=anchors['before'][t].copy()
        before.append(z.copy());z[av]*=.8
        selected=np.asarray(av)[z[ix].astype(int)]
        z[cv]=.8*z[cv]+.2*alpha*z[selected]
        if noisy:z[cv]+=sigma*np.sqrt(.2)*np.array([normal(p['seed'],0,0,1,j,t,0) for j in range(2)])
        v=z[av+cv].copy();hard=(v>np.array([10,10,.5,.5])).astype(float);soft=hard.copy()
        if anchors is not None:
            old=anchors['v'][t];hard=anchors['hard'][t]
            soft=hard+p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(old-np.array([10,10,.5,.5])))**2*(v-old)
        voltages.append(v);hard_spikes.append(hard);spikes.append(soft)
        z[av]+=drive*inp;z[av]*=1-(hard[:2] if p['detach_reset'] else soft[:2])
        for j in range(2):
            pick=int(z[ix[j]]);new=1-pick;g=hard[2+j] if p['detach_reset'] else soft[2+j]
            value=z[av[pick]]+.1;target=av[new]
            z[target]+=g*(value-z[target]);z[cv[j]]-=.3*g
            if hard[2+j] and 'zpick' not in layout['runtime_c']:z[ix[j]]=new
    logits=np.array(spikes)[:,2:].mean(axis=0)*p['logit_scale'];loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
    return loss,z,np.array(spikes),dict(before=before,v=voltages,hard=hard_spikes)


from test_training_integer_ir import engine


@pytest.mark.parametrize('index_alias',[False,True])
@pytest.mark.parametrize('noisy',[False,True])
@pytest.mark.parametrize('detach',[False,True])
@pytest.mark.parametrize('window',[None,2])
def test_frontend_every_float_state_parameter_vjp(engine,noisy,detach,window,index_alias):
    *_,x,bundle=model(noisy=noisy,detach_reset=detach,tbptt_window=window,backend=engine,index_alias=index_alias)
    out=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights).gradients(x[None],[0])
    loss,z,spikes,anchors=reference(bundle,x)
    tol=3e-6 if engine=='cpu' else 8e-4;absolute=3e-8 if engine=='cpu' else 5e-6
    assert out['loss']==pytest.approx(loss,abs=absolute)
    np.testing.assert_array_equal(out['spikes'][0],spikes)
    np.testing.assert_allclose(out['final_state'][0],z,rtol=tol,atol=absolute)
    if engine!='cpu':assert out['gpu_dispatches']>0
    assert 'index-routing' in out['gradient_scope']
    eps=1e-6
    for bank,row in enumerate(bundle.weights):
        for k in range(len(row)):
            hi=copy.deepcopy(bundle.weights);lo=copy.deepcopy(bundle.weights);hi[bank][k]+=eps;lo[bank][k]-=eps
            fd=(reference(bundle,x,weights=hi,anchors=anchors)[0]-reference(bundle,x,weights=lo,anchors=anchors)[0])/(2*eps)
            assert out['gradients'][bank][k]==pytest.approx(fd,rel=tol,abs=absolute)
    for k in range(len(z)):
        if k in bundle.plan['dynamic']['integer_states']:
            assert out['initial_state_gradients'][0][k]==0;continue
        hi=np.array(bundle.initial_state);lo=hi.copy();hi[k]+=eps;lo[k]-=eps
        fd=(reference(bundle,x,hi,anchors=anchors)[0]-reference(bundle,x,lo,anchors=anchors)[0])/(2*eps)
        assert out['initial_state_gradients'][0][k]==pytest.approx(fd,rel=tol,abs=absolute)


@pytest.mark.parametrize('ranks',[None,2,8])
def test_frontend_batch_carry_checkpoint_and_rollback(engine,ranks,tmp_path):
    if ranks and os.environ.get('B2_TEST_MPI')!='1':pytest.skip('local MPI required')
    *_,x,bundle=model(noisy=True,detach_reset=False,tbptt_window=2,backend=engine,mpi_ranks=ranks)
    p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights)
    trainer=NativeLIFTrainer(p,runner=RUNNER,weights=bundle.weights)
    q=copy.deepcopy(p);q.update(backend='cpu',mpi_ranks=None)
    cpu=NativeLIFTrainer(q,runner=RUNNER,weights=bundle.weights)
    initial=np.tile(bundle.initial_state,(2,1));ix=bundle.provenance['neuron_state_layout']['runtime_c']['pick'];initial[1,ix]=[1,0]
    xx=np.stack([x,x[:,::-1]])
    for t in (0,2,4):
        kw=dict(initial='carry') if t else dict(initial=initial,start_tick=3,noise_sequence=7)
        out=trainer.step(xx[:,t:t+2],[0,1],**kw);expected=cpu.step(xx[:,t:t+2],[0,1],**kw)
        for key in ('final_state','initial_state_gradients','spikes'):
            np.testing.assert_allclose(out[key],expected[key],rtol=8e-4,atol=5e-6)
        for actual,want in zip(out['gradients'],expected['gradients']):np.testing.assert_allclose(actual,want,rtol=8e-4,atol=5e-6)
        path=tmp_path/'runtime.json';trainer.store(path);restored=NativeLIFTrainer(p,runner=RUNNER);restored.restore(path);trainer=restored
        assert trainer.clock_tick==cpu.clock_tick and trainer.noise_sequence==cpu.noise_sequence
    before=snapshot(trainer);bad=initial.copy();bad[1,ix[1]]=-1
    with pytest.raises(ValueError):trainer.step(xx,[0,1],initial=bad)
    assert snapshot(trainer)==before


@pytest.mark.parametrize('ranks',[None,2,8])
@pytest.mark.parametrize('write_index',[False,True])
def test_synaptic_runtime_index_matches_compiled_brian(engine,ranks,write_index):
    if ranks and os.environ.get('B2_TEST_MPI')!='1':pytest.skip('local MPI required')
    net,inp,a,c,drive,x,_=model()
    syn=b.Synapses(inp,c,'w:1\nedgepeer:1 (linked)\nedgepick:integer',
        on_pre=('edgepick=1-edgepick\n' if write_index else '')+'edgepeer+=w\nv_post+=.1*edgepeer',dt=.2*b.ms,name='runtime_event')
    syn.connect(i=[0,1],j=[0,1]);syn.edgepick=[0,1];syn.edgepeer=b.linked_var(c,'v',index='edgepick');syn.w=[.09,.13];net.add(syn)
    bundle=lower_brian_dynamic_training(net,input_group=inp,layers=[a,c],backend=engine,mpi_ranks=ranks)
    out=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights).gradients(x[None],[0])
    before=copy.deepcopy(bundle.initial_state)
    monitors=[b.SpikeMonitor(g) for g in (a,c)];net.add(*monitors);net.run(len(x)*.2*b.ms,namespace={})
    assert syn.pre.codeobj.compiled_code['run'] is not None
    expected=np.zeros((len(x),4))
    for l,m in enumerate(monitors):expected[np.rint(np.asarray(m.t/b.second)/.0002).astype(int),2*l+np.asarray(m.i)]=1
    np.testing.assert_array_equal(out['spikes'][0],expected)
    layout=bundle.provenance['neuron_state_layout'];tol=1e-13 if engine=='cpu' else 4e-6
    for g in (a,c):np.testing.assert_allclose(np.asarray(out['final_state'])[0,layout[g.name]['v']],np.asarray(g.v[:]),atol=tol,rtol=tol)
    indices=bundle.provenance['dynamic_state_layout'][syn.name]['edgepick']
    np.testing.assert_array_equal(np.asarray(out['final_state'])[0,indices],np.asarray(syn.edgepick[:]))
    assert bundle.initial_state==before


def test_runtime_index_frontend_budget_and_source_scope():
    from brian2_rust import TrainingConversionError
    net,inp,a,c,syn,x,_=model()
    with pytest.raises(TrainingConversionError,match='budget'):
        lower_brian_dynamic_training(net,input_group=inp,layers=[a,c],max_tape_bytes=500)
    outside=b.NeuronGroup(2,'pick:integer',name='outside_index');outside.pick=[0,1]
    # The index source must belong to selected canonical storage too.
    d=b.NeuronGroup(2,'dv/dt=(-v+.1*peer)/ms:1\npeer:1 (linked)\npick:integer (linked)',threshold='v>.5',reset='v-=.3',method='euler',dt=.2*b.ms,name='runtime_external')
    d.pick=b.linked_var(outside,'pick');d.peer=b.linked_var(a,'v',index='pick');net.add(d)
    with pytest.raises(TrainingConversionError,match='selected'):
        lower_brian_dynamic_training(net,input_group=inp,layers=[a,c,d])


def test_unused_old_index_does_not_block_constant_write():
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='cython';dt=.2*b.ms
    inp=b.SpikeGeneratorGroup(2,[],[]*b.second,dt=dt,name='unused_input')
    a=b.NeuronGroup(2,'dv/dt=-v/ms:1',threshold='v>10',reset='v=0',method='euler',dt=dt,name='unused_a')
    c=b.NeuronGroup(2,'dv/dt=-v/ms:1\npeer:1 (linked)\npick:integer',threshold='v>.5',
        reset='pick=0\npeer=.25\nv-=.3',method='euler',dt=dt,name='unused_c')
    c.peer=b.linked_var(a,'v',index='pick');c.pick=[0,1];c.v=[1.4,1.7]
    net=b.Network(inp,a,c);bundle=lower_brian_dynamic_training(net,input_group=inp,layers=[a,c])
    out=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights).evaluate(np.zeros((1,1,2)),[0])
    net.run(dt,namespace={});assert c.resetter['spike'].codeobj.compiled_code['run'] is not None
    layout=bundle.provenance['neuron_state_layout']
    np.testing.assert_allclose(np.asarray(out['final_state'])[0,layout[a.name]['v']],np.asarray(a.v[:]))
    # No expression consumes the old selector or old linked value. A constant
    # assignment of the selector makes the write valid, including on first run.
    initial=np.array(bundle.initial_state);initial[layout[c.name]['pick']]=[-1,999]
    actual=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights).evaluate(np.zeros((1,1,2)),[0],initial=initial[None])
    np.testing.assert_array_equal(actual['final_state'],out['final_state'])
