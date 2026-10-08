"""Authenticated Python captures bound to native persistent canonical arrays."""
import copy
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer
from brian2_rust.training import lif_training_plan
from brian2_rust.training_equations import neuron_parameter_bank
from brian2_rust.training_effects import lower_state_effect_function,bind_state_effect_captures,compile_state_effect_transform
from test_training_integer_ir import engine
from test_training_poisson_zero_vjp import mpi
from test_native_training import RUNNER


def callback(array):
    def captured(x):
        saved=array
        saved*=.8
        return x+array+saved
    return captured


def model(backend,ranks,window):
    array=np.array([.113,.217]);function=callback(array)
    descriptor=bind_state_effect_captures(lower_state_effect_function(function),{'array':'capture'})
    states={'v':0,'capture':1}
    effect=compile_state_effect_transform('v = f(v)',states=states,parameters={'f':descriptor},array_states=set(states),writable_states=set(states),
        physical_slots={0:0,1:1},writeback_order=('v',),state_types={0:'float',1:'float'})
    assert set(effect['effect_writes'])=={0,1}
    identity=[[[dict(op='state',index=0)]]]*2
    plan=lif_training_plan([1,1,2],projections=[neuron_parameter_bank(1),neuron_parameter_bank(2)],state_equations=identity,state_resets=identity,
        threshold=.5,detach_reset=False,clock=dict(origin=0.,dt=.0002),backend=backend,mpi_ranks=ranks,tbptt_window=window,trainable=[False,True])
    actions=[];programs=[effect['programs'],[[dict(op='state',index=0),dict(op='constant',value=.5),dict(op='sub',left=0,right=1)]]]
    for j in range(2):actions.append(dict(owner=1+j,reads=[1+j,3+j],writes=[([1+j,3+j])[slot] for slot in effect['writes']],program_set=0,threshold=None,trigger=None))
    for j in range(3):actions.append(dict(owner=j,reads=[j],writes=[],program_set=None,threshold=j,trigger=None))
    for j in range(2):actions.append(dict(owner=1+j,reads=[1+j],writes=[1+j],program_set=1,threshold=None,trigger=dict(external=False,index=1+j),detach_trigger=False))
    initial=[0.,.173,.719,*array]
    plan.update(schema='b2-dynamic-training-plan-v5',dynamic=dict(initial=initial,initial_parameters=[None,None,None,[1,0],[1,1]],detached=[False]*5,
        voltage=[0,1,2],integer_states=[],binary_states=[],integer_parameters=[],program_sets=programs,actions=actions))
    return plan,[[0.],array.tolist()],initial,function,array


def oracle(plan,weights,initial=None,anchors=None):
    z=np.array(plan['dynamic']['initial'] if initial is None else initial,float)
    if initial is None:z[3:]=weights[1]
    old=[];margins=[];hard=[];spikes=[]
    for tick in range(4):
        if anchors is not None and plan['tbptt_window'] and tick and tick%plan['tbptt_window']==0:z=anchors['old'][tick].copy()
        old.append(z.copy());z[3:]*=.8;z[1:3]+=2*z[3:]
        margin=z[1:3]-.5;event=(margin>0).astype(float);margins.append(margin.copy());hard.append(event.copy())
        if anchors is not None:
            base=anchors['margins'][tick];event=anchors['hard'][tick]+plan['surrogate']['scale']/(1+plan['surrogate']['slope']*abs(base))**2*(margin-base)
        spikes.append(np.r_[0.,event]);z[1:3]-=.5*event
    logits=np.array(spikes)[:,1:].mean(0)*plan['logit_scale'];loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
    return loss,z,np.array(spikes),dict(old=old,margins=margins,hard=hard)


@pytest.mark.parametrize('ranks',[None,2])
@pytest.mark.parametrize('window',[None,2])
def test_capture_native_all_vjps_and_original_python(engine,ranks,window):
    mpi(ranks);plan,weights,initial,function,array=model(engine,ranks,window);x=np.zeros((1,4,1))
    out=NativeLIFTrainer(plan,weights=weights,runner=RUNNER).gradients(x,[0]);loss,z,spikes,anchors=oracle(plan,weights)
    np.testing.assert_allclose(out['final_state'][0],z,rtol=8e-5,atol=8e-6);np.testing.assert_array_equal(out['spikes'][0],spikes);assert out['loss']==pytest.approx(loss,abs=8e-6)
    for j in range(2):
        hi=copy.deepcopy(weights);lo=copy.deepcopy(weights);hi[1][j]+=1e-6;lo[1][j]-=1e-6
        fd=(oracle(plan,hi,anchors=anchors)[0]-oracle(plan,lo,anchors=anchors)[0])/2e-6
        assert out['gradients'][1][j]==pytest.approx(fd,rel=1e-3,abs=1e-5)
    for j in range(5):
        hi=np.array(initial);lo=hi.copy();hi[j]+=1e-6;lo[j]-=1e-6
        fd=(oracle(plan,weights,hi,anchors)[0]-oracle(plan,weights,lo,anchors)[0])/2e-6
        assert out['initial_state_gradients'][0][j]==pytest.approx(fd,rel=1e-3,abs=1e-5),j
    value=np.array(initial[1:3]);np.testing.assert_array_equal(array,weights[1])
    for tick in range(4):value=function(value);value-=.5*(value>.5)
    np.testing.assert_allclose(value,z[1:3]);np.testing.assert_allclose(array,z[3:]);assert (out['gpu_dispatches']>0)==(engine!='cpu')


@pytest.mark.parametrize('ranks',[None,2])
def test_capture_checkpoint_carries_shared_alias(engine,ranks,tmp_path):
    mpi(ranks);plan,weights,*_=model(engine,ranks,None);plan['trainable']=[False,False];x=np.zeros((1,4,1))
    whole=NativeLIFTrainer(plan,weights=weights,runner=RUNNER).evaluate(x,[0]);trainer=NativeLIFTrainer(plan,weights=weights,runner=RUNNER)
    first=trainer.step(x[:,:2],[0]);path=tmp_path/'capture';trainer.store(path);trainer=NativeLIFTrainer(plan,runner=RUNNER);trainer.restore(path);last=trainer.step(x[:,2:],[0],initial='carry')
    np.testing.assert_allclose(last['final_state'],whole['final_state'],rtol=8e-5,atol=8e-6);np.testing.assert_array_equal(np.concatenate([first['spikes'],last['spikes']],axis=1),whole['spikes'])


def test_unbound_capture_is_rejected_without_execution():
    array=np.array([.113,.217]);function=callback(array);descriptor=lower_state_effect_function(function)
    with pytest.raises(ValueError,match='persistent state bindings'):
        compile_state_effect_transform('v=f(v)',states={'v':0},parameters={'f':descriptor},array_states={'v'},writable_states={'v'})
    np.testing.assert_array_equal(array,[.113,.217])
    with pytest.raises(ValueError,match='every mutable capture'):bind_state_effect_captures(descriptor,{})


def integer_callback(array):
    def captured(x):
        saved=array
        saved+=1
        return x+array
    return captured


@pytest.mark.parametrize('dtype',['float64','int32','bool'])
def test_capture_binding_dtype_ownership_and_shared_identity(dtype):
    from brian2_rust.training_effects import StateEffectFunction
    array=np.array([1,2],dtype=dtype)
    first=lower_state_effect_function(integer_callback(array) if dtype=='int32' else callback(array))
    second=lower_state_effect_function(integer_callback(array) if dtype=='int32' else callback(array))
    assert dict(first.captured_arrays)['array'] is dict(second.captured_arrays)['array'] is array
    first=bind_state_effect_captures(first,{'array':'capture'});second=bind_state_effect_captures(second,{'array':'capture'})
    kind='integer' if dtype=='int32' else 'boolean' if dtype=='bool' else 'float'
    kwargs=dict(states={'v':0,'capture':1},parameters={'f':first,'g':second},array_states={'v','capture'},writable_states={'v','capture'},
                state_types={0:'float',1:kind},physical_slots={0:0,1:1},writeback_order=('v',))
    if dtype=='bool':
        with pytest.raises(ValueError,match='changes array dtype'):compile_state_effect_transform('v=f(v)',**kwargs)
    else:
        effect=compile_state_effect_transform('v=f(v)+g(v)',**kwargs)
        assert 1 in effect['effect_writes']
        wrong=dict(kwargs,state_types={0:'float',1:'boolean' if kind!='boolean' else 'integer'})
        with pytest.raises(ValueError,match='incompatible dtype'):compile_state_effect_transform('v=f(v)',**wrong)
        readonly=dict(kwargs,writable_states={'v'})
        with pytest.raises(ValueError,match='incompatible dtype'):compile_state_effect_transform('v=f(v)',**readonly)
    np.testing.assert_array_equal(array,np.array([1,2],dtype=dtype))


@pytest.mark.parametrize('array',[np.array([np.nan]),np.zeros((1,2)),np.array([1],dtype='int64'),np.arange(4.)[::2]])
def test_invalid_capture_storage_refused_without_execution(array):
    with pytest.raises(ValueError,match='captures require'):lower_state_effect_function(callback(array))


def brian_model(backend,ranks,synaptic):
    import brian2 as b
    from brian2_rust import lower_brian_dynamic_training
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='numpy';dt=.2*b.ms
    array=np.array([.113,.217]);function=callback(array)
    f=b.Function(function,arg_units=[1],arg_names=['x'],return_unit=1,stateless=False)
    source=b.SpikeGeneratorGroup(1,[0,0],np.array([0,2])*dt,dt=dt)
    hidden=b.NeuronGroup(1,'dv/dt=0/second:1',threshold='v>100',reset='v=0',method='euler',dt=dt)
    g=b.NeuronGroup(2,'dv/dt=0/second:1',threshold='v>.5',reset='v-=.5',method='euler',dt=dt,namespace={'f':f});g.v=[.173,.719]
    objects=[source,hidden,g]
    if synaptic:
        owner=b.Synapses(source,g,'h:1',on_pre='v_post+=h',namespace={'f':f},dt=dt);owner.connect(i=[0,0],j=[0,1]);owner.h=[.137,.223];objects.append(owner)
        owner.run_regularly('h=f(h)',when='groups')
    else:owner=g;g.run_regularly('v=f(v)',when='groups')
    net=b.Network(*objects);bundle=lower_brian_dynamic_training(net,input_group=source,layers=[hidden,g],backend=backend,mpi_ranks=ranks,detach_reset=False)
    x=np.zeros((1,4,1));x[0,[0,2],0]=1
    return b,net,g,owner,array,dt,bundle,x


@pytest.mark.parametrize('synaptic',[False,True])
@pytest.mark.parametrize('ranks',[None,2])
def test_public_brian_capture_snapshot_forward_and_restore(engine,synaptic,ranks,tmp_path):
    mpi(ranks);b,net,g,owner,array,dt,bundle,x=brian_model(engine,ranks,synaptic)
    np.testing.assert_array_equal(array,[.113,.217]);entry=bundle.provenance['mutable_capture_layout'][0]
    whole=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).evaluate(x,[0]);plan=copy.deepcopy(bundle.plan);plan['trainable']=[False]*len(bundle.weights)
    t=NativeLIFTrainer(plan,weights=bundle.weights,runner=RUNNER);first=t.step(x[:,:2],[0]);path=tmp_path/'brian-capture';t.store(path);t=NativeLIFTrainer(plan,runner=RUNNER);t.restore(path);last=t.step(x[:,2:],[0],initial='carry')
    np.testing.assert_allclose(last['final_state'],whole['final_state'],rtol=8e-5,atol=8e-6);np.testing.assert_array_equal(np.concatenate([first['spikes'],last['spikes']],axis=1),whole['spikes'])
    monitor=b.SpikeMonitor(g);net.add(monitor);net.run(4*dt,namespace={})
    expected=np.zeros((4,2));expected[np.rint(monitor.t/dt).astype(int),np.asarray(monitor.i)]=1
    np.testing.assert_array_equal(np.asarray(whole['spikes'])[0,:,1:],expected)
    np.testing.assert_allclose(np.asarray(whole['final_state'])[0,entry['cells']],array,rtol=8e-5,atol=8e-6)
    for obj,key in [(g,'neuron_state_layout')]+([(owner,'dynamic_state_layout')] if synaptic else []):
        for field,cells in bundle.provenance[key][obj.name].items():np.testing.assert_allclose(np.asarray(whole['final_state'])[0,cells],obj.variables[field].get_value(),rtol=8e-5,atol=8e-6)
    assert (whole['gpu_dispatches']>0)==(engine!='cpu')


@pytest.mark.parametrize('synaptic',[False,True])
@pytest.mark.parametrize('ranks',[None,2])
@pytest.mark.parametrize('window',[None,2])
def test_public_capture_all_initial_vjps(engine,synaptic,ranks,window):
    mpi(ranks);b,net,g,owner,array,dt,bundle,x=brian_model(engine,ranks,synaptic);plan=bundle.plan;plan['tbptt_window']=window
    v=bundle.provenance['neuron_state_layout'][g.name]['v'];capture=bundle.provenance['mutable_capture_layout'][0]['cells']
    h=bundle.provenance['dynamic_state_layout'][owner.name]['h'] if synaptic else None
    def reference(initial,anchors=None):
        z=np.array(initial,float);old=[];margins=[];hard=[];spikes=[]
        for tick in range(4):
            if anchors is not None and window and tick and tick%window==0:z=anchors['old'][tick].copy()
            old.append(z.copy());z[capture]*=.8;z[h if synaptic else v]+=2*z[capture]
            margin=z[v]-.5;event=(margin>0).astype(float);margins.append(margin.copy());hard.append(event.copy())
            if anchors is not None:
                base=anchors['margins'][tick];event=anchors['hard'][tick]+plan['surrogate']['scale']/(1+plan['surrogate']['slope']*abs(base))**2*(margin-base)
            spikes.append(event.copy())
            if synaptic and tick in (0,2):z[v]+=z[h]
            z[v]-=.5*event
        logits=np.array(spikes).mean(0)*plan['logit_scale'];loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
        return loss,z,dict(old=old,margins=margins,hard=hard)
    out=NativeLIFTrainer(plan,weights=bundle.weights,runner=RUNNER).gradients(x,[0]);loss,z,anchors=reference(bundle.initial_state)
    np.testing.assert_allclose(out['final_state'][0],z,rtol=8e-5,atol=8e-6);assert out['loss']==pytest.approx(loss,abs=8e-6)
    for j in [*v,*capture,*([*h] if synaptic else [])]:
        hi=np.array(bundle.initial_state);lo=hi.copy();hi[j]+=1e-6;lo[j]-=1e-6
        fd=(reference(hi,anchors)[0]-reference(lo,anchors)[0])/2e-6
        assert out['initial_state_gradients'][0][j]==pytest.approx(fd,rel=1e-3,abs=1e-5),j


def boolean_callback(array):
    def captured(x):
        saved=array
        saved*=False
        return x+array
    return captured


@pytest.mark.parametrize('dtype',['int32','bool'])
@pytest.mark.parametrize('ranks',[None,2])
def test_public_typed_capture_exact_execution(engine,dtype,ranks):
    import brian2 as b
    from brian2_rust import lower_brian_dynamic_training
    mpi(ranks);b.set_device('runtime');b.start_scope();b.prefs.codegen.target='numpy';dt=.2*b.ms
    array=np.array([16777217,2147483647],dtype='int32') if dtype=='int32' else np.array([True,False])
    function=integer_callback(array) if dtype=='int32' else boolean_callback(array)
    f=b.Function(function,arg_units=[1],arg_names=['x'],return_unit=1,stateless=False)
    source=b.SpikeGeneratorGroup(1,[],np.array([])*b.second,dt=dt)
    hidden=b.NeuronGroup(1,'dv/dt=0/second:1',threshold='v>100',reset='v=0',method='euler',dt=dt)
    g=b.NeuronGroup(2,'dv/dt=0/second:1',threshold='v>1e12',reset='v=0',method='euler',dt=dt,namespace={'f':f});g.run_regularly('v=f(v)',when='groups')
    net=b.Network(source,hidden,g);bundle=lower_brian_dynamic_training(net,input_group=source,layers=[hidden,g],backend=engine,mpi_ranks=ranks)
    out=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).gradients(np.zeros((1,3,1)),[0]);net.run(3*dt,namespace={})
    cells=bundle.provenance['mutable_capture_layout'][0]['cells'];np.testing.assert_array_equal(np.asarray(out['final_state'])[0,cells],array)
    np.testing.assert_array_equal(np.asarray(out['initial_state_gradients'])[0,cells],[0.,0.])
    v=bundle.provenance['neuron_state_layout'][g.name]['v'];np.testing.assert_allclose(np.asarray(out['final_state'])[0,v],g.v[:],rtol=8e-5,atol=8e-6)
