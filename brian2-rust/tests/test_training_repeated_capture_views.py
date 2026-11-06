"""Writable zero-stride captures follow NumPy's buffered whole-vector writes."""
import copy
import brian2 as b
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer, lower_brian_dynamic_training
from test_training_mixed_capture_lengths import model
from test_training_event_captures import event_capture, integer_event_capture, boolean_event_capture
from test_training_integer_ir import engine
from test_training_poisson_zero_vjp import mpi
from test_native_training import RUNNER


def repeated_capture(array, increments):
    def f(x):
        captured=array
        captured+=increments
        x*=.7
        return x
    return f


@pytest.mark.parametrize('mode',['array','vectorised','scalar'])
@pytest.mark.parametrize('dtype',['float','integer','boolean','varying'])
@pytest.mark.parametrize('ranks',[None,2])
def test_writable_repeated_capture_original_restore(engine,mode,dtype,ranks,tmp_path):
    mpi(ranks)
    net,g,syn,_,_,dt,_,x=model(mode,1,ranks,engine)
    parent=np.array([2147483646],dtype=np.int32) if dtype=='integer' else np.array([False]) if dtype=='boolean' else np.array([.23])
    array=np.lib.stride_tricks.as_strided(parent,shape=(3,),strides=(0,),writeable=True)
    callback=integer_event_capture(array) if dtype=='integer' else boolean_event_capture(array) if dtype=='boolean' else repeated_capture(array,np.array([.01,.02,.03])) if dtype=='varying' else event_capture(array)
    syn.namespace['f']=b.Function(callback,arg_units=[1],arg_names=['x'],return_unit=1,stateless=False)
    # Two calls per nonempty batch; scalar fallback calls twice per arrival.
    syn.namespace['q']=syn.namespace['f']
    syn.pre.code='h=f('+('v_post' if mode=='scalar' else 'h')+');h=q(h)'+(';v_post+=gain*h' if mode=='scalar' else ';u_post+=gain*h' if mode=='vectorised' else '')
    source=next(obj for obj in net.objects if isinstance(obj,b.SpikeGeneratorGroup))
    hidden=next(obj for obj in net.objects if isinstance(obj,b.NeuronGroup) and obj is not g)
    bundle=lower_brian_dynamic_training(net,input_group=source,layers=[hidden,g],backend=engine,mpi_ranks=ranks,detach_reset=False,trainable_synapse_parameters={syn.name:['h','gain']})
    cells=next(row['cells'] for row in bundle.provenance['mutable_capture_layout'] if any(alias['capture']=='array' for alias in row['aliases']))
    assert len(cells)==3 and len(set(cells))==1
    p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights)
    trainer=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER)
    for tick in range(4):
        out=trainer.step(x[:,tick:tick+1],[0],initial='carry' if tick else None);net.run(dt,namespace={})
        actual=np.asarray(out['final_state'])[0,cells]
        if dtype in ('integer','boolean'):
            np.testing.assert_array_equal(actual,array.astype(float))
            np.testing.assert_array_equal(np.asarray(out['initial_state_gradients'])[0,cells],0.)
        else:np.testing.assert_allclose(actual,array,rtol=8e-5,atol=8e-6)
        for obj,key,names in [(g,'neuron_state_layout',['v','u']),(syn,'dynamic_state_layout',['h'])]:
            for name in names:
                ids=bundle.provenance[key][obj.name][name]
                np.testing.assert_allclose(np.asarray(out['final_state'])[0,ids],obj.variables[name].get_value(),rtol=8e-5,atol=8e-6)
        path=tmp_path/str(tick);trainer.store(path);trainer=NativeLIFTrainer(trainer.plan,runner=RUNNER);trainer.restore(path)
    assert (out['gpu_dispatches']>0)==(engine!='cpu')


@pytest.mark.parametrize('mode',['array','vectorised','scalar'])
@pytest.mark.parametrize('window',[None,2])
@pytest.mark.parametrize('ranks',[None,2])
def test_repeated_physical_capture_all_vjps(engine,mode,window,ranks):
    from test_training_partial_overlap_vjps import model as base_model,oracle
    from test_training_event_captures import readonly_event_capture
    mpi(ranks);net,g,syn,dt,_,x=base_model(mode,False,1,ranks,engine)
    array=np.lib.stride_tricks.as_strided(g.variables['v'].get_value()[1:2],shape=(3,),strides=(0,),writeable=True)
    view=array.view();view.flags.writeable=False
    syn.namespace.update(f=b.Function(event_capture(array),arg_units=[1],arg_names=['x'],return_unit=1,stateless=False),q=b.Function(readonly_event_capture(view),arg_units=[1],arg_names=['x'],return_unit=1,stateless=False))
    source=next(obj for obj in net.objects if isinstance(obj,b.SpikeGeneratorGroup));hidden=next(obj for obj in net.objects if isinstance(obj,b.NeuronGroup) and obj is not g)
    bundle=lower_brian_dynamic_training(net,input_group=source,layers=[hidden,g],backend=engine,mpi_ranks=ranks,detach_reset=False,trainable_synapse_parameters={syn.name:['h','gain']})
    data=net,g,syn,dt,bundle,x;p=copy.deepcopy(bundle.plan);p['tbptt_window']=window
    out=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER).gradients(x,[0]);loss,z,anchors=oracle(data,mode,False,bundle.weights,window)
    cells=[*bundle.provenance['neuron_state_layout'][g.name]['v'],*bundle.provenance['neuron_state_layout'][g.name]['u'],*bundle.provenance['dynamic_state_layout'][syn.name]['h']]
    np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],z[cells],rtol=8e-5,atol=8e-6)
    assert out['loss']==pytest.approx(loss,abs=8e-6)
    for bank,row in enumerate(bundle.weights):
        for index in range(len(row)):
            hi=copy.deepcopy(bundle.weights);lo=copy.deepcopy(bundle.weights);hi[bank][index]+=1e-6;lo[bank][index]-=1e-6
            fd=(oracle(data,mode,False,hi,window,anchors=anchors)[0]-oracle(data,mode,False,lo,window,anchors=anchors)[0])/2e-6
            assert out['gradients'][bank][index]==pytest.approx(fd,rel=1e-3,abs=1e-5),(bank,index)
    for index in range(len(bundle.initial_state)):
        if p['dynamic']['detached'][index] or index in p['dynamic']['integer_states']:continue
        hi=np.array(bundle.initial_state);lo=hi.copy();hi[index]+=1e-6;lo[index]-=1e-6
        fd=(oracle(data,mode,False,bundle.weights,window,hi,anchors)[0]-oracle(data,mode,False,bundle.weights,window,lo,anchors)[0])/2e-6
        assert out['initial_state_gradients'][0][index]==pytest.approx(fd,rel=1e-3,abs=1e-5),index
    slot=bundle.provenance['neuron_state_layout'][g.name]['v'][1]
    if window is None:assert abs(out['initial_state_gradients'][0][slot])>1e-5
    net.run(4*dt,namespace={});np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells[:3]],g.v[:],rtol=8e-5,atol=8e-6)


def test_neuron_repeated_capture_requires_whole_vector_schedule():
    from test_training_neuron_shared_capture_views import model as neuron_model
    from brian2_rust import TrainingConversionError
    net,g,_,_,_=neuron_model('integrator',False,'euler',None,'cpu')
    parent=np.array([.23]);array=np.lib.stride_tricks.as_strided(parent,shape=(2,),strides=(0,),writeable=True)
    g.namespace['f']=b.Function(event_capture(array),arg_units=[1],arg_names=['x'],return_unit=1,stateless=False)
    source=next(obj for obj in net.objects if isinstance(obj,b.SpikeGeneratorGroup));hidden=next(obj for obj in net.objects if isinstance(obj,b.NeuronGroup) and obj is not g)
    with pytest.raises(TrainingConversionError,match='whole event binding'):
        lower_brian_dynamic_training(net,input_group=source,layers=[hidden,g])
    np.testing.assert_array_equal(parent,[.23])
