"""Typed shared capture views preserve exact values and eager source order."""
import copy
import brian2 as b
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer,lower_brian_dynamic_training
from test_training_shared_capture_views import model
from test_training_event_captures import readonly_event_capture,integer_event_capture,boolean_event_capture
from test_training_integer_ir import engine
from test_training_poisson_zero_vjp import mpi
from test_native_training import RUNNER


@pytest.mark.parametrize('dtype',['integer','boolean'])
@pytest.mark.parametrize('readonly_first',[False,True])
@pytest.mark.parametrize('ranks',[None,2])
def test_shared_typed_views_latest_values_and_eager_order(engine,dtype,readonly_first,ranks):
    mpi(ranks);net,g,syn,_,dt,_,x=model(False,readonly_first,0,ranks,engine)
    array=np.array([16777217,2147483646],dtype=np.int32) if dtype=='integer' else np.array([True,False])
    view=array.view();view.flags.writeable=False
    readonly=b.Function(readonly_event_capture(view),arg_units=[1],arg_names=['x'],return_unit=1,stateless=False)
    mutable=b.Function((integer_event_capture if dtype=='integer' else boolean_event_capture)(array),arg_units=[1],arg_names=['x'],return_unit=1,stateless=False)
    syn.namespace.update(f=readonly if readonly_first else mutable,q=mutable if readonly_first else readonly)
    source=next(obj for obj in net.objects if isinstance(obj,b.SpikeGeneratorGroup));hidden=next(obj for obj in net.objects if isinstance(obj,b.NeuronGroup) and obj is not g)
    bundle=lower_brian_dynamic_training(net,input_group=source,layers=[hidden,g],backend=engine,mpi_ranks=ranks,trainable_synapse_parameters={syn.name:['h']})
    t=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER)
    if dtype=='boolean' and readonly_first:
        t.evaluate(np.zeros((1,1,2)),[0]);before=copy.deepcopy(t.state)
        with pytest.raises(ValueError,match='[Nn]on.?finite|domain'):t.step(x[:,:1],[0])
        assert t.state==before and t.neuron_state is None and t.clock_tick==0
        return
    p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights);out=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER).evaluate(x,[0]);net.run(4*dt,namespace={})
    cells=bundle.provenance['mutable_capture_layout'][0]['cells'];np.testing.assert_array_equal(np.asarray(out['final_state'])[0,cells],array.astype(float))
    cells=bundle.provenance['dynamic_state_layout'][syn.name]['h'];np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],syn.h[:],rtol=8e-5,atol=8e-6)
