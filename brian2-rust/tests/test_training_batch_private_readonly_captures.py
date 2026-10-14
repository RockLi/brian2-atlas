"""Whole external readonly arrays check every column, not only selected rows."""
import copy
import brian2 as b
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer,lower_brian_dynamic_training
from test_training_batch_event_captures import batch_model
from test_training_event_captures import readonly_event_capture
from test_training_integer_ir import engine
from test_training_poisson_zero_vjp import mpi
from test_native_training import RUNNER


@pytest.mark.parametrize('dtype',['float','integer','boolean'])
@pytest.mark.parametrize('repeated',[False,True])
@pytest.mark.parametrize('ranks',[None,2])
@pytest.mark.parametrize('readonly_flag',[False,True])
def test_private_readonly_unselected_domain_empty_atomic(engine,dtype,repeated,ranks,readonly_flag):
    mpi(ranks);net,g,syn,_,_,_,_=batch_model(repeated,False,0,ranks,engine)
    array=np.array([1,0],dtype={'float':np.float64,'integer':np.int32,'boolean':bool}[dtype]);array.flags.writeable=not readonly_flag
    syn.namespace['f']=b.Function(readonly_event_capture(array),arg_units=[1],arg_names=['x'],return_unit=1,stateless=False)
    source=next(obj for obj in net.objects if isinstance(obj,b.SpikeGeneratorGroup));hidden=next(obj for obj in net.objects if isinstance(obj,b.NeuronGroup) and obj is not g)
    bundle=lower_brian_dynamic_training(net,input_group=source,layers=[hidden,g],backend=engine,mpi_ranks=ranks,trainable_synapse_parameters={syn.name:['h']})
    t=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER);t.evaluate(np.zeros((1,1,2)),[0]);before=copy.deepcopy(t.state)
    with pytest.raises(ValueError,match='[Nn]on.?finite|domain'):t.step(np.array([[[1.,0.]]]),[0])
    assert t.state==before and t.neuron_state is None and t.clock_tick==0


def test_private_readonly_capture_write_rejected_without_mutation():
    from test_training_event_captures import event_capture
    net,g,syn,_,_,_,_=batch_model(False,False,0,None,'cpu')
    array=np.ones(2);array.flags.writeable=False
    syn.namespace['f']=b.Function(event_capture(array),arg_units=[1],arg_names=['x'],return_unit=1,stateless=False)
    source=next(obj for obj in net.objects if isinstance(obj,b.SpikeGeneratorGroup));hidden=next(obj for obj in net.objects if isinstance(obj,b.NeuronGroup) and obj is not g)
    with pytest.raises(ValueError,match='readonly|ownership'):
        lower_brian_dynamic_training(net,input_group=source,layers=[hidden,g],trainable_synapse_parameters={syn.name:['h']})
    np.testing.assert_array_equal(array,[1.,1.])
