"""Discarded readonly work observes the latest shared capture, even on empty reset."""
import copy
import brian2 as b
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer, lower_brian_dynamic_training
from test_training_neuron_shared_capture_views import model
from test_training_shared_capture_views import zero_capture
from test_training_integer_ir import engine
from test_training_poisson_zero_vjp import mpi
from test_native_training import RUNNER


@pytest.mark.parametrize('where,empty',[('integrator',False),('threshold',False),('reset',False),('reset',True)])
@pytest.mark.parametrize('ranks',[None,2])
def test_neuron_readonly_domain_after_shared_write_is_atomic(engine,where,empty,ranks):
    mpi(ranks);net,g,array,_,_=model(where,False,'euler',ranks,engine)
    g.namespace['f']=b.Function(zero_capture(array),arg_units=[1],arg_names=['x'],return_unit=1,stateless=False)
    if empty:g.v=[.173,.219]
    source=next(obj for obj in net.objects if isinstance(obj,b.SpikeGeneratorGroup))
    hidden=next(obj for obj in net.objects if isinstance(obj,b.NeuronGroup) and obj is not g)
    bundle=lower_brian_dynamic_training(net,input_group=source,layers=[hidden,g],backend=engine,mpi_ranks=ranks,detach_reset=False)
    trainer=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER)
    before=copy.deepcopy(trainer.state)
    with pytest.raises(ValueError,match='[Nn]on.?finite|domain'):
        trainer.step(np.zeros((1,1,1)),[0])
    assert trainer.state==before
    np.testing.assert_array_equal(array,[.23,.31])
