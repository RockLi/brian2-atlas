"""The last column of a longer readonly capture is checked only on arrivals."""
import copy
import brian2 as b
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer,lower_brian_dynamic_training
from test_training_mixed_capture_lengths import model
from test_training_integer_ir import engine
from test_training_poisson_zero_vjp import mpi
from test_native_training import RUNNER


@pytest.mark.parametrize('mode',['array','vectorised','scalar'])
@pytest.mark.parametrize('ranks',[None,2])
def test_longer_readonly_capture_last_column_empty_and_atomic(engine,mode,ranks):
    mpi(ranks);net,g,syn,a,c,_,_,_=model(mode,0,ranks,engine,'readonly')
    c.flags.writeable=True;c[-1]=0.;c.flags.writeable=False
    source=next(obj for obj in net.objects if isinstance(obj,b.SpikeGeneratorGroup))
    hidden=next(obj for obj in net.objects if isinstance(obj,b.NeuronGroup) and obj is not g)
    bundle=lower_brian_dynamic_training(net,input_group=source,layers=[hidden,g],backend=engine,mpi_ranks=ranks,detach_reset=False,
        trainable_synapse_parameters={syn.name:['h','gain']})
    trainer=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER)
    trainer.evaluate(np.zeros((1,1,2)),[0]);before=copy.deepcopy(trainer.state);original=a.copy()
    with pytest.raises(ValueError,match='[Nn]on.?finite|domain'):
        trainer.step(np.array([[[1.,0.]]]),[0])
    assert trainer.state==before
    np.testing.assert_array_equal(a,original)
    np.testing.assert_array_equal(c,[.29,.37,0.])
