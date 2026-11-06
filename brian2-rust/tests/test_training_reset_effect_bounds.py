"""Scalar reset work remains unconditional even with an empty event list."""
import brian2 as b
import copy
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer,lower_brian_dynamic_training
from test_training_callback_effects import invalid_curve
from test_training_integer_ir import engine
from test_training_poisson_zero_vjp import mpi
from test_native_training import RUNNER

@pytest.mark.parametrize('discard',[False,True])
@pytest.mark.parametrize('ranks',[None,2])
def test_empty_reset_scalar_error_is_not_silently_gated(engine,discard,ranks):
 mpi(ranks)
 b.set_device('runtime');b.start_scope();b.prefs.codegen.target='numpy'
 b.prefs.codegen.runtime.numpy.discard_units=discard
 dt=.2*b.ms;source=b.SpikeGeneratorGroup(1,[],np.array([])*b.second,dt=dt)
 callback=b.Function(invalid_curve,arg_units=[1],arg_names=['x'],return_unit=1,stateless=False)
 groups=[b.NeuronGroup(1,'dv/dt=0/second:1',threshold='v>100',reset='v=bad(.5)',
                       method='euler',dt=dt,namespace={'bad':callback}) for _ in range(2)]
 net=b.Network(source,*groups)
 bundle=lower_brian_dynamic_training(net,input_group=source,layers=groups,backend=engine,mpi_ranks=ranks)
 trainer=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER);before=copy.deepcopy(trainer.state)
 with pytest.raises(ValueError,match='[Nn]on.?finite|domain'):
  trainer.step(np.zeros((1,1,1)),[0])
 assert trainer.state==before and trainer.neuron_state is None
 assert trainer.clock_tick==0 and trainer.next_noise_sequence==0
 # Brian's scalar reset block actually executes despite the empty spike set.
 with pytest.raises(b.core.base.BrianObjectException) as caught:
  net.run(dt,namespace={})
 assert isinstance(caught.value.__cause__,ZeroDivisionError)
