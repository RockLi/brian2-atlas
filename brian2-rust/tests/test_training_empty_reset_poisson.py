"""Scalar Poisson lambda validation remains eager for an empty reset array."""
import copy
import brian2 as b
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer,lower_brian_dynamic_training
from test_training_typed_callback_effects import increment
from test_training_integer_ir import engine
from test_training_poisson_zero_vjp import mpi
from test_native_training import RUNNER

@pytest.mark.parametrize('rate',[-1.,1e20])
@pytest.mark.parametrize('discard',[False,True])
@pytest.mark.parametrize('ranks',[None,2])
def test_empty_poisson_scalar_domain_native_and_original(engine,rate,discard,ranks):
 mpi(ranks);b.set_device('runtime');b.start_scope();b.prefs.codegen.target='numpy';b.prefs.codegen.runtime.numpy.discard_units=discard
 dt=.2*b.ms;source=b.SpikeGeneratorGroup(1,[],np.array([])*b.second,dt=dt)
 f=b.Function(increment,arg_units=[1],arg_names=['x'],return_unit=1,stateless=False)
 groups=[b.NeuronGroup(2,'dv/dt=0/second:1\nrate:1 (constant, shared)',threshold='v>100',
                       reset='v=change(poisson(rate))',method='euler',dt=dt,namespace={'change':f}) for _ in range(2)]
 for group in groups:group.rate=rate
 net=b.Network(source,*groups);bundle=lower_brian_dynamic_training(net,input_group=source,layers=groups,backend=engine,mpi_ranks=ranks)
 trainer=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER);before=copy.deepcopy(trainer.state)
 with pytest.raises(ValueError,match='[Nn]on.?finite|domain'):trainer.step(np.zeros((1,1,1)),[0])
 assert trainer.state==before and trainer.neuron_state is None and trainer.clock_tick==0 and trainer.next_noise_sequence==0
 with pytest.raises(b.core.base.BrianObjectException) as caught:net.run(dt,namespace={})
 assert isinstance(caught.value.__cause__,ValueError)

@pytest.mark.parametrize('rate',[0.,2.,3e9])
@pytest.mark.parametrize('ranks',[None,2])
def test_empty_poisson_scalar_rate_validation_does_not_sample(engine,rate,ranks):
 mpi(ranks);b.set_device('runtime');b.start_scope();b.prefs.codegen.target='numpy';b.prefs.codegen.runtime.numpy.discard_units=True
 dt=.2*b.ms;source=b.SpikeGeneratorGroup(1,[],np.array([])*b.second,dt=dt)
 f=b.Function(increment,arg_units=[1],arg_names=['x'],return_unit=1,stateless=False)
 groups=[b.NeuronGroup(2,'dv/dt=0/second:1\nrate:1 (constant, shared)',threshold='v>100',
                       reset='v=change(poisson(rate))',method='euler',dt=dt,namespace={'change':f}) for _ in range(2)]
 for group in groups:group.rate=rate
 net=b.Network(source,*groups);bundle=lower_brian_dynamic_training(net,input_group=source,layers=groups,backend=engine,mpi_ranks=ranks)
 out=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).gradients(np.zeros((1,2,1)),[0])
 net.run(2*dt,namespace={})
 np.testing.assert_array_equal(np.asarray(out['spikes']),0.)
 for group in groups:
  np.testing.assert_array_equal(np.asarray(out['final_state'])[0,bundle.provenance['neuron_state_layout'][group.name]['v']],group.v[:])
 assert all(np.all(np.asarray(row)==0) for row in out['gradients'])
 assert (out['gpu_dispatches']>0)==(engine!='cpu')
