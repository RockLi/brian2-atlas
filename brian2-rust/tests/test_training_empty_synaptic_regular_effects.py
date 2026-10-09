"""Empty Synapses retain NumPy callback scalar work and skip array domains."""
import copy
import brian2 as b
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer,lower_brian_dynamic_training
from test_training_empty_reset_effects import reciprocal_curve
from test_training_callback_effects import curve
from test_training_integer_ir import engine
from test_training_poisson_zero_vjp import mpi
from test_native_training import RUNNER


def model(kind,discard,ranks,backend,invalid=False):
 b.set_device('runtime');b.start_scope();b.prefs.codegen.target='numpy';b.prefs.codegen.runtime.numpy.discard_units=discard
 dt=.2*b.ms;inp=b.SpikeGeneratorGroup(1,[],np.array([])*b.second,dt=dt)
 a=b.NeuronGroup(1,'dv/dt=0/second:1',threshold='v>1',reset='v=0',method='euler',dt=dt)
 c=b.NeuronGroup(2,'dv/dt=0/second:1',threshold='v>1',reset='v=0',method='euler',dt=dt)
 c.v=[.1,.2]
 callback=b.Function(reciprocal_curve,arg_units=[1,1],arg_names=['x','rate'],return_unit=1,stateless=False)
 scalar=b.Function(curve,arg_units=[1],arg_names=['x'],return_unit=1,stateless=False)
 syn=b.Synapses(a,c,'h:1\ngain:1 (shared)\nrate:1 (constant'+(', shared' if kind=='scalar_domain' else '')+')',dt=dt,
                namespace={'check':callback,'curve':scalar})
 syn.connect(False);syn.gain=.4
 if kind=='scalar_domain':syn.rate=0 if invalid else 2
 code='gain=curve(gain)+.1' if kind=='shared' else 'h=check(h,rate)+h'
 runner=syn.run_regularly(code,when='end')
 net=b.Network(inp,a,c,syn)
 bundle=lower_brian_dynamic_training(net,input_group=inp,layers=[a,c],backend=backend,mpi_ranks=ranks,detach_reset=False)
 return net,syn,runner,dt,bundle


@pytest.mark.parametrize('kind',['shared','scalar_domain','vector_domain'])
@pytest.mark.parametrize('discard',[False,True])
@pytest.mark.parametrize('ranks',[None,2])
def test_empty_synaptic_regular_original_carry(engine,kind,discard,ranks,tmp_path):
 mpi(ranks);net,syn,runner,dt,bundle=model(kind,discard,ranks,engine)
 trainer=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER)
 whole=trainer.gradients(np.zeros((1,4,1)),[0])
 for slots in bundle.provenance['dynamic_state_layout'][syn.name].values():
  assert not np.any(np.asarray(whole['initial_state_gradients'])[0,slots])
 assert not any(np.any(row) for row in whole['gradients'])
 trainer=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER)
 for tick in range(4):
  out=trainer.step(np.zeros((1,1,1)),[0],initial='carry' if tick else None)
  net.run(dt,namespace={})
  for name,slots in bundle.provenance['dynamic_state_layout'][syn.name].items():
   np.testing.assert_allclose(np.asarray(out['final_state'])[0,slots],syn.variables[name].get_value(),rtol=8e-5,atol=8e-6)
  path=tmp_path/str(tick);trainer.store(path)
  trainer=NativeLIFTrainer(trainer.plan,runner=RUNNER);trainer.restore(path)
 np.testing.assert_allclose(out['final_state'],whole['final_state'],rtol=8e-5,atol=8e-6)
 assert (out['gpu_dispatches']>0)==(engine!='cpu')


@pytest.mark.parametrize('discard',[False,True])
@pytest.mark.parametrize('ranks',[None,2])
def test_empty_synaptic_regular_scalar_error_atomic(engine,discard,ranks):
 mpi(ranks);net,syn,runner,dt,bundle=model('scalar_domain',discard,ranks,engine,invalid=True)
 trainer=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER);before=copy.deepcopy(trainer.state)
 with pytest.raises(ValueError,match='[Nn]on.?finite|domain'):trainer.step(np.zeros((1,1,1)),[0])
 assert trainer.state==before and trainer.neuron_state is None and trainer.clock_tick==0 and trainer.next_noise_sequence==0
 try:
  with np.errstate(divide='ignore',invalid='ignore'):net.run(dt,namespace={})
 except b.core.base.BrianObjectException as error:assert isinstance(error.__cause__,ZeroDivisionError)


@pytest.mark.parametrize('rate',[-1.,1e20,0.,2.,3e9])
@pytest.mark.parametrize('discard',[False,True])
@pytest.mark.parametrize('ranks',[None,2])
def test_empty_synaptic_regular_poisson_validation(engine,rate,discard,ranks):
 from test_training_typed_callback_effects import increment
 mpi(ranks);net,syn,runner,dt,_=model('scalar_domain',discard,ranks,engine)
 syn.rate=rate
 syn.namespace['change']=b.Function(increment,arg_units=[1],arg_names=['x'],return_unit=1,stateless=False)
 runner.abstract_code='h=h+change(poisson(rate))'
 inp=next(o for o in net.objects if isinstance(o,b.SpikeGeneratorGroup))
 groups=sorted((o for o in net.objects if isinstance(o,b.NeuronGroup)),key=lambda g:len(g))
 bundle=lower_brian_dynamic_training(net,input_group=inp,layers=groups,backend=engine,mpi_ranks=ranks)
 trainer=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER);before=copy.deepcopy(trainer.state)
 if rate<0 or rate>1e19:
  with pytest.raises(ValueError,match='[Nn]on.?finite|domain'):trainer.step(np.zeros((1,1,1)),[0])
  assert trainer.state==before and trainer.neuron_state is None and trainer.clock_tick==0 and trainer.next_noise_sequence==0
  with pytest.raises(b.core.base.BrianObjectException) as caught:net.run(dt,namespace={})
  assert isinstance(caught.value.__cause__,ValueError)
 else:
  out=trainer.gradients(np.zeros((1,2,1)),[0]);net.run(2*dt,namespace={})
  assert not any(node['op']=='poisson' for programs in bundle.plan['dynamic']['program_sets'] for program in programs for node in program)
  assert (out['gpu_dispatches']>0)==(engine!='cpu')
