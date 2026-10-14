"""Actual Brian run_regularly state-effect conversion and native execution."""
import copy
import ast
import brian2 as b
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer, lower_brian_dynamic_training, TrainingConversionError
from test_training_integer_ir import engine
from test_native_training import RUNNER


def curve(x):
    saved=x
    x*=.8
    return saved


def invalid_curve(x):
    x*=.8
    ignored=1./0.
    return x


def promoted_curve(x,y):
    saved=x
    x+=y
    alias=x
    x*=.8
    return alias+saved


CODES={
 'borrowed':'v=curve(v)+v',
 'temporary':'v=curve(v+.125)+v-.1',
 'scalar':'v=curve(.5)+v',
 'hidden_write':'a=curve(v)',
 'sequential_alias':'a=curve(v);v+=a',
 'left_reference':'v=v+curve(v)',
 'left_value':'v=2*v+curve(v)',
 'identity_product':'v=curve(1.*v)+v',
 'identity_sum':'v=curve(v+0.)+v',
 'identity_division':'v=curve(v/1.)+v',
 'linked_single_borrowed':'a=curve(u)',
 'linked_single_copy':'u=curve(u)+u',
 'linked_overlap_return':'a=curve(v);u+=a',
 'linked_overlap_borrowed':'a=curve(u)+v',
 'linked_overlap_copy':'v=curve(u)+v',
 'linked_overlap_two_writes':'v=curve(u)+v;u+=.2',
 'linked_overlap_rebind':'a=curve(v);u=a+.1;a+=.2',
 'linked_overlap_chain':'a=curve(v);u=curve(a+.1)+v',
}


def network(kind,discard=True,stateless=False,link_index=None):
 b.set_device('runtime');b.start_scope();b.prefs.codegen.target='numpy'
 b.prefs.codegen.runtime.numpy.discard_units=discard
 dt=b.ms;source=b.SpikeGeneratorGroup(1,[],np.array([])*b.second,dt=dt)
 function=b.Function(curve,arg_units=[1],arg_names=['x'],return_unit=1,stateless=stateless)
 groups=[b.NeuronGroup(2,'dv/dt=0/second:1\na:1'+('\nu:1 (linked)' if kind.startswith('linked_') else ''),threshold='v>100',reset='v=0',
                      method='euler',dt=dt,namespace={'curve':function}) for _ in range(2)]
 for group in groups:
  group.v=[1.,2.];group.a=0
  if kind.startswith('linked_'):group.u=b.linked_var(group,'v',index=link_index)
  group.run_regularly(CODES[kind],dt=dt,when='start')
 return b.Network(source,*groups),source,groups,dt


@pytest.mark.parametrize('kind',list(CODES))
@pytest.mark.parametrize('discard',[False,True])
@pytest.mark.parametrize('stateless',[False,True])
@pytest.mark.parametrize('optimise',[False,True])
def test_original_brian_regular_effect_writeback(engine,kind,discard,stateless,optimise):
 b.prefs.codegen.loop_invariant_optimisations=optimise
 try:
  _original_brian_regular_effect_writeback(engine,kind,discard,stateless)
 finally:
  b.prefs.codegen.loop_invariant_optimisations=True


def _original_brian_regular_effect_writeback(engine,kind,discard,stateless):
 net,source,groups,dt=network(kind,discard,stateless)
 before=[np.asarray(group.v[:]).copy() for group in groups]
 bundle=lower_brian_dynamic_training(net,input_group=source,layers=groups,backend=engine)
 for group,value in zip(groups,before):np.testing.assert_array_equal(group.v[:],value)
 result=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).evaluate(np.zeros((1,1,1)),[0])
 net.run(dt,namespace={})
 assert_numpy_write_order(bundle,groups)
 for group in groups:
  for variable in ('v','a'):
   slots=bundle.provenance['neuron_state_layout'][group.name][variable]
   np.testing.assert_allclose(np.asarray(result['final_state'])[0,slots],np.asarray(group.variables[variable].get_value()),atol=3e-6)
 assert (result['gpu_dispatches']>0)==(engine!='cpu')


def test_discarded_callback_error_preserves_trainer_state(engine):
 net,source,groups,dt=network('borrowed')
 function=b.Function(invalid_curve,arg_units=[1],arg_names=['x'],return_unit=1,stateless=False)
 for group in groups:group.namespace['curve']=function
 bundle=lower_brian_dynamic_training(net,input_group=source,layers=groups,backend=engine)
 trainer=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER)
 before=copy.deepcopy(trainer.state)
 with pytest.raises(ValueError,match='[Nn]on.?finite|domain'):
  trainer.step(np.zeros((1,1,1)),[0])
 assert trainer.state==before and trainer.neuron_state is None
 assert trainer.clock_tick==0 and trainer.next_noise_sequence==0


def assert_numpy_write_order(bundle,groups):
 for group in groups:
  runner=next(r for r in group.contained_objects if hasattr(r,'abstract_code') and 'curve' in r.abstract_code)
  order=bundle.provenance['regular_runner_layout'][runner.name]['numpy_write_order']
  writes=[]
  for statement in ast.parse(str(runner.codeobj.code['run'])).body:
   if (isinstance(statement,ast.Assign) and isinstance(statement.value,ast.Name)
       and any(isinstance(t,ast.Subscript) and isinstance(t.value,ast.Name)
               and t.value.id.startswith('_array_') for t in statement.targets)):
    writes.append(statement.value.id)
  assert writes==order['scalar']+order['vector']


@pytest.mark.parametrize('discard',[False,True])
@pytest.mark.parametrize('optimise',[False,True])
def test_shared_scalar_alias_writeback_order(engine,discard,optimise):
 b.set_device('runtime');b.start_scope();b.prefs.codegen.target='numpy'
 b.prefs.codegen.runtime.numpy.discard_units=discard
 b.prefs.codegen.loop_invariant_optimisations=optimise
 try:
  f=b.Function(curve,arg_units=[1],arg_names=['x'],return_unit=1,stateless=False)
  source=b.SpikeGeneratorGroup(1,[],np.array([])*b.second,dt=b.ms)
  groups=[b.NeuronGroup(2,'dv/dt=0/second:1\ng:1 (shared)\nu:1 (linked)',threshold='v>100',reset='v=0',
                       method='euler',dt=b.ms,namespace={'curve':f}) for _ in range(2)]
  for group in groups:
   group.v=[1.,2.];group.g=.5;group.u=b.linked_var(group,'g')
   group.run_regularly('g=curve(g)+.1;u=g+.2;v+=g+u',when='start')
  net=b.Network(source,*groups)
  bundle=lower_brian_dynamic_training(net,input_group=source,layers=groups,backend=engine)
  result=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).evaluate(np.zeros((1,1,1)),[0])
  net.run(b.ms,namespace={});assert_numpy_write_order(bundle,groups)
  for group in groups:
   for name in ('v','g'):
    slots=bundle.provenance['neuron_state_layout'][group.name][name]
    np.testing.assert_allclose(np.asarray(result['final_state'])[0,slots],np.broadcast_to(group.variables[name].get_value(),(len(group),)),atol=3e-6)
  assert (result['gpu_dispatches']>0)==(engine!='cpu')
 finally:b.prefs.codegen.loop_invariant_optimisations=True


@pytest.mark.parametrize('code',['v=curve(i+1)','v=curve(i*2)','v=curve(i+2)-.1'])
def test_integer_temporary_cannot_erase_numpy_casting_error(code):
 net,source,groups,dt=network('borrowed')
 for group in groups:
  regular=next(r for r in group.contained_objects if hasattr(r,'abstract_code') and 'curve' in r.abstract_code)
  regular.abstract_code=code
 before=[np.asarray(g.v[:]).copy() for g in groups]
 with pytest.raises(TrainingConversionError,match='effect augmented assignment changes array dtype'):
  lower_brian_dynamic_training(net,input_group=source,layers=groups)
 for group,value in zip(groups,before):np.testing.assert_array_equal(group.v[:],value)
 with pytest.raises(b.core.base.BrianObjectException) as caught:net.run(dt,namespace={})
 assert isinstance(caught.value.__cause__,TypeError)
 assert 'cast' in str(caught.value.__cause__).lower()


@pytest.mark.parametrize('discard',[False,True])
def test_scalar_promotion_keeps_new_array_alias(engine,discard):
 net,source,groups,dt=network('borrowed',discard)
 function=b.Function(promoted_curve,arg_units=[1,1],arg_names=['x','y'],return_unit=1,stateless=False)
 for group in groups:
  group.namespace['curve']=function
  regular=next(r for r in group.contained_objects if hasattr(r,'abstract_code') and 'curve' in r.abstract_code)
  regular.abstract_code='v=curve(.5,v)'
 bundle=lower_brian_dynamic_training(net,input_group=source,layers=groups,backend=engine)
 result=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).evaluate(np.zeros((1,1,1)),[0])
 net.run(dt,namespace={})
 for group in groups:
  slots=bundle.provenance['neuron_state_layout'][group.name]['v']
  np.testing.assert_allclose(np.asarray(result['final_state'])[0,slots],group.v[:],atol=3e-6)
 assert (result['gpu_dispatches']>0)==(engine!='cpu')


def test_constant_indexed_link_is_not_borrowed_storage(engine):
 net,source,groups,dt=network('linked_single_borrowed',link_index=np.array([1,0],dtype=int))
 before=[np.asarray(g.v[:]).copy() for g in groups]
 bundle=lower_brian_dynamic_training(net,input_group=source,layers=groups,backend=engine)
 for group,value in zip(groups,before):np.testing.assert_array_equal(group.v[:],value)
 out=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).evaluate(np.zeros((1,1,1)),[0])
 net.run(dt,namespace={})
 for group,value in zip(groups,before):
  layout=bundle.provenance['neuron_state_layout'][group.name]
  np.testing.assert_array_equal(group.v[:],value)
  np.testing.assert_allclose(group.a[:],.8*value[[1,0]],atol=3e-6)
  for name in ('v','a'):
   np.testing.assert_allclose(np.asarray(out['final_state'])[0,layout[name]],group.variables[name].get_value(),atol=3e-6)
 assert (out['gpu_dispatches']>0)==(engine!='cpu')
