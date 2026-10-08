"""Actual NumPy selected-copy callbacks and indexed conditional writeback."""
import copy
import numpy as np
import brian2 as b
import pytest
from brian2_rust import NativeLIFTrainer,lower_brian_dynamic_training
from test_training_callback_effects import curve,invalid_curve
from test_training_integer_ir import engine
from test_training_poisson_zero_vjp import mpi
from test_native_training import RUNNER

def vector_invalid_curve(x):
 x*=.8
 unused=1./(x-x)
 return x

CODES={
 'indexed_repeat':'v=curve(a)+a',
 'indexed_self':'v=curve(v)+v',
 'target_add':'v+=curve(a)+a',
 'unguarded_first':'a=curve(a)+a;v+=curve(a)',
 'temporary_alias':'temp=a;v=curve(temp)+a',
 'target_alias':'temp=v;v+=curve(a);a=curve(temp)+temp',
 'hidden_only':'a=curve(v)+v',
 'scalar':'v=curve(.125)+v',
 'coefficient':'v=curve(gain)+gain',
}


def model(kind,discard=True,ranks=None,backend='cpu',when='start'):
 b.set_device('runtime');b.start_scope();b.prefs.codegen.target='numpy';b.prefs.codegen.runtime.numpy.discard_units=discard
 dt=.2*b.ms;source=b.SpikeGeneratorGroup(1,[],np.array([])*b.second,dt=dt,name='guarded_input')
 hidden=b.NeuronGroup(1,'dv/dt=0/second:1',method='euler',threshold='v>100',reset='v=0',dt=dt,name='guarded_hidden')
 function=b.Function(curve,arg_units=[1],arg_names=['x'],return_unit=1,stateless=False)
 g=b.NeuronGroup(2,'dv/dt=0/second:1 (unless refractory)\na:1\ngain:1 (constant)',method='euler',threshold='v>.5',reset='v-=.5',refractory=.4*b.ms,
                 dt=dt,namespace={'curve':function},name='guarded_neurons')
 g.v=[.4,.7];g.a=[.4,.6];g.lastspike=[0.,-1.]*b.second;g.not_refractory=[False,True]
 g.gain=[.3,.5]
 runner=g.run_regularly(CODES[kind],when=when,name='guarded_regular')
 net=b.Network(source,hidden,g)
 bundle=lower_brian_dynamic_training(net,input_group=source,layers=[hidden,g],backend=backend,mpi_ranks=ranks,detach_reset=False,
  trainable_neuron_parameters={g.name:['gain']})
 return net,g,runner,dt,bundle


INDEXED_CODES={
 'indexed_repeat':'v[flag]=curve(a[flag])+a[flag]',
 'indexed_self':'v[flag]=curve(v[flag])+v[flag]',
 'target_add':'v[flag]+=curve(a[flag])+a[flag]',
 'unguarded_first':'a=curve(a)+a;v[flag]+=curve(a[flag])',
 'temporary_alias':'temp=a;v[flag]=curve(temp[flag])+a[flag]',
 'target_alias':'temp=v;v[flag]+=curve(a[flag]);a=curve(temp)+temp',
 'hidden_only':'a=curve(v)+v',
 'scalar':'v[flag]=curve(.125)+v[flag]',
 'coefficient':'v[flag]=curve(gain[flag])+gain[flag]',
}


def reference(bundle,g,runner,kind,when,initial=None,anchors=None,weights=None):
 p=bundle.plan;z=np.array(bundle.initial_state if initial is None else initial,float)
 layout=bundle.provenance['neuron_state_layout'][g.name];vs=layout['v'];aa=layout['a']
 v=z[vs].copy();a=z[aa].copy();last=np.array([0,-5000]);flag=np.array([False,True]);before=[];margins=[];hard_spikes=[];spikes=[]
 bank=next(row['bank'] for row in bundle.provenance['bindings'] if row['object']==g.name and row['variables']==['gain'])
 gain=np.array((bundle.weights if weights is None else weights)[bank])
 def regular():
  nonlocal v,a
  env=dict(v=v.copy(),a=a.copy(),gain=gain.copy(),flag=flag.copy(),curve=curve)
  exec(INDEXED_CODES[kind],env);v=env['v'];a=env['a']
 for tick in range(4):
  before.append((v.copy(),a.copy()))
  if when=='start':regular()
  flag=tick-last>=2
  if when=='after_groups':regular()
  margin=v-.5;hard=(margin>0)&flag;gate=hard.astype(float)
  if anchors is not None:
   baseline=anchors['margins'][tick]
   hard=anchors['hard_spikes'][tick]
   gate=hard.astype(float)+flag*p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(baseline))**2*(margin-baseline)
  margins.append(margin.copy());hard_spikes.append(hard.copy());spikes.append(gate.copy())
  last[hard]=tick;flag[hard]=False;v=v-.5*gate
  if when=='end':regular()
 logits=np.array(spikes).mean(0)*p['logit_scale'];loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
 return loss,v,a,np.array(spikes),dict(margins=margins,hard_spikes=hard_spikes,before=before)


@pytest.mark.parametrize('kind',list(CODES))
@pytest.mark.parametrize('when',['start','after_groups','end'])
@pytest.mark.parametrize('ranks',[None,2])
def test_masked_callback_all_initial_vjps(engine,kind,when,ranks):
 mpi(ranks);_,g,runner,_,bundle=model(kind,True,ranks,engine,when)
 out=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).gradients(np.zeros((1,4,1)),[0])
 loss,v,a,spikes,anchors=reference(bundle,g,runner,kind,when)
 layout=bundle.provenance['neuron_state_layout'][g.name]
 np.testing.assert_allclose(np.asarray(out['final_state'])[0,layout['v']],v,rtol=7e-5,atol=7e-6)
 np.testing.assert_allclose(np.asarray(out['final_state'])[0,layout['a']],a,rtol=7e-5,atol=7e-6)
 np.testing.assert_array_equal(np.asarray(out['spikes'])[0,:,1:],spikes);assert out['loss']==pytest.approx(loss,abs=7e-6)
 for bank,row in enumerate(bundle.weights):
  for index in range(len(row)):
   hi=copy.deepcopy(bundle.weights);lo=copy.deepcopy(bundle.weights);hi[bank][index]+=1e-6;lo[bank][index]-=1e-6
   fd=(reference(bundle,g,runner,kind,when,anchors=anchors,weights=hi)[0]-reference(bundle,g,runner,kind,when,anchors=anchors,weights=lo)[0])/2e-6
   assert out['gradients'][bank][index]==pytest.approx(fd,rel=9e-4,abs=9e-6)
 for index in range(len(bundle.initial_state)):
  # Discrete refractory/clock storage has a stopped derivative; only model
  # float slots and overwritten temporary cells are continuous inputs.
  if index in bundle.plan['dynamic'].get('binary_states',[]) or index in bundle.plan['dynamic'].get('integer_states',[]) or bundle.plan['dynamic']['detached'][index]:
   assert out['initial_state_gradients'][0][index]==0.;continue
  hi=np.array(bundle.initial_state);lo=hi.copy();hi[index]+=1e-6;lo[index]-=1e-6
  fd=(reference(bundle,g,runner,kind,when,hi,anchors)[0]-reference(bundle,g,runner,kind,when,lo,anchors)[0])/2e-6
  assert out['initial_state_gradients'][0][index]==pytest.approx(fd,rel=9e-4,abs=9e-6)
 assert (out['gpu_dispatches']>0)==(engine!='cpu')


@pytest.mark.parametrize('invalid_scalar',[False,True])
def test_empty_boolean_selection_keeps_scalar_errors_and_skips_array_work(invalid_scalar,monkeypatch):
 code='v=bad(a)+a'
 monkeypatch.setitem(CODES,'empty',code)
 # Rebuild with the authenticated selected callback and an all-empty mask.
 net,g,runner,dt,_=model('indexed_repeat')
 g.not_refractory=False;g.lastspike=0*b.second
 runner.abstract_code=code
 g.namespace['bad']=b.Function(invalid_curve if invalid_scalar else vector_invalid_curve,arg_units=[1],arg_names=['x'],return_unit=1,stateless=False)
 source=next(obj for obj in net.objects if obj.name=='guarded_input');hidden=next(obj for obj in net.objects if obj.name=='guarded_hidden')
 bundle=lower_brian_dynamic_training(net,input_group=source,layers=[hidden,g])
 trainer=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER)
 if invalid_scalar:
  before=copy.deepcopy((trainer.plan,trainer.state,trainer.neuron_state,trainer.clock_tick))
  with pytest.raises(ValueError):trainer.step(np.zeros((1,1,1)),[0])
  assert (trainer.plan,trainer.state,trainer.neuron_state,trainer.clock_tick)==before
  with pytest.raises(b.BrianObjectException):net.run(dt,namespace={})
 else:
  out=trainer.evaluate(np.zeros((1,1,1)),[0]);net.run(dt,namespace={})
  slots=bundle.provenance['neuron_state_layout'][g.name]['v']
  np.testing.assert_array_equal(np.asarray(out['final_state'])[0,slots],g.v[:])


@pytest.mark.parametrize('kind',list(CODES))
@pytest.mark.parametrize('discard',[False,True])
@pytest.mark.parametrize('when',['start','after_groups','end'])
@pytest.mark.parametrize('ranks',[None,2])
def test_masked_original_hidden_writes_and_checkpoint(engine,kind,discard,when,ranks,tmp_path):
 mpi(ranks);net,g,runner,dt,bundle=model(kind,discard,ranks,engine,when)
 bundle.plan['trainable']=[False]*len(bundle.weights)
 trainer=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER)
 monitor=b.SpikeMonitor(g);net.add(monitor);allspikes=[]
 for cursor,length in [(0,1),(1,2),(3,1)]:
  out=trainer.step(np.zeros((1,length,1)),[0],initial='carry' if cursor else None)
  net.run(length*dt,namespace={});allspikes.extend(np.asarray(out['spikes'])[0,:,1:])
  for name,slots in bundle.provenance['neuron_state_layout'][g.name].items():
   if name.startswith('__'):continue
   np.testing.assert_allclose(np.asarray(out['final_state'])[0,slots],g.variables[name].get_value(),rtol=7e-5,atol=7e-6)
  path=tmp_path/('checkpoint-'+str(cursor));trainer.store(path)
  restored=NativeLIFTrainer(trainer.plan,runner=RUNNER);restored.restore(path)
  assert restored.neuron_state==trainer.neuron_state and restored.clock_tick==trainer.clock_tick
  trainer=restored
 expected=np.zeros((4,2));expected[np.rint(np.asarray(monitor.t)/float(dt)).astype(int),np.asarray(monitor.i)]=1.
 np.testing.assert_array_equal(allspikes,expected)
 assert (out['gpu_dispatches']>0)==(engine!='cpu')
