"""Composed array effects, pure return ownership, timed inputs and Poisson."""
import copy
import math
import brian2 as b
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer, lower_brian_dynamic_training
from test_training_callback_effects import curve
from test_training_integer_ir import engine
from test_training_poisson_zero_vjp import mpi
from test_training_poisson_core import mix, MASK, uniform
from test_native_training import RUNNER


def identity(x): return x
def constant(x): return .125
def square(x): return x*x
def nested_identity(x): return identity(x)
def private_constant(x): return np.array(.125, dtype=np.float64)
def selection(x): return np.where(x>.5,x,2*x)
CAPTURED_UNIT = None
def captured_unit_value(x): return CAPTURED_UNIT
def copied_unit_value(x): return np.array(CAPTURED_UNIT,dtype=np.float64)


CODES={
 'identity':'v=curve(pure(v))+v+.1*gain',
 'nested_identity':'v=curve(pure(v))+v+.1*gain',
 'constant':'v=curve(pure(v))+v+.1*gain',
 'private_constant':'v=curve(pure(v))+v+.1*gain',
 'square':'v=curve(pure(v))+v+.1*gain',
 'selection':'v=curve(pure(v))+v+.1*gain',
 'timed_scalar':'v=curve(drive(t))+v+.1*gain',
 'timed_vector':'v=curve(drive(t+v*ms))+v+.1*gain',
 'poisson':'v=curve(v)+v+.01*poisson(rate)+.1*gain',
}


def model(kind,discard=True,ranks=None,window=None,backend='cpu',lower=True):
 b.set_device('runtime');b.start_scope();b.prefs.codegen.target='numpy'
 b.prefs.codegen.runtime.numpy.discard_units=discard;dt=.2*b.ms
 source=b.SpikeGeneratorGroup(1,[],np.array([])*b.second,dt=dt,name='composed_input')
 hidden=b.NeuronGroup(1,'dv/dt=0/second:1',method='euler',threshold='v>100',reset='v=0',dt=dt,name='composed_hidden')
 namespace={'curve':b.Function(curve,arg_units=[1],arg_names=['x'],return_unit=1,stateless=False)}
 if kind.startswith('timed'):namespace['drive']=b.TimedArray(np.array([.4,.8,.6,1.2]),dt=.3*b.ms,name='composed_drive')
 elif kind!='poisson':namespace['pure']=b.Function(globals()[kind],arg_units=[1],arg_names=['x'],return_unit=1)
 group=b.NeuronGroup(2,'dv/dt=0/second:1\ngain:1 (constant)\nrate:1 (constant)',method='euler',threshold='v>.5',reset='v-=.5',dt=dt,namespace=namespace,name='composed_neurons')
 group.v=[.4,.7];group.gain=[.3,.5];group.rate=[.3,.5]
 runner=group.run_regularly(CODES[kind],dt=dt,when='start',name='composed_regular')
 net=b.Network(source,hidden,group)
 if not lower:return net,group,runner,dt,None
 bundle=lower_brian_dynamic_training(net,input_group=source,layers=[hidden,group],backend=backend,mpi_ranks=ranks,
   tbptt_window=window,detach_reset=False,trainable_neuron_parameters={group.name:['gain','rate']})
 bundle.provenance['callback_discard_units']=discard
 return net,group,runner,dt,bundle


def poisson_count(rate,seed,domain,entity,tick):
 address=mix(seed^0x4232504f49533031)
 for field in (9,0,domain,entity,0,tick,0):address=mix(address^mix((field+0x9e3779b97f4a7c15)&MASK))
 arrival=0.;count=0
 while True:
  arrival-=math.log1p(-uniform(address,count))
  if arrival>=rate:return count
  count+=1


def oracle(bundle,group,runner,kind,weights,initial=None,anchors=None):
 p=bundle.plan;z=np.array(bundle.initial_state if initial is None else initial,float)
 vslots=bundle.provenance['neuron_state_layout'][group.name]['v'];v=z[vslots].copy()
 banks={name:next(row['bank'] for row in bundle.provenance['bindings'] if row['object']==group.name and row['variables']==[name]) for name in ('gain','rate')}
 gain=np.array(weights[banks['gain']]);rate=np.array(weights[banks['rate']]);before=[];margins=[];spikes=[];counts=[];logp=0.
 if kind.startswith('timed'):
  entry=bundle.provenance['timed_inputs'][0];table=np.array(weights[entry['bank']])
  k=max(1,int(2**np.ceil(np.log2(8*entry['dt_seconds']/.0002))));epsilon=entry['dt_seconds']/k
  def timed(t):return table[np.clip(np.floor((np.asarray(t)/epsilon+.5)/k).astype(int),0,len(table)-1)]
 for tick in range(4):
  if anchors is not None and p['tbptt_window'] and tick and tick%p['tbptt_window']==0:v=anchors['before'][tick].copy()
  before.append(v.copy())
  if kind in ('identity','nested_identity','poisson'):v=1.6*v
  elif kind=='constant':v=v+.125
  elif kind=='private_constant':v=v+(.1 if bundle.provenance['callback_discard_units'] else .125)
  elif kind=='square':v=v+.8*v*v
  elif kind=='selection':v=v+.8*np.where(v>.5,v,2*v)
  elif kind=='timed_scalar':v=v+timed(tick*.0002)
  else:v=v+.8*timed(tick*.0002+v*.001)
  if kind=='poisson':
   domain=bundle.provenance['regular_runner_layout'][runner.name]['noise_domain']
   count=np.array([poisson_count(r,p['seed'],domain,j,tick) for j,r in enumerate(rate)]) if anchors is None else anchors['counts'][tick]
   counts.append(count);v=v+.01*count
   logp+=sum(n*math.log(r)-r-math.lgamma(n+1) for n,r in zip(count,rate))
  v=v+.1*gain;margin=v-.5;gate=(margin>0).astype(float)
  if anchors is not None:
   old=anchors['margins'][tick];gate=(old>0).astype(float)+p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(old))**2*(margin-old)
  margins.append(margin);spikes.append(gate);v=v-.5*gate
 logits=np.array(spikes).mean(0)*p['logit_scale'];loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
 objective=loss+(anchors['loss']*logp if kind=='poisson' and anchors is not None else 0.)
 return objective,v,np.array(spikes),dict(before=before,margins=margins,counts=counts,loss=loss)


@pytest.mark.parametrize('kind',list(CODES))
@pytest.mark.parametrize('discard',[False,True])
@pytest.mark.parametrize('ranks',[None,2])
@pytest.mark.parametrize('window',[None,2])
def test_composed_calls_original_and_all_vjps(engine,kind,discard,ranks,window,monkeypatch):
 mpi(ranks);net,group,runner,dt,bundle=model(kind,discard,ranks,window,engine)
 result=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).gradients(np.zeros((1,4,1)),[0],**({'noise_sequence':9} if kind=='poisson' else {}))
 loss,v,spikes,anchors=oracle(bundle,group,runner,kind,bundle.weights)
 slots=bundle.provenance['neuron_state_layout'][group.name]['v']
 np.testing.assert_allclose(np.asarray(result['final_state'])[0,slots],v,rtol=7e-5,atol=7e-6)
 np.testing.assert_array_equal(np.asarray(result['spikes'])[0,:,1:],spikes);assert result['loss']==pytest.approx(loss,abs=7e-6)
 for bank,values in enumerate(bundle.weights):
  for index in range(len(values)):
   hi=copy.deepcopy(bundle.weights);lo=copy.deepcopy(bundle.weights);hi[bank][index]+=1e-6;lo[bank][index]-=1e-6
   fd=(oracle(bundle,group,runner,kind,hi,anchors=anchors)[0]-oracle(bundle,group,runner,kind,lo,anchors=anchors)[0])/2e-6
   assert result['gradients'][bank][index]==pytest.approx(fd,rel=9e-4,abs=9e-6)
 for index in range(len(bundle.initial_state)):
  hi=np.array(bundle.initial_state);lo=hi.copy();hi[index]+=1e-6;lo[index]-=1e-6
  fd=(oracle(bundle,group,runner,kind,bundle.weights,initial=hi,anchors=anchors)[0]-oracle(bundle,group,runner,kind,bundle.weights,initial=lo,anchors=anchors)[0])/2e-6
  assert result['initial_state_gradients'][0][index]==pytest.approx(fd,rel=9e-4,abs=9e-6)
 if kind=='poisson':
  values=iter(anchors['counts']);calls=[]
  def sample(lam,size=None):
   np.testing.assert_array_equal(lam,np.array([.3,.5]));assert size==2
   calls.append(size);return next(values).copy()
  monkeypatch.setattr(np.random,'poisson',sample)
 net.run(4*dt,namespace={})
 np.testing.assert_allclose(group.v[:],v,rtol=7e-5,atol=7e-6)
 if kind=='poisson':assert calls==[2]*4 and next(values,None) is None
 assert (result['gpu_dispatches']>0)==(engine!='cpu')


@pytest.mark.parametrize('discard',[False,True])
@pytest.mark.parametrize('copy_capture',[False,True])
def test_captured_unit_arrays_keep_borrowing_or_private_copy(discard,copy_capture,monkeypatch):
 from brian2_rust import TrainingConversionError
 global CAPTURED_UNIT
 from brian2.units.fundamentalunits import DIMENSIONLESS
 CAPTURED_UNIT=b.Quantity(np.array(.125),dim=DIMENSIONLESS,force_quantity=True)
 kind='copied_unit_value' if copy_capture else 'captured_unit_value'
 monkeypatch.setitem(CODES,kind,CODES['constant'])
 if discard and not copy_capture:
  with pytest.raises(TrainingConversionError,match='readonly'):model(kind,discard)
  net,group,_,dt,_=model(kind,discard,lower=False);net.run(4*dt,namespace={})
  # Original unitless wrappers expose shared captured storage. A native
  # constant snapshot must explicitly refuse its mutation until capture state
  # is represented, rather than silently returning a scalar approximation.
  assert float(CAPTURED_UNIT)==pytest.approx(.125*.8**4)
 else:
  net,group,_,dt,bundle=model(kind,discard)
  out=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).evaluate(np.zeros((1,4,1)),[0])
  net.run(4*dt,namespace={})
  slots=bundle.provenance['neuron_state_layout'][group.name]['v']
  np.testing.assert_allclose(np.asarray(out['final_state'])[0,slots],group.v[:],rtol=0,atol=1e-13)
  assert float(CAPTURED_UNIT)==.125
