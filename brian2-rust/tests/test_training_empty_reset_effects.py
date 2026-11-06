"""Scalar reset domains execute for empty selections; vector domains do not."""
import copy
import brian2 as b
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer,lower_brian_dynamic_training
from test_training_integer_ir import engine
from test_training_poisson_zero_vjp import mpi
from test_native_training import RUNNER

def reciprocal_curve(x,rate):
 x*=.8
 ignored=1./rate
 return x

def model(kind,discard,ranks,window,backend,invalid=False):
 b.set_device('runtime');b.start_scope();b.prefs.codegen.target='numpy';b.prefs.codegen.runtime.numpy.discard_units=discard
 dt=.2*b.ms;source=b.SpikeGeneratorGroup(1,[],np.array([])*b.second,dt=dt)
 hidden=b.NeuronGroup(1,'dv/dt=0/second:1',threshold='v>100',reset='v=0',method='euler',dt=dt)
 callback=b.Function(reciprocal_curve,arg_units=[1,1],arg_names=['x','rate'],return_unit=1,stateless=False)
 scalar=kind!='vector_domain'
 code='v=curve(gain,rate)+v-.5' if kind=='scalar_argument' else 'v=curve(v,rate)+v-.5'
 g=b.NeuronGroup(2,'dv/dt=gain/ms:1\ngain:1 (constant'+(', shared' if kind=='scalar_argument' else '')+')\nrate:1 (constant'+(', shared' if scalar else '')+')',
                 method='euler',threshold='v>.5',reset=code,dt=dt,namespace={'curve':callback})
 g.v=[.05,.1] if invalid else [.05,.7]
 g.gain=.1 if kind=='scalar_argument' else [.02,.03]
 g.rate=0 if invalid else 2 if scalar else [0,2]
 net=b.Network(source,hidden,g)
 bundle=lower_brian_dynamic_training(net,input_group=source,layers=[hidden,g],backend=backend,mpi_ranks=ranks,
                                    tbptt_window=window,detach_reset=False,trainable_neuron_parameters={g.name:['gain','rate']})
 return net,g,dt,bundle

def reference(bundle,g,kind,weights,initial=None,anchors=None):
 p=bundle.plan;layout=bundle.provenance['neuron_state_layout'][g.name];z=np.array(bundle.initial_state if initial is None else initial,float)
 # Shared scalar constants may share a bank; provenance supplies its order.
 def coefficient(name):
  row=next(row for row in bundle.provenance['bindings'] if row['object']==g.name and name in row['variables'])
  value=np.array(weights[row['bank']]);return value[row['variables'].index(name)] if len(row['variables'])>1 or row['kind']=='neuron' else value
 gain=coefficient('gain');v=z[layout['v']].copy();before=[];margins=[];hard=[];spikes=[]
 for tick in range(4):
  if anchors is not None and p['tbptt_window'] and tick and tick%p['tbptt_window']==0:v=anchors['before'][tick].copy()
  before.append(v.copy());v+=.2*gain;margin=v-.5;event=(margin>0).astype(float)
  margins.append(margin.copy());hard.append(event.copy())
  if anchors is not None:
   old=anchors['margins'][tick];event=anchors['hard'][tick]+p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(old))**2*(margin-old)
  spikes.append(event.copy())
  # The reset candidate's scalar arithmetic is independent of vector reads.
  # rate is discarded after validation; never divide inactive vector rows.
  candidate=v+.8*gain-.5 if kind=='scalar_argument' else 1.6*v-.5
  v+=event*(candidate-v)
 logits=np.array(spikes).mean(0)*p['logit_scale'];loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
 return loss,v,np.array(spikes),dict(before=before,margins=margins,hard=hard)

@pytest.mark.parametrize('kind',['scalar_argument','scalar_domain','vector_domain'])
@pytest.mark.parametrize('discard',[False,True])
@pytest.mark.parametrize('ranks',[None,2])
def test_empty_reset_original_and_restored_carry(engine,kind,discard,ranks,tmp_path):
 mpi(ranks);net,g,dt,bundle=model(kind,discard,ranks,None,engine)
 bundle.plan['trainable']=[False]*len(bundle.weights);trainer=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER)
 for tick in range(3):
  out=trainer.step(np.zeros((1,1,1)),[0],initial='carry' if tick else None);net.run(dt,namespace={})
  np.testing.assert_allclose(np.asarray(out['final_state'])[0,bundle.provenance['neuron_state_layout'][g.name]['v']],g.v[:],rtol=7e-5,atol=7e-6)
  path=tmp_path/str(tick);trainer.store(path);restored=NativeLIFTrainer(trainer.plan,runner=RUNNER);restored.restore(path);trainer=restored
 assert (out['gpu_dispatches']>0)==(engine!='cpu')

@pytest.mark.parametrize('kind',['scalar_argument','scalar_domain','vector_domain'])
@pytest.mark.parametrize('window',[None,2])
@pytest.mark.parametrize('ranks',[None,2])
def test_empty_reset_all_parameters_and_initial_vjps(engine,kind,window,ranks):
 mpi(ranks);_,g,_,bundle=model(kind,True,ranks,window,engine)
 out=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).gradients(np.zeros((1,4,1)),[0])
 loss,v,spikes,anchors=reference(bundle,g,kind,bundle.weights)
 np.testing.assert_allclose(np.asarray(out['final_state'])[0,bundle.provenance['neuron_state_layout'][g.name]['v']],v,rtol=7e-5,atol=7e-6)
 np.testing.assert_array_equal(np.asarray(out['spikes'])[0,:,1:],spikes);assert out['loss']==pytest.approx(loss,abs=7e-6)
 for bank,row in enumerate(bundle.weights):
  for index in range(len(row)):
   hi=copy.deepcopy(bundle.weights);lo=copy.deepcopy(bundle.weights);hi[bank][index]+=1e-6;lo[bank][index]-=1e-6
   fd=(reference(bundle,g,kind,hi,anchors=anchors)[0]-reference(bundle,g,kind,lo,anchors=anchors)[0])/2e-6
   assert out['gradients'][bank][index]==pytest.approx(fd,rel=9e-4,abs=9e-6),(bank,index)
 for index in range(len(bundle.initial_state)):
  if bundle.plan['dynamic']['detached'][index] or index in bundle.plan['dynamic']['integer_states']:
   assert out['initial_state_gradients'][0][index]==0.;continue
  hi=np.array(bundle.initial_state);lo=hi.copy();hi[index]+=1e-6;lo[index]-=1e-6
  fd=(reference(bundle,g,kind,bundle.weights,hi,anchors)[0]-reference(bundle,g,kind,bundle.weights,lo,anchors)[0])/2e-6
  assert out['initial_state_gradients'][0][index]==pytest.approx(fd,rel=9e-4,abs=9e-6),index
 assert (out['gpu_dispatches']>0)==(engine!='cpu')

@pytest.mark.parametrize('kind',['scalar_argument','scalar_domain'])
@pytest.mark.parametrize('discard',[False,True])
@pytest.mark.parametrize('ranks',[None,2])
def test_empty_selected_scalar_domain_error_is_atomic(engine,kind,discard,ranks):
 mpi(ranks);net,g,dt,bundle=model(kind,discard,ranks,None,engine,invalid=True)
 trainer=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER);before=copy.deepcopy(trainer.state)
 with pytest.raises(ValueError,match='[Nn]on.?finite|domain'):trainer.step(np.zeros((1,1,1)),[0])
 assert trainer.state==before and trainer.neuron_state is None and trainer.clock_tick==0 and trainer.next_noise_sequence==0
 # A warm unit wrapper can supply a Python float (raising) or a NumPy scalar
 # (warning on division). Both execute; native checked arithmetic refuses the
 # invalid eager work and preserves its state under either original form.
 try:
  with np.errstate(divide='ignore',invalid='ignore'):net.run(dt,namespace={})
 except b.core.base.BrianObjectException as error:
  assert isinstance(error.__cause__,ZeroDivisionError)
