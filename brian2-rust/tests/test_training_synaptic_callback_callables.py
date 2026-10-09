"""Pure alias ownership and explicit Poisson in generated synaptic ODE/SDEs."""
import ast
import copy
import brian2 as b
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer,lower_brian_dynamic_training
from brian2_rust.training_brian import _INTEGRATORS
from test_training_callback_effects import curve
from test_training_callback_callables import identity,square,private_constant,poisson_count
from test_training_integer_ir import engine
from test_training_poisson_zero_vjp import mpi
from test_training_stochastic import normal
from test_native_training import RUNNER


def model(kind,method,noisy,discard,ranks,window,backend):
 b.set_device('runtime');b.start_scope();b.prefs.codegen.target='numpy';b.prefs.codegen.runtime.numpy.discard_units=discard
 dt=.2*b.ms;source=b.SpikeGeneratorGroup(1,[],np.array([])*b.second,dt=dt,name='syn_call_input')
 a=b.NeuronGroup(2,'dv/dt=(.5-v)/ms:1',method='euler',threshold='v>.5',reset='v-=.5',dt=dt,name='syn_call_a')
 c=b.NeuronGroup(2,'dv/dt=(.5-v+q)/ms:1\nq:1',method='euler',threshold='v>.5',reset='v-=.5',dt=dt,name='syn_call_c')
 a.v=[.6,.1];c.v=[.4,.7]
 namespace={'curve':b.Function(curve,arg_units=[1],arg_names=['x'],return_unit=1,stateless=False),
            'pure':b.Function({'identity':identity,'square':square,'private_constant':private_constant}[kind],arg_units=[1],arg_names=['x'],return_unit=1)}
 syn=b.Synapses(a,c,'dh/dt=gain*curve(pure(a))/ms+.01*poisson(rate)/ms'+('+.1*xi/ms**.5' if noisy else '')+':1 (clock-driven)\na:1\ngain:1 (constant)\nrate:1 (constant)\nq_post=h:1 (summed)',
               method=method,dt=dt,namespace=namespace,name='syn_call_syn')
 syn.connect(i=[1,0],j=[0,1]);syn.a=[.3,.5];syn.h=[.6,.4];syn.gain=[.2,.3];syn.rate=[.3,.5]
 net=b.Network(source,a,c,syn)
 bundle=lower_brian_dynamic_training(net,input_group=source,layers=[a,c],backend=backend,mpi_ranks=ranks,tbptt_window=window,
  detach_reset=False,trainable_synapse_parameters={syn.name:['a','h','gain','rate']},seed=7123)
 code=_INTEGRATORS[method](syn.equations,variables=syn.variables)
 return net,(a,c),syn,dt,bundle,code


def reference(bundle,groups,syn,kind,noisy,discard,code,weights,initial=None,anchors=None):
 z=np.array(bundle.initial_state if initial is None else initial,float);p=bundle.plan
 if initial is None:
  for slot,parameter in enumerate(p['dynamic']['initial_parameters']):
   if parameter is not None:z[slot]=weights[parameter[0]][parameter[1]]
 nl=bundle.provenance['neuron_state_layout'];av=nl[groups[0].name]['v'];cv=nl[groups[1].name]['v'];q=nl[groups[1].name]['q']
 sl=bundle.provenance['dynamic_state_layout'][syn.name];h=sl['h'];aux=sl.get('a')
 banks={name:next(row['bank'] for row in bundle.provenance['bindings'] if row['object']==syn.name and row['variables']==[name]) for name in ('gain','rate')}
 abank=next((row['bank'] for row in bundle.provenance['bindings'] if row['object']==syn.name and row['variables']==['a'] and row['kind']=='synapse_constant'),None)
 domain=bundle.provenance['synaptic_noise_domains'][syn.name] if 'synaptic_noise_domains' in bundle.provenance else 2
 margins=[];spikes=[];before=[];allcounts=[];logp=0.
 stage=compile(code,'<independent synaptic composed calls>','exec')
 for tick in range(4):
  if anchors is not None and p['tbptt_window'] and tick and tick%p['tbptt_window']==0:z=anchors['before'][tick].copy()
  before.append(z.copy());z[q]=z[h];z[av]=.8*z[av]+.1;z[cv]=.8*z[cv]+.1+.2*z[q]
  counts=[];stream=int(noisy)
  def poisson(rate):
   nonlocal stream,logp
   value=np.array([count_at(r,p['seed'],domain,j,tick,stream) for j,r in enumerate(rate)]) if anchors is None else anchors['counts'][tick][len(counts)]
   counts.append(value);stream+=1;logp+=sum(n*np.log(r)-r for n,r in zip(value,rate))
   return value.copy()
  def effect(value):
   if not discard and np.asarray(value).shape==():value=np.asarray(value).item()
   result=curve(value);return np.asarray(result) if not discard else result
  environment=dict(h=z[h].copy(),a=z[aux].copy() if aux is not None else np.array(weights[abank]),gain=np.array(weights[banks['gain']]),rate=np.array(weights[banks['rate']]),
   dt=.0002,ms=.001,curve=effect,pure={'identity':identity,'square':square,'private_constant':private_constant}[kind],poisson=poisson,
   randn=lambda:np.array([normal(p['seed'],9,0,domain,j,tick,0) for j in range(2)]),sqrt=np.sqrt)
  exec(stage,environment);z[h]=environment['h']
  if aux is not None:z[aux]=environment['a']
  allcounts.append(counts);margin=z[av+cv]-.5;gate=(margin>0).astype(float)
  if anchors is not None:
   old=anchors['margins'][tick];gate=(old>0).astype(float)+p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(old))**2*(margin-old)
  margins.append(margin.copy());spikes.append(gate);z[av+cv]-=.5*gate
 logits=np.array(spikes)[:,2:].mean(0)*p['logit_scale'];loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
 return loss+(anchors['loss']*logp if anchors is not None else 0.),z,np.array(spikes),dict(before=before,margins=margins,counts=allcounts,loss=loss)


def count_at(rate,seed,domain,entity,tick,stream):
 # Independent Poisson counter address; stream zero helper is shared with the
 # regular-call test, other streams retain the same published mixer algorithm.
 from test_training_poisson_core import mix,MASK,uniform
 import math
 address=mix(seed^0x4232504f49533031)
 for field in (9,0,domain,entity,0,tick,stream):address=mix(address^mix((field+0x9e3779b97f4a7c15)&MASK))
 arrival=0.;count=0
 while True:
  arrival-=math.log1p(-uniform(address,count))
  if arrival>=rate:return count
  count+=1


@pytest.mark.parametrize('kind',['identity','square','private_constant'])
@pytest.mark.parametrize('method,noisy',[('euler',False),('rk2',False),('euler',True)])
@pytest.mark.parametrize('discard',[False,True])
@pytest.mark.parametrize('ranks',[None,2])
@pytest.mark.parametrize('window',[None,2])
def test_synaptic_composed_sde_all_vjps_and_original(engine,kind,method,noisy,discard,ranks,window,monkeypatch):
 mpi(ranks);net,groups,syn,dt,bundle,code=model(kind,method,noisy,discard,ranks,window,engine)
 result=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).gradients(np.zeros((1,4,1)),[0],noise_sequence=9)
 loss,z,spikes,anchors=reference(bundle,groups,syn,kind,noisy,discard,code,bundle.weights)
 np.testing.assert_allclose(result['final_state'][0],z,rtol=7e-5,atol=7e-6);np.testing.assert_array_equal(result['spikes'][0],spikes)
 assert result['loss']==pytest.approx(loss,abs=7e-6)
 for bank,row in enumerate(bundle.weights):
  for index in range(len(row)):
   hi=copy.deepcopy(bundle.weights);lo=copy.deepcopy(bundle.weights);hi[bank][index]+=1e-6;lo[bank][index]-=1e-6
   fd=(reference(bundle,groups,syn,kind,noisy,discard,code,hi,anchors=anchors)[0]-reference(bundle,groups,syn,kind,noisy,discard,code,lo,anchors=anchors)[0])/2e-6
   assert result['gradients'][bank][index]==pytest.approx(fd,rel=9e-4,abs=9e-6)
 for index in range(len(bundle.initial_state)):
  hi=np.array(bundle.initial_state);lo=hi.copy();hi[index]+=1e-6;lo[index]-=1e-6
  fd=(reference(bundle,groups,syn,kind,noisy,discard,code,bundle.weights,initial=hi,anchors=anchors)[0]-reference(bundle,groups,syn,kind,noisy,discard,code,bundle.weights,initial=lo,anchors=anchors)[0])/2e-6
  assert result['initial_state_gradients'][0][index]==pytest.approx(fd,rel=9e-4,abs=9e-6)
 values=iter(value for tick in anchors['counts'] for value in tick);calls=[]
 def sample(lam,size=None):
  assert size==2;np.testing.assert_array_equal(lam,[.3,.5]);calls.append(size);return next(values).copy()
 monkeypatch.setattr(np.random,'poisson',sample)
 if noisy:
  noises=iter(np.array([normal(bundle.plan['seed'],9,0,2,j,tick,0) for j in range(2)]) for tick in range(4))
  monkeypatch.setattr(np.random,'randn',lambda n:next(noises).copy())
 net.run(4*dt,namespace={});assert next(values,None) is None
 for group in groups:
  for name,slots in bundle.provenance['neuron_state_layout'][group.name].items():np.testing.assert_allclose(np.asarray(result['final_state'])[0,slots],group.variables[name].get_value(),rtol=7e-5,atol=7e-6)
 for name,slots in bundle.provenance['dynamic_state_layout'][syn.name].items():np.testing.assert_allclose(np.asarray(result['final_state'])[0,slots],syn.variables[name].get_value(),rtol=7e-5,atol=7e-6)
 assert ('a' in bundle.provenance['dynamic_state_layout'][syn.name])==(kind=='identity')
 assert (result['gpu_dispatches']>0)==(engine!='cpu')
