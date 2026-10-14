"""Continuous own-array synaptic callbacks: actual Brian and all VJPs."""
import copy
import brian2 as b
import numpy as np
import pytest
from brian2_rust import lower_brian_dynamic_training,NativeLIFTrainer
from brian2_rust.training_brian import _INTEGRATORS
from test_training_callback_effects import curve
from test_training_integer_ir import engine
from test_training_poisson_zero_vjp import mpi
from test_training_stochastic import normal
from test_native_training import RUNNER


def model(method,discard,engine,ranks,window,kind):
 b.set_device('runtime');b.start_scope();b.prefs.codegen.target='numpy'
 b.prefs.codegen.runtime.numpy.discard_units=discard;dt=.2*b.ms
 source=b.SpikeGeneratorGroup(1,[],np.array([])*b.second,dt=dt,name='effect_input')
 a=b.NeuronGroup(2,'dv/dt=(.5-v)/ms:1',method='euler',threshold='v>.5',reset='v-=.5',dt=dt,name='effect_a')
 c=b.NeuronGroup(2,'dv/dt=(.5-v+q)/ms:1\nq:1',method='euler',threshold='v>.5',reset='v-=.5',dt=dt,name='effect_c')
 a.v=[.6,.1];c.v=[.4,.7]
 function=b.Function(curve,arg_units=[1],arg_names=['x'],return_unit=1,stateless=False)
 noisy=method in ('heun','milstein')
 argument={'drift':'h','hidden':'a','temporary':'.5*a'}[kind]
 syn=b.Synapses(a,c,'dh/dt=gain*curve('+argument+')/ms'+('+.1*xi/ms**.5' if noisy else '')+':1 (clock-driven)\nq_post=h:1 (summed)\ngain:1 (constant)'+('\na:1' if kind!='drift' else ''),
                method=method,dt=dt,namespace={'curve':function},name='effect_syn')
 syn.connect(i=[1,0],j=[0,1]);syn.h=[.6,.4];syn.gain=[.2,.3]
 if kind!='drift':syn.a=[.3,.5]
 net=b.Network(source,a,c,syn)
 bundle=lower_brian_dynamic_training(net,input_group=source,layers=[a,c],backend=engine,mpi_ranks=ranks,
  tbptt_window=window,detach_reset=False,seed=7123,trainable_synapse_parameters={syn.name:['gain','h']+(['a'] if kind!='drift' else [])})
 stage=compile(_INTEGRATORS[method](syn.equations,variables=syn.variables),'<independent synaptic stages>','exec')
 return net,source,(a,c),syn,dt,bundle,stage,noisy


def reference(bundle,groups,syn,stage,noisy,weights,initial=None,anchors=None):
 p=bundle.plan;z=np.array(bundle.initial_state if initial is None else initial,float)
 if initial is None:
  for slot,parameter in enumerate(p['dynamic']['initial_parameters']):
   if parameter is not None:z[slot]=weights[parameter[0]][parameter[1]]
 layouts=bundle.provenance['neuron_state_layout'];a=layouts[groups[0].name]['v'];c=layouts[groups[1].name]['v'];q=layouts[groups[1].name]['q']
 h=bundle.provenance['dynamic_state_layout'][syn.name]['h']
 bank=next(row['bank'] for row in bundle.provenance['bindings'] if row['object']==syn.name and row['variables']==['gain'])
 auxiliary=bundle.provenance['dynamic_state_layout'][syn.name].get('a')
 constant_a=next((row['bank'] for row in bundle.provenance['bindings'] if row['object']==syn.name and row['variables']==['a'] and row['kind']=='synapse_constant'),None)
 margins=[];spikes=[];before=[]
 for tick in range(4):
  if anchors is not None and p['tbptt_window'] and tick and tick%p['tbptt_window']==0:z=anchors[1][tick].copy()
  before.append(z.copy());z[q]=z[h]
  z[a]=.8*z[a]+.1;z[c]=.8*z[c]+.1+.2*z[q]
  noise=np.array([normal(p['seed'],9,0,2,j,tick,0) for j in range(2)])
  environment=dict(h=z[h].copy(),gain=np.array(weights[bank]),dt=.0002,ms=.001,curve=curve,randn=lambda:noise.copy(),sqrt=np.sqrt)
  if auxiliary is not None:environment['a']=z[auxiliary].copy()
  elif constant_a is not None:environment['a']=np.array(weights[constant_a])
  exec(stage,environment);z[h]=environment['h']
  if auxiliary is not None:z[auxiliary]=environment['a']
  margin=z[a+c]-.5;gate=(margin>0).astype(float)
  if anchors is not None:
   old=anchors[0][tick];gate=(old>0).astype(float)+p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(old))**2*(margin-old)
  margins.append(margin.copy());spikes.append(gate.copy());z[a+c]-=.5*gate
 spikes=np.array(spikes);logits=spikes[:,2:].mean(0)*p['logit_scale']
 loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
 return loss,z,spikes,(np.array(margins),np.array(before))


@pytest.mark.parametrize('kind',['drift','hidden','temporary'])
@pytest.mark.parametrize('method',['euler','rk2','rk4','heun','milstein'])
@pytest.mark.parametrize('discard',[False,True])
@pytest.mark.parametrize('ranks',[None,2])
@pytest.mark.parametrize('window',[None,2])
def test_synaptic_integrator_callback_original_and_all_vjps(engine,method,discard,ranks,window,kind,monkeypatch):
 mpi(ranks);net,source,groups,syn,dt,bundle,stage,noisy=model(method,discard,engine,ranks,window,kind)
 options={'noise_sequence':9} if noisy else {}
 result=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).gradients(np.zeros((1,4,1)),[0],**options)
 loss,z,spikes,anchors=reference(bundle,groups,syn,stage,noisy,bundle.weights)
 np.testing.assert_allclose(result['final_state'][0],z,rtol=5e-5,atol=5e-6)
 np.testing.assert_array_equal(result['spikes'][0],spikes);assert result['loss']==pytest.approx(loss,abs=5e-6)
 for bank,row in enumerate(bundle.weights):
  for j in range(len(row)):
   hi=copy.deepcopy(bundle.weights);lo=copy.deepcopy(bundle.weights);hi[bank][j]+=1e-6;lo[bank][j]-=1e-6
   fd=(reference(bundle,groups,syn,stage,noisy,hi,anchors=anchors)[0]-reference(bundle,groups,syn,stage,noisy,lo,anchors=anchors)[0])/2e-6
   assert result['gradients'][bank][j]==pytest.approx(fd,rel=6e-4,abs=6e-6)
 initial_slots={j for group in groups for row in bundle.provenance['neuron_state_layout'][group.name].values() for j in row}
 for row in bundle.provenance['dynamic_state_layout'][syn.name].values():initial_slots.update(row)
 for j in sorted(initial_slots):
  hi=np.array(bundle.initial_state,float);lo=hi.copy();hi[j]+=1e-6;lo[j]-=1e-6
  fd=(reference(bundle,groups,syn,stage,noisy,bundle.weights,initial=hi,anchors=anchors)[0]-reference(bundle,groups,syn,stage,noisy,bundle.weights,initial=lo,anchors=anchors)[0])/2e-6
  assert result['initial_state_gradients'][0][j]==pytest.approx(fd,rel=6e-4,abs=6e-6)
 calls=[]
 def replay(*shape):
  assert shape==(2,) and len(calls)<4;tick=len(calls);calls.append(tick)
  return np.array([normal(7123,9,0,2,j,tick,0) for j in range(2)])
 if noisy:monkeypatch.setattr(np.random,'randn',replay)
 net.run(4*dt,namespace={})
 if noisy:assert len(calls)==4
 for group in groups:
  for name,slots in bundle.provenance['neuron_state_layout'][group.name].items():
   np.testing.assert_allclose(np.asarray(result['final_state'])[0,slots],group.variables[name].get_value(),rtol=5e-5,atol=5e-6)
 for name,slots in bundle.provenance['dynamic_state_layout'][syn.name].items():
  np.testing.assert_allclose(np.asarray(result['final_state'])[0,slots],syn.variables[name].get_value(),rtol=5e-5,atol=5e-6)
 if kind=='temporary':
  assert 'a' not in bundle.provenance['dynamic_state_layout'][syn.name]
  np.testing.assert_array_equal(syn.a[:],[.3,.5])
 if kind=='hidden':assert 'a' in bundle.provenance['dynamic_state_layout'][syn.name]
 assert (result['gpu_dispatches']>0)==(engine!='cpu')


@pytest.mark.parametrize('method',['euler','milstein'])
@pytest.mark.parametrize('ranks',[None,2])
def test_synaptic_hidden_callback_carry_restore_preserves_all_states(engine,method,ranks,tmp_path):
 mpi(ranks);net,source,groups,syn,dt,bundle,stage,noisy=model(method,True,engine,ranks,None,'hidden')
 plan=copy.deepcopy(bundle.plan);plan['trainable']=[False]*len(plan['trainable'])
 x=np.zeros((1,4,1));options={'noise_sequence':9} if noisy else {}
 complete=NativeLIFTrainer(plan,weights=bundle.weights,runner=RUNNER).evaluate(x,[0],**options)
 trainer=NativeLIFTrainer(plan,weights=bundle.weights,runner=RUNNER)
 first=trainer.step(x[:,:2],[0],**options);checkpoint=tmp_path/'checkpoint';trainer.store(checkpoint)
 restored=NativeLIFTrainer(plan,runner=RUNNER);restored.restore(checkpoint)
 last=restored.step(x[:,2:],[0],initial='carry')
 np.testing.assert_allclose(last['final_state'],complete['final_state'],rtol=5e-5,atol=5e-6)
 np.testing.assert_array_equal(np.concatenate([first['spikes'],last['spikes']],axis=1),complete['spikes'])
 assert restored.clock_tick==4
 if noisy:assert restored.noise_sequence==9 and restored.next_noise_sequence==10
 for name in ('h','a'):assert name in bundle.provenance['dynamic_state_layout'][syn.name]


@pytest.mark.parametrize('ranks',[None,2])
def test_synaptic_callback_discarded_error_is_atomic(engine,ranks):
 from test_training_callback_effects import invalid_curve
 mpi(ranks);net,source,groups,syn,dt,_,_,_=model('milstein',True,engine,ranks,None,'hidden')
 syn.namespace['curve']=b.Function(invalid_curve,arg_units=[1],arg_names=['x'],return_unit=1,stateless=False)
 bundle=lower_brian_dynamic_training(net,input_group=source,layers=groups,backend=engine,mpi_ranks=ranks,seed=7123)
 trainer=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER);before=copy.deepcopy(trainer.state)
 with pytest.raises(ValueError,match='[Nn]on.?finite|domain'):trainer.step(np.zeros((1,1,1)),[0],noise_sequence=9)
 assert trainer.state==before and trainer.neuron_state is None and trainer.clock_tick==0 and trainer.next_noise_sequence==0
