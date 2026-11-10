"""Typed hidden synaptic storage in generated continuous updates and VJPs."""
import copy
import brian2 as b
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer,lower_brian_dynamic_training
from brian2_rust.training_brian import _INTEGRATORS
from test_training_typed_callback_effects import increment,fill
from test_training_integer_ir import engine
from test_training_poisson_zero_vjp import mpi
from test_training_stochastic import normal
from test_native_training import RUNNER

def model(kind,method,discard,ranks,window,backend):
 b.set_device('runtime');b.start_scope();b.prefs.codegen.target='numpy';b.prefs.codegen.runtime.numpy.discard_units=discard
 dt=.2*b.ms;source=b.SpikeGeneratorGroup(1,[],np.array([])*b.second,dt=dt)
 a=b.NeuronGroup(2,'dv/dt=(.5-v)/ms:1',method='euler',threshold='v>.5',reset='v-=.5',dt=dt)
 c=b.NeuronGroup(2,'dv/dt=(.5-v+q)/ms:1\nq:1',method='euler',threshold='v>.5',reset='v-=.5',dt=dt)
 a.v=[.6,.1];c.v=[.4,.7]
 callback=b.Function(fill if 'boolean' in kind else increment,arg_units=[1],arg_names=['x'],return_unit=1,stateless=False)
 argument={'integer_state':'k','boolean_state':'flag','integer_coefficient':'offset+1','boolean_coefficient':'mask==False'}[kind]
 noisy=method in ('heun','milstein')
 syn=b.Synapses(a,c,'dh/dt=.1*gain*change('+argument+')/ms'+('+.1*xi/ms**.5' if noisy else '')+':1 (clock-driven)\nq_post=h:1 (summed)\nk:integer\nflag:boolean\noffset:integer (constant)\nmask:boolean (constant)\ngain:1 (constant)',
                method=method,dt=dt,namespace={'change':callback})
 syn.connect(i=[1,0],j=[0,1]);syn.h=[.6,.4];syn.k=[2,3];syn.flag=[False,True];syn.offset=[1,2];syn.mask=[True,False];syn.gain=[.2,.3]
 net=b.Network(source,a,c,syn)
 bundle=lower_brian_dynamic_training(net,input_group=source,layers=[a,c],backend=backend,mpi_ranks=ranks,
                                    tbptt_window=window,detach_reset=False,seed=7123,trainable_synapse_parameters={syn.name:['gain','h']})
 stage=compile(_INTEGRATORS[method](syn.equations,variables={**syn.variables,'change':callback}),'<independent typed synaptic stages>','exec')
 return net,(a,c),syn,dt,bundle,stage,noisy

def reference(bundle,groups,syn,kind,stage,noisy,weights,initial=None,anchors=None):
 p=bundle.plan;z=np.array(bundle.initial_state if initial is None else initial,float)
 if initial is None:
  for index,parameter in enumerate(p['dynamic']['initial_parameters']):
   if parameter is not None:z[index]=weights[parameter[0]][parameter[1]]
 ns=bundle.provenance['neuron_state_layout'];ss=bundle.provenance['dynamic_state_layout'][syn.name]
 a=ns[groups[0].name]['v'];c=ns[groups[1].name]['v'];q=ns[groups[1].name]['q'];h=ss['h']
 bank=next(row['bank'] for row in bundle.provenance['bindings'] if row['object']==syn.name and row['variables']==['gain'])
 before=[];margins=[];hard=[];spikes=[]
 for tick in range(4):
  if anchors is not None and p['tbptt_window'] and tick and tick%p['tbptt_window']==0:z=anchors['before'][tick].copy()
  before.append(z.copy());z[q]=z[h];z[a]=.8*z[a]+.1;z[c]=.8*z[c]+.1+.2*z[q]
  noise=np.array([normal(p['seed'],9,0,2,j,tick,0) for j in range(2)])
  env=dict(h=z[h].copy(),k=z[ss['k']].astype(np.int32) if 'k' in ss else np.array([2,3],np.int32),
           flag=z[ss['flag']].astype(bool) if 'flag' in ss else np.array([False,True]),offset=np.array([1,2],np.int32),mask=np.array([True,False]),
           gain=np.array(weights[bank]),change=fill if 'boolean' in kind else increment,ms=.001,dt=.0002,randn=lambda:noise.copy(),sqrt=np.sqrt)
  exec(stage,env);z[h]=env['h']
  for name in ('k','flag'):
   if name in ss:z[ss[name]]=env[name]
  margin=z[a+c]-.5;event=(margin>0).astype(float);margins.append(margin.copy());hard.append(event.copy())
  if anchors is not None:
   old=anchors['margins'][tick];event=anchors['hard'][tick]+p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(old))**2*(margin-old)
  spikes.append(event.copy());z[a+c]-=.5*event
 logits=np.array(spikes)[:,2:].mean(0)*p['logit_scale'];loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
 return loss,z,np.array(spikes),dict(before=before,margins=margins,hard=hard)

@pytest.mark.parametrize('kind',['integer_state','boolean_state','integer_coefficient','boolean_coefficient'])
@pytest.mark.parametrize('method',['euler','rk2','rk4','heun','milstein'])
@pytest.mark.parametrize('discard',[False,True])
@pytest.mark.parametrize('ranks',[None,2])
@pytest.mark.parametrize('window',[None,2])
def test_typed_synaptic_integrator_original_and_all_vjps(engine,kind,method,discard,ranks,window,monkeypatch):
 mpi(ranks);net,groups,syn,dt,bundle,stage,noisy=model(kind,method,discard,ranks,window,engine)
 out=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).gradients(np.zeros((1,4,1)),[0],**({'noise_sequence':9} if noisy else {}))
 loss,z,spikes,anchors=reference(bundle,groups,syn,kind,stage,noisy,bundle.weights)
 np.testing.assert_allclose(np.asarray(out['final_state'])[0],z,rtol=8e-5,atol=8e-6)
 np.testing.assert_array_equal(np.asarray(out['spikes'])[0],spikes);assert out['loss']==pytest.approx(loss,abs=8e-6)
 for bank,row in enumerate(bundle.weights):
  for index in range(len(row)):
   hi=copy.deepcopy(bundle.weights);lo=copy.deepcopy(bundle.weights);hi[bank][index]+=1e-6;lo[bank][index]-=1e-6
   fd=(reference(bundle,groups,syn,kind,stage,noisy,hi,anchors=anchors)[0]-reference(bundle,groups,syn,kind,stage,noisy,lo,anchors=anchors)[0])/2e-6
   assert out['gradients'][bank][index]==pytest.approx(fd,rel=1e-3,abs=1e-5),(bank,index)
 for index in range(len(bundle.initial_state)):
  if bundle.plan['dynamic']['detached'][index] or index in bundle.plan['dynamic']['integer_states']:
   assert out['initial_state_gradients'][0][index]==0.;continue
  hi=np.array(bundle.initial_state);lo=hi.copy();hi[index]+=1e-6;lo[index]-=1e-6
  fd=(reference(bundle,groups,syn,kind,stage,noisy,bundle.weights,hi,anchors)[0]-reference(bundle,groups,syn,kind,stage,noisy,bundle.weights,lo,anchors)[0])/2e-6
  assert out['initial_state_gradients'][0][index]==pytest.approx(fd,rel=1e-3,abs=1e-5),index
 if noisy:
  values=iter(np.array([normal(bundle.plan['seed'],9,0,2,j,tick,0) for j in range(2)]) for tick in range(4))
  monkeypatch.setattr(np.random,'randn',lambda n:next(values).copy())
 net.run(4*dt,namespace={})
 for group in groups:
  for name,indices in bundle.provenance['neuron_state_layout'][group.name].items():
   np.testing.assert_allclose(group.variables[name].get_value(),np.asarray(out['final_state'])[0,indices],rtol=8e-5,atol=8e-6)
 for name,indices in bundle.provenance['dynamic_state_layout'][syn.name].items():
  np.testing.assert_allclose(syn.variables[name].get_value(),np.asarray(out['final_state'])[0,indices],rtol=8e-5,atol=8e-6)
 assert (out['gpu_dispatches']>0)==(engine!='cpu')

@pytest.mark.parametrize('kind',['integer_state','boolean_state'])
@pytest.mark.parametrize('method',['euler','milstein'])
@pytest.mark.parametrize('ranks',[None,2])
def test_typed_synaptic_hidden_carry_and_fresh_restore(engine,kind,method,ranks,tmp_path):
 mpi(ranks);_,_,syn,_,bundle,_,noisy=model(kind,method,True,ranks,None,engine)
 plan=copy.deepcopy(bundle.plan);plan['trainable']=[False]*len(plan['trainable'])
 inputs=np.zeros((1,4,1));options={'noise_sequence':9} if noisy else {}
 complete=NativeLIFTrainer(plan,weights=bundle.weights,runner=RUNNER).evaluate(inputs,[0],**options)
 trainer=NativeLIFTrainer(plan,weights=bundle.weights,runner=RUNNER)
 first=trainer.step(inputs[:,:2],[0],**options);path=tmp_path/'typed-synaptic';trainer.store(path)
 restored=NativeLIFTrainer(plan,runner=RUNNER);restored.restore(path)
 last=restored.step(inputs[:,2:],[0],initial='carry')
 np.testing.assert_allclose(last['final_state'],complete['final_state'],rtol=8e-5,atol=8e-6)
 np.testing.assert_array_equal(np.concatenate([first['spikes'],last['spikes']],axis=1),complete['spikes'])
 field='k' if kind=='integer_state' else 'flag';indices=bundle.provenance['dynamic_state_layout'][syn.name][field]
 np.testing.assert_array_equal(np.asarray(last['final_state'])[0,indices],np.asarray(complete['final_state'])[0,indices])
 assert restored.clock_tick==4
 if noisy:assert restored.noise_sequence==9 and restored.next_noise_sequence==10
 assert (last['gpu_dispatches']>0)==(engine!='cpu')
