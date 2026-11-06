"""Synapses run_regularly whole-array ownership, callbacks and all VJPs."""
import copy
import brian2 as b
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer,lower_brian_dynamic_training
from test_training_callback_effects import curve
from test_training_typed_callback_effects import increment,fill
from test_training_integer_ir import engine
from test_training_poisson_zero_vjp import mpi
from test_training_stochastic import normal
from test_native_training import RUNNER

CODES={
 'integer':'h=h+.1*gain*change(k)',
 'integer_alias':'temp=k;h=h+.1*gain*change(temp)',
 'boolean':'h=h+.1*gain*change(flag)',
 'coefficient_copy':'h=h+.1*gain*change(offset+1)',
 'float_alias':'temp=h;h=.1*gain*curve(temp)+h',
 'random':'h=h+.1*gain*change(k)+.01*curve(randn())',
}

def model(kind,when,discard,ranks,window,backend):
 b.set_device('runtime');b.start_scope();b.prefs.codegen.target='numpy';b.prefs.codegen.runtime.numpy.discard_units=discard
 dt=.2*b.ms;source=b.SpikeGeneratorGroup(1,[],np.array([])*b.second,dt=dt)
 a=b.NeuronGroup(2,'dv/dt=(.5-v)/ms:1',method='euler',threshold='v>.5',reset='v-=.5',dt=dt)
 c=b.NeuronGroup(2,'dv/dt=(.5-v+q)/ms:1\nq:1',method='euler',threshold='v>.5',reset='v-=.5',dt=dt)
 a.v=[.6,.1];c.v=[.4,.7]
 change=b.Function(fill if kind=='boolean' else increment,arg_units=[1],arg_names=['x'],return_unit=1,stateless=False)
 f=b.Function(curve,arg_units=[1],arg_names=['x'],return_unit=1,stateless=False)
 syn=b.Synapses(a,c,'h:1\nk:integer\nflag:boolean\noffset:integer (constant)\ngain:1 (constant)\nq_post=h:1 (summed)',
                dt=dt,namespace={'change':change,'curve':f})
 syn.connect(i=[1,0],j=[0,1]);syn.h=[.6,.4];syn.k=[2,3];syn.flag=[False,True];syn.offset=[1,2];syn.gain=[.2,.3]
 runner=syn.run_regularly(CODES[kind],when=when)
 net=b.Network(source,a,c,syn)
 bundle=lower_brian_dynamic_training(net,input_group=source,layers=[a,c],backend=backend,mpi_ranks=ranks,
                                    tbptt_window=window,detach_reset=False,seed=7123,trainable_synapse_parameters={syn.name:['gain','h']})
 from brian2.codegen.translation import make_statements
 scalar,vector=make_statements(CODES[kind],{**b.core.functions.DEFAULT_FUNCTIONS,**syn.variables,**syn.namespace},np.float64,optimise=True)
 stage=compile('\n'.join(f'{s.var} {"=" if s.op==":=" else s.op} {s.expr}' for s in [*scalar,*vector]),'<independent optimized NumPy statements>','exec')
 return net,(a,c),syn,runner,dt,bundle,stage

def reference(bundle,groups,syn,runner,kind,when,stage,weights,initial=None,anchors=None):
 p=bundle.plan;z=np.array(bundle.initial_state if initial is None else initial,float)
 if initial is None:
  for index,parameter in enumerate(p['dynamic']['initial_parameters']):
   if parameter is not None:z[index]=weights[parameter[0]][parameter[1]]
 ns=bundle.provenance['neuron_state_layout'];ss=bundle.provenance['dynamic_state_layout'][syn.name]
 a=ns[groups[0].name]['v'];c=ns[groups[1].name]['v'];q=ns[groups[1].name]['q'];h=ss['h']
 bank=next(row['bank'] for row in bundle.provenance['bindings'] if row['object']==syn.name and row['variables']==['gain'])
 domain=bundle.provenance['regular_runner_layout'][runner.name]['noise_domain'];before=[];margins=[];hard=[];spikes=[]
 for tick in range(4):
  if anchors is not None and p['tbptt_window'] and tick and tick%p['tbptt_window']==0:z=anchors['before'][tick].copy()
  before.append(z.copy())
  def update():
   env=dict(h=z[h].copy(),k=z[ss['k']].astype(np.int32) if 'k' in ss else np.array([2,3],np.int32),
            flag=z[ss['flag']].astype(bool) if 'flag' in ss else np.array([False,True]),offset=np.array([1,2],np.int32),gain=np.array(weights[bank]),
            change=fill if kind=='boolean' else increment,curve=curve,
            randn=lambda:np.array([normal(p['seed'],9,0,domain,j,tick,0) for j in range(2)]))
   exec(stage,env);z[h]=env['h']
   for name,index in bundle.provenance['regular_runner_layout'][runner.name]['temporary'].items():
    if name in env:z[index]=np.asarray(env[name]).reshape(-1)[-1]
   for name in ('k','flag'):
    if name in ss:z[ss[name]]=env[name]
  if when=='start':update()
  z[q]=z[h];z[a]=.8*z[a]+.1;z[c]=.8*z[c]+.1+.2*z[q]
  if when=='after_groups':update()
  margin=z[a+c]-.5;event=(margin>0).astype(float);margins.append(margin.copy());hard.append(event.copy())
  if anchors is not None:
   old=anchors['margins'][tick];event=anchors['hard'][tick]+p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(old))**2*(margin-old)
  spikes.append(event.copy());z[a+c]-=.5*event
 logits=np.array(spikes)[:,2:].mean(0)*p['logit_scale'];loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
 return loss,z,np.array(spikes),dict(before=before,margins=margins,hard=hard)

@pytest.mark.parametrize('kind',list(CODES))
@pytest.mark.parametrize('when',['start','after_groups'])
@pytest.mark.parametrize('discard',[False,True])
@pytest.mark.parametrize('ranks',[None,2])
@pytest.mark.parametrize('window',[None,2])
def test_synaptic_regular_effect_original_and_all_vjps(engine,kind,when,discard,ranks,window,monkeypatch):
 mpi(ranks);net,groups,syn,runner,dt,bundle,stage=model(kind,when,discard,ranks,window,engine)
 noisy=kind=='random';out=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).gradients(np.zeros((1,4,1)),[0],**({'noise_sequence':9} if noisy else {}))
 def oracle(weights,initial=None,anchors=None):return reference(bundle,groups,syn,runner,kind,when,stage,weights,initial,anchors)
 loss,z,spikes,anchors=oracle(bundle.weights)
 np.testing.assert_allclose(np.asarray(out['final_state'])[0,:len(z)],z,rtol=8e-5,atol=8e-6)
 np.testing.assert_array_equal(np.asarray(out['spikes'])[0],spikes);assert out['loss']==pytest.approx(loss,abs=8e-6)
 for bank,row in enumerate(bundle.weights):
  for index in range(len(row)):
   hi=copy.deepcopy(bundle.weights);lo=copy.deepcopy(bundle.weights);hi[bank][index]+=1e-6;lo[bank][index]-=1e-6
   fd=(oracle(hi,anchors=anchors)[0]-oracle(lo,anchors=anchors)[0])/2e-6
   assert out['gradients'][bank][index]==pytest.approx(fd,rel=1e-3,abs=1e-5),(bank,index)
 for index in range(len(bundle.initial_state)):
  if bundle.plan['dynamic']['detached'][index] or index in bundle.plan['dynamic']['integer_states']:
   assert out['initial_state_gradients'][0][index]==0.;continue
  hi=np.array(bundle.initial_state);lo=hi.copy();hi[index]+=1e-6;lo[index]-=1e-6
  fd=(oracle(bundle.weights,hi,anchors)[0]-oracle(bundle.weights,lo,anchors)[0])/2e-6
  assert out['initial_state_gradients'][0][index]==pytest.approx(fd,rel=1e-3,abs=1e-5),index
 if noisy:
  domain=bundle.provenance['regular_runner_layout'][runner.name]['noise_domain']
  values=iter(np.array([normal(bundle.plan['seed'],9,0,domain,j,tick,0) for j in range(2)]) for tick in range(4))
  monkeypatch.setattr(np.random,'randn',lambda n:next(values).copy())
 net.run(4*dt,namespace={})
 for group in groups:
  for name,indices in bundle.provenance['neuron_state_layout'][group.name].items():
   np.testing.assert_allclose(group.variables[name].get_value(),np.asarray(out['final_state'])[0,indices],rtol=8e-5,atol=8e-6)
 for name,indices in bundle.provenance['dynamic_state_layout'][syn.name].items():
  np.testing.assert_allclose(syn.variables[name].get_value(),np.asarray(out['final_state'])[0,indices],rtol=8e-5,atol=8e-6)
 assert (out['gpu_dispatches']>0)==(engine!='cpu')

@pytest.mark.parametrize('kind',['integer_alias','boolean','float_alias','random'])
@pytest.mark.parametrize('when',['start','after_groups'])
@pytest.mark.parametrize('ranks',[None,2])
def test_synaptic_regular_effect_carry_and_fresh_restore(engine,kind,when,ranks,tmp_path):
 mpi(ranks);_,_,syn,_,_,bundle,_=model(kind,when,True,ranks,None,engine)
 plan=copy.deepcopy(bundle.plan);plan['trainable']=[False]*len(plan['trainable']);inputs=np.zeros((1,4,1))
 options={'noise_sequence':9} if kind=='random' else {}
 complete=NativeLIFTrainer(plan,weights=bundle.weights,runner=RUNNER).evaluate(inputs,[0],**options)
 trainer=NativeLIFTrainer(plan,weights=bundle.weights,runner=RUNNER);first=trainer.step(inputs[:,:2],[0],**options)
 path=tmp_path/'synaptic-regular';trainer.store(path);restored=NativeLIFTrainer(plan,runner=RUNNER);restored.restore(path)
 last=restored.step(inputs[:,2:],[0],initial='carry')
 np.testing.assert_allclose(last['final_state'],complete['final_state'],rtol=8e-5,atol=8e-6)
 np.testing.assert_array_equal(np.concatenate([first['spikes'],last['spikes']],axis=1),complete['spikes'])
 assert restored.clock_tick==4
 if kind=='random':assert restored.noise_sequence==9 and restored.next_noise_sequence==10
 for name in ('k','flag'):
  indices=bundle.provenance['dynamic_state_layout'][syn.name].get(name)
  if indices:np.testing.assert_array_equal(np.asarray(last['final_state'])[0,indices],np.asarray(complete['final_state'])[0,indices])
 assert (last['gpu_dispatches']>0)==(engine!='cpu')
