"""Subgroup callbacks receive indexed copies; only explicit writes persist."""
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
 'hidden':'v=v+.1*gain*curve(a)',
 'integer':'v=v+.1*gain*change(k)',
 'boolean':'v=v+.1*gain*change(flag)',
 'alias':'temp=a;a=curve(temp)+temp;v=v+.1*gain*a',
 'coefficient':'v=v+.1*curve(gain)',
 'noise':'temp=curve(a);v=v+.1*gain*temp+.01*curve(randn())',
}

def model(kind,discard,ranks,window,backend):
 b.set_device('runtime');b.start_scope();b.prefs.codegen.target='numpy';b.prefs.codegen.runtime.numpy.discard_units=discard
 dt=.2*b.ms;source=b.SpikeGeneratorGroup(1,[],np.array([])*b.second,dt=dt)
 a=b.NeuronGroup(1,'dv/dt=0/second:1',threshold='v>100',reset='v=0',method='euler',dt=dt)
 c=b.NeuronGroup(4,'dv/dt=.1*a/ms:1\na:1\nk:integer\nflag:boolean\ngain:1 (constant)',
                 threshold='v>.5',reset='v-=.5',method='euler',dt=dt)
 c.v=[.1,.4,.7,.2];c.a=[.2,.6,.4,.3];c.k=[1,2,3,4];c.flag=[True,False,True,False];c.gain=[.1,.2,.3,.4]
 c.namespace['curve']=b.Function(curve,arg_units=[1],arg_names=['x'],return_unit=1,stateless=False)
 c.namespace['change']=b.Function(fill if kind=='boolean' else increment,arg_units=[1],arg_names=['x'],return_unit=1,stateless=False)
 sub=c[1:3];runner=sub.run_regularly(CODES[kind],when='start')
 net=b.Network(source,a,c)
 bundle=lower_brian_dynamic_training(net,input_group=source,layers=[a,c],backend=backend,mpi_ranks=ranks,tbptt_window=window,
                                    detach_reset=False,seed=7123,trainable_neuron_parameters={c.name:['gain']})
 from brian2.codegen.translation import make_statements
 scalar,vector=make_statements(CODES[kind],{**b.core.functions.DEFAULT_FUNCTIONS,**sub.variables,**sub.namespace},np.float64,optimise=True)
 stage=compile('\n'.join(f'{s.var} {"=" if s.op==":=" else s.op} {s.expr}' for s in [*scalar,*vector]),'<independent NumPy subgroup code>','exec')
 return net,c,runner,dt,bundle,stage


def reference(bundle,g,runner,kind,stage,weights,initial=None,anchors=None):
 p=bundle.plan;z=np.array(bundle.initial_state if initial is None else initial,float);layout=bundle.provenance['neuron_state_layout'][g.name]
 bank=next(row['bank'] for row in bundle.provenance['bindings'] if row['object']==g.name and row['variables']==['gain'])
 regular=bundle.provenance['regular_runner_layout'][runner.name];before=[];margins=[];hard=[];spikes=[]
 for tick in range(4):
  if anchors is not None and p['tbptt_window'] and tick and tick%p['tbptt_window']==0:z=anchors['before'][tick].copy()
  before.append(z.copy())
  env={name:z[slots][1:3].copy() for name,slots in layout.items() if name in ('v','a','k','flag')}
  env['k']=env['k'].astype(np.int32);env['flag']=env['flag'].astype(bool)
  env.update(gain=np.array(weights[bank])[1:3].copy(),curve=curve,change=fill if kind=='boolean' else increment,
             randn=lambda:np.array([normal(p['seed'],9,0,regular['noise_domain'],j,tick,0) for j in range(2)]))
  exec(stage,env)
  for name in regular['numpy_write_order']['vector']:
   slots=np.array(layout[name])[1:3];z[slots]=env[name]
  for name,index in regular['temporary'].items():
   if name in env:z[index]=np.asarray(env[name]).reshape(-1)[-1]
  z[layout['v']]+=.02*z[layout['a']]
  margin=z[layout['v']]-.5;event=(margin>0).astype(float);margins.append(margin.copy());hard.append(event.copy())
  if anchors is not None:
   old=anchors['margins'][tick];event=anchors['hard'][tick]+p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(old))**2*(margin-old)
  spikes.append(event.copy());z[layout['v']]-=.5*event
 logits=np.array(spikes).mean(0)*p['logit_scale'];loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
 return loss,z,np.array(spikes),dict(before=before,margins=margins,hard=hard)


@pytest.mark.parametrize('kind',list(CODES))
@pytest.mark.parametrize('discard',[False,True])
@pytest.mark.parametrize('ranks',[None,2])
@pytest.mark.parametrize('window',[None,2])
def test_subgroup_effect_original_and_all_vjps(engine,kind,discard,ranks,window,monkeypatch):
 mpi(ranks);net,g,runner,dt,bundle,stage=model(kind,discard,ranks,window,engine)
 out=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).gradients(np.zeros((1,4,1)),[0],**({'noise_sequence':9} if kind=='noise' else {}))
 def oracle(weights,initial=None,anchors=None):return reference(bundle,g,runner,kind,stage,weights,initial,anchors)
 loss,z,spikes,anchors=oracle(bundle.weights)
 np.testing.assert_allclose(out['final_state'][0],z,rtol=8e-5,atol=8e-6);np.testing.assert_array_equal(np.array(out['spikes'])[0,:,1:],spikes)
 assert out['loss']==pytest.approx(loss,abs=8e-6)
 for bank,row in enumerate(bundle.weights):
  for index in range(len(row)):
   hi=copy.deepcopy(bundle.weights);lo=copy.deepcopy(bundle.weights);hi[bank][index]+=1e-6;lo[bank][index]-=1e-6
   fd=(oracle(hi,anchors=anchors)[0]-oracle(lo,anchors=anchors)[0])/2e-6
   assert out['gradients'][bank][index]==pytest.approx(fd,rel=1e-3,abs=1e-5),(bank,index)
 for index in range(len(bundle.initial_state)):
  if bundle.plan['dynamic']['detached'][index] or index in bundle.plan['dynamic']['integer_states']:
   assert out['initial_state_gradients'][0][index]==0.;continue
  hi=np.array(bundle.initial_state);lo=hi.copy();hi[index]+=1e-6;lo[index]-=1e-6
  fd=(oracle(bundle.weights,hi,anchors=anchors)[0]-oracle(bundle.weights,lo,anchors=anchors)[0])/2e-6
  assert out['initial_state_gradients'][0][index]==pytest.approx(fd,rel=1e-3,abs=1e-5),index
 if kind=='noise':
  draws=iter(np.array([normal(bundle.plan['seed'],9,0,bundle.provenance['regular_runner_layout'][runner.name]['noise_domain'],j,tick,0) for j in range(2)]) for tick in range(4))
  monkeypatch.setattr(np.random,'randn',lambda n:next(draws).copy())
 net.run(4*dt,namespace={})
 for name,slots in bundle.provenance['neuron_state_layout'][g.name].items():
  np.testing.assert_allclose(np.array(out['final_state'])[0,slots],g.variables[name].get_value(),rtol=8e-5,atol=8e-6)
 np.testing.assert_array_equal(g.gain[:], [.1,.2,.3,.4])
 assert (out['gpu_dispatches']>0)==(engine!='cpu')


@pytest.mark.parametrize('kind',['hidden','integer','boolean','alias','coefficient','noise'])
@pytest.mark.parametrize('ranks',[None,2])
def test_subgroup_effect_carry_and_fresh_restore(engine,kind,ranks,tmp_path):
 mpi(ranks);_,_,_,_,bundle,_=model(kind,True,ranks,None,engine)
 plan=copy.deepcopy(bundle.plan);plan['trainable']=[False]*len(plan['trainable']);x=np.zeros((1,4,1))
 options={'noise_sequence':9} if kind=='noise' else {}
 whole=NativeLIFTrainer(plan,weights=bundle.weights,runner=RUNNER).evaluate(x,[0],**options)
 trainer=NativeLIFTrainer(plan,weights=bundle.weights,runner=RUNNER);first=trainer.step(x[:,:2],[0],**options)
 path=tmp_path/'subgroup';trainer.store(path);restored=NativeLIFTrainer(plan,runner=RUNNER);restored.restore(path)
 last=restored.step(x[:,2:],[0],initial='carry')
 np.testing.assert_allclose(last['final_state'],whole['final_state'],rtol=8e-5,atol=8e-6)
 np.testing.assert_array_equal(np.concatenate([first['spikes'],last['spikes']],axis=1),whole['spikes'])
 assert restored.clock_tick==4
 if kind=='noise':assert restored.noise_sequence==9 and restored.next_noise_sequence==10
 assert (last['gpu_dispatches']>0)==(engine!='cpu')
