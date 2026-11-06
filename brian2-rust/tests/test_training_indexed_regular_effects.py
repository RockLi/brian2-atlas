"""Indexed NumPy callback reads precede every row's ordered scatter."""
import copy
import brian2 as b
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer,lower_brian_dynamic_training
from test_training_callback_effects import curve
from test_training_typed_callback_effects import increment
from test_training_integer_ir import engine
from test_training_poisson_zero_vjp import mpi
from test_native_training import RUNNER

CODES={
 'read':'v=.8*v+.1*curve(peer)+gain',
 'scatter':'peer=curve(peer)+gain;v=.8*v+.1*peer',
 'index':'pick=(pick+1)%4;peer=curve(peer);v=v+.1*peer+gain',
 'integer':'v=.8*v+.1*gain*change(kpeer)',
 'coefficient':'v=.8*v+.1*curve(peer_gain)',
}

def model(kind,discard,ranks,window,backend):
 b.set_device('runtime');b.start_scope();b.prefs.codegen.target='numpy';b.prefs.codegen.runtime.numpy.discard_units=discard
 dt=.2*b.ms;inp=b.SpikeGeneratorGroup(1,[],np.array([])*b.second,dt=dt)
 a=b.NeuronGroup(1,'dv/dt=0/second:1',threshold='v>100',reset='v=0',method='euler',dt=dt)
 g=b.NeuronGroup(4,'dv/dt=0/second:1\nk:integer\npick:integer\npeer:1 (linked)\nkpeer:integer (linked)\ngain:1 (constant)\npeer_gain:1 (linked)',
                 threshold='v>.5',reset='v-=.5',method='euler',dt=dt)
 g.v=[.1,.4,.7,.2];g.k=[1,2,3,4];g.pick=[1,1,0,2];g.gain=[.1,.2,.3,.4]
 g.peer=b.linked_var(g,'v',index='pick');g.kpeer=b.linked_var(g,'k',index='pick');g.peer_gain=b.linked_var(g,'gain',index='pick')
 g.namespace['curve']=b.Function(curve,arg_units=[1],arg_names=['x'],return_unit=1,stateless=False)
 g.namespace['change']=b.Function(increment,arg_units=[1],arg_names=['x'],return_unit=1,stateless=False)
 runner=g.run_regularly(CODES[kind],when='start');net=b.Network(inp,a,g)
 bundle=lower_brian_dynamic_training(net,input_group=inp,layers=[a,g],backend=backend,mpi_ranks=ranks,tbptt_window=window,detach_reset=False,
                                    trainable_neuron_parameters={g.name:['gain']})
 from brian2.codegen.translation import make_statements
 scalar,vector=make_statements(CODES[kind],{**b.core.functions.DEFAULT_FUNCTIONS,**g.variables,**g.namespace},np.float64,optimise=True)
 stage=compile('\n'.join(f'{s.var} {"=" if s.op==":=" else s.op} {s.expr}' for s in [*scalar,*vector]),'<independent indexed NumPy code>','exec')
 return net,g,runner,dt,bundle,stage


def reference(bundle,g,runner,stage,weights,initial=None,anchors=None):
 p=bundle.plan;z=np.array(bundle.initial_state if initial is None else initial,float);layout=bundle.provenance['neuron_state_layout'][g.name]
 bank=next(row['bank'] for row in bundle.provenance['bindings'] if row['object']==g.name and row['variables']==['gain'])
 regular=bundle.provenance['regular_runner_layout'][runner.name];before=[];margins=[];hard=[];spikes=[]
 for tick in range(4):
  if anchors is not None and p['tbptt_window'] and tick and tick%p['tbptt_window']==0:z=anchors['before'][tick].copy()
  before.append(z.copy());v=z[layout['v']].copy();k=z[layout['k']].astype(np.int32);pick=z[layout['pick']].astype(np.int32);gain=np.array(weights[bank])
  if regular['numpy_mode']=='scalar':
   for j in range(4):
    env=dict(v=v[j],k=k[j],pick=pick[j],peer=v[pick[j]],kpeer=k[pick[j]],gain=gain[j],peer_gain=gain[pick[j]],curve=curve,change=increment)
    exec(stage,env)
    for name,index in regular['temporary'].items():
     if name in env:z[index]=env[name]
    for name in regular['numpy_write_order']['vector']:
     if name=='peer':v[env['pick']]=env[name]
     elif name=='v':v[j]=env[name]
     elif name=='pick':pick[j]=env[name]
  else:
   env=dict(v=v.copy(),k=k,pick=pick.copy(),peer=v[pick].copy(),kpeer=k[pick].copy(),gain=gain.copy(),peer_gain=gain[pick].copy(),curve=curve,change=increment)
   for j,row in enumerate(regular.get('staged',[])):
    for name,slot in row['captures'].items():z[slot]=env[name][j]
   exec(stage,env)
   for j,row in enumerate(regular.get('staged',[])):
    for name,slot in row['outputs'].items():z[slot]=env[name][j]
   for name,index in regular['temporary'].items():
    if name in env:z[index]=np.asarray(env[name]).reshape(-1)[-1]
   for name in regular['numpy_write_order']['vector']:
    if name=='peer':v[env['pick']]=env[name]
    elif name=='v':v[:]=env[name]
    elif name=='pick':pick[:]=env[name]
  z[layout['v']]=v;z[layout['pick']]=pick;z[layout['k']]=k
  margin=v-.5;event=(margin>0).astype(float);margins.append(margin.copy());hard.append(event.copy())
  if anchors is not None:
   old=anchors['margins'][tick];event=anchors['hard'][tick]+p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(old))**2*(margin-old)
  spikes.append(event.copy());z[layout['v']]-=.5*event
 logits=np.array(spikes).mean(0)*p['logit_scale'];loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
 return loss,z,np.array(spikes),dict(before=before,margins=margins,hard=hard)


@pytest.mark.parametrize('kind',list(CODES))
@pytest.mark.parametrize('discard',[False,True])
@pytest.mark.parametrize('ranks',[None,2])
@pytest.mark.parametrize('window',[None,2])
def test_indexed_effect_original_and_all_vjps(engine,kind,discard,ranks,window):
 mpi(ranks);net,g,runner,dt,bundle,stage=model(kind,discard,ranks,window,engine)
 out=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).gradients(np.zeros((1,4,1)),[0])
 def oracle(weights,initial=None,anchors=None):return reference(bundle,g,runner,stage,weights,initial,anchors)
 loss,z,spikes,anchors=oracle(bundle.weights)
 np.testing.assert_allclose(out['final_state'][0],z,rtol=8e-5,atol=8e-6);np.testing.assert_array_equal(np.array(out['spikes'])[0,:,1:],spikes);assert out['loss']==pytest.approx(loss,abs=8e-6)
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
 net.run(4*dt,namespace={})
 for name in ['v','k','pick']:np.testing.assert_allclose(np.array(out['final_state'])[0,bundle.provenance['neuron_state_layout'][g.name][name]],g.variables[name].get_value(),rtol=8e-5,atol=8e-6)
 np.testing.assert_array_equal(g.gain[:],[.1,.2,.3,.4])
 assert (out['gpu_dispatches']>0)==(engine!='cpu')


@pytest.mark.parametrize('kind',list(CODES))
@pytest.mark.parametrize('ranks',[None,2])
def test_indexed_effect_carry_and_fresh_restore(engine,kind,ranks,tmp_path):
 mpi(ranks);_,_,_,_,bundle,_=model(kind,True,ranks,None,engine)
 plan=copy.deepcopy(bundle.plan);plan['trainable']=[False]*len(plan['trainable']);x=np.zeros((1,4,1))
 whole=NativeLIFTrainer(plan,weights=bundle.weights,runner=RUNNER).evaluate(x,[0])
 trainer=NativeLIFTrainer(plan,weights=bundle.weights,runner=RUNNER);first=trainer.step(x[:,:2],[0])
 path=tmp_path/'indexed';trainer.store(path);restored=NativeLIFTrainer(plan,runner=RUNNER);restored.restore(path)
 last=restored.step(x[:,2:],[0],initial='carry')
 np.testing.assert_allclose(last['final_state'],whole['final_state'],rtol=8e-5,atol=8e-6)
 np.testing.assert_array_equal(np.concatenate([first['spikes'],last['spikes']],axis=1),whole['spikes'])
 assert restored.clock_tick==4 and (last['gpu_dispatches']>0)==(engine!='cpu')


@pytest.mark.parametrize('ranks',[None,2])
def test_indexed_effect_invalid_output_index_is_atomic(engine,ranks):
 mpi(ranks);net,g,runner,dt,_,_=model('index',True,ranks,None,engine)
 runner.abstract_code='pick=pick+4;peer=curve(peer);v=v+.1*peer+gain'
 inp=next(o for o in net.objects if isinstance(o,b.SpikeGeneratorGroup))
 hidden=next(o for o in net.objects if isinstance(o,b.NeuronGroup) and len(o)==1)
 bundle=lower_brian_dynamic_training(net,input_group=inp,layers=[hidden,g],backend=engine,mpi_ranks=ranks)
 trainer=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER);before=copy.deepcopy(trainer.state)
 with pytest.raises(ValueError,match='[Ii]ndex|outside|domain|nonfinite dynamic GPU result|nonfinite or invalid dynamic (?:Metal|CUDA)(?: MPI)? action'):
  trainer.step(np.zeros((1,1,1)),[0])
 assert trainer.state==before and trainer.neuron_state is None and trainer.clock_tick==0 and trainer.next_noise_sequence==0
 with pytest.raises(b.core.base.BrianObjectException) as caught:net.run(dt,namespace={})
 assert isinstance(caught.value.__cause__,IndexError)
