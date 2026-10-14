"""Original NumPy conditional event copies, scatter and scalar fallback."""
import copy
import brian2 as b
import numpy as np
import pytest
from brian2_rust import lower_brian_dynamic_training,NativeLIFTrainer
from test_training_callback_effects import curve
from test_training_integer_ir import engine
from test_training_poisson_zero_vjp import mpi
from test_native_training import RUNNER
from test_training_event_noise import draw

CODES={
 'copy':'v_post=curve(a_post)+a_post',
 'add':'v_post+=curve(a_post)+a_post',
 'coefficient':'v_post+=curve(gain)+gain',
 'scalar':'h=curve(v_post);v_post+=gain*h',
 'temporary':'temp=a_post;v_post+=curve(temp)+temp',
 'hidden':'a_post=curve(v_post)+v_post;v_post+=gain*h',
 'noise':'v_post+=curve(gain)+.05*randn()',
 'staged':'temp=a_post;v_post+=curve(temp);h=temp+v_post',
}

def model(kind,repeat,discard,ranks,window,delay,backend):
 b.set_device('runtime');b.start_scope();b.prefs.codegen.target='numpy';b.prefs.codegen.runtime.numpy.discard_units=discard
 dt=.2*b.ms;source=b.SpikeGeneratorGroup(1,[0,0,0,0],np.array([0,1,3,5])*dt,dt=dt,name='guarded_event_input')
 hidden=b.NeuronGroup(1,'dv/dt=0/second:1',method='euler',threshold='v>100',reset='v=0',dt=dt,name='guarded_event_hidden')
 g=b.NeuronGroup(2,'dv/dt=0/second:1 (unless refractory)\na:1',method='euler',threshold='v>.5',reset='v-=.5',
                 refractory=.4*b.ms,dt=dt,name='guarded_event_neurons')
 g.v=[.4,.7];g.a=[.3,.5];g.lastspike=[0.,-1.]*b.second;g.not_refractory=[False,True]
 f=b.Function(curve,arg_units=[1],arg_names=['x'],return_unit=1,stateless=False)
 syn=b.Synapses(source,g,'h:1\ngain:1 (constant)',on_pre=CODES[kind],dt=dt,namespace={'curve':f},name='guarded_event_syn')
 syn.connect(i=[0,0],j=[0,0] if repeat else [0,1]);syn.h=[.2,.3];syn.gain=[.15,.25];syn.delay=delay*dt
 net=b.Network(source,hidden,g,syn)
 bundle=lower_brian_dynamic_training(net,input_group=source,layers=[hidden,g],backend=backend,mpi_ranks=ranks,
                                    tbptt_window=window,detach_reset=False,seed=3129,trainable_synapse_parameters={syn.name:['h','gain']})
 x=np.zeros((1,6,1));x[0,[0,1,3,5],0]=1.
 return net,g,syn,dt,bundle,x

@pytest.mark.parametrize('kind',list(CODES))
@pytest.mark.parametrize('repeat',[False,True])
@pytest.mark.parametrize('discard',[False,True])
@pytest.mark.parametrize('delay',[0,1])
@pytest.mark.parametrize('ranks',[None,2])
def test_guarded_event_original_and_restored_carry(engine,kind,repeat,discard,delay,ranks,tmp_path,monkeypatch):
 mpi(ranks);net,g,syn,dt,bundle,x=model(kind,repeat,discard,ranks,None,delay,engine)
 bundle.plan['trainable']=[False]*len(bundle.weights)
 trainer=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER)
 for start,length in [(0,1),(1,2),(3,3)]:
  out=trainer.step(x[:,start:start+length],[0],initial='carry' if start else None)
  if kind=='noise':install_original_noise(monkeypatch,bundle,g,syn,dt,repeat,delay,lambda tick:trainer.noise_sequence)
  net.run(length*dt,namespace={})
  for name,slots in bundle.provenance['neuron_state_layout'][g.name].items():
   if name.startswith('__'):continue
   np.testing.assert_allclose(np.asarray(out['final_state'])[0,slots],g.variables[name].get_value(),rtol=7e-5,atol=7e-6)
  for name,slots in bundle.provenance['dynamic_state_layout'][syn.name].items():
   np.testing.assert_allclose(np.asarray(out['final_state'])[0,slots],syn.variables[name].get_value(),rtol=7e-5,atol=7e-6)
  np.testing.assert_array_equal(syn.gain[:],[.15,.25])
  path=tmp_path/('checkpoint-'+str(start));trainer.store(path)
  restored=NativeLIFTrainer(trainer.plan,runner=RUNNER);restored.restore(path)
  assert restored.neuron_state==trainer.neuron_state and restored.clock_tick==trainer.clock_tick
  trainer=restored
 assert (out['gpu_dispatches']>0)==(engine!='cpu')


def reference(bundle,g,syn,kind,repeat,delay,weights,initial=None,anchors=None):
 """Explicit NumPy event ownership and detached refractory control, no SSA."""
 p=bundle.plan;z=np.array(bundle.initial_state if initial is None else initial,float)
 if initial is None:
  for slot,parameter in enumerate(p['dynamic']['initial_parameters']):
   if parameter is not None:z[slot]=weights[parameter[0]][parameter[1]]
 layout=bundle.provenance['neuron_state_layout'][g.name];v=layout['v'];a=layout['a'];counter=layout['__refractory_ticks']
 activity=bundle.provenance['refractory_activity_layout'][g.name]
 banks={name:next(row['bank'] for row in bundle.provenance['bindings'] if row['object']==syn.name and row['variables']==[name]) for name in ('h','gain')}
 h=bundle.provenance['dynamic_state_layout'][syn.name].get('h');gain=np.array(weights[banks['gain']]);posts=np.array([0,0] if repeat else [0,1])
 path=syn.pre.name;snapshots=bundle.provenance['event_callback_snapshots'].get(path,[])
 before=[];margins=[];hard_spikes=[];free_rows=[];spikes=[]
 for tick in range(6):
  if anchors is not None and p['tbptt_window'] and tick and tick%p['tbptt_window']==0:z=anchors['before'][tick].copy()
  before.append(z.copy());free=z[counter]<.5;z[counter]=np.maximum(z[counter]-1,0.)
  if anchors is not None:free=anchors['free'][tick]
  free_rows.append(free.copy());margin=z[v]-.5;hard=(margin>0)&free;gate=hard.astype(float)
  if anchors is not None:
   hard=anchors['hard'][tick];old=anchors['margins'][tick]
   gate=hard.astype(float)+free*p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(old))**2*(margin-old)
  margins.append(margin.copy());hard_spikes.append(hard.copy());spikes.append(gate.copy());flag=free&~hard;z[activity]=flag
  for row in snapshots:z[row['cache']]=z[row['source']]
  queues=bundle.provenance['delay_queues'][path]['new']
  event=np.array([z[row['states'][0]] for row in queues]) if delay else np.full(2,float(tick in (0,1,3,5)))
  old_a=z[a].copy()
  for edge,post in enumerate(posts):
   amplitude=event[edge];old_v=z[v[post]]
   if kind=='copy':new_v=1.6*old_a[post]  # vectorised plain assignment has no mask
   elif kind in ('add','temporary'):new_v=old_v+(1.8*old_a[post] if flag[post] else 0.)
   elif kind=='staged':new_v=old_v+(.8*old_a[post] if flag[post] else 0.)
   elif kind=='coefficient':new_v=old_v+(1.8*gain[edge] if flag[post] else 0.)
   elif kind=='noise':
    domain=bundle.provenance['scheduled_noise_domains'][path]
    noise=draw('randn',p['seed'],7,0,domain,edge,tick-delay,0,0) if tick>=delay else 0.
    new_v=old_v+(.8*gain[edge]+.05*noise if flag[post] else 0.)
   elif kind=='scalar':
    # curve saves an alias before *=; immutable scalar rebinding leaves the
    # saved scalar unchanged, unlike its borrowed array execution.
    old_h=z[h[edge]];new_h=old_v;z[h[edge]]=old_h+amplitude*(new_h-old_h)
    new_v=old_v+(gain[edge]*new_h if flag[post] else 0.)
   else:
    old=z[a[post]];z[a[post]]=old+amplitude*(2.*old_v-old)
    weight=weights[banks['h']][edge] if h is None else z[h[edge]]
    new_v=old_v+(gain[edge]*weight if flag[post] else 0.)
   z[v[post]]=old_v+amplitude*(new_v-old_v)
  if kind=='staged':
   later=path+'::numpy-stage:1'
   for row in bundle.provenance['event_callback_snapshots'][later]:z[row['cache']]=z[row['source']]
   later_queues=bundle.provenance['delay_queues'][later]['new']
   later_event=np.array([z[row['states'][0]] for row in later_queues]) if delay else event
   for edge,post in enumerate(posts):
    old_h=z[h[edge]];new_h=old_a[post]+z[v[post]];z[h[edge]]=old_h+later_event[edge]*(new_h-old_h)
   for row in later_queues:
    slots=row['states']
    if slots:z[slots[:-1]]=z[slots[1:]];z[slots[-1]]=float(tick in (0,1,3,5))
  for row in queues:
   slots=row['states']
   if slots:z[slots[:-1]]=z[slots[1:]];z[slots[-1]]=float(tick in (0,1,3,5))
  z[v]-=.5*gate;z[counter]=np.where(hard,1.,z[counter])
 logits=np.array(spikes).mean(0)*p['logit_scale'];loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
 return loss,z,np.array(spikes),dict(before=before,margins=margins,hard=hard_spikes,free=free_rows)


@pytest.mark.parametrize('kind',list(CODES))
@pytest.mark.parametrize('repeat',[False,True])
@pytest.mark.parametrize('delay',[0,1])
@pytest.mark.parametrize('ranks',[None,2])
@pytest.mark.parametrize('window',[None,2])
def test_guarded_event_all_parameters_and_initial_vjps(engine,kind,repeat,delay,ranks,window,monkeypatch):
 mpi(ranks);net,g,syn,dt,bundle,x=model(kind,repeat,True,ranks,window,delay,engine)
 out=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).gradients(x,[0],**({'noise_sequence':7} if kind=='noise' else {}))
 loss,z,spikes,anchors=reference(bundle,g,syn,kind,repeat,delay,bundle.weights)
 np.testing.assert_allclose(out['final_state'][0],z,rtol=7e-5,atol=7e-6)
 np.testing.assert_array_equal(np.asarray(out['spikes'])[0,:,1:],spikes);assert out['loss']==pytest.approx(loss,abs=7e-6)
 for bank,row in enumerate(bundle.weights):
  for index in range(len(row)):
   hi=copy.deepcopy(bundle.weights);lo=copy.deepcopy(bundle.weights);hi[bank][index]+=1e-6;lo[bank][index]-=1e-6
   fd=(reference(bundle,g,syn,kind,repeat,delay,hi,anchors=anchors)[0]-reference(bundle,g,syn,kind,repeat,delay,lo,anchors=anchors)[0])/2e-6
   assert out['gradients'][bank][index]==pytest.approx(fd,rel=9e-4,abs=9e-6),(bank,index)
 for index in range(len(bundle.initial_state)):
  if bundle.plan['dynamic']['detached'][index] or index in bundle.plan['dynamic']['binary_states']:
   assert out['initial_state_gradients'][0][index]==0.;continue
  hi=np.array(bundle.initial_state);lo=hi.copy();hi[index]+=1e-6;lo[index]-=1e-6
  fd=(reference(bundle,g,syn,kind,repeat,delay,bundle.weights,hi,anchors)[0]-reference(bundle,g,syn,kind,repeat,delay,bundle.weights,lo,anchors)[0])/2e-6
  assert out['initial_state_gradients'][0][index]==pytest.approx(fd,rel=9e-4,abs=9e-6),index
 if kind=='noise':install_original_noise(monkeypatch,bundle,g,syn,dt,repeat,delay,lambda tick:7)
 net.run(6*dt,namespace={})
 for name in ('v','a'):np.testing.assert_allclose(z[bundle.provenance['neuron_state_layout'][g.name][name]],g.variables[name].get_value(),rtol=7e-5,atol=7e-6)
 assert (out['gpu_dispatches']>0)==(engine!='cpu')


def install_original_noise(monkeypatch,bundle,g,syn,dt,repeat,delay,sequence):
 posts=np.array([0,0] if repeat else [0,1]);domain=bundle.provenance['scheduled_noise_domains'][syn.pre.name]
 def randn(size):
  tick=int(round(float(g.clock.variables['t'].get_value()[0])/float(dt)));emission=tick-delay
  active=np.flatnonzero(np.asarray(g.not_refractory[:])[posts]);assert size==len(active)
  return np.array([draw('randn',bundle.plan['seed'],sequence(emission),0,domain,int(edge),emission,0,0) for edge in active])
 monkeypatch.setattr(np.random,'randn',randn)


@pytest.mark.parametrize('scalar',[False,True])
def test_empty_event_mask_preserves_vector_scalar_work_and_scalar_if(scalar):
 from test_training_callback_effects import invalid_curve
 net,g,syn,dt,_,x=model('scalar' if scalar else 'add',False,True,None,None,0,'cpu')
 syn.pre.code=('h=v_post;' if scalar else '')+'v_post+=bad(.125)'
 syn.namespace['bad']=b.Function(invalid_curve,arg_units=[1],arg_names=['x'],return_unit=1,stateless=False)
 source=next(obj for obj in net.objects if obj.name=='guarded_event_input');hidden=next(obj for obj in net.objects if obj.name=='guarded_event_hidden')
 bundle=lower_brian_dynamic_training(net,input_group=source,layers=[hidden,g])
 trainer=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER)
 if scalar:
  # Both conditions are false after the tick-0 threshold: scalar if skips bad.
  trainer.step(x[:,:1],[0]);net.run(dt,namespace={})
  before=copy.deepcopy((trainer.plan,trainer.state,trainer.neuron_state,trainer.clock_tick,trainer.next_noise_sequence))
  with pytest.raises(ValueError):trainer.step(x[:,1:4],[0],initial='carry')
  assert (trainer.plan,trainer.state,trainer.neuron_state,trainer.clock_tick,trainer.next_noise_sequence)==before
  with pytest.raises(b.BrianObjectException):net.run(3*dt,namespace={})
 else:
  # The empty index vector does not skip a scalar RHS of numpy.add.at.
  before=copy.deepcopy((trainer.plan,trainer.state,trainer.neuron_state,trainer.clock_tick,trainer.next_noise_sequence))
  with pytest.raises(ValueError):trainer.step(x[:,:1],[0])
  assert (trainer.plan,trainer.state,trainer.neuron_state,trainer.clock_tick,trainer.next_noise_sequence)==before
  with pytest.raises(b.BrianObjectException):net.run(dt,namespace={})


@pytest.mark.parametrize('kind',['add','scalar','noise','staged'])
@pytest.mark.parametrize('ranks',[None,2])
def test_guarded_event_delay_rebuild_keeps_masks_and_pending_arrivals(engine,kind,ranks,tmp_path,monkeypatch):
 mpi(ranks);net,g,syn,dt,bundle,x=model(kind,True,True,ranks,None,1,engine)
 bundle.plan['trainable']=[False]*len(bundle.weights)
 trainer=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER)
 for start,length,delay in [(0,1,1),(1,2,0),(3,3,1)]:
  if start:
   before=copy.deepcopy((trainer.state,trainer.clock_tick,trainer.next_noise_sequence))
   trainer.update_delays({syn.pre.name:float(delay*dt)});syn.delay=delay*dt
   assert (trainer.state,trainer.clock_tick,trainer.next_noise_sequence)==before
  out=trainer.step(x[:,start:start+length],[0],initial='carry' if start else None)
  if kind=='noise':install_original_noise(monkeypatch,bundle,g,syn,dt,True,delay,lambda tick:trainer.noise_sequence)
  net.run(length*dt,namespace={})
  for name in ('v','a'):
   np.testing.assert_allclose(np.asarray(out['final_state'])[0,bundle.provenance['neuron_state_layout'][g.name][name]],g.variables[name].get_value(),rtol=7e-5,atol=7e-6)
  for name,slots in bundle.provenance['dynamic_state_layout'][syn.name].items():
   np.testing.assert_allclose(np.asarray(out['final_state'])[0,slots],syn.variables[name].get_value(),rtol=7e-5,atol=7e-6)
  checkpoint=tmp_path/('changed-delay-'+str(start));trainer.store(checkpoint)
  restored=NativeLIFTrainer(trainer.plan,runner=RUNNER);restored.restore(checkpoint)
  assert restored.neuron_state==trainer.neuron_state and restored.clock_tick==trainer.clock_tick
  trainer=restored
 assert (out['gpu_dispatches']>0)==(engine!='cpu')


@pytest.mark.parametrize('kind',['noise','staged'])
def test_guarded_event_eight_ranks_preserve_mask_and_replay_vjps(engine,kind,monkeypatch):
 test_guarded_event_all_parameters_and_initial_vjps(engine,kind,True,1,8,2,monkeypatch)
