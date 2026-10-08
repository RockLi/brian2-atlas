"""Cross-edge scatter re-reads: NumPy stage order, operational VJPs and queues."""
import copy
import brian2 as b
import numpy as np
import pytest
from brian2_rust import lower_brian_dynamic_training,NativeLIFTrainer
from test_training_integer_ir import engine
from test_training_poisson_zero_vjp import mpi
from test_training_event_callback_effects import model as original_model
from test_native_training import RUNNER

CODES={
 'reread':'v_post+=gain*curve(v_pre);h=v_pre',
 'twice':'v_post+=gain*curve(v_pre);h=v_pre;v_post+=.5*gain*curve(v_pre);h=v_pre',
 'callback_reread':'v_post+=gain*curve(v_pre);h=curve(v_pre)',
 'temporary':'temp=curve(v_pre);v_post+=gain*temp;h=temp+v_pre',
 'alias_mutate':'temp=curve(v_pre);alias=temp;v_post+=gain*temp;temp*=.8;h=alias+v_pre',
 'random_temporary':'temp=curve(randn());v_post+=gain*temp;h=temp+v_pre+.05*randn()',
}


def model(kind,repeat,discard,ranks,window,delay,backend):
 net,source,groups,syn,dt,_,x=original_model('autapse',repeat,discard,ranks,window,delay,backend)
 syn.pre.code=CODES[kind]
 bundle=lower_brian_dynamic_training(net,input_group=source,layers=groups,backend=backend,mpi_ranks=ranks,
  tbptt_window=window,detach_reset=False,trainable_synapse_parameters={syn.name:['h','a','gain']})
 return net,source,groups,syn,dt,bundle,x


def reference(bundle,groups,syn,kind,repeat,weights,initial=None,anchors=None,event_delay=False):
 p=bundle.plan;z=np.array(bundle.initial_state if initial is None else initial,float)
 if initial is None:
  for slot,parameter in enumerate(p['dynamic']['initial_parameters']):
   if parameter is not None:z[slot]=weights[parameter[0]][parameter[1]]
 layouts=bundle.provenance['neuron_state_layout'];av=layouts[groups[0].name]['v'];cv=layouts[groups[1].name]['v'];q=layouts[groups[1].name]['q']
 h=bundle.provenance['dynamic_state_layout'][syn.name]['h']
 bank=next(row['bank'] for row in bundle.provenance['bindings'] if row['object']==syn.name and row['variables']==['gain'])
 gain=np.array(weights[bank]);posts=[0,0] if repeat else [0,1]
 stages=bundle.provenance['event_callback_stage_groups'][syn.pre.name];spikes=[];margins=[];before=[]
 if event_delay:
  # Runtime routing first rebuilds the initial empty histories into private
  # routes. This discrete boundary drops empty imported queue cells and their
  # adjoints; the reference retains virtual queue addresses for tick execution.
  for name in stages:
   for row in bundle.provenance['delay_queues'][name]['new']:z[row['states']]=0.
 for tick in range(4):
  if anchors is not None and p['tbptt_window'] and tick and tick%p['tbptt_window']==0:z=anchors[1][tick].copy()
  before.append(z.copy());z[q]=0.
  for edge,post in enumerate(posts):z[q[post]]+=z[h[edge]]
  z[av]=.8*z[av]+.1;z[cv]=.8*z[cv]+.1+.2*z[q]
  margin=z[av+cv]-.5;gate=(margin>0).astype(float)
  if anchors is not None:
   old=anchors[0][tick];gate=(old>0).astype(float)+p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(old))**2*(margin-old)
  margins.append(margin.copy());spikes.append(gate.copy())
  original_pre=z[cv].copy()
  for stage,name in enumerate(stages):
   for row in bundle.provenance['event_callback_snapshots'].get(name,[]):z[row['cache']]=z[row['source']]
   routes=bundle.provenance['delay_queues'][name]['new']
   event=np.array([z[row['states'][0]] if row['states'] else gate[2+row['edge']] for row in routes])
   pre=z[cv].copy()
   if kind=='random_temporary':
    from test_training_event_noise import draw
    domain=bundle.provenance['scheduled_noise_domains'][syn.pre.name]
    delay_ticks=routes[0]['delay']
    temporary=np.array([.8*draw('randn',p['seed'],9,0,domain,edge,tick-delay_ticks,0,0) for edge in range(2)])
    second=np.array([draw('randn',p['seed'],9,0,domain,edge,tick-delay_ticks,0,1) for edge in range(2)])
   else:temporary=.8*original_pre
   if stage==0:
    delta=temporary if kind=='random_temporary' else .8*pre
    for edge,post in enumerate(posts):z[cv[post]]+=event[edge]*gain[edge]*delta[edge]
   else:
    factor=.8 if kind=='callback_reread' else 1.
    value=factor*pre
    if kind=='temporary':value=value+temporary
    elif kind=='alias_mutate':value=value+.8*temporary
    elif kind=='random_temporary':value=value+temporary+.05*second
    if event_delay and stage==len(stages)-1:
     delay_slots=bundle.provenance['pathway_state_layout'][syn.pre.name]['delay']
     value=value+.1*z[delay_slots]/.001
    for edge in range(2):z[h[edge]]+=event[edge]*(value[edge]-z[h[edge]])
    if event_delay and stage==len(stages)-1:
     z[delay_slots]+=event*float(syn.clock.dt)
    if kind=='twice' and stage==1:
     for edge,post in enumerate(posts):z[cv[post]]+=event[edge]*.5*gain[edge]*.8*pre[edge]
   for row in routes:
    history=row['states']
    if history:z[history[:-1]]=z[history[1:]];z[history[-1]]=gate[2+row['edge']]
  z[av+cv]-=.5*gate
 spikes=np.array(spikes);logits=spikes[:,2:].mean(0)*p['logit_scale']
 loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
 return loss,z,spikes,(np.array(margins),np.array(before))


@pytest.mark.parametrize('kind',list(CODES))
@pytest.mark.parametrize('repeat',[False,True])
@pytest.mark.parametrize('discard',[False,True])
@pytest.mark.parametrize('ranks',[None,2])
@pytest.mark.parametrize('window',[None,2])
@pytest.mark.parametrize('delay',[0,1])
def test_staged_scatter_actual_brian_and_all_vjps(engine,kind,repeat,discard,ranks,window,delay,monkeypatch):
 mpi(ranks);net,source,groups,syn,dt,bundle,x=model(kind,repeat,discard,ranks,window,delay,engine)
 result=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).gradients(x,[0],**({'noise_sequence':9} if kind=='random_temporary' else {}))
 loss,z,spikes,anchors=reference(bundle,groups,syn,kind,repeat,bundle.weights)
 np.testing.assert_allclose(result['final_state'][0],z,rtol=7e-5,atol=7e-6)
 np.testing.assert_array_equal(result['spikes'][0],spikes);assert result['loss']==pytest.approx(loss,abs=7e-6)
 for bank,row in enumerate(bundle.weights):
  for j in range(len(row)):
   hi=copy.deepcopy(bundle.weights);lo=copy.deepcopy(bundle.weights);hi[bank][j]+=1e-6;lo[bank][j]-=1e-6
   fd=(reference(bundle,groups,syn,kind,repeat,hi,anchors=anchors)[0]-reference(bundle,groups,syn,kind,repeat,lo,anchors=anchors)[0])/2e-6
   assert result['gradients'][bank][j]==pytest.approx(fd,rel=9e-4,abs=9e-6)
 for j in range(len(bundle.initial_state)):
  hi=np.array(bundle.initial_state,float);lo=hi.copy();hi[j]+=1e-6;lo[j]-=1e-6
  fd=(reference(bundle,groups,syn,kind,repeat,bundle.weights,initial=hi,anchors=anchors)[0]-reference(bundle,groups,syn,kind,repeat,bundle.weights,initial=lo,anchors=anchors)[0])/2e-6
  assert result['initial_state_gradients'][0][j]==pytest.approx(fd,rel=9e-4,abs=9e-6)
 if kind=='random_temporary':
  from test_training_event_noise import draw
  queues={};domain=bundle.provenance['scheduled_noise_domains'][syn.pre.name]
  for tick in range(4):
   for emission in range(tick+1):
    if tick-emission!=delay:continue
    row=[edge for edge in range(2) if spikes[emission,2+edge]]
    if row:
     queues.setdefault(tick,[]).extend([np.array([draw('randn',bundle.plan['seed'],9,0,domain,edge,emission,0,stream) for edge in row]) for stream in (0,1)])
  for tick in range(4):
   values=iter(queues.get(tick,[]))
   def replay(*shape):
    value=next(values);assert shape==(len(value),);return value.copy()
   monkeypatch.setattr(np.random,'randn',replay);net.run(dt,namespace={})
   assert next(values,None) is None
 else:net.run(4*dt,namespace={})
 for group in groups:
  for name,slots in bundle.provenance['neuron_state_layout'][group.name].items():
   np.testing.assert_allclose(np.asarray(result['final_state'])[0,slots],group.variables[name].get_value(),rtol=7e-5,atol=7e-6)
 for name,slots in bundle.provenance['dynamic_state_layout'][syn.name].items():
  np.testing.assert_allclose(np.asarray(result['final_state'])[0,slots],syn.variables[name].get_value(),rtol=7e-5,atol=7e-6)
 assert len(bundle.provenance['event_callback_stage_groups'][syn.pre.name])==(3 if kind=='twice' else 2)
 if kind=='random_temporary':
  for name in bundle.provenance['event_callback_stage_groups'][syn.pre.name]:
   path=next(row for row in bundle.plan['dynamic']['delay_layout']['paths'] if row['name']==name)
   assert all(bundle.plan['dynamic']['actions'][row['event']]['noise_domain']==bundle.provenance['scheduled_noise_domains'][syn.pre.name] for row in path['edges'])
 assert (result['gpu_dispatches']>0)==(engine!='cpu')


@pytest.mark.parametrize('kind',list(CODES))
@pytest.mark.parametrize('ranks',[None,2])
def test_staged_path_delay_changes_pending_arrivals_and_checkpoint(engine,kind,ranks,tmp_path,monkeypatch):
 mpi(ranks);net,source,groups,syn,dt,bundle,x=model(kind,True,True,ranks,None,1,engine)
 plan=copy.deepcopy(bundle.plan);plan['trainable']=[False]*len(plan['trainable'])
 trainer=NativeLIFTrainer(plan,weights=bundle.weights,runner=RUNNER)
 trainer.step(x[:,:1],[0]);net.run(dt,namespace={})
 for cursor,length,delay in [(1,1,0),(2,2,1)]:
  before=copy.deepcopy((trainer.state,trainer.clock_tick,trainer.next_noise_sequence))
  trainer.update_delays({syn.pre.name:float(delay*dt)})
  assert (trainer.state,trainer.clock_tick,trainer.next_noise_sequence)==before
  syn.delay=delay*dt
  if kind=='random_temporary':
   for tick in range(cursor,cursor+length):
    calls=replay_current_event_arrivals(trainer,syn,groups[1],tick,monkeypatch)
    out=trainer.step(x[:,tick:tick+1],[0],initial='carry');net.run(dt,namespace={})
    assert len(calls) in (0,2)
  else:out=trainer.step(x[:,cursor:cursor+length],[0],initial='carry');net.run(length*dt,namespace={})
  for group in groups:
   for name,slots in bundle.provenance['neuron_state_layout'][group.name].items():
    np.testing.assert_allclose(np.asarray(out['final_state'])[0,slots],group.variables[name].get_value(),rtol=7e-5,atol=7e-6)
  for name,slots in bundle.provenance['dynamic_state_layout'][syn.name].items():
   np.testing.assert_allclose(np.asarray(out['final_state'])[0,slots],syn.variables[name].get_value(),rtol=7e-5,atol=7e-6)
  checkpoint=tmp_path/('checkpoint-'+str(cursor));trainer.store(checkpoint)
  restored=NativeLIFTrainer(trainer.plan,runner=RUNNER);restored.restore(checkpoint)
  assert restored.neuron_state==trainer.neuron_state and restored.clock_tick==trainer.clock_tick
  trainer=restored
  # Stage grouping survives a fresh trainer constructed from the saved plan.
  assert all(name in {row['name'] for row in trainer.plan['dynamic']['delay_layout']['paths']}
             for name in bundle.provenance['event_callback_stage_groups'][syn.pre.name])
 stage=bundle.provenance['event_callback_stage_groups'][syn.pre.name][-1]
 snapshot=copy.deepcopy((trainer.plan,trainer.state,trainer.neuron_state))
 with pytest.raises(ValueError,match='original pathway'):trainer.update_delays({stage:0.})
 assert (trainer.plan,trainer.state,trainer.neuron_state)==snapshot


def test_staged_array_temporary_preserves_original_read_phase():
 net,source,groups,syn,dt,_,_=model('reread',True,True,None,None,0,'cpu')
 syn.pre.code='tmp=curve(v_pre);v_post+=gain*tmp;h=tmp+v_pre'
 bundle=lower_brian_dynamic_training(net,input_group=source,layers=groups)
 result=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).evaluate(np.zeros((1,1,1)),[0])
 net.run(dt,namespace={})
 np.testing.assert_allclose(syn.h[:],[1.3736,1.188],rtol=0,atol=1e-12)
 slots=bundle.provenance['dynamic_state_layout'][syn.name]['h']
 np.testing.assert_allclose(np.array(result['final_state'])[0,slots],syn.h[:],rtol=0,atol=1e-12)


def replay_current_event_arrivals(trainer,syn,group,tick,monkeypatch):
 from test_training_event_noise import draw
 path=next(row for row in trainer.plan['dynamic']['delay_layout']['paths'] if row['name']==syn.pre.name)
 state=trainer.neuron_state[0];arrivals=None;calls=[]
 def sample(*shape):
  nonlocal arrivals
  if arrivals is None:
   arrivals=[]
   for row in path['pending']:
    if state[row['states'][0]]:
     action=trainer.plan['dynamic']['actions'][row['event']]
     address=action['event_noise'];pending=address.get('pending',0)
     arrivals.append((row['edge'],0 if pending else tick-address['delay'],pending))
   for edge,row in sorted(enumerate(path['edges']),key=lambda item:(-len(item[1]['states']),item[1]['source']['index'],item[0])):
    if (state[row['states'][0]] if row['states'] else edge in group.spikes):arrivals.append((edge,tick-len(row['states']),0))
   np.testing.assert_array_equal(syn.pre.queue.peek(),[edge for edge,_,_ in arrivals])
  stream=len(calls);calls.append(stream);assert stream<2 and shape==(len(arrivals),)
  action=trainer.plan['dynamic']['actions'][path['edges'][0]['event']]
  return np.array([draw('randn',trainer.plan['seed'],trainer.noise_sequence,0,action['noise_domain'],edge,emission,pending,stream) for edge,emission,pending in arrivals])
 monkeypatch.setattr(np.random,'randn',sample)
 return calls


def test_staged_delay_write_preserves_an_original_emission_latch():
 net,source,groups,syn,dt,_,_=model('reread',True,True,None,None,0,'cpu')
 syn.pre.code='v_post+=gain*curve(v_pre);h=v_pre;delay+=dt'
 bundle=lower_brian_dynamic_training(net,input_group=source,layers=groups)
 result=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).evaluate(np.zeros((1,1,1)),[0])
 net.run(dt,namespace={})
 np.testing.assert_allclose(syn.h[:],[.8776,.66],rtol=0,atol=1e-12)
 np.testing.assert_allclose(np.asarray(syn.pre.delay[:]),float(dt),rtol=0,atol=1e-15)
 slots=bundle.provenance['pathway_state_layout'][syn.pre.name]['delay']
 np.testing.assert_allclose(np.asarray(result['final_state'])[0,slots],float(dt),rtol=0,atol=1e-15)
 slots=bundle.provenance['dynamic_state_layout'][syn.name]['h']
 np.testing.assert_allclose(np.asarray(result['final_state'])[0,slots],syn.h[:],rtol=0,atol=1e-12)
