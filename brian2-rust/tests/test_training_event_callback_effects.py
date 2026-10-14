"""NumPy event array copies, scalar fallback, batch snapshots and VJPs."""
import copy
import brian2 as b
import numpy as np
import pytest
from brian2_rust import lower_brian_dynamic_training,NativeLIFTrainer
from test_training_callback_effects import curve
from test_training_integer_ir import engine
from test_training_poisson_zero_vjp import mpi
from test_native_training import RUNNER

CODES={
 'own':'h=curve(h)+h',
 'parameter':'h=curve(a)+a',
 'shared_parameter':'h=curve(a)+a',
 'scatter':'h=curve(h)+h;v_post+=gain*h',
 'reload':'h=curve(a);v_post+=gain*a',
 'temporary':'temp=curve(a);alias=temp;temp*=.8;v_post+=gain*alias;h=temp',
 'scalar':'h=curve(v_post);v_post+=gain*h',
 'autapse':'h=curve(v_pre);v_post+=gain*h',
}


def model(kind,repeat,discard,ranks,window,delay,engine):
 b.set_device('runtime');b.start_scope();b.prefs.codegen.target='numpy';b.prefs.codegen.runtime.numpy.discard_units=discard
 dt=.2*b.ms;times=np.array([]) if kind=='autapse' else np.array([0,2])
 source=b.SpikeGeneratorGroup(1,np.zeros(len(times),int),times*dt,dt=dt,name='callback_input')
 a=b.NeuronGroup(2,'dv/dt=(.5-v)/ms:1',method='euler',threshold='v>.5',reset='v-=.5',dt=dt,name='callback_a')
 c=b.NeuronGroup(2,'dv/dt=(.5-v+q)/ms:1\nq:1',method='euler',threshold='v>.5',reset='v-=.5',dt=dt,name='callback_c')
 a.v=[.6,.1];c.v=[.4,.7]
 f=b.Function(curve,arg_units=[1],arg_names=['x'],return_unit=1,stateless=False)
 shared=kind=='shared_parameter'
 syn=b.Synapses(c if kind=='autapse' else source,c,'h:1\na:1 (constant'+(', shared' if shared else '')+')\ngain:1 (constant)\nq_post=h:1 (summed)',
                on_pre=CODES[kind],dt=dt,namespace={'curve':f},name='callback_syn')
 syn.connect(i=[0,1] if kind=='autapse' else [0,0],j=[0,0] if repeat else [0,1]);syn.h=[.6,.4];syn.gain=[.2,.3];syn.a=.3 if shared else [.3,.5]
 syn.delay=delay*dt;net=b.Network(source,a,c,syn)
 bundle=lower_brian_dynamic_training(net,input_group=source,layers=[a,c],backend=engine,mpi_ranks=ranks,
   tbptt_window=window,detach_reset=False,trainable_synapse_parameters={syn.name:['h','a','gain']})
 x=np.zeros((1,4,1))
 if kind!='autapse':x[0,[0,2],0]=1
 return net,source,(a,c),syn,dt,bundle,x


def reference(bundle,groups,syn,kind,repeat,delay,weights,initial=None,anchors=None):
 p=bundle.plan;z=np.array(bundle.initial_state if initial is None else initial,float)
 if initial is None:
  for slot,parameter in enumerate(p['dynamic']['initial_parameters']):
   if parameter is not None:z[slot]=weights[parameter[0]][parameter[1]]
 layouts=bundle.provenance['neuron_state_layout'];av=layouts[groups[0].name]['v'];cv=layouts[groups[1].name]['v'];q=layouts[groups[1].name]['q']
 h=bundle.provenance['dynamic_state_layout'][syn.name]['h']
 banks={name:next(row['bank'] for row in bundle.provenance['bindings'] if row['object']==syn.name and row['variables']==[name]) for name in ('a','gain')}
 aa=np.broadcast_to(np.array(weights[banks['a']]),(2,));gain=np.array(weights[banks['gain']]);posts=[0,0] if repeat else [0,1]
 spikes=[];margins=[];before=[]
 path=syn.pre.name;snapshots=bundle.provenance['event_callback_snapshots'].get(path,[])
 for tick in range(4):
  if anchors is not None and p['tbptt_window'] and tick and tick%p['tbptt_window']==0:z=anchors[1][tick].copy()
  before.append(z.copy());z[q]=0.
  for edge,post in enumerate(posts):z[q[post]]+=z[h[edge]]
  z[av]=.8*z[av]+.1;z[cv]=.8*z[cv]+.1+.2*z[q]
  margin=z[av+cv]-.5;gate=(margin>0).astype(float)
  if anchors is not None:
   old=anchors[0][tick];gate=(old>0).astype(float)+p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(old))**2*(margin-old)
  margins.append(margin.copy());spikes.append(gate.copy())
  for row in snapshots:z[row['cache']]=z[row['source']]
  emission=tick-delay
  # History cells are binary in real forwards; their reported initial VJP
  # uses the documented continuous proxy event gate. Include a perturbed
  # initial queue cell rather than deriving every arrival from fixed emissions.
  routes=bundle.provenance['delay_queues'][path]['new']
  event=np.array([z[row['states'][0]] for row in routes]) if delay else (gate[2:] if kind=='autapse' else np.full(2,float(tick in (0,2))))
  prior_h=z[h].copy();pre=z[cv].copy()
  for edge,post in enumerate(posts):
   old_h=z[h[edge]];old_v=z[cv[post]];g=event[edge]
   if kind=='own' or kind=='scatter':new_h=1.6*prior_h[edge]
   elif kind=='parameter':new_h=1.6*aa[edge]
   elif kind=='shared_parameter':new_h=2.*aa[edge]
   elif kind=='reload':new_h=.8*aa[edge]
   elif kind=='temporary':new_h=.64*aa[edge]
   elif kind=='scalar':new_h=old_v
   else:new_h=.8*pre[edge]
   z[h[edge]]=old_h+g*(new_h-old_h)
   if kind in ('scatter','temporary','scalar','autapse'):z[cv[post]]=old_v+g*gain[edge]*new_h
   elif kind=='reload':z[cv[post]]=old_v+g*gain[edge]*aa[edge]
  for entry in bundle.provenance['delay_queues'][path]['new']:
   states=entry['states']
   if states:
    z[states[:-1]]=z[states[1:]]
    z[states[-1]]=gate[2+entry['edge']] if kind=='autapse' else float(tick in (0,2))
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
def test_event_callback_modes_actual_brian_and_all_vjps(engine,kind,repeat,discard,ranks,window,delay):
 mpi(ranks);net,source,groups,syn,dt,bundle,x=model(kind,repeat,discard,ranks,window,delay,engine)
 result=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).gradients(x,[0])
 loss,z,spikes,anchors=reference(bundle,groups,syn,kind,repeat,delay,bundle.weights)
 np.testing.assert_allclose(result['final_state'][0],z,rtol=6e-5,atol=6e-6)
 np.testing.assert_array_equal(result['spikes'][0],spikes);assert result['loss']==pytest.approx(loss,abs=6e-6)
 for bank,row in enumerate(bundle.weights):
  for j in range(len(row)):
   hi=copy.deepcopy(bundle.weights);lo=copy.deepcopy(bundle.weights);hi[bank][j]+=1e-6;lo[bank][j]-=1e-6
   fd=(reference(bundle,groups,syn,kind,repeat,delay,hi,anchors=anchors)[0]-reference(bundle,groups,syn,kind,repeat,delay,lo,anchors=anchors)[0])/2e-6
   assert result['gradients'][bank][j]==pytest.approx(fd,rel=8e-4,abs=8e-6)
 for j in range(len(bundle.initial_state)):
  hi=np.array(bundle.initial_state,float);lo=hi.copy();hi[j]+=1e-6;lo[j]-=1e-6
  fd=(reference(bundle,groups,syn,kind,repeat,delay,bundle.weights,initial=hi,anchors=anchors)[0]-reference(bundle,groups,syn,kind,repeat,delay,bundle.weights,initial=lo,anchors=anchors)[0])/2e-6
  assert result['initial_state_gradients'][0][j]==pytest.approx(fd,rel=8e-4,abs=8e-6)
 net.run(4*dt,namespace={})
 for group in groups:
  for name,slots in bundle.provenance['neuron_state_layout'][group.name].items():
   np.testing.assert_allclose(np.asarray(result['final_state'])[0,slots],group.variables[name].get_value(),rtol=6e-5,atol=6e-6)
 slots=bundle.provenance['dynamic_state_layout'][syn.name]['h']
 np.testing.assert_allclose(np.asarray(result['final_state'])[0,slots],syn.h[:],rtol=6e-5,atol=6e-6)
 np.testing.assert_allclose(syn.a[:],.3 if kind=='shared_parameter' else [.3,.5],rtol=0,atol=0)
 mode=bundle.provenance['event_callback_modes'][syn.pre.name]['mode']
 assert mode==('scalar' if kind=='scalar' else 'array' if kind in ('own','parameter','shared_parameter') else 'vectorised')
 assert (result['gpu_dispatches']>0)==(engine!='cpu')


@pytest.mark.parametrize('kind',['own','scatter','reload','scalar','autapse'])
@pytest.mark.parametrize('ranks',[None,2])
def test_event_callback_snapshots_survive_delay_rebuild_and_checkpoint(engine,kind,ranks,tmp_path):
 mpi(ranks);net,source,groups,syn,dt,bundle,x=model(kind,True,True,ranks,None,1,engine)
 plan=copy.deepcopy(bundle.plan);plan['trainable']=[False]*len(plan['trainable'])
 trainer=NativeLIFTrainer(plan,weights=bundle.weights,runner=RUNNER)
 trainer.step(x[:,:1],[0]);net.run(dt,namespace={})
 def compare(result):
  for group in groups:
   for name,slots in bundle.provenance['neuron_state_layout'][group.name].items():
    np.testing.assert_allclose(np.asarray(result['final_state'])[0,slots],group.variables[name].get_value(),rtol=6e-5,atol=6e-6)
  for name,slots in bundle.provenance['dynamic_state_layout'][syn.name].items():
   np.testing.assert_allclose(np.asarray(result['final_state'])[0,slots],syn.variables[name].get_value(),rtol=6e-5,atol=6e-6)
 for cursor,length,delay in [(1,1,0),(2,2,1)]:
  before=copy.deepcopy((trainer.state,trainer.clock_tick,trainer.next_noise_sequence))
  trainer.update_delays({syn.pre.name:float(delay*dt)})
  assert (trainer.state,trainer.clock_tick,trainer.next_noise_sequence)==before
  syn.delay=delay*dt
  result=trainer.step(x[:,cursor:cursor+length],[0],initial='carry');net.run(length*dt,namespace={});compare(result)
  checkpoint=tmp_path/('checkpoint-'+str(cursor));trainer.store(checkpoint)
  restored=NativeLIFTrainer(trainer.plan,runner=RUNNER);restored.restore(checkpoint)
  assert restored.neuron_state==trainer.neuron_state and restored.clock_tick==trainer.clock_tick
  trainer=restored
 for row in bundle.provenance['event_callback_snapshots'].get(syn.pre.name,[]):
  assert row['cache']<len(trainer.neuron_state[0])


def test_vectorised_scatter_alias_reread_matches_original():
 net,source,groups,syn,dt,_,x=model('autapse',True,True,None,None,0,'cpu')
 syn.pre.code='v_post+=curve(v_pre);h=v_pre'
 bundle=lower_brian_dynamic_training(net,input_group=source,layers=groups)
 result=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).evaluate(x[:,:1],[0])
 net.run(dt,namespace={})
 np.testing.assert_allclose(syn.h[:],[1.644,.66],rtol=0,atol=1e-12)
 slots=bundle.provenance['dynamic_state_layout'][syn.name]['h']
 np.testing.assert_allclose(np.array(result['final_state'])[0,slots],syn.h[:],rtol=0,atol=1e-12)
 for group in groups:
  for name,slots in bundle.provenance['neuron_state_layout'][group.name].items():
   np.testing.assert_allclose(np.array(result['final_state'])[0,slots],group.variables[name].get_value(),rtol=0,atol=1e-12)
 assert len(bundle.provenance['event_callback_stage_groups'][syn.pre.name])==2


@pytest.mark.parametrize('ranks',[None,2])
def test_event_callback_inactive_errors_and_active_failure_are_atomic(engine,ranks):
 from test_training_callback_effects import invalid_curve
 mpi(ranks);net,source,groups,syn,dt,_,x=model('scatter',True,True,ranks,None,1,engine)
 syn.namespace['curve']=b.Function(invalid_curve,arg_units=[1],arg_names=['x'],return_unit=1,stateless=False)
 bundle=lower_brian_dynamic_training(net,input_group=source,layers=groups,backend=engine,mpi_ranks=ranks)
 trainer=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER)
 # No arrival in this tick: callback errors must not run during copy preparation.
 trainer.evaluate(np.zeros((1,1,1)),[0])
 before=copy.deepcopy((trainer.state,trainer.neuron_state,trainer.clock_tick,trainer.next_noise_sequence,trainer.plan))
 with pytest.raises(ValueError,match='[Nn]on.?finite|domain'):trainer.step(x[:,:2],[0])
 assert (trainer.state,trainer.neuron_state,trainer.clock_tick,trainer.next_noise_sequence,trainer.plan)==before
