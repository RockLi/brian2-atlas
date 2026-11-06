"""Typed event gathers, per-statement storage casts and exact int snapshots."""
import copy
import brian2 as b
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer,lower_brian_dynamic_training
from test_training_typed_callback_effects import increment,clear,fill
from test_training_integer_ir import engine
from test_training_poisson_zero_vjp import mpi
from test_native_training import RUNNER

CODES={
 'own':'k=change(k);h=.1*k;v_post+=gain*h',
 'wrap':'k=change(k);h=1e-10*k;v_post+=gain*h',
 'cast':'k=change(gain+1.25);h=.1*k;v_post+=gain*h',
 'scalar':'h=change(k+int(v_post));v_post+=.1*gain*h',
 'clear':'flag=change(flag);h=.1*flag;v_post+=gain*h',
 'fill':'flag=change(flag);h=.1*flag;v_post+=gain*h',
}

def model(kind,repeat,discard,ranks,window,delay,backend):
 b.set_device('runtime');b.start_scope();b.prefs.codegen.target='numpy';b.prefs.codegen.runtime.numpy.discard_units=discard
 dt=.2*b.ms;source=b.SpikeGeneratorGroup(1,[0,0],np.array([0,2])*dt,dt=dt,name='typed_event_input')
 hidden=b.NeuronGroup(1,'dv/dt=0/second:1',method='euler',threshold='v>100',reset='v=0',dt=dt,name='typed_event_hidden')
 g=b.NeuronGroup(2,'dv/dt=0/second:1',method='euler',threshold='v>.5',reset='v-=.5',dt=dt,name='typed_event_neurons');g.v=[.3,.7]
 callback=clear if kind=='clear' else fill if kind=='fill' else increment
 f=b.Function(callback,arg_units=[1],arg_names=['x'],return_unit=1,stateless=False)
 syn=b.Synapses(source,g,'k:integer\nflag:boolean\nh:1\ngain:1 (constant)',on_pre=CODES[kind],dt=dt,namespace={'change':f},name='typed_event_syn')
 syn.connect(i=[0,0],j=[0,0] if repeat else [0,1]);syn.k=[2147483647,-2147483648] if kind=='wrap' else [2,3];syn.flag=[True,False];syn.h=[.2,.3];syn.gain=[.2,.3];syn.delay=delay*dt
 net=b.Network(source,hidden,g,syn)
 bundle=lower_brian_dynamic_training(net,input_group=source,layers=[hidden,g],backend=backend,mpi_ranks=ranks,tbptt_window=window,
                                    detach_reset=False,trainable_synapse_parameters={syn.name:['h','gain']})
 x=np.zeros((1,4,1));x[0,[0,2],0]=1.
 return net,g,syn,dt,bundle,x

@pytest.mark.parametrize('kind',list(CODES))
@pytest.mark.parametrize('repeat',[False,True])
@pytest.mark.parametrize('discard',[False,True])
@pytest.mark.parametrize('delay',[0,1])
@pytest.mark.parametrize('ranks',[None,2])
def test_typed_event_original_and_restored_carry(engine,kind,repeat,discard,delay,ranks,tmp_path):
 mpi(ranks);net,g,syn,dt,bundle,x=model(kind,repeat,discard,ranks,None,delay,engine)
 bundle.plan['trainable']=[False]*len(bundle.weights)
 trainer=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER)
 for start,length in [(0,1),(1,2),(3,1)]:
  out=trainer.step(x[:,start:start+length],[0],initial='carry' if start else None);net.run(length*dt,namespace={})
  np.testing.assert_allclose(np.asarray(out['final_state'])[0,bundle.provenance['neuron_state_layout'][g.name]['v']],g.v[:],rtol=7e-5,atol=7e-6)
  for name,slots in bundle.provenance['dynamic_state_layout'][syn.name].items():
   values=np.asarray(out['final_state'])[0,slots]
   if name in ('k','flag'):np.testing.assert_array_equal(values,syn.variables[name].get_value())
   else:np.testing.assert_allclose(values,syn.variables[name].get_value(),rtol=7e-5,atol=7e-6)
  path=tmp_path/('typed-event-'+str(start));trainer.store(path)
  restored=NativeLIFTrainer(trainer.plan,runner=RUNNER);restored.restore(path);trainer=restored
 assert (out['gpu_dispatches']>0)==(engine!='cpu')


def reference(bundle,g,syn,kind,repeat,delay,weights,initial=None,anchors=None):
 p=bundle.plan;z=np.array(bundle.initial_state if initial is None else initial,float)
 if initial is None:
  for slot,ref in enumerate(p['dynamic']['initial_parameters']):
   if ref is not None:z[slot]=weights[ref[0]][ref[1]]
 v=bundle.provenance['neuron_state_layout'][g.name]['v'];layout=bundle.provenance['dynamic_state_layout'][syn.name]
 h=layout['h'];k=layout.get('k');flag=layout.get('flag');posts=[0,0] if repeat else [0,1]
 bank=next(row['bank'] for row in bundle.provenance['bindings'] if row['object']==syn.name and row['variables']==['gain']);gain=np.array(weights[bank])
 path=syn.pre.name;before=[];margins=[];hard=[];spikes=[];events=[]
 for tick in range(4):
  if anchors is not None and p['tbptt_window'] and tick and tick%p['tbptt_window']==0:z=anchors['before'][tick].copy()
  before.append(z.copy());margin=z[v]-.5;gate=(margin>0).astype(float)
  if anchors is not None:
   old=anchors['margins'][tick];gate=anchors['hard'][tick]+p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(old))**2*(margin-old)
  margins.append(margin.copy());hard.append((margin>0).astype(float));spikes.append(gate.copy())
  for row in bundle.provenance['event_callback_snapshots'].get(path,[]):z[row['cache']]=z[row['source']]
  queues=bundle.provenance['delay_queues'][path]['new'];amplitudes=np.array([z[row['states'][0]] for row in queues]) if delay else np.full(2,float(tick in (0,2)))
  actual=amplitudes!=0 if anchors is None else anchors['events'][tick];events.append(actual.copy())
  for edge,post in enumerate(posts):
   old_h=z[h[edge]];old_v=z[v[post]];new_h=old_h
   if kind in ('own','wrap'):
    count=np.array([int(z[k[edge]])],dtype=np.int32);count+=1;new_count=int(count[0]);new_h=(1e-10 if kind=='wrap' else .1)*new_count
    if actual[edge]:z[k[edge]]=new_count
   elif kind=='cast':
    new_count=int(gain[edge]+2.25);new_h=.1*new_count
    if actual[edge]:z[k[edge]]=new_count
   elif kind=='scalar':new_h=2+edge+int(old_v)
   else:
    new_flag=kind=='fill';new_h=.1*new_flag
    if actual[edge]:z[flag[edge]]=new_flag
   z[h[edge]]=old_h+amplitudes[edge]*(new_h-old_h)
   z[v[post]]=old_v+amplitudes[edge]*gain[edge]*new_h*(.1 if kind=='scalar' else 1.)
  for row in queues:
   slots=row['states']
   if slots:z[slots[:-1]]=z[slots[1:]];z[slots[-1]]=float(tick in (0,2))
  z[v]-=.5*gate
 logits=np.array(spikes).mean(0)*p['logit_scale'];loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
 return loss,z,np.array(spikes),dict(before=before,margins=margins,hard=hard,events=events)


@pytest.mark.parametrize('kind',list(CODES))
@pytest.mark.parametrize('repeat',[False,True])
@pytest.mark.parametrize('delay',[0,1])
@pytest.mark.parametrize('ranks',[None,2])
@pytest.mark.parametrize('window',[None,2])
def test_typed_event_all_parameters_and_initial_vjps(engine,kind,repeat,delay,ranks,window):
 mpi(ranks);_,g,syn,_,bundle,x=model(kind,repeat,True,ranks,window,delay,engine)
 out=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).gradients(x,[0])
 loss,z,spikes,anchors=reference(bundle,g,syn,kind,repeat,delay,bundle.weights)
 np.testing.assert_allclose(out['final_state'][0],z,rtol=7e-5,atol=7e-6)
 np.testing.assert_array_equal(np.asarray(out['spikes'])[0,:,1:],spikes);assert out['loss']==pytest.approx(loss,abs=7e-6)
 for bank,row in enumerate(bundle.weights):
  for index in range(len(row)):
   hi=copy.deepcopy(bundle.weights);lo=copy.deepcopy(bundle.weights);hi[bank][index]+=1e-6;lo[bank][index]-=1e-6
   fd=(reference(bundle,g,syn,kind,repeat,delay,hi,anchors=anchors)[0]-reference(bundle,g,syn,kind,repeat,delay,lo,anchors=anchors)[0])/2e-6
   assert out['gradients'][bank][index]==pytest.approx(fd,rel=9e-4,abs=9e-6),(bank,index)
 for index in range(len(bundle.initial_state)):
  # Binary arrival buffers have a declared continuous proxy gate VJP. Only
  # detached model control and integer storage have stopped initial adjoints.
  if bundle.plan['dynamic']['detached'][index] or index in bundle.plan['dynamic']['integer_states']:
   assert out['initial_state_gradients'][0][index]==0.,index;continue
  hi=np.array(bundle.initial_state);lo=hi.copy();hi[index]+=1e-6;lo[index]-=1e-6
  fd=(reference(bundle,g,syn,kind,repeat,delay,bundle.weights,hi,anchors)[0]-reference(bundle,g,syn,kind,repeat,delay,bundle.weights,lo,anchors)[0])/2e-6
  assert out['initial_state_gradients'][0][index]==pytest.approx(fd,rel=9e-4,abs=9e-6),index
 assert (out['gpu_dispatches']>0)==(engine!='cpu')
