"""Threshold borrowed-array effects, operand order, refractory and VJPs."""
import copy
import brian2 as b
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer,lower_brian_dynamic_training
from test_training_callback_effects import curve
from test_training_integer_ir import engine
from test_training_poisson_zero_vjp import mpi
from test_native_training import RUNNER

THRESHOLDS={
 'borrowed':'curve(v)>.5',
 'hidden':'v+.2*curve(a)>.5',
 'temporary':'curve(v+.125)>.5',
 'right':'v>curve(a)',
 'left_reference':'v>curve(v)',
 'less_order':'v<curve(v)',
 'left_value':'2*v>curve(v)',
 'inclusive':'curve(v)>=.5',
}

def model(kind,discard,ranks,window,backend,refractory=False,threshold=None):
 b.set_device('runtime');b.start_scope();b.prefs.codegen.target='numpy';b.prefs.codegen.runtime.numpy.discard_units=discard
 dt=.2*b.ms;source=b.SpikeGeneratorGroup(1,[],np.array([])*b.second,dt=dt,name='threshold_effect_input')
 hidden=b.NeuronGroup(1,'dv/dt=0/second:1',threshold='v>100',reset='v=0',method='euler',dt=dt,name='threshold_effect_hidden')
 f=b.Function(curve,arg_units=[1],arg_names=['x'],return_unit=1,stateless=False)
 g=b.NeuronGroup(2,'dv/dt=gain/ms:1\na:1\ngain:1 (constant)',method='euler',
                 threshold=THRESHOLDS[kind] if threshold is None else threshold,reset='v-=.5',dt=dt,namespace={'curve':f},
                 refractory=.4*b.ms if refractory else False,name='threshold_effect_neurons')
 g.v=[.3,.7];g.a=[.4,.6];g.gain=[.2,.3]
 if refractory:g.lastspike=[0.,-1.]*b.second;g.not_refractory=[False,True]
 net=b.Network(source,hidden,g)
 bundle=lower_brian_dynamic_training(net,input_group=source,layers=[hidden,g],backend=backend,mpi_ranks=ranks,
                                    tbptt_window=window,detach_reset=False,trainable_neuron_parameters={g.name:['gain']})
 return net,g,dt,bundle

@pytest.mark.parametrize('kind',list(THRESHOLDS))
@pytest.mark.parametrize('discard',[False,True])
@pytest.mark.parametrize('ranks',[None,2])
def test_threshold_effect_original_and_restored_carry(engine,kind,discard,ranks,tmp_path):
 mpi(ranks);net,g,dt,bundle=model(kind,discard,ranks,None,engine)
 bundle.plan['trainable']=[False]*len(bundle.weights)
 trainer=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER)
 for tick in range(3):
  out=trainer.step(np.zeros((1,1,1)),[0],initial='carry' if tick else None);net.run(dt,namespace={})
  for name in ('v','a'):
   value=np.asarray(out['final_state'])[0,bundle.provenance['neuron_state_layout'][g.name][name]]
   np.testing.assert_allclose(value,g.variables[name].get_value(),rtol=7e-5,atol=7e-6)
  path=tmp_path/str(tick);trainer.store(path)
  restored=NativeLIFTrainer(trainer.plan,runner=RUNNER);restored.restore(path);trainer=restored
 assert (out['gpu_dispatches']>0)==(engine!='cpu')

def reference(bundle,g,kind,weights,initial=None,anchors=None,refractory=False):
 p=bundle.plan;layout=bundle.provenance['neuron_state_layout'][g.name]
 z=np.array(bundle.initial_state if initial is None else initial,float)
 bank=next(row['bank'] for row in bundle.provenance['bindings'] if row['object']==g.name and row['variables']==['gain'])
 v=z[layout['v']].copy();a=z[layout['a']].copy();last=np.array([0,-5000]);before=[];margins=[];hard=[];spikes=[]
 for tick in range(4):
  if anchors is not None and p['tbptt_window'] and tick and tick%p['tbptt_window']==0:
   v,a=(row.copy() for row in anchors['before'][tick])
  before.append((v.copy(),a.copy()));v+=.2*np.array(weights[bank])
  # Evaluate operands separately, retaining references until comparison.
  # The Python/NumPy callback itself supplies the independent mutation rule.
  if kind in ('borrowed','inclusive'):left=curve(v);right=.5
  elif kind=='hidden':left=v+.2*curve(a);right=.5
  elif kind=='temporary':left=curve(v+.125);right=.5
  elif kind=='right':left=v;right=curve(a)
  elif kind in ('left_reference','less_order'):left=v;right=curve(v)
  else:left=2*v;right=curve(v)
  margin=right-left if kind=='less_order' else left-right
  active=tick-last>=2 if refractory else np.ones(2,dtype=bool)
  event=((margin>=0) if kind=='inclusive' else (margin>0)).astype(float)*active
  margins.append(margin.copy());hard.append(event.copy())
  if anchors is not None:
   old=anchors['margins'][tick]
   event=anchors['hard'][tick]+active*p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(old))**2*(margin-old)
  spikes.append(event.copy());last[hard[-1].astype(bool)]=tick;v-=.5*event
 logits=np.array(spikes).mean(0)*p['logit_scale'];loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
 return loss,v,a,np.array(spikes),dict(before=before,margins=margins,hard=hard)

@pytest.mark.parametrize('kind',list(THRESHOLDS))
@pytest.mark.parametrize('window',[None,2])
@pytest.mark.parametrize('ranks',[None,2])
def test_threshold_effect_all_parameters_and_initial_vjps(engine,kind,window,ranks):
 mpi(ranks);_,g,_,bundle=model(kind,True,ranks,window,engine)
 out=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).gradients(np.zeros((1,4,1)),[0])
 loss,v,a,spikes,anchors=reference(bundle,g,kind,bundle.weights)
 for name,value in (('v',v),('a',a)):
  np.testing.assert_allclose(np.asarray(out['final_state'])[0,bundle.provenance['neuron_state_layout'][g.name][name]],value,rtol=7e-5,atol=7e-6)
 np.testing.assert_array_equal(np.asarray(out['spikes'])[0,:,1:],spikes);assert out['loss']==pytest.approx(loss,abs=7e-6)
 for bank,row in enumerate(bundle.weights):
  for index in range(len(row)):
   hi=copy.deepcopy(bundle.weights);lo=copy.deepcopy(bundle.weights);hi[bank][index]+=1e-6;lo[bank][index]-=1e-6
   fd=(reference(bundle,g,kind,hi,anchors=anchors)[0]-reference(bundle,g,kind,lo,anchors=anchors)[0])/2e-6
   assert out['gradients'][bank][index]==pytest.approx(fd,rel=9e-4,abs=9e-6),(bank,index)
 for index in range(len(bundle.initial_state)):
  if bundle.plan['dynamic']['detached'][index] or index in bundle.plan['dynamic']['integer_states']:
   assert out['initial_state_gradients'][0][index]==0.;continue
  hi=np.array(bundle.initial_state);lo=hi.copy();hi[index]+=1e-6;lo[index]-=1e-6
  fd=(reference(bundle,g,kind,bundle.weights,hi,anchors)[0]-reference(bundle,g,kind,bundle.weights,lo,anchors)[0])/2e-6
  assert out['initial_state_gradients'][0][index]==pytest.approx(fd,rel=9e-4,abs=9e-6),index
 assert (out['gpu_dispatches']>0)==(engine!='cpu')

@pytest.mark.parametrize('kind',['borrowed','hidden','right','left_value'])
@pytest.mark.parametrize('ranks',[None,2])
def test_refractory_threshold_still_mutates_borrowed_arrays(engine,kind,ranks):
 mpi(ranks);net,g,dt,bundle=model(kind,True,ranks,None,engine,refractory=True)
 out=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).gradients(np.zeros((1,4,1)),[0])
 loss,v,a,spikes,_=reference(bundle,g,kind,bundle.weights,refractory=True)
 net.run(4*dt,namespace={})
 for name,value in (('v',v),('a',a)):
  np.testing.assert_allclose(g.variables[name].get_value(),value,rtol=7e-5,atol=7e-6)
  np.testing.assert_allclose(np.asarray(out['final_state'])[0,bundle.provenance['neuron_state_layout'][g.name][name]],value,rtol=7e-5,atol=7e-6)
 np.testing.assert_array_equal(np.asarray(out['spikes'])[0,:,1:],spikes);assert out['loss']==pytest.approx(loss,abs=7e-6)
 assert (out['gpu_dispatches']>0)==(engine!='cpu')
