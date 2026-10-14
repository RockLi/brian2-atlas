"""Typed threshold mutations stop control gradients, retaining float VJPs."""
import copy
import brian2 as b
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer,lower_brian_dynamic_training
from test_training_typed_callback_effects import increment,fill
from test_training_integer_ir import engine
from test_training_poisson_zero_vjp import mpi
from test_native_training import RUNNER

def model(kind,discard,ranks,window,backend):
 b.set_device('runtime');b.start_scope();b.prefs.codegen.target='numpy';b.prefs.codegen.runtime.numpy.discard_units=discard
 dt=.2*b.ms;source=b.SpikeGeneratorGroup(1,[],np.array([])*b.second,dt=dt)
 hidden=b.NeuronGroup(1,'dv/dt=0/second:1',threshold='v>100',reset='v=0',method='euler',dt=dt)
 callback=b.Function(fill if kind=='boolean' else increment,arg_units=[1],arg_names=['x'],
                     return_unit=bool if kind=='boolean' else 1,return_type='boolean' if kind=='boolean' else 'integer',stateless=False)
 g=b.NeuronGroup(2,'dv/dt=gain/ms:1\nk:integer\nflag:boolean\ngain:1 (constant)',method='euler',
                 threshold='change(flag) and (v>.5)' if kind=='boolean' else '(change(k)>3) and (v>.5)',
                 reset='v-=.5',dt=dt,namespace={'change':callback})
 g.v=[.3,.7];g.k=[2,3];g.flag=[False,True];g.gain=[.2,.3]
 net=b.Network(source,hidden,g)
 bundle=lower_brian_dynamic_training(net,input_group=source,layers=[hidden,g],backend=backend,mpi_ranks=ranks,
                                    tbptt_window=window,detach_reset=False,trainable_neuron_parameters={g.name:['gain']})
 return net,g,dt,bundle

def reference(bundle,g,kind,weights,initial=None,anchors=None):
 p=bundle.plan;layout=bundle.provenance['neuron_state_layout'][g.name];z=np.array(bundle.initial_state if initial is None else initial,float)
 bank=next(row['bank'] for row in bundle.provenance['bindings'] if row['object']==g.name and row['variables']==['gain'])
 v=z[layout['v']].copy();k=z[layout['k']].astype(np.int32);flag=z[layout['flag']].astype(bool)
 before=[];margins=[];hard=[];spikes=[]
 for tick in range(4):
  if anchors is not None and p['tbptt_window'] and tick and tick%p['tbptt_window']==0:
   v,k,flag=(a.copy() for a in anchors['before'][tick])
  before.append((v.copy(),k.copy(),flag.copy()));v+=.2*np.array(weights[bank])
  control=fill(flag).astype(float) if kind=='boolean' else (increment(k)>3).astype(float)
  margin=v-.5;event=(margin>0).astype(float);margins.append(margin.copy());hard.append(event.copy())
  if anchors is not None:
   old=anchors['margins'][tick];event=anchors['hard'][tick]+p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(old))**2*(margin-old)
  event*=control;spikes.append(event.copy());v-=.5*event
 logits=np.array(spikes).mean(0)*p['logit_scale'];loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
 return loss,v,k,flag,np.array(spikes),dict(before=before,margins=margins,hard=hard)

@pytest.mark.parametrize('kind',['integer','boolean'])
@pytest.mark.parametrize('discard',[False,True])
@pytest.mark.parametrize('ranks',[None,2])
@pytest.mark.parametrize('window',[None,2])
def test_typed_threshold_original_and_all_vjps(engine,kind,discard,ranks,window):
 mpi(ranks);net,g,dt,bundle=model(kind,discard,ranks,window,engine)
 out=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).gradients(np.zeros((1,4,1)),[0])
 loss,v,k,flag,spikes,anchors=reference(bundle,g,kind,bundle.weights);net.run(4*dt,namespace={})
 for name,value in (('v',v),('k',k),('flag',flag)):
  np.testing.assert_allclose(g.variables[name].get_value(),value,rtol=7e-5,atol=7e-6)
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
