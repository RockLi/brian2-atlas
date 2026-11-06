"""NumPy eager threshold predicates retain discarded branch mutations."""
import copy
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer
from test_training_threshold_callback_effects import model
from test_training_callback_effects import curve
from test_training_integer_ir import engine
from test_training_poisson_zero_vjp import mpi
from test_native_training import RUNNER

PREDICATES={
 'and':'(v>.5) and (curve(a)>.3)',
 'or':'(v>.5) or (curve(a)>.3)',
 'not':'not (curve(v)>.5)',
 'equality':'(curve(a)==.32) or (v>.5)',
 'shared_alias':'(curve(v)>.4) and (curve(v)>.3)',
}

def reference(bundle,g,kind,weights,initial=None,anchors=None):
 p=bundle.plan;layout=bundle.provenance['neuron_state_layout'][g.name]
 z=np.array(bundle.initial_state if initial is None else initial,float)
 bank=next(row['bank'] for row in bundle.provenance['bindings'] if row['object']==g.name and row['variables']==['gain'])
 v=z[layout['v']].copy();a=z[layout['a']].copy();before=[];comparisons=[];hard=[];spikes=[]
 for tick in range(4):
  if anchors is not None and p['tbptt_window'] and tick and tick%p['tbptt_window']==0:
   v,a=(row.copy() for row in anchors['before'][tick])
  before.append((v.copy(),a.copy()));v+=.2*np.array(weights[bank]);margins=[];gates=[]
  def compare(value,bound,equality=False):
   margin=(value-bound).copy();gate=(value==bound).astype(float) if equality else (margin>0).astype(float)
   index=len(gates);margins.append(margin);gates.append(gate.copy())
   if anchors is not None and not equality:
    old=anchors['comparisons'][tick][index]
    gate=anchors['hard'][tick][index]+p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(old))**2*(margin-old)
   return gate
  if kind=='not':event=1-compare(curve(v),.5)
  elif kind=='shared_alias':
   left=compare(curve(v),.4);right=compare(curve(v),.3);event=left*right
  elif kind=='equality':
   left=compare(curve(a),.32,equality=True);right=compare(v,.5);event=left+right-left*right
  else:
   left=compare(v,.5);right=compare(curve(a),.3)
   event=left*right if kind=='and' else left+right-left*right
  comparisons.append(margins);hard.append(gates);spikes.append(event.copy());v-=.5*event
 logits=np.array(spikes).mean(0)*p['logit_scale'];loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
 return loss,v,a,np.array(spikes),dict(before=before,comparisons=comparisons,hard=hard)

@pytest.mark.parametrize('kind',list(PREDICATES))
@pytest.mark.parametrize('window',[None,2])
@pytest.mark.parametrize('ranks',[None,2])
@pytest.mark.parametrize('discard',[False,True])
def test_eager_effect_predicate_original_and_all_vjps(engine,kind,window,ranks,discard):
 mpi(ranks);net,g,dt,bundle=model('borrowed',discard,ranks,window,engine,threshold=PREDICATES[kind])
 out=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).gradients(np.zeros((1,4,1)),[0])
 loss,v,a,spikes,anchors=reference(bundle,g,kind,bundle.weights)
 net.run(4*dt,namespace={})
 for name,value in (('v',v),('a',a)):
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
