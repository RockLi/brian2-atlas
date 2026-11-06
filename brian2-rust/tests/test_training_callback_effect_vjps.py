"""Public Brian effect conversion: independent noisy full/TBPTT recurrence."""
import copy
import brian2 as b
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer,lower_brian_dynamic_training
from test_training_callback_effects import curve, promoted_curve
from test_training_integer_ir import engine
from test_training_poisson_zero_vjp import mpi
from test_training_stochastic import normal
from test_native_training import RUNNER

CODES={
 'borrowed':'v=curve(v)+v+gain+.05*randn()',
 'temporary':'v=curve(v+.125)+v-.1+gain+.05*randn()',
 'scalar':'v=curve(.5)+v+gain+.05*randn()',
 'linked':'u=curve(u)+u+gain+.05*randn()',
 'promoted':'v=curve(.5,v)+gain+.05*randn()',
 'overlap_return':'a=curve(v);u+=a+gain+.05*randn()',
 'overlap_borrowed':'a=curve(u)+v+gain+.05*randn()',
 'overlap_rebind':'v=curve(u);a=v+gain+.05*randn()',
}


def model(kind,engine,ranks,window):
 b.set_device('runtime');b.start_scope();b.prefs.codegen.target='numpy';b.prefs.codegen.runtime.numpy.discard_units=True
 dt=.2*b.ms;source=b.SpikeGeneratorGroup(1,[],np.array([])*b.second,dt=dt)
 f=(b.Function(promoted_curve,arg_units=[1,1],arg_names=['x','y'],return_unit=1,stateless=False)
    if kind=='promoted' else b.Function(curve,arg_units=[1],arg_names=['x'],return_unit=1,stateless=False))
 overlap=kind.startswith('overlap_')
 equations=('dv/dt=.1*a/ms:1\na:1' if overlap else 'dv/dt=0/second:1')+'\ngain:1 (constant)'
 if kind=='linked' or overlap:equations+='\nu:1 (linked)'
 groups=[b.NeuronGroup(2,equations,threshold='v>.5',reset='v-=.5',
                      method='euler',dt=dt,namespace={'curve':f}) for _ in range(2)]
 runners=[]
 for group,initial,gain in zip(groups,[[.6,.1],[.7,.3]],[[.2,.3],[.4,.5]]):
  group.v=initial;group.gain=gain;runners.append(group.run_regularly(CODES[kind],dt=dt,when='groups',order=-2))
  if kind=='linked' or overlap:group.u=b.linked_var(group,'v')
  if overlap:group.a=0
 net=b.Network(source,*groups)
 bundle=lower_brian_dynamic_training(net,input_group=source,layers=groups,backend=engine,mpi_ranks=ranks,
          detach_reset=False,tbptt_window=window,seed=7123,trainable_neuron_parameters={g.name:['gain'] for g in groups})
 return net,source,groups,runners,dt,bundle


def reference(bundle,groups,runners,kind,w,initial=None,anchors=None):
 p=bundle.plan;z=np.array(bundle.initial_state if initial is None else initial,float);layout=bundle.provenance['neuron_state_layout']
 slots=[layout[g.name]['v'] for g in groups];spikes=[];margins=[];old=[]
 banks=[next(q['bank'] for q in bundle.provenance['bindings'] if q['object']==g.name and q['variables']==['gain']) for g in groups]
 domains=[bundle.provenance['regular_runner_layout'][r.name]['noise_domain'] for r in runners]
 for tick in range(4):
  if anchors is not None and p['tbptt_window'] and tick and tick%p['tbptt_window']==0:z=anchors[2][tick].copy()
  old.append(z.copy())
  for group,runner,indices,bank,domain in zip(groups,runners,slots,banks,domains):
   value=z[indices].copy();noise=np.array([normal(p['seed'],9,0,domain,j,tick,0) for j in range(2)])
   if kind.startswith('overlap_'):
    physical=value;gain=np.array(w[bank]);aux=z[layout[group.name]['a']].copy()
    if kind=='overlap_return':
     u=physical.copy();a=curve(physical);u+=a+gain+.05*noise
     locals_={'u':u,'a':a}
    elif kind=='overlap_borrowed':locals_={'a':curve(physical)+physical+gain+.05*noise}
    else:locals_={'v':curve(physical),'a':physical+gain+.05*noise}
    for name in bundle.provenance['regular_runner_layout'][runner.name]['numpy_write_order']['vector']:
     if name=='a':aux[:]=locals_[name]
     else:physical[:]=locals_[name]
    z[layout[group.name]['a']]=aux;value=physical+.02*aux
   elif kind in ('borrowed','linked'):value=curve(value)+value+np.array(w[bank])+.05*noise
   elif kind=='temporary':value=curve(value+.125)+value-.1+np.array(w[bank])+.05*noise
   elif kind=='promoted':value=promoted_curve(.5,value)+np.array(w[bank])+.05*noise
   else:value=curve(.5)+value+np.array(w[bank])+.05*noise
   z[indices]=value
  margin=z[slots[0]+slots[1]]-.5;gate=(margin>0).astype(float)
  if anchors is not None:
   original=anchors[0][tick]
   gate=(original>0).astype(float)+p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(original))**2*(margin-original)
  margins.append(margin.copy());spikes.append(gate.copy());z[slots[0]+slots[1]]-=.5*gate
 logits=np.array(spikes)[:,2:].mean(0)*p['logit_scale'];loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
 return loss,z,np.array(spikes),(np.array(margins),np.array(spikes),np.array(old))


@pytest.mark.parametrize('kind',list(CODES))
@pytest.mark.parametrize('ranks',[None,2])
@pytest.mark.parametrize('window',[None,2])
def test_public_effect_noise_bank_initial_and_tbptt_vjps(engine,kind,ranks,window):
 mpi(ranks);net,source,groups,runners,dt,bundle=model(kind,engine,ranks,window)
 x=np.zeros((1,4,1));w=bundle.weights
 result=NativeLIFTrainer(bundle.plan,weights=w,runner=RUNNER).gradients(x,[0],noise_sequence=9)
 loss,z,spikes,anchors=reference(bundle,groups,runners,kind,w)
 np.testing.assert_allclose(result['final_state'][0],z,rtol=3e-5,atol=3e-6)
 np.testing.assert_array_equal(result['spikes'][0],spikes);assert result['loss']==pytest.approx(loss,abs=3e-6)
 for bank,row in enumerate(w):
  for j in range(len(row)):
   hi=copy.deepcopy(w);lo=copy.deepcopy(w);hi[bank][j]+=1e-6;lo[bank][j]-=1e-6
   fd=(reference(bundle,groups,runners,kind,hi,anchors=anchors)[0]-reference(bundle,groups,runners,kind,lo,anchors=anchors)[0])/2e-6
   assert result['gradients'][bank][j]==pytest.approx(fd,rel=5e-4,abs=5e-6)
 initial_slots={j for group in groups for indices in bundle.provenance['neuron_state_layout'][group.name].values() for j in indices}
 for j in sorted(initial_slots):
  hi=np.array(bundle.initial_state,float);lo=hi.copy();hi[j]+=1e-6;lo[j]-=1e-6
  fd=(reference(bundle,groups,runners,kind,w,initial=hi,anchors=anchors)[0]-reference(bundle,groups,runners,kind,w,initial=lo,anchors=anchors)[0])/2e-6
  assert result['initial_state_gradients'][0][j]==pytest.approx(fd,rel=5e-4,abs=5e-6)
 assert (result['gpu_dispatches']>0)==(engine!='cpu')
