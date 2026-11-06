"""Reset subset-copy effects against actual Brian and independent VJPs."""
import copy
import brian2 as b
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer,lower_brian_dynamic_training
from test_training_callback_effects import curve
from test_training_typed_callback_effects import increment,fill
from test_training_integer_ir import engine
from test_training_poisson_zero_vjp import mpi
from test_native_training import RUNNER

CODES={
 'borrowed':'v=.2*curve(v)+v',
 'hidden_copy':'v=.2*curve(a)+v-.5',
 'sequential':'a=curve(v);v+=a-.5',
 'parameter_copy':'v=.2*curve(gain)+v-.5',
 'integer':'k=change(k);v-=.1*k',
 'boolean':'flag=change(flag);v-=.5*int(flag)',
}

def model(kind,discard,ranks,window,backend):
 b.set_device('runtime');b.start_scope();b.prefs.codegen.target='numpy';b.prefs.codegen.runtime.numpy.discard_units=discard
 dt=.2*b.ms;source=b.SpikeGeneratorGroup(1,[],np.array([])*b.second,dt=dt,name='reset_effect_input')
 hidden=b.NeuronGroup(1,'dv/dt=0/second:1',threshold='v>100',reset='v=0',method='euler',dt=dt,name='reset_effect_hidden')
 f=b.Function(curve,arg_units=[1],arg_names=['x'],return_unit=1,stateless=False)
 change=b.Function(fill if kind=='boolean' else increment,arg_units=[1],arg_names=['x'],return_unit=1,stateless=False)
 g=b.NeuronGroup(2,'dv/dt=gain/ms:1\na:1\nk:integer\nflag:boolean\ngain:1 (constant)',
                 method='euler',threshold='v>.5',reset=CODES[kind],dt=dt,
                 namespace={'curve':f,'change':change},name='reset_effect_neurons')
 g.v=[.3,.7];g.a=[.4,.6];g.k=[2,3];g.flag=[False,True];g.gain=[.2,.3]
 net=b.Network(source,hidden,g)
 bundle=lower_brian_dynamic_training(net,input_group=source,layers=[hidden,g],backend=backend,mpi_ranks=ranks,
                                    tbptt_window=window,detach_reset=False,trainable_neuron_parameters={g.name:['gain']})
 return net,g,dt,bundle

@pytest.mark.parametrize('kind',list(CODES))
@pytest.mark.parametrize('discard',[False,True])
@pytest.mark.parametrize('ranks',[None,2])
def test_reset_subset_original_and_restored_carry(engine,kind,discard,ranks,tmp_path):
 mpi(ranks);net,g,dt,bundle=model(kind,discard,ranks,None,engine)
 bundle.plan['trainable']=[False]*len(bundle.weights)
 trainer=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER)
 for tick in range(3):
  out=trainer.step(np.zeros((1,1,1)),[0],initial='carry' if tick else None);net.run(dt,namespace={})
  for name in ('v','a','k','flag'):
   value=np.asarray(out['final_state'])[0,bundle.provenance['neuron_state_layout'][g.name][name]]
   np.testing.assert_allclose(value,g.variables[name].get_value(),rtol=7e-5,atol=7e-6)
  path=tmp_path/str(tick);trainer.store(path)
  restored=NativeLIFTrainer(trainer.plan,runner=RUNNER);restored.restore(path);trainer=restored
 assert (out['gpu_dispatches']>0)==(engine!='cpu')

def reference(bundle,g,kind,weights,initial=None,anchors=None):
 p=bundle.plan;layout=bundle.provenance['neuron_state_layout'][g.name]
 z=np.array(bundle.initial_state if initial is None else initial,float)
 bank=next(row['bank'] for row in bundle.provenance['bindings'] if row['object']==g.name and row['variables']==['gain'])
 values={name:z[layout[name]].copy() for name in ('v','a','k','flag')}
 values['k']=values['k'].astype(np.int32);values['flag']=values['flag'].astype(bool)
 before=[];margins=[];hard=[];spikes=[]
 assigned={'borrowed':('v',),'hidden_copy':('v',),'sequential':('a','v'),
           'parameter_copy':('v',),'integer':('k','v'),'boolean':('flag','v')}[kind]
 for tick in range(4):
  if anchors is not None and p['tbptt_window'] and tick and tick%p['tbptt_window']==0:
   values={name:a.copy() for name,a in anchors['before'][tick].items()}
  before.append({name:a.copy() for name,a in values.items()})
  values['v']+=.2*np.array(weights[bank]);margin=values['v']-.5;event=(margin>0).astype(float)
  margins.append(margin.copy());hard.append(event.copy())
  if anchors is not None:
   old=anchors['margins'][tick]
   event=anchors['hard'][tick]+p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(old))**2*(margin-old)
  spikes.append(event.copy())
  # Elementwise callback bodies commute across selected rows. Compute each
  # reset candidate on independent copies; only explicit targets scatter back.
  env={name:a.copy() for name,a in values.items()}
  env.update(gain=np.array(weights[bank]),curve=curve,change=fill if kind=='boolean' else increment,
             int=lambda a:np.asarray(a,dtype=np.int32))
  exec(CODES[kind],env)
  for name in assigned:
   if name in ('k','flag'):
    gate=anchors['hard'][tick] if anchors is not None else hard[-1]
    values[name]=np.where(gate.astype(bool),env[name],values[name]).astype(np.int32 if name=='k' else bool)
   else:values[name]=values[name]+event*(env[name]-values[name])
 logits=np.array(spikes).mean(0)*p['logit_scale'];loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
 return loss,values,np.array(spikes),dict(before=before,margins=margins,hard=hard)

@pytest.mark.parametrize('kind',list(CODES))
@pytest.mark.parametrize('window',[None,2])
@pytest.mark.parametrize('ranks',[None,2])
def test_reset_subset_all_parameters_and_initial_vjps(engine,kind,window,ranks):
 mpi(ranks);_,g,_,bundle=model(kind,True,ranks,window,engine)
 out=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).gradients(np.zeros((1,4,1)),[0])
 loss,values,spikes,anchors=reference(bundle,g,kind,bundle.weights)
 for name,value in values.items():
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
