"""NumPy int32/Boolean callback aliases and physical writeback casts."""
import copy
import warnings
import brian2 as b
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer,lower_brian_dynamic_training,TrainingConversionError
from test_training_integer_ir import engine
from test_training_poisson_zero_vjp import mpi
from test_native_training import RUNNER

def increment(x):
 saved=x
 x+=1
 return saved

def clear(x):
 saved=x
 x*=False
 return saved

def fill(x):
 saved=x
 x+=True
 return saved

def modulo(x):
 saved=x
 x%=3
 return saved

def square(x):
 x*=1
 return x**2

CODES={
 'borrowed':'v=.8*v+.1*gain*change(k)',
 'integer_copy':'v=.8*v+.1*gain*change(k+1)',
 'float_copy':'v=.8*v+.1*gain*change(k*1.)',
 'rebind':'k=.8*change(k);v=.8*v+.1*gain*k',
 'cast':'k=change(gain+.125);v=.8*v+.1*k',
 'bool_clear':'v=.8*v+.1*gain*change(flag)',
 'bool_fill':'v=.8*v+.1*gain*change(flag)',
 'wrap':'v=1e-10*change(k)',
 'modulo':'v=.8*v+.1*gain*change(k)',
 'power':'v=1e-10*change(k)',
 'overflow_cast':'k=change(gain+2147483648.);v=1e-10*k',
}

def model(kind,discard,ranks,window,backend):
 b.set_device('runtime');b.start_scope();b.prefs.codegen.target='numpy';b.prefs.codegen.runtime.numpy.discard_units=discard
 dt=.2*b.ms;source=b.SpikeGeneratorGroup(1,[],np.array([])*b.second,dt=dt,name='typed_effect_input')
 hidden=b.NeuronGroup(1,'dv/dt=0/second:1',threshold='v>100',reset='v=0',method='euler',dt=dt,name='typed_effect_hidden')
 callback=clear if kind=='bool_clear' else fill if kind=='bool_fill' else modulo if kind=='modulo' else square if kind=='power' else increment
 f=b.Function(callback,arg_units=[1],arg_names=['x'],return_unit=1,stateless=False)
 g=b.NeuronGroup(2,'dv/dt=0/second:1\nk:integer\nflag:boolean\ngain:1 (constant)',method='euler',threshold='v>.5',reset='v-=.5',
                 dt=dt,namespace={'change':f},name='typed_effect_neurons')
 g.v=[.3,.7];g.k=[2147483647,-2147483648] if kind in ('wrap','power') else [2,3];g.flag=[True,False];g.gain=[.2,.3]
 g.run_regularly(CODES[kind],when='start')
 net=b.Network(source,hidden,g)
 bundle=lower_brian_dynamic_training(net,input_group=source,layers=[hidden,g],backend=backend,mpi_ranks=ranks,tbptt_window=window,
                                    detach_reset=False,trainable_neuron_parameters={g.name:['gain']})
 return net,g,dt,bundle

@pytest.mark.parametrize('kind',list(CODES))
@pytest.mark.parametrize('discard',[False,True])
@pytest.mark.parametrize('ranks',[None,2])
def test_typed_effect_original_and_restored_carry(engine,kind,discard,ranks,tmp_path):
 mpi(ranks);net,g,dt,bundle=model(kind,discard,ranks,None,engine)
 bundle.plan['trainable']=[False]*len(bundle.weights)
 trainer=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER)
 for tick in range(3):
  out=trainer.step(np.zeros((1,1,1)),[0],initial='carry' if tick else None);net.run(dt,namespace={})
  for name in ('v','k','flag'):
   values=np.asarray(out['final_state'])[0,bundle.provenance['neuron_state_layout'][g.name][name]]
   np.testing.assert_allclose(values,g.variables[name].get_value(),rtol=7e-5,atol=7e-6) if name=='v' else np.testing.assert_array_equal(values,g.variables[name].get_value())
  path=tmp_path/('typed-'+str(tick));trainer.store(path)
  restored=NativeLIFTrainer(trainer.plan,runner=RUNNER);restored.restore(path);trainer=restored
 assert (out['gpu_dispatches']>0)==(engine!='cpu')


def reference(bundle,g,kind,weights,initial=None,anchors=None):
 p=bundle.plan;layout=bundle.provenance['neuron_state_layout'][g.name]
 z=np.array(bundle.initial_state if initial is None else initial,float)
 v=z[layout['v']].copy();k=z[layout['k']].astype(np.int32);flag=z[layout['flag']].astype(bool)
 bank=next(row['bank'] for row in bundle.provenance['bindings'] if row['object']==g.name and row['variables']==['gain'])
 gain=np.array(weights[bank]);callback=clear if kind=='bool_clear' else fill if kind=='bool_fill' else modulo if kind=='modulo' else square if kind=='power' else increment
 before=[];margins=[];hard=[];spikes=[]
 for tick in range(4):
  if anchors is not None and p['tbptt_window'] and tick and tick%p['tbptt_window']==0:
   v,k,flag=(a.copy() for a in anchors['before'][tick])
  before.append((v.copy(),k.copy(),flag.copy()))
  env=dict(v=v.copy(),k=k.copy(),flag=flag.copy(),gain=gain.copy(),change=callback)
  with warnings.catch_warnings():
   warnings.simplefilter('ignore',RuntimeWarning)
   exec(CODES[kind],env)
   # The local k can be a floating array through the next statement. Apply
   # array storage conversion only after all statements have read that local.
   v=env['v'].copy();target=np.empty_like(k);target[:]=env['k'];k=target;flag=np.asarray(env['flag'],bool).copy()
  margin=v-.5;event=(margin>0).astype(float)
  if anchors is not None:
   old=anchors['margins'][tick];event=anchors['hard'][tick]+p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(old))**2*(margin-old)
  margins.append(margin.copy());hard.append((margin>0).astype(float));spikes.append(event.copy());v-=.5*event
 logits=np.array(spikes).mean(0)*p['logit_scale'];loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
 return loss,v,k,flag,np.array(spikes),dict(before=before,margins=margins,hard=hard)


@pytest.mark.parametrize('kind',list(CODES))
@pytest.mark.parametrize('window',[None,2])
@pytest.mark.parametrize('ranks',[None,2])
def test_typed_effect_all_parameters_and_initial_vjps(engine,kind,window,ranks):
 mpi(ranks);_,g,_,bundle=model(kind,True,ranks,window,engine)
 out=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).gradients(np.zeros((1,4,1)),[0])
 loss,v,k,flag,spikes,anchors=reference(bundle,g,kind,bundle.weights)
 layout=bundle.provenance['neuron_state_layout'][g.name]
 np.testing.assert_allclose(np.asarray(out['final_state'])[0,layout['v']],v,rtol=7e-5,atol=7e-6)
 np.testing.assert_array_equal(np.asarray(out['final_state'])[0,layout['k']],k)
 np.testing.assert_array_equal(np.asarray(out['final_state'])[0,layout['flag']],flag)
 np.testing.assert_array_equal(np.asarray(out['spikes'])[0,:,1:],spikes);assert out['loss']==pytest.approx(loss,abs=7e-6)
 for bank,row in enumerate(bundle.weights):
  for index in range(len(row)):
   hi=copy.deepcopy(bundle.weights);lo=copy.deepcopy(bundle.weights);hi[bank][index]+=1e-6;lo[bank][index]-=1e-6
   fd=(reference(bundle,g,kind,hi,anchors=anchors)[0]-reference(bundle,g,kind,lo,anchors=anchors)[0])/2e-6
   assert out['gradients'][bank][index]==pytest.approx(fd,rel=9e-4,abs=9e-6),(bank,index)
 for index in range(len(bundle.initial_state)):
  if bundle.plan['dynamic']['detached'][index] or index in bundle.plan['dynamic']['integer_states'] or index in bundle.plan['dynamic']['binary_states']:
   assert out['initial_state_gradients'][0][index]==0.;continue
  hi=np.array(bundle.initial_state);lo=hi.copy();hi[index]+=1e-6;lo[index]-=1e-6
  fd=(reference(bundle,g,kind,bundle.weights,hi,anchors)[0]-reference(bundle,g,kind,bundle.weights,lo,anchors)[0])/2e-6
  assert out['initial_state_gradients'][0][index]==pytest.approx(fd,rel=9e-4,abs=9e-6),index
 assert (out['gpu_dispatches']>0)==(engine!='cpu')
