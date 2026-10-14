"""Actual NumPy per-statement ufunc.at and retained callback array aliases."""
import copy
import brian2 as b
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer,lower_brian_dynamic_training
from test_training_integer_ir import engine
from test_training_poisson_zero_vjp import mpi
from test_training_stochastic import normal
from test_native_training import RUNNER

CODES={
 'repeat':'peer+=gain*curve(v);peer+=.05*curve(v)',
 'multiply':'peer*=1+.1*curve(v);peer/=(1+.05*curve(v))',
 'borrowed':'tmp=curve(v);peer+=gain*tmp;v+=.1*tmp',
 'private_alias':'tmp=curve(v+.1);alias=tmp;peer+=gain*tmp;tmp*=.8;v+=.1*alias',
 'rebind':'tmp=curve(v+.1);alias=tmp;tmp=tmp+.2;peer+=gain*tmp;v+=.1*alias',
 'overlap_rhs':'peer+=curve(v)',
 'noise':'tmp=curve(randn());peer+=.01*tmp;v+=.1*gain',
 'integer_copy':'tmp=change(k+1);kpeer+=tmp;v+=.01*k',
 'integer_wrap':'tmp=change(k+1);kpeer+=tmp;v+=.01*k',
 'boolean':'flagpeer+=fill(flag);flagpeer*=fill(flag);v+=.1*gain*int(flag)',
 'integer_float':'kpeer+=.25*curve(v);v+=.01*k',
 'integer_divide':'kpeer/=(1+.1*curve(v));v+=.01*k',
 'boolean_integer':'flagpeer+=change(k);v+=.1*gain*int(flag)',
 'boolean_subfloat':'flagpeer-=.1*curve(v);v+=.1*gain*int(flag)',
}

def model(kind,discard,ranks,window,backend):
 from brian2.codegen.translation import make_statements
 from brian2.codegen.generators.numpy_generator import NumpyCodeGenerator
 from brian2.codegen.runtime.numpy_rt import NumpyCodeObject
 from test_training_callback_effects import curve
 from test_training_typed_callback_effects import increment,fill
 b.set_device('runtime');b.start_scope();b.prefs.codegen.target='numpy';b.prefs.codegen.runtime.numpy.discard_units=discard
 dt=.2*b.ms;inp=b.SpikeGeneratorGroup(1,[],np.array([])*b.second,dt=dt)
 hidden=b.NeuronGroup(1,'dv/dt=0/second:1',threshold='v>100',reset='v=0',method='euler',dt=dt)
 g=b.NeuronGroup(4,'dv/dt=0/second:1\nk:integer\nflag:boolean\npick:integer\npeer:1 (linked)\nkpeer:integer (linked)\nflagpeer:boolean (linked)\ngain:1 (constant)',
                 threshold='v>.5',reset='v-=.5',method='euler',dt=dt)
 g.v=[.1,.4,.7,.2];g.k=[2147483647,-2147483648,2,3] if kind=='integer_wrap' else [1,2,3,4]
 if kind in ('integer_float','integer_divide'):g.k=[-1,-2,3,4]
 g.flag=[True,False,True,False];g.pick=[1,1,0,2];g.gain=[.1,.2,.3,.4]
 for peer,field in [('peer','v'),('kpeer','k'),('flagpeer','flag')]:setattr(g,peer,b.linked_var(g,field,index='pick'))
 for name,callback in [('curve',curve),('change',increment),('fill',fill)]:g.namespace[name]=b.Function(callback,arg_units=[1],arg_names=['x'],return_unit=1,stateless=False)
 runner=g.run_regularly(CODES[kind],when='start');net=b.Network(inp,hidden,g)
 bundle=lower_brian_dynamic_training(net,input_group=inp,layers=[hidden,g],backend=backend,mpi_ranks=ranks,tbptt_window=window,detach_reset=False,
                                    seed=7123,trainable_neuron_parameters={g.name:['gain']})
 assert bundle.provenance['regular_runner_layout'][runner.name]['numpy_mode']=='vectorised'
 variables={**b.core.functions.DEFAULT_FUNCTIONS,**g.variables,**g.namespace}
 scalar,vector=make_statements(CODES[kind],variables,np.float64,optimise=True)
 generator=NumpyCodeGenerator(variables,g.variables.indices,runner,{'_idx'},NumpyCodeObject,runner.name,'stateupdate',allows_scalar_write=True)
 snippets=[compile('\n'.join(generator.vectorise_code([s],variables,g.variables.indices)),'<actual NumPy statement>','exec') for s in vector]
 return net,g,runner,dt,bundle,variables,generator,snippets

class NumpyProxy:
 def __init__(self,env):self.env=env;self.operand=None;self.before_at={}
 def __getattr__(self,name):
  function=getattr(np,name)
  if name not in ('add','subtract','multiply','divide'):return function
  proxy=self
  class Ufunc:
   @staticmethod
   def at(target,index,value):
    proxy.operand=np.asarray(value).copy()
    proxy.before_at={key:np.asarray(value).copy() for key,value in proxy.env.items() if key.startswith('_array_')}
    function.at(target,index,value)
  return Ufunc


def reference(bundle,g,runner,variables,generator,snippets,weights,initial=None,anchors=None):
 from test_training_callback_effects import curve
 from test_training_typed_callback_effects import increment,fill
 p=bundle.plan;z=np.array(bundle.initial_state if initial is None else initial,float);layout=bundle.provenance['neuron_state_layout'][g.name]
 bank=next(row['bank'] for row in bundle.provenance['bindings'] if row['object']==g.name and row['variables']==['gain'])
 regular=bundle.provenance['regular_runner_layout'][runner.name];before=[];margins=[];hard=[];spikes=[]
 for tick in range(4):
  if anchors is not None and p['tbptt_window'] and tick and tick%p['tbptt_window']==0:z=anchors['before'][tick].copy()
  before.append(z.copy());env=dict(curve=curve,change=increment,fill=fill,int=np.int32,_vectorisation_idx=np.arange(4));raw={}
  for name,var in variables.items():
   if not isinstance(var,b.core.variables.ArrayVariable):continue
   key=generator.get_array_name(var)
   if key in raw:continue
   physical=next((field for field in ['v','k','pick','flag'] if var is g.variables[field]),None)
   value=(z[layout[physical]].copy() if physical else np.array(weights[bank]) if var is g.variables['gain'] else np.asarray(var.get_value()).copy())
   raw[key]=np.asarray(value,dtype=var.dtype);env[key]=raw[key]
  random_stream=[0]
  def randn(index):
   stream=random_stream[0];random_stream[0]+=1
   return np.array([normal(p['seed'],9,0,regular['noise_domain'],j,tick,stream) for j in range(4)])
  env['randn']=randn;env['_randn']=randn;proxy=NumpyProxy(env);env['_numpy']=proxy
  for stage,snippet in zip(regular['vectorised_stages'],snippets):
   proxy.operand=None;proxy.before_at={}
   for j,row in enumerate(stage['rows']):
    for name,slot in row['captures'].items():
     if isinstance(variables.get(name),b.core.variables.ArrayVariable):
      var=variables[name];array=env[generator.get_array_name(var)];index=g.variables.indices[name]
      value=array if index=='_idx' else array[env[generator.get_array_name(variables[index])]]
     else:value=env[name]
     z[slot]=np.asarray(value).reshape(-1)[j] if np.asarray(value).ndim else value
   exec(snippet,env)
   for j,row in enumerate(stage['rows']):
    for name,slot in {**row['outputs'],**row['direct_outputs']}.items():
     if name==stage['target'] and stage['accumulate'] or name==stage['result'] and stage['accumulate']:value=proxy.operand
     elif name==stage['result'] and stage['result']!=stage['target']:value=env[stage['target']]
     elif proxy.before_at and name in variables and isinstance(variables[name],b.core.variables.ArrayVariable):value=proxy.before_at.get(generator.get_array_name(variables[name]),env.get(name))
     else:value=env[name]
     z[slot]=np.asarray(value).reshape(-1)[j] if np.asarray(value).ndim else value
    for name in row['outputs']:
     if name in row.get('array_locals',[]):
      cell=row['origins'][name]
      if cell not in layout['v']:z[cell]=np.asarray(env[name]).reshape(-1)[j]
   for field in ['v','k','pick','flag']:z[layout[field]]=env[generator.get_array_name(g.variables[field])]
  v=z[layout['v']];margin=v-.5;event=(margin>0).astype(float);margins.append(margin.copy());hard.append(event.copy())
  if anchors is not None:
   old=anchors['margins'][tick];event=anchors['hard'][tick]+p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(old))**2*(margin-old)
  spikes.append(event.copy());z[layout['v']]-=.5*event
 logits=np.array(spikes).mean(0)*p['logit_scale'];loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
 return loss,z,np.array(spikes),dict(before=before,margins=margins,hard=hard)

@pytest.mark.parametrize('kind',list(CODES))
@pytest.mark.parametrize('discard',[False,True])
@pytest.mark.parametrize('ranks',[None,2])
@pytest.mark.parametrize('window',[None,2])
def test_regular_ufunc_original_and_all_vjps(engine,kind,discard,ranks,window,monkeypatch):
 mpi(ranks);net,g,runner,dt,bundle,variables,generator,snippets=model(kind,discard,ranks,window,engine)
 options={'noise_sequence':9} if kind=='noise' else {}
 out=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).gradients(np.zeros((1,4,1)),[0],**options)
 def oracle(weights,initial=None,anchors=None):return reference(bundle,g,runner,variables,generator,snippets,weights,initial,anchors)
 loss,z,spikes,anchors=oracle(bundle.weights)
 np.testing.assert_allclose(out['final_state'][0],z,rtol=8e-5,atol=8e-6);np.testing.assert_array_equal(np.asarray(out['final_state'])[0,bundle.plan['dynamic']['integer_states']],z[bundle.plan['dynamic']['integer_states']]);np.testing.assert_array_equal(np.array(out['spikes'])[0,:,1:],spikes);assert out['loss']==pytest.approx(loss,abs=8e-6)
 for bank,row in enumerate(bundle.weights):
  for index in range(len(row)):
   hi=copy.deepcopy(bundle.weights);lo=copy.deepcopy(bundle.weights);hi[bank][index]+=1e-6;lo[bank][index]-=1e-6
   fd=(oracle(hi,anchors=anchors)[0]-oracle(lo,anchors=anchors)[0])/2e-6
   assert out['gradients'][bank][index]==pytest.approx(fd,rel=1e-3,abs=1e-5),(bank,index)
 for index in range(len(bundle.initial_state)):
  if bundle.plan['dynamic']['detached'][index] or index in bundle.plan['dynamic']['integer_states']:
   assert out['initial_state_gradients'][0][index]==0.;continue
  hi=np.array(bundle.initial_state);lo=hi.copy();hi[index]+=1e-6;lo[index]-=1e-6
  fd=(oracle(bundle.weights,hi,anchors=anchors)[0]-oracle(bundle.weights,lo,anchors=anchors)[0])/2e-6
  assert out['initial_state_gradients'][0][index]==pytest.approx(fd,rel=1e-3,abs=1e-5),index
 if kind=='noise':
  draws=iter(np.array([normal(bundle.plan['seed'],9,0,bundle.provenance['regular_runner_layout'][runner.name]['noise_domain'],j,tick,0) for j in range(4)]) for tick in range(4))
  monkeypatch.setattr(np.random,'randn',lambda n:next(draws).copy())
 net.run(4*dt,namespace={})
 np.testing.assert_allclose(np.asarray(out['final_state'])[0,bundle.provenance['neuron_state_layout'][g.name]['v']],g.v[:],rtol=8e-5,atol=8e-6)
 assert (out['gpu_dispatches']>0)==(engine!='cpu')


@pytest.mark.parametrize('kind',list(CODES))
@pytest.mark.parametrize('ranks',[None,2])
def test_regular_ufunc_carry_and_fresh_restore(engine,kind,ranks,tmp_path):
 mpi(ranks);_,_,_,_,bundle,_,_,_=model(kind,True,ranks,None,engine)
 plan=copy.deepcopy(bundle.plan);plan['trainable']=[False]*len(plan['trainable']);x=np.zeros((1,4,1))
 options={'noise_sequence':9} if kind=='noise' else {}
 whole=NativeLIFTrainer(plan,weights=bundle.weights,runner=RUNNER).evaluate(x,[0],**options)
 trainer=NativeLIFTrainer(plan,weights=bundle.weights,runner=RUNNER);first=trainer.step(x[:,:2],[0],**options)
 path=tmp_path/'ufunc';trainer.store(path);restored=NativeLIFTrainer(plan,runner=RUNNER);restored.restore(path)
 last=restored.step(x[:,2:],[0],initial='carry')
 np.testing.assert_allclose(last['final_state'],whole['final_state'],rtol=8e-5,atol=8e-6)
 np.testing.assert_array_equal(np.concatenate([first['spikes'],last['spikes']],axis=1),whole['spikes'])
 indices=plan['dynamic']['integer_states'];np.testing.assert_array_equal(np.asarray(last['final_state'])[0,indices],np.asarray(whole['final_state'])[0,indices])
 assert restored.clock_tick==4
 if kind=='noise':assert restored.noise_sequence==9 and restored.next_noise_sequence==10
 assert (last['gpu_dispatches']>0)==(engine!='cpu')


@pytest.mark.parametrize('ranks',[None,2])
def test_regular_ufunc_refractory_copies_and_carry(engine,ranks):
 from test_training_callback_effects import curve
 mpi(ranks);b.set_device('runtime');b.start_scope();b.prefs.codegen.target='numpy';dt=.2*b.ms
 inp=b.SpikeGeneratorGroup(1,[],np.array([])*b.second,dt=dt)
 hidden=b.NeuronGroup(1,'dv/dt=0/second:1',threshold='v>100',reset='v=0',method='euler',dt=dt)
 g=b.NeuronGroup(4,'dv/dt=.1/ms:1 (unless refractory)\npick:integer\npeer:1 (linked)\ngain:1 (constant)',
                 threshold='v>.5',reset='v-=.5',refractory=.4*b.ms,method='euler',dt=dt)
 g.v=[.6,.4,.7,.2];g.pick=[1,1,0,2];g.gain=[.1,.2,.3,.4];g.peer=b.linked_var(g,'v',index='pick')
 net=b.Network(inp,hidden,g);net.run(dt,namespace={})
 g.namespace['curve']=b.Function(curve,arg_units=[1],arg_names=['x'],return_unit=1,stateless=False)
 r=g.run_regularly('peer+=gain*curve(v)',when='start')
 bundle=lower_brian_dynamic_training(net,input_group=inp,layers=[hidden,g],backend=engine,mpi_ranks=ranks,detach_reset=False)
 assert bundle.provenance['regular_runner_layout'][r.name]['numpy_mode']=='vectorised'
 trainer=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER)
 for tick in range(4):
  out=trainer.step(np.zeros((1,1,1)),[0],initial='carry' if tick else None);net.run(dt,namespace={})
  np.testing.assert_allclose(np.asarray(out['final_state'])[0,bundle.provenance['neuron_state_layout'][g.name]['v']],g.v[:],rtol=8e-5,atol=8e-6)
 assert (out['gpu_dispatches']>0)==(engine!='cpu')


def test_regular_ufunc_boolean_subtraction_matches_numpy_rejection():
 from brian2_rust import TrainingConversionError
 net,g,r,dt,_,_,_,_=model('boolean',True,None,None,'cpu');before=np.asarray(g.flag[:]).copy()
 r.abstract_code='flagpeer-=fill(flag)'
 inp=next(o for o in net.objects if isinstance(o,b.SpikeGeneratorGroup));hidden=next(o for o in net.objects if isinstance(o,b.NeuronGroup) and len(o)==1)
 with pytest.raises(TrainingConversionError,match='boolean subtraction'):
  lower_brian_dynamic_training(net,input_group=inp,layers=[hidden,g])
 np.testing.assert_array_equal(g.flag[:],before)
 with pytest.raises(b.core.base.BrianObjectException) as caught:net.run(dt,namespace={})
 assert isinstance(caught.value.__cause__,TypeError)


@pytest.mark.parametrize('ranks',[None,2])
def test_regular_ufunc_empty_mask_scalar_domain_is_atomic(engine,ranks):
 import warnings
 from test_training_empty_reset_effects import reciprocal_curve
 mpi(ranks);b.set_device('runtime');b.start_scope();b.prefs.codegen.target='numpy';dt=.2*b.ms
 inp=b.SpikeGeneratorGroup(1,[],np.array([])*b.second,dt=dt)
 hidden=b.NeuronGroup(1,'dv/dt=0/second:1',threshold='v>100',reset='v=0',method='euler',dt=dt)
 g=b.NeuronGroup(4,'dv/dt=0/second:1 (unless refractory)\npick:integer\npeer:1 (linked)\nrate:1 (constant, shared)',
                 threshold='v>.5',reset='v-=.5',refractory=.4*b.ms,method='euler',dt=dt)
 g.v=[.6,.7,.8,.9];g.pick=[1,1,0,2];g.peer=b.linked_var(g,'v',index='pick');g.rate=0
 net=b.Network(inp,hidden,g);net.run(dt,namespace={});np.testing.assert_array_equal(g.not_refractory[:],False)
 g.namespace['check']=b.Function(reciprocal_curve,arg_units=[1,1],arg_names=['x','rate'],return_unit=1,stateless=False)
 r=g.run_regularly('peer+=check(v,rate)',when='start')
 bundle=lower_brian_dynamic_training(net,input_group=inp,layers=[hidden,g],backend=engine,mpi_ranks=ranks,trainable_neuron_parameters={g.name:['rate']})
 assert bundle.provenance['regular_runner_layout'][r.name]['numpy_mode']=='vectorised'
 trainer=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER);before=copy.deepcopy(trainer.state);tick=trainer.clock_tick
 with pytest.raises(ValueError,match='non.?finite|Non.?finite|domain|invalid dynamic'):trainer.step(np.zeros((1,1,1)),[0])
 assert trainer.state==before and trainer.neuron_state is None and trainer.clock_tick==tick and trainer.next_noise_sequence==0
 try:
  with warnings.catch_warnings(record=True) as caught:
   warnings.simplefilter('always',RuntimeWarning);net.run(dt,namespace={})
  assert any(isinstance(w.message,RuntimeWarning) for w in caught)
 except b.core.base.BrianObjectException as error:assert isinstance(error.__cause__,ZeroDivisionError)
