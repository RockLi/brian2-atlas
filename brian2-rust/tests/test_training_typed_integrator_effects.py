"""Typed borrowed states and coefficient copies in generated integrators."""
import ast
import copy
import brian2 as b
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer,lower_brian_dynamic_training
from brian2_rust.training_brian import _INTEGRATORS
from test_training_typed_callback_effects import increment,fill
from test_training_integer_ir import engine
from test_training_poisson_zero_vjp import mpi
from test_training_stochastic import normal
from test_native_training import RUNNER

def model(kind,method,discard,ranks,window,backend,refractory=False):
 b.set_device('runtime');b.start_scope();b.prefs.codegen.target='numpy';b.prefs.codegen.runtime.numpy.discard_units=discard
 dt=.2*b.ms;source=b.SpikeGeneratorGroup(1,[],np.array([])*b.second,dt=dt)
 hidden=b.NeuronGroup(1,'dv/dt=0/second:1',threshold='v>100',reset='v=0',method='euler',dt=dt)
 callback=b.Function(fill if 'boolean' in kind else increment,arg_units=[1],arg_names=['x'],return_unit=1,stateless=False)
 argument={'integer_state':'k','boolean_state':'flag','integer_coefficient':'offset+1','boolean_coefficient':'mask==False'}[kind]
 noisy=method in ('heun','milstein')
 equations='dv/dt=.1*gain*change('+argument+')/ms'+('+gain*xi/ms**.5' if noisy else '')+':1\nk:integer\nflag:boolean\noffset:integer (constant)\nmask:boolean (constant)\ngain:1 (constant)'
 if refractory:equations=equations.replace(':1\nk:integer',':1 (unless refractory)\nk:integer')
 g=b.NeuronGroup(2,equations,method=method,threshold='v>.5',reset='v-=.5',dt=dt,namespace={'change':callback},
                 refractory=.4*b.ms if refractory else False)
 g.v=[.3,.7];g.k=[2,3];g.flag=[False,True];g.offset=[1,2];g.mask=[True,False];g.gain=[.2,.3]
 if refractory:g.lastspike=[0.,-1.]*b.second;g.not_refractory=[False,True]
 net=b.Network(source,hidden,g)
 bundle=lower_brian_dynamic_training(net,input_group=source,layers=[hidden,g],backend=backend,mpi_ranks=ranks,
                                    tbptt_window=window,detach_reset=False,seed=7123,trainable_neuron_parameters={g.name:['gain']})
 # This independent executable is Brian's generated stage block, with the
 # Python/NumPy callback itself providing storage and alias behavior.
 code=_INTEGRATORS[method](g.equations,variables={**g.variables,'change':callback})
 streams=sorted(g.equations.stochastic_variables)
 order=[streams.index(stmt.targets[0].id) for stmt in ast.parse(code).body
        if isinstance(stmt,ast.Assign) and stmt.targets[0].id in streams]
 if refractory:
  from brian2.codegen.translation import make_statements
  from brian2.codegen.generators.numpy_generator import NumpyCodeGenerator
  from brian2.codegen.runtime.numpy_rt import NumpyCodeObject
  variables={**b.core.functions.DEFAULT_FUNCTIONS,**g.variables,'change':callback}
  scalar,vector=make_statements(code,variables,np.float64,optimise=True)
  generator=NumpyCodeGenerator(variables,g.variables.indices,g,{'_idx'},NumpyCodeObject,
                               g.state_updater.name,'stateupdate',allows_scalar_write=True)
  tree=ast.parse('\n'.join(generator.translate_statement(stmt) for stmt in [*scalar,*vector]))
  class Indexed(ast.NodeTransformer):
   def visit_Name(self,node):
    return ast.copy_location(ast.Subscript(value=ast.Name(id=node.id,ctx=ast.Load()),slice=ast.Name(id='not_refractory',ctx=ast.Load()),ctx=node.ctx),node) if node.id in ('v','_v') else node
  for stmt in tree.body:
   if isinstance(stmt,ast.Assign) and stmt.targets[0].id=='v':
    stmt.targets[0]=Indexed().visit(stmt.targets[0]);stmt.value=Indexed().visit(stmt.value)
  code=ast.unparse(ast.fix_missing_locations(tree))
 return net,g,dt,bundle,compile(code,'<independent typed Brian stages>','exec'),order

def reference(bundle,g,kind,stage,order,weights,initial=None,anchors=None,refractory=False):
 p=bundle.plan;layout=bundle.provenance['neuron_state_layout'][g.name];z=np.array(bundle.initial_state if initial is None else initial,float)
 bank=next(row['bank'] for row in bundle.provenance['bindings'] if row['object']==g.name and row['variables']==['gain'])
 v=z[layout['v']].copy();k=z[layout['k']].astype(np.int32);flag=z[layout['flag']].astype(bool)
 before=[];margins=[];hard=[];spikes=[];last=np.array([0,-5000])
 for tick in range(4):
  if anchors is not None and p['tbptt_window'] and tick and tick%p['tbptt_window']==0:
   v,k,flag=(value.copy() for value in anchors['before'][tick])
  before.append((v.copy(),k.copy(),flag.copy()))
  draws=iter(np.array([normal(p['seed'],9,0,1,j,tick,stream) for j in range(2)]) for stream in order)
  env=dict(v=v.copy(),k=k.copy(),flag=flag.copy(),offset=np.array([1,2],np.int32),mask=np.array([True,False]),gain=np.array(weights[bank]),
           change=fill if 'boolean' in kind else increment,ms=.001,dt=.0002,randn=lambda *args:next(draws).copy(),sqrt=np.sqrt,
           not_refractory=tick-last>=2,_numpy=np,_vectorisation_idx=np.arange(2),int=lambda value:np.asarray(value,dtype=np.int32))
  exec(stage,env);v=env['v'];k=env['k'];flag=env['flag'];margin=v-.5
  active=tick-last>=2 if refractory else np.ones(2,dtype=bool);event=(margin>0).astype(float)*active
  margins.append(margin.copy());hard.append(event.copy())
  if anchors is not None:
   old=anchors['margins'][tick];event=anchors['hard'][tick]+active*p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(old))**2*(margin-old)
  spikes.append(event.copy());last[hard[-1].astype(bool)]=tick;v-=.5*event
 logits=np.array(spikes).mean(0)*p['logit_scale'];loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
 return loss,v,k,flag,np.array(spikes),dict(before=before,margins=margins,hard=hard)

@pytest.mark.parametrize('kind',['integer_state','boolean_state','integer_coefficient','boolean_coefficient'])
@pytest.mark.parametrize('method',['euler','rk2','rk4','heun','milstein'])
@pytest.mark.parametrize('discard',[False,True])
@pytest.mark.parametrize('ranks',[None,2])
@pytest.mark.parametrize('window',[None,2])
def test_typed_integrator_original_and_all_vjps(engine,kind,method,discard,ranks,window,monkeypatch):
 check_typed_integrator(engine,kind,method,discard,ranks,window,monkeypatch)

def check_typed_integrator(engine,kind,method,discard,ranks,window,monkeypatch,refractory=False):
 mpi(ranks);net,g,dt,bundle,stage,order=model(kind,method,discard,ranks,window,engine,refractory)
 noisy=bool(order);out=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).gradients(np.zeros((1,4,1)),[0],**({'noise_sequence':9} if noisy else {}))
 def oracle(weights,initial=None,anchors=None):return reference(bundle,g,kind,stage,order,weights,initial,anchors,refractory)
 loss,v,k,flag,spikes,anchors=oracle(bundle.weights)
 for name,value in (('v',v),('k',k),('flag',flag)):
  np.testing.assert_allclose(np.asarray(out['final_state'])[0,bundle.provenance['neuron_state_layout'][g.name][name]],value,rtol=7e-5,atol=7e-6)
 np.testing.assert_array_equal(np.asarray(out['spikes'])[0,:,1:],spikes);assert out['loss']==pytest.approx(loss,abs=7e-6)
 for bank,row in enumerate(bundle.weights):
  for index in range(len(row)):
   hi=copy.deepcopy(bundle.weights);lo=copy.deepcopy(bundle.weights);hi[bank][index]+=1e-6;lo[bank][index]-=1e-6
   fd=(oracle(hi,anchors=anchors)[0]-oracle(lo,anchors=anchors)[0])/2e-6
   assert out['gradients'][bank][index]==pytest.approx(fd,rel=9e-4,abs=9e-6),(bank,index)
 for index in range(len(bundle.initial_state)):
  if bundle.plan['dynamic']['detached'][index] or index in bundle.plan['dynamic']['integer_states']:
   assert out['initial_state_gradients'][0][index]==0.;continue
  hi=np.array(bundle.initial_state);lo=hi.copy();hi[index]+=1e-6;lo[index]-=1e-6
  fd=(oracle(bundle.weights,hi,anchors)[0]-oracle(bundle.weights,lo,anchors)[0])/2e-6
  assert out['initial_state_gradients'][0][index]==pytest.approx(fd,rel=9e-4,abs=9e-6),index
 if noisy:
  values=iter(np.array([normal(bundle.plan['seed'],9,0,1,j,tick,stream) for j in range(2)]) for tick in range(4) for stream in order)
  monkeypatch.setattr(np.random,'randn',lambda n:next(values).copy())
 net.run(4*dt,namespace={})
 for name,value in (('v',v),('k',k),('flag',flag)):
  np.testing.assert_allclose(g.variables[name].get_value(),value,rtol=7e-5,atol=7e-6)
 assert (out['gpu_dispatches']>0)==(engine!='cpu')

@pytest.mark.parametrize('kind',['integer_state','boolean_state','integer_coefficient','boolean_coefficient'])
@pytest.mark.parametrize('method',['euler','rk2','rk4','heun','milstein'])
@pytest.mark.parametrize('ranks',[None,2])
def test_typed_refractory_integrator_original_and_all_vjps(engine,kind,method,ranks,monkeypatch):
 check_typed_integrator(engine,kind,method,True,ranks,2,monkeypatch,refractory=True)
