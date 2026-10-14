"""NumPy refractory integrator temporaries and callback writes before masks."""
import ast
import copy
import numpy as np
import brian2 as b
import pytest
from brian2_rust import NativeLIFTrainer,lower_brian_dynamic_training
from brian2_rust.training_brian import _INTEGRATORS
from test_training_callback_effects import curve
from test_training_integer_ir import engine
from test_training_poisson_zero_vjp import mpi
from test_training_stochastic import normal
from test_native_training import RUNNER


def model(kind,method,discard,ranks,window,backend):
 b.set_device('runtime');b.start_scope();b.prefs.codegen.target='numpy';b.prefs.codegen.runtime.numpy.discard_units=discard
 dt=.2*b.ms;source=b.SpikeGeneratorGroup(1,[],np.array([])*b.second,dt=dt,name='ref_effect_input')
 hidden=b.NeuronGroup(1,'dv/dt=0/second:1',method='euler',threshold='v>100',reset='v=0',dt=dt,name='ref_effect_hidden')
 f=b.Function(curve,arg_units=[1],arg_names=['x'],return_unit=1,stateless=False)
 noisy=method in ('heun','milstein');noise='gain*xi/ms**.5' if noisy else '.1*gain/ms'
 expr='curve('+('v' if kind=='drift' else 'a')+')/ms+'+noise
 g=b.NeuronGroup(2,'dv/dt='+expr+':1 (unless refractory)\na:1\ngain:1 (constant)',method=method,threshold='v>.5',reset='v-=.5',refractory=.4*b.ms,
                 dt=dt,namespace={'curve':f},name='ref_effect_neurons')
 g.v=[.4,.7];g.a=[.4,.6];g.gain=[.2,.3];g.lastspike=[0.,-1.]*b.second;g.not_refractory=[False,True]
 net=b.Network(source,hidden,g)
 bundle=lower_brian_dynamic_training(net,input_group=source,layers=[hidden,g],backend=backend,mpi_ranks=ranks,tbptt_window=window,
                                    detach_reset=False,seed=7123,trainable_neuron_parameters={g.name:['gain']})
 code=_INTEGRATORS[method](g.equations,variables=g.variables)
 # NumPy's eager where rewrite evaluates callback-bearing branches twice.
 # Generate the reference with Brian's actual statement translator, then
 # apply explicit conditional indexing independently of the native compiler.
 from brian2.codegen.translation import make_statements
 from brian2.codegen.generators.numpy_generator import NumpyCodeGenerator
 from brian2.codegen.runtime.numpy_rt import NumpyCodeObject
 variables={**b.core.functions.DEFAULT_FUNCTIONS,**g.variables,'curve':f}
 scalar,vector=make_statements(code,variables,np.float64,optimise=True)
 generator=NumpyCodeGenerator(variables,g.variables.indices,g,{'_idx'},NumpyCodeObject,
                             g.state_updater.name,'stateupdate',allows_scalar_write=True)
 tree=ast.parse('\n'.join(generator.translate_statement(s) for s in [*scalar,*vector]))
 for stmt in tree.body:
  if isinstance(stmt,ast.Assign) and stmt.targets[0].id=='v':
   class Indexed(ast.NodeTransformer):
    def visit_Name(self,node):return ast.copy_location(ast.Subscript(value=ast.Name(id=node.id,ctx=ast.Load()),slice=ast.Name(id='not_refractory',ctx=ast.Load()),ctx=node.ctx),node) if node.id in ('v','_v') else node
   stmt.targets[0]=Indexed().visit(stmt.targets[0]);stmt.value=Indexed().visit(stmt.value)
 return net,g,dt,bundle,compile(ast.fix_missing_locations(tree),'<independent masked generated stages>','exec'),noisy


def reference(bundle,g,stage,noisy,weights,initial=None,anchors=None):
 p=bundle.plan;z=np.array(bundle.initial_state if initial is None else initial,float);layout=bundle.provenance['neuron_state_layout'][g.name]
 vs=layout['v'];aa=layout['a'];gainbank=next(row['bank'] for row in bundle.provenance['bindings'] if row['object']==g.name and row['variables']==['gain'])
 v=z[vs].copy();a=z[aa].copy();last=np.array([0,-5000]);spikes=[];margins=[];before=[];hard_spikes=[]
 for tick in range(4):
  if anchors is not None and p['tbptt_window'] and tick and tick%p['tbptt_window']==0:
   v,a=(row.copy() for row in anchors['before'][tick])
  before.append((v.copy(),a.copy()));flag=tick-last>=2
  noise=np.array([normal(p['seed'],9,0,1,j,tick,0) for j in range(2)])
  env=dict(v=v.copy(),a=a.copy(),gain=np.array(weights[gainbank]),dt=.0002,ms=.001,curve=curve,
           not_refractory=flag.copy(),randn=lambda *args:noise.copy(),_vectorisation_idx=np.arange(2),_numpy=np,
           sqrt=np.sqrt,int=lambda value:np.asarray(value,dtype=np.int32))
  exec(stage,env);v=env['v'];a=env['a'];margin=v-.5;hard=(margin>0)&flag;gate=hard.astype(float)
  if anchors is not None:
   old=anchors['margins'][tick];hard=anchors['hard_spikes'][tick]
   gate=hard.astype(float)+flag*p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(old))**2*(margin-old)
  margins.append(margin.copy());hard_spikes.append(hard.copy());spikes.append(gate.copy());last[hard]=tick;v-=.5*gate
 logits=np.array(spikes).mean(0)*p['logit_scale'];loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
 return loss,v,a,np.array(spikes),dict(before=before,margins=margins,hard_spikes=hard_spikes)


@pytest.mark.parametrize('kind',['drift','hidden'])
@pytest.mark.parametrize('method',['euler','rk2','rk4','heun','milstein'])
@pytest.mark.parametrize('discard',[False,True])
@pytest.mark.parametrize('ranks',[None,2])
@pytest.mark.parametrize('window',[None,2])
def test_guarded_generated_effects_original_and_all_vjps(engine,kind,method,discard,ranks,window,monkeypatch):
 mpi(ranks);net,g,dt,bundle,stage,noisy=model(kind,method,discard,ranks,window,engine)
 out=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).gradients(np.zeros((1,4,1)),[0],**({'noise_sequence':9} if noisy else {}))
 loss,v,a,spikes,anchors=reference(bundle,g,stage,noisy,bundle.weights)
 layout=bundle.provenance['neuron_state_layout'][g.name]
 np.testing.assert_allclose(np.asarray(out['final_state'])[0,layout['v']],v,rtol=7e-5,atol=7e-6)
 np.testing.assert_allclose(np.asarray(out['final_state'])[0,layout['a']],a,rtol=7e-5,atol=7e-6)
 np.testing.assert_array_equal(np.asarray(out['spikes'])[0,:,1:],spikes);assert out['loss']==pytest.approx(loss,abs=7e-6)
 for bank,row in enumerate(bundle.weights):
  for index in range(len(row)):
   hi=copy.deepcopy(bundle.weights);lo=copy.deepcopy(bundle.weights);hi[bank][index]+=1e-6;lo[bank][index]-=1e-6
   fd=(reference(bundle,g,stage,noisy,hi,anchors=anchors)[0]-reference(bundle,g,stage,noisy,lo,anchors=anchors)[0])/2e-6
   assert out['gradients'][bank][index]==pytest.approx(fd,rel=9e-4,abs=9e-6)
 for index in range(len(bundle.initial_state)):
  if index in bundle.plan['dynamic'].get('binary_states',[]) or index in bundle.plan['dynamic'].get('integer_states',[]) or bundle.plan['dynamic']['detached'][index]:
   assert out['initial_state_gradients'][0][index]==0.;continue
  hi=np.array(bundle.initial_state);lo=hi.copy();hi[index]+=1e-6;lo[index]-=1e-6
  fd=(reference(bundle,g,stage,noisy,bundle.weights,hi,anchors)[0]-reference(bundle,g,stage,noisy,bundle.weights,lo,anchors)[0])/2e-6
  assert out['initial_state_gradients'][0][index]==pytest.approx(fd,rel=9e-4,abs=9e-6)
 if noisy:
  values=iter(np.array([normal(bundle.plan['seed'],9,0,1,j,tick,0) for j in range(2)]) for tick in range(4))
  monkeypatch.setattr(np.random,'randn',lambda n:next(values).copy())
 net.run(4*dt,namespace={})
 np.testing.assert_allclose(g.v[:],v,rtol=7e-5,atol=7e-6);np.testing.assert_allclose(g.a[:],a,rtol=7e-5,atol=7e-6)
 assert (out['gpu_dispatches']>0)==(engine!='cpu')
