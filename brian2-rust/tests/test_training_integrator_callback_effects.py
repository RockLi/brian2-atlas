"""Generated Brian SDE array callbacks, original forwards and coupled VJPs."""
import copy
import ast
import brian2 as b
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer,lower_brian_dynamic_training
from test_training_callback_effects import curve, invalid_curve
from test_training_integer_ir import engine
from test_training_poisson_zero_vjp import mpi
from test_training_stochastic import normal
from test_native_training import RUNNER
from brian2_rust.training_brian import _INTEGRATORS

CODES={'drift':'curve(v)/ms+gain*xi/ms**.5','hidden':'curve(a)/ms+gain*xi/ms**.5',
       'coupled':'curve(v)/ms+gain*xi_shared/ms**.5'}


def equations(kind,noisy=True,method="euler"):
 expression=CODES[kind] if noisy else CODES[kind].split('+gain')[0]
 auxiliary=('da/dt=curve(v)/ms-.2*a/ms'+('+gain*xi_shared/ms**.5' if noisy else '')+':1'
            if kind=='coupled' else 'a:1')
 if kind=='coupled' and method=='milstein':
  expression=expression.replace('xi_shared','xi_v');auxiliary=auxiliary.replace('xi_shared','xi_a')
 return 'dv/dt='+expression+':1\n'+auxiliary+'\ngain:1 (constant)'


def model(kind,engine,ranks,window,method='euler'):
 b.set_device('runtime');b.start_scope();b.prefs.codegen.target='numpy';b.prefs.codegen.runtime.numpy.discard_units=True
 dt=.2*b.ms;source=b.SpikeGeneratorGroup(1,[],np.array([])*b.second,dt=dt)
 f=b.Function(curve,arg_units=[1],arg_names=['x'],return_unit=1,stateless=False)
 groups=[b.NeuronGroup(2,equations(kind,method=method),threshold='v>.5',reset='v-=.5',
                      method=method,dt=dt,namespace={'curve':f}) for _ in range(2)]
 runners=[g.state_updater for g in groups]
 for group,initial,gain in zip(groups,[[.6,.1],[.7,.3]],[[.2,.3],[.4,.5]]):
  group.v=initial;group.a=[.4,.6];group.gain=gain
 net=b.Network(source,*groups)
 bundle=lower_brian_dynamic_training(net,input_group=source,layers=groups,backend=engine,mpi_ranks=ranks,
          detach_reset=False,tbptt_window=window,seed=7123,trainable_neuron_parameters={g.name:['gain'] for g in groups})
 return net,source,groups,runners,dt,bundle


def reference(bundle,groups,runners,kind,w,initial=None,anchors=None):
 p=bundle.plan;z=np.array(bundle.initial_state if initial is None else initial,float);layout=bundle.provenance['neuron_state_layout']
 slots=[layout[g.name]['v'] for g in groups];spikes=[];margins=[];old=[]
 banks=[next(q['bank'] for q in bundle.provenance['bindings'] if q['object']==g.name and q['variables']==['gain']) for g in groups]
 domains=list(range(len(groups)))
 stage_sources=[_INTEGRATORS[g.state_updater.method_choice](g.equations,variables=g.variables) for g in groups]
 stage_codes=[compile(code,'<independent Brian stages>','exec') for code in stage_sources]
 noise_orders=[[sorted(g.equations.stochastic_variables).index(stmt.targets[0].id) for stmt in ast.parse(code).body
                if isinstance(stmt,ast.Assign) and stmt.targets[0].id in g.equations.stochastic_variables]
               for g,code in zip(groups,stage_sources)]
 for tick in range(4):
  if anchors is not None and p['tbptt_window'] and tick and tick%p['tbptt_window']==0:z=anchors[2][tick].copy()
  old.append(z.copy())
  for group,runner,indices,bank,domain,stage_code,noise_order in zip(groups,runners,slots,banks,domains,stage_codes,noise_orders):
   value=z[indices].copy();noises=iter([np.array([normal(p['seed'],9,0,domain,j,tick,stream) for j in range(2)]) for stream in noise_order])
   auxiliary=z[layout[group.name]['a']].copy()
   namespace=dict(v=value,a=auxiliary,gain=np.array(w[bank]),dt=.0002,ms=.001,
                  curve=curve,randn=lambda:next(noises).copy(),sqrt=np.sqrt)
   exec(stage_code,namespace)
   value=namespace['v'];z[layout[group.name]['a']]=namespace['a']
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
@pytest.mark.parametrize('method',['euler','heun','milstein'])
def test_generated_integrator_effect_noise_bank_initial_and_tbptt_vjps(engine,kind,ranks,window,method):
 mpi(ranks);net,source,groups,runners,dt,bundle=model(kind,engine,ranks,window,method)
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


@pytest.mark.parametrize('method,noisy',[('euler',False),('euler',True),('rk2',False),('rk4',False),('heun',True),('milstein',True)])
@pytest.mark.parametrize('discard',[False,True])
@pytest.mark.parametrize('kind',list(CODES))
def test_integrator_effect_matches_actual_brian(engine,method,noisy,discard,kind,monkeypatch):
 b.set_device('runtime');b.start_scope();b.prefs.codegen.target='numpy'
 b.prefs.codegen.runtime.numpy.discard_units=discard
 dt=.2*b.ms;source=b.SpikeGeneratorGroup(1,[],np.array([])*b.second,dt=dt)
 f=b.Function(curve,arg_units=[1],arg_names=['x'],return_unit=1,stateless=False)
 groups=[b.NeuronGroup(2,equations(kind,noisy,method),threshold='v>100',reset='v=0',
                      method=method,dt=dt,namespace={'curve':f}) for _ in range(2)]
 for group in groups:group.v=[1.,2.];group.a=[.4,.6];group.gain=[.2,.3]
 net=b.Network(source,*groups)
 bundle=lower_brian_dynamic_training(net,input_group=source,layers=groups,backend=engine,seed=7123)
 result=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).evaluate(np.zeros((1,1,1)),[0],**({'noise_sequence':9} if noisy else {}))
 draws=[]
 for obj in net.sorted_objects:
  for domain,group in enumerate(groups):
   if obj is group.state_updater:
    stage_code=_INTEGRATORS[method](group.equations,variables=group.variables)
    for stmt in ast.parse(stage_code).body:
     if isinstance(stmt,ast.Assign) and stmt.targets[0].id in group.equations.stochastic_variables:
      stream=sorted(group.equations.stochastic_variables).index(stmt.targets[0].id)
      draws.append(np.array([normal(bundle.plan['seed'],9,0,domain,j,0,stream) for j in range(2)]))
 called=[]
 def randn(*shape):
  assert shape==(2,) and len(called)<len(draws)
  value=draws[len(called)].copy();called.append(value);return value
 if noisy:monkeypatch.setattr(np.random,'randn',randn)
 net.run(dt,namespace={})
 if noisy:assert len(called)==len(draws)
 for group in groups:
  for name in ('v','a'):
   slots=bundle.provenance['neuron_state_layout'][group.name][name]
   np.testing.assert_allclose(np.asarray(result['final_state'])[0,slots],group.variables[name].get_value(),rtol=4e-5,atol=3e-6)
 assert (result['gpu_dispatches']>0)==(engine!='cpu')


def test_integrator_discarded_error_does_not_commit_noise_or_carry(engine):
 net,source,groups,_,_,_=model('drift',engine,None,None)
 function=b.Function(invalid_curve,arg_units=[1],arg_names=['x'],return_unit=1,stateless=False)
 for group in groups:group.namespace['curve']=function
 bundle=lower_brian_dynamic_training(net,input_group=source,layers=groups,backend=engine,seed=7123)
 t=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER);before=copy.deepcopy(t.state)
 with pytest.raises(ValueError,match='[Nn]on.?finite|domain'):
  t.step(np.zeros((1,1,1)),[0],noise_sequence=9)
 assert t.state==before and t.neuron_state is None and t.clock_tick==0 and t.next_noise_sequence==0


@pytest.mark.parametrize('draw',['randn','rand'])
@pytest.mark.parametrize('discard',[False,True])
@pytest.mark.parametrize('ranks',[None,2])
def test_integrator_callback_random_argument_is_a_mutable_array(engine,draw,discard,ranks,monkeypatch):
 from test_training_uniform import uniform
 mpi(ranks);b.set_device('runtime');b.start_scope();b.prefs.codegen.target='numpy'
 b.prefs.codegen.runtime.numpy.discard_units=discard;dt=.2*b.ms
 source=b.SpikeGeneratorGroup(1,[],np.array([])*b.second,dt=dt)
 function=b.Function(curve,arg_units=[1],arg_names=['x'],return_unit=1,stateless=False)
 groups=[b.NeuronGroup(2,'dv/dt=gain*curve('+draw+'())/ms:1\ngain:1 (constant)',
         method='euler',threshold='v>.5',reset='v-=.5',dt=dt,namespace={'curve':function}) for _ in range(2)]
 for group in groups:group.v=[.7,.1];group.gain=[.2,.3]
 bundle=lower_brian_dynamic_training(b.Network(source,*groups),input_group=source,layers=groups,
   seed=7123,backend=engine,mpi_ranks=ranks,detach_reset=False,
   trainable_neuron_parameters={g.name:['gain'] for g in groups})
 result=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).gradients(np.zeros((1,1,1)),[0],noise_sequence=9)
 sampler=normal if draw=='randn' else uniform
 values=[np.array([sampler(7123,9,0,domain,j,0,0) for j in range(2)]) for domain in range(2)]
 voltages=np.concatenate([np.array([.7,.1])+.2*np.array([.2,.3])*.8*value for value in values])
 margin=voltages-.5;gate=(margin>0).astype(float)
 expected=np.array(bundle.initial_state,float)
 slots=[bundle.provenance['neuron_state_layout'][g.name]['v'] for g in groups]
 expected[slots[0]+slots[1]]=voltages-.5*gate
 np.testing.assert_allclose(result['final_state'][0],expected,rtol=3e-5,atol=3e-6)
 # Independently differentiate the one-tick surrogate loss at the hard anchors.
 def loss(perturbed):
  soft=gate+bundle.plan['surrogate']['scale']/(1+bundle.plan['surrogate']['slope']*abs(margin))**2*(perturbed-voltages)
  logits=soft[2:]*bundle.plan['logit_scale']
  return np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
 banks=[next(row['bank'] for row in bundle.provenance['bindings'] if row['object']==g.name and row['variables']==['gain']) for g in groups]
 for domain,bank in enumerate(banks):
  for j in range(2):
   delta=np.zeros(4);delta[2*domain+j]=.2*.8*values[domain][j]*1e-6
   fd=(loss(voltages+delta)-loss(voltages-delta))/2e-6
   assert result['gradients'][bank][j]==pytest.approx(fd,rel=5e-4,abs=5e-6)
 for j,slot in enumerate(slots[0]+slots[1]):
  delta=np.zeros(4);delta[j]=1e-6
  fd=(loss(voltages+delta)-loss(voltages-delta))/2e-6
  assert result['initial_state_gradients'][0][slot]==pytest.approx(fd,rel=5e-4,abs=5e-6)
 # Actual NumPy callback must see an array (saved aliases x after x *= .8).
 net=b.Network(source,*groups);ordered=[]
 for obj in net.sorted_objects:
  for domain,group in enumerate(groups):
   if obj is group.state_updater:ordered.append(values[domain])
 called=[]
 def replay(*shape):
  assert shape==(2,) and len(called)<2
  value=ordered[len(called)].copy();called.append(value);return value
 monkeypatch.setattr(np.random,draw,replay);net.run(dt,namespace={});assert len(called)==2
 for group,indices in zip(groups,slots):
  np.testing.assert_allclose(np.array(result['final_state'])[0,indices],group.v[:],rtol=3e-5,atol=3e-6)
 assert (result['gpu_dispatches']>0)==(engine!='cpu')
