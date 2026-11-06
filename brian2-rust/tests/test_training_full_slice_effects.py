"""Python ndarray full-view type, alias and native random callback behavior."""
import ast
import copy
import textwrap
import numpy as np
import brian2 as b
import pytest
from brian2_rust import lower_brian_dynamic_training,NativeLIFTrainer
from brian2_rust.training_effects import Effects,lower_state_effect_function,compile_state_effect_transform
from brian2_rust.training_callback_noise import lift_callback_noise
from test_training_integer_ir import engine
from test_training_poisson_zero_vjp import mpi
from test_training_stochastic import normal
from test_native_training import RUNNER


def plain(x):
    x[:]*=.8
    return x


def local(x):
    y=x
    y[:]*=.8
    return y


def noisy(x,mean,spread):
    x[:]*=.8
    draw=np.random.normal(mean,spread)
    return x+.1*draw


def noisy_local(x,mean,spread):
    y=x
    y[:]*=.8
    draw=np.random.normal(mean,spread)
    return y+.1*draw


def reserved(x):
    y=_b2_full_slice(x)
    y*=.8
    return y


@pytest.mark.parametrize('function',[plain,local])
@pytest.mark.parametrize('kind',['scalar','zero_dim','readonly','array'])
def test_full_slice_python_type_matches_effect_compilation(function,kind):
    value=.473 if kind=='scalar' else np.array(.473) if kind=='zero_dim' else np.array([.473,.719])
    if kind=='readonly':value.flags.writeable=False
    descriptor=lower_state_effect_function(function)
    def compile():
        executor=Effects({'v':0},{'f':descriptor},array_states=() if kind=='scalar' else {'v'},writable_states=() if kind in ('scalar','readonly') else {'v'},state_types={0:'float'})
        if kind=='zero_dim':executor.environment['v'].zero_dim=True
        return executor.expression(ast.parse('f(v)',mode='eval').body,executor.environment)
    if kind!='array':
        with pytest.raises((TypeError,IndexError,ValueError)):function(value)
        with pytest.raises(ValueError,match='one-dimensional array|readonly'):compile()
    else:
        np.testing.assert_allclose(function(value),np.array([.473,.719])*.8)
        result=compile_state_effect_transform('v=f(v)',states={'v':0},parameters={'f':descriptor},array_states={'v'},writable_states={'v'},state_types={0:'float'})
        assert result['programs']==[[dict(op='state',index=0),dict(op='constant',value=.8),dict(op='mul',left=0,right=1)]]


@pytest.mark.parametrize('function',[noisy,noisy_local])
def test_full_slice_lifted_python_keeps_array_and_scalar_errors(function,monkeypatch):
    descriptor=lower_state_effect_function(function);code='out=f(x,mean,spread)';text,helpers,sites,_=lift_callback_noise(code,{'f':descriptor})
    def whole(value):
        if not isinstance(value,np.ndarray) or value.ndim!=1:raise ValueError('full array slice requires a one-dimensional array')
        return value
    def execute(lifted,value):
        scope=dict(x=value,mean=.043,spread=.13,f=function,_b2_full_slice=whole)
        for name,helper in helpers.items():
            context=dict(_b2_scalar_noise=lambda first,second,noise:noise,_b2_shape_noise=lambda first,second,noise:noise[0])
            exec('def helper('+','.join(helper.arguments)+'):\n'+textwrap.indent(helper.body,'    '),context);scope[name]=context['helper']
        for name in sites:scope[name]=lambda *args:np.array([.7,.99])
        exec(text if lifted else code,scope);return scope['out'],scope['x']
    monkeypatch.setattr(np.random,'normal',lambda *args:.043+.13*.7)
    expected=execute(False,np.array([.473,.719]));actual=execute(True,np.array([.473,.719]))
    for a,c in zip(expected,actual):np.testing.assert_allclose(a,c)
    for scalar in (.473,np.array(.473)):
        for lifted in (False,True):
            with pytest.raises((ValueError,IndexError,TypeError)):execute(lifted,scalar)


def test_full_slice_reserved_names_rejected():
    with pytest.raises(ValueError,match='reserved'):lower_state_effect_function(reserved)
    with pytest.raises(ValueError,match='reserved'):lift_callback_noise('out=_b2_full_slice(v)',{})


@pytest.mark.parametrize('phase',['threshold','update','regular'])
@pytest.mark.parametrize('function',[noisy,noisy_local])
def test_full_slice_native_original_restore(engine,phase,function,monkeypatch,tmp_path):
    mpi(2);b.set_device('runtime');b.start_scope();b.prefs.codegen.target='numpy';dt=.2*b.ms
    source=b.SpikeGeneratorGroup(1,[],np.array([])*dt,dt=dt)
    hidden=b.NeuronGroup(2,'dv/dt=0/second:1',threshold='v>100',reset='v=0',method='euler',dt=dt)
    expression='gain+.1*f(v,mean,spread)' if phase=='update' else 'gain'
    threshold='f(v,mean,spread)>.5' if phase=='threshold' else 'v>.5'
    g=b.NeuronGroup(2,'dv/dt=('+expression+')/ms:1\nmean:1 (constant,shared)\nspread:1 (constant,shared)\ngain:1 (constant)',threshold=threshold,reset='v-=.5',method='euler',dt=dt);g.v=[.473,.719];g.mean=.043;g.spread=.13;g.gain=[.217,.319]
    g.namespace['f']=b.Function(function,arg_units=[1,1,1],return_unit=1,stateless=False)
    runner=g.run_regularly('v=f(v,mean,spread)',when='start',dt=dt) if phase=='regular' else None
    net=b.Network(source,hidden,g);bundle=lower_brian_dynamic_training(net,input_group=source,layers=[hidden,g],backend=engine,mpi_ranks=2,detach_reset=False,trainable_neuron_parameters={g.name:['gain','mean','spread']})
    monitor=b.SpikeMonitor(g);net.add(monitor);p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights);trainer=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER)
    domain=bundle.provenance['regular_runner_layout'][runner.name]['noise_domain'] if runner is not None else 1
    calls=[]
    for tick in range(4):
        def sample(*args,**kwargs):
            assert kwargs.get('size') is None;calls.append(tick);return float(g.mean[:])+float(g.spread[:])*normal(p['seed'],9,0,domain,0,tick,0)
        monkeypatch.setattr(np.random,'normal',sample)
        out=trainer.step(np.zeros((1,1,1)),[0],**({'initial':'carry'} if tick else {'noise_sequence':9}));net.run(dt,namespace={})
        cells=bundle.provenance['neuron_state_layout'][g.name]['v'];np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],np.asarray(g.v[:]),rtol=8e-5,atol=8e-6)
        recorded=np.asarray(monitor.i)[np.isclose(np.asarray(monitor.t/b.second),tick*float(dt),rtol=0.,atol=1e-12)];np.testing.assert_array_equal(np.asarray(out['spikes'])[0,-1,2:],np.isin(np.arange(2),recorded).astype(float))
        path=tmp_path/str(tick);trainer.store(path);trainer=NativeLIFTrainer(p,runner=RUNNER);trainer.restore(path)
    assert len(calls)==4


@pytest.mark.parametrize('function',[noisy,noisy_local])
def test_full_slice_shared_scalar_conversion_matches_brian_failure(function):
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='numpy';dt=.2*b.ms
    source=b.SpikeGeneratorGroup(1,[],np.array([])*dt,dt=dt)
    hidden=b.NeuronGroup(1,'dv/dt=0/second:1',threshold='v>100',reset='v=0',method='euler',dt=dt)
    g=b.NeuronGroup(2,'dv/dt=0/second:1\nmean:1 (constant,shared)\nspread:1 (constant,shared)',threshold='v>.5',reset='v=0',method='euler',dt=dt);g.mean=.043;g.spread=.13
    g.namespace['f']=b.Function(function,arg_units=[1,1,1],return_unit=1,stateless=False);g.run_regularly('v=f(mean,mean,spread)',when='start',dt=dt);net=b.Network(source,hidden,g)
    with pytest.raises(ValueError,match='one-dimensional array'):lower_brian_dynamic_training(net,input_group=source,layers=[hidden,g])
    with pytest.raises(b.core.base.BrianObjectException) as raised:net.run(dt,namespace={})
    assert isinstance(raised.value.__cause__,(ValueError,IndexError,TypeError))


@pytest.mark.parametrize('phase',['threshold','update','regular'])
@pytest.mark.parametrize('function',[noisy,noisy_local])
def test_full_slice_native_invalid_spread_atomic(engine,phase,function,tmp_path):
    mpi(2);b.set_device('runtime');b.start_scope();b.prefs.codegen.target='numpy';dt=.2*b.ms
    source=b.SpikeGeneratorGroup(1,[],np.array([])*dt,dt=dt);hidden=b.NeuronGroup(2,'dv/dt=0/second:1',threshold='v>100',reset='v=0',method='euler',dt=dt)
    expression='gain+.1*f(v,mean,spread)' if phase=='update' else 'gain';threshold='f(v,mean,spread)>.5' if phase=='threshold' else 'v>.5'
    g=b.NeuronGroup(2,'dv/dt=('+expression+')/ms:1\nmean:1 (constant,shared)\nspread:1 (constant,shared)\ngain:1 (constant)',threshold=threshold,reset='v-=.5',method='euler',dt=dt);g.v=[.473,.719];g.mean=.043;g.spread=.13;g.gain=[.217,.319]
    g.namespace['f']=b.Function(function,arg_units=[1,1,1],return_unit=1,stateless=False)
    if phase=='regular':g.run_regularly('v=f(v,mean,spread)',when='start',dt=dt)
    bundle=lower_brian_dynamic_training(b.Network(source,hidden,g),input_group=source,layers=[hidden,g],backend=engine,mpi_ranks=2,detach_reset=False,trainable_neuron_parameters={g.name:['gain','mean','spread']});p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights);trainer=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER);x=np.zeros((1,1,1));trainer.step(x,[0],noise_sequence=9)
    path=tmp_path/'prefix';trainer.store(path);trainer=NativeLIFTrainer(p,runner=RUNNER);trainer.restore(path)
    row=next(row for row in bundle.provenance['bindings'] if row['object']==g.name and 'spread' in row['variables']);trainer.state['weights'][row['bank']][row['variables'].index('spread')]=-1.
    before=copy.deepcopy((trainer.state,trainer.neuron_state,trainer.clock_state,trainer.poisson_state,trainer.elapsed_ticks,trainer.next_noise_sequence))
    for _ in range(2):
        with pytest.raises(ValueError):trainer.step(x,[0],initial='carry')
        assert (trainer.state,trainer.neuron_state,trainer.clock_state,trainer.poisson_state,trainer.elapsed_ticks,trainer.next_noise_sequence)==before
