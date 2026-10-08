"""Pure callback lowering: native forward/VJP, lexical scope and refusal."""
import copy
import math
import importlib.util
import os

import brian2 as b
import numpy as np
import pytest

from brian2_rust import NativeLIFTrainer, PureFunction, compile_training_equation
from brian2_rust.training_brian import lower_brian_training, TrainingConversionError
from brian2_rust.training_functions import lower_pure_function
from brian2_rust.training_dynamic import compile_dynamic_transform
from brian2_rust.training_brian_dynamic import lower_brian_dynamic_training
from test_native_training import RUNNER
from test_native_training_equations import fixture_equation, oracle
from test_training_integer_ir import engine, model as integer_model, oracle as integer_oracle
from test_training_poisson_zero_vjp import mpi


@b.check_units(v=1, result=1)
def smooth(v):
    return .05*np.sin(v)+.02*np.cos(v)+.03*np.tanh(v)+.02*np.exp(-v*v)+.01*np.log(v*v+1)+.01*np.sqrt(v*v+1)+.002*v**2


def square(x):
    return x*x


def nested(x):
    return square(square(x))+math.sin(x)


def mutable(x):
    x[0] += 1
    return x


def random_body(x):
    return x+np.random.rand()


def recursive(x):
    return recursive(x)


@pytest.mark.parametrize('ranks', [None, 2, 8])
@pytest.mark.parametrize('reset', ['zero', 'subtract'])
def test_native_callback_all_parameter_and_initial_vjps(engine, ranks, reset):
    mpi(ranks)
    p,w,x,y,initial=fixture_equation(reset,False,None,engine)
    function=lower_pure_function(b.Function(smooth))
    p['equations']=[compile_training_equation('v+dt*(-v/tau+curve(v))+bias',
        parameters={'dt':.25,'tau':(4,2*l),'bias':(4,2*l+1),'curve':function}) for l in range(2)]
    if ranks is not None:p['mpi_ranks']=ranks
    result=NativeLIFTrainer(p,weights=w,runner=RUNNER).gradients(x,y,initial=initial)
    loss,spikes,final,anchors=oracle(p,w,x,y,initial)
    np.testing.assert_array_equal(result['spikes'],spikes)
    np.testing.assert_allclose(result['final_membrane'],final,rtol=2e-5,atol=3e-6)
    assert result['loss']==pytest.approx(loss,abs=3e-6)
    for bank,row in enumerate(w):
        for i in range(len(row)):
            plus=copy.deepcopy(w);minus=copy.deepcopy(w);plus[bank][i]+=1e-6;minus[bank][i]-=1e-6
            fd=(oracle(p,plus,x,y,initial,anchors)[0]-oracle(p,minus,x,y,initial,anchors)[0])/2e-6
            assert result['gradients'][bank][i]==pytest.approx(fd,abs=3e-6,rel=3e-4)
    for sample in range(len(initial)):
        for i in range(initial.shape[1]):
            plus=initial.copy();minus=initial.copy();plus[sample,i]+=1e-6;minus[sample,i]-=1e-6
            fd=(oracle(p,w,x,y,plus,anchors)[0]-oracle(p,w,x,y,minus,anchors)[0])/2e-6
            assert result['initial_gradients'][sample][i]==pytest.approx(fd,abs=3e-6,rel=3e-4)


def test_lexical_scope_nested_calls_and_single_argument_evaluation():
    function=lower_pure_function(nested)
    compiled=compile_training_equation('f(f(v))+f(2*v)',parameters={'f':function})
    # Every expansion owns a fresh memo and the formal v/x shadows caller v.
    explicit=compile_training_equation('(v**4+sin(v))**4+sin(v**4+sin(v))+(2*v)**4+sin(2*v)')
    assert sum(n['op']=='sin' for n in compiled)==3
    from brian2_rust.training_equations import NormalNoise
    draw=compile_training_equation('f(z)',parameters={'f':PureFunction(('x',),'x*x+x'),'z':NormalNoise(0)})
    assert sum(n['op']=='noise' for n in draw)==1
    assert draw[-1]['op']=='sequence' and explicit[-1]['op']=='add'
    assert compile_training_equation('f(v)',parameters={'f':PureFunction(('v',),'v+1')})[-1]['op']=='sequence'
    captured=compile_training_equation('f(v)',parameters={'f':PureFunction(('x',),'x+v',(('v',2.),))})
    assert captured[1]=={'op':'constant','value':2.}
    with pytest.raises(ValueError,match='unknown pure function'):
        compile_training_equation('f(v)',parameters={'f':PureFunction(('x',),'x+v')})


def test_closure_snapshot_is_detached_and_not_executed():
    gain=.125
    def curve(x):
        return gain*x+np.sin(x)
    function=lower_pure_function(curve);gain=.5
    assert dict(function.parameters)=={'gain':.125}
    # The callback itself is never invoked, even with a singular placeholder.
    def singular(x):
        return 1/x
    assert lower_pure_function(singular).expression=='1 / x'


@pytest.mark.parametrize('function', [mutable, random_body, recursive, lambda x:x+1])
def test_opaque_stateful_or_nonexpression_callbacks_are_refused(function):
    with pytest.raises(ValueError):lower_pure_function(function)


@pytest.mark.parametrize('function', [PureFunction(('x',),'(lambda y:y)(x)'),
    PureFunction(('x','y'),'x'),PureFunction(('x',),'x+t'),
    PureFunction(('x',),'x+rand()'),PureFunction(('x',),'x+a',(('a',np.inf),)),
    PureFunction(('x',),'x+x',(('x',2.),))])
def test_descriptor_bounds_and_lexical_capture_refusal(function):
    with pytest.raises(ValueError):
        compile_training_equation('f(v)',parameters={'f':function})


@pytest.mark.parametrize('arguments', [(), ('x','x'), ([],), ('bad-name',), tuple('a'+str(i) for i in range(17))])
def test_descriptor_argument_bounds(arguments):
    with pytest.raises(ValueError,match='distinct positional'):
        compile_training_equation('f(v)',parameters={'f':PureFunction(arguments,'v')})


def test_inline_arity_node_budget_and_recursion_bounds():
    function=PureFunction(('x',),'x*x+x')
    for expression in ('f()', 'f(v,v)', 'f(x=v)'):
        with pytest.raises(ValueError,match='exact arity'):
            compile_training_equation(expression,parameters={'f':function})
    large=PureFunction(('x',),'+'.join('x*x' for _ in range(40)))
    with pytest.raises(ValueError,match='128 nodes'):
        compile_training_equation('f(v)+f(v)',parameters={'f':large})
    cycle=PureFunction(('x',),'f(x)')
    object.__setattr__(cycle,'parameters',(('f',cycle),))
    with pytest.raises(ValueError,match='recursive'):
        compile_training_equation('f(v)',parameters={'f':cycle})


MILLIVOLT=b.mV


@b.check_units(x=b.volt,result=b.volt)
def unit_curve(x):
    return .02*x+.01*MILLIVOLT*np.sin(x/MILLIVOLT)


def test_scalar_unit_closure_snapshots_si_and_rejects_arrays():
    function=lower_pure_function(b.Function(unit_curve))
    assert dict(function.parameters)=={'MILLIVOLT':.001}
    array=np.array([1.,2.])*b.mV
    def bad(x):
        return x+array
    with pytest.raises(ValueError,match='real scalar'):lower_pure_function(bad)


def test_changed_source_cannot_replace_loaded_function(tmp_path):
    path=tmp_path/'callback.py';path.write_text('def curve(x):\n    return x+1\n')
    spec=importlib.util.spec_from_file_location('test_b2_pure_callback',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert lower_pure_function(module.curve).expression=='x + 1'
    modified=path.stat().st_mtime+10
    path.write_text('def curve(x):\n    return x+2\n');os.utime(path,(modified,modified))
    with pytest.raises(ValueError,match='differs from loaded'):
        lower_pure_function(module.curve)


def test_replaced_standard_math_attribute_is_not_assumed_native(monkeypatch):
    monkeypatch.setattr(np,'sin',lambda value:value+17)
    with pytest.raises(ValueError,match='unsupported pure function math attribute'):
        lower_pure_function(b.Function(smooth))


def test_opaque_wrapper_metadata_is_not_executed():
    seen=[]
    class Opaque:
        @property
        def __class__(self):
            seen.append(True)
            raise AssertionError('opaque callback class metadata executed')
        @property
        def __wrapped__(self):
            seen.append(True)
            raise AssertionError('opaque callback metadata executed')
    opaque=Opaque()
    def wrapper(x):
        return x
    wrapper.__wrapped__=opaque
    for value in (opaque,wrapper):
        with pytest.raises(ValueError,match='opaque'):
            lower_pure_function(value)
    assert seen==[]


@pytest.mark.parametrize('engine_name', ['cpu','metal'])
def test_unitful_brian_callback_forward(engine_name):
    if engine_name=='metal':
        import os
        if os.environ.get('B2_TEST_GPU')!='1':pytest.skip('actual metal hardware required')
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='numpy';dt=.25*b.ms
    source=b.SpikeGeneratorGroup(1,[0,0],np.array([0,2])*dt,dt=dt)
    groups=[b.NeuronGroup(2,'dv/dt=(-v+curve(v))/ms:volt\ntheta:volt (constant,shared)',threshold='v>theta',reset='v-=theta',
        method='euler',dt=dt,namespace={'curve':b.Function(unit_curve)}) for _ in range(2)]
    for g in groups:g.v=[.3,.7]*b.mV;g.theta=.5*b.mV
    syn=b.Synapses(source,groups[1],'w:volt',on_pre='v_post+=w');syn.connect();syn.w=[.4,.6]*b.mV
    net=b.Network(source,*groups,syn)
    bundle=lower_brian_training(net,input_group=source,layers=groups,backend=engine_name)
    result=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).evaluate(
        np.array([[[1.],[0.],[1.]]]),[0],initial=[bundle.initial_membrane])
    net.run(3*dt,namespace={})
    np.testing.assert_allclose(result['final_membrane'][0],np.concatenate([np.asarray(g.v[:]) for g in groups]),rtol=2e-5,atol=3e-8)


@pytest.mark.parametrize('condition', ['stateful','vectorise','override'])
def test_brian_semantic_overrides_are_refused(condition):
    function=b.Function(smooth,stateless=condition!='stateful',auto_vectorise=condition=='vectorise')
    if condition=='override':function.implementations.add_implementation('cpp','return 99;')
    with pytest.raises(ValueError,match='stateless non-vectorised'):
        lower_pure_function(function)


@pytest.mark.parametrize('engine_name', ['cpu', 'metal'])
def test_brian_custom_neuron_function_snapshot_and_forward(engine_name):
    if engine_name=='metal':
        import os
        if os.environ.get('B2_TEST_GPU')!='1':pytest.skip('actual metal hardware required')
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='numpy'
    dt=.25*b.ms;x=np.array([[1],[0],[1],[1],[0],[1]],float)
    source=b.SpikeGeneratorGroup(1,[0,0,0,0],np.array([0,2,3,5])*dt,dt=dt)
    groups=[b.NeuronGroup(2,'dv/dt=(-v+curve(v))/ms:1',threshold='v>.5',reset='v-=.5',
        method='euler',dt=dt,namespace={'curve':b.Function(smooth)}) for _ in range(2)]
    synapses=[]
    for src,dst in [(source,groups[0]),(groups[0],groups[1])]:
        syn=b.Synapses(src,dst,'w:1',on_pre='v_post+=w');syn.connect();syn.w=.4;synapses.append(syn)
    for g in groups:g.v=[.3,.7]
    net=b.Network(source,*groups,*synapses)
    before=[np.asarray(g.v[:]).copy() for g in groups]
    bundle=lower_brian_training(net,input_group=source,layers=groups,backend=engine_name)
    result=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).evaluate(x[None],[0],initial=[bundle.initial_membrane])
    for g,v in zip(groups,before):np.testing.assert_array_equal(g.v[:],v)
    monitors=[b.SpikeMonitor(g) for g in groups];net.add(*monitors);net.run(len(x)*dt,namespace={})
    spikes=np.zeros((len(x),4))
    for l,m in enumerate(monitors):
        ticks=np.rint(np.asarray(m.t/b.second)/float(dt/b.second)).astype(int)
        spikes[ticks,2*l+np.asarray(m.i)]=1
    np.testing.assert_array_equal(result['spikes'][0],spikes)
    np.testing.assert_allclose(result['final_membrane'][0],np.concatenate([np.asarray(g.v[:]) for g in groups]),rtol=2e-5,atol=3e-6)


@pytest.mark.parametrize('ranks', [None,2,8])
def test_dynamic_callback_with_noise_all_float_vjps(engine,ranks):
    from brian2_rust.training_equations import NeuronParameter, NormalNoise
    mpi(ranks);p,w,x=integer_model(noisy=True,engine=engine,ranks=ranks)
    function=PureFunction(('voltage','gate','gain','draw'),
        '3.2*scale(scale(voltage))+gain*gate+.05*gate*draw',
        (('scale',PureFunction(('voltage',),'.5*voltage')),))
    for action in p['dynamic']['actions']:
        if action.get('noise_streams')!=1:continue
        p['dynamic']['program_sets'][action['program_set']]=compile_dynamic_transform(
            'v=f(v,flag,gain,draw)',states={'v':0,'flag':1},
            parameters={'f':function,'gain':NeuronParameter(0),'draw':NormalNoise(0)})['programs']
    result=NativeLIFTrainer(p,weights=w,runner=RUNNER).gradients(x,[0],noise_sequence=7)
    loss,final,spikes,anchors=integer_oracle(p,w,noisy=True)
    np.testing.assert_array_equal(result['spikes'][0],spikes)
    np.testing.assert_allclose(result['final_state'][0],final,rtol=2e-5,atol=3e-6)
    assert result['loss']==pytest.approx(loss,abs=3e-6)
    for i in range(4):
        plus=copy.deepcopy(w);minus=copy.deepcopy(w);plus[0][i]+=1e-6;minus[0][i]-=1e-6
        fd=(integer_oracle(p,plus,noisy=True,anchors=anchors)[0]-integer_oracle(p,minus,noisy=True,anchors=anchors)[0])/2e-6
        assert result['gradients'][0][i]==pytest.approx(fd,abs=3e-6,rel=3e-4)
    assert result['gradients'][1]==[0.]*4


@b.check_units(x=1,result=1)
def decay(x):
    return .8*x+.01*np.sin(x)


@pytest.mark.parametrize('position', ['pre','post','regular','synaptic_ode'])
def test_dynamic_brian_callback_positions_and_forward(engine,position):
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='numpy';dt=.25*b.ms
    source=b.SpikeGeneratorGroup(1,[0,0,0],np.array([0,2,4])*dt,dt=dt)
    groups=[b.NeuronGroup(2,'dv/dt=-v/ms:1',threshold='v>.5',reset='v-=.5',
                         method='euler',dt=dt) for _ in range(2)]
    for g in groups:g.v=[.3,.8]
    equations='dw/dt=(-w+curve(w))/ms:1 (clock-driven)' if position=='synaptic_ode' else 'w:1'
    pre='w=curve(w)\nv_post+=w' if position=='pre' else 'v_post+=w'
    post='w=curve(w)' if position=='post' else None
    syn=b.Synapses(source,groups[1],equations,on_pre=pre,on_post=post,dt=dt,method='euler',namespace={'curve':b.Function(decay)})
    syn.connect();syn.w=[.4,.6]
    if position=='regular':groups[1].namespace['curve']=b.Function(decay);groups[1].run_regularly('v=curve(v)',dt=dt)
    net=b.Network(source,*groups,syn)
    before=np.asarray(syn.w[:]).copy()
    bundle=lower_brian_dynamic_training(net,input_group=source,layers=groups,backend=engine)
    np.testing.assert_array_equal(syn.w[:],before)
    result=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).evaluate(
        np.array([[[1.],[0.],[1.],[0.],[1.]]]),[0])
    monitors=[b.SpikeMonitor(g) for g in groups];net.add(*monitors);net.run(5*dt,namespace={})
    spikes=np.zeros((5,4))
    for l,m in enumerate(monitors):
        ticks=np.rint(np.asarray(m.t/b.second)/float(dt/b.second)).astype(int)
        spikes[ticks,2*l+np.asarray(m.i)]=1
    np.testing.assert_array_equal(result['spikes'][0],spikes)
    for g in groups:
        slots=bundle.provenance['neuron_state_layout'][g.name]['v']
        np.testing.assert_allclose(np.asarray(result['final_state'])[0,slots],g.v[:],rtol=2e-5,atol=3e-6)
    if position=='regular':
        np.testing.assert_array_equal(syn.w[:],before)
    else:
        slots=bundle.provenance['dynamic_state_layout'][syn.name]['w']
        np.testing.assert_allclose(np.asarray(result['final_state'])[0,slots],syn.w[:],rtol=2e-5,atol=3e-6)
