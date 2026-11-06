"""Warm default NumPy bindings preserve actual source and SI namespaces."""
import brian2 as b
import numpy as np
import pytest

from brian2_rust import NativeLIFTrainer, lower_brian_dynamic_training
from brian2_rust.training_functions import lower_pure_function
from test_native_training import RUNNER
from test_training_integer_ir import engine

UNIT_GAIN = 2*b.mV


@b.check_units(x=1, result=b.volt)
def unit_gain(x):
    return UNIT_GAIN*x


@b.check_units(x=1, result=1)
def simple(x):
    return .7*x


def warm(function):
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='numpy'
    group=b.NeuronGroup(1,'v:1',namespace={'curve':function},dt=b.ms)
    group.run_regularly('v=curve(v)',dt=b.ms)
    network=b.Network(group);network.run(0*b.ms,namespace={})
    return function.implementations['numpy']


@pytest.mark.parametrize('discard',[False,True])
def test_default_binding_uses_actual_unit_namespace_and_native_physics(engine,discard,monkeypatch):
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='numpy'
    b.prefs.codegen.runtime.numpy.discard_units=discard
    monkeypatch.setitem(globals(),'UNIT_GAIN',2*b.mV)
    function=b.Function(unit_gain)
    source=b.SpikeGeneratorGroup(1,[],np.array([])*b.second,dt=b.ms)
    groups=[b.NeuronGroup(1,'dv/dt=0*volt/second:volt',threshold='v>100*mV',
        reset='v=0*volt',method='euler',dt=b.ms,namespace={'curve':function}) for _ in range(2)]
    for group in groups:group.run_regularly('v=curve(1.)',dt=b.ms)
    network=b.Network(source,*groups);network.run(0*b.ms,namespace={})
    # discard=True copied the namespace when NumPy materialized its default.
    # The unit-preserving wrapper still reads the original function globals.
    monkeypatch.setitem(globals(),'UNIT_GAIN',4*b.mV)
    expected=.002 if discard else .004
    descriptor=lower_pure_function(function)
    assert dict(descriptor.parameters)['UNIT_GAIN']==pytest.approx(expected)
    bundle=lower_brian_dynamic_training(network,input_group=source,layers=groups,backend=engine)
    result=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).evaluate(np.zeros((1,1,1)),[0])
    network.run(b.ms,namespace={})
    for group in groups:
        np.testing.assert_allclose(group.v[:]/b.volt,expected,atol=1e-12)
        slots=bundle.provenance['neuron_state_layout'][group.name]['v']
        np.testing.assert_allclose(np.asarray(result['final_state'])[0,slots],np.asarray(group.v[:]),atol=2e-9)
    assert (result['gpu_dispatches']>0)==(engine!='cpu')


@pytest.mark.parametrize('discard',[False,True])
def test_explicit_numpy_registration_still_refuses(discard):
    function=b.Function(simple)
    function.implementations.add_numpy_implementation(function.pyfunc,discard_units=discard)
    with pytest.raises(ValueError,match='target overrides'):lower_pure_function(function)


@pytest.mark.parametrize('change',['code','closure','availability','dependencies'])
def test_modified_automatic_binding_refuses_without_running_hooks(change):
    b.prefs.codegen.runtime.numpy.discard_units=False
    function=b.Function(simple);impl=warm(function);calls=[]
    if change=='code':impl._code=lambda x:99.*x
    elif change=='closure':
        index=impl._code.__code__.co_freevars.index('orig_func')
        impl._code.__closure__[index].cell_contents=lambda x:99.*x
    elif change=='availability':
        def hook():calls.append('availability')
        impl.availability_check=hook
    else:
        class Opaque:
            def __bool__(self):calls.append('truth');return True
            def __len__(self):calls.append('length');return 1
        impl.dependencies=Opaque()
    with pytest.raises(ValueError):lower_pure_function(function)
    assert calls==[]


@pytest.mark.parametrize('field',['stateless','auto_vectorise'])
def test_opaque_function_flags_refuse_without_truth_calls(field):
    calls=[]
    class Opaque:
        def __bool__(self):calls.append(field);return True
    function=b.Function(simple)
    setattr(function,field,Opaque())
    with pytest.raises(ValueError):lower_pure_function(function)
    assert calls==[]
