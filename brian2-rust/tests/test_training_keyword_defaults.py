"""Loaded keyword-only scalar defaults in real SDE and dynamic synapse AD."""
from functools import wraps
import math
import sys
import numpy as np
import pytest
import brian2 as b
from brian2_rust import compile_training_equation
from brian2_rust import NativeLIFTrainer, lower_brian_dynamic_training
from brian2_rust.training_functions import lower_pure_function
from test_training_integer_ir import engine
import test_training_wrapper_bodies as wrappers
import test_training_positional_only_callbacks as positional
import test_training_static_control_sde as static_sde
import test_training_dynamic_static_sde as dynamic_sde
import test_training_pure_functions as callbacks
from test_native_training import RUNNER


def keyword_diffusion(x, *, factor=.25):
    """The local scalar default can be rebound without mutating caller state."""
    factor *= 2
    return factor*(2.+2.*x+.4*x*x)


def keyword_decay(x, *, factor=.25):
    factor *= 2
    return factor*(3.2*x+.04*np.sin(x))


def loaded_diffusion(x, *, factor=.75):
    return factor*(2.+2.*x+.4*x*x)


def loaded_decay(x, *, factor=.75):
    return factor*(3.2*x+.04*np.sin(x))


def keyword_prefix(f):
    @wraps(f)
    def wrapped(x, /, *args, factor=.25, **kwargs):
        factor *= 2
        return factor*f(x,*args,**kwargs)
    return wrapped


def doubled_decay(x):
    return 3.2*x+.04*np.sin(x)


def callback(kind, position, monkeypatch):
    if kind=='fixed':return keyword_diffusion if position=='sde' else keyword_decay
    if kind=='wrapper':return keyword_prefix(wrappers.diffusion_base if position=='sde' else doubled_decay)
    function=loaded_diffusion if position=='sde' else loaded_decay
    monkeypatch.setattr(function,'__kwdefaults__',{'factor':.5})
    return function


@pytest.mark.parametrize('kind',['fixed','wrapper','loaded'])
@pytest.mark.parametrize('dynamic',[False,True])
@pytest.mark.parametrize('method,shared',[('heun',True),('milstein',False)])
@pytest.mark.parametrize('ranks,window',[(None,None),(2,3)])
def test_keyword_defaults_sde_brian_and_all_bank_initial_vjps(
        engine,kind,dynamic,method,shared,ranks,window,monkeypatch):
    monkeypatch.setattr(static_sde,'diffusion',wrappers.metadata(callback(kind,'sde',monkeypatch)))
    verify=(dynamic_sde.test_dynamic_sde_callback_physics_all_banks_and_initial_vjps
            if dynamic else static_sde.test_sde_static_callback_physics_all_bank_initial_and_tbptt_vjps)
    verify(engine,ranks,method,shared,False,window)


@pytest.mark.parametrize('kind',['fixed','wrapper','loaded'])
@pytest.mark.parametrize('position',['pre','post','regular','synaptic_ode'])
def test_keyword_defaults_dynamic_synapse_brian_forward(engine,kind,position,monkeypatch):
    monkeypatch.setattr(callbacks,'decay',wrappers.metadata(callback(kind,position,monkeypatch)))
    callbacks.test_dynamic_brian_callback_positions_and_forward(engine,position)


SOURCE_FACTOR=.25

def source_default(x, *, factor=SOURCE_FACTOR):
    return factor*x


def test_default_snapshot_uses_loaded_value_without_reevaluating_source(monkeypatch):
    monkeypatch.setattr(sys.modules[__name__],'SOURCE_FACTOR',.75)
    result=lower_pure_function(source_default)
    assert result.statements[0]==('factor','0.25')
    monkeypatch.setattr(source_default,'__kwdefaults__',{'factor':.5})
    updated=lower_pure_function(source_default)
    assert updated.statements[0]==('factor','0.5')
    assert result.statements[0]==('factor','0.25')


@pytest.mark.parametrize('value',[[],math.nan,math.inf,np.float32(.5),2**31])
def test_non_scalar_or_unrepresentable_keyword_defaults_refused(value,monkeypatch):
    monkeypatch.setattr(source_default,'__kwdefaults__',{'factor':value})
    with pytest.raises(ValueError,match='keyword-only'):
        lower_pure_function(source_default)


def required_default(x, *, factor):
    return factor*x


def test_required_keyword_only_parameter_is_refused():
    with pytest.raises(ValueError,match='keyword-only'):
        lower_pure_function(required_default)


def test_default_hooks_and_mapping_hooks_never_execute(monkeypatch):
    calls=[]
    class Trap:
        def __float__(self):calls.append('float');raise AssertionError
        def __bool__(self):calls.append('bool');raise AssertionError
    class Mapping(dict):
        def __getitem__(self,name):calls.append('lookup');raise AssertionError
    for defaults in ({'factor':Trap()},Mapping(factor=.5)):
        monkeypatch.setattr(source_default,'__kwdefaults__',defaults)
        with pytest.raises(ValueError,match='keyword-only'):
            lower_pure_function(source_default)
    assert calls==[]


def packet_default(f):
    @wraps(f)
    def wrapped(*args, _b2_wrapper_arg_0=9., **kwargs):
        return f(*args,**kwargs)
    return wrapped


two_arguments=positional.two_arguments


def explicit_unused_default(x,y):
    _b2_wrapper_arg_0=9.
    return two_arguments(x,y)


def test_default_name_cannot_overwrite_generated_packet_argument():
    function=lower_pure_function(packet_default(positional.two_arguments))
    assert function.arguments[0]!='_b2_wrapper_arg_0'
    assert function.statements[0]==('_b2_wrapper_arg_0','9.0')
    actual=compile_training_equation('f(v,2*v)',parameters={'f':function})
    explicit=compile_training_equation('f(v,2*v)',parameters={'f':lower_pure_function(explicit_unused_default)})
    assert actual==explicit


def forwarded_keyword_default(f):
    @wraps(f)
    def wrapped(*args,**kwargs):
        return f(*args,**kwargs)
    return wrapped


def test_raw_wrapper_infers_positional_arity_of_keyword_default_target():
    function=lower_pure_function(forwarded_keyword_default(source_default))
    assert len(function.arguments)==1
    assert compile_training_equation('f(v)',parameters={'f':function})[-1]['op']=='sequence'


@pytest.mark.parametrize('discard',[False,True])
def test_warm_default_binding_keeps_actual_keyword_defaults(engine,discard,monkeypatch):
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='numpy'
    monkeypatch.setitem(b.prefs,'codegen.runtime.numpy.discard_units',discard)
    monkeypatch.setattr(loaded_decay,'__kwdefaults__',{'factor':.5})
    function=b.Function(wrappers.metadata(loaded_decay))
    source=b.SpikeGeneratorGroup(1,[],np.array([])*b.second,dt=b.ms)
    groups=[b.NeuronGroup(1,'dv/dt=0/second:1',threshold='v>100',reset='v=0',dt=b.ms,method='euler',
                          namespace={'curve':function}) for _ in range(2)]
    for group,value in zip(groups,[.2,-.3]):
        group.v=value
        group.run_regularly('v=curve(v)',dt=b.ms)
    network=b.Network(source,*groups);network.run(0*b.ms,namespace={})
    # Unit-discarding NumPy owns the defaults copied when it was materialized;
    # the unit-preserving adapter continues to call the original function.
    monkeypatch.setattr(loaded_decay,'__kwdefaults__',{'factor':.75})
    factor=.5 if discard else .75
    descriptor=lower_pure_function(function)
    assert descriptor.statements[0]==('factor',str(factor))
    bundle=lower_brian_dynamic_training(network,input_group=source,layers=groups,backend=engine)
    result=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).evaluate(np.zeros((1,1,1)),[0])
    network.run(b.ms,namespace={})
    for group,value in zip(groups,[.2,-.3]):
        expected=factor*(3.2*value+.04*np.sin(value))
        np.testing.assert_allclose(np.asarray(group.v[:]),expected,atol=1e-12)
        slots=bundle.provenance['neuron_state_layout'][group.name]['v']
        np.testing.assert_allclose(np.asarray(result['final_state'])[0,slots],np.asarray(group.v[:]),atol=3e-6)
    assert (result['gpu_dispatches']>0)==(engine!='cpu')
