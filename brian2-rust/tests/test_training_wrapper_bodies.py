"""Numeric decorators execute their real bodies, including fixed call packets."""
from functools import wraps
import numpy as np
import pytest

from brian2_rust.training_functions import lower_pure_function
from test_training_integer_ir import engine
import test_training_pure_functions as callbacks
import test_training_static_control_sde as static_sde
import test_training_dynamic_static_sde as dynamic_sde


def scale_fixed(f):
    @wraps(f)
    def wrapped(x):
        return .5*f(x)
    return wrapped


def scale_forwarded(f):
    @wraps(f)
    def wrapped(*args,**kwargs):
        return .5*f(*args,**kwargs)
    return wrapped


def scale_packet(f):
    @wraps(f)
    def wrapped(*args,**kwargs):
        factor = kwargs.get('factor',.5)
        if kwargs:
            return np.sqrt(-1.)
        if len(args) != 1:
            return np.sqrt(-1.)
        return factor*f(args[0])
    return wrapped


WRAPPERS=[scale_fixed,scale_forwarded,scale_packet]


def diffusion_base(x):return 2.+2.*x+.4*x*x


def decay_base(x):return 1.6*x+.02*np.sin(x)


def metadata(function):
    # Public Brian Function metadata, without copying a unit wrapper's
    # _orig_func shortcut onto a numeric decorator.
    function._arg_units=[1];function._return_unit=1
    function._arg_names=['x'];function._returns_bool=False
    return function


@pytest.mark.parametrize('wrapper',WRAPPERS)
@pytest.mark.parametrize('dynamic',[False,True])
@pytest.mark.parametrize('method,shared',[('heun',False),('heun',True),('heun','mixed'),('milstein',False)])
@pytest.mark.parametrize('ranks',[None,2,8])
@pytest.mark.parametrize('detach,window',[(False,None),(True,None),(False,3)])
def test_wrapped_sde_original_brian_and_all_bank_initial_vjps(
        engine,wrapper,dynamic,method,shared,ranks,detach,window,monkeypatch):
    monkeypatch.setattr(static_sde,'diffusion',metadata(wrapper(diffusion_base)))
    verify=(dynamic_sde.test_dynamic_sde_callback_physics_all_banks_and_initial_vjps
            if dynamic else static_sde.test_sde_static_callback_physics_all_bank_initial_and_tbptt_vjps)
    verify(engine,ranks,method,shared,detach,window)


@pytest.mark.parametrize('wrapper',WRAPPERS)
@pytest.mark.parametrize('position',['pre','post','regular','synaptic_ode'])
def test_wrapped_dynamic_synapse_positions_match_original_brian(engine,wrapper,position,monkeypatch):
    monkeypatch.setattr(callbacks,'decay',metadata(wrapper(decay_base)))
    callbacks.test_dynamic_brian_callback_positions_and_forward(engine,position)


def unsafe_tuple_condition(f):
    @wraps(f)
    def wrapped(*args,**kwargs):
        if (np.sqrt(-1.),args[0]):
            return f(*args,**kwargs)
        return f(*args,**kwargs)
    return wrapped


def test_packet_specialization_does_not_erase_tuple_operand_execution():
    with pytest.raises(ValueError):lower_pure_function(unsafe_tuple_condition(decay_base))


def unsafe_tuple_selection(f):
    @wraps(f)
    def wrapped(*args,**kwargs):
        return f((np.sqrt(-1.),args[0])[1])
    return wrapped


def test_packet_specialization_does_not_erase_unselected_tuple_elements():
    with pytest.raises(ValueError):lower_pure_function(unsafe_tuple_selection(decay_base))


def bad_packet_index(f):
    @wraps(f)
    def wrapped(*args,**kwargs):
        return f(args[1])
    return wrapped


def test_packet_out_of_bounds_is_refused():
    with pytest.raises(ValueError,match='packet index'):
        lower_pure_function(bad_packet_index(decay_base))
