"""Positional-only callback signatures preserve numeric bodies and packet order."""
from functools import wraps
import numpy as np
import pytest

from brian2_rust.training_functions import lower_pure_function
from brian2_rust import compile_training_equation
from test_training_integer_ir import engine
import test_training_wrapper_bodies as wrappers
import test_training_static_control_sde as static_sde
import test_training_dynamic_static_sde as dynamic_sde
import test_training_pure_functions as callbacks


def positional_diffusion(x, /):
    return 1.+x+.2*x*x


def doubled_positional_diffusion(x, /):
    return 2.+2.*x+.4*x*x


def positional_decay(x, /):
    return 1.6*x+.02*np.sin(x)


def positional_prefix(f):
    @wraps(f)
    def wrapped(x, /, *args, **kwargs):
        if len(args):
            return np.sqrt(-1.)
        return .5*f(x, *args, **kwargs)
    return wrapped


@pytest.mark.parametrize('wrapped',[False,True])
@pytest.mark.parametrize('dynamic',[False,True])
@pytest.mark.parametrize('ranks,window',[(None,None),(2,3)])
def test_positional_only_sde_original_brian_and_bank_initial_vjps(
        engine,wrapped,dynamic,ranks,window,monkeypatch):
    # Both actual Python bodies must describe the oracle's 1+x+.2*x*x;
    # the wrapper halves its doubled underlying numeric body.
    curve=positional_prefix(doubled_positional_diffusion) if wrapped else positional_diffusion
    monkeypatch.setattr(static_sde,'diffusion',wrappers.metadata(curve))
    verify=(dynamic_sde.test_dynamic_sde_callback_physics_all_banks_and_initial_vjps
            if dynamic else static_sde.test_sde_static_callback_physics_all_bank_initial_and_tbptt_vjps)
    verify(engine,ranks,'heun',True,False,window)


@pytest.mark.parametrize('wrapped',[False,True])
@pytest.mark.parametrize('position',['pre','post','regular','synaptic_ode'])
def test_positional_only_synapse_forward(engine,wrapped,position,monkeypatch):
    curve=positional_prefix(positional_decay) if wrapped else positional_decay
    monkeypatch.setattr(callbacks,'decay',wrappers.metadata(curve))
    callbacks.test_dynamic_brian_callback_positions_and_forward(engine,position)


def two_arguments(x, /, y):
    return x+2*y


def two_prefix(f):
    @wraps(f)
    def wrapped(x, /, *args, **kwargs):
        return f(x,args[0],**kwargs)+3*x
    return wrapped


def explicit_two_prefix(x, y):
    return two_arguments(x,y)+3*x


def test_positional_only_prefix_keeps_remaining_packet_order():
    actual=compile_training_equation('f(v,2*v)',parameters={'f':lower_pure_function(two_prefix(two_arguments))})
    expected=compile_training_equation('f(v,2*v)',parameters={'f':lower_pure_function(explicit_two_prefix)})
    assert actual==expected


def test_positional_only_counts_toward_fixed_arity_bound():
    def too_many(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,/):
        return a+q
    with pytest.raises(ValueError,match='0..16 positional'):
        lower_pure_function(too_many)
