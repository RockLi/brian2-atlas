"""Immutable real control keeps original array callback and SDE semantics."""
import brian2 as b
import numpy as np
import pytest

from brian2_rust.training_functions import lower_pure_function
from test_training_integer_ir import engine
import test_training_pure_functions as callbacks
import test_training_static_control_sde as static_sde
import test_training_dynamic_static_sde as dynamic_sde


STEP = np.float64(.2)
LIMIT = .4


@b.check_units(x=1, result=1)
def diffusion(x):
    y = 1.+x
    clock = 0.
    while clock < LIMIT:
        if STEP/2. > 0. and clock <= .2:
            y += .1*x*x
        else:
            return np.sqrt(-1.)
        clock += STEP
    else:
        if not 0. or x:
            return y
    return np.sqrt(-1.)


@pytest.mark.parametrize('dynamic', [False, True])
@pytest.mark.parametrize('method,shared', [('heun',False),('heun',True),('heun','mixed'),('milstein',False)])
@pytest.mark.parametrize('ranks', [None,2,8])
@pytest.mark.parametrize('detach,window', [(False,None),(True,None),(False,3)])
def test_real_control_sde_original_forward_all_bank_initial_vjps(
        engine,dynamic,method,shared,ranks,detach,window,monkeypatch):
    monkeypatch.setattr(static_sde,'diffusion',diffusion)
    verify = (dynamic_sde.test_dynamic_sde_callback_physics_all_banks_and_initial_vjps
              if dynamic else static_sde.test_sde_static_callback_physics_all_bank_initial_and_tbptt_vjps)
    verify(engine,ranks,method,shared,detach,window)


def float_if(x):
    gain = .2/.5
    if .1 < gain <= .4:
        return gain*x
    return np.sqrt(-1.)


def float_while(x):
    y = 0.*x
    i = 0.
    while i < .4:
        y += .2*x
        i += .2
    return y


@pytest.mark.parametrize('function', [float_if,float_while])
def test_real_control_original_brian_callback_positions(engine,function,monkeypatch):
    monkeypatch.setattr(callbacks,'decay',b.check_units(x=1,result=1)(function))
    callbacks.test_dynamic_brian_callback_positions_and_forward(engine,'regular')


def float_range(x):
    for i in range(2.):
        x = .5*x
    return x


NONFINITE = np.float64(np.nan)


def nonfinite_control(x):
    if NONFINITE:
        return x
    return .5*x


def divide_zero(x):
    if .2/0.:
        return x
    return .5*x


def runtime_control(x):
    if x > .2:
        return x
    return .5*x


@pytest.mark.parametrize('function', [float_range,nonfinite_control,divide_zero,runtime_control])
def test_invalid_real_control_is_refused(function):
    with pytest.raises(ValueError):lower_pure_function(function)


class Opaque:
    def __bool__(self):raise AssertionError('opaque truth hook executed')
    def __lt__(self,other):raise AssertionError('opaque comparison hook executed')
    def __float__(self):raise AssertionError('opaque conversion hook executed')


OPAQUE = Opaque()


def opaque_truth(x):
    if OPAQUE:
        return x
    return .5*x


def opaque_comparison(x):
    if OPAQUE < .4:
        return x
    return .5*x


@pytest.mark.parametrize('function', [opaque_truth,opaque_comparison])
def test_real_control_never_invokes_opaque_hooks(function):
    with pytest.raises(ValueError):lower_pure_function(function)


SINGLE_STEP = np.float32(.1)
SINGLE_LIMIT = np.float32(.3)


def single_precision_while(x):
    y = 0.*x
    clock = 0.
    while clock < SINGLE_LIMIT:
        y += .2*x
        clock += SINGLE_STEP
    return y


def test_numpy_single_precision_control_original_brian_forward(engine,monkeypatch):
    monkeypatch.setattr(callbacks,'decay',b.check_units(x=1,result=1)(single_precision_while))
    callbacks.test_dynamic_brian_callback_positions_and_forward(engine,'regular')
