"""Statically bounded callback loops: original Brian physics and native VJPs."""
import copy
import builtins
import dis
import importlib.util
from types import SimpleNamespace

import brian2 as b
import numpy as np
import pytest

from brian2_rust import NativeLIFTrainer, compile_training_equation
from brian2_rust.training_functions import lower_pure_function
from brian2_rust.training_dynamic import compile_dynamic_transform
from brian2_rust.training_equations import NeuronParameter, NormalNoise
from test_native_training import RUNNER
from test_training_integer_ir import engine, model, oracle
from test_training_poisson_zero_vjp import mpi
import test_training_pure_functions as callbacks


@b.check_units(x=1, result=1)
def loop_decay(x):
    y = x
    for i in range(2):
        y = .8*y+.01*np.sin(y)
    return y


def loop_noisy(voltage, gate, gain, draw):
    next_voltage = 0.*voltage
    i = 0
    while i < 4:
        next_voltage += .2*voltage
        i += 1
        continue
        next_voltage += np.sqrt(-1.)
    return next_voltage+gain*gate+.05*gate*draw


CAPTURED_STEPS = 3
RANGE_ALIAS = builtins.range


def nested_curve(x):
    y = 0.*x
    for i in RANGE_ALIAS(CAPTURED_STEPS):
        for j in range(i+1):
            y = y+x*.1
    else:
        y = y+.25*x
    return y


def descending_curve(x):
    y = 0.*x
    for i in range(3, -1, -1):
        y = y+i*x
    return y+i


def empty_curve(x):
    for i in range(0):
        x[0] = 99.
    else:
        x = .5*x
    return x


def break_curve(x):
    y = 0.*x
    for i in range(3):
        y = y+.7*x
        break
        y = np.sqrt(-1.)
    else:
        y = y+99.
    return y+i


def nested_continue_curve(x):
    y = 0.*x
    for i in range(3):
        for j in range(3):
            y = y+.1*x
            break
            y = np.sqrt(-1.)
        else:
            y = y+99.
        continue
        y = np.sqrt(-1.)
    else:
        y = y+.25*x
    return y


def inner_else_continue_curve(x):
    y = 0.*x
    for i in range(3):
        for j in range(0):
            y = np.sqrt(-1.)
        else:
            continue
        y = np.sqrt(-1.)
    else:
        y = y+.5*x
    return y


def inner_else_break_curve(x):
    y = .2*x
    for i in range(3):
        for j in range(0):
            y = np.sqrt(-1.)
        else:
            break
        y = np.sqrt(-1.)
    else:
        y = y+99.
    return y


def conditional_index_curve(x):
    y = 0.*x
    for i in range(5):
        if i%2 == 0:
            continue
        if i >= 3:
            break
        y = y+.6*x
    else:
        y = y+99.
    return y


def while_break_curve(x):
    y = 0.*x
    i = 0
    while i < 5:
        i += 1
        if i%2 == 0:
            continue
        y = y+.2*x
        if i == 3:
            break
    else:
        y = y+99.
    return y


def while_else_curve(x):
    y = 0.*x
    i = 0
    while i < 3:
        y = y+.1*x
        i += 1
    else:
        y = y+.2*x
    return y


def empty_while_curve(x):
    while False and x:
        x[0] = 99.
    else:
        x = .3*x
    return x


def while_else_continue_curve(x):
    y = 0.*x
    for i in range(3):
        while False:
            y = np.sqrt(-1.)
        else:
            continue
        y = np.sqrt(-1.)
    else:
        y = y+.55*x
    return y


def lazy_static_condition_curve(x):
    y = 0.*x
    while True:
        if not ((0 and x) or (1 < 2 <= 3)):
            y = np.sqrt(-1.)
        y = y+.7*x
        break
    else:
        y = np.sqrt(-1.)
    return y


@pytest.mark.parametrize('function', [nested_curve, descending_curve, empty_curve,
    break_curve, nested_continue_curve, inner_else_continue_curve, inner_else_break_curve,
    conditional_index_curve, while_break_curve, while_else_curve, empty_while_curve,
    while_else_continue_curve, lazy_static_condition_curve])
def test_nested_negative_and_empty_ranges_match_original_array_callback(engine, function):
    # Feed the same original callback through Brian's normal NumPy execution;
    # the lowerer snapshots code, never invokes its body.
    original = callbacks.decay
    try:
        callbacks.decay = b.check_units(x=1, result=1)(function)
        callbacks.test_dynamic_brian_callback_positions_and_forward(engine, 'regular')
    finally:
        callbacks.decay = original


@pytest.mark.parametrize('position', ['pre', 'post', 'regular', 'synaptic_ode'])
def test_loop_functions_in_dynamic_brian_positions(engine, position, monkeypatch):
    monkeypatch.setattr(callbacks, 'decay', loop_decay)
    callbacks.test_dynamic_brian_callback_positions_and_forward(engine, position)


@pytest.mark.parametrize('ranks', [None, 2, 8])
def test_noisy_loop_all_bank_and_physical_initial_vjps(engine, ranks):
    mpi(ranks); p,w,x = model(noisy=True, engine=engine, ranks=ranks)
    function = lower_pure_function(loop_noisy)
    for action in p['dynamic']['actions']:
        if action.get('noise_streams') != 1: continue
        p['dynamic']['program_sets'][action['program_set']] = compile_dynamic_transform(
            'v=f(v,flag,gain,draw)', states={'v':0, 'flag':1},
            parameters={'f':function, 'gain':NeuronParameter(0), 'draw':NormalNoise(0)})['programs']
    result = NativeLIFTrainer(p, weights=w, runner=RUNNER).gradients(x,[0],noise_sequence=7)
    loss, final, spikes, anchors = oracle(p,w,noisy=True)
    np.testing.assert_array_equal(result['spikes'][0],spikes)
    np.testing.assert_allclose(result['final_state'][0],final,rtol=2e-5,atol=3e-6)
    assert result['loss'] == pytest.approx(loss,abs=3e-6)
    for i in range(4):
        plus=copy.deepcopy(w);minus=copy.deepcopy(w);plus[0][i]+=1e-6;minus[0][i]-=1e-6
        fd=(oracle(p,plus,noisy=True,anchors=anchors)[0]-oracle(p,minus,noisy=True,anchors=anchors)[0])/2e-6
        assert result['gradients'][0][i] == pytest.approx(fd,abs=3e-6,rel=3e-4)
        plus=np.array(p['dynamic']['initial']);minus=plus.copy();plus[i]+=1e-6;minus[i]-=1e-6
        fd=(oracle(p,w,noisy=True,anchors=anchors,initial=plus)[0]-oracle(p,w,noisy=True,anchors=anchors,initial=minus)[0])/2e-6
        assert result['initial_state_gradients'][0][i] == pytest.approx(fd,abs=3e-6,rel=3e-4)
    np.testing.assert_array_equal(result['gradients'][1],0.)
    assert (result['gpu_dispatches']>0) == (engine!='cpu')


def test_immutable_loop_capture_is_snapshotted_without_execution(monkeypatch):
    function=lower_pure_function(nested_curve)
    assert len(function.statements)==17
    program=compile_training_equation('f(v)',parameters={'f':function})
    assert program[-1]['op']=='sequence'
    monkeypatch.setitem(nested_curve.__globals__, 'CAPTURED_STEPS', 0)
    assert compile_training_equation('f(v)',parameters={'f':function}) == program
    assert len(lower_pure_function(nested_curve).statements) == 2


def dynamic_bound(x, n):
    for i in range(n): x = x*.5
    return x


def oversized(x):
    for i in range(65): x = x+.1
    return x


def mutable_loop(x):
    for i in range(2): x += .1
    return x


def bad_step(x):
    for i in range(0, 1, 0): x = x+.1
    return x


def bad_index(x):
    for i in range(2**40, 2**40+1): x = x+.1
    return x


def undefined_zero_index(x):
    for i in range(0): x = x+.1
    return x+i


def dynamic_while_bound(x, n):
    i = 0
    while i < n:
        x = .5*x
        i += 1
    return x


def unbounded_while(x):
    while True:
        continue
    return x


def array_condition_loop(x):
    for i in range(2):
        if x:
            break
    return x


@pytest.mark.parametrize('function', [dynamic_bound, oversized, mutable_loop, bad_step, bad_index,
    undefined_zero_index, dynamic_while_bound, unbounded_while, array_condition_loop])
def test_unsupported_loop_semantics_refuse_without_invoking_callback(function):
    with pytest.raises(ValueError): lower_pure_function(function)


def test_shadowed_range_is_not_executed():
    calls=[]
    def impostor(n):
        calls.append(n)
        return builtins.range(n)
    range=impostor
    def callback(x):
        for i in range(3): x = x+.1
        return x
    with pytest.raises(ValueError,match='unmodified builtin range'): lower_pure_function(callback)
    assert not calls


@pytest.mark.parametrize('source', [
    'def curve(x):\n    for i in range(n): x=.5*x\n    n=2\n    return x\n',
    'def curve(x):\n    for i in range(2): x=.5*x\n    range=2\n    return x\n',
    'def curve(x):\n    for i in range(0): x=.5*x\n    for j in range(i): x=.5*x\n    return x\n',
])
def test_unbound_loop_locals_do_not_fall_back_to_captured_values(tmp_path, source):
    file=tmp_path/'unbound_loop.py'
    file.write_text('n=2\ni=2\n'+source)
    spec=importlib.util.spec_from_file_location('b2_unbound_loop',file)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    with pytest.raises(UnboundLocalError):module.curve(np.array([1.]))
    with pytest.raises(ValueError):lower_pure_function(module.curve)


def test_stale_loop_control_flow_source_is_refused(tmp_path):
    file=tmp_path/'loop_source.py'
    original='def curve(x):\n    y=x\n    for i in range(2):\n        y=.5*y\n    return y\n'
    file.write_text(original)
    spec=importlib.util.spec_from_file_location('b2_loop_stale_source',file)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert lower_pure_function(module.curve)
    file.write_text(original.replace('        y=.5*y\n','        y=.5*y\n        return y\n'))
    with pytest.raises(ValueError,match='differs from loaded Python code'):lower_pure_function(module.curve)


def test_loop_source_fingerprint_without_new_dis_jump_inventory(monkeypatch):
    import brian2_rust.training_functions as functions
    # Hide only the adapter's inventory; current dis itself needs hasjump to
    # decode bytecode and must retain its own version's internal globals.
    facade=SimpleNamespace(get_instructions=dis.get_instructions,hasjabs=dis.hasjabs,hasjrel=dis.hasjrel)
    monkeypatch.setattr(functions,'dis',facade)
    function=lower_pure_function(loop_decay)
    assert len(function.statements)==5


def test_static_conditions_never_invoke_opaque_truth_or_comparison():
    calls=[]
    class Opaque:
        def __bool__(self):
            calls.append('truth')
            return True
        def __lt__(self, other):
            calls.append('comparison')
            return True
    guard=Opaque()
    def while_callback(x):
        while guard:
            x=.5*x
            break
        return x
    def if_callback(x):
        if guard:
            x=.5*x
        return x
    def comparison_callback(x):
        while guard < 2:
            x=.5*x
            break
        return x
    for callback in (while_callback,if_callback,comparison_callback):
        with pytest.raises(ValueError):lower_pure_function(callback)
    assert calls==[]
