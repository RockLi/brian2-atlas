"""Exact int32 masks/shifts in stochastic dynamic paths and detached VJPs."""
import copy
import importlib
import os
import subprocess

import numpy as np
import pytest

from brian2_rust import NativeLIFTrainer, compile_training_equation
from brian2_rust.training_dynamic import compile_dynamic_transform
from test_native_training import RUNNER
from test_training_integer_ir import engine, model, wrap
from test_training_poisson_zero_vjp import mpi
from test_training_stochastic import normal


def bitwise(a, b, kind):
    return wrap({'bit_and': lambda: a & b, 'bit_or': lambda: a | b,
                 'bit_xor': lambda: a ^ b, 'left_shift': lambda: a << b,
                 'right_shift': lambda: a >> b}[kind]())


def fixture(kind, engine, ranks=None):
    p, w, x = model(kind=kind, noisy=True, engine=engine, ranks=ranks)
    w[1] = [0., 1., 7., 31.] if 'shift' in kind else [-1., 16777217., 1431655765., -2147483648.]
    return p, w, x


def oracle(p, w, kind, anchors=None):
    voltage = np.asarray(p['dynamic']['initial'][:4], float)
    integer = list(map(int, p['dynamic']['initial'][4:8])); margins = []; spikes = []
    for tick in range(4):
        integer = [bitwise(a, int(b), kind) for a, b in zip(integer, w[1])]
        flag = (np.asarray(integer) > 0).astype(float)
        voltage = .8*voltage+np.asarray(w[0])*flag+.05*flag*np.asarray([
            normal(p['seed'], 7, 0, 5+j//2, j % 2, tick, 0) for j in range(4)])
        margin = voltage-.5; event = (margin > 0).astype(float)
        if anchors is not None:
            base = anchors[tick]
            event = (base > 0).astype(float)+p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(base))**2*(margin-base)
        margins.append(margin.copy()); spikes.append(event.copy()); voltage -= .5*event
    logits = np.asarray(spikes)[:, 2:].mean(0)*p['logit_scale']
    loss = np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
    return loss, np.r_[voltage, integer, flag], np.asarray(spikes), margins


@pytest.mark.parametrize('kind', ['bit_and', 'bit_or', 'bit_xor', 'left_shift', 'right_shift'])
@pytest.mark.parametrize('ranks', [None, 2, 8])
def test_exact_stochastic_integer_masks_and_all_float_vjps(engine, kind, ranks):
    mpi(ranks); p, w, x = fixture(kind, engine, ranks)
    result = NativeLIFTrainer(p, weights=w, runner=RUNNER).gradients(x, [0], noise_sequence=7)
    loss, final, spikes, anchors = oracle(p, w, kind)
    np.testing.assert_array_equal(result['final_state'][0][4:], final[4:])
    np.testing.assert_array_equal(result['spikes'][0], spikes)
    np.testing.assert_allclose(result['final_state'][0][:4], final[:4], rtol=2e-5, atol=2e-6)
    assert result['loss'] == pytest.approx(loss, abs=2e-6)
    for j in range(4):
        plus = copy.deepcopy(w); minus = copy.deepcopy(w); plus[0][j] += 1e-6; minus[0][j] -= 1e-6
        fd = (oracle(p, plus, kind, anchors)[0]-oracle(p, minus, kind, anchors)[0])/2e-6
        assert result['gradients'][0][j] == pytest.approx(fd, abs=2e-6, rel=2e-4)
    assert result['gradients'][1] == [0.]*4
    assert result['initial_state_gradients'][0][4:] == [0.]*8


@pytest.mark.parametrize('kind', ['left_shift', 'right_shift'])
@pytest.mark.parametrize('count', [-1, 32, 2147483647])
@pytest.mark.parametrize('ranks', [None, 8])
def test_invalid_shift_on_nonroot_owner_is_atomic(engine, kind, count, ranks):
    mpi(ranks); p, w, x = fixture(kind, engine, ranks)
    w[1][-1] = float(count)
    trainer = NativeLIFTrainer(p, weights=w, runner=RUNNER); before = copy.deepcopy(trainer.state)
    with pytest.raises(ValueError, match='shift|nonfinite|GPU|native training'):
        trainer.step(x[:, :1], [0], noise_sequence=7)
    assert trainer.state == before and trainer.neuron_state is None and trainer.clock_tick == 0


@pytest.mark.parametrize('symbol,kind', [('&', 'bit_and'), ('|', 'bit_or'), ('^', 'bit_xor'),
                                      ('<<', 'left_shift'), ('>>', 'right_shift')])
def test_typed_frontend_augmented_bitwise_assignments(symbol, kind):
    expression = 'k '+symbol+'= rhs\nk = ~k\nv += gain*(k & 1)'
    result = compile_dynamic_transform(expression, states={'v': 0, 'k': 1},
        parameters={'rhs': 1, 'gain': (0, 0)}, state_types={1: 'integer'})
    nodes = [node for program in result['programs'] for node in program]
    assert any(node.get('op') == 'integer_binary' and node.get('kind') == kind for node in nodes)
    assert any(node.get('kind') == 'bit_xor' for node in nodes)
    assert any(node.get('op') == 'integer_float' for node in nodes)
    with pytest.raises(ValueError, match='typed int32'):
        compile_training_equation('v '+symbol+' 1', state_types=[])


@pytest.mark.parametrize('backend', ['metal', 'cuda'])
@pytest.mark.parametrize('version', [None, 0, 2])
def test_bitwise_versioned_capability(backend, version, tmp_path, monkeypatch):
    p, w, x = fixture('bit_xor', backend)
    source = tmp_path/'old.c'; library = tmp_path/'old.so'
    source.write_text('#include <stdint.h>\nuint64_t b2_train_math_v1(void){return 1;}\n'+
        ('' if version is None else 'uint64_t b2_train_bitwise_v1(void){return '+str(version)+';}\n'))
    subprocess.run(['cc', '-shared', '-fPIC', str(source), '-o', str(library)], check=True, capture_output=True)
    monkeypatch.setattr(importlib.import_module('brian2_rust.training_'+backend), 'build', lambda directory: library)
    trainer = NativeLIFTrainer(p, weights=w, runner=RUNNER); before = copy.deepcopy(trainer.state)
    with pytest.raises(ValueError, match='GPU bitwise capability'):
        trainer.gradients(x, [0], noise_sequence=7)
    assert trainer.state == before and trainer.neuron_state is None


@pytest.mark.parametrize('selected', [False, True])
@pytest.mark.parametrize('ranks', [None, 8])
def test_bitwise_lazy_branch_preserves_integer_nan_payload(engine, selected, ranks):
    mpi(ranks); p, w, x = fixture('left_shift', engine, ranks)
    w[1] = [-1., -1., -1., -1.]
    for group in p['dynamic']['program_sets']:
        for i, program in enumerate(group):
            if not any(node.get('op') == 'integer_binary' for node in program):
                continue
            # The selected invalid shift must fail; a skipped one must return
            # exact int32 -1 (an f32 NaN payload), rather than an error flag.
            target = 2
            if program[-1]['op'] == 'integer_compare':
                program = program[:3]
            group[i] = program[:3]+[dict(op='constant', value=float(selected)),
                dict(op='integer_constant', value=-1),
                dict(op='integer_select', condition=3, yes=target, no=4)]
            if i == 1:
                group[i] += [dict(op='integer_constant', value=0),
                             dict(op='integer_compare', left=5, right=6, kind='gt')]
    trainer = NativeLIFTrainer(p, weights=w, runner=RUNNER); before = copy.deepcopy(trainer.state)
    if selected:
        with pytest.raises(ValueError, match='shift|nonfinite|GPU|native training'):
            trainer.step(x[:, :1], [0], noise_sequence=7)
        assert trainer.state == before and trainer.neuron_state is None
    else:
        result = trainer.gradients(x[:, :1], [0], noise_sequence=7)
        assert result['final_state'][0][4:8] == [-1.]*4
        assert result['initial_state_gradients'][0][4:] == [0.]*8


@pytest.mark.parametrize('vector', [False, True])
def test_bitwise_scalar_vector_float_parameter_vjp(engine, vector):
    from brian2_rust import lif_training_plan
    from brian2_rust.training_equations import neuron_parameter_bank
    program = compile_training_equation('gain*((-2147483648 >> 31) ^ -2)+.2*v',
        parameters={'gain': (0, 0)}, states=['v'] if vector else None,
        state_types=['float'] if vector else [])
    programs = dict(state_equations=[[program], [program]],
        state_resets=[[[dict(op='state', index=0)]], [[dict(op='state', index=0)]]]) if vector else dict(equations=[program, program])
    p = lif_training_plan([1, 1, 2], projections=[neuron_parameter_bank(1)],
                         backend=engine, threshold=.5, surrogate_slope=2, **programs)
    # -1 xor -2 == 1; therefore every pre-threshold margin is gain+.2*v-.5.
    initial = np.asarray([[.1, .9, .2]]); weights = [[.4]]
    result = NativeLIFTrainer(p, weights=weights, runner=RUNNER).gradients(np.zeros((1, 1, 1)), [0], initial=initial)
    u = .4+.2*initial[0]; hard = (u > .5).astype(float)
    logits = hard[1:]*p['logit_scale']; probability = np.exp(logits-logits.max()); probability /= probability.sum(); probability[0] -= 1
    phi = p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(u[1:]-.5))**2
    expected = (probability*phi*p['logit_scale']).sum()
    assert result['gradients'][0][0] == pytest.approx(expected, abs=2e-6, rel=2e-5)
    np.testing.assert_array_equal(result['spikes'][0][0], hard)
