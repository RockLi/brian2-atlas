"""Direct scalar contexts against physical recurrence and stopped-law VJPs."""
import copy
import math
import os
import subprocess
import importlib

import numpy as np
import pytest

from brian2_rust import NativeLIFTrainer, lif_training_plan, compile_training_equation
from brian2_rust.training_equations import (
    SimulationTime, NormalNoise, UniformNoise, PoissonNoise, TimedInput,
    NeuronParameter, neuron_parameter_bank,
)
from test_native_training import RUNNER
from test_native_training_graph import fixture_graph
from test_training_poisson_static import draw
from test_training_poisson_zero_vjp import mpi
from test_training_stochastic import normal
from test_training_uniform import uniform


@pytest.fixture(params=['cpu','metal','cuda'])
def engine(request):
    name=request.param
    flag={'metal':'B2_TEST_GPU','cuda':'B2_TEST_CUDA_TRAIN'}.get(name)
    if flag and os.environ.get(flag)!='1':pytest.skip('actual '+name+' hardware required')
    return name


def model(reset='subtract', detach=False, window=None, ranks=None, poisson=True, zero=False, backend='cpu'):
    old, weights, x, y, initial = fixture_graph(reset, detach, window)
    parameters = dict(t=SimulationTime(), n=NormalNoise(1), u=UniformNoise(2),
                      draw=PoissonNoise(0), decay=(4, 0), amp=(4, 1), scale=(4, 2),
                      bias=(4, 3), sigma=NeuronParameter(4, 6),
                      table=TimedInput(5, 4, 1, .0125, 8, 1))
    rate = 'scale*(1+v*v)' if zero else 'scale*(1+v*v)+.2*table(t-.05)'
    expr = 'decay*v+bias+.04*t+.08*table(t-.05)+sigma*(v+1)*n+.03*u'
    if poisson:
        expr += '+amp*draw(' + rate + ')'
    programs = [compile_training_equation(expr, parameters=parameters) for _ in range(2)]
    p = lif_training_plan([2, 2, 2], projections=old['projections'] +
                         [neuron_parameter_bank(8), neuron_parameter_bank(4)],
                         equations=programs, clock=dict(origin=.05, dt=.1),
                         noise_streams=[3, 3], seed=731, threshold=[.6, .7],
                         threshold_parameters=[[4, 4], [4, 5]], surrogate_slope=2,
                         reset=reset, detach_reset=detach, tbptt_window=window,
                         mpi_ranks=ranks, backend=backend, trainable=[True]*5+[False], logit_scale=1.7)
    weights += [[.55, .36, 0. if zero else .4, .025, .6, .7, .02, .04], [.3, .7, .5, .9]]
    # Negative/fractional amplitudes distinguish scalar analog projections from
    # a binary-event adapter, including tied recurrence and feedback.
    x = x[:, :4].copy()*np.array([.25, 1.3])
    x[0, 1, 1] = -.5
    return p, weights, x, np.asarray(y), initial


def oracle(p, w, x, labels, initial, anchors=None, forced=None, sequence=9,
           start=0, zero=False, poisson=True):
    v = initial.copy(); batch, length, _ = x.shape
    theta = np.repeat(w[4][4:6], 2); before = []; margins = []; counts = []; spikes = []
    logp = np.zeros(batch)
    for t in range(length):
        if anchors is not None and p['tbptt_window'] and t and t % p['tbptt_window'] == 0:
            v = anchors['before'][t].copy()
        before.append(v.copy())
        timestamp = p['clock']['origin']+(start+t)*p['clock']['dt']
        # Brian TimedArray uses round((t/epsilon+0.5)/K), then clamps its row.
        row = min(3, max(0, math.floor(((timestamp-.05)/.0125+.5)/8)))
        table = w[5][row]
        rate = w[4][2]*(1+v*v)+(0. if zero else .2*table)
        if poisson:
            k = np.asarray([[draw(rate[b, j], p['seed'], sequence, b, j//2, j % 2, start+t)
                             for j in range(4)] for b in range(batch)]) if anchors is None else anchors['counts'][t]
            if forced is not None and forced[0] == t:
                k = k.copy(); k[forced[1], forced[2]] = 1
            if anchors is not None and not zero:
                logp += np.asarray([sum(c*math.log(r)-r-math.lgamma(c+1)
                                      for c, r in zip(ks, rs)) for ks, rs in zip(k, rate)])
        else:
            k = np.zeros_like(v)
        counts.append(k.copy())
        n = np.asarray([[normal(p['seed'], sequence, b, j//2, j % 2, start+t, 1)
                         for j in range(4)] for b in range(batch)])
        udraw = np.asarray([[uniform(p['seed'], sequence, b, j//2, j % 2, start+t, 2)
                             for j in range(4)] for b in range(batch)])
        u = w[4][0]*v+w[4][3]+.04*timestamp+.08*table+np.tile(w[4][6:8], 2)*(v+1)*n+.03*udraw
        if poisson:
            u += w[4][1]*k
        margin = u-theta; hard = (margin > 0).astype(float); event = hard
        if anchors is not None:
            base = anchors['margins'][t]
            event = (base > 0).astype(float)+p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(base))**2*(margin-base)
        gate = hard if anchors is None or p['detach_reset'] else event
        # Match the scalar contract's subtraction before all projections.
        v = u.copy() if p['reset'] == 'zero' else u-theta*gate
        for projection, bank in zip(p['projections'][:4], w):
            source = projection['source_layer']; target = projection['target_layer']
            for a, b, slot in zip(projection['sources'], projection['targets'], projection['parameter_ids']):
                value = x[:, t, a] if source == 0 else event[:, 2*(source-1)+a]
                v[:, 2*(target-1)+b] += bank[slot]*value
        if p['reset'] == 'zero':
            v *= 1-gate
        margins.append(margin.copy()); spikes.append(event.copy())
    spikes = np.stack(spikes, 1); logits = spikes[:, :, 2:].mean(1)*p['logit_scale']
    maximum = logits.max(1)
    losses = maximum+np.log(np.exp(logits-maximum[:, None]).sum(1))-logits[np.arange(batch), labels]
    objective = losses.mean()+(0. if anchors is None else np.mean(anchors['losses']*logp))
    return objective, v, spikes, dict(before=before, margins=margins, counts=counts, losses=losses)


@pytest.mark.parametrize('reset', ['subtract', 'zero'])
@pytest.mark.parametrize('detach,window', [(False, None), (True, 2)])
@pytest.mark.parametrize('poisson', [False, True])
def test_physical_scalar_contexts_and_all_vjps(engine, reset, detach, window, poisson):
    p, w, x, y, initial = model(reset, detach, window, poisson=poisson, backend=engine)
    result = NativeLIFTrainer(p, weights=w, runner=RUNNER).gradients(x, y, initial=initial, start_tick=3, noise_sequence=9)
    loss, final, spikes, anchors = oracle(p, w, x, y, initial, start=3, poisson=poisson)
    assert 'final_state' not in result and 'initial_state_gradients' not in result
    assert result['final_tick'] == 7 and result['noise_sequence'] == 9
    assert result['loss'] == pytest.approx(loss, abs=2e-14 if engine=='cpu' else 2e-6)
    np.testing.assert_array_equal(result['spikes'], spikes)
    np.testing.assert_allclose(result['final_membrane'], final, rtol=2e-14 if engine=='cpu' else 2e-5, atol=2e-14 if engine=='cpu' else 2e-6)
    epsilon = 1e-6
    def objective(weights, live):
        return oracle(p, weights, x, y, live, anchors, start=3, poisson=poisson)[0]
    for bank, row in enumerate(w):
        for slot in range(len(row)):
            plus = copy.deepcopy(w); minus = copy.deepcopy(w)
            plus[bank][slot] += epsilon; minus[bank][slot] -= epsilon
            fd = (objective(plus, initial)-objective(minus, initial))/(2*epsilon)
            assert result['gradients'][bank][slot] == pytest.approx(fd, abs=2e-7 if engine=='cpu' else 2e-5, rel=2e-5 if engine=='cpu' else 3e-4)
    for b in range(len(initial)):
        for k in range(4):
            plus = initial.copy(); minus = initial.copy(); plus[b, k] += epsilon; minus[b, k] -= epsilon
            fd = (objective(w, plus)-objective(w, minus))/(2*epsilon)
            assert result['initial_gradients'][b][k] == pytest.approx(fd, abs=2e-7 if engine=='cpu' else 2e-5, rel=2e-5 if engine=='cpu' else 3e-4)


@pytest.mark.parametrize('ranks', [None, 2, 8])
@pytest.mark.parametrize('reset', ['subtract', 'zero'])
@pytest.mark.parametrize('detach,window', [(False, None), (True, 2)])
def test_scalar_zero_rate_complete_sample_weak_vjp(engine, ranks, reset, detach, window):
    mpi(ranks)
    p, w, x, y, initial = model(reset, detach, window, ranks, zero=True, backend=engine)
    result = NativeLIFTrainer(p, weights=w, runner=RUNNER).gradients(x, y, initial=initial, noise_sequence=9)
    baseline, final, spikes, anchors = oracle(p, w, x, y, initial, zero=True)
    expected = 0.
    for t, old in enumerate(anchors['before']):
        for b in range(len(x)):
            for k in range(4):
                alternate = oracle(p, w, x, y, initial, forced=(t, b, k), zero=True)[0]
                expected += (alternate-baseline)*(1+old[b, k]**2)
    assert result['gradients'][4][2] == pytest.approx(expected, abs=2e-13 if engine=='cpu' else 2e-5)
    assert result['loss'] == pytest.approx(baseline, abs=2e-14 if engine=='cpu' else 2e-6)
    np.testing.assert_array_equal(result['spikes'], spikes)
    np.testing.assert_allclose(result['final_membrane'], final, rtol=2e-14 if engine=='cpu' else 2e-5, atol=2e-14 if engine=='cpu' else 2e-6)
    assert len(result['poisson_state']['entries']) == len(x)*len(x[0])*4
    assert all(e['count'] == 0 and e['rate'] == 0 for e in result['poisson_state']['entries'])


@pytest.mark.parametrize('ranks', [None, 2, 8])
def test_scalar_context_carry_restore_timed_replacement_and_rewind(engine, ranks, tmp_path):
    mpi(ranks)
    p, w, x, y, initial = model(ranks=ranks, backend=engine)
    p['trainable'] = [False]*len(w)
    trainer = NativeLIFTrainer(p, weights=w, runner=RUNNER)
    first = trainer.step(x[:, :2], y, initial=initial, noise_sequence=9)
    checkpoint = tmp_path/'state.json'; trainer.store(checkpoint)
    restored = NativeLIFTrainer(p, runner=RUNNER); restored.restore(checkpoint)
    state = copy.deepcopy(restored.__dict__)
    replacement = [.8, .2, .4, 1.1]
    restored.update_timed_input(5, replacement); trainer.update_timed_input(5, replacement)
    assert restored.clock_tick == 2 and restored.noise_sequence == 9
    assert restored.neuron_state == state['neuron_state'] and restored.poisson_state == state['poisson_state']
    assert restored.state['step'] == state['state']['step']
    changed = copy.deepcopy(w); changed[5] = replacement
    expected = oracle(p, changed, x[:, 2:], y, np.asarray(first['final_membrane']), start=2)
    a = trainer.step(x[:, 2:], y, initial='carry'); b = restored.step(x[:, 2:], y, initial='carry')
    assert a == b
    np.testing.assert_array_equal(a['spikes'], expected[2])
    np.testing.assert_allclose(a['final_membrane'], expected[1], rtol=2e-14 if engine=='cpu' else 2e-5, atol=2e-14 if engine=='cpu' else 2e-6)
    assert a['final_tick'] == 4 and len(a['poisson_state']['entries']) == 32
    # Rewinding with a committed cache detaches every old rate and draw. Compare
    # against the ordinary scalar VJP with exactly the same cached counts.
    old = NativeLIFTrainer(p, weights=w, runner=RUNNER).gradients(x[:, :2], y, initial=initial,
        noise_sequence=9, poisson_state=first['poisson_state'])
    base = oracle(p, w, x[:, :2], y, initial)
    # Scale is read only inside the detached imported rate.
    assert old['gradients'][4][2] == 0.
    np.testing.assert_array_equal(old['spikes'], base[2])


@pytest.mark.parametrize('ranks', [None, 2, 8])
def test_scalar_nonroot_failure_is_atomic(ranks):
    mpi(ranks)
    p, w, x, y, initial = model(ranks=ranks)
    trainer = NativeLIFTrainer(p, weights=w, runner=RUNNER)
    trainer.step(x[:, :1], y, initial=initial, noise_sequence=9)
    fields = ('state', 'neuron_state', 'clock_tick', 'clock_state', 'poisson_state',
              'noise_sequence', 'next_noise_sequence', 'elapsed_ticks')
    def runtime(trainer):
        return copy.deepcopy({key: getattr(trainer, key) for key in fields})
    before = runtime(trainer)
    broken = copy.deepcopy(trainer.neuron_state); broken[0][-1] = -2.
    # A rate that is invalid only on the final neuron, owned by a nonroot rank.
    p, _, _, _, _ = model(ranks=ranks)
    p['equations'][1] = compile_training_equation('v+.1*draw(v)', parameters={'draw': PoissonNoise(0)})
    bad = NativeLIFTrainer(p, weights=w, runner=RUNNER)
    snapshot = runtime(bad)
    with pytest.raises(ValueError, match='Poisson'):
        bad.step(x[:, :1], y, initial=broken, noise_sequence=9)
    assert runtime(bad) == snapshot and runtime(trainer) == before


@pytest.mark.parametrize('backend', ['metal', 'cuda'])
@pytest.mark.parametrize('version', [None, 0, 2])
def test_scalar_context_versioned_capability(backend, version, tmp_path, monkeypatch):
    p, w, x, y, initial = model(backend=backend)
    source=tmp_path/'old.c'; library=tmp_path/'old.so'
    source.write_text('#include <stdint.h>\nuint64_t b2_train_math_v1(void){return 1;}\n'+
        ('' if version is None else 'uint64_t b2_train_scalar_context_v1(void){return '+str(version)+';}\n'))
    subprocess.run(['cc','-shared','-fPIC',str(source),'-o',str(library)],check=True,capture_output=True)
    monkeypatch.setattr(importlib.import_module('brian2_rust.training_'+backend),'build',lambda directory:library)
    trainer=NativeLIFTrainer(p,weights=w,runner=RUNNER); before=copy.deepcopy(trainer.state)
    with pytest.raises(ValueError,match='GPU scalar context capability'):
        trainer.gradients(x,y,initial=initial)
    assert trainer.state==before and trainer.neuron_state is None and trainer.poisson_state is None


@pytest.mark.parametrize('ranks', [None, 2, 8])
@pytest.mark.parametrize('rate', ['scale-scale', '0.*scale', 'v-v'])
def test_scalar_cancelled_rate_does_not_replay_singular_alternate(engine, ranks, rate):
    mpi(ranks)
    p, w, x, y, initial = model(ranks=ranks, backend=engine)
    program = compile_training_equation('1./(1.-draw('+rate+'))',
        parameters={'draw': PoissonNoise(0), 'scale': (4, 2)})
    p['equations'] = [program, program]
    result = NativeLIFTrainer(p, weights=w, runner=RUNNER).gradients(x[:, :1], y, initial=initial, noise_sequence=9)
    assert result['gradients'][4][2] == 0.
    np.testing.assert_array_equal(result['initial_gradients'], np.zeros_like(initial))


@pytest.mark.parametrize('start', [0, 1, 4])
def test_scalar_timed_2d_rows_columns_and_value_vjp(engine, start):
    p, w, x, y, initial = model(poisson=False, backend=engine)
    p['projections'].append(neuron_parameter_bank(2))
    p['masks'].append([1., 1.]); p['trainable'].append(False); w.append([0., 1.])
    program = compile_training_equation('.55*v+.1*table(t-.05,column)',
        parameters={'t': SimulationTime(), 'table': TimedInput(5, 2, 2, .0125, 8, 2),
                    'column': NeuronParameter(6)})
    p['equations'] = [program, program]
    w[:4] = [[0.]*len(bank) for bank in w[:4]]
    result = NativeLIFTrainer(p, weights=w, runner=RUNNER).gradients(x[:, :1], y, initial=initial, start_tick=start, noise_sequence=9)
    row = min(start, 1); values = np.tile(w[5][2*row:2*row+2], 2)
    u = .55*initial+.1*values; theta = np.repeat(w[4][4:6], 2)
    hard = (u > theta).astype(float); final = u-theta*hard
    np.testing.assert_array_equal(result['spikes'], hard[:, None])
    np.testing.assert_allclose(result['final_membrane'], final, atol=2e-14 if engine=='cpu' else 2e-6, rtol=2e-14 if engine=='cpu' else 2e-5)
    # For one tick the only loss path is output voltage -> surrogate -> table.
    logits = hard[:, 2:]*p['logit_scale']; maximum = logits.max(1)
    probability = np.exp(logits-maximum[:, None]); probability /= probability.sum(1)[:, None]
    probability[np.arange(len(y)), y] -= 1
    phi = p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(u[:, 2:]-theta[2:]))**2
    expected = np.zeros(4); expected[2*row:2*row+2] = (.1*probability*phi*p['logit_scale']/len(y)).sum(0)
    np.testing.assert_allclose(result['gradients'][5], expected, atol=2e-14 if engine=='cpu' else 2e-6, rtol=2e-14 if engine=='cpu' else 2e-5)
    assert result['gradients'][6] == [0., 0.]


@pytest.mark.parametrize('ranks', [None, 2, 8])
def test_scalar_invalid_imported_cache_and_input_replacement_are_atomic(engine, ranks, tmp_path):
    mpi(ranks)
    p, w, x, y, initial = model(ranks=ranks, backend=engine)
    trainer = NativeLIFTrainer(p, weights=w, runner=RUNNER)
    result = trainer.step(x[:, :1], y, initial=initial, noise_sequence=9)
    fields = ('state', 'neuron_state', 'clock_tick', 'clock_state', 'poisson_state',
              'noise_sequence', 'next_noise_sequence', 'elapsed_ticks')
    before = copy.deepcopy({key: getattr(trainer, key) for key in fields})
    bad = copy.deepcopy(result['poisson_state']); bad['entries'][0]['count'] += 1
    with pytest.raises(ValueError, match='Poisson'):
        trainer.step(x[:, 1:2], y, initial=trainer.neuron_state, start_tick=1, noise_sequence=9, poisson_state=bad)
    with pytest.raises(ValueError, match='input update'):
        trainer.update_timed_input(5, [1.])
    assert {key: getattr(trainer, key) for key in fields} == before
    path = tmp_path/'missing-cache.json'; trainer.store(path)
    import json
    envelope = json.loads(path.read_text())
    payload = json.loads(envelope['payload']); payload['poisson_state'] = None
    # Recalculate the public integrity field as a valid but incomplete payload.
    from brian2_rust.training import canonical_bytes
    import hashlib
    envelope['payload'] = canonical_bytes(payload).decode()
    envelope['sha256'] = hashlib.sha256(envelope['payload'].encode()).hexdigest()
    path.write_bytes(canonical_bytes(envelope))
    other = NativeLIFTrainer(p, weights=w, runner=RUNNER)
    with pytest.raises(ValueError, match='missing its Poisson draw state'):
        other.restore(path)
