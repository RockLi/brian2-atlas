"""Static v4 Poisson: physical f64 recurrence and stopped-likelihood VJP oracle."""
import copy
import ast
import hashlib
import json
import math
import subprocess
from unittest.mock import patch

import numpy as np
import pytest
import brian2 as b
from brian2_rust import NativeLIFTrainer, lif_training_plan, compile_training_equation
from brian2_rust.training_equations import PoissonNoise, neuron_parameter_bank, _compile_training_ast
from test_native_training import RUNNER
from test_training_poisson_core import mix, MASK, uniform
from test_training_poisson_zero_vjp import mpi
from test_training_linked import cython_cache


def draw(rate, seed, sequence, batch, layer, neuron, tick, stream=0):
    key = mix(seed ^ 0x4232504f49533031)
    for field in (sequence, batch, layer, neuron, 0, tick, stream):
        key = mix(key ^ mix((field + 0x9e3779b97f4a7c15) & MASK))
    arrival = 0.
    for count in range(100000):
        arrival -= math.log1p(-uniform(key, count))
        if arrival >= rate:
            return count
    raise AssertionError('reference draw budget')


def model(window=None, detach=False, ranks=None, rate='scale*r', reset_draw=True):
    params = {'draw': PoissonNoise(0), 'scale': (3, 0), 'amp': (3, 1), 'offset': (3, 2)}
    def compile(expr):
        return expression(expr, params)
    updates = [[compile(f'.4*v+amp*draw({rate})'), compile('.8*r+.2*offset')] for _ in range(2)]
    resets = [[compile('v-.6'), compile('.9*r+.1*offset'+(f'+.05*draw({rate})' if reset_draw else ''))] for _ in range(2)]
    projections = [dict(source_layer=0, target_layer=1, parameter_count=1, sources=[0], targets=[0], parameter_ids=[0]),
                   dict(source_layer=1, target_layer=2, parameter_count=2, sources=[0, 0], targets=[0, 1], parameter_ids=[0, 1]),
                   dict(source_layer=2, target_layer=1, parameter_count=1, sources=[0, 1], targets=[0, 0], parameter_ids=[0, 0]),
                   neuron_parameter_bank(3)]
    p = lif_training_plan([1, 1, 2], projections=projections, state_equations=updates, state_resets=resets,
                          clock={'origin': 0., 'dt': .001}, noise_streams=[1, 1], seed=731,
                          threshold=.6, detach_reset=detach, tbptt_window=window, mpi_ranks=ranks, logit_scale=1.7)
    return p, [[.31], [.28, -.12], [.04], [1.2, .38, 1.3]]


def expression(code, parameters):
    # The public scalar compiler deliberately excludes conditional syntax;
    # the existing frontend's expression-DAG entry point admits lazy selects.
    return _compile_training_ast(ast.parse(code, mode='eval').body, states=['v', 'r'],
                                 parameters=parameters, allow_select=True, typed=True)


def fixture(batch):
    initial = np.tile([.7, 1.1, .2, .9, 1.2, 1.4], (batch, 1))
    initial[:, [1, 4, 5]] += np.arange(batch)[:, None]*.1
    x = np.tile(np.array([.25, -.5, 1.3, .125])[None, :, None], (batch, 1, 1))
    return x, np.arange(batch)%2, initial


def oracle(p, w, x, labels, initial, anchors=None, forced=None, sequence=9):
    live = initial.copy(); batch, length, _ = x.shape
    scale, amp, offset = w[3]
    before = []; margins = []; counts = []; spikes = []; rates = []; logp = np.zeros(batch)
    for tick in range(length):
        if anchors is not None and p['tbptt_window'] and tick and tick % p['tbptt_window'] == 0:
            live = anchors['before'][tick].copy()
        before.append(live.copy()); rate = scale*live[:, [1, 4, 5]]; rates.append(rate.copy())
        k = np.array([[draw(rate[b, j], p['seed'], sequence, b, int(j > 0), max(j-1, 0), tick)
                       for j in range(3)] for b in range(batch)]) if anchors is None else anchors['counts'][tick]
        if forced is not None and forced[0] == tick:
            k[forced[1], forced[2]] = 1
        counts.append(k.copy())
        if anchors is not None:
            logp += np.array([sum(c*math.log(r)-r-math.lgamma(c+1) for c, r in zip(ks, rs)) for ks, rs in zip(k, rate)])
        u = live.copy(); u[:, [0, 2, 3]] = .4*live[:, [0, 2, 3]]+amp*k
        u[:, [1, 4, 5]] = .8*live[:, [1, 4, 5]]+.2*offset
        margin = u[:, [0, 2, 3]]-.6; hard = (margin > 0).astype(float); event = hard
        if anchors is not None:
            base = anchors['margins'][tick]
            event = (base > 0).astype(float)+p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(base))**2*(margin-base)
        gate = hard if anchors is None or p['detach_reset'] else event
        u[:, 0] += w[0][0]*x[:, tick, 0]+w[2][0]*event[:, 1:].sum(1)
        u[:, 2:4] += event[:, :1]*np.array(w[1])
        live = u.copy(); live[:, [0, 2, 3]] -= .6*gate
        live[:, [1, 4, 5]] += gate*(-.1*u[:, [1, 4, 5]]+.1*offset+.05*k)
        margins.append(margin.copy()); spikes.append(event.copy())
    spikes = np.stack(spikes, 1); logits = spikes[:, :, 1:].mean(1)*p['logit_scale']
    maximum = logits.max(1); losses = maximum+np.log(np.exp(logits-maximum[:, None]).sum(1))-logits[np.arange(batch), labels]
    objective = losses.mean()+(0. if anchors is None else np.mean(anchors['losses']*logp))
    return objective, live, spikes, dict(before=before, margins=margins, counts=counts, rates=rates, losses=losses)


@pytest.mark.parametrize('window', [None, 1, 2])
@pytest.mark.parametrize('detach', [False, True])
@pytest.mark.parametrize('batch', [1, 3])
@pytest.mark.parametrize('ranks', [None, 2, 8])
def test_physical_recurrence_and_all_vjps(window, detach, batch, ranks):
    mpi(ranks); p, w = model(window, detach, ranks); x, y, initial = fixture(batch)
    r = NativeLIFTrainer(p, weights=w, runner=RUNNER).gradients(x, y, initial=initial, noise_sequence=9)
    loss, final, spikes, anchors = oracle(p, w, x, y, initial)
    assert r['loss'] == pytest.approx(loss, abs=3e-14)
    np.testing.assert_array_equal(r['spikes'], spikes)
    np.testing.assert_allclose(r['final_state'], final, rtol=2e-14, atol=2e-14)
    assert len(r['poisson_state']['entries']) == batch*4*3
    eps = 1e-6
    for bank, row in enumerate(w):
        for j in range(len(row)):
            plus = copy.deepcopy(w); minus = copy.deepcopy(w); plus[bank][j] += eps; minus[bank][j] -= eps
            fd = (oracle(p, plus, x, y, initial, anchors)[0]-oracle(p, minus, x, y, initial, anchors)[0])/(2*eps)
            assert r['gradients'][bank][j] == pytest.approx(fd, rel=5e-5, abs=5e-8)
    for b in range(batch):
        for j in range(6):
            plus = initial.copy(); minus = initial.copy(); plus[b, j] += eps; minus[b, j] -= eps
            fd = (oracle(p, w, x, y, plus, anchors)[0]-oracle(p, w, x, y, minus, anchors)[0])/(2*eps)
            assert r['initial_state_gradients'][b][j] == pytest.approx(fd, rel=5e-5, abs=5e-8)


@pytest.mark.parametrize('window', [None, 1, 2])
@pytest.mark.parametrize('batch', [1, 3])
@pytest.mark.parametrize('ranks', [None, 2, 8])
def test_zero_rate_full_counterfactual(window, batch, ranks):
    mpi(ranks); p, w = model(window, ranks=ranks); w[3][0] = 0.
    x, y, initial = fixture(batch)
    r = NativeLIFTrainer(p, weights=w, runner=RUNNER).gradients(x, y, initial=initial, noise_sequence=9)
    baseline, final, spikes, anchors = oracle(p, w, x, y, initial)
    expected = 0.
    for tick in range(4):
        for b in range(batch):
            for j in range(3):
                alternate = oracle(p, w, x, y, initial, forced=(tick, b, j))[0]
                expected += anchors['before'][tick][b, [1, 4, 5][j]]*(alternate-baseline)
    assert r['gradients'][3][0] == pytest.approx(expected, rel=1e-12, abs=5e-14)
    np.testing.assert_allclose(r['final_state'], final, rtol=2e-14, atol=2e-14)
    np.testing.assert_array_equal(r['spikes'], spikes)


@pytest.mark.parametrize('rate', ['scale-scale', '0.*scale', 'r-r', 'scale if r<0 else 0.', 'sin(scale)-sin(scale)'])
@pytest.mark.parametrize('ranks', [None, 2])
def test_cancelled_rate_does_not_replay_invalid_alternate(rate, ranks):
    mpi(ranks); p, w = model(ranks=ranks, rate=rate, reset_draw=False)
    for layer in p['state_equations']:
        layer[0] = expression(f'1./(1.-draw({rate}))', {'draw':PoissonNoise(0), 'scale':(3, 0)})
    x, y, initial = fixture(1)
    result = NativeLIFTrainer(p, weights=w, runner=RUNNER).gradients(x[:, :2], y, initial=initial)
    assert result['gradients'][3][0] == 0.
    assert all(e['count'] == 0 for e in result['poisson_state']['entries'])


@pytest.mark.parametrize('ranks', [None, 2])
def test_invalid_active_alternate_rolls_back(ranks):
    mpi(ranks); p, w = model(ranks=ranks, rate='scale', reset_draw=False); w[3][0] = 0.
    for layer in p['state_equations']:
        layer[0] = compile_training_equation('1./(1.-draw(scale))', states=['v', 'r'],
            parameters={'draw':PoissonNoise(0), 'scale':(3, 0)})
    x, y, initial = fixture(1); t = NativeLIFTrainer(p, weights=w, runner=RUNNER); before = copy.deepcopy(t.state)
    t.evaluate(x[:, :1], y, initial=initial)
    with pytest.raises(ValueError, match='nonfinite equation'):
        t.step(x[:, :1], y, initial=initial)
    assert t.state == before and t.poisson_state is None and t.clock_tick == 0


@pytest.mark.parametrize('ranks', [None, 2, 8])
def test_carry_store_restore_and_detached_rewind(ranks, tmp_path):
    mpi(ranks); p, w = model(ranks=ranks); p['trainable'] = [False]*4
    x, y, initial = fixture(3); t = NativeLIFTrainer(p, weights=w, runner=RUNNER)
    whole = t.gradients(x, y, initial=initial, noise_sequence=9)
    first = t.step(x[:, :2], y, initial=initial, noise_sequence=9)
    path = tmp_path/'static'; t.store(path); restored = NativeLIFTrainer(p, weights=w, runner=RUNNER); restored.restore(path)
    a = t.step(x[:, 2:], y, initial='carry'); assert restored.step(x[:, 2:], y, initial='carry') == a
    np.testing.assert_allclose(a['final_state'], whole['final_state'], rtol=2e-14, atol=2e-14)
    assert a['poisson_state'] == whole['poisson_state']
    # An imported observation has no score owner, even after a clock rewind.
    restored.restore(path); restored.clock_tick = 0
    restored.state['weights'][3][0] = -1.
    result = restored.gradients(x[:, :2], y, initial='carry')
    assert result['gradients'][3][0] == 0.
    assert result['poisson_state'] == first['poisson_state']


@pytest.mark.parametrize('malformation', ['duplicate', 'count', 'rate', 'missing'])
def test_restore_validation_is_atomic(malformation, tmp_path):
    p, w = model(); x, y, initial = fixture(1); t = NativeLIFTrainer(p, weights=w, runner=RUNNER)
    t.step(x[:, :1], y, initial=initial); path = tmp_path/'static'; t.store(path)
    envelope = json.loads(path.read_text()); payload = json.loads(envelope['payload'])
    if malformation == 'duplicate': payload['poisson_state']['entries'] *= 2
    elif malformation == 'missing': payload.pop('poisson_state')
    elif malformation == 'rate': payload['poisson_state']['entries'][0]['rate'] = -1.
    else: payload['poisson_state']['entries'][0]['count'] += 1
    from brian2_rust.training import canonical_bytes
    raw = canonical_bytes(payload); envelope['payload'] = raw.decode(); envelope['sha256'] = hashlib.sha256(raw).hexdigest()
    path.write_text(json.dumps(envelope)); other = NativeLIFTrainer(p, weights=w, runner=RUNNER); before = copy.deepcopy(other.state)
    with pytest.raises(ValueError, match='Poisson|poisson'): other.restore(path)
    assert other.state == before and other.poisson_state is None


def test_static_gpu_requires_a_selected_native_library(tmp_path):
    p, w = model(); x, y, initial = fixture(1)
    state = NativeLIFTrainer(p, weights=w, runner=RUNNER).state
    p['backend'] = 'metal'
    request = tmp_path/'request'; output = tmp_path/'output'
    request.write_text(json.dumps(dict(plan=p, state=state, operation='gradients',
        inputs=x.tolist(), labels=y.tolist(), initial=initial.tolist())))
    # Capability tests in the GPU module cover old libraries. Without any
    # selected native library the CLI must fail before dispatch, on all hosts.
    result = subprocess.run([str(RUNNER), str(request), str(output)], capture_output=True, text=True, timeout=30,
        env={k:v for k,v in __import__('os').environ.items() if k!='B2_TRAIN_METAL_LIB'})
    assert result.returncode != 0
    assert 'B2_TRAIN_METAL_LIB' in result.stderr
    assert not output.exists()


@pytest.mark.parametrize('ranks', [None, 2, 8])
def test_reset_first_observation_uses_post_update_rate_context(ranks):
    mpi(ranks); p, w = model(ranks=ranks); x, y, initial = fixture(3)
    initial[:, [0, 2, 3]] = .7
    params = {'draw': PoissonNoise(0), 'scale': (3, 0)}
    for layer in p['state_equations']:
        layer[0] = expression('v+draw(scale*r) if v>2. else v', params)
    for layer in p['state_resets']:
        layer[0] = expression('v-.6+.05*draw(scale*r)', params)
        layer[1] = expression('r', params)
    result = NativeLIFTrainer(p, weights=w, runner=RUNNER).gradients(x[:, :1], y, initial=initial, noise_sequence=9)
    entries = result['poisson_state']['entries']; assert len(entries) == 9
    # All output neurons spike; sample CE is log(2), and scale only enters a
    # detached count, so its VJP consists solely of the first-observation score.
    expected = sum(math.log(2.)/3*(e['count']/e['rate']-1)*e['rate']/w[3][0] for e in entries)
    assert result['gradients'][3][0] == pytest.approx(expected, abs=3e-14)
    for e in entries:
        id = e['identity']; j = id['site']['entity']; l = id['site']['domain']; sample = id['batch']
        source = [1, 4, 5][0 if l == 0 else 1+j]
        assert e['rate'] == pytest.approx(w[3][0]*(.8*initial[sample, source]+.2*w[3][2]))
        assert result['initial_state_gradients'][sample][source] == pytest.approx(
            math.log(2.)/3*(e['count']/e['rate']-1)*w[3][0]*.8, abs=3e-14)


@pytest.mark.parametrize('ranks', [None, 2])
@pytest.mark.parametrize('masked', [False, True])
def test_singular_rate_activity_and_rollback(ranks, masked):
    mpi(ranks); p, w = model(ranks=ranks, rate='sqrt(scale)', reset_draw=False); w[3][0] = 0.
    for layer in p['state_equations']:
        layer[0] = expression('1./(1.-draw(sqrt(scale)))', {'draw': PoissonNoise(0), 'scale': (3, 0)})
    if masked: p['masks'][3][0] = 0.
    x, y, initial = fixture(1); t = NativeLIFTrainer(p, weights=w, runner=RUNNER); before = copy.deepcopy(t.state)
    if masked:
        result = t.gradients(x[:, :1], y, initial=initial)
        assert result['gradients'][3][0] == 0.
    else:
        with pytest.raises(ValueError, match='nonfinite equation derivative'):
            t.step(x[:, :1], y, initial=initial)
        assert t.state == before and t.poisson_state is None


@pytest.mark.parametrize('problem', ['different_rate', 'mixed_distribution', 'budget', 'invalid_rate'])
@pytest.mark.parametrize('ranks', [None, 2])
def test_admission_and_partial_draw_failure_are_atomic(problem, ranks):
    mpi(ranks); p, w = model(ranks=ranks); x, y, initial = fixture(1)
    match = 'rate expressions'
    if problem == 'different_rate':
        p['state_resets'][1][1] = expression('r+.05*draw(scale*r+1.)', {'draw': PoissonNoise(0), 'scale':(3, 0)})
    elif problem == 'mixed_distribution':
        p['state_resets'][1][1] = [{'op':'noise', 'stream':0}]; match = 'mixes distributions'
    elif problem == 'budget':
        good = NativeLIFTrainer(p, weights=w, runner=RUNNER).gradients(x, y, initial=initial)
        p['max_tape_bytes'] = good['tape_bytes']-1; match = 'budget'
    else:
        initial[0, 5] = -1.; match = 'Poisson sample'
    t = NativeLIFTrainer(p, weights=w, runner=RUNNER); before = copy.deepcopy(t.state)
    with pytest.raises(ValueError, match=match): t.step(x, y, initial=initial)
    assert t.state == before and t.poisson_state is None and t.clock_tick == 0 and t.next_noise_sequence == 0


@pytest.mark.parametrize('warm', [0, 2])
@pytest.mark.parametrize('ranks', [None, 2])
def test_static_frontend_matches_actual_cython(warm, ranks, tmp_path):
    _static_frontend_cython(warm, ranks, tmp_path)


def _static_frontend_cython(warm, ranks, tmp_path, backend='cpu'):
    mpi(ranks); b.set_device('runtime'); b.start_scope(); b.prefs.codegen.target = 'cython'; dt = .125*b.ms
    inp = b.SpikeGeneratorGroup(1, [], []*b.ms, dt=dt, name='static_poisson_input')
    groups = []
    for l in range(2):
        g = b.NeuronGroup(2, 'dv/dt=(-v+.5*poisson(rate))/ms:1\nda/dt=-a/ms:1\nrate:1 (constant)',
                          threshold='v>.3', reset='a+=.125*poisson(rate)\nv-=.25+.125*a',
                          method='euler', dt=dt, name=f'static_poisson_{l}')
        g.rate = [1.25, 2.5]; g.v = [.45, .75]; g.a = [.125, .25]; groups.append(g)
    syn = b.Synapses(groups[0], groups[1], 'w:1', on_pre='v_post+=w', name='static_poisson_syn')
    syn.connect(j='i'); syn.w = [.0625, .125]
    net = b.Network(inp, *groups, syn)
    if warm: b.seed(731); net.run(warm*dt, namespace={})
    net.run(0*b.ms, namespace={})
    from brian2_rust.training_brian import lower_brian_training
    bundle = lower_brian_training(net, input_group=inp, layers=groups, mpi_ranks=ranks,
        trainable_neuron_parameters={g.name:['rate'] for g in groups})
    assert bundle.plan['schema'] == 'b2-state-training-plan-v4' and 'dynamic' not in bundle.plan
    assert bundle.provenance['noise_gradient'] == 'likelihood-score-poisson-and-pathwise-fixed-draws'
    bundle.plan['backend'] = backend
    bundle.plan['trainable'] = [False]*len(bundle.weights)
    t = NativeLIFTrainer(bundle.plan, weights=bundle.weights, runner=RUNNER)
    cursor = 0
    for length in (2, 3):
        result = t.step(np.zeros((1, length, 1)), [0], **({'initial':'carry'} if cursor else {'initial':[bundle.initial_state], 'noise_sequence':9}))
        draws = []
        for tick in range(cursor, cursor+length):
            for l in range(2):
                for j, rate in enumerate((1.25, 2.5)):
                    draws.extend(cython_uniforms(bundle.plan['seed'], 9, l, j, tick, 0, rate))
            for l in range(2):
                for j, rate in enumerate((1.25, 2.5)):
                    if result['spikes'][0][tick-cursor][l*2+j]:
                        draws.extend(cython_uniforms(bundle.plan['seed'], 9, l, j, tick, 1, rate))
        calls = []; device = b.get_device(); device.rand_buffer_index[:] = 0
        def refill(n):
            assert n == 20000 and not calls; calls.append(n); out = np.full(n, .5); out[:len(draws)] = draws; return out
        with patch('numpy.random.rand', refill): net.run(length*dt, namespace={})
        assert calls == [20000] and device.rand_buffer_index[0] == len(draws)
        device.rand_buffer_index[:] = 0
        physical = np.concatenate([g.variables[name].get_value() for g, names in zip(groups, bundle.provenance['state_names']) for name in names])
        np.testing.assert_allclose(result['final_state'][0], physical,
            rtol=3e-13 if backend=='cpu' else 4e-5, atol=3e-14 if backend=='cpu' else 5e-6)
        assert result['backend']==backend and (result['gpu_dispatches']>0)==(backend!='cpu')
        path = tmp_path/'static_brian'; t.store(path); restored = NativeLIFTrainer(bundle.plan, runner=RUNNER); restored.restore(path); t = restored
        cursor += length


def cython_uniforms(seed, sequence, layer, neuron, tick, stream, rate):
    key = mix(seed ^ 0x4232504f49533031)
    for field in (sequence, 0, layer, neuron, 0, tick, stream):
        key = mix(key ^ mix((field+0x9e3779b97f4a7c15) & MASK))
    out = []; total = 0.
    for count in range(100000):
        u = uniform(key, count); out.append(1-u); total -= math.log1p(-u)
        if total >= rate: return out
    raise AssertionError('reference draw budget')


@pytest.mark.parametrize('ranks', [None, 2, 8])
def test_refractory_clamps_draw_only_reached_outputs(ranks):
    mpi(ranks); p, w = model(ranks=ranks, reset_draw=False)
    params = {'draw':PoissonNoise(1), 'offset':(3, 2)}
    for layer in p['state_equations']:
        layer[1] = expression('.8*r+.1*draw(offset)', params)
        layer.append([{'op':'state', 'index':2}])
    for layer in p['state_resets']: layer.append([{'op':'state', 'index':2}])
    p['noise_streams'] = [2, 2]; p['refractory'] = [{'steps':3, 'clamp':[0]}]*2
    initial = np.array([[.7, 1.1, 2., .2, .9, 1.2, 1.4, 0., 1.]])
    x = np.array([.25, -.5, 1.3, .125, .4])[None, :, None]
    result = NativeLIFTrainer(p, weights=w, runner=RUNNER).gradients(x, [0], initial=initial, noise_sequence=9)
    live = initial[0].copy(); entries = []; spikes = []
    vslots = [0, 3, 4]; rslots = [1, 5, 6]; cslots = [2, 7, 8]
    for tick in range(5):
        old = live.copy(); active = old[cslots] == 0.
        live[cslots] = np.maximum(old[cslots]-1, 0.)
        for j in range(3):
            l, neuron = int(j > 0), max(j-1, 0)
            if active[j]:
                rate = w[3][0]*old[rslots[j]]; count = draw(rate, p['seed'], 9, 0, l, neuron, tick)
                entries.append((l, neuron, 0, tick, count, rate)); live[vslots[j]] = .4*old[vslots[j]]+w[3][1]*count
            count = draw(w[3][2], p['seed'], 9, 0, l, neuron, tick, 1)
            entries.append((l, neuron, 1, tick, count, w[3][2])); live[rslots[j]] = .8*old[rslots[j]]+.1*count
        hard = (live[vslots] > .6) & active; spikes.append(hard.copy())
        if active[0] and not hard[0]: live[0] += w[0][0]*x[0, tick, 0]+w[2][0]*hard[1:].sum()
        for j in (1, 2):
            if active[j] and not hard[j]: live[vslots[j]] += w[1][j-1]*hard[0]
        for j in range(3):
            if hard[j]:
                live[vslots[j]] -= .6; live[rslots[j]] = .9*live[rslots[j]]+.1*w[3][2]; live[cslots[j]] = 2.
    np.testing.assert_array_equal(result['spikes'][0], spikes)
    np.testing.assert_allclose(result['final_state'][0], live, rtol=2e-14, atol=2e-14)
    actual = [(e['identity']['site']['domain'], e['identity']['site']['entity'], e['identity']['site']['stream'],
               e['identity']['instant'], e['count'], e['rate']) for e in result['poisson_state']['entries']]
    assert sorted(actual) == sorted(entries)
    np.testing.assert_array_equal(np.asarray(result['initial_state_gradients'])[:, cslots], 0.)


@pytest.mark.parametrize('ranks', [None, 2, 8])
@pytest.mark.parametrize('scale', [0., 1.2])
def test_nested_sites_and_duplicate_inner_consumer(ranks, scale):
    mpi(ranks); p, w = model(ranks=ranks, reset_draw=False); w[3][0] = scale
    params = {'inner':PoissonNoise(0), 'outer':PoissonNoise(1), 'scale':(3, 0)}
    for layer in p['state_equations']:
        layer[0] = expression('.4*v+.8*outer(.4+2.*inner(scale*r))+.05*inner(scale*r)', params)
    p['noise_streams'] = [2, 2]; x, y, initial = fixture(1); x = x[:, :3]
    def physical(force=None):
        live = initial[0].copy(); events = []; records = []
        for tick in range(3):
            old = live.copy(); inner = []; outer = []
            for j in range(3):
                l, entity = int(j>0), max(j-1, 0); rate = scale*old[[1, 4, 5][j]]
                a = 1 if force == (tick, j) else draw(rate, p['seed'], 9, 0, l, entity, tick)
                bcount = draw(.4+2*a, p['seed'], 9, 0, l, entity, tick, 1)
                inner.append(a); outer.append(bcount); records.append((tick, j, rate, a, bcount, old[[1, 4, 5][j]]))
            live[[0, 2, 3]] = .4*old[[0, 2, 3]]+.8*np.array(outer)+.05*np.array(inner)
            live[[1, 4, 5]] = .8*old[[1, 4, 5]]+.2*w[3][2]
            event = live[[0, 2, 3]] > .6; events.append(event.copy())
            live[0] += w[0][0]*x[0, tick, 0]+w[2][0]*event[1:].sum()
            live[2:4] += np.array(w[1])*event[0]
            live[[0, 2, 3]] -= .6*event
            live[[1, 4, 5]] += event*(-.1*live[[1, 4, 5]]+.1*w[3][2])
        logits = np.array(events)[:, 1:].mean(0)*p['logit_scale']; maximum = logits.max()
        loss = maximum+np.log(np.exp(logits-maximum).sum())-logits[0]
        return loss, live, events, records
    loss, final, events, records = physical()
    result = NativeLIFTrainer(p, weights=w, runner=RUNNER).gradients(x, y, initial=initial, noise_sequence=9)
    assert result['loss'] == pytest.approx(loss, abs=2e-14)
    np.testing.assert_allclose(result['final_state'][0], final, rtol=2e-14, atol=2e-14)
    np.testing.assert_array_equal(result['spikes'][0], events)
    assert len(result['poisson_state']['entries']) == 18
    if scale:
        expected = sum(loss*(a/rate-1)*source for tick, j, rate, a, bcount, source in records)
    else:
        expected = sum(source*(physical((tick, j))[0]-loss) for tick, j, rate, a, bcount, source in records)
    assert result['gradients'][3][0] == pytest.approx(expected, rel=3e-13, abs=3e-14)
