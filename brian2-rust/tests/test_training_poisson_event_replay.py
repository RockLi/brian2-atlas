"""Independent Poisson arrival bins, checkpoints and actual Cython queues.

The oracle implements the physical equations and a Python FIFO. It never
interprets native programs/actions or reads native sample counts to refill
Cython's RNG. Both native profiles must agree with these independently keyed
observations and Cython's original Poisson implementation.
"""
import copy
import math
import json
import os
from pathlib import Path
from unittest.mock import patch

import brian2 as b
import numpy as np
import pytest

from brian2_rust import NativeLIFTrainer, lower_brian_dynamic_training
from test_native_training import RUNNER
from test_training_integer_ir import engine
from test_training_linked import cython_cache
from test_training_poisson_core import MASK, mix, uniform
from test_training_poisson_zero_vjp import mpi


def observation(seed, sequence, domain, edge, emitted, pending, rate, *, clock=False):
    value = mix(seed ^ 0x4232504F49533031)
    kind, instant = (0, emitted) if clock else (2, pending) if pending else (1, emitted & MASK)
    for field in (sequence, 0, domain, edge, kind, instant, 0):
        value = mix(value ^ mix((field + 0x9E3779B97F4A7C15) & MASK))
    total, draws = 0., []
    for j in range(10000):
        u = uniform(value, j)
        draws.append(1. - u)  # Cython multiplies U; native sums -log(1-U).
        total -= math.log1p(-u)
        if total >= rate:
            identity = dict(site=dict(domain=domain, entity=edge, stream=0,
                                     kind=kind, pending=pending or 0),
                            batch=0, instant=instant)
            return dict(identity=identity, count=j, rate=rate), draws
    raise AssertionError('independent small-rate draw budget exceeded')


def identity(entry):
    row = entry['identity']; site = row['site']
    return tuple(site[k] for k in ('domain', 'entity', 'stream', 'kind', 'pending')) + (row['batch'], row['instant'])


def model(warm, variable, event_driven, post_first, backend, ranks):
    b.set_device('runtime'); b.start_scope(); b.prefs.codegen.target = 'cython'
    dt = .2*b.ms
    x = np.ones((14, 2)); ticks, ids = np.nonzero(x)
    inp = b.SpikeGeneratorGroup(2, ids, ticks*dt, dt=dt, name='poisson_event_input')
    a = b.NeuronGroup(2, 'dv/dt=(.7-v)/ms:1', threshold='v>.6', reset='v-=.4',
                      method='euler', dt=dt, name='poisson_event_a')
    c = b.NeuronGroup(2, 'dv/dt=(.5-v)/ms:1', threshold='v>.6', reset='v-=.4',
                      method='euler', dt=dt, name='poisson_event_c')
    a.v = [1.2, .95]; c.v = [.8, .4]
    drive = b.Synapses(inp, a, 'w:1', on_pre='v_post+=w', dt=dt, name='poisson_event_drive')
    drive.connect(j='i'); drive.w = .18
    pre = 'sample=poisson(1.25)\nw=.95*w+.015*sample\nv_post+=.7*w+.01*a\na+=.03*sample\ncount+=1\ntotal+=sample'
    if variable: pre += '\ndelay=(count%3)*dt'
    flag = 'event-driven' if event_driven else 'clock-driven'
    s = b.Synapses(a, c, 'w:1\nda/dt=-a/(2*ms):1 ('+flag+')\ncount:integer\n'
                  'sample:integer\npost_sample:integer\ntotal:integer', on_pre=pre,
                  on_post='post_sample=poisson(2.5)\nw+=.003*post_sample\na*=.9\ntotal+=post_sample',
                  method='exact', dt=dt, name='poisson_event_s')
    s.connect(i=[0, 1, 0, 1], j=[0, 1, 1, 0]); s.w = [.15, .18, .21, .12]
    s.a = [.1, .2, .3, .4]; s.pre.delay = .6*b.ms; s.post.delay = .2*b.ms
    if post_first: s.post.order = -2
    net = b.Network(inp, a, c, drive, s)
    if warm:
        b.seed(991); net.run(warm*dt, namespace={}); x = x[warm:]
    bundle = lower_brian_dynamic_training(net, input_group=inp, layers=[a, c],
                                        backend=backend, mpi_ranks=ranks, detach_reset=False)
    bundle.plan['trainable'] = [False]*len(bundle.weights)
    return net, [a, c], s, x, bundle


class ArrivalOracle:
    def __init__(self, bundle, variable, event_driven, post_first):
        self.bundle = bundle; self.variable = variable; self.event_driven = event_driven
        self.order = [1, 0] if post_first else [0, 1]
        self.z = np.asarray(bundle.initial_state, float).copy()
        self.layout = bundle.provenance['dynamic_state_layout']['poisson_event_s']
        self.voltage = sum([bundle.provenance['neuron_state_layout'][name]['v']
                            for name in ('poisson_event_a', 'poisson_event_c')], [])
        self.sources = [np.array([0, 1, 0, 1]), np.array([0, 1, 1, 0])]
        self.targets = self.sources[1]
        self.names = ['poisson_event_s_pre', 'poisson_event_s_post']
        self.domains = [bundle.provenance['scheduled_noise_domains'][name] for name in self.names]
        self.bins = [{}, {}]; self.records = {}; self.tick = 0; self.collisions = []
        self.delays = [np.full(4, 3), np.ones(4, dtype=int)]
        self.delay_slots = bundle.provenance['pathway_state_layout'].get(self.names[0], {}).get('delay')
        self.pending = 0
        for path, name in enumerate(self.names):
            for number, row in enumerate(bundle.provenance['delay_queues'][name]['pending'], 1):
                assert self.z[row['states'][0]] == (1. if row['remaining'] == 0 else 0.)
                self.bins[path].setdefault(row['remaining'], []).append((row['edge'], None, number))
                self.pending += 1

    def change_delays(self, pattern):
        self.delays = [np.array(pattern[0], dtype=int), np.array(pattern[1], dtype=int)]

    def advance(self, x):
        out = []; draws = []
        state = self.z; a, c = self.voltage[:2], self.voltage[2:]
        w, trace = self.layout['w'], self.layout['a']
        for external in x:
            tick = self.tick
            if self.variable:
                self.delays[0] = np.floor(state[self.delay_slots]/.0002+.5).astype(int)
            state[a] = .8*state[a]+.14; state[c] = .8*state[c]+.1
            hard = (state[self.voltage] > .6).astype(float); out.append(hard.copy())
            if not self.event_driven: state[trace] *= math.exp(-.0002/.002)
            state[a] += .18*external
            stamp = self.bundle.plan['clock']['origin']+tick*.0002
            for path in self.order:
                source = self.sources[path]; current = hard[:2] if path == 0 else hard[2:]
                for edge in sorted(range(4), key=lambda e: (source[e], e)):
                    if current[source[edge]]:
                        self.bins[path].setdefault(tick+int(self.delays[path][edge]), []).append((edge, tick, None))
                arrivals = self.bins[path].pop(tick, [])
                edges = [row[0] for row in arrivals]
                if len(edges) != len(set(edges)): self.collisions.append((tick, path, edges))
                for edge, emitted, pending in arrivals:
                    entry, samples = observation(self.bundle.plan['seed'], 9, self.domains[path],
                                                 edge, emitted, pending, 1.25 if path == 0 else 2.5)
                    key = identity(entry); assert key not in self.records
                    self.records[key] = entry; draws.extend(samples); count = entry['count']
                    if self.event_driven:
                        last = self.layout['lastupdate'][edge]
                        state[trace[edge]] *= math.exp(-(stamp-state[last])/.002)
                        state[last] = stamp
                    if path == 0:
                        state[self.layout['sample'][edge]] = count
                        state[w[edge]] = .95*state[w[edge]]+.015*count
                        state[c[self.targets[edge]]] += .7*state[w[edge]]+.01*state[trace[edge]]
                        state[trace[edge]] += .03*count
                        state[self.layout['count'][edge]] += 1
                        if self.variable:
                            state[self.delay_slots[edge]] = (int(state[self.layout['count'][edge]]) % 3)*.0002
                    else:
                        state[self.layout['post_sample'][edge]] = count
                        state[w[edge]] += .003*count; state[trace[edge]] *= .9
                    state[self.layout['total'][edge]] += count
            state[self.voltage] -= .4*hard; self.tick += 1
        return np.asarray(out), draws

    def entries(self):
        return [self.records[key] for key in sorted(self.records)]


def replay_cython(net, draws, length):
    device = b.get_device(); device.rand_buffer_index[:] = 0; calls = []
    def refill(n):
        assert n == 20000 and not calls and len(draws) < n
        calls.append(n); result = np.full(n, .5); result[:len(draws)] = draws
        return result
    try:
        with patch('numpy.random.rand', refill): net.run(length*.2*b.ms, namespace={})
        assert calls == ([20000] if draws else [])
        assert device.rand_buffer_index[0] == len(draws)
    finally: device.rand_buffer_index[:] = 0


def compare_pending_queues(syn, oracle):
    """Inspect real queues at a boundary, including duplicate arrival order."""
    pending = []
    for index, path in enumerate((syn.pre, syn.post)):
        offset, bins = path.queue._full_state()
        actual = bins[offset:]+bins[:offset]
        remaining = max(oracle.bins[index], default=oracle.tick-1)-oracle.tick+1
        expected = [[entry[0] for entry in oracle.bins[index].get(oracle.tick+j, [])]
                    for j in range(remaining)]
        while actual and not actual[-1]: actual.pop()
        while expected and not expected[-1]: expected.pop()
        assert actual == expected
        pending.append(expected)
    return pending


@pytest.mark.parametrize('ranks', [None, 2])
@pytest.mark.parametrize('warm', [0, 4])
@pytest.mark.parametrize('variable', [False, True])
@pytest.mark.parametrize('event_driven', [False, True])
@pytest.mark.parametrize('post_first', [False, True])
def test_positive_poisson_queues_match_independent_arrivals_and_cython(
        engine, ranks, warm, variable, event_driven, post_first, tmp_path):
    mpi(ranks)
    net, groups, syn, x, bundle = model(warm, variable, event_driven, post_first, engine, ranks)
    oracle = ArrivalOracle(bundle, variable, event_driven, post_first)
    if warm: assert oracle.pending > 0
    monitors = [b.SpikeMonitor(group) for group in groups]; net.add(*monitors)
    net.run(0*b.ms, namespace={})
    assert syn.pre.codeobj.compiled_code['run'] is not None
    assert syn.post.codeobj.compiled_code['run'] is not None
    trainer = NativeLIFTrainer(bundle.plan, weights=bundle.weights, runner=RUNNER)
    lengths = [1]*len(x) if variable else [3, 2, len(x)-5]
    cursor = 0; tolerance = 6e-5 if engine != 'cpu' else 4e-12; native_spikes = []; boundaries = []
    for phase, length in enumerate(lengths):
        if not variable and phase:
            pattern = ([0]*4, [0]*4) if phase == 1 else ([2, 0, 3, 1], [0, 2, 0, 3])
            cache = copy.deepcopy(trainer.poisson_state)
            trainer.update_delays({syn.pre.name: np.array(pattern[0])*.0002,
                                   syn.post.name: np.array(pattern[1])*.0002})
            assert trainer.poisson_state == cache
            syn.pre.delay = np.array(pattern[0])*.2*b.ms; syn.post.delay = np.array(pattern[1])*.2*b.ms
            oracle.change_delays(pattern)
        expected, draws = oracle.advance(x[cursor:cursor+length])
        result = trainer.step(x[None, cursor:cursor+length], [0],
                              **({'initial': 'carry'} if cursor else {'noise_sequence': 9}))
        np.testing.assert_array_equal(result['spikes'][0], expected)
        native_spikes.extend(result['spikes'][0])
        assert result['poisson_state']['entries'] == oracle.entries()
        physical = oracle.voltage + sum(oracle.layout.values(), [])
        if oracle.delay_slots: physical += oracle.delay_slots
        np.testing.assert_allclose(np.asarray(result['final_state'])[0, physical], oracle.z[physical],
                                   rtol=tolerance, atol=tolerance*.02)
        assert (result['gpu_dispatches'] > 0) == (engine != 'cpu')
        replay_cython(net, draws, length)
        pending = compare_pending_queues(syn, oracle)
        for group in groups:
            slots = bundle.provenance['neuron_state_layout'][group.name]['v']
            np.testing.assert_allclose(oracle.z[slots], group.v[:], rtol=4e-12, atol=4e-14)
        for name, slots in oracle.layout.items():
            np.testing.assert_allclose(oracle.z[slots], syn.variables[name].get_value(), rtol=4e-12, atol=4e-14)
        if oracle.delay_slots: np.testing.assert_allclose(oracle.z[oracle.delay_slots], syn.pre.delay[:]/b.second, rtol=4e-12, atol=4e-14)
        checkpoint = tmp_path/'events.json'; trainer.store(checkpoint)
        restored = NativeLIFTrainer(trainer.plan, weights=bundle.weights, runner=RUNNER)
        restored.restore(checkpoint); assert restored.poisson_state == trainer.poisson_state
        boundaries.append(dict(tick=oracle.tick, uniform_draws=len(draws),
                               records=len(oracle.records), pending=pending,
                               checkpoint_entries=result['poisson_state']['entries']))
        trainer = restored; cursor += length
    # Compare actual threshold events, not just native hard decisions.
    actual = np.zeros((len(x), 4)); origin = bundle.plan['clock']['origin']
    for layer, monitor in enumerate(monitors):
        ticks = np.rint((np.asarray(monitor.t/b.second)-origin)/.0002).astype(int)
        actual[ticks, layer*2+np.asarray(monitor.i)] = 1.
    np.testing.assert_array_equal(actual, native_spikes)
    assert oracle.records and any(e['count'] > 0 for e in oracle.entries())
    if not variable and not warm: assert oracle.collisions
    if os.environ.get('B2_EVENTS_EVIDENCE'):
        name = f'{engine}-r{ranks}-w{warm}-v{variable}-e{event_driven}-p{post_first}.json'
        (Path(os.environ['B2_EVENTS_EVIDENCE'])/name).write_text(json.dumps(
            dict(backend=engine, ranks=ranks, warm=warm, variable=variable,
                 event_driven=event_driven, post_first=post_first, imported_pending=oracle.pending,
                 collisions=oracle.collisions, boundaries=boundaries, spikes=native_spikes), indent=2)+'\n')


def assert_cached_poisson_clock(backend, fast=False, ranks=None, checkpoint=None):
    """Clock-addressed observations persist until the next cache refresh."""
    mpi(ranks); b.set_device('runtime'); b.start_scope(); b.prefs.codegen.target = 'cython'
    dt = .2*b.ms
    source = b.SpikeGeneratorGroup(2, [], []*b.ms, dt=dt, name='clock_cache_input')
    groups = []
    for k in range(2):
        g = b.NeuronGroup(2, 'dv/dt=-v/ms:1\nq=.1*poisson(1.25)+.1*v:1 (constant over dt)',
                          threshold='v>.6+.1*q', reset='v-=.3+.02*q', method='euler', dt=dt,
                          name=f'clock_cache_{k}')
        g.v = [.9, .8]; g.q = [.2, .3]
        g.subexpression_updater.when = 'before_start'
        g.subexpression_updater._clock = b.Clock(dt=(.4 if k == 0 else .1 if fast else .2)*b.ms)
        groups.append(g)
    net = b.Network(source, *groups)
    bundle = lower_brian_dynamic_training(net, input_group=source, layers=groups,
                                        backend=backend, mpi_ranks=ranks, detach_reset=False)
    bundle.plan['trainable'] = [False]*len(bundle.weights)
    domains = [bundle.provenance['regular_runner_layout'][g.subexpression_updater.name]['noise_domain'] for g in groups]
    v = np.array([[.9, .8], [.9, .8]]); q = np.array([[.2, .3], [.2, .3]])
    spikes = []; draws = []; entries = []
    # An independent .1ms lattice includes the idle fast-cache visits after
    # each main .2ms transition, including the final call boundary.
    periods = [4, 1 if fast else 2]
    for half in range(12):
        for k in range(2):
            if half % periods[k] == 0:
                for j in range(2):
                    entry, random = observation(bundle.plan['seed'], 9, domains[k], j,
                                                half//periods[k], None, 1.25, clock=True)
                    entries.append(entry); draws.extend(random); q[k, j] = .1*entry['count']+.1*v[k, j]
        if half % 2 == 0:
            v *= .8; event = v > .6+.1*q; spikes.append(event.ravel().copy())
            v -= event*(.3+.02*q)
    entries.sort(key=identity)
    trainer = NativeLIFTrainer(bundle.plan, weights=bundle.weights, runner=RUNNER)
    x = np.zeros((1, 6, 2)); full = trainer.evaluate(x, [0], noise_sequence=9)
    np.testing.assert_array_equal(full['spikes'][0], spikes)
    assert full['poisson_state']['entries'] == entries
    tolerance = 5e-5 if backend != 'cpu' else 4e-12
    for k, group in enumerate(groups):
        layout = bundle.provenance['neuron_state_layout'][group.name]
        for name, expected in [('v', v[k]), ('q', q[k])]:
            np.testing.assert_allclose(np.asarray(full['final_state'])[0, layout[name]], expected,
                                       rtol=tolerance, atol=tolerance*.02)
    if checkpoint is not None:
        head = trainer.step(x[:, :3], [0], noise_sequence=9); trainer.store(checkpoint)
        restored = NativeLIFTrainer(trainer.plan, weights=bundle.weights, runner=RUNNER)
        restored.restore(checkpoint); tail = restored.step(x[:, 3:], [0], initial='carry')
        assert tail['poisson_state'] == full['poisson_state']
        np.testing.assert_array_equal(head['spikes'][0]+tail['spikes'][0], spikes)
        np.testing.assert_allclose(tail['final_state'], full['final_state'], rtol=tolerance, atol=tolerance*.02)
    assert (full['gpu_dispatches'] > 0) == (backend != 'cpu')
    monitors = [b.SpikeMonitor(g) for g in groups]; net.add(*monitors); net.run(0*b.ms, namespace={})
    replay_cython(net, draws, 6)
    actual = np.zeros((6, 4))
    for k, (group, monitor) in enumerate(zip(groups, monitors)):
        assert group.subexpression_updater.codeobj.compiled_code['run'] is not None
        actual[np.rint(np.asarray(monitor.t/b.second)/.0002).astype(int), k*2+np.asarray(monitor.i)] = 1.
        np.testing.assert_allclose(group.v[:], v[k], rtol=4e-12, atol=4e-14)
        np.testing.assert_allclose(group.q[:], q[k], rtol=4e-12, atol=4e-14)
    np.testing.assert_array_equal(actual, spikes)
    if os.environ.get('B2_EVENTS_EVIDENCE') and checkpoint is not None:
        directory = Path(os.environ['B2_EVENTS_EVIDENCE']).parent/'clock-cases'
        directory.mkdir(exist_ok=True)
        (directory/f'{backend}-r{ranks}-fast{fast}.json').write_text(json.dumps(
            dict(backend=backend, ranks=ranks, fast=fast, periods=periods,
                 full_entries=full['poisson_state']['entries'], tail_entries=tail['poisson_state']['entries'],
                 full_state=full['final_state'], tail_state=tail['final_state'],
                 spikes=full['spikes'][0], cython_spikes=actual.tolist(),
                 uniform_draws=len(draws), cython_v=[g.v[:].tolist() for g in groups],
                 cython_q=[g.q[:].tolist() for g in groups]), indent=2)+'\n')


@pytest.mark.parametrize('ranks', [None, 2])
@pytest.mark.parametrize('fast', [False, True])
def test_cached_poisson_threshold_reset_with_distinct_clocks(engine, ranks, fast, tmp_path):
    assert_cached_poisson_clock(engine, fast, ranks, tmp_path/'clock-cache.json')
