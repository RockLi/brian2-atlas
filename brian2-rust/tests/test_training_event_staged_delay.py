"""Shared stage delay latches: original NumPy runs, all VJPs and recovery."""
import copy
import brian2 as b
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer, lower_brian_dynamic_training
from test_native_training import RUNNER
from test_training_integer_ir import engine
from test_training_poisson_zero_vjp import mpi
from test_training_event_staged_callbacks import CODES, model, reference


def delay_model(kind, ranks, window, delay, backend, discard=True):
    net, source, groups, syn, dt, _, x = model(kind, True, discard, ranks, window, delay, backend)
    syn.pre.code = CODES[kind] + ';h+=.1*delay/ms;delay+=dt'
    bundle = lower_brian_dynamic_training(net, input_group=source, layers=groups,
        backend=backend, mpi_ranks=ranks, tbptt_window=window, detach_reset=False,
        trainable_synapse_parameters={syn.name: ['h', 'a', 'gain']})
    return net, source, groups, syn, dt, bundle, x


def compare_original(result, bundle, groups, syn):
    z = np.asarray(result['final_state'])[0]
    for group in groups:
        for name, slots in bundle.provenance['neuron_state_layout'][group.name].items():
            np.testing.assert_allclose(z[slots], group.variables[name].get_value(), rtol=7e-5, atol=7e-6)
    for name, slots in bundle.provenance['dynamic_state_layout'][syn.name].items():
        np.testing.assert_allclose(z[slots], syn.variables[name].get_value(), rtol=7e-5, atol=7e-6)
    slots = bundle.provenance['pathway_state_layout'][syn.pre.name]['delay']
    np.testing.assert_allclose(z[slots], np.asarray(syn.pre.delay[:]), rtol=7e-5, atol=7e-10)


@pytest.mark.parametrize('kind', list(CODES))
@pytest.mark.parametrize('ranks', [None, 2])
@pytest.mark.parametrize('delay', [0, 1])
@pytest.mark.parametrize('discard', [False, True])
def test_event_written_staged_delay_original_carry_and_checkpoint(engine, kind, ranks, delay, discard, tmp_path, monkeypatch):
    mpi(ranks)
    net, _, groups, syn, dt, bundle, x = delay_model(kind, ranks, None, delay, engine, discard)
    bundle.plan['trainable'] = [False] * len(bundle.weights)
    trainer = NativeLIFTrainer(bundle.plan, weights=bundle.weights, runner=RUNNER)
    pending = {}
    monitor = b.SpikeMonitor(groups[1]); net.add(monitor); seen = 0
    # Separate calls expose changes to the next run's emission route. Queued
    # arrivals retain their previous route, across checkpoint and API changes.
    for tick in range(4):
        if tick == 2:
            trainer.update_delays({syn.pre.name: 0.})
            syn.delay = 0. * dt
        calls = []
        latched = np.floor(np.asarray(syn.pre.delay[:])/float(dt)+.5).astype(int)
        if kind == 'random_temporary':
            from test_training_event_noise import draw
            arrivals = list(pending.pop(tick, []))
            domain = bundle.provenance['scheduled_noise_domains'][syn.pre.name]
            def replay(*shape):
                if not calls:
                    emitted = set(np.asarray(monitor.i)[seen:])
                    arrivals.extend((edge, tick) for edge in range(2)
                        if latched[edge] == 0 and edge in emitted)
                    np.testing.assert_array_equal(syn.pre.queue.peek(), [edge for edge, _ in arrivals])
                stream = len(calls); calls.append(stream)
                assert stream < 2 and shape == (len(arrivals),)
                return np.array([draw('randn', bundle.plan['seed'], trainer.noise_sequence,
                    0, domain, edge, emission, 0, stream) for edge, emission in arrivals])
            monkeypatch.setattr(np.random, 'randn', replay)
        result = trainer.step(x[:, tick:tick+1], [0], initial='carry' if tick else None)
        net.run(dt, namespace={})
        if kind == 'random_temporary':
            assert len(calls) in (0, 2)
            # Track emissions with the original Brian source spikes and the
            # delay latched at this run's entry, independently of Native routes.
            emitted = set(np.asarray(monitor.i)[seen:])
            for edge in range(2):
                if latched[edge] > 0 and edge in emitted:
                    pending.setdefault(tick+int(latched[edge]), []).append((edge, tick))
        seen = len(monitor.i)
        compare_original(result, bundle, groups, syn)
        checkpoint = tmp_path / ('checkpoint-' + str(tick))
        trainer.store(checkpoint)
        restored = NativeLIFTrainer(trainer.plan, runner=RUNNER)
        restored.restore(checkpoint)
        assert (restored.neuron_state, restored.clock_tick, restored.noise_sequence) == (trainer.neuron_state, trainer.clock_tick, trainer.noise_sequence)
        paths = restored.plan['dynamic']['delay_layout']['paths']
        rows = [p for p in paths if p['name'] in bundle.provenance['event_callback_stage_groups'][syn.pre.name]]
        assert all([e['delay_state'] for e in p['edges']] == [e['delay_state'] for e in rows[0]['edges']] for p in rows)
        trainer = restored


@pytest.mark.parametrize('kind', list(CODES))
@pytest.mark.parametrize('ranks', [None, 2])
@pytest.mark.parametrize('window', [None, 2])
@pytest.mark.parametrize('delay', [0, 1])
def test_staged_delay_all_parameter_and_initial_vjps(engine, kind, ranks, window, delay, monkeypatch):
    mpi(ranks)
    net, _, groups, syn, dt, bundle, x = delay_model(kind, ranks, window, delay, engine)
    actual = NativeLIFTrainer(bundle.plan, weights=bundle.weights, runner=RUNNER).gradients(x, [0], **({'noise_sequence': 9} if kind == 'random_temporary' else {}))
    loss, z, spikes, anchors = reference(bundle, groups, syn, kind, True, bundle.weights, event_delay=True)
    original_queues = {slot for name in bundle.provenance['event_callback_stage_groups'][syn.pre.name]
        for row in bundle.provenance['delay_queues'][name]['new'] for slot in row['states']}
    physical = sorted(set(range(len(z))) - original_queues)
    final = np.asarray(actual['final_state'])[0]
    np.testing.assert_allclose(final[physical], z[physical], rtol=7e-5, atol=7e-6)
    # Queue addresses may move during the native boundary rebuild. Verify the
    # complete actual route representation against the independent histories.
    for path in actual['updated_dynamic']['delay_layout']['paths']:
        old = {row['edge']: row for row in bundle.provenance['delay_queues'][path['name']]['new']}
        assert not path['pending']
        for route in path['routes']:
            assert final[route['selection']] == 1.
            np.testing.assert_array_equal(final[route['states']], z[old[route['edge']]['states']])
            if route.get('zero_gate') is not None:
                assert final[route['zero_gate']] == spikes[-1, 2+route['edge']]
    np.testing.assert_array_equal(final[actual['updated_dynamic']['delay_layout']['free_cells']], 0.)
    np.testing.assert_array_equal(actual['spikes'][0], spikes)
    assert actual['loss'] == pytest.approx(loss, abs=7e-6)
    for bank, row in enumerate(bundle.weights):
        for index in range(len(row)):
            hi = copy.deepcopy(bundle.weights); lo = copy.deepcopy(bundle.weights)
            hi[bank][index] += 1e-6; lo[bank][index] -= 1e-6
            fd = (reference(bundle, groups, syn, kind, True, hi, anchors=anchors, event_delay=True)[0]
                - reference(bundle, groups, syn, kind, True, lo, anchors=anchors, event_delay=True)[0]) / 2e-6
            assert actual['gradients'][bank][index] == pytest.approx(fd, rel=9e-4, abs=9e-6)
    for index in range(len(bundle.initial_state)):
        hi = np.array(bundle.initial_state); lo = hi.copy()
        hi[index] += 1e-6; lo[index] -= 1e-6
        fd = (reference(bundle, groups, syn, kind, True, bundle.weights, initial=hi, anchors=anchors, event_delay=True)[0]
            - reference(bundle, groups, syn, kind, True, bundle.weights, initial=lo, anchors=anchors, event_delay=True)[0]) / 2e-6
        assert actual['initial_state_gradients'][0][index] == pytest.approx(fd, rel=9e-4, abs=9e-6)
    assert (actual['gpu_dispatches'] > 0) == (engine != 'cpu')
    if kind == 'random_temporary':
        from test_training_event_noise import draw
        domain = bundle.provenance['scheduled_noise_domains'][syn.pre.name]
        draws = []
        for tick in range(4):
            emission = tick - delay
            edges = [edge for edge in range(2) if emission >= 0 and spikes[emission, 2+edge]]
            if edges:
                draws.extend(np.array([draw('randn', bundle.plan['seed'], 9, 0, domain, edge, emission, 0, stream)
                    for edge in edges]) for stream in (0, 1))
        values = iter(draws)
        def replay(*shape):
            value = next(values)
            assert shape == (len(value),)
            return value.copy()
        monkeypatch.setattr(np.random, 'randn', replay)
        net.run(4*dt, namespace={})
        assert next(values, None) is None
    else:
        net.run(4*dt, namespace={})
    compare_original(actual, bundle, groups, syn)


@pytest.mark.parametrize('issue', ['origin', 'gap', 'cell', 'source', 'plain'])
def test_native_rejects_unrelated_delay_aliases_atomically(issue):
    _, _, _, syn, _, bundle, x = delay_model('reread', None, None, 0, 'cpu')
    plan = copy.deepcopy(bundle.plan)
    original, stage = plan['dynamic']['delay_layout']['paths'][-2:]
    assert stage['name'] == syn.pre.name + '::numpy-stage:1'
    if issue == 'origin': stage['name'] = 'missing::numpy-stage:1'
    elif issue == 'gap': stage['name'] = syn.pre.name + '::numpy-stage:2'
    elif issue == 'cell': stage['edges'][0]['delay_state'] = original['edges'][1]['delay_state']
    elif issue == 'source': stage['edges'][0]['source']['index'] = original['edges'][1]['source']['index']
    else: stage['name'] = 'unrelated'
    trainer = NativeLIFTrainer(plan, weights=bundle.weights, runner=RUNNER)
    before = copy.deepcopy((trainer.plan, trainer.state, trainer.neuron_state, trainer.clock_tick))
    with pytest.raises(ValueError, match='NumPy delay stage|pathway delay storage'):
        trainer.step(x, [0])
    assert (trainer.plan, trainer.state, trainer.neuron_state, trainer.clock_tick) == before
