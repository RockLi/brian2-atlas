"""Zero-rate weak derivatives from independent hard-trajectory loss oracles."""
import copy
import math
import os

import numpy as np
import pytest

from brian2_rust.training import NativeLIFTrainer
from test_native_training import RUNNER
from test_training_linked import cython_cache
from test_training_poisson_ssa import single_transform
from test_training_poisson_core import mix, MASK, uniform as counter_uniform


def hard_loss(logit, label=0):
    return np.logaddexp(0., logit) - np.where(np.asarray(label) == 0, logit, 0.)


def zero_model(code='k=draw(scale)\nv=v+amp*k', ranks=None):
    plan, weights = single_transform(code)
    plan['mpi_ranks'] = ranks
    weights[0][0] = 0.
    weights[0][1] = 1.
    return plan, weights


@pytest.mark.parametrize('length', [1, 4])
@pytest.mark.parametrize('window', [None, 1, 2])
@pytest.mark.parametrize('batch', [1, 3])
def test_zero_parameter_full_counterfactual_loss(length, window, batch):
    plan, weights = zero_model(); plan['tbptt_window'] = window
    labels = np.arange(batch) % 2
    result = NativeLIFTrainer(plan, weights=weights, runner=RUNNER).gradients(
        np.zeros((batch, length, 1)), labels)
    expected = sum((hard_loss(plan['logit_scale']*(length-t)/length, labels)-math.log(2)).mean()
                   for t in range(length))
    assert result['gradients'][0][0] == pytest.approx(expected, abs=2e-14)
    assert result['gradients'][0][1] == 0.
    assert result['loss'] == pytest.approx(math.log(2))
    np.testing.assert_array_equal(np.asarray(result['final_state'])[:, 2], 0)


@pytest.mark.parametrize('window', [None, 1, 2])
@pytest.mark.parametrize('binding', [False, True])
def test_rate_state_vjp_uses_baseline_and_tbptt(window, binding):
    plan, weights = zero_model('k=draw(scale*r)\nv=v+amp*k\nr=.8*r')
    plan['tbptt_window'] = window; weights[0][0] = 1.2
    plan['dynamic']['initial'][4] = 0.
    weights[0][2] = 0.
    if binding: plan['dynamic']['initial_parameters'][4] = [0, 2]
    result = NativeLIFTrainer(plan, weights=weights, runner=RUNNER).gradients(np.zeros((1, 4, 1)), [0])
    end = 4 if window is None else window
    expected = sum(1.2*.8**t*(hard_loss(plan['logit_scale']*(4-t)/4)-math.log(2)) for t in range(end))
    assert result['initial_state_gradients'][0][4] == pytest.approx(expected, abs=3e-14)
    assert result['gradients'][0][2] == pytest.approx(expected if binding else 0., abs=3e-14)
    assert result['gradients'][0][0] == 0.


def stream_draw(rate, seed, sequence, batch, stream):
    address = mix(seed ^ 0x4232504f49533031)
    for field in (sequence, batch, 71, 0, 0, 0, stream):
        address = mix(address ^ mix((field + 0x9e3779b97f4a7c15) & MASK))
    arrival = 0.; count = 0
    while True:
        arrival -= math.log1p(-counter_uniform(address, count))
        if arrival >= rate: return count
        count += 1


def test_nested_zero_sites_resample_live_rate_with_original_batch_keys():
    plan, weights = zero_model('k=draw(scale)\nv=amp*other(k+scale)')
    batch = 64; sequence = 2**64-7; labels = np.arange(batch) % 2
    result = NativeLIFTrainer(plan, weights=weights, runner=RUNNER).gradients(
        np.zeros((batch, 1, 1)), labels, noise_sequence=sequence)
    counts = np.array([stream_draw(1., plan['seed'], sequence, b, 1) for b in range(batch)])
    assert np.any(counts == 0) and np.any(counts > 0)
    first = hard_loss(plan['logit_scale']*(counts > 0), labels)-math.log(2)
    second = hard_loss(plan['logit_scale'], labels)-math.log(2)
    assert result['gradients'][0][0] == pytest.approx((first+second).mean(), abs=2e-14)


@pytest.mark.parametrize('code', [
    'k=draw(scale)\nv=amp*other(scale) if k > 0 else 0.',
    'k=draw(scale)\nv=amp*other(-1.) if k < 0 else 0.',
])
def test_conditionally_unreached_zero_sites_do_not_contribute(code):
    plan, weights = zero_model(code)
    result = NativeLIFTrainer(plan, weights=weights, runner=RUNNER).gradients(np.zeros((1, 1, 1)), [0])
    assert result['gradients'][0][0] == 0.


@pytest.mark.parametrize('ranks', [None, 2, 8])
def test_indirect_counterfactual_writes_and_owner_mpi(ranks):
    if ranks and os.environ.get('B2_TEST_MPI') != '1': pytest.skip('local MPI opt-in')
    plan, weights = zero_model('k=draw(scale)\nv=amp', ranks)
    # k=0 routes v to an unused rate cell; k=1 routes it to output voltage.
    action = plan['dynamic']['actions'][0]
    action['indirect'] = {'reads': {}, 'writes': {'0': {'index': {'kind': 'output', 'slot': 1}, 'tables': [[4, 0]]}}}
    result = NativeLIFTrainer(plan, weights=weights, runner=RUNNER).gradients(np.zeros((2, 1, 1)), [0, 1])
    expected = np.mean(hard_loss(plan['logit_scale'], np.array([0, 1]))-math.log(2))
    assert result['gradients'][0][0] == pytest.approx(expected, abs=2e-14)
    np.testing.assert_array_equal(np.asarray(result['final_state'])[:, 2], 0.)


@pytest.mark.parametrize('ranks', [None, 2])
def test_invalid_positive_count_branch_rolls_back(ranks):
    if ranks and os.environ.get('B2_TEST_MPI') != '1': pytest.skip('local MPI opt-in')
    plan, weights = zero_model('k=draw(scale)\nv=amp*other(-1.) if k > 0 else 0.', ranks)
    trainer = NativeLIFTrainer(plan, weights=weights, runner=RUNNER)
    before = copy.deepcopy(trainer.state)
    trainer.evaluate(np.zeros((1, 1, 1)), [0])
    with pytest.raises(ValueError, match='Poisson sample'):
        trainer.step(np.zeros((1, 1, 1)), [0])
    assert trainer.state == before and trainer.clock_tick == 0 and trainer.next_noise_sequence == 0


def test_replay_memory_reserved_before_allocation():
    plan, weights = zero_model()
    trainer = NativeLIFTrainer(plan, weights=weights, runner=RUNNER)
    inputs = np.zeros((2, 3, 1))
    forward = trainer.evaluate(inputs, [0, 1])
    reverse = trainer.gradients(inputs, [0, 1])
    assert reverse['tape_bytes'] >= 2*forward['tape_bytes']
    plan['max_tape_bytes'] = reverse['tape_bytes']-1
    with pytest.raises(ValueError, match='budget'):
        NativeLIFTrainer(plan, weights=weights, runner=RUNNER).gradients(inputs, [0, 1])


@pytest.mark.parametrize('ranks', [None, 2])
def test_replay_activates_later_event_and_preserves_checkpoint(ranks, tmp_path):
    if ranks and os.environ.get('B2_TEST_MPI') != '1': pytest.skip('local MPI opt-in')
    from brian2_rust.training_dynamic import compile_dynamic_transform, dynamic_action
    plan, weights = zero_model('k=draw(scale)\nv=amp*k', ranks)
    transform = compile_dynamic_transform('v=v+amp', states={'v': 0}, parameters={'amp': (0, 1)})
    plan['dynamic']['program_sets'].append(transform['programs'])
    # Threshold 1 sees the forced draw; its event changes output neuron 2
    # before threshold 2 in the same visit.
    event = dynamic_action(transform, [1], owner=2, program_set=1)
    event['trigger'] = {'external': False, 'index': 1}
    event['detach_trigger'] = True
    plan['dynamic']['actions'].insert(2, event)
    plan['trainable'] = [False]
    trainer = NativeLIFTrainer(plan, weights=weights, runner=RUNNER)
    trainer.step(np.zeros((1, 2, 1)), [0], noise_sequence=2**64-7)
    path = tmp_path/'checkpoint.json'; trainer.store(path)
    restored = NativeLIFTrainer(plan, runner=RUNNER); restored.restore(path)
    actual = restored.gradients(np.zeros((1, 2, 1)), [0], initial='carry')
    same = trainer.gradients(np.zeros((1, 2, 1)), [0], initial='carry')
    for key in ('gradients', 'initial_state_gradients', 'final_state', 'spikes', 'loss'):
        np.testing.assert_array_equal(actual[key], same[key])
    # A force at t=0 produces [1, 1] then [0, 1] for neuron 1/2:
    # the late output retains its voltage, so its second spike changes the loss.
    expected = hard_loss(-plan['logit_scale']/2)-math.log(2)
    assert actual['gradients'][0][0] == pytest.approx(expected, abs=2e-14)
    assert actual['final_tick'] == 4 and actual['noise_sequence'] == 2**64-7


@pytest.mark.parametrize('ranks', [None, 2])
def test_asynchronous_zero_sites_between_primary_frames(ranks):
    if ranks and os.environ.get('B2_TEST_MPI') != '1': pytest.skip('local MPI opt-in')
    plan, weights = zero_model(ranks=ranks)
    plan['dynamic']['clocks'] = {'start': 0., 'dts': [.001, .0005], 'epsilon': 1e-4, 'order': [0, 1]}
    plan['dynamic']['actions'][0]['clock'] = 1
    result = NativeLIFTrainer(plan, weights=weights, runner=RUNNER).gradients(np.zeros((1, 2, 1)), [0])
    # Draws at 0, .5, 1, 1.5 ms: respectively 2, 1, 1, 0 output spikes.
    expected = (hard_loss(plan['logit_scale'])-math.log(2)
                + 2*(hard_loss(plan['logit_scale']/2)-math.log(2)))
    assert result['gradients'][0][0] == pytest.approx(expected, abs=2e-14)


def test_brian_zero_rate_frontend_and_actual_cython_counterfactual():
    import brian2 as b
    from unittest.mock import patch
    from brian2_rust import lower_brian_dynamic_training
    b.set_device('runtime'); b.start_scope(); b.prefs.codegen.target = 'cython'
    dt = .2*b.ms
    inp = b.SpikeGeneratorGroup(1, [], []*b.ms, dt=dt, name='zero_input')
    hidden = b.NeuronGroup(1, 'dv/dt=0*Hz:1', method='euler', threshold='v>1', reset='v=0', dt=dt, name='zero_hidden')
    output = b.NeuronGroup(2, 'dv/dt=0*Hz:1\nrate:1 (constant)', method='euler', threshold='v>.8', reset='v=v', dt=dt, name='zero_output')
    output.run_regularly('v+=1.0*poisson(rate)', dt=dt, when='start', name='zero_regular')
    net = b.Network(inp, hidden, output)
    net.run(0*b.ms, namespace={}); net.store('baseline')
    bundle = lower_brian_dynamic_training(net, input_group=inp, layers=[hidden, output],
        trainable_neuron_parameters={output.name: ['rate']})
    result = NativeLIFTrainer(bundle.plan, weights=bundle.weights, runner=RUNNER).gradients(np.zeros((1, 1, 1)), [0])
    bank = next(row['bank'] for row in bundle.provenance['bindings']
                if row['object'] == output.name and row['variables'] == ['rate'])
    net.run(dt, namespace={})
    slots = bundle.provenance['neuron_state_layout'][output.name]['v']
    np.testing.assert_array_equal(np.asarray(result['final_state'])[0, slots], output.v[:])
    # Actual compiled Brian alternate trajectories: each of the two possible
    # one-count sites is tested separately, using Knuth uniforms for exactly one.
    expected = []
    for j in range(2):
        net.restore('baseline'); output.rate[j] = 1.
        device = b.get_device(); device.rand_buffer_index[:] = 0
        calls = []
        def refill(n):
            calls.append(n); values = np.full(n, .5); values[:2] = [.9, .1]; return values
        with patch('numpy.random.rand', refill): net.run(dt, namespace={})
        assert calls == [20000] and device.rand_buffer_index[0] == 2
        device.rand_buffer_index[:] = 0
        np.testing.assert_array_equal(output.v[:], np.eye(2)[j])
        logits = (np.asarray(output.v[:]) > .8)*bundle.plan['logit_scale']
        expected.append(np.logaddexp(logits[0], logits[1])-logits[0]-math.log(2))
    np.testing.assert_allclose(result['gradients'][bank], expected, rtol=1e-13, atol=1e-14)


def test_training_step_uses_weak_derivative_and_commits_baseline_state():
    plan, weights = zero_model(); plan['optimizer'].update(kind='sgd', learning_rate=.05)
    trainer = NativeLIFTrainer(plan, weights=weights, runner=RUNNER)
    result = trainer.step(np.zeros((1, 1, 1)), [0], noise_sequence=19)
    expected = hard_loss(plan['logit_scale'])-math.log(2)
    assert trainer.state['weights'][0][0] == pytest.approx(-.05*expected)
    assert result['final_state'][0][2] == 0.
    assert trainer.clock_tick == 1 and trainer.state['step'] == 1
