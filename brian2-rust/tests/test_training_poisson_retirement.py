"""Explicit retirement preserves trajectories/VJPs and rejects lost-history replay."""
import copy
import json
import subprocess
import sys
from pathlib import Path

import brian2 as b
import numpy as np
import pytest

from brian2_rust import BatchTimedArray, NativeLIFTrainer, lower_brian_dynamic_training
from test_native_training import RUNNER
from test_training_integer_ir import engine
from test_training_poisson_shared import model as shared_model
from test_training_poisson_static import model as static_model, fixture
from test_training_poisson_zero_vjp import mpi
from test_training_poisson_checkpoint_gpu import snapshot
from test_training_delays import cython_cache


def equivalent(actual, expected):
    for field in ('loss', 'final_state', 'spikes', 'gradients', 'initial_state_gradients'):
        if actual.get(field) is not None:
            if field == 'gradients':
                assert len(actual[field]) == len(expected[field])
                for left, right in zip(actual[field], expected[field]):
                    np.testing.assert_allclose(left, right, rtol=5e-5, atol=6e-6, err_msg=field)
            else:
                np.testing.assert_allclose(actual[field], expected[field], rtol=5e-5, atol=6e-6, err_msg=field)


@pytest.mark.parametrize('kind', ['static', 'dynamic'])
@pytest.mark.parametrize('ranks', [None, 2, 8])
@pytest.mark.parametrize('zero', [False, True])
def test_long_carry_retirement_bounds_memory_and_preserves_all_vjps(engine, kind, ranks, zero):
    mpi(ranks)
    if kind == 'static':
        plan, weights = static_model(window=2, ranks=ranks)
        x, labels, initial = fixture(2)
        start = dict(initial=initial, noise_sequence=9)
        if zero: weights[3][0] = 0.
    else:
        plan, weights = shared_model(window=2, ranks=ranks)
        x = np.zeros((2, 2, 1)); labels = [0, 1]; start = dict(noise_sequence=9)
        if zero: weights[0][0] = 0.
    plan.update(backend=engine, trainable=[False]*len(weights))
    actual = NativeLIFTrainer(plan, runner=RUNNER, weights=weights)
    reference = NativeLIFTrainer(plan, runner=RUNNER, weights=weights)
    for phase in range(12):
        kw = start if phase == 0 else dict(initial='carry')
        equivalent(actual.step(x, labels, **kw), reference.step(x, labels, **kw))
        before = snapshot(actual)
        result = actual.retire_poisson_history()
        assert result['poisson_state']['entries'] == []
        assert len(result['poisson_state']['continuation']['sha256']) == 64
        after = snapshot(actual)
        assert after[:3] == before[:3] and after[4:] == before[4:]
        assert reference.poisson_state['entries']
        before = snapshot(actual)
        equivalent(actual.gradients(x, labels, initial='carry'), reference.gradients(x, labels, initial='carry'))
        assert snapshot(actual) == before
    assert len(reference.poisson_state['entries']) >= 48


@pytest.mark.parametrize('ranks', [None, 2, 8])
def test_wrapped_emissions_and_delayed_aliases_retain_saved_rates(engine, ranks):
    mpi(ranks); plan, weights = shared_model(window=2, ranks=ranks)
    plan.update(backend=engine, trainable=[False])
    plan['dynamic']['actions'].insert(2, copy.deepcopy(plan['dynamic']['actions'][1]))
    for delay, action in zip((0, 2, 3), plan['dynamic']['actions'][:3]):
        action.update(event_noise={'delay': delay}, trigger={'external': True, 'index': 0})
    actual = NativeLIFTrainer(plan, runner=RUNNER, weights=weights)
    reference = NativeLIFTrainer(plan, runner=RUNNER, weights=weights)
    x = np.ones((2, 1, 1)); labels = [0, 1]
    for phase in range(9):
        kw = dict(noise_sequence=9) if phase == 0 else dict(initial='carry')
        equivalent(actual.step(x, labels, **kw), reference.step(x, labels, **kw))
        saved = copy.deepcopy(actual.poisson_state['entries'])
        result = actual.retire_poisson_history()
        # All aliases have one site. The delay-two negative emission at tick
        # zero must survive for the delay-three consumer at tick one.
        floor = actual.clock_tick - 3
        expected = [entry for entry in saved if
                    (entry['identity']['instant'] if entry['identity']['instant'] < 2**63
                     else entry['identity']['instant']-2**64) >= floor]
        assert result['poisson_state']['entries'] == expected
        assert len(expected) <= 6
        equivalent(actual.gradients(x, labels, initial='carry'), reference.gradients(x, labels, initial='carry'))


def test_pending_alias_and_unknown_imported_sites_remain_detached(engine):
    plan, weights = shared_model(); plan.update(backend=engine, trainable=[False])
    for action in plan['dynamic']['actions'][:2]:
        action.update(event_noise={'delay': 0, 'pending': 73}, trigger={'external': True, 'index': 0})
    actual = NativeLIFTrainer(plan, runner=RUNNER, weights=weights)
    x = np.ones((1, 1, 1)); actual.step(x, [0], noise_sequence=9)
    # A valid caller-supplied orphan is deliberately outside the current graph.
    actual.poisson_state['entries'].append(dict(identity=dict(site=dict(domain=999, entity=0,
        stream=0, kind=0, pending=0), batch=0, instant=0), count=0, rate=0.))
    actual.poisson_state['entries'].sort(key=lambda e: (e['identity']['site']['domain'],
        e['identity']['site']['entity'], e['identity']['site']['stream'], e['identity']['site']['kind'],
        e['identity']['site']['pending'], e['identity']['batch'], e['identity']['instant']))
    before = copy.deepcopy(actual.poisson_state['entries'])
    actual.retire_poisson_history(); assert actual.poisson_state['entries'] == before
    actual.state['weights'][0][0] = -3.  # A cache hit must still skip the invalid new rate.
    result = actual.gradients(x, [0], initial='carry')
    assert result['poisson_state']['entries'] == before
    assert result['gradients'][0][0] == 0.


@pytest.mark.parametrize('issue', ['tick', 'live', 'clock', 'digest', 'missing_initial'])
def test_retirement_rewind_or_corruption_rejects_atomically(engine, issue):
    plan, weights = shared_model(); plan.update(backend=engine, trainable=[False])
    actual = NativeLIFTrainer(plan, runner=RUNNER, weights=weights)
    x = np.zeros((1, 2, 1)); actual.step(x, [0], noise_sequence=9); actual.retire_poisson_history()
    before = snapshot(actual)
    kwargs = dict(initial=copy.deepcopy(actual.neuron_state), start_tick=actual.clock_tick,
                  noise_sequence=9, poisson_state=copy.deepcopy(actual.poisson_state))
    if issue == 'tick': kwargs['start_tick'] -= 1
    elif issue == 'live': kwargs['initial'][0][0] += .125
    elif issue == 'clock': kwargs['clock_state'] = dict(next_tick=2, start=0., initial_calls=[0], calls=[2], ticks=[2], visits=2)
    elif issue == 'digest': kwargs['poisson_state']['continuation']['sha256'] = '0'*64
    else: kwargs['initial'] = None
    with pytest.raises(ValueError, match='retired|retirement'): actual.step(x, [0], **kwargs)
    assert snapshot(actual) == before
    actual.step(x, [0], initial='carry')


def test_retired_checkpoint_new_process_and_older_checkpoint_rewind(engine, tmp_path):
    plan, weights = shared_model(); plan.update(backend=engine, trainable=[False])
    actual = NativeLIFTrainer(plan, runner=RUNNER, weights=weights)
    x = np.zeros((1, 2, 1)); actual.step(x, [0], noise_sequence=9)
    old = tmp_path/'before.json'; new = tmp_path/'after.json'; actual.store(old)
    actual.retire_poisson_history(); actual.store(new)
    expected = actual.gradients(x, [0], initial='carry')
    code = '''import json,sys
import numpy as np
from brian2_rust import NativeLIFTrainer
payload=json.loads(json.loads(open(sys.argv[1]).read())['payload'])
t=NativeLIFTrainer(payload['plan'],runner=sys.argv[2],weights=payload['state']['weights'])
t.restore(sys.argv[1]);r=t.gradients(np.zeros((1,2,1)),[0],initial='carry')
print(json.dumps(r))
'''
    python = Path(sys.executable)
    child = subprocess.run([str(python), '-c', code, str(new), str(RUNNER)], capture_output=True, text=True, check=True)
    equivalent(json.loads(child.stdout), expected)
    restored = NativeLIFTrainer(plan, runner=RUNNER, weights=weights); restored.restore(old)
    assert 'continuation' not in restored.poisson_state
    result = restored.gradients(x, [0], initial=restored.neuron_state, start_tick=0,
                                noise_sequence=9, poisson_state=restored.poisson_state)
    assert result['poisson_state']['entries']


@pytest.mark.parametrize('fast', [False, True])
def test_async_cached_fields_survive_draw_retirement(engine, fast):
    b.set_device('runtime'); b.start_scope(); dt = .2*b.ms
    source = b.SpikeGeneratorGroup(2, [], []*b.ms, dt=dt, name='retire_clock_input')
    groups = []
    for k in range(2):
        g = b.NeuronGroup(2, 'dv/dt=-v/ms:1\nq=.1*poisson(1.25)+.1*v:1 (constant over dt)',
            threshold='v>.6+.1*q', reset='v-=.3+.02*q', method='euler', dt=dt, name=f'retire_clock_{k}')
        g.v = [.9, .8]; g.q = [.2, .3]
        g.subexpression_updater._clock = b.Clock(dt=(.4 if k == 0 else .1 if fast else .2)*b.ms)
        groups.append(g)
    bundle = lower_brian_dynamic_training(b.Network(source, *groups), input_group=source, layers=groups,
                                        backend=engine, detach_reset=False)
    bundle.plan['trainable'] = [False]*len(bundle.weights)
    actual = NativeLIFTrainer(bundle.plan, runner=RUNNER, weights=bundle.weights)
    reference = NativeLIFTrainer(bundle.plan, runner=RUNNER, weights=bundle.weights)
    for phase, length in enumerate((1, 2, 1, 3)):
        x = np.zeros((1, length, 2)); kw = dict(noise_sequence=9) if phase == 0 else dict(initial='carry')
        equivalent(actual.step(x, [0], **kw), reference.step(x, [0], **kw))
        actual.retire_poisson_history(); assert actual.poisson_state['entries'] == []
        equivalent(actual.gradients(x, [0], initial='carry'), reference.gradients(x, [0], initial='carry'))


@pytest.mark.parametrize('rate', [0., 1.3])
@pytest.mark.parametrize('ranks', [None, 2])
def test_delay_migration_old_arrivals_keep_poisson_history(engine, rate, ranks, tmp_path):
    from test_training_delays import model as delay_model
    from test_training_delay_update import change
    mpi(ranks); net, source, groups, static, syn, x, _ = delay_model(order_sensitive=True)
    syn.pre.code += '\nw+=.01*poisson('+str(rate)+')'
    bundle = lower_brian_dynamic_training(net, input_group=source, layers=groups,
                                        backend=engine, mpi_ranks=ranks)
    bundle.plan['trainable'] = [False]*len(bundle.weights)
    actual = NativeLIFTrainer(bundle.plan, runner=RUNNER, weights=bundle.weights)
    reference = NativeLIFTrainer(bundle.plan, runner=RUNNER, weights=bundle.weights)
    cursor = 0
    for phase, length in enumerate((2, 1, 3, 2)):
        if phase:
            update = change(phase-1, static, syn)
            actual.update_delays(update); reference.update_delays(update)
        kw = dict(noise_sequence=9) if phase == 0 else dict(initial='carry')
        part = x[None, cursor:cursor+length]
        equivalent(actual.step(part, [0], **kw), reference.step(part, [0], **kw))
        actual.retire_poisson_history()
        filename = tmp_path/f'phase-{phase}.json'; actual.store(filename)
        restored = NativeLIFTrainer(actual.plan, runner=RUNNER, weights=bundle.weights); restored.restore(filename)
        assert restored.poisson_state == actual.poisson_state
        actual = restored; cursor += length


def test_imported_pending_exhaustion_matches_actual_cython(engine, cython_cache):
    from test_training_delays import model as delay_model
    net, source, groups, static, syn, x, _ = delay_model(warmup=.8, order_sensitive=True)
    syn.pre.code += '\nw+=.01*poisson(0.)'
    bundle = lower_brian_dynamic_training(net, input_group=source, layers=groups, backend=engine)
    bundle.plan['trainable'] = [False]*len(bundle.weights)
    assert any(path['pending'] for path in bundle.plan['dynamic']['delay_layout']['paths'])
    actual = NativeLIFTrainer(bundle.plan, runner=RUNNER, weights=bundle.weights)
    saw_pending = False; saw_removed = False
    for tick in range(min(5, len(x))):
        result = actual.step(x[None, tick:tick+1], [0], **(dict(noise_sequence=9) if tick == 0 else dict(initial='carry')))
        pending = [e for e in actual.poisson_state['entries'] if e['identity']['site']['kind'] == 2]
        saw_pending |= bool(pending)
        actual.retire_poisson_history()
        retained = [e for e in actual.poisson_state['entries'] if e['identity']['site']['kind'] == 2]
        saw_removed |= len(retained) < len(pending)
        net.run(.2*b.ms, namespace={})
        np.testing.assert_allclose(result['final_membrane'][0], np.r_[groups[0].v[:], groups[1].v[:]], rtol=5e-5, atol=6e-6)
        state = np.array(result['final_state'])[0]
        for name, cells in bundle.provenance['dynamic_state_layout'][syn.name].items():
            np.testing.assert_allclose(state[cells], syn.variables[name].get_value(), rtol=5e-5, atol=6e-6)
    assert saw_pending and saw_removed
    assert syn.pre.codeobj.compiled_code['run'] is not None


def test_per_sample_rate_table_update_rebinds_retired_continuation(engine):
    b.set_device('runtime'); b.start_scope(); dt = .2*b.ms
    source = b.NeuronGroup(1, 'rate:1 (shared)', threshold='False', reset='', dt=dt)
    hidden = b.NeuronGroup(1, 'dv/dt=-v/ms:1', threshold='v>.5', reset='v-=.5', method='euler', dt=dt)
    output = b.NeuronGroup(2, 'dv/dt=-v/ms:1\nrate:1 (linked)\nk:integer',
        threshold='v>.5', reset='v-=.5', method='euler', dt=dt)
    output.rate = b.linked_var(source, 'rate'); output.run_regularly('k=poisson(rate);v+=k')
    bundle = lower_brian_dynamic_training(b.Network(source, hidden, output), input_group=source,
        layers=[hidden, output], external_state_inputs={'rate': BatchTimedArray([[1.3], [0.]], dt=dt)}, backend=engine)
    bundle.plan['trainable'] = [False]*len(bundle.weights)
    actual = NativeLIFTrainer(bundle.plan, runner=RUNNER, weights=bundle.weights)
    reference = NativeLIFTrainer(bundle.plan, runner=RUNNER, weights=bundle.weights)
    x = np.zeros((2, 2, 1)); labels = [0, 1]
    equivalent(actual.step(x, labels, noise_sequence=9), reference.step(x, labels, noise_sequence=9))
    actual.retire_poisson_history()
    descriptor = bundle.provenance['external_state_inputs']['sources']['rate']
    for values in (np.array([[0.], [2.1]]), np.array([[1.9], [0.]])):
        actual.update_external_state_input(descriptor, values)
        reference.update_external_state_input(descriptor, values)
        equivalent(actual.step(x, labels, initial='carry'), reference.step(x, labels, initial='carry'))
        actual.retire_poisson_history()
        equivalent(actual.gradients(x, labels, initial='carry'), reference.gradients(x, labels, initial='carry'))


def test_structural_mask_updates_rebind_retired_physical_state(engine):
    from test_training_delays import model as delay_model
    net, source, groups, static, syn, x, _ = delay_model(order_sensitive=True)
    syn.pre.code += '\nw+=.01*poisson(1.3)'
    bundle = lower_brian_dynamic_training(net, input_group=source, layers=groups, backend=engine)
    bundle.plan['trainable'] = [False]*len(bundle.weights)
    actual = NativeLIFTrainer(bundle.plan, runner=RUNNER, weights=bundle.weights)
    reference = NativeLIFTrainer(bundle.plan, runner=RUNNER, weights=bundle.weights)
    part = x[None, :2]; equivalent(actual.step(part, [0], noise_sequence=9), reference.step(part, [0], noise_sequence=9))
    actual.retire_poisson_history()
    bank, index = bundle.plan['dynamic']['migration']['controlled_masks'][0]
    for phase, active in enumerate((0., 1.)):
        masks = copy.deepcopy(actual.plan['masks']); masks[bank][index] = active
        actual.update_mask(masks, growth_weight=.25); reference.update_mask(masks, growth_weight=.25)
        part = x[None, 2+phase:3+phase]
        equivalent(actual.step(part, [0], initial='carry'), reference.step(part, [0], initial='carry'))
        actual.retire_poisson_history()


@pytest.mark.parametrize('kind', ['static', 'dynamic'])
def test_retirement_combined_budget_rejects_before_draw_validation(engine, kind):
    if kind == 'static':
        plan, weights = static_model()
        x, labels, initial = fixture(1)
        start = dict(initial=initial, noise_sequence=9)
        nodes = sum(len(program) for groups in (plan['state_equations'], plan['state_resets'])
                    for group in groups for program in group)
        budget = max(8192, nodes*1024)
    else:
        plan, weights = shared_model()
        x = np.zeros((1, 1, 1)); labels = [0]; start = dict(noise_sequence=9)
        budget = 8192
    plan.update(backend=engine, trainable=[False]*len(weights))
    actual = NativeLIFTrainer(plan, runner=RUNNER, weights=weights)
    actual.step(x, labels, **start)
    limit = actual.plan['max_tape_bytes']
    # Each reserve fits individually, but model/cache/state/working sets do not
    # fit together. Unknown zero-rate identities are valid retained records.
    for entity in range(budget//768-1-len(actual.poisson_state['entries'])):
        actual.poisson_state['entries'].append(dict(identity=dict(site=dict(domain=999, entity=entity,
            stream=0, kind=0, pending=0), batch=0, instant=0), count=0, rate=0.))
    assert len(actual.poisson_state['entries'])*768 < budget
    actual.plan['max_tape_bytes'] = budget
    actual.poisson_state['entries'][-1]['count'] = 1  # Auth would reject zero-rate count one.
    before = snapshot(actual)
    with pytest.raises(ValueError, match='Poisson retirement workspace budget'):
        actual.retire_poisson_history()
    assert snapshot(actual) == before
    actual.plan['max_tape_bytes'] = limit
    actual.poisson_state['entries'][-1]['count'] = 0
    result = actual.retire_poisson_history()
    assert 0 < result['tape_bytes'] <= limit
    actual.step(x, labels, initial='carry')
