"""Event-written pathway delays: real Brian runs, independent queue/VJP oracle."""
import copy
import os
import numpy as np
import pytest
import brian2 as b
from brian2_rust import NativeLIFTrainer, TrainingConversionError, lower_brian_dynamic_training
from test_native_training import RUNNER
from test_training_brian_dynamic import network
from test_training_delays import cython_cache
from test_training_integer_ir import engine
from test_training_delay_update import snapshot


def model(event_driven=False, noisy=False, post_first=False, warmup=0, **options):
    net, inp, layers, static, syn, _, _, x = network(event_driven, noisy)
    b.prefs.codegen.target = 'cython'
    static.pre.delay = np.array([0, .4, .2, .6])*b.ms
    static.pre.code = 'v_post += w + .05*delay/ms\ndelay=(.08+.4*int(v_post>1.2)+.4*int(w>.9))*ms'
    syn.pre.delay = np.array([.4, .2, .6, 0])*b.ms
    syn.pre.code += '\nv_post+=.01*delay/ms\ndelay=(.08+.2*int(apre>.12)+.4*int(w>1))*ms'
    syn.post.delay = np.array([0, .6, .2, .4])*b.ms
    syn.post.code += '\napost-=.003*delay/ms\ndelay=(.08+.4*int(apost<-.07))*ms'
    if post_first:
        syn.post.order = -2
    if warmup:
        net.run(warmup*.2*b.ms, namespace={})
    bundle = lower_brian_dynamic_training(net, input_group=inp, layers=layers, **options)
    bundle.plan['trainable'] = [False]*len(bundle.weights)
    return net, layers, [static, syn], x[warmup:], bundle


def physical(bundle):
    cells = set(range(len(bundle.initial_state)))
    for path in bundle.plan['dynamic']['delay_layout']['paths']:
        for e in [*path['pending'], *path['edges']]:
            cells.difference_update(e['states'])
    return sorted(cells)


@pytest.mark.parametrize('event_driven', [False, True])
@pytest.mark.parametrize('post_first', [False, True])
@pytest.mark.parametrize('warmup', [0, 4])
def test_event_delay_changes_match_compiled_brian_across_runs(event_driven, post_first, warmup):
    net, layers, synapses, x, bundle = model(event_driven, post_first=post_first, warmup=warmup)
    trainer = NativeLIFTrainer(bundle.plan, runner=RUNNER, weights=bundle.weights)
    monitors = [b.SpikeMonitor(g) for g in layers];net.add(*monitors)
    actual = [];cursor = 0
    for length in [2, 1, 3, len(x)-6]:
        result = trainer.step(x[None, cursor:cursor+length], [0], initial='carry' if cursor else None)
        net.run(length*.2*b.ms, namespace={});actual.extend(result['spikes'][0]);cursor += length
        np.testing.assert_allclose(result['final_membrane'][0], np.r_[layers[0].v[:], layers[1].v[:]], rtol=3e-13, atol=8e-14)
        for syn in synapses:
            for name, indices in bundle.provenance['dynamic_state_layout'][syn.name].items():
                np.testing.assert_allclose(np.asarray(result['final_state'])[0, indices], syn.variables[name].get_value(), rtol=3e-13, atol=8e-14)
            for path in syn._pathways:
                indices = bundle.provenance['pathway_state_layout'][path.name]['delay']
                np.testing.assert_allclose(np.asarray(result['final_state'])[0, indices], np.asarray(path.delay[:]), rtol=3e-13, atol=8e-14)
                assert path.codeobj.compiled_code['run'] is not None
    expected = np.zeros((len(x), 4))
    for layer, monitor in enumerate(monitors):
        ticks = np.rint(monitor.t/(.2*b.ms)).astype(int)-warmup
        expected[ticks, 2*layer+np.asarray(monitor.i)] = 1
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('ranks', [None, 2, 8])
def test_batch_specific_delay_routes_and_gradients_match_independent_runs(engine, ranks, tmp_path):
    if ranks and os.environ.get('B2_TEST_MPI') != '1':
        pytest.skip('local MPI required')
    _, _, _, x, bundle = model(post_first=True)
    bundle.plan.update(backend=engine, mpi_ranks=ranks)
    xx = np.stack([x, x.copy()]);xx[1, :3] = 0;xx[1, 4] = [0, 1]
    trainer = NativeLIFTrainer(bundle.plan, runner=RUNNER, weights=bundle.weights)
    p = copy.deepcopy(bundle.plan);p.update(backend='cpu', mpi_ranks=None)
    singles = [NativeLIFTrainer(p, runner=RUNNER, weights=bundle.weights) for _ in range(2)]
    indices = physical(bundle);cursor = 0;different = False
    for length in [2, 1, 3, 6]:
        tail = xx[:, cursor:cursor+length]
        before = snapshot(trainer)
        actual = trainer.gradients(tail, [0, 1], initial='carry' if cursor else None)
        assert snapshot(trainer)[:-1] == before[:-1]
        assert actual['backend'] == engine
        if engine != 'cpu':
            assert actual['gpu_dispatches'] > 0
        references = [single.gradients(tail[i:i+1], [i], initial='carry' if cursor else None) for i, single in enumerate(singles)]
        for i, reference in enumerate(references):
            np.testing.assert_allclose(np.asarray(actual['final_state'])[i, indices], np.asarray(reference['final_state'])[0, indices], rtol=6e-5, atol=8e-6)
            np.testing.assert_array_equal(actual['spikes'][i], reference['spikes'][0])
            np.testing.assert_allclose(np.asarray(actual['initial_state_gradients'])[i, indices]*2, np.asarray(reference['initial_state_gradients'])[0, indices], rtol=8e-4, atol=1e-5)
        for bank, values in enumerate(actual['gradients']):
            expected = (np.array(references[0]['gradients'][bank])+references[1]['gradients'][bank])/2
            np.testing.assert_allclose(values, expected, rtol=8e-4, atol=1e-5)
        trainer.step(tail, [0, 1], initial='carry' if cursor else None)
        for i, single in enumerate(singles):
            single.step(tail[i:i+1], [i], initial='carry' if cursor else None)
        for path in trainer.plan['dynamic']['delay_layout']['paths']:
            for r in path.get('routes', []):
                different |= trainer.neuron_state[0][r['selection']] != trainer.neuron_state[1][r['selection']]
        cursor += length
    assert different
    saved = tmp_path/'event-delay.json';trainer.store(saved)
    restored = NativeLIFTrainer(trainer.plan, runner=RUNNER);restored.restore(saved)
    result = trainer.gradients(xx[:, :3], [0, 1], initial='carry')
    reference = restored.gradients(xx[:, :3], [0, 1], initial='carry')
    for key in ('final_state', 'spikes', 'initial_state_gradients'):
        np.testing.assert_array_equal(result[key], reference[key])


@pytest.mark.parametrize('window', [None, 2])
@pytest.mark.parametrize('ranks', [None, 2])
def test_noisy_delay_mutations_carry_mask_and_readonly_plan(engine, window, ranks, tmp_path):
    if ranks and os.environ.get('B2_TEST_MPI') != '1':
        pytest.skip('local MPI required')
    _, _, _, x, bundle = model(noisy=True, tbptt_window=window)
    bundle.plan.update(backend=engine, mpi_ranks=ranks)
    xx = np.stack([x, x.copy()]);xx[1, :2] = 0
    actual = NativeLIFTrainer(bundle.plan, runner=RUNNER, weights=bundle.weights)
    p = copy.deepcopy(bundle.plan);p.update(backend='cpu', mpi_ranks=None)
    reference = NativeLIFTrainer(p, runner=RUNNER, weights=bundle.weights)
    for trainer in (actual, reference):
        trainer.step(xx[:, :2], [0, 1], noise_sequence=9)
        bank, edge = trainer.plan['dynamic']['migration']['controlled_masks'][0]
        masks = copy.deepcopy(trainer.plan['masks']);masks[bank][edge] = 0
        trainer.update_mask(masks)
        trainer.step(xx[:, 2:4], [0, 1], initial='carry')
        masks[bank][edge] = 1;trainer.update_mask(masks, growth_weight=.5)
    before = snapshot(actual)
    result = actual.gradients(xx[:, 4:], [0, 1], initial='carry')
    expected = reference.gradients(xx[:, 4:], [0, 1], initial='carry')
    assert snapshot(actual)[:-1] == before[:-1]
    assert actual.clock_tick == 4 and actual.noise_sequence == 9
    for key in ('final_state', 'spikes', 'initial_state_gradients', 'logits'):
        np.testing.assert_allclose(result[key], expected[key], rtol=6e-4, atol=1e-5)
    for a, c in zip(result['gradients'], expected['gradients']):
        np.testing.assert_allclose(a, c, rtol=1e-3, atol=1e-5)


def scalar_model(statement):
    net, inp, layers, old, syn, _, _, x = network()
    net.remove(old)
    scalar = b.Synapses(inp, layers[0], 'w:1', on_pre=statement, delay=.24*b.ms, dt=.2*b.ms, name='event_scalar')
    scalar.connect();scalar.w = [.85, .95, 1.05, .75];net.add(scalar)
    return net, inp, layers, scalar, x


@pytest.mark.parametrize('statement', ['delay=0*ms', 'delay+=.2*ms', 'delay*=2'])
def test_scalar_delay_event_writes_remain_invalid(statement):
    net, inp, layers, scalar, _ = scalar_model('v_post+=w\n'+statement)
    with pytest.raises(TrainingConversionError, match='scalar/shared delay'):
        lower_brian_dynamic_training(net, input_group=inp, layers=layers)
    assert float(scalar.pre.delay[:]/b.ms) == .24


def test_shared_scalar_delay_reads_and_explicit_updates_match_brian():
    net, inp, layers, scalar, x = scalar_model('v_post+=w+.1*delay/ms')
    b.prefs.codegen.target = 'cython'
    bundle = lower_brian_dynamic_training(net, input_group=inp, layers=layers)
    bundle.plan['trainable'] = [False]*len(bundle.weights)
    trainer = NativeLIFTrainer(bundle.plan, runner=RUNNER, weights=bundle.weights)
    ids = bundle.provenance['pathway_state_layout'][scalar.pre.name]['delay']
    assert len(ids) == 1
    trainer.step(x[None, :2], [0]);net.run(.4*b.ms, namespace={})
    before = snapshot(trainer)
    with pytest.raises(ValueError, match='shared pathway delay'):
        trainer.update_delays({scalar.pre.name: [.0002, .0004, .0002, .0002]})
    assert snapshot(trainer) == before
    trainer.update_delays({scalar.pre.name: .64*b.ms});scalar.pre.delay = .64*b.ms
    result = trainer.step(x[None, 2:], [0], initial='carry');net.run(2*b.ms, namespace={})
    np.testing.assert_allclose(result['final_membrane'][0], np.r_[layers[0].v[:], layers[1].v[:]], rtol=3e-13, atol=1e-13)
    assert result['final_state'][0][ids[0]] == .00064
    assert scalar.pre.codeobj.compiled_code['run'] is not None


@pytest.mark.parametrize('issue', ['negative', 'huge', 'budget', 'duplicate_selector', 'foreign_gate'])
def test_bad_runtime_delay_configuration_is_atomic(issue):
    _, _, _, x, bundle = model()
    trainer = NativeLIFTrainer(bundle.plan, runner=RUNNER, weights=bundle.weights)
    trainer.step(x[None, :2], [0])
    path = trainer.plan['dynamic']['delay_layout']['paths'][0]
    if issue in ('negative', 'huge', 'budget'):
        trainer.neuron_state[0][path['edges'][0]['delay_state']] = {'negative':-.01, 'huge':1e9, 'budget':20.}[issue]
    elif issue == 'duplicate_selector':
        for route in path['routes']:
            trainer.neuron_state[0][route['selection']] = 0.
    else:
        p = copy.deepcopy(trainer.plan);p['dynamic']['delay_layout']['paths'][0]['routes'][0]['selection'] = 0
        broken = NativeLIFTrainer(p, runner=RUNNER, weights=bundle.weights)
        with pytest.raises(ValueError):
            broken.step(x[None, :2], [0])
        assert broken.neuron_state is None
        return
    before = snapshot(trainer)
    with pytest.raises(ValueError):
        trainer.step(x[None, 2:], [0], initial='carry')
    assert snapshot(trainer) == before


def test_explicit_updates_compose_with_event_writes_and_fresh_sequences():
    net, layers, synapses, x, bundle = model(post_first=True)
    trainer = NativeLIFTrainer(bundle.plan, runner=RUNNER, weights=bundle.weights)
    path = synapses[0].pre
    ids = bundle.provenance['pathway_state_layout'][path.name]['delay']
    trainer.step(x[None,:2], [0]);net.run(.4*b.ms, namespace={})
    cursor = 2
    for values, length in [([0.,0.,0.,0.],2), ([.00064,0.,.00024,.00044],4), ([.00024]*4,4)]:
        before = snapshot(trainer)
        configured = trainer.update_delays({path.name:values});path.delay = np.array(values)*b.second
        assert configured['gpu_dispatches'] == 0
        assert trainer.state == before[1]
        assert (trainer.elapsed_ticks, trainer.clock_tick, trainer.noise_sequence, trainer.next_noise_sequence) == before[3:7]
        np.testing.assert_array_equal(np.asarray(trainer.neuron_state)[0,ids], values)
        np.testing.assert_array_equal(np.asarray(trainer.plan['dynamic']['initial'])[ids], values)
        result = trainer.step(x[None,cursor:cursor+length], [0], initial='carry')
        net.run(length*.2*b.ms, namespace={});cursor += length
        np.testing.assert_allclose(result['final_membrane'][0], np.r_[layers[0].v[:],layers[1].v[:]], rtol=4e-13, atol=1e-13)
    before = snapshot(trainer)
    fresh = trainer.evaluate(x[None], [0])
    assert snapshot(trainer)[:-1] == before[:-1]
    reference, groups, objects, _, _ = model(post_first=True)
    objects[0].pre.delay = .24*b.ms;reference.run(len(x)*.2*b.ms, namespace={})
    np.testing.assert_allclose(fresh['final_membrane'][0], np.r_[groups[0].v[:],groups[1].v[:]], rtol=4e-13, atol=1e-13)


@pytest.mark.parametrize('ranks', [None,2,8])
def test_execution_failure_after_delay_rebuild_does_not_commit_plan_or_state(engine, ranks):
    if ranks and os.environ.get('B2_TEST_MPI') != '1':
        pytest.skip('local MPI required')
    net, inp, layers, _, _, _, _, x = network()
    fault = b.Synapses(inp,layers[-1],'w:1',on_pre='v_post+=1/w\ndelay=.64*ms',dt=.2*b.ms,name='event_fault')
    fault.connect();fault.w=1.;net.add(fault)
    bundle = lower_brian_dynamic_training(net, input_group=inp, layers=layers,backend=engine,mpi_ranks=ranks)
    bundle.plan['trainable'] = [False]*len(bundle.weights)
    trainer = NativeLIFTrainer(bundle.plan, runner=RUNNER, weights=bundle.weights)
    trainer.step(x[None,:2], [0])
    bank = next(e['bank'] for e in bundle.provenance['bindings'] if e['object']==fault.name and e['variables']==['w'])
    # Edge 1 targets neuron 3, owned by a non-root rank for both rank counts.
    trainer.state['weights'][bank][1] = 0.
    before = snapshot(trainer)
    with pytest.raises(ValueError):
        trainer.step(x[None,2:], [0], initial='carry')
    assert snapshot(trainer) == before
