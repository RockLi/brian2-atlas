"""Semantic admission tests for the bounded temporal simulator profile."""
import copy
import importlib.util
import json
import os
from pathlib import Path

import brian2 as b
import pytest
from brian2_atlas.neuromorphic import convert_ir, export_ir, run_simulator
from brian2_rust.export import lower_network
from brian2_rust.protocol import canonical_bytes
from brian2_rust.spec import bits
from brian2_rust.spinnaker2_b2ir import _lower_validated

EXAMPLE = Path(__file__).resolve().parents[1] / 'examples/neuromorphic/spinnaker2/compare.py'
spec = importlib.util.spec_from_file_location('spinnaker2_example', EXAMPLE)
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


@pytest.fixture
def model():
    previous_device = b.get_device()
    previous_optimisation = b.prefs.codegen.loop_invariant_optimisations
    previous_target = b.prefs.codegen.target
    try:
        config = example.cases()['recurrent_16']
        yield lower_network(example.build(config), config['steps'] * b.ms)
    finally:
        b.set_device(previous_device)
        b.prefs.codegen.loop_invariant_optimisations = previous_optimisation
        b.prefs.codegen.target = previous_target
        b.start_scope()


def test_preserves_spike_times_delays_and_signed_recurrent_weights(model):
    plan = _lower_validated(model)
    assert plan['steps'] == 16
    assert plan['populations'][0]['params'] == {'0': [0, 4], '1': [1, 8]}
    assert plan['populations'][1]['params']['alpha_decay'] == [.5, .75, .875]
    assert plan['projections'][0]['connections'] == [[0, 0, 3, 0], [0, 1, -2, 1], [1, 2, 4, 3]]
    assert plan['projections'][1]['connections'] == [[0, 2, 2, 1], [2, 0, -1, 2]]


@pytest.mark.parametrize('change,message', [
    ('schedule', 'scheduling'), ('dt', 'dt=1 ms'), ('continuation', 'fresh run'),
    ('initial_v', 'initial voltage'), ('threshold_ge', 'threshold AST'),
    ('non_lif', 'state_update AST'), ('extra_code', 'only Euler'),
    ('reset', 'reset AST'), ('refractory', 'refractory'),
    ('fractional_weight', 'weights'), ('oversized_weight', 'weights'),
    ('delay_overflow', '3 bits'), ('fractional_delay', 'exact integer'),
    ('multapses', 'multapses'), ('pending', 'pending'), ('start_monitor', 'end-of-step'),
    ('unrepresentable_tau', 'round-trip|float32'), ('inexact_parameter', 'float32'),
])
def test_rejects_semantics_the_sdk_cannot_preserve(model, change, message):
    m = copy.deepcopy(model)
    p = m['definition']['populations'][1]
    state = m['instance']['populations'][1]
    synapse = m['instance']['synapses'][0]
    if change == 'schedule':
        m['definition']['schedule']['base_slots'] = ['start', 'groups', 'thresholds', 'synapses', 'resets', 'end']
    elif change == 'dt':
        m['definition']['clocks'][0]['dt'] = bits(.002)
    elif change == 'continuation':
        m['run']['start'] = bits(.001)
    elif change == 'initial_v':
        state['initial_state']['v'][0] = bits(1.)
    elif change == 'threshold_ge':
        p['code_objects'][1]['vector'][0]['value']['op'] = 'ge'
    elif change == 'non_lif':
        p['code_objects'][0]['vector'][0]['value']['op'] = 'sub'
    elif change == 'extra_code':
        p['code_objects'].append(copy.deepcopy(p['code_objects'][0]))
    elif change == 'reset':
        p['code_objects'][2]['vector'][0]['value'] = {'op': 'load', 'name': 'threshold'}
    elif change == 'refractory':
        p['refractory'] = {'mode': 'fixed'}
    elif change == 'fractional_weight':
        synapse['parameters']['w'][0] = bits(1.5)
    elif change == 'oversized_weight':
        synapse['parameters']['w'][0] = bits(16.)
    elif change == 'delay_overflow':
        synapse['pathways'][0]['delay_ticks'][0] = 8
    elif change == 'fractional_delay':
        synapse['pathways'][0]['delay'][0] = bits(.0005)
    elif change == 'multapses':
        synapse['source'][1] = synapse['source'][0]
        synapse['target'][1] = synapse['target'][0]
    elif change == 'pending':
        synapse['pathways'][0]['pending'] = [[1, 0]]
    elif change == 'start_monitor':
        p['state_monitors'][0]['when'] = 'start'
    elif change == 'unrepresentable_tau':
        state['parameters']['tau'][0] = bits(.003)
    elif change == 'inexact_parameter':
        state['parameters']['threshold'][0] = bits(1.1)
    with pytest.raises(ValueError, match=message):
        _lower_validated(m)


def test_validated_export_preserves_source_and_refuses_tampering(model, tmp_path):
    runner = Path(os.environ.get('B2_RUNNER', 'missing'))
    if not runner.is_file():
        pytest.skip('set B2_RUNNER to the current Atlas native runner')
    source = tmp_path / 'source.json'
    source.write_bytes(canonical_bytes(model))
    bundle = export_ir(source, tmp_path / 'converted', target='spinnaker2',
                       profile='spinnaker2-lif-v1', runner=runner)
    assert (tmp_path / 'converted/source.json').read_bytes() == source.read_bytes()
    assert bundle['physical_hardware'] is False and bundle['hardware_mapping_tested'] is False
    assert bundle['source_hashes'] == model['protocol']['layers']
    with pytest.raises(FileExistsError):
        export_ir(source, tmp_path / 'converted', target='spinnaker2',
                  profile='spinnaker2-lif-v1', runner=runner)
    model['instance']['populations'][1]['initial_state']['v'][0] = bits(1.)
    source.write_bytes(canonical_bytes(model))
    with pytest.raises(ValueError, match='hash mismatch'):
        convert_ir(source, target='spinnaker2', profile='spinnaker2-lif-v1', runner=runner)


def test_static_and_temporal_profiles_cannot_be_mixed():
    for target, profile in [('specksim', 'dense-int8-v1'), ('lava', 'spinnaker2-lif-v1')]:
        with pytest.raises(ValueError, match='only supported|binary-count-v1'):
            convert_ir('missing', target=target, profile=profile, runner='missing')
    bundle = dict(schema='atlas.neuromorphic.simulator-bundle.v1', target='spinnaker2',
                  profile='spinnaker2-lif-v1', plan={})
    with pytest.raises(ValueError, match='embedded spike schedule'):
        run_simulator(bundle, [[1]])


def test_sdk_float_timestamps_use_nearest_clock_tick():
    import numpy as np
    from brian2_rust.spinnaker2_runtime import spike_ticks
    times=np.array([1,1001,2025,4002,9999],dtype=float)
    below=np.nextafter(times,-np.inf)
    assert np.any(below.astype(int)!=times)
    np.testing.assert_array_equal(spike_ticks(below),times.astype(int))
    assert spike_ticks([]).shape==(0,)


@pytest.mark.parametrize('times',[[1.25],[float('nan')],[float('inf')]])
def test_sdk_off_clock_timestamps_rejected(times):
    from brian2_rust.spinnaker2_runtime import spike_ticks
    with pytest.raises(ValueError,match='clock'):
        spike_ticks(times)


def test_scalar_delay_broadcast_preserves_every_connection(model):
    for synapse in model['instance']['synapses']:
        synapse['pathways'][0]['delay'] = [bits(0.)]
        synapse['pathways'][0]['delay_ticks'] = [0]
    plan = _lower_validated(model)
    for original, converted in zip(model['instance']['synapses'], plan['projections']):
        assert len(converted['connections']) == len(original['source'])
        assert all(row[3] == 0 for row in converted['connections'])


def test_scalar_weight_broadcast_preserves_every_connection(model):
    for synapse in model['instance']['synapses']:
        synapse['parameters']['w'] = [bits(2.)]
    plan = _lower_validated(model)
    for original, converted in zip(model['instance']['synapses'], plan['projections']):
        assert len(converted['connections']) == len(original['source'])
        assert all(row[2] == 2 for row in converted['connections'])


def test_invalid_delay_shape_is_not_silently_zipped(model):
    model['instance']['synapses'][0]['pathways'][0]['delay_ticks'] = [0, 1]
    with pytest.raises(ValueError, match='scalar or per-synapse'):
        _lower_validated(model)
