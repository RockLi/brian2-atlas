"""Preparation must reject oversized MAM requests before creating neurons."""
import gzip
import importlib.util
import json
from pathlib import Path
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'python'))
spec = importlib.util.spec_from_file_location('mam_pilot_budget', ROOT/'examples/mpi_multi_area.py')
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


@pytest.mark.parametrize('limits,reason', [
    ({'max_neurons':70996}, 'neuron preparation budget'),
    ({'max_recurrent_edges':84835659}, 'recurrent-edge preparation budget'),
    ({'max_neurons':0}, 'positive integers'),
    ({'max_recurrent_edges':-1}, 'positive integers'),
    ({'max_neurons':True}, 'positive integers'),
    ({'max_recurrent_edges':1.5}, 'positive integers'),
])
def test_budget_rejection_precedes_allocation(monkeypatch, limits, reason):
    parameters = json.loads(gzip.decompress((ROOT/'mpi-evidence/n02-k04/parameters.json.gz').read_bytes()))
    def unexpected(*args, **kwargs):
        pytest.fail('NeuronGroup allocation occurred before preparation budget rejection')
    monkeypatch.setattr(pilot.b, 'NeuronGroup', unexpected)
    with pytest.raises(ValueError, match=reason):
        pilot.make_model(parameters, ['V1', 'V2'], **limits)


def test_initial_value_budget_is_checked_before_mam_neuron_allocation(monkeypatch):
    parameters = json.loads(gzip.decompress((ROOT/'mpi-evidence/n02-k04/parameters.json.gz').read_bytes()))
    monkeypatch.setenv('B2_MAX_INITIAL_VALUES',str(4*70997-1))
    def unexpected(*args,**kwargs):
        pytest.fail('allocated neurons before initial-value admission')
    monkeypatch.setattr(pilot.b,'NeuronGroup',unexpected)
    with pytest.raises(ValueError,match='initial-value preparation budget'):
        pilot.make_model(parameters,['V1','V2'])


def test_nest_grid_adapter_preserves_inputs_and_changes_only_refractory_instance():
    import copy
    import gc
    from brian2_rust.protocol import attach_protocol, verify_protocol
    parameters = json.loads(gzip.decompress((ROOT/'mpi-evidence/n02-k04/parameters.json.gz').read_bytes()))
    # Keep the real adapter and population/projection structure, but construct
    # only two neurons and one procedural edge per selected population/pair.
    for population in parameters['populations']:
        population['count'] = 2
    for projection in parameters['projections']:
        projection['count'] = 1
    pilot.b.set_device('rust_standalone',runner=ROOT/'target/release/b2-runner')
    try:
        legacy, legacy_owners = pilot.make_model(parameters,['V1'],steps=2)
        gc.collect()
        translated, translated_owners = pilot.make_model(parameters,['V1'],steps=2,nest_grid=True)
        verify_protocol(legacy)
        verify_protocol(translated)
        assert legacy_owners == translated_owners
        normalized = copy.deepcopy(translated)
        for old, new, normal in zip(legacy['instance']['populations'],translated['instance']['populations'],
                                    normalized['instance']['populations'],strict=True):
            assert old['refractory']['period_ticks'] == 20
            assert new['refractory']['period_ticks'] == 21
            normal['refractory']['period'] = old['refractory']['period']
            normal['refractory']['period_ticks'] = old['refractory']['period_ticks']
        # Protocol identities must change with the refractory instance. Rehash
        # the normalized content before comparing the entire exported model.
        assert attach_protocol(normalized) == legacy
    finally:
        pilot.b.device.reinit()
        pilot.b.set_device('runtime')
