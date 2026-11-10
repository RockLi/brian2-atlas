"""Tick-only reads preserve all retained values and reject corrupt dumps."""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('result_reader_times', ROOT / 'python/brian2_rust/results.py')
reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader)


def equal_without_times(full, lean):
    if isinstance(full, dict):
        assert set(lean) == set(full) - {'times', 'spike_times'}
        for key in lean:
            equal_without_times(full[key], lean[key])
    elif isinstance(full, list):
        assert len(full) == len(lean)
        for x, y in zip(full, lean, strict=True):equal_without_times(x, y)
    elif isinstance(full, np.ndarray):
        assert full.dtype == lean.dtype
        np.testing.assert_array_equal(full, lean)
    else:
        assert full == lean


def test_retained_values_and_default_api_match_existing_result():
    folder = ROOT / 'tests/fixtures/reference-v1'
    model = json.loads((folder / 'model.json').read_text())
    full = reader.load_results(model, folder / 'reference')
    lean = reader.load_results(model, folder / 'reference', include_times=False)
    equal_without_times(full, lean)
    assert full['metadata']['spike_count'] > 0
    assert any(p['event_streams'] for p in full['populations'])
    assert all('spike_times' in p and 'times' in p for p in full['populations'])
    assert not lean['populations'][0]['trace']['v'].flags.owndata


@pytest.mark.parametrize('include_times', [True, False])
@pytest.mark.parametrize('corruption', ['version', 'truncated', 'counts'])
def test_both_modes_keep_validation(tmp_path, include_times, corruption):
    folder = ROOT / 'tests/fixtures/reference-v1'
    model = json.loads((folder / 'model.json').read_text())
    for name in ['results.bin', 'events.bin', 'summary.json']:
        (tmp_path / name).write_bytes((folder / 'reference' / name).read_bytes())
    payload = bytearray((tmp_path / 'results.bin').read_bytes())
    if corruption == 'version':payload[8:12] = (99).to_bytes(4, 'little')
    elif corruption == 'truncated':payload = payload[:-1]
    else:
        metadata = json.loads((tmp_path / 'summary.json').read_text())
        metadata['spike_count'] += 1
        (tmp_path / 'summary.json').write_text(json.dumps(metadata))
    (tmp_path / 'results.bin').write_bytes(payload)
    with pytest.raises(RuntimeError, match='inconsistent results'):
        reader.load_results(model, tmp_path, include_times=include_times)


def test_time_mode_requires_explicit_bool():
    with pytest.raises(TypeError, match='include_times'):
        reader.load_results({}, Path('/nonexistent'), include_times=0)


def test_custom_event_monitor_and_stream_times(tmp_path):
    import brian2 as b
    sys.path.insert(0, str(ROOT / 'python'))
    from brian2_rust.export import export_network
    b.start_scope()
    b.set_device('runtime')
    group = b.NeuronGroup(3, 'dv/dt=1000*Hz:1', events={'crossing': 'v>=1'},
                          dt=.1*b.ms, method='euler')
    group.run_on_event('crossing', 'v=0')
    monitor = b.EventMonitor(group, 'crossing', variables='v')
    network = b.Network(group, monitor)
    model = export_network(network, 3*b.ms, tmp_path / 'model.json')
    run = subprocess.run([str(ROOT / 'target/release/b2-runner'), str(tmp_path / 'model.json'), str(tmp_path / 'out')], capture_output=True, text=True, timeout=30)
    assert run.returncode == 0, run.stderr
    full = reader.load_results(model, tmp_path / 'out')
    lean = reader.load_results(model, tmp_path / 'out', include_times=False)
    equal_without_times(full, lean)
    population = full['populations'][0]
    assert population['event_monitors'] and population['event_streams']
    assert any(len(event['times']) for event in population['event_monitors'].values())
