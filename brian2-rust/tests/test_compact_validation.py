"""Large-document validation retains the public loader's acceptance contract."""
import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
from brian2_rust.protocol import attach_protocol  # noqa: E402


@pytest.mark.parametrize('case,accepted', [
    ('current', True), ('omitted_optional', True), ('duplicate_key', True),
    ('tampered', False), ('invalid_shape', False), ('unknown_field', False),
])
def test_compact_validation_matches_ordinary_loader(tmp_path, case, accepted):
    model = json.loads((ROOT / 'tests/golden/b2ir-v1/minimal-v1.json').read_text())
    model = copy.deepcopy(model)
    if case == 'omitted_optional':
        assert model['instance']['populations'][0]['spike_generator'] is None
        del model['instance']['populations'][0]['spike_generator']
    elif case in {'tampered', 'invalid_shape'}:
        model['instance']['neuron_count'] += 1
    elif case == 'unknown_field':
        model['instance']['unexpected'] = []
    if case != 'tampered':
        attach_protocol(model)
    wire = json.dumps(model, sort_keys=True, separators=(',', ':'))
    if case == 'duplicate_key':
        wire = wire[:-1] + ',"schema":"b2ir-v1"}'
    results = []
    for label, padding in [('ordinary', ''), ('large', ' ' * (8 * 1024 * 1024))]:
        path = tmp_path / f'{label}.json'
        path.write_text(wire + padding)
        result = subprocess.run([str(ROOT / 'target/release/b2-runner'),
                                 '--validate', str(path)], capture_output=True, text=True)
        results.append(result.returncode == 0)
    assert results == [accepted, accepted]
