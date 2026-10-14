"""Keep the Xylo recurrent diagnostic separate from exact SpiNNaker2 exports."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess

import brian2 as b
import numpy as np
import pytest
from brian2_rust.export import lower_network
from brian2_rust.native_xylo_conversion import build_native_xylo_network
from brian2_rust.spinnaker2_b2ir import _lower_validated
from brian2_rust.spinnaker2_runtime import run_native_batch

EXAMPLE = Path(__file__).resolve().parents[1] / 'examples/neuromorphic/native_xylo_recurrent'


def frozen():
    return json.loads((EXAMPLE / 'fixtures/frozen.json').read_text())


def test_recurrent_integer_source_is_not_ordinary_spinnaker2_lif():
    previous = b.prefs.codegen.loop_invariant_optimisations
    try:
        b.prefs.codegen.loop_invariant_optimisations = False
        model = lower_network(build_native_xylo_network(frozen(), np.zeros((40, 3))), 40 * b.ms)
        with pytest.raises(ValueError, match='SpiNNaker2:.*scheduling'):
            _lower_validated(model)
    finally:
        b.prefs.codegen.loop_invariant_optimisations = previous
        b.start_scope()
    with pytest.raises(ValueError, match='bridge contract'):
        run_native_batch({}, dict(schema='atlas.native-xylo-frozen.v2',
            fresh_state_per_sample=True, source_is_zero_input_template=True), np.zeros((1,40,3)), 1)


def test_diagnostic_reports_small_state_and_clock_errors(monkeypatch):
    monkeypatch.syspath_prepend(str(EXAMPLE))
    spec = importlib.util.spec_from_file_location('diagnostic', EXAMPLE / 'probe_spinnaker2.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = np.array([[[0., 1.], [2., 0.]]])
    tiny = source.copy(); tiny[0, 1, 0] += .125
    report = module.difference(source, tiny)
    assert report['differences'] == 1 and report['max_abs_error'] == .125
    assert report['first_difference'] == dict(index=[0, 1, 0], native=2., candidate=2.125)
    assert module.difference(source, np.roll(source, 1, axis=1))['differences'] == 4
    with pytest.raises(AssertionError):
        module.difference(source, np.full_like(source, np.nan))


def test_stock_sdk_diagnostic_is_batch_independent_and_not_exact(tmp_path, monkeypatch):
    sdk = os.environ.get('B2_SPINNAKER2_PYTHON')
    runner = os.environ.get('B2_TRAIN_RUNNER')
    if not sdk or not runner:
        pytest.skip('set B2_SPINNAKER2_PYTHON and B2_TRAIN_RUNNER')
    monkeypatch.syspath_prepend(str(EXAMPLE))
    from train import data
    from brian2_atlas import NativeXyloTrainer
    x, y, _ = data('test'); x, y = x[:2], y[:2]
    np.save(tmp_path / 'input.npy', x)
    script = '''import importlib.util,json,numpy as np,sys
spec=importlib.util.spec_from_file_location('worker',sys.argv[1]);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
f=json.load(open(sys.argv[2]));x=np.load(sys.argv[3]);batch,banks=m.run_batch(f,x)
for i in range(len(x)):
 solo,_=m.run_batch(f,x[i:i+1])
 for key in batch: np.testing.assert_array_equal(batch[key][i:i+1],solo[key])
np.savez(sys.argv[4],**batch)
'''
    result = subprocess.run([sdk, '-c', script, str(EXAMPLE / 'spinnaker2_worker.py'),
        str(EXAMPLE / 'fixtures/frozen.json'), str(tmp_path / 'input.npy'), str(tmp_path / 'sdk.npz')],
        cwd=tmp_path, env={**os.environ, 'PYTHONPATH': '', 'PYTHONNOUSERSITE': '1'},
        capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    f = frozen()
    trainer = NativeXyloTrainer(f['plan'], runner=runner,
        weights=[np.asarray(w) / f['plan']['weight_scale'] for w in f['target_weights']])
    native = trainer.evaluate(x, y)
    sdk_result = np.load(tmp_path / 'sdk.npz')
    assert np.max(native['spikes']) > 1 and np.max(sdk_result['spikes']) == 1
    assert np.array_equal(np.argmax(native['logits'], axis=1), y)
    for offset in (0, 1):
        for key in ('spikes', 'membranes', 'currents'):
            assert np.count_nonzero(np.asarray(native[key]) != sdk_result[key][:, offset:40+offset]) > 0
