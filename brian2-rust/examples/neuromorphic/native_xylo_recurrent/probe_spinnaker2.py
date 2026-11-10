"""Compare the frozen recurrent task with a stock current-LIF substitution.

This diagnostic must not be described as an equivalent SpiNNaker2 conversion.
It reuses the held-out split and all weights without training or selection.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess

import brian2 as b
import numpy as np
from brian2_atlas import NativeXyloTrainer
from brian2_atlas.neuromorphic import convert_ir, export_native_xylo
from brian2_rust.native_xylo_conversion import validate_frozen
from train import data, sha, STEPS, READOUT_START
from verify import accuracy, native


def difference(reference, candidate):
    a, c = np.asarray(reference), np.asarray(candidate)
    assert a.shape == c.shape and np.isfinite(c).all()
    delta = c.astype(float) - a
    indices = np.argwhere(a != c)
    first = None
    if len(indices):
        index = tuple(indices[0])
        first = dict(index=[int(i) for i in index], native=float(a[index]), candidate=float(c[index]))
    return dict(values=int(a.size), differences=int(len(indices)),
                max_abs_error=float(np.max(np.abs(delta))), mean_abs_error=float(np.mean(np.abs(delta))),
                first_difference=first)


def sdk_run(python, worker, args, output):
    with output.open('w') as log:
        subprocess.run([str(python.absolute()), str(worker), *map(str, args)],
            cwd=output.parent, env={**os.environ, 'PYTHONPATH': '', 'PYTHONNOUSERSITE': '1'},
            stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('train-runner', 'runner', 'xylo-python', 'spinnaker2-python', 'output'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--trained', type=Path, default=Path(__file__).with_name('fixtures'))
    a = p.parse_args()
    a.output = a.output.resolve()
    a.output.mkdir(parents=True, exist_ok=False)
    f = json.loads((a.trained / 'frozen.json').read_text())
    sizes = validate_frozen(f)
    training = json.loads((a.trained / 'training.json').read_text())
    assert sha(a.trained / 'frozen.json') == training['frozen_file_sha256']
    assert f['schema'] == 'atlas.native-xylo-frozen.v2'
    assert f['plan']['readout_start'] == READOUT_START
    x, y, ids = data('test')
    np.savez_compressed(a.output / 'inputs.npz', spikes=x)
    weights = [np.asarray(w) / f['plan']['weight_scale'] for w in f['target_weights']]
    trainer = NativeXyloTrainer(f['plan'], runner=a.train_runner, weights=weights)
    ref = native(trainer, x, y)
    b.prefs.codegen.target = 'numpy'
    bundle = export_native_xylo(f, a.output / 'xylosim', steps=STEPS, runner=a.runner)
    try:
        convert_ir(a.output / 'xylosim/template.json', target='spinnaker2',
                   profile='spinnaker2-lif-v1', runner=a.runner)
    except ValueError as error:
        admission = dict(admitted=False, reason=str(error))
    else:
        raise AssertionError('Xylo integer dynamics were admitted as ordinary SpiNNaker2 LIF')
    here = Path(__file__).resolve().parent
    runtime = here.parents[2] / 'python/brian2_rust/neuromorphic_runtime.py'
    sdk_run(a.xylo_python, here / 'sdk_worker.py',
        ['--runtime', runtime, '--bundle', a.output / 'xylosim/converted/bundle.json',
         '--inputs', a.output / 'inputs.npz', '--output', a.output / 'xylo'], a.output / 'xylo.log')
    sdk_run(a.spinnaker2_python, here / 'spinnaker2_worker.py',
        ['--frozen', (a.trained / 'frozen.json').resolve(), '--inputs', a.output / 'inputs.npz',
         '--output', a.output / 'spinnaker2'], a.output / 'spinnaker2.log')
    xy = np.load(a.output / 'xylo/results.npz', allow_pickle=False)
    xy_arrays = dict(spikes=xy['spike_raster'], membranes=xy['voltage'],
        currents=np.concatenate((xy['hidden_I'], xy['output_I']), axis=2), logits=xy['logits'])
    xy_comparisons = {k: difference(ref[k], xy_arrays[k]) for k in ref}
    assert all(v['differences'] == 0 for v in xy_comparisons.values())
    sp = np.load(a.output / 'spinnaker2/results.npz', allow_pickle=False)
    views, per_sample = {}, dict(labels=y, template_ids=ids, native_logits=ref['logits'], xylo_logits=xy['logits'])
    # Both predeclared clock conventions are reported; neither is selected by accuracy.
    for name, offset in [('raw_clock', 0), ('sdk_shifted_clock', 1)]:
        candidate = {k: sp[k][:, offset:offset + STEPS] for k in ('spikes', 'membranes', 'currents')}
        counts = candidate['spikes'][:, READOUT_START:, -sizes[2]:].sum(1)
        logits = counts * f['plan']['logit_scale'] / (STEPS - READOUT_START)
        candidate['logits'] = logits
        views[name] = dict(accuracy=accuracy(logits, y), comparisons={k: difference(ref[k], candidate[k]) for k in ref},
            prediction_disagreements=int(np.count_nonzero(np.argmax(logits, 1) != np.argmax(ref['logits'], 1))),
            tied_readouts=int(np.count_nonzero(logits[:, 0] == logits[:, 1])),
            all_zero_readouts=int(np.count_nonzero(np.all(logits == 0, axis=1))))
        per_sample[name + '_logits'] = logits
        per_sample[name + '_counts'] = counts
    np.savez_compressed(a.output / 'per_sample.npz', **per_sample)
    executions = {name: json.loads((a.output / name / 'execution.json').read_text()) for name in ('xylo', 'spinnaker2')}
    for name, execution in executions.items():
        assert execution['inputs_sha256'] == sha(a.output / 'inputs.npz')
        assert execution['results_sha256'] == sha(a.output / name / 'results.npz')
    assert executions['spinnaker2']['frozen_file_sha256'] == sha(a.trained / 'frozen.json')
    report = dict(schema='atlas.recurrent-spinnaker2-diagnostic-audit.v1',
        diagnostic_completed=True, exact_conversion_supported=False, strict_conversion=admission,
        task='Delayed AB/BA discrimination', samples=len(y), templates=len(set(ids)), sizes=sizes,
        steps=STEPS, readout_start=READOUT_START, same_weight_banks=True, retraining=False,
        model_substitution='Xylo integer current dynamics -> stock lif_curr_exp_32bit',
        native=accuracy(ref['logits'], y), xylosim=accuracy(xy['logits'], y), native_vs_xylosim=xy_comparisons,
        spinnaker2_views=views, native_hidden_max_spike_count=int(ref['spikes'][:, :, :sizes[1]].max()),
        native_hidden_multispike_entries=int(np.count_nonzero(ref['spikes'][:, :, :sizes[1]] > 1)),
        native_current_range=[float(ref['currents'].min()), float(ref['currents'].max())],
        frozen_sha256=f['sha256'], frozen_file_sha256=sha(a.trained / 'frozen.json'),
        train_runner_sha256=sha(a.train_runner), atlas_runner_sha256=sha(a.runner),
        inputs_sha256=sha(a.output / 'inputs.npz'), per_sample_sha256=sha(a.output / 'per_sample.npz'),
        sdk_executions=executions, physical_hardware=False,
        source_hashes={path.name: sha(path) for path in (Path(__file__), here / 'spinnaker2_worker.py', here / 'train.py', here / 'verify.py')})
    (a.output / 'audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('strict_conversion', 'native', 'xylosim', 'spinnaker2_views',
          'native_hidden_max_spike_count', 'native_hidden_multispike_entries')}, indent=2))


if __name__ == '__main__':
    main()
