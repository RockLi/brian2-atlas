"""Export native Brian2 LIF cases to AtlasIR; compare Atlas with the real SDK."""
import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import brian2 as b
import numpy as np
from brian2_atlas import export_network
from brian2_atlas.neuromorphic import export_ir
from brian2_rust.results import load_results


def cases():
    baseline = dict(steps=64, events=[(0, 0), (1, 1), (4, 0), (8, 1), (31, 0)],
                    tau_ms=[2, 4, 8], drive=[1, 0, 2], threshold=[3, 2, 5], reset_v=[0, 0, 0],
                    reset='v = v_reset', inputs=[[0, 0, 3, 0], [0, 1, -2, 1], [1, 2, 4, 3]],
                    recurrent=[[0, 2, 2, 1], [2, 0, -1, 2]])
    result = {}
    def add(name, **changes):
        result[name] = {**baseline, **changes}
    add('recurrent_16', steps=16, events=baseline['events'][:-1])
    add('recurrent_64')
    add('subtract_reset', reset='v -= threshold', drive=[9, 7, 4])
    add('nonzero_reset', reset_v=[-.5, .5, 1])
    add('zero_delay', steps=12, events=[(0, 0)], drive=[0, 0, 0],
        inputs=[[0, 0, 3, 0]], recurrent=[])
    add('seven_tick_delay', steps=12, events=[(0, 0)], drive=[0, 0, 0],
        inputs=[[0, 0, 3, 7]], recurrent=[])
    add('pending_at_end', steps=12, events=[(11, 0)], drive=[0, 0, 0],
        inputs=[[0, 0, 3, 7]], recurrent=[])
    add('threshold_equality', steps=16, events=[], tau_ms=[1, 1, 1], drive=[1, 2, 3],
        threshold=[1, 2, 3], recurrent=[])
    add('threshold_above', steps=16, events=[], tau_ms=[1, 1, 1], drive=[1, 2, 3],
        threshold=[.5, 1, 2], recurrent=[])
    add('weight_limits', inputs=[[0, 0, 15, 0], [0, 1, -15, 7], [1, 2, 15, 3]])
    add('silent', events=[], drive=[0, 0, 0])
    rng = np.random.default_rng(20261013)
    events = [(tick, i) for tick in range(128) for i in range(2) if rng.random() < .2]
    add('seeded_recurrent', steps=128, events=events, tau_ms=[1, 2, 4], drive=[.5, 1, 2],
        threshold=[3, 4, 5], inputs=[[0, 0, 3, 0], [0, 1, -2, 7], [1, 2, 4, 2]])
    return result


def build(config):
    b.start_scope()
    b.set_device('runtime')
    b.prefs.codegen.target = 'numpy'
    b.prefs.codegen.loop_invariant_optimisations = False
    clock = b.Clock(dt=b.ms, name='clock')
    events = config['events']
    source = b.SpikeGeneratorGroup(2, [i for _, i in events],
                                  np.array([t for t, _ in events]) * b.ms, clock=clock, name='input_spikes')
    group = b.NeuronGroup(3,
        'dv/dt = -v/tau + i_offset/dt : 1\n'
        'tau : second (constant)\ni_offset : 1 (constant)\n'
        'v_reset : 1 (constant)\nthreshold : 1 (constant)',
        threshold='v > threshold', reset=config['reset'], method='euler', clock=clock, name='lif')
    group.tau = np.array(config['tau_ms']) * b.ms
    group.i_offset = config['drive']
    group.threshold = config['threshold']
    group.v_reset = config['reset_v']
    objects = [source, group]
    for name, pre, rows in [('input_projection', source, config['inputs']),
                            ('recurrent_projection', group, config['recurrent'])]:
        if not rows:
            continue
        s = b.Synapses(pre, group, 'w : 1 (constant)', on_pre='v_post += w', clock=clock, name=name)
        s.connect(i=[r[0] for r in rows], j=[r[1] for r in rows])
        s.w = [r[2] for r in rows]
        s.delay = np.array([r[3] for r in rows]) * b.ms
        objects.append(s)
    objects += [b.StateMonitor(group, 'v', record=True, when='end', name='voltage'),
                b.SpikeMonitor(group, name='spikes')]
    net = b.Network(*objects)
    net.schedule = ['start', 'groups', 'synapses', 'thresholds', 'resets', 'end']
    return net


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runner', type=Path, required=True)
    p.add_argument('--sdk-python', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output = a.output.resolve()
    a.output.mkdir(parents=True, exist_ok=False)
    bundles, model_hashes = [], {}
    for name, config in cases().items():
        root = a.output / name
        root.mkdir()
        source = root / 'model.json'
        model = export_network(build(config), config['steps'] * b.ms, source)
        bundle = export_ir(source, root / 'converted', target='spinnaker2',
                           profile='spinnaker2-lif-v1', runner=a.runner)
        subprocess.run([str(a.runner.resolve()), str(source), str(root / 'atlas')], check=True,
                       capture_output=True, text=True)
        result = load_results(model, root / 'atlas')['populations'][bundle['plan']['lif_population']]
        np.savez_compressed(root / 'atlas-results.npz', voltage=result['trace']['v'],
                            final_v=result['states']['v'],
                            spikes=np.column_stack((result['spike_ticks'], result['indices'])))
        bundles.append(root / 'converted' / 'bundle.json')
        model_hashes[name] = bundle['source_model_sha256']
        print(name, 'Atlas finished', flush=True)
    # Run vendor SDK under upstream Brian2, with no Atlas package on sys.path.
    script = Path(__file__).with_name('sdk_worker.py')
    runtime = Path(__file__).resolve().parents[3] / 'python/brian2_rust/neuromorphic_runtime.py'
    env = {**os.environ, 'PYTHONPATH': '', 'PYTHONNOUSERSITE': '1'}
    process = subprocess.run([str(a.sdk_python.absolute()), str(script), '--runtime', str(runtime),
                              '--bundles', *map(str, bundles)], env=env, cwd=a.output,
                             capture_output=True, text=True, timeout=300)
    (a.output / 'sdk.log').write_text(process.stdout + process.stderr)
    if process.returncode:
        raise RuntimeError(process.stderr[-5000:])
    reports = []
    for path in bundles:
        name = path.parent.parent.name
        with np.load(path.parent.parent / 'atlas-results.npz', allow_pickle=False) as atlas, \
             np.load(path.parent / 'sdk-results.npz', allow_pickle=False) as sdk:
            np.testing.assert_allclose(sdk['voltage_f64'], atlas['voltage'], rtol=0, atol=1e-12)
            np.testing.assert_allclose(sdk['final_v'], atlas['final_v'], rtol=0, atol=1e-12)
            np.testing.assert_array_equal(sdk['voltage'], atlas['voltage'].astype(np.float32))
            np.testing.assert_array_equal(sdk['spikes'], atlas['spikes'])
            np.testing.assert_array_equal(sdk['raw_spikes'], atlas['spikes'])
            reports.append(dict(case=name, steps=len(sdk['voltage']), state_values=sdk['voltage'].size,
                                spikes=len(sdk['spikes']), spike_differences=0,
                                max_abs_voltage_error=float(np.max(np.abs(sdk['voltage_f64'] - atlas['voltage']))),
                                public_float32_voltage_differences=0,
                                source_model_sha256=model_hashes[name],
                                atlas_results_sha256=sha(path.parent.parent / 'atlas-results.npz'),
                                sdk_execution=json.loads((path.parent / 'sdk-execution.json').read_text())))
    report = dict(schema='atlas.spinnaker2.lif-audit.v1', passed=True, cases=reports,
                  runner_sha256=sha(a.runner), sdk_backend='upstream Brian2 NumPy via py-spinnaker2',
                  physical_hardware=False, hardware_mapping_tested=False,
                  voltage_atol=1e-12, voltage_rtol=0, spike_comparison='exact tick and neuron',
                  includes_mnist=False)
    (a.output / 'audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(passed=True, cases=len(reports), state_values=sum(r['state_values'] for r in reports),
                         spikes=sum(r['spikes'] for r in reports),
                         max_abs_voltage_error=max(r['max_abs_voltage_error'] for r in reports)), indent=2))


if __name__ == '__main__':
    main()
