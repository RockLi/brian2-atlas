"""Diagnostic stock SpiNNaker2 current-LIF substitution, not an exact export.

No labels, native traces, retraining, weight rescaling or neuron equation patches.
The observer adds current recording to the SDK's existing voltage monitor.
"""
import argparse
import hashlib
import importlib.metadata as md
import json
from pathlib import Path
import time

import brian2 as b
import numpy as np
from spinnaker2 import brian2_sim, snn
from spinnaker2.neuron_models import lif_curr_exp


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class CurrentObserver(brian2_sim.Brian2Backend):
    def _build_population(self, pop, timestep):
        group, spikes, voltage = super()._build_population(pop, timestep)
        if pop.neuron_model == 'lif_curr_exp_32bit':
            # Observation only: keep the stock group, threshold, reset and schedule.
            voltage = b.StateMonitor(group, ['v', 'I_syn_exc', 'I_syn_inh'], record=True, when='end')
        return group, spikes, voltage


def run_batch(frozen, samples):
    if b.get_device().__class__.__name__ != 'RuntimeDevice':
        raise ValueError('use an isolated upstream Brian2 environment')
    b.start_scope()
    n, steps, inputs = samples.shape
    _, hidden, outputs = frozen['plan']['sizes']
    net = snn.Network('recurrent_diagnostic')
    source = snn.Population(n * inputs, 'spike_list',
        {r * inputs + i: np.flatnonzero(samples[r, :, i]).tolist()
         for r in range(n) for i in range(inputs)}, name='input')
    populations = [source]
    for role, size, threshold in zip(('hidden', 'output'), (hidden, outputs), frozen['plan']['threshold']):
        populations.append(snn.Population(n * size, 'lif_curr_exp_32bit',
            dict(alpha_decay=.5, exc_decay=.5, inh_decay=.5, threshold=threshold,
                 reset='reset_by_subtraction', t_refract=0, i_offset=0, v_init=0),
            name=role, record=['v', 'spikes']))
    net.add(*populations)
    banks = []
    for bank, (src, dst, ni, no) in enumerate(((0, 1, inputs, hidden), (1, 2, hidden, outputs), (1, 1, hidden, hidden))):
        weights = np.asarray(frozen['target_weights'][bank], dtype=np.int64).reshape(ni, no)
        # Use the stock 32-bit synapse format to avoid the ordinary LIF weight cap.
        limit = lif_curr_exp.LIFCurrExp32BitApplication.sw_spec.max_weight()
        if np.max(np.abs(weights)) > limit:
            raise ValueError('SDK would clip source weights')
        ii, jj = np.nonzero(weights)
        connections = [[r * ni + int(i), r * no + int(j), int(weights[i, j]), 0]
                       for r in range(n) for i, j in zip(ii, jj)]
        projection = snn.Projection(populations[src], populations[dst], connections, name=f'bank_{bank}')
        net.add(projection)
        # Verify all replicas and the absence of cross-replica connections.
        dense = projection.sparse_weights.tocsr()
        assert dense.nnz == n * len(ii)
        for r in range(n):
            np.testing.assert_array_equal(dense[r*ni:(r+1)*ni, r*no:(r+1)*no].toarray(), weights)
        banks.append(dict(bank=bank, nonzero=int(len(ii)), max_abs=int(np.max(np.abs(weights)))))
    backend = CurrentObserver()
    # One extra tick also exposes the SDK's documented current-LIF -1 tick view.
    backend.run(net, steps + 1, quantize_weights=True)
    result = {key: [] for key in ('spikes', 'membranes', 'currents')}
    for pop, size in zip(populations[1:], (hidden, outputs)):
        monitor = backend.state_monitors[pop]
        spike_monitor = backend.spike_monitors[pop]
        raw_times = np.asarray(spike_monitor.t / b.ms)
        ticks = np.rint(raw_times).astype(int)
        np.testing.assert_allclose(ticks, raw_times, rtol=0, atol=1e-8)
        spikes = np.zeros((steps + 1, n * size), dtype=np.uint8)
        np.add.at(spikes, (ticks, np.asarray(spike_monitor.i, dtype=int)), 1)
        assert np.max(spikes) <= 1
        for key, values in [('spikes', spikes), ('membranes', np.asarray(monitor.v).T),
                            ('currents', (np.asarray(monitor.I_syn_exc) - np.asarray(monitor.I_syn_inh)).T)]:
            result[key].append(values.reshape(steps + 1, n, size).transpose(1, 0, 2))
    return {k: np.concatenate(v, axis=2) for k, v in result.items()}, banks


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('frozen', 'inputs', 'output'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--batch-size', type=int, default=64)
    a = p.parse_args()
    if not 1 <= a.batch_size <= 128:
        raise ValueError('batch size must be 1..128')
    f = json.loads(a.frozen.read_text())
    x = np.load(a.inputs, allow_pickle=False)['spikes']
    assert f['schema'] == 'atlas.native-xylo-frozen.v2'
    assert x.ndim == 3 and len(x) and x.shape[2] == f['plan']['sizes'][0]
    assert np.isin(x, (0, 1)).all()
    a.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    arrays = {key: [] for key in ('spikes', 'membranes', 'currents')}
    for offset in range(0, len(x), a.batch_size):
        batch, banks = run_batch(f, x[offset:offset + a.batch_size])
        for key in arrays:
            arrays[key].append(batch[key])
    np.savez_compressed(a.output / 'results.npz', **{k: np.concatenate(v) for k, v in arrays.items()})
    report = dict(schema='atlas.recurrent-spinnaker2-diagnostic-execution.v1',
        dynamics_equivalent=False, stock_neuron_model='lif_curr_exp_32bit',
        stock_equations_unchanged=True, observation_extension='record I_syn_exc and I_syn_inh',
        weight_banks=banks, weights_unchanged=True, quantize_weights=True, weight_clipping=False,
        projection_delays=0, decay_parameters=.5, threshold_unchanged=True, reset='reset_by_subtraction',
        t_refract_parameter=0, sdk_refractory_duration_ticks=1,
        raw_ticks=int(x.shape[1] + 1), extra_tick_for_sdk_shifted_view=True,
        current_observation='I_syn_exc - I_syn_inh at end of raw tick',
        batch_size=a.batch_size, independent_replicas=True,
        versions={name: md.version(name) for name in ('spinnaker2', 'brian2', 'numpy')},
        sdk_source_hashes={Path(module.__file__).name: sha(module.__file__) for module in (brian2_sim, lif_curr_exp)},
        worker_sha256=sha(__file__), frozen_file_sha256=sha(a.frozen), inputs_sha256=sha(a.inputs),
        results_sha256=sha(a.output / 'results.npz'), execution_seconds=time.monotonic() - start,
        labels_or_native_reference_read=False, physical_hardware=False)
    (a.output / 'execution.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
