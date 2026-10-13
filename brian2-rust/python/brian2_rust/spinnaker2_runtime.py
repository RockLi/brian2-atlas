"""Isolated, optional execution of trusted SpiNNaker2 LIF conversion plans."""
import hashlib
from pathlib import Path

import numpy as np


def run(plan):
    import brian2 as b
    from spinnaker2 import brian2_sim, snn

    if plan.get('schema') != 'atlas.spinnaker2.lif-plan.v1':
        raise ValueError('unknown SpiNNaker2 plan schema')
    # The worker must use upstream Brian2's NumPy runtime, never the Atlas
    # Device. Matching codegen settings retain the source Euler operation order.
    if b.get_device().__class__.__name__ != 'RuntimeDevice':
        raise ValueError('SpiNNaker2 must run in an isolated Brian2 runtime environment')
    previous = b.prefs.codegen.loop_invariant_optimisations
    b.prefs.codegen.loop_invariant_optimisations = False
    try:
        b.start_scope()
        net = snn.Network('atlas_lif')
        pops = []
        for item in plan['populations']:
            params = item['params']
            is_lif = item['neuron_model'] == 'lif'
            if not is_lif:
                params = {int(k): v for k, v in params.items()}
            pop = snn.Population(item['size'], item['neuron_model'], params,
                                 name=item['name'], record=['v', 'spikes'] if is_lif else [])
            pops.append(pop)
            net.add(pop)
        for item in plan['projections']:
            net.add(snn.Projection(pops[item['source']], pops[item['target']], item['connections'],
                                   name=item['name']))
        backend = brian2_sim.Brian2Backend()
        backend.run(net, plan['steps'], quantize_weights=True)
        pop = pops[plan['lif_population']]
        public_spikes = sorted((tick, int(i)) for i, ticks in pop.get_spikes().items() for tick in ticks)
        monitor = backend.spike_monitors[pop]
        raw_ticks = np.rint(np.asarray(monitor.t / b.ms)).astype(np.int64)
        raw_spikes = np.column_stack((raw_ticks, np.asarray(monitor.i, dtype=np.int64)))
        return dict(
            voltage=np.stack([pop.get_voltages()[i] for i in range(pop.size)], axis=1),
            voltage_f64=np.asarray(backend.state_monitors[pop].v).T.copy(),
            final_v=np.asarray(backend.pops[pop].v[:]).copy(),
            spikes=np.asarray(public_spikes, dtype=np.int64).reshape(-1, 2),
            raw_spikes=raw_spikes.reshape(-1, 2),
            sdk_simulator_sha256=hashlib.sha256(Path(brian2_sim.__file__).read_bytes()).hexdigest(),
        )
    finally:
        b.prefs.codegen.loop_invariant_optimisations = previous
