"""Isolated, optional execution of trusted SpiNNaker2 LIF conversion plans."""
import hashlib
from pathlib import Path

import numpy as np


def spike_ticks(times_in_steps):
    """Recover aligned SDK clock ticks without truncating float timestamps."""
    times=np.asarray(times_in_steps,dtype=np.float64)
    ticks=np.rint(times)
    if not np.isfinite(times).all() or not np.allclose(times,ticks,rtol=0,atol=1e-8):
        raise ValueError('SDK spike timestamp is not on the admitted clock')
    return ticks.astype(np.int64)


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
        raw_ticks = spike_ticks(monitor.t / b.ms)
        raw_spikes = np.column_stack((raw_ticks, np.asarray(monitor.i, dtype=np.int64)))
        return dict(
            voltage=np.stack([pop.get_voltages()[i] for i in range(pop.size)], axis=1),
            voltage_f64=np.asarray(backend.state_monitors[pop].v).T.copy(),
            final_v=np.asarray(backend.pops[pop].v[:]).copy(),
            # SDK get_spikes truncates t/dt and can report a tick too early.
            # Use its raw monitor clock; preserve the public result for audits.
            spikes=raw_spikes.reshape(-1, 2).copy(),
            sdk_public_spikes=np.asarray(public_spikes, dtype=np.int64).reshape(-1, 2),
            raw_spikes=raw_spikes.reshape(-1, 2),
            sdk_simulator_sha256=hashlib.sha256(Path(brian2_sim.__file__).read_bytes()).hexdigest(),
        )
    finally:
        b.prefs.codegen.loop_invariant_optimisations = previous


def run_native_batch(plan, bridge, samples, batch_size):
    """Independent replica batching of an admitted zero-input LIF template.

    Only input transport, replica offsets and observations occur here; all
    neural updates execute in the stock SDK backend. No labels are accepted.
    """
    import copy
    x=np.asarray(samples)
    if (bridge.get('schema') != 'atlas.native-lif-frozen.v1'
            or not bridge.get('fresh_state_per_sample') or not bridge.get('source_is_zero_input_template')):
        raise ValueError('unknown native training bridge contract')
    if type(batch_size) is not int or not 1 <= batch_size <= 128:
        raise ValueError('native LIF batch size must be in [1,128]')
    sizes=bridge['sizes'];cells=sum(sizes[1:]);steps=plan['steps']
    if (x.ndim != 3 or x.shape[0] < 1 or x.shape[1:] != (steps,sizes[0])
            or not np.isfinite(x).all() or not ((x == 0) | (x == 1)).all()):
        raise ValueError('native LIF inference requires binary batch × time × input')
    lif=plan['lif_population'];source=1-lif
    if (len(plan['populations']) != 2 or plan['populations'][lif]['size'] != cells
            or plan['populations'][source]['size'] != sizes[0]
            or any(plan['populations'][source]['params'].values())):
        raise ValueError('native LIF requires an empty input template with matching sizes')
    voltages=[];finals=[];rasters=[];sdk_hash=None
    for start in range(0,len(x),batch_size):
        chunk=x[start:start+batch_size];n=len(chunk);replica=copy.deepcopy(plan)
        for item in replica['populations']:
            item['size']*=n
            if item['neuron_model']=='lif':
                item['params']={k:v*n if isinstance(v,list) else v for k,v in item['params'].items()}
            else:
                item['params']={str(k*sizes[0]+i):np.flatnonzero(chunk[k,:,i]).tolist()
                                for k in range(n) for i in range(sizes[0])}
        for item in replica['projections']:
            src_size=plan['populations'][item['source']]['size']
            dst_size=plan['populations'][item['target']]['size']
            item['connections']=[[i+k*src_size,j+k*dst_size,w,d]
                                 for k in range(n) for i,j,w,d in item['connections']]
        result=run(replica);sdk_hash=result['sdk_simulator_sha256']
        raster=np.zeros((n,steps,cells),dtype=np.uint8)
        spikes=result['spikes']
        if len(spikes): raster[spikes[:,1]//cells,spikes[:,0],spikes[:,1]%cells]=1
        rasters.append(raster)
        voltages.append(result['voltage_f64'].reshape(steps,n,cells).transpose(1,0,2))
        finals.append(result['final_v'].reshape(n,cells))
    spikes=np.concatenate(rasters)
    counts=spikes[:,:,-sizes[-1]:].sum(axis=1,dtype=np.int64)
    return dict(spike_raster=spikes,voltage_f64=np.concatenate(voltages),
                final_v=np.concatenate(finals),counts=counts,
                logits=counts*bridge['logit_scale']/steps,sdk_simulator_sha256=sdk_hash)
