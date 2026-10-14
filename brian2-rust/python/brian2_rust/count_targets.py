"""Bounded static count plans for SpiNNaker2 and XyloSim.

Consumes the existing exact binary-count admission proof. These encodings have
no source spike-time contract. They do not imply a hardware placement.
"""
import numpy as np


def graph_plan(count, target):
    if count['schema'] != 'atlas.specksim.binary-count-plan.v1':
        raise ValueError('requires an admitted binary-count plan')
    n, h = len(count['inputs']), len(count['hidden_states'])
    if not 1 <= n <= 1024 or not 1 <= h <= 32:
        raise ValueError('count target admits 1..1024 inputs and 1..32 class counts')
    peaks = count['binary_max_counts']
    plan = dict(count)
    plan['schema'] = f'atlas.{target}.binary-count-plan.v1'
    if target == 'spinnaker2':
        # One comparator for each possible integer quotient. At alpha=0,
        # Euler clears v exactly before each sample's weighted spike sum.
        banks = [[j, k] for j, peak in enumerate(peaks) for k in range(1, peak + 1)]
        plan.update(neuron_model='lif_no_delay', alpha_decay=0, reset='reset_to_v_reset',
                    comparators=banks, thresholds=[k*count['threshold']-.5 for _, k in banks],
                    pipeline_delay_ticks=1,
                    resources=dict(input_neurons=n, comparator_neurons=len(banks)),
                    event_encoding='one image per tick; one spike per active pixel',
                    proof='For integer S, sum(k=1..peak)[S > k*T-0.5] = floor(S/T). '
                          'alpha=0 clears state; all sums <=8191 are exactly representable.')
    elif target == 'xylosim':
        # Hidden cells emit at most 31 spikes. Offsets partition the quotient
        # into nonoverlapping groups of 31; unused hardware output is silent.
        banks = [[j, k] for j, peak in enumerate(peaks) for k in range((peak+30)//31)]
        plan.update(neuron_model='XyloSim v2 hidden IAF', banks=banks,
                    biases=[-31*k*count['threshold'] for _, k in banks],
                    weight_shift_in=1, dash_syn=1, dash_mem=0,
                    resource_limits=dict(input_channels=16, hidden_neurons=1000, output_neurons=8),
                    resources=dict(input_channels=n, hidden_neurons=max(1,len(banks)), output_neurons=1),
                    hardware_input_limit_exceeded=n>16,
                    readout='recorded hidden spike counts grouped by source state',
                    event_encoding='one image in one tick; reset all SDK state before each image',
                    proof='Fresh state: input current 2*S decays by (2*S)>>1 to S. '
                          'sum(k)[clip(floor((S-31*k*T)/T),0,31)] = floor(S/T). '
                          'S<=8191 ensures no signed 16-bit saturation, including shifted current.')
    else:
        raise ValueError('unsupported count target')
    plan['observation'] = 'sum of recorded spikes per class; final static values only'
    plan['hardware_mapping_tested'] = False
    return plan


def run_spinnaker2(plan, x):
    import brian2 as b
    from spinnaker2 import snn, brian2_sim
    if type(b.get_device()).__name__ != 'RuntimeDevice':
        raise RuntimeError('SpiNNaker2 requires a separate upstream Brian2 runtime environment')
    from spinnaker2.neuron_models import NEURON_MODEL_TO_APPLICATION
    app = NEURON_MODEL_TO_APPLICATION['lif_no_delay']
    if app.sw_spec.max_weight() < np.max(plan['weights']) or app.sw_spec.max_delay() != 0:
        raise RuntimeError('installed SDK weight/delay contract differs from the compiled plan')
    b.start_scope()
    h = len(plan['hidden_states'])
    banks = plan['comparators']
    counts = np.zeros((len(x),h),np.int64)
    if not banks:
        return dict(states=counts[:,plan['output_indices']], counts=counts)
    input_spikes = {i:np.flatnonzero(x[:,i]).tolist() for i in range(x.shape[1])}
    source = snn.Population(x.shape[1], 'spike_list', input_spikes, name='pixels')
    post = snn.Population(len(banks), 'lif_no_delay',
        dict(alpha_decay=0., threshold=plan['thresholds'], reset='reset_to_v_reset', v_reset=0., i_offset=0.),
        name='count_comparators', record=['spikes'])
    weights = np.asarray(plan['weights'])
    connections = [[int(i),j,int(weights[c,i]),0]
                   for j,(c,_) in enumerate(banks) for i in np.flatnonzero(weights[c])]
    net = snn.Network('atlas_binary_count')
    net.add(source, post, snn.Projection(source,post,connections))
    backend = brian2_sim.Brian2Backend()
    backend.run(net,len(x)+plan['pipeline_delay_ticks'],quantize_weights=True)
    # Read exact ticks from the SDK's monitor. Public get_spikes uses int(t/dt),
    # which can truncate a float timestamp by one tick on long runs.
    monitor = backend.spike_monitors[post]
    ticks = np.rint(np.asarray(monitor.t/b.ms)).astype(np.int64)-plan['pipeline_delay_ticks']
    neurons = np.asarray(monitor.i,dtype=np.int64)
    if np.any((ticks<0)|(ticks>=len(x))):
        raise RuntimeError('unexpected output outside encoded sample window')
    classes = np.array([c for c,_ in banks],np.int64)
    np.add.at(counts,(ticks,classes[neurons]),1)
    return dict(states=counts[:,plan['output_indices']],counts=counts,
                output_spikes=np.array(len(ticks),np.int64))


def run_xylosim(plan, x):
    from xylosim.v2 import XyloLayer, XyloSynapse
    banks = plan['banks']; h = len(plan['hidden_states'])
    neurons = max(1,len(banks))
    weights = np.asarray(plan['weights'])
    # Keep pybind-owned synapses alive throughout SDK execution.
    synapses = [[XyloSynapse(j,0,int(weights[c,i])) for j,(c,_) in enumerate(banks) if weights[c,i]]
                for i in range(x.shape[1])]
    layer = XyloLayer(synapses_in=synapses, synapses_rec=[[] for _ in range(neurons)],
        synapses_out=[[] for _ in range(neurons)], aliases=[[] for _ in range(neurons)],
        threshold=[plan['threshold']]*neurons, threshold_out=[1], has_bias=True,
        bias=plan['biases'] or [0], bias_out=[0], weight_shift_inp=plan['weight_shift_in'],
        weight_shift_rec=0,weight_shift_out=0,dash_mem=[0]*neurons,dash_mem_out=[0],
        dash_syns=[[1,0] for _ in range(neurons)],dash_syns_out=[[0]],name='atlas_binary_count')
    counts = np.zeros((len(x),h),np.int64)
    bank_counts = np.zeros((len(x),len(banks)),np.int64)
    currents = np.zeros_like(bank_counts); voltage = np.zeros_like(bank_counts)
    for i,row in enumerate(x):
        layer.reset_all()
        layer.evolve([row.tolist()])
        if banks:
            bank_counts[i] = np.asarray(layer.rec_recurrent_spikes)[0]
            currents[i] = np.asarray(layer.rec_i_syn)[:,0]
            voltage[i] = np.asarray(layer.rec_v_mem)[:,0]
        if i%1000 == 0:
            print(f'XyloSim {i+1}/{len(x)} samples',flush=True)
    # Only spike observation/reordering on host; no weighted sum or floor.
    for j,(c,_) in enumerate(banks):
        counts[:,c] += bank_counts[:,j]
    return dict(states=counts[:,plan['output_indices']],counts=counts,
                bank_spikes=bank_counts,bank_current=currents,bank_voltage=voltage)
