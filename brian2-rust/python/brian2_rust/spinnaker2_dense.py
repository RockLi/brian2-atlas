"""Exact static integer MLP encoding for the stock SpiNNaker2 CPU SDK.

Seven successive binary decisions replace 127 parallel threshold comparators.
The SDK performs every affine sum, comparison and output reconstruction.
"""
import numpy as np


def graph_plan(dense):
    if dense['schema'] != 'atlas.akida.dense-int8-plan.v1':
        raise ValueError('requires a validated dense-int8 plan')
    step = 1 << np.asarray(dense['shift1'], dtype=np.int64)
    bias = (np.asarray(dense['bias1'], dtype=np.int64)
            * (1 << np.asarray(dense['bias_shift1'], dtype=np.int64)) + step // 2)
    w1, w2 = np.asarray(dense['weights1']), np.asarray(dense['weights2'])
    # lif_32bit has a 20-bit magnitude and 3-bit delay in the pinned SDK.
    maximum = max(int(np.abs(w1).max()) * 64, int(np.abs(w2).max()) * 64,
                  int(np.abs(bias).max()), int(step.max()) * 64)
    if maximum > (1 << 20) - 1:
        raise ValueError('SpiNNaker2 dense encoding exceeds 20-bit synaptic weight magnitude')
    plan = dict(dense)
    # Compiler probes are not runtime inputs or inference references.
    plan.pop('verification_inputs', None)
    plan.pop('verification_outputs', None)
    plan.update(schema='atlas.spinnaker2.dense-int8-plan.v1',
        neuron_model='lif_32bit', pipeline_delay_ticks=8, bit_order=[6,5,4,3,2,1,0],
        maximum_encoded_weight=maximum, maximum_encoded_delay=6,
        hidden_rounding_bias=bias.tolist(),
        resources=dict(input_neurons=len(dense['inputs'])*7+1,
            comparator_neurons=len(step)*7, readout_neurons=len(dense['outputs']),
            first_layer_connections=int(np.count_nonzero(w1))*49,
            bias_connections=int(np.count_nonzero(bias))*7,
            decision_connections=len(step)*21,
            readout_connections=int(np.count_nonzero(w2))*7),
        state_readback='SDK affine readout voltages, sliced at fixed eight-tick latency',
        event_encoding='seven binary input planes and a constant spike, one sample per tick',
        proof='For S=affine+half_step and bits k=6..0, emit bit k iff '
              'S-step*sum(previous bits*2**bit) > step*2**k-0.5. '
              'This greedy expansion equals clip(floor(S/step),0,127), including '
              'negative S and saturation. Delays align each sample; alpha=0 '
              'clears all voltages each tick. The readout sums weighted bits in the SDK.',
        hardware_mapping_tested=False)
    return plan


def run(plan, x):
    import brian2 as b
    from spinnaker2 import snn, brian2_sim
    from spinnaker2.neuron_models import NEURON_MODEL_TO_APPLICATION
    if type(b.get_device()).__name__ != 'RuntimeDevice':
        raise RuntimeError('SpiNNaker2 requires an isolated upstream Brian2 runtime')
    if plan['schema'] != 'atlas.spinnaker2.dense-int8-plan.v1':
        raise ValueError('unknown SpiNNaker2 dense plan')
    spec = NEURON_MODEL_TO_APPLICATION['lif_32bit'].sw_spec
    if spec.max_weight() < plan['maximum_encoded_weight'] or spec.max_delay() < 6:
        raise RuntimeError('installed SDK weight/delay contract differs from compiled plan')
    b.start_scope()
    w1, w2 = np.asarray(plan['weights1']), np.asarray(plan['weights2'])
    h = w1.shape[1]
    step = 1 << np.asarray(plan['shift1'], dtype=np.int64)
    bias = np.asarray(plan['hidden_rounding_bias'])
    b2 = np.asarray(plan['bias2']) * (1 << np.asarray(plan['bias_shift2']))
    spikes = {i*7+bit:np.flatnonzero((x[:,i] >> bit) & 1).tolist()
              for i in range(x.shape[1]) for bit in range(7)}
    spikes[x.shape[1]*7] = list(range(len(x)))
    source = snn.Population(x.shape[1]*7+1, 'spike_list', spikes, name='input_bits')
    net = snn.Network('atlas_dense_binary_decisions')
    net.add(source)
    stages = []

    def connect(pre, post, rows):
        # Avoid millions of Python list objects and omit zero-weight bias edges.
        rows = np.asarray(rows, dtype=np.int32).reshape(-1,4)
        rows = rows[rows[:,2] != 0]
        if len(rows):
            net.add(snn.Projection(pre, post, rows))

    ni, nj = np.nonzero(w1)
    for stage, bit in enumerate(plan['bit_order']):
        pop = snn.Population(h, 'lif_32bit',
            dict(alpha_decay=0., threshold=(step*(1 << bit)-.5).tolist(),
                 reset='reset_to_v_reset', v_reset=0., i_offset=0.), name=f'bit_{bit}')
        net.add(pop)
        stages.append(pop)
        # Source spike t arrives at stage k on t+k+1 under the SDK schedule.
        rows = [np.column_stack((ni*7+j, nj, w1[ni,nj]*(1 << j),
                                 np.full(len(ni),stage))) for j in range(7)]
        rows.append(np.column_stack((np.full(h,source.size-1), np.arange(h),
                                     bias, np.full(h,stage))))
        connect(source,pop,np.concatenate(rows))
        for earlier in range(stage):
            connect(stages[earlier],pop,np.column_stack((np.arange(h), np.arange(h),
                -step*(1 << (6-earlier)), np.full(h,stage-earlier-1))))
    readout = snn.Population(w2.shape[1], 'lif_32bit',
        dict(alpha_decay=0., threshold=float(2**24), i_offset=b2.tolist(),
             reset='reset_to_v_reset', v_reset=0.), name='affine_readout', record=['v'])
    net.add(readout)
    ni, nj = np.nonzero(w2)
    for stage, pop in enumerate(stages):
        connect(pop,readout,np.column_stack((ni,nj,w2[ni,nj]*(1 << (6-stage)),
                                            np.full(len(ni),6-stage))))
    backend = brian2_sim.Brian2Backend()
    backend.run(net,len(x)+8,quantize_weights=True)
    # Output observation only: no host weighted sum, activation, or rounding.
    states = np.stack([readout.get_voltages()[i] for i in range(readout.size)],axis=1)[8:len(x)+8]
    return dict(states=states)
