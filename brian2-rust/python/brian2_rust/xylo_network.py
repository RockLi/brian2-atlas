"""Build the shared, exact integer Xylo AtlasIR subset (no SDK dependency)."""
import numpy as np

UPDATE = """I = I - (floor(I/syn_div) + int(I>0)*int(I<syn_div))
delta = clip(-(floor(v/mem_div) + int(v>0)*int(v<mem_div)) + I, -32768, 32767)
delta = clip(delta+bias, -32768, 32767)
v = clip(v+delta, -32768, 32767)
q = clip(floor(v/threshold), 0, spike_cap)"""


def build_xylo_network(sizes, weight_banks, thresholds, raster):
    """Arguments are checked by each source bridge before construction."""
    import brian2 as b
    clock = b.Clock(dt=b.ms)
    table = b.TimedArray(raster.astype(float), dt=b.ms)
    source = b.NeuronGroup(sizes[0], 'q:1', threshold='q>0', clock=clock,
                          name='xylo_input*', namespace={'stimulus': table})
    source.run_regularly('q=stimulus(t,i)', when='start', order=0)
    source.thresholder['spike'].when = 'start'
    source.thresholder['spike'].order = 1
    objects = [source]
    groups = []
    for layer, (role, cap) in enumerate((('hidden', 31), ('output', 1))):
        group = b.NeuronGroup(sizes[layer+1],
            'I:1\nv:1\nq:1\nsyn_div:1 (constant)\nmem_div:1 (constant)\n'
            'bias:1 (constant)\nthreshold:1 (constant)\nspike_cap:1 (constant)',
            threshold='q>0', reset='v=v-q*threshold', clock=clock, name='xylo_'+role+'*')
        group.syn_div = 2
        group.mem_div = 2
        group.bias = 0
        group.threshold = thresholds[layer]
        group.spike_cap = cap
        group.run_regularly(UPDATE, when='groups')
        objects += [group, b.SpikeMonitor(group),
                    b.StateMonitor(group, ['I', 'v', 'q'], record=True, when='end')]
        groups.append(group)
    for layer, (pre, post, name) in enumerate(((source, groups[0], 'a_xylo_input*'),
                                              (groups[0], groups[1], 'c_xylo_readout*'))):
        weights = np.asarray(weight_banks[layer]).reshape(sizes[layer], sizes[layer+1])
        i, j = np.nonzero(weights)
        syn = b.Synapses(pre, post, 'w:1 (constant)',
            on_pre='I_post=clip(I_post+w*q_pre,-32768,32767)', clock=clock, name=name)
        syn.connect(i=i, j=j)
        syn.w = weights[i, j]
        objects.append(syn)
    if len(weight_banks)==3:
        weights=np.asarray(weight_banks[2]).reshape(sizes[1],sizes[1]);i,j=np.nonzero(weights)
        if len(i):
            syn=b.Synapses(groups[0],groups[0],'w:1 (constant)',
                on_pre='I_post=clip(I_post+w*q_pre,-32768,32767)',clock=clock,name='b_xylo_recurrent*')
            syn.connect(i=i,j=j);syn.w=weights[i,j];objects.append(syn)
    network = b.Network(*objects)
    network.schedule = ['start', 'synapses', 'groups', 'thresholds', 'resets', 'end']
    return network
