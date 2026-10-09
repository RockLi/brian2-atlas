"""Sparse zero-delay event expansion with deterministic target consumption.

Each active source expands its outgoing edges once. Integer atomic reservations
write ranks encoding the original target/source/edge order into unordered target
queues. The consumer sorts sparse ranks or scans the original ordered CSR before
performing floating-point work. Capacity
is exactly indegree: a threshold stream contains each source at most once, and
one enqueue/consume pair is completed before the next pathway reuses the queue.
"""
from .metal import MetalKernel, _PRELUDE
from .metal_event_layout import event_offset


SORT_SOURCE = '''
inline void b2_sift(device uint *data, uint base, uint root, uint count) {
    while (root < count/2) {
        uint child = 2*root+1;
        if (child+1<count && data[base+child]<data[base+child+1]) ++child;
        if (data[base+root]>=data[base+child]) return;
        uint value=data[base+root]; data[base+root]=data[base+child]; data[base+child]=value;
        root=child;
    }
}
inline void b2_sort_active(device uint *data, uint base, uint count) {
    for (uint root=count/2; root>0; --root) b2_sift(data,base,root-1,count);
    for (uint end=count; end>1; --end) {
        uint value=data[base]; data[base]=data[base+end-1]; data[base+end-1]=value;
        b2_sift(data,base,0,end-1);
    }
}
'''


# CPU controls use compiler atomic intrinsics on the same uint32 storage. No
# std::atomic object is overlaid on a NumPy allocation with an invalid lifetime.
CPU_ATOMICS = '''
using atomic_uint = uint;
constexpr int memory_order_relaxed = __ATOMIC_RELAXED;
inline uint atomic_fetch_add_explicit(uint *p, uint value, int order) {
    return __atomic_fetch_add(p,value,order);
}
inline uint atomic_fetch_or_explicit(uint *p, uint value, int order) { return __atomic_fetch_or(p,value,order); }
inline uint atomic_load_explicit(uint *p, int order) { return __atomic_load_n(p,order); }
inline void atomic_store_explicit(uint *p, uint value, int order) { __atomic_store_n(p,value,order); }
'''


def enqueue_kernel(model, logical, node, ordinal):
    syn = model["definition"]["synapses"][node.owner_index]
    code=syn['code_objects'][node.item_index]
    offset=event_offset(model['definition']['populations'][syn['source_population']],code['event_name'])
    entry = f"stage_{ordinal}_enqueue"
    source = _PRELUDE + f'''
kernel void {entry}(device const uchar *fired [[buffer(0)]],
    device const uint *source_offsets [[buffer(1)]], device const uint *source_ranks [[buffer(2)]],
    device const uint *rank_targets [[buffer(3)]], device const uint *target_offsets [[buffer(4)]],
    device uint *active [[buffer(5)]], device atomic_uint *counts [[buffer(6)]],
    constant long &tick [[buffer(7)]], uint i [[thread_position_in_grid]]) {{
    if (i >= {syn['source_count']}u || !fired[i+{syn['source_start']+offset}]) return;
    for (uint at=source_offsets[i]; at<source_offsets[i+1]; ++at) {{
        uint rank=source_ranks[at];
        uint target=rank_targets[rank];
        uint slot=atomic_fetch_add_explicit(&counts[target],1u,memory_order_relaxed);
        active[target_offsets[target]+slot]=rank;
    }}
}}
'''
    clock = logical.clocks[node.clock]
    return MetalKernel(syn["source_population"], entry, (), syn["source_count"],
                       clock.start_tick, clock.steps, 0, 0, source)
