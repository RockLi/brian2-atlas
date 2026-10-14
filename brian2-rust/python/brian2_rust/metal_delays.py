"""Bounded source-history rings and sparse queues for delayed pre pathways.

An edge's arrival tick determines its emission tick uniquely. Sorting incoming
edges by decreasing delay, then source and creation index, therefore recovers
the reference queue's insertion order without floating atomics or runtime sort.
Pending events predate this run and are consumed first in their supplied order.
The scan route checks incoming edges; the sparse route expands active source /
delay groups into bounded target queues. Both sample history at the pathway slot.
"""
from dataclasses import replace
import re

import numpy as np

from .metal import MetalKernel, _PRELUDE
from .metal_event_layout import event_offset


BUFFER_KINDS = ("delays", "ordered_edges", "history", "pending_offsets",
                "pending_edges", "pending_ticks", "pending_cursor")
SPARSE_BUFFER_KINDS = ("source_group_offsets", "group_delays", "group_offsets",
                       "group_ranks", "rank_targets")


def pathway_for_node(model, node):
    code = model["definition"]["synapses"][node.owner_index]["code_objects"][node.item_index]
    if code["kind"] not in {"synapses", "synapses_post"}:
        return None
    return next(path for path in model["instance"]["synapses"][node.owner_index]["pathways"]
                if path["name"] == code["pathway_name"])


def needs_delay(path):
    return path is not None and (any(path["delay_ticks"]) or bool(path["pending"]))


def ring_slots(path, clock):
    return max(1, min(max(path["delay_ticks"], default=0)+1, clock.steps))


def history_kernel(model, logical, node, ordinal):
    syn = model["definition"]["synapses"][node.owner_index]
    offset=event_offset(model['definition']['populations'][syn['source_population']],pathway_for_node(model,node)['event'])
    clock = logical.clocks[node.clock]
    slots = ring_slots(pathway_for_node(model, node), clock)
    entry = f"stage_{ordinal}_delay_enqueue"
    source = _PRELUDE + f'''
kernel void {entry}(device const uchar *fired [[buffer(0)]],
    device uchar *history [[buffer(1)]], constant long &tick [[buffer(2)]],
    uint i [[thread_position_in_grid]]) {{
    if (i >= {syn['source_count']}u) return;
    history[ulong(tick%{slots})*{syn['source_count']}+i] = fired[i+{syn['source_start']+offset}];
}}
'''
    return MetalKernel(syn["source_population"], entry, (), syn["source_count"],
                       clock.start_tick, clock.steps, 0, 0, source)


def sparse_history_kernel(model, logical, node, ordinal, *, saturate=False, bitset=False, compact_bitset=False):
    """One source owns its history and all its delay groups, so no extra barrier.

    Each edge can arrive at most once per tick, even with heterogeneous delays.
    Thus target indegree bounds the active queue. Ranks use decreasing delay,
    source and creation index: for this arrival tick they encode enqueue order.

    ``saturate`` is a private experiment, selected only by the comparison harness.
    Native validation passed, but dense replay regressed; production keeps False.
    ``bitset`` scans ordered bits instead of sorting rank reservations. Public
    bitmap plans set ``compact_bitset`` and use immutable word offsets in binding
    nine; the earlier private experiment retains its original rank-sized layout.
    Neither bitmap layout can use saturation.
    """
    if compact_bitset and not bitset:raise ValueError('Compact bitmaps require bitset enqueue')
    if bitset and saturate:raise ValueError('Bitset enqueue cannot saturate rank reservations')
    syn = model["definition"]["synapses"][node.owner_index]
    offset=event_offset(model['definition']['populations'][syn['source_population']],pathway_for_node(model,node)['event'])
    clock = logical.clocks[node.clock]
    slots = ring_slots(pathway_for_node(model, node), clock)
    entry = f"stage_{ordinal}_delay_sparse_enqueue"
    reservation="""uint slot=atomic_fetch_add_explicit(&counts[target],1u,memory_order_relaxed);
            active[target_offsets[target]+slot]=rank;"""
    if saturate:
        # ceil(indegree/4) is the consumer's dense-fallback threshold. Avoid
        # degree+3 overflow, and never skip events before that threshold.
        # Concurrent stale loads can overshoot the threshold, but counts only
        # increase and never exceed the number of actual current edges.
        reservation="""uint degree=target_offsets[target+1]-target_offsets[target];
            uint limit=degree/4+uint(degree%4!=0);
            if (atomic_load_explicit(&counts[target],memory_order_relaxed)>=limit) continue;
            uint slot=atomic_fetch_add_explicit(&counts[target],1u,memory_order_relaxed);
            if (slot<limit) active[target_offsets[target]+slot]=rank;"""
    if bitset:
        # Each target owns ceil(indegree/32) words at the start of its existing
        # rank allocation. Different producers atomically set distinct bits;
        # the ordered consumer clears the words before the next producer stage.
        reservation="""uint base=target_offsets[target], relative=rank-base;
            atomic_fetch_or_explicit((device atomic_uint *)&active[base+relative/32],
                1u<<(relative%32),memory_order_relaxed);"""
    if compact_bitset:
        reservation=reservation.replace('active[base+relative/32]','active[counts[target]+relative/32]')
    counts_type='const uint' if compact_bitset else 'atomic_uint'
    source = _PRELUDE + f'''
kernel void {entry}(device const uchar *fired [[buffer(0)]],
    device uchar *history [[buffer(1)]], device const uint *source_groups [[buffer(2)]],
    device const uint *group_delays [[buffer(3)]], device const uint *group_offsets [[buffer(4)]],
    device const uint *group_ranks [[buffer(5)]], device const uint *rank_targets [[buffer(6)]],
    device const uint *target_offsets [[buffer(7)]], device uint *active [[buffer(8)]],
    device {counts_type} *counts [[buffer(9)]], constant long &tick [[buffer(10)]],
    uint i [[thread_position_in_grid]]) {{
    if (i >= {syn['source_count']}u) return;
    history[ulong(tick%{slots})*{syn['source_count']}+i] = fired[i+{syn['source_start']+offset}];
    for (uint group=source_groups[i]; group<source_groups[i+1]; ++group) {{
        long emission=tick-long(group_delays[group]);
        if (emission < {clock.start_tick} || !history[ulong(emission%{slots})*{syn['source_count']}+i]) continue;
        for (uint at=group_offsets[group]; at<group_offsets[group+1]; ++at) {{
            uint rank=group_ranks[at];
            uint target=rank_targets[rank];
            {reservation}
        }}
    }}
}}
'''
    return MetalKernel(syn["source_population"], entry, (), syn["source_count"],
                       clock.start_tick, clock.steps, 0, 0, source)


def sparse_delay_arrays(syn, inst, delays, ordered_edges, max_buffer_bytes):
    count, ns = len(inst["source"]), syn["source_count"]
    # Worst case one group per edge; validate before building sorted tables.
    if max((count+1)*4, (ns+1)*4) > max_buffer_bytes:
        raise MemoryError("Metal sparse delay groups exceed configured memory limit")
    source = np.asarray(inst["source"], np.uint32)
    order = np.lexsort((np.arange(count), delays, source))
    sorted_source, sorted_delays = source[order], delays[order]
    starts = np.flatnonzero(np.r_[True, (sorted_source[1:] != sorted_source[:-1]) |
                                (sorted_delays[1:] != sorted_delays[:-1])]) if count else np.empty(0, np.int64)
    group_sources = sorted_source[starts]
    source_groups = np.concatenate(([0], np.cumsum(np.bincount(group_sources.astype(np.int64), minlength=ns)))).astype(np.uint32)
    ranks = np.empty(count, np.uint32)
    ranks[ordered_edges] = np.arange(count, dtype=np.uint32)
    return [source_groups, sorted_delays[starts], np.r_[starts, count].astype(np.uint32),
            ranks[order], np.asarray(inst["target"], np.uint32)[ordered_edges]]


def fuse_source_enqueue(population, enqueue, population_types, enqueue_types):
    """Fuse adjacent full-population operations whose dependency is lane-local.

    Caller proves source_start=0 and equal population/source lane domains. Each
    lane writes its fired flag before consuming that same flag and history row.
    Queue reservations may cross targets, but target consumption retains its
    separate dispatch/barrier. Subgroups must not use this transformation.
    """
    def helper(kernel):
        source = kernel.source.replace(_PRELUDE, "").replace("kernel void", "inline void")
        return re.sub(r"\[\[(?:buffer\(\d+\)|thread_position_in_grid)\]\]", "", source)
    types = population_types+enqueue_types
    arguments = ",\n    ".join(f"device {dtype} *b{i} [[buffer({i})]]" for i,dtype in enumerate(types))
    count = len(population_types)
    pop_args = ", ".join(f"b{i}" for i in range(count))
    enqueue_args = ", ".join(f"b{i}" for i in range(count,len(types)))
    entry = population.entry+"_with_delay_enqueue"
    source = _PRELUDE+helper(population)+helper(enqueue)+f'''
kernel void {entry}({arguments}, constant long &tick [[buffer({len(types)})]],
    uint i [[thread_position_in_grid]]) {{
    {population.entry}({pop_args},tick,i);
    {enqueue.entry}({enqueue_args},tick,i);
}}
'''
    return replace(population,entry=entry,source=source)


def delay_arrays(syn, inst, path, clock, max_buffer_bytes):
    """Check ring and expanded pending capacities before allocating either."""
    ns, nt, count = syn["source_count"], syn["target_count"], len(inst["source"])
    slots = ring_slots(path, clock)
    if max(ns*slots, count*4, (nt+1)*4) > max_buffer_bytes:
        raise MemoryError("Metal delay ring/topology exceeds configured memory limit")
    source, target = np.asarray(inst["source"], np.uint32), np.asarray(inst["target"], np.uint32)
    raw = path["delay_ticks"]
    delays = np.full(count, raw[0], np.uint32) if len(raw) == 1 else np.asarray(raw, np.uint32)
    uniform = len(raw) > 0 and all(d == raw[0] for d in raw)
    source_edges = np.argsort(source, kind="stable")
    source_offsets = np.concatenate(([0], np.cumsum(np.bincount(source.astype(np.int64), minlength=ns))))
    pending_count = sum(int(source_offsets[e["item"]+1]-source_offsets[e["item"]]) if uniform else 1
                        for e in path["pending"])
    if pending_count > np.iinfo(np.uint32).max or pending_count*8 > max_buffer_bytes:
        raise MemoryError("Metal expanded pending delay events exceed configured memory limit")
    pending_edges = np.empty(pending_count, np.uint32)
    pending_ticks = np.empty(pending_count, np.int64)
    at = 0
    for event in path["pending"]:
        item = event["item"]
        edges = source_edges[source_offsets[item]:source_offsets[item+1]] if uniform else [item]
        end = at+len(edges)
        pending_edges[at:end] = edges
        pending_ticks[at:end] = event["delivery_tick"]
        at = end
    # Stable for equal target/tick: never reconstruct chronology from edge IDs.
    order = np.lexsort((np.arange(pending_count), pending_ticks, target[pending_edges]))
    pending_offsets = np.concatenate(([0], np.cumsum(np.bincount(target[pending_edges].astype(np.int64), minlength=nt)))).astype(np.uint32)
    ordered_edges = np.lexsort((np.arange(count), source, -delays.astype(np.int64), target)).astype(np.uint32)
    return [delays, ordered_edges, np.zeros((slots, ns), np.uint8), pending_offsets,
            pending_edges[order], pending_ticks[order], np.zeros(nt, np.uint32)]
