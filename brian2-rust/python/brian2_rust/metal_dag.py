"""Canonical, target-owned Metal dispatches for a bounded coupled subset.

The coupled profile supports clock-coalesced scheduling and explicit topology. Immutable
projections use target ownership; order-dependent pre/post pathways retain a
canonical GPU lane. Independent synapse-state nodes use one lane per edge.
Explicit buffer barriers synchronize successive canonical stages. No float
atomics or tree reassociation are used. Event delivery follows source/edge order,
whereas summed follows creation order, as in the reference executor.
"""
from dataclasses import asdict, dataclass, replace
import ctypes
import hashlib
import platform
import re
import subprocess
import time

import numpy as np

from .metal import (MetalKernel, MetalPlan, METAL_PROFILE, _kernel, _Expressions,
                    _PRELUDE, _literal, number, population_types)
from .plan import PlanValidationError
from .metal_events import SORT_SOURCE, CPU_ATOMICS, enqueue_kernel
from .metal_delays import (BUFFER_KINDS, pathway_for_node, needs_delay,
                           ring_slots, history_kernel, delay_arrays,
                           SPARSE_BUFFER_KINDS, sparse_history_kernel, sparse_delay_arrays,
                           fuse_source_enqueue)
from .metal_synapses import canonical_projection, canonical_kernel, canonical_delay_arrays, endpoint, pathway_history_kernel
from .metal_random import RNG_PROFILE, has_random
from .metal_timed_array import TimedTables
from .gpu_initialization import initialization_records, initialization_seconds
from . import gpu_types as gt
from . import gpu_links
from .metal_monitors import monitors, buffer_names, monitor_arrays, monitor_kernel, monitor_results
from .metal_event_layout import event_lanes, event_offset, event_coordinates
from .gpu_dispatch_fusion import fuse_dispatches


@dataclass(frozen=True)
class Dispatch:
    entry: str
    lanes: int
    bindings: tuple[int, ...]
    types: tuple[str, ...]
    dependencies: tuple[str, ...]
    role: str = "compute"
    clock: int = 0


def _population_kernel(model, logical, nodes, ordinal):
    return _kernel(model, replace(logical, nodes=tuple(nodes)), nodes[0].owner_index,
                   entry=f"stage_{ordinal}", single_tick=True, dispatch_clock=nodes[0].clock)


def _synapse_kernel(model, logical, node, ordinal, *, event_delivery):
    d = model["definition"]
    q = node.owner_index
    syn = d["synapses"][q]
    inst = model["instance"]["synapses"][q]
    code = syn["code_objects"][node.item_index]
    summed = code["kind"] == "summed_variable"
    if not summed and code["kind"] != "synapses":
        raise PlanValidationError("Metal DAG supports only immutable pre pathways and post summed reductions")
    if summed and code["summed_target"] != "post":
        raise PlanValidationError("Metal DAG summed requires post ownership")
    source, target = syn["source_population"], syn["target_population"]
    pre, post = d["populations"][source], d["populations"][target]
    writes = set(code["effects"]["writes"])
    if any(name not in syn["post_state_aliases"] for name in writes):
        raise PlanValidationError("Metal DAG pathways may only write target neuron states; plasticity is unsupported")
    written = {syn["post_state_aliases"][name] for name in writes}
    if len(written) != len(writes):
        raise PlanValidationError("Metal DAG rejects multiple write aliases for one target state")
    if summed:
        written.add(code["summed_state"])
    # No lane may read a state concurrently written by a different lane.
    for name in code["effects"]["reads"]:
        if source == target and syn["pre_state_aliases"].get(name) in written:
            raise PlanValidationError("Metal DAG rejects cross-lane read/write aliasing")
        if summed and syn["post_state_aliases"].get(name) in written:
            raise PlanValidationError("Metal DAG summed cannot read its destination")
    symbols = {"i": "float(pre_index)", "j": "float(i)", "N": f"float({len(inst['source'])})",
               "N_pre": f"float({syn['source_count']})", "N_post": f"float({syn['target_count']})",
               "t": "time", "dt": _literal(number(logical.clocks[node.clock].dt))}
    for side, pop, buffer, index in (("pre", pre, "pre_state", "source_index"), ("post", post, "post_state", "target_index")):
        positions = {state["name"]: n for n, state in enumerate(pop["states"])}
        for alias, name in syn[f"{side}_state_aliases"].items():
            symbols[alias] = f"{buffer}[{positions[name]*pop['count']}+{index}]"
    offset = 0
    for state in syn["states"] + syn["parameters"]:
        scalar = state["index_domain"] == "scalar"
        symbols[state["name"]] = f"values[{offset}{'' if scalar else '+edge'}]"
        offset += 1 if scalar else len(inst["source"])
    block = _Expressions(symbols, {f["name"]: f for f in d["functions"]}, math_error="math_error")
    block.statements(code["scalar"])
    block.statements(code["vector"])
    body = list(block.lines)
    if summed:
        position = next(i for i, state in enumerate(post["states"]) if state["name"] == code["summed_state"])
        location = f"post_state[{position*post['count']}+target_index]"
        body.append(f"total = b2_finite(total + {block.symbols['_synaptic_var']}, &math_error);")
        before, after = "float total = 0.0f;", f"{location} = total;"
    else:
        for name in sorted(writes):
            body.append(f"{symbols[name]} = {block.symbols[name]};")
        body.append("delivered[i] += 1;")
        before = after = ""
    path = pathway_for_node(model, node)
    delayed = needs_delay(path)
    delayed_sparse = delayed and event_delivery == "sparse"
    sparse = not summed and not delayed and event_delivery == "sparse"
    arguments = "device uint *active [[buffer(8)]], device atomic_uint *active_counts [[buffer(9)]], " if sparse else ""
    tick_binding = 10 if sparse else 8
    prepare = ""
    loop = "for (uint at=offsets[i]; at<offsets[i+1]; ++at)"
    edge = "edges[at]"
    pending_body = ""
    offset = event_offset(pre,path['event']) if path is not None else 0
    event_check = "" if summed else f"if (!fired[{offset}+source_index]) continue;"
    if delayed:
        clock = logical.clocks[node.clock]
        slots = ring_slots(path, clock)
        arguments = """device const uint *delays [[buffer(8)]], device const uchar *history [[buffer(9)]],
    device const uint *pending_offsets [[buffer(10)]], device const uint *pending_edges [[buffer(11)]],
    device const long *pending_ticks [[buffer(12)]], device uint *pending_cursor [[buffer(13)]], """
        tick_binding = 14
        event_check = f"""long emission = tick-long(delays[edge]);
        if (emission < {clock.start_tick} || !history[ulong(emission%{slots})*{syn['source_count']}+pre_index]) continue;"""
        pending_body = f"""uint cursor = pending_offsets[i]+pending_cursor[i];
    while (cursor < pending_offsets[i+1] && pending_ticks[cursor] <= tick) {{
        uint edge = pending_edges[cursor++];
        uint pre_index = sources[edge];
        uint source_index = pre_index + {syn['source_start']};
        {' '.join(body)}
    }}
    pending_cursor[i] = cursor-pending_offsets[i];"""
    if sparse:
        prepare = """uint count=atomic_load_explicit(&active_counts[i],memory_order_relaxed);
    if (!count) return;
    uint base=offsets[i];
    uint degree=offsets[i+1]-base;
    bool compact = ulong(count)*4 < degree;
    if (compact) b2_sort_active(active,base,count);"""
        loop = "for (uint at=0; at<(compact ? count : degree); ++at)"
        edge = "edges[compact ? active[base+at] : base+at]"
        after += "atomic_store_explicit(&active_counts[i],0u,memory_order_relaxed);"
    if delayed_sparse:
        arguments += "device uint *active [[buffer(14)]], device atomic_uint *active_counts [[buffer(15)]], "
        tick_binding = 16
        # Pending is independent of the current queue: never skip it on count=0.
        prepare = """uint count=atomic_load_explicit(&active_counts[i],memory_order_relaxed);
    uint base=offsets[i];
    uint degree=offsets[i+1]-base;
    bool compact=ulong(count)*4 < degree;
    if (compact) b2_sort_active(active,base,count);"""
        loop = "for (uint at=0; at<(compact ? count : degree); ++at)"
        edge = "edges[compact ? active[base+at] : base+at]"
        event_check = "if (!compact) { " + event_check + " }"
        after += "atomic_store_explicit(&active_counts[i],0u,memory_order_relaxed);"
    entry = f"stage_{ordinal}"
    text = _PRELUDE + (SORT_SOURCE if sparse or delayed_sparse else "") + f'''
kernel void {entry}(device const float *pre_state [[buffer(0)]],
    device float *post_state [[buffer(1)]], device const uchar *fired [[buffer(2)]],
    device const float *values [[buffer(3)]], device const uint *offsets [[buffer(4)]],
    device const uint *edges [[buffer(5)]], device const uint *sources [[buffer(6)]],
    device ulong *delivered [[buffer(7)]], {arguments}constant long &tick [[buffer({tick_binding})]],
    uint i [[thread_position_in_grid]]) {{
    if (i >= {syn['target_count']}u) return;
    uint target_index = i + {syn['target_start']};
    bool math_error = false;
    float time = float(tick)*{_literal(number(logical.clocks[node.clock].dt))};
    {prepare}
    {before}
    {pending_body}
    {loop} {{
        uint edge = {edge};
        uint pre_index = sources[edge];
        uint source_index = pre_index + {syn['source_start']};
        {event_check}
        {' '.join(body)}
    }}
    {after}
    if (math_error) delivered[i] |= 0x8000000000000000ul;
}}
'''
    clock = logical.clocks[node.clock]
    return MetalKernel(target, entry, (node.id,), syn["target_count"], clock.start_tick,
                       clock.steps, 0, 0, text)


def derive_dag(model, logical, *, event_delivery="scan", synapse_prefix=False, synapse_fusion=False, synapse_sparse=False, _keep_unused_projection_queues=False):
    d = model["definition"]
    canonical = {q for q in range(len(d["synapses"])) if canonical_projection(model,q)}
    for syn, inst in zip(d["synapses"], model["instance"]["synapses"], strict=True):
        if inst.get("topology", {"kind": "explicit"})["kind"] != "explicit":
            raise PlanValidationError("Metal DAG requires explicit topology")
        for path in inst["pathways"]:
            if path["kind"] not in {"pre","post"}:
                raise PlanValidationError("Metal DAG requires pre/post event pathways")
            if path['event'] not in d['populations'][endpoint(syn,path)[0]]['events']:
                raise PlanValidationError('Metal DAG pathway event must be declared by its endpoint')
            if path.get("delay_initializer") is not None:
                raise PlanValidationError("Metal DAG requires materialized integer delays")
    # Buffer inventory is stable and hashed with each dispatch binding.
    buffers = [f"population/{p}/{i}" for p in range(len(d["populations"])) for i in range(12)]
    syn_buffers, delay_buffers, target_buffers, pre_summed_buffers = [], {}, {}, {}
    for q in range(len(d["synapses"])):
        syn_buffers.append(len(buffers))
        buffers += [f"synapse/{q}/{kind}" for kind in ("values", "offsets", "event_edges", "summed_edges", "sources", "delivered")]
        # Canonical pathways own separate delay/target queues; no dispatch binds
        # these generic projection queues. The private retention switch exists
        # only for same-model allocation ablation against the former inventory.
        if event_delivery == "sparse" and (q not in canonical or _keep_unused_projection_queues):
            buffers += [f"synapse/{q}/{kind}" for kind in ("source_offsets", "source_ranks", "rank_targets", "active_ranks", "active_counts")]
        if q in canonical:
            target_buffers[q] = len(buffers)
            buffers.append(f"synapse/{q}/canonical_targets")
            if any(c['kind']=='summed_variable' and c['summed_target']=='pre' for c in d['synapses'][q]['code_objects']):
                pre_summed_buffers[q] = len(buffers)
                buffers += [f"synapse/{q}/pre_summed_offsets", f"synapse/{q}/pre_summed_edges"]
        for r, path in enumerate(model["instance"]["synapses"][q]["pathways"]):
            if needs_delay(path) or q in canonical:
                delay_buffers[q, path["name"]] = len(buffers)
                buffers += [f"synapse/{q}/pathway/{r}/{kind}" for kind in BUFFER_KINDS]
                if event_delivery == "sparse" and q not in canonical:
                    buffers += [f"synapse/{q}/pathway/{r}/{kind}" for kind in SPARSE_BUFFER_KINDS]
    monitor_buffers = {}
    for p,m,pop,monitor in monitors(model):
        monitor_buffers[p,m] = len(buffers)
        buffers.extend(buffer_names(p,m))
    link_buffers={}
    for p,pop in enumerate(d['populations']):
        if pop.get('linked_variables'):
            link_buffers[p]=len(buffers);buffers.extend(gpu_links.buffer_names(p))
    kernels, dispatches = [], []
    groups, elided = [], []
    recorded_populations = set()
    for node in logical.nodes:
        if node.operation == 'state_monitor':
            if node.owner_index in recorded_populations:
                elided.append(node.id)
                continue
            recorded_populations.add(node.owner_index)
        if node.owner_kind == "population" and node.operation == "code_object":
            pop = d["populations"][node.owner_index]
            code = pop["code_objects"][node.item_index]
            if code["kind"] == "state_update" and pop["refractory"] is None and not code["scalar"] and not code["vector"]:
                elided.append(node.id)
                continue
        if (groups and node.owner_kind == "population" and groups[-1][0].owner_kind == "population"
                and node.operation != 'event_monitor' and groups[-1][0].operation != 'event_monitor'
                and not d['populations'][node.owner_index].get('linked_variables')
                and (node.owner_index,node.clock) == (groups[-1][0].owner_index,groups[-1][0].clock)):
            groups[-1].append(node)
        else:
            groups.append([node])
    # Keep one identity dispatch for an entirely inert network so the normal
    # DAG result/checkpoint path still returns all populations and synapses.
    if not groups and logical.nodes:
        groups = [[logical.nodes[0]]]
        elided.remove(logical.nodes[0].id)

    def append(kernel, bindings, types, role, clock):
        dependencies = (dispatches[-1].entry,) if dispatches else ()
        kernels.append(kernel)
        dispatches.append(Dispatch(kernel.entry,kernel.neurons,bindings,types,dependencies,role,clock))

    prefix_masks={};sparse_queues={}
    for ordinal, nodes in enumerate(groups):
        node = nodes[0]
        serial_links=gpu_links.serial_self_links(model,node)
        for link in gpu_links.links_for_node(model,node):
            if link in serial_links:continue
            cache=link_buffers[node.owner_index]
            gather=gpu_links.gather_kernel(model,logical,node,ordinal,link)
            bindings=(link['source_population']*12,node.owner_index*12,cache+1,cache,
                      node.owner_index*12+8,node.owner_index*12+6,node.owner_index*12+3)
            append(gather,bindings,('const float','const float','const ulong','float','long','const uchar','const int'),
                   'linked-gather',node.clock)
        if node.operation == 'event_monitor':
            kernel = monitor_kernel(model,logical,node,ordinal)
            base = monitor_buffers[node.owner_index,node.item_index]
            bindings = (node.owner_index*12,node.owner_index*12+1,node.owner_index*12+6,base,base+1)
            types = ('const float','const float','const uchar','uchar','float')
            if node.owner_index in link_buffers:
                bindings+=(link_buffers[node.owner_index],);types+=('const float',)
            role = 'event-monitor'
        elif node.owner_kind == "population":
            kernel = _population_kernel(model, logical, nodes, ordinal)
            bindings = tuple(range(node.owner_index*12, (node.owner_index+1)*12))
            if node.owner_index in link_buffers:bindings+=(link_buffers[node.owner_index],)
            types = population_types(model, node.owner_index, single_tick=True)
            role = "population"
            if serial_links:
                types=(*types[:-1],'float')
                kernel=gpu_links.self_link_kernel(model,logical,node,ordinal,kernel,types)
                bindings+=(link_buffers[node.owner_index]+1,);types+=('const ulong',)
                role='canonical-linked-population'
        elif node.owner_index in canonical:
            syn = d["synapses"][node.owner_index]
            base = syn_buffers[node.owner_index]
            kernel, types = canonical_kernel(model,logical,node,ordinal)
            bindings = (syn["source_population"]*12,syn["target_population"]*12,base,base+4,
                        target_buffers[node.owner_index],base+5,syn["target_population"]*12+9)
            path = pathway_for_node(model,node)
            if path is not None:
                delay_base = delay_buffers[node.owner_index,path["name"]]
                bindings += (*range(delay_base,delay_base+7),endpoint(syn,path)[0]*12+6)
            if kernel.entry.endswith('_owned_summed'):
                side=syn['code_objects'][node.item_index]['summed_target']
                bindings += (pre_summed_buffers[node.owner_index],pre_summed_buffers[node.owner_index]+1) if side=='pre' else (base+1,base+3)
                role = 'endpoint-owned-summed'
            elif kernel.entry.endswith(('_edge_pathway','_target_pathway')):
                ownership='target' if kernel.entry.endswith('_target_pathway') else 'edge'
                queued=synapse_sparse and ownership=='target'
                if queued:
                    from .gpu_target_sparse import buffer_names as sparse_buffer_names,ROLE,BITSET_ROLE
                    key=(node.owner_index,path['name'])
                    if key not in sparse_queues:
                        r=next(i for i,p in enumerate(model['instance']['synapses'][node.owner_index]['pathways']) if p['name']==path['name'])
                        sparse_queues[key]=len(buffers);buffers.extend(sparse_buffer_names(node.owner_index,r,bitset=synapse_sparse=='bitset'))
                    queue=sparse_queues[key]
                    history=sparse_history_kernel(model,logical,node,ordinal,bitset=synapse_sparse=='bitset',compact_bitset=synapse_sparse=='bitset')
                    append(history,(syn['source_population']*12+6,delay_base+2,*range(queue,queue+5),base+1,queue+5,queue+6),
                        ('const uchar','uchar','const uint','const uint','const uint','const uint','const uint','const uint','uint','const uint' if synapse_sparse=='bitset' else 'atomic_uint'),
                        'delay-source-group-enqueue',node.clock)
                else:
                    history=pathway_history_kernel(model,logical,node,ordinal,ownership=ownership)
                    append(history,(endpoint(syn,path)[0]*12+6,delay_base+2),('const uchar','uchar'),ownership+'-pathway-history',node.clock)
                if kernel.entry.endswith('_target_pathway'):
                    bindings+=(base+1,);role='target-owned-synapse-pathway'
                    from .gpu_synapse_prefix import prefix_length
                    prefix=prefix_length(model,syn,syn['code_objects'][node.item_index],path) if synapse_prefix else 0
                    if prefix:
                        from .gpu_prefix_pending import pending_spec
                        spec=pending_spec(model,syn,syn['code_objects'][node.item_index],path)
                        prefix_bindings=bindings
                        if spec['span']:
                            key=(node.owner_index,path['name'])
                            if key not in prefix_masks:
                                r=next(i for i,p in enumerate(model['instance']['synapses'][node.owner_index]['pathways']) if p['name']==path['name'])
                                prefix_masks[key]=len(buffers)
                                buffers.append(f'synapse/{node.owner_index}/pathway/{r}/prefix_pending_mask')
                            prefix_bindings+=(prefix_masks[key],)
                        precompute,precompute_types=canonical_kernel(model,logical,node,ordinal,prefix_count=prefix,phase='prefix',pending_span=spec['span'])
                        append(precompute,prefix_bindings,precompute_types,'edge-synapse-prefix',node.clock)
                        kernel,types=canonical_kernel(model,logical,node,ordinal,prefix_count=prefix,phase='remainder')
                    if queued:
                        kernel,types=canonical_kernel(model,logical,node,ordinal,prefix_count=prefix,phase='remainder' if prefix else None,sparse_target=True,bitset=synapse_sparse=='bitset',compact_bitset=synapse_sparse=='bitset')
                        bindings+=(queue+5,queue+6);role=BITSET_ROLE if synapse_sparse=='bitset' else ROLE
                else:role='edge-owned-synapse-pathway'
            else:
                role = "parallel-synapse-state" if kernel.entry.endswith("_parallel_synapse") else "canonical-synapse"
        else:
            kernel = _synapse_kernel(model, logical, node, ordinal, event_delivery=event_delivery)
            syn = d["synapses"][node.owner_index]
            base = syn_buffers[node.owner_index]
            summed = syn["code_objects"][node.item_index]["kind"] == "summed_variable"
            bindings = (syn["source_population"]*12, syn["target_population"]*12,
                        syn["source_population"]*12+6, base, base+1, base+(3 if summed else 2), base+4, base+5)
            types = ("const float", "float", "const uchar", "const float", "const uint", "const uint", "const uint", "ulong")
            role = "summed" if summed else "target-delivery"
            path = pathway_for_node(model, node)
            if needs_delay(path):
                delay_base = delay_buffers[node.owner_index, path["name"]]
                if event_delivery == "sparse":
                    enqueue = sparse_history_kernel(model, logical, node, ordinal)
                    enqueue_bindings = (syn["source_population"]*12+6,delay_base+2,
                                        *range(delay_base+7,delay_base+12),base+1,base+9,base+10)
                    enqueue_types = ("const uchar","uchar",*("const uint",)*6,"uint","atomic_uint")
                    if (dispatches and dispatches[-1].role == "population"
                            and dispatches[-1].clock == node.clock
                            and kernels[-1].population == syn["source_population"]
                            and syn["source_start"] == 0 and kernels[-1].neurons == syn["source_count"]
                            and len(dispatches[-1].bindings)+len(enqueue_bindings) <= 30):
                        population, dispatch = kernels.pop(), dispatches.pop()
                        fused = fuse_source_enqueue(population,enqueue,dispatch.types,enqueue_types)
                        append(fused,dispatch.bindings+enqueue_bindings,dispatch.types+enqueue_types,
                               "population-delay-source-enqueue",node.clock)
                    else:
                        append(enqueue,enqueue_bindings,enqueue_types,"delay-source-group-enqueue",node.clock)
                else:
                    enqueue = history_kernel(model, logical, node, ordinal)
                    append(enqueue,(syn["source_population"]*12+6,delay_base+2),
                           ("const uchar","uchar"),"delay-history-enqueue",node.clock)
                bindings = bindings[:5]+(delay_base+1,)+bindings[6:]+(delay_base,delay_base+2,delay_base+3,delay_base+4,delay_base+5,delay_base+6)
                types += ("const uint","const uchar","const uint","const uint","const long","uint")
                role = "delayed-target-scan"
                if event_delivery == "sparse":
                    bindings += (base+9,base+10)
                    types += ("uint","atomic_uint")
                    role = "delayed-target-sparse"
            elif not summed and event_delivery == "sparse":
                enqueue = enqueue_kernel(model,logical,node,ordinal)
                append(enqueue,(syn["source_population"]*12+6,base+6,base+7,base+8,base+1,base+9,base+10),
                       ("const uchar","const uint","const uint","const uint","const uint","uint","atomic_uint"),"source-enqueue",node.clock)
                bindings += (base+9,base+10)
                types += ("uint","atomic_uint")
        # Conservative chain orders expansion, consumption and queue reuse.
        append(kernel,bindings,types,role,node.clock)
    kernels,dispatches=fuse_dispatches(model,logical,kernels,dispatches)
    if synapse_fusion:
        from .gpu_synapse_fusion import fuse_pairs
        kernels,dispatches=fuse_pairs(model,logical,kernels,dispatches)
    hashes = model["protocol"]["layers"]
    return MetalPlan("b2-metal-plan-v0", METAL_PROFILE, hashes["definition"], hashes["instance"],
                     hashes["run"], logical, tuple(kernels), tuple(dispatches), tuple(buffers), "canonical-target-owned-dag", event_delivery, tuple(elided),
                     rng_profile=RNG_PROFILE if has_random(d) else None,
                     initializations=initialization_records(model))


def _cpu_dag(executor):
    """Same generated f32 operations, persistent CPU workers and stage barriers."""
    from .gpu_functions import require_cpu_mirror
    require_cpu_mirror(executor.model)
    if hasattr(executor, "cpu_dag"):
        return executor.cpu_dag
    from .metal import _CPU_PRELUDE
    parts = [_CPU_PRELUDE, CPU_ATOMICS, '''#include <mutex>
#include <condition_variable>
class Barrier {
    std::mutex mutex; std::condition_variable condition;
    uint arrived=0, generation=0, workers;
public:
    explicit Barrier(uint n): workers(n) {}
    void wait() {
        if (workers==1) return;
        std::unique_lock<std::mutex> lock(mutex);
        uint old=generation;
        if (++arrived==workers) { arrived=0; ++generation; condition.notify_all(); }
        else condition.wait(lock,[&]{ return generation!=old; });
    }
};''']
    calls = []
    for kernel, dispatch in zip(executor.plan.kernels, executor.plan.dispatches, strict=True):
        source = kernel.source.replace("#include <metal_stdlib>", "").replace("using namespace metal;", "")
        source = source.replace("kernel void", "void").replace("device ", "").replace("thread ", "").replace("constant ", "const ")
        source = re.sub(r"\[\[(?:buffer\(\d+\)|thread_position_in_grid)\]\]", "", source)
        parts += [f"namespace {kernel.entry} {{", source, "}"]
        args = ", ".join(f"static_cast<{dtype} *>(data[{binding}])" for dtype, binding in zip(dispatch.types, dispatch.bindings, strict=True))
        calls.append(f"if (active[{dispatch.clock}]) {{ long tick=ticks[{dispatch.clock}]; for (uint i=uint(uint64_t({dispatch.lanes})*rank/workers); i<uint(uint64_t({dispatch.lanes})*(rank+1)/workers); ++i) {kernel.entry}::{kernel.entry}({args},tick,i); barrier.wait(); }}")
    from .gpu_schedule import clock_arrays, native_scheduler_source
    starts, ends, dts = clock_arrays(executor.plan.logical.clocks)
    parts.append(native_scheduler_source())
    count = len(starts)
    parts.append(f'''extern "C" void cpu_dag(void **data, uint workers) {{
    Barrier barrier(workers);
    auto run = [&](uint rank) {{
        int64_t ticks[]={{{','.join(map(str,starts))}}}, ends[]={{{','.join(map(str,ends))}}};
        const double dt[]={{{','.join(map(repr,dts))}}};
        uint8_t active[{count}];
        while (b2_active_clocks(ticks,ends,dt,{count},active)) {{
            {' '.join(calls)}
            for (uint c=0;c<{count};++c) ticks[c]+=active[c];
        }}
    }};
    std::vector<std::thread> threads;
    for (uint rank=1; rank<workers; ++rank) threads.emplace_back(run,rank);
    run(0);
    for (auto &thread: threads) thread.join();
}}''')
    source = "\n".join(parts)
    digest = hashlib.sha256(source.encode()).hexdigest()[:16]
    path = executor.directory / f"cpu-f32-dag-{digest}.cpp"
    library = path.with_suffix(".dylib")
    path.write_text(source)
    if not library.exists():
        subprocess.run(["clang++", "-std=c++17", "-O3", "-ffp-contract=off", "-fno-fast-math",
                        *(["-dynamiclib"] if platform.system()=="Darwin" else ["-shared","-fPIC"]),
                        "-pthread", str(path), "-o", str(library)],
                       check=True, capture_output=True, text=True)
    executor.cpu_dag = ctypes.CDLL(str(library))
    executor.cpu_dag.cpu_dag.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint32]
    executor.cpu_dag.cpu_dag.restype = None
    return executor.cpu_dag


def _prepare_dag_storage(executor, max_buffer_bytes):
    from .metal import population_arrays
    model, plan = executor.model, executor.plan
    d = model["definition"]
    nodes={node.id:node for node in plan.logical.nodes}
    edge_paths=set();target_paths=set()
    for kernel,dispatch in zip(plan.kernels,plan.dispatches,strict=True):
        if dispatch.role=='fused-target-pre-post':
            pre,post=(nodes[n] for n in kernel.nodes)
            target_paths.add((pre.owner_index,pathway_for_node(model,pre)['name']))
            edge_paths.add((post.owner_index,pathway_for_node(model,post)['name']))
        if dispatch.role in {'edge-owned-synapse-pathway','target-owned-synapse-pathway','target-owned-sparse-synapse-pathway','target-owned-bitset-synapse-pathway'}:
            node=nodes[kernel.nodes[0]]
            (edge_paths if dispatch.role=='edge-owned-synapse-pathway' else target_paths).add((node.owner_index,pathway_for_node(model,node)['name']))
    arrays, population_data, synapse_data = [], [], []
    for p in range(len(d["populations"])):
        # Storage/recording metadata uses the population clock. Every code
        # object, including other clocks, is executed by its own DAG dispatch.
        storage_logical = replace(plan.logical, nodes=tuple(node for node in plan.logical.nodes
            if node.clock == d['populations'][p]['clock']))
        template = _kernel(model, storage_logical, p)
        storage, last = population_arrays(model, p, template, max_buffer_bytes)
        count_events=len(event_lanes(d['populations'][p]))
        history_bytes = count_events*template.neurons*template.monitor_steps
        if history_bytes > max_buffer_bytes:
            raise MemoryError("Metal DAG event history exceeds configured memory limit")
        history = np.zeros((count_events, template.monitor_steps, template.neurons), np.uint8)
        population_data.append((template, storage, last, history))
        arrays.extend(storage + [history])
    for q, (syn, inst) in enumerate(zip(d["synapses"], model["instance"]["synapses"], strict=True)):
        count = len(inst["source"])
        values = []
        states = {}
        for symbol in syn["states"]+syn["parameters"]:
            field = "initial_state" if symbol in syn["states"] else "parameters"
            data = gt.pack(inst[field][symbol["name"]],symbol["dtype"])
            if field == "initial_state":
                states[symbol["name"]] = (symbol["dtype"],len(values),len(data)//gt.width(symbol["dtype"]))
            values.extend(data)
        tables = TimedTables(syn["code_objects"], {f["name"]:f for f in d["functions"]}, buffer="values", offset=len(values))
        values.extend(tables.values)
        source = np.asarray(inst["source"], np.uint32)
        target = np.asarray(inst["target"], np.uint32)
        edges = np.arange(count, dtype=np.uint32)
        event_order = np.lexsort((edges, source, target)).astype(np.uint32)
        summed_order = np.argsort(target, kind="stable").astype(np.uint32)
        offsets = np.concatenate(([0], np.cumsum(np.bincount(target.astype(np.int64), minlength=syn["target_count"]))))
        delivery_slots=max(syn['target_count'],count,1) if any(owner==q for owner,_ in edge_paths) else syn['target_count']
        if delivery_slots*8 > max_buffer_bytes:raise MemoryError("GPU per-edge delivery counters exceed configured memory limit")
        delivered = np.zeros(delivery_slots, np.uint64)
        values_binding = len(arrays)
        arrays.extend([np.asarray(values, np.float32), offsets.astype(np.uint32), event_order, summed_order, source, delivered])
        # Consume the validated inventory, also permitting the private legacy
        # plan in paired diagnostics. Absent queues are never constructed.
        if f"synapse/{q}/source_offsets" in plan.buffers:
            source_offsets = np.concatenate(([0],np.cumsum(np.bincount(source.astype(np.int64),minlength=syn["source_count"])))).astype(np.uint32)
            ranks = np.empty(count,np.uint32)
            ranks[event_order] = edges
            source_ranks = ranks[np.argsort(source,kind="stable")]
            arrays.extend([source_offsets,source_ranks,target[event_order],np.zeros(count,np.uint32),np.zeros(syn["target_count"],np.uint32)])
        canonical = canonical_projection(model,q)
        if canonical:
            arrays.append(target)
            if any(c['kind']=='summed_variable' and c['summed_target']=='pre' for c in syn['code_objects']):
                if (syn['source_count']+1)*4 > max_buffer_bytes:raise MemoryError('GPU source summed offsets exceed configured memory limit')
                pre_offsets=np.concatenate(([0],np.cumsum(np.bincount(source.astype(np.int64),minlength=syn['source_count'])))).astype(np.uint32)
                arrays.extend([pre_offsets,np.argsort(source,kind='stable').astype(np.uint32)])
        for path in inst["pathways"]:
            clock_index = next(code['clock'] for code in syn['code_objects']
                               if code.get('pathway_name') == path['name'] and code['kind'] in {'synapses','synapses_post'})
            clock = plan.logical.clocks[clock_index]
            if canonical:
                arrays.extend(canonical_delay_arrays(syn,inst,path,clock,max_buffer_bytes,edge_owned=(q,path["name"]) in edge_paths,target_owned=(q,path["name"]) in target_paths))
            elif needs_delay(path):
                storage = delay_arrays(syn,inst,path,clock,max_buffer_bytes)
                arrays.extend(storage)
                if plan.event_delivery == "sparse":
                    arrays.extend(sparse_delay_arrays(syn,inst,storage[0],storage[1],max_buffer_bytes))
        synapse_data.append((states, delivered, values_binding))
    for p,m,pop,monitor in monitors(model):
        arrays.extend(monitor_arrays(pop,monitor,max_buffer_bytes))
    for p,pop in enumerate(d['populations']):
        if pop.get('linked_variables'):arrays.extend(gpu_links.arrays(model,p,max_buffer_bytes))
    from .gpu_prefix_pending import pending_spec,pending_mask
    while len(arrays)<len(plan.buffers):
        name=plan.buffers[len(arrays)]
        parts=name.split('/')
        if len(parts)==5 and parts[0]=='synapse' and parts[2]=='pathway' and parts[4]=='target_sparse_source_group_offsets':
            from . import gpu_target_sparse
            q,r=int(parts[1]),int(parts[3])
            bitset=plan.buffers[len(arrays)+5].endswith('/target_sparse_bitmap_words')
            names=gpu_target_sparse.buffer_names(q,r,bitset=bitset)
            if tuple(plan.buffers[len(arrays):len(arrays)+len(names)])!=names:
                raise PlanValidationError('Invalid GPU target sparse queue storage bindings')
            arrays.extend(gpu_target_sparse.arrays(model,plan,arrays,q,r,max_buffer_bytes,bitset=bitset))
            continue
        if len(parts)!=5 or parts[0]!='synapse' or parts[2]!='pathway' or parts[4]!='prefix_pending_mask':
            raise PlanValidationError('Unknown GPU prefix storage binding')
        q,r=int(parts[1]),int(parts[3]);syn=d['synapses'][q];path=model['instance']['synapses'][q]['pathways'][r]
        code=next(c for c in syn['code_objects'] if c.get('pathway_name')==path['name'] and c['kind']=='synapses')
        spec=pending_spec(model,syn,code,path)
        if spec is None or not spec['span']:raise PlanValidationError('GPU pending-prefix proof no longer holds')
        arrays.append(pending_mask(spec,max_buffer_bytes))
    if any(a.nbytes > max_buffer_bytes for a in arrays):
        raise MemoryError("Metal DAG buffer exceeds configured memory limit")
    # Metal disallows zero-length allocations, including an empty edge set.
    arrays = [a if a.size else np.zeros(1, a.dtype) for a in arrays]
    bindings = {id(array): i for i,array in enumerate(arrays)}
    return arrays, population_data, [(states,bindings[id(delivered)],values) for states,delivered,values in synapse_data]


def run_dag(executor, *, max_buffer_bytes, compute, workers):
    """Replay an executor's validated snapshot with fresh writable storage.

    Sorting topology and decoding initial values belong to preparation. Cached
    arrays are never passed to writable bindings; returned state dictionaries
    and refractory arrays cannot mutate the initial snapshot either.
    """
    from .metal import population_result
    model, plan = executor.model, executor.plan
    d = model["definition"]
    cpu = _cpu_dag(executor) if compute == "cpu-f32" else None
    started = time.perf_counter()
    preparation_seconds = 0.0
    if not hasattr(executor, "_dag_initial_storage"):
        executor._dag_initial_storage = _prepare_dag_storage(executor,max_buffer_bytes)
        preparation_seconds = time.perf_counter()-started
    initial, pop_initial, syn_initial = executor._dag_initial_storage
    if any(a.nbytes > max_buffer_bytes for a in initial):
        raise MemoryError("Metal DAG buffer exceeds configured memory limit")
    writable = {binding for dispatch in plan.dispatches
                for binding,dtype in zip(dispatch.bindings,dispatch.types,strict=True)
                if not dtype.startswith("const ")}
    from .gpu_readback import fresh_dag_arrays,host_spike_cache
    host_reset_started=time.perf_counter()
    arrays,host_storage=fresh_dag_arrays(plan,initial,writable,compute,spike_cache=host_spike_cache(executor,compute))
    host_storage['reset_seconds']=time.perf_counter()-host_reset_started
    population_data = [(template,arrays[p*12:p*12+11],None if last is None else last.copy(),arrays[p*12+11])
                       for p,(template,storage,last,history) in enumerate(pop_initial)]
    pointers = (ctypes.c_void_p*len(arrays))(*(a.ctypes.data for a in arrays))
    timing = (ctypes.c_double*4)()
    if cpu is not None:
        begin = time.perf_counter()
        cpu.cpu_dag(pointers, workers)
        timing[1] = time.perf_counter()-begin
    elif compute == "cuda":
        timing = executor._execute_dag(arrays,max_buffer_bytes=max_buffer_bytes)
    else:
        timing = executor._execute_dag(arrays, max_buffer_bytes=max_buffer_bytes)
    populations = [population_result(model, p, template, storage, last)
                   for p, (template, storage, last, history) in enumerate(population_data)]
    for p, (values, (template, storage, last, history)) in enumerate(zip(populations, population_data, strict=True)):
        values['event_streams'] = {}
        for event in d['populations'][p]['events']:
            slot=event_lanes(d['populations'][p]).index(event)
            ticks,indices=event_coordinates(history[slot])
            values['event_streams'][event] = dict(ticks=ticks+template.start_tick+template.steps-template.monitor_steps,indices=indices)
    synapse_data=[({name:gt.unpack(arrays[values],field) for name,field in states.items()},arrays[binding])
                  for states,binding,values in syn_initial]
    monitor_results(model,plan,arrays,populations)
    if not gt.finite(value for states,delivered in synapse_data for value in states.values()):
        raise FloatingPointError("Metal float32 produced non-finite synaptic state")
    if any(np.any(delivered >> np.uint64(63)) for states,delivered in synapse_data):
        raise FloatingPointError("GPU numeric evaluation failed in floating arithmetic, synaptic division, timestep/tick_offset, random sampler or TimedArray")
    return {"populations": populations,
            "synapses": [{"states": {name:value.copy() for name,value in states.items()}, "events": int(delivered.sum())} for states, delivered in synapse_data],
            "numeric_profile": plan.numeric_profile if cpu is None else "b2-cpu-f32-mirror-v0",
            "device": executor.device_name if cpu is None else f"CPU f32 DAG mirror ({workers} workers)",
            "timings": [dict(zip(("input_seconds", "command_seconds", "gpu_seconds", "readback_seconds"), timing, strict=True))],
            "run_seconds": time.perf_counter()-started, "compile_seconds": executor.compile_seconds,
            "storage_preparation_seconds": preparation_seconds,
            "host_storage": host_storage,
            "initializations": [asdict(record) for record in plan.initializations],
            "initialization_seconds": initialization_seconds(model),
            "plan_sha256": plan.sha256, "rng_profile": plan.rng_profile}
