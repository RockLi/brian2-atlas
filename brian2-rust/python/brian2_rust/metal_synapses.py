"""Canonical GPU synapse execution for mutable and order-dependent programs.

Order-dependent pathways and reductions retain a single canonical GPU lane.
Independent summed nodes assign one lane per endpoint with stable edge order.
Clock-driven, subexpression and pre/post nodes with edge-owned writes use one
lane per edge; population inputs remain read-only behind stage barriers. Delayed
pathways record endpoint history first and retain pending chronology per edge.
Pre pathways with target-neuron stores serialize incoming edges in target lanes
when source reads cannot observe another lane's writes.
"""
import numpy as np

from .metal import MetalKernel, _PRELUDE, _Expressions, _literal, number
from .metal_delays import pathway_for_node, ring_slots, delay_arrays
from .metal_random import has_random
from . import gpu_types as gt
from .plan import PlanValidationError
from .metal_timed_array import TimedTables, timed_nodes


def canonical_projection(model, q):
    syn = model["definition"]["synapses"][q]
    populations=model['definition']['populations']
    owners=[syn,populations[syn['source_population']],populations[syn['target_population']]]
    if any(s['dtype'] not in {'f32','f64'} for owner in owners for s in owner['states']+owner['parameters']):
        return True
    if any(path["kind"] == "post" for path in model["instance"]["synapses"][q]["pathways"]):
        return True
    for code in syn["code_objects"]:
        # Scalar expressions execute even without arriving events. The fast
        # target-delivery kernel evaluates only active edges.
        if code["scalar"]:return True
        if checked_numeric_operation(code, {f['name']:f for f in model['definition']['functions']}):
            return True
        if any(timed_nodes(code, {f["name"]:f for f in model["definition"]["functions"]})):
            return True
        if has_random(code):
            return True
        if "not_refractory_post" in code["effects"]["reads"]:
            return True
        if code["kind"] not in {"synapses", "summed_variable"}:
            return True
        if code["kind"] == "summed_variable" and code["summed_target"] != "post":
            return True
        if (code["kind"] == "summed_variable"
                and code["clock"] != populations[syn["source_population"]]["clock"]
                and {"t", "dt"} & set(code["effects"]["reads"])):
            return True
        writes = code["effects"]["writes"]
        if any(name not in syn["post_state_aliases"] for name in writes):
            return True
        written = {syn["post_state_aliases"][name] for name in writes}
        if code["kind"] == "summed_variable":
            written.add(code["summed_state"])
            if any(syn["post_state_aliases"].get(name) in written for name in code["effects"]["reads"]):
                return True
        if syn["source_population"] == syn["target_population"] and any(
                syn["pre_state_aliases"].get(name) in written for name in code["effects"]["reads"]):
            return True
    return False


def checked_numeric_operation(value, functions):
    if isinstance(value,dict):
        if value.get('op') in {'mod','floor_div','timestep','tick_offset'}:return True
        if value.get('op')=='call' and checked_numeric_operation(functions[value['function']]['body'],functions):return True
        return any(checked_numeric_operation(v,functions) for v in value.values())
    return isinstance(value,(list,tuple)) and any(checked_numeric_operation(v,functions) for v in value)


def endpoint(syn, path):
    side = "source" if path["kind"] == "pre" else "target"
    return syn[f"{side}_population"], syn[f"{side}_start"], syn[f"{side}_count"], side


def canonical_delay_arrays(syn, inst, path, clock, budget, *, edge_owned=False, target_owned=False):
    _, _, count, side = endpoint(syn, path)
    # A stable bucket per edge preserves every duplicate pending event and its
    # tick/order within that edge. Canonical paths retain one global bucket.
    edges = len(inst["source"])
    if edge_owned and (edges+1)*4 > budget:
        raise MemoryError("GPU per-edge pending offsets exceed configured memory limit")
    if edge_owned and target_owned:raise ValueError("Pathway ownership must be exclusive")
    virtual = {"source_count": count, "target_count": max(1,edges) if edge_owned else syn["target_count"] if target_owned else 1}
    topology = {"source": inst[side], "target": np.arange(edges,dtype=np.uint32) if edge_owned else np.zeros(edges,np.uint32)}
    if target_owned:topology["target"]=np.asarray(inst["target"],np.uint32)
    return delay_arrays(virtual,topology,path,clock,budget)


BASE_TYPES = ("float","float","float","const uint","const uint","ulong","const uchar")
PATH_TYPES = ("const uint","const uint","uchar","const uint","const uint","const long","uint","const uchar")


def independent_state_update(syn, code):
    """Validated all-edge nodes can reorder only edge-owned state stores.

    Scalar locals are pure and can be recomputed in each lane. Scalar RNG is
    excluded to preserve once-per-node draw semantics. Population/shared writes,
    event delivery and reductions retain canonical ordering.
    """
    import sys
    if sys.byteorder != 'little':return False
    if code['kind'] not in {'synapse_state_update','synapse_subexpression_update'}:return False
    if code['iteration_domain']!='all_synapses' or has_random(code['scalar']):return False
    states={state['name']:state for state in syn['states']}
    return all(name in states and states[name]['index_domain']=='synapse' for name in code['effects']['writes'])


def independent_pathway(syn, code):
    """Only own-edge stores may reorder across edges in a pre/post pathway.

    Population/shared stores remain canonical. Reads of population state are
    stable at this node's barrier; pending and current events stay sequential
    within each edge. Validated portable functions are pure.
    """
    if code['kind'] not in {'synapses','synapses_post'} or code['iteration_domain']!='active_synapses':return False
    if has_random(code['scalar']):return False
    states={state['name']:state for state in syn['states']}
    return all(name in states and states[name]['index_domain']=='synapse' for name in code['effects']['writes'])


def target_owned_pathway(model, syn, code):
    """Target lanes may serialize all their incoming edges without atomics.

    A pathway may write own-edge state and target-neuron state. Recurrent reads
    of a source alias that another target lane writes are not independent. All
    scalar/shared or source-population stores retain the canonical route.
    """
    if code['kind']!='synapses' or code['iteration_domain']!='active_synapses':return False
    if has_random(code['scalar']):return False
    states={state['name']:state for state in syn['states']}
    population=model['definition']['populations'][syn['target_population']]
    target_states={state['name']:state for state in population['states']}
    written=set()
    for name in code['effects']['writes']:
        if name in states and states[name]['index_domain']=='synapse':continue
        target=syn['post_state_aliases'].get(name)
        if target not in target_states or target_states[target]['index_domain']!='neuron':return False
        written.add(target)
    if not written:return False  # The existing per-edge route is better here.
    if syn['source_population']==syn['target_population']:
        if any(syn['pre_state_aliases'].get(name) in written for name in code['effects']['reads']):return False
    return True


def independent_summed(model, syn, code):
    """One endpoint lane sums its edges in original edge order.

    The destination must not be read by the reduction: canonical execution zeros
    all destinations first and then accumulates in global edge order. Reordering
    that zeroing or another endpoint's partial sums is observable through aliases.
    Pure scalar work can repeat, while RNG retains its once-per-node semantics.
    """
    import sys
    if sys.byteorder != 'little':return False
    if code['kind']!='summed_variable' or code['iteration_domain']!='all_synapses':return False
    if has_random(code['scalar']):return False
    side=code['summed_target'];endpoint='source' if side=='pre' else 'target'
    population=syn[endpoint+'_population'];name=code['summed_state']
    state=next(s for s in model['definition']['populations'][population]['states'] if s['name']==name)
    if state['dtype'] not in {'f32','f64'} or state['index_domain']!='neuron':return False
    for alias_side,owner in (('pre','source'),('post','target')):
        if syn[owner+'_population']==population:
            if any(syn[alias_side+'_state_aliases'].get(read)==name for read in code['effects']['reads']):return False
    return True


def pathway_history_kernel(model, logical, node, ordinal, *, ownership="edge"):
    from .metal_event_layout import event_offset
    syn=model['definition']['synapses'][node.owner_index];path=pathway_for_node(model,node)
    population,start,length,_=endpoint(syn,path);clock=logical.clocks[node.clock]
    offset=event_offset(model['definition']['populations'][population],path['event']);slots=ring_slots(path,clock)
    entry=f'stage_{ordinal}_{ownership}_pathway_history'
    source=_PRELUDE+f"""
kernel void {entry}(device const uchar *fired [[buffer(0)]], device uchar *history [[buffer(1)]],
    constant long &tick [[buffer(2)]], uint lane [[thread_position_in_grid]]) {{
    if (lane >= {length}u) return;
    history[ulong(tick%{slots})*{length}+lane]=fired[lane+{start+offset}];
}}
"""
    return MetalKernel(population,entry,(),length,clock.start_tick,clock.steps,0,0,source)


def pending_code(path, source):
    """Specialize the immutable activation-entry queue, not the live history ring.

    New delayed events use history and never append to this pending input. A
    subsequent activation is replanned against its own pending list. Keep the
    buffer ABI stable, including cursors used by continuation reconstruction.
    """
    return source if path['pending'] else ''


def canonical_kernel(model, logical, node, ordinal, *, prefix_count=0, phase=None, pending_span=0, sparse_target=False, bitset=False, compact_bitset=False):
    from .metal_event_layout import event_offset
    d = model["definition"]
    syn, inst = d["synapses"][node.owner_index], model["instance"]["synapses"][node.owner_index]
    code = syn["code_objects"][node.item_index]
    clock = logical.clocks[node.clock]
    count = len(inst["source"])
    parallel = independent_state_update(syn,code)
    edge_path = independent_pathway(syn,code)
    target_path = target_owned_pathway(model,syn,code)
    summed = independent_summed(model,syn,code)
    summed_endpoint = 'source' if code.get('summed_target')=='pre' else 'target'
    if sparse_target and (not target_path or phase=='prefix'):raise ValueError('Sparse target queues require the target consumer')
    if compact_bitset and not bitset:raise ValueError('Compact bitmaps require bitset consumer')
    if bitset and not sparse_target:raise ValueError('Bitset consumer requires sparse target ownership')
    if pending_span and phase!='prefix':raise ValueError('Pending prefix mask requires the prefix phase')
    if phase is not None:
        if phase not in {'prefix','remainder'} or not prefix_count or not target_path:
            raise ValueError('Invalid target pathway split')
        vector=code['vector'][:prefix_count] if phase=='prefix' else code['vector'][prefix_count:]
        code={**code,'vector':vector,'effects':{**code['effects'],'writes':sorted({s['target'] for s in vector} & set(code['effects']['writes']))}}
    lanes = count if phase=='prefix' else syn[summed_endpoint+'_count'] if summed else syn["target_count"] if target_path else max(1,count) if parallel or edge_path else 1
    # Synaptic CodeRunners activate on their own clock, but Brian exposes
    # their source Synapses clock through the dt variable.
    owner_clock_index = (d["populations"][syn["source_population"]]["clock"]
                         if code["kind"] in {"synapse_run_regularly", "summed_variable"}
                         else node.clock)
    dt_clock = logical.clocks[owner_clock_index]
    from .gpu_schedule import owner_tick_expression
    owner_tick = (owner_tick_expression(logical, node.clock, owner_clock_index)
                  if "t" in code["effects"]["reads"] else "tick")
    symbols = {"i":"pre_index","j":"post_index","N":f"{count}u",
               "N_pre":f"{syn['source_count']}u","N_post":f"{syn['target_count']}u",
               "t":"time","dt":_literal(number(dt_clock.dt)),
               "not_refractory_post":"bool(available[target_index])"}
    dtypes=dict(i='index',j='index',N='index',N_pre='index',N_post='index',t='f64',dt='f64',not_refractory_post='bool')
    stores={}
    for side, pop_index, buffer, index in (("pre",syn["source_population"],"pre_state","source_index"),
                                           ("post",syn["target_population"],"post_state","target_index")):
        pop = d["populations"][pop_index]
        fields,_=gt.layout(pop['states'],pop['count'])
        for alias,name in syn[f"{side}_state_aliases"].items():
            field=fields[name]
            symbols[alias]=gt.read(buffer,field,index)
            dtypes[alias]=field[0]
            stores[alias]=(buffer,field,index)
    fields,offset=gt.layout(syn['states']+syn['parameters'],count)
    for state in syn['states']+syn['parameters']:
        index='0' if state['index_domain']=='scalar' else 'edge'
        name=state['name'];field=fields[name]
        symbols[name]=gt.read('values',field,index)
        dtypes[name]=field[0]
        stores[name]=('values',field,index)
    functions = {f["name"]:f for f in d["functions"]}
    tables = TimedTables(syn["code_objects"], functions, buffer="values", offset=offset)
    block = _Expressions(symbols,functions, math_error="math_error",
                         rng={"seed":model["instance"]["rng_seed"], "index":"ulong(edge)"},
                         timed_tables=tables, dtypes=dtypes)
    block.statements(code["scalar"])
    # Scalar expressions execute once per scheduled node, including nodes
    # with no arriving events. Keep their checked faults outside edge loops.
    scalar_body = "\n".join(block.lines)
    block.lines = []
    if code["kind"] == "synapse_state_update":
        block.snapshot = dict(symbols)
    block.statements(code["vector"])
    body = [f"uint pre_index=sources[edge], post_index=targets[edge];",
            f"uint source_index=pre_index+{syn['source_start']}, target_index=post_index+{syn['target_start']};",
            *block.lines]
    before = ""
    if code["kind"] == "summed_variable":
        side = code["summed_target"]
        pop = d["populations"][syn["source_population" if side=="pre" else "target_population"]]
        field=gt.layout(pop["states"],pop["count"])[0][code["summed_state"]]
        if field[0] not in {'f32','f64'}:
            raise PlanValidationError('GPU summed variables require floating-point destination state')
        start = syn["source_start" if side=="pre" else "target_start"]
        length = syn["source_count" if side=="pre" else "target_count"]
        buffer, index = ("pre_state","source_index") if side=="pre" else ("post_state","target_index")
        before = f"for (uint j=0;j<{length};++j) {{ "+gt.write(buffer,field,f'{start}+j',f'{gt.CTYPES[field[0]]}(0)')+" }"
        if summed:before = gt.write(buffer,field,f'{start}+lane',f'{gt.CTYPES[field[0]]}(0)')
        value=f"b2_finite(({gt.read(buffer,field,index)}) + ({block.symbols['_synaptic_var']}), &math_error)"
        body.append(gt.write(buffer,field,index,value))
    else:
        for name in code["effects"]["writes"]:
            body.append(gt.write(*stores[name],block.symbols[name]))
    path = pathway_for_node(model,node)
    if path is not None and phase!='prefix':
        body.append("delivered[edge] += 1;" if edge_path else "delivered[lane] += 1;" if target_path else "delivered[0] += 1;")
    body = "\n".join(body)
    loop = f"for (uint edge=0;edge<{count};++edge) {{ {body} }}"
    types, names = BASE_TYPES, ("pre_state","post_state","values","sources","targets","delivered","available")
    if path is not None:
        population, start, length, side = endpoint(syn,path)
        offset=event_offset(d['populations'][population],path['event'])
        slots = ring_slots(path,clock)
        types += PATH_TYPES
        names += ("delays","edges","history","pending_offsets","pending_edges","pending_ticks","cursor","fired")
        loop = f'''
    for (uint j=0;j<{length};++j) history[ulong(tick%{slots})*{length}+j]=fired[j+{start+offset}];
    {pending_code(path, f"""while (cursor[0]<pending_offsets[1] && pending_ticks[cursor[0]]<=tick) {{
        uint edge=pending_edges[cursor[0]++]; {body}
    }}""")}
    for (uint at=0;at<{count};++at) {{
        uint edge=edges[at];
        long emission=tick-long(delays[edge]);
        if (emission<{clock.start_tick} || !history[ulong(emission%{slots})*{length}+{'sources' if side=='source' else 'targets'}[edge]]) continue;
        {body}
    }}
'''
    if parallel:
        # Reinterpret the uint64 delivery counter as two uint32 words only in
        # this stage. No lane increments the count here; integer atomic OR on
        # its high word preserves all event-count bits and concurrent faults.
        types = ("const float","const float","float","const uint","const uint","atomic_uint","const uchar")
        loop = f"if (lane < {count}u) {{ uint edge=lane; {body} }}"
    if edge_path:
        types = ("const float","const float","float","const uint","const uint","ulong","const uchar",
                 "const uint","const uint","const uchar","const uint","const uint","const long","uint","const uchar")
        loop = f"""
    if (lane < {count}u) {{
        uint edge=lane;
        {pending_code(path, f'''uint begin=pending_offsets[edge], at=begin+cursor[edge];
        while (at<pending_offsets[edge+1] && pending_ticks[at]<=tick) {{
            {body}
            ++at;
        }}
        cursor[edge]=at-begin;''')}
        long emission=tick-long(delays[edge]);
        if (emission>={clock.start_tick} && history[ulong(emission%{slots})*{length}+{'sources' if side=='source' else 'targets'}[edge]]) {{
            {body}
        }}
    }}
"""
    if target_path:
        types = ("const float","float","float","const uint","const uint","ulong","const uchar",
                 "const uint","const uint","const uchar","const uint","const uint","const long","uint","const uchar","const uint")
        names += ("target_offsets",)
        loop = f"""
    {pending_code(path, f'''uint begin=pending_offsets[lane], at=begin+cursor[lane];
    while (at<pending_offsets[lane+1] && pending_ticks[at]<=tick) {{
        uint edge=pending_edges[at]; {body}
        ++at;
    }}
    cursor[lane]=at-begin;''')}
    for (uint pos=target_offsets[lane];pos<target_offsets[lane+1];++pos) {{
        uint edge=edges[pos];
        long emission=tick-long(delays[edge]);
        if (emission<{clock.start_tick} || !history[ulong(emission%{slots})*{length}+{'sources' if side=='source' else 'targets'}[edge]]) continue;
        {body}
    }}
"""
    if sparse_target:
        from .metal_events import SORT_SOURCE
        ordered_loop=loop
        types+=('uint','atomic_uint');names+=('active_ranks','active_counts')
        prepare="""uint active_count=atomic_load_explicit(&active_counts[lane],memory_order_relaxed);
    uint queue_base=target_offsets[lane], degree=target_offsets[lane+1]-queue_base;
    bool compact=ulong(active_count)*4<degree;
    if (compact) b2_sort_active(active_ranks,queue_base,active_count);"""
        old="for (uint pos=target_offsets[lane];pos<target_offsets[lane+1];++pos) {\n        uint edge=edges[pos];"
        new="for (uint at=0;at<(compact ? active_count : degree);++at) {\n        uint edge=edges[compact ? active_ranks[queue_base+at] : queue_base+at];"
        if old not in loop:raise PlanValidationError('Sparse target consumer requires canonical incoming order')
        loop=prepare+'\n'+loop.replace(old,new)
        # Queue ranks already encode the original delay/source/creation order.
        # The dense fallback retains the history predicate; pending stays first.
        loop+= "\n    atomic_store_explicit(&active_counts[lane],0u,memory_order_relaxed);"
        if bitset:
            # The source/delay producer emits each current edge at most once.
            # Ordered ranks already include delay/source/creation chronology;
            # duplicate imported pending events remain in the preceding loop.
            # Keep scalar execution and fault publication outside both loops.
            pending=ordered_loop[:ordered_loop.index('    for (uint pos=target_offsets[lane]')]
            loop=pending+f'''
    uint queue_base=target_offsets[lane], degree=target_offsets[lane+1]-queue_base;
    uint words=degree/32+uint(degree%32!=0);
    for (uint word=0;word<words;++word) {{
        uint mask=active_ranks[queue_base+word];
        active_ranks[queue_base+word]=0u;
        while (mask) {{
            uint low=mask & (0u-mask);
            uint bit=uint((low&0xffff0000u)!=0)*16u+uint((low&0xff00ff00u)!=0)*8u
                +uint((low&0xf0f0f0f0u)!=0)*4u+uint((low&0xccccccccu)!=0)*2u+uint((low&0xaaaaaaaau)!=0);
            uint edge=edges[queue_base+word*32+bit];
            {body}
            mask&=mask-1u;
        }}
    }}
'''
            if compact_bitset:
                types=(*types[:-1],'const uint')
                loop=loop.replace('active_ranks[queue_base+word]','active_ranks[active_counts[lane]+word]')
    if phase=='prefix':
        types=("const float","const float","float","const uint","const uint","atomic_uint","const uchar",
               "const uint","const uint","const uchar","const uint","const uint","const long","const uint","const uchar","const uint")
        active=f"emission>={clock.start_tick} && history[ulong(emission%{slots})*{length}+sources[edge]]"
        pending=''
        if pending_span:
            types+=('const uint',);names+=('prefix_pending_mask',)
            pending=f"ulong relative=ulong(tick-{clock.start_tick});\n    "
            active=f"(relative<{pending_span}ul && (prefix_pending_mask[(relative/32)*{count}ul+edge] & (1u<<uint(relative%32)))) || ({active})"
        loop=f"""
    uint edge=lane;
    {pending}long emission=tick-long(delays[edge]);
    if ({active}) {{
        {body}
    }}
"""
    if summed:
        types = ("float","float","const float","const uint","const uint","atomic_uint","const uchar","const uint","const uint")
        names += ("sum_offsets","sum_edges")
        loop = f"for (uint pos=sum_offsets[lane];pos<sum_offsets[lane+1];++pos) {{ uint edge=sum_edges[pos]; {body} }}"
    arguments = ",\n".join(f"device {t} *{name} [[buffer({i})]]" for i,(t,name) in enumerate(zip(types,names,strict=True)))
    entry = f"stage_{ordinal}_edge_prefix" if phase=='prefix' else f"stage_{ordinal}_owned_summed" if summed else f"stage_{ordinal}_target_pathway" if target_path else f"stage_{ordinal}_edge_pathway" if edge_path else f"stage_{ordinal}_{'parallel' if parallel else 'canonical'}_synapse"
    guard = f"if (lane >= {lanes}u) return;" if parallel or edge_path or target_path or summed else "if (lane) return;"
    fault = ("atomic_fetch_or_explicit(&delivered[1],0x80000000u,memory_order_relaxed);"
             if parallel or summed or phase=='prefix' else "delivered[lane] |= 0x8000000000000000ul;" if edge_path or target_path else "delivered[0] |= 0x8000000000000000ul;")
    source = _PRELUDE+(SORT_SOURCE if sparse_target and not bitset else '')+f'''
kernel void {entry}({arguments}, constant long &tick [[buffer({len(types)})]], uint lane [[thread_position_in_grid]]) {{
    {guard}
    bool math_error=false;
    float time=float({owner_tick})*{_literal(number(dt_clock.dt))};
    {scalar_body}
    {before}
    {loop}
    if (math_error) {fault}
}}
'''
    return MetalKernel(syn["target_population"],entry,(node.id,),lanes,clock.start_tick,clock.steps,0,0,source), types
