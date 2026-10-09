"""Conservative lane-local fusion after canonical GPU DAG expansion.

Delayed sparse enqueue and mutable pathway-history stages participate.
Buffer hazards prove source enqueue/history commutation; equal target ownership and absence of source-state reads
prove target delivery composition. Every remaining dispatch stays a barrier.
"""
from dataclasses import replace
import re


def _writes(dispatch):
    return {b for b,t in zip(dispatch.bindings,dispatch.types,strict=True) if not t.startswith('const ')}


def _commute(a,b):
    return not (_writes(a)&set(b.bindings) or _writes(b)&set(a.bindings))


def _arguments(a,b):
    bindings=[];types=[];positions={};calls=[]
    for dispatch in (a,b):
        call=[]
        for binding,dtype in zip(dispatch.bindings,dispatch.types,strict=True):
            base=dtype.removeprefix('const ')
            if binding not in positions:
                positions[binding]=len(bindings);bindings.append(binding);types.append(dtype)
            at=positions[binding]
            if types[at].removeprefix('const ')!=base:return None
            if not dtype.startswith('const '):types[at]=base
            call.append(at)
        calls.append(call)
    # Metal has 31 buffer slots; the absolute tick consumes the last slot.
    if len(bindings)>30:return None
    return tuple(bindings),tuple(types),calls


def _fuse(left,right,a,b,suffix,*,right_edges=None):
    from .metal import _PRELUDE
    from .metal_events import SORT_SOURCE
    arguments=_arguments(a,b)
    if arguments is None:return None
    bindings,types,calls=arguments
    sort=SORT_SOURCE if SORT_SOURCE in left.source or SORT_SOURCE in right.source else ''
    def helper(kernel):
        text=kernel.source.replace(_PRELUDE,'').replace(SORT_SOURCE,'').replace('kernel void','inline void')
        return re.sub(r'\[\[(?:buffer\(\d+\)|thread_position_in_grid)\]\]','',text)
    parameters=',\n    '.join(f'device {dtype} *b{i} [[buffer({i})]]' for i,dtype in enumerate(types))
    first=', '.join('b'+str(i) for i in calls[0]);second=', '.join('b'+str(i) for i in calls[1])
    guard=''
    right_call=f'{right.entry}({second},tick,i);'
    if right_edges is not None:
        offsets,edges,count=right_edges
        offsets=bindings.index(offsets);edges=bindings.index(edges)
        guard=f'if(i>={a.lanes}u) return;\n    '
        right_call=f'''for(uint pos=b{offsets}[i];pos<b{offsets}[i+1];++pos) {{
        {right.entry}({second},tick,b{edges}[pos]);
    }}'''
        if not count:right_call+=f'\n    if(i==0u) {right.entry}({second},tick,0u);'
    entry=left.entry+suffix
    source=_PRELUDE+sort+helper(left)+helper(right)+f'''
kernel void {entry}({parameters}, constant long &tick [[buffer({len(types)})]],
    uint i [[thread_position_in_grid]]) {{
    {guard}{left.entry}({first},tick,i);
    {right_call}
}}
'''
    kernel=replace(left,entry=entry,nodes=left.nodes+right.nodes,source=source)
    dispatch=replace(a,entry=entry,bindings=bindings,types=types)
    return kernel,dispatch


def fuse_dispatches(model,logical,kernels,dispatches,*,target_delivery=True,pathway_history=True):
    """Preserve per-clock semantics while reducing proven lane-local barriers."""
    kernels=list(kernels);dispatches=list(dispatches)
    populations=model['definition']['populations']
    i=0
    while i<len(dispatches):
        enqueue=dispatches[i];right=kernels[i]
        history=enqueue.role in {'edge-pathway-history','target-pathway-history'}
        if enqueue.role!='delay-source-group-enqueue' and not (pathway_history and history):i+=1;continue
        for j in range(i-1,-1,-1):
            previous=dispatches[j];left=kernels[j]
            if previous.clock!=enqueue.clock:break
            if (previous.role in {'population','population-delay-source-enqueue','population-pathway-history'}
                    and left.population==right.population and left.neurons==right.neurons
                    and right.neurons==populations[right.population]['count']):
                fused=_fuse(left,right,previous,enqueue,'_with_pathway_history' if history else '_with_source_enqueue')
                if fused is not None:
                    kernels[j],dispatches[j]=fused
                    dispatches[j]=replace(dispatches[j],role='population-pathway-history' if history else 'population-delay-source-enqueue')
                    del kernels[i];del dispatches[i];i-=1
                break
            # Moving the enqueue across even one aliasing access or a different
            # clock is forbidden. Mutable bindings count as reads AND writes.
            if not _commute(previous,enqueue):break
        i+=1
    nodes={n.id:n for n in logical.nodes}
    def target(kernel,dispatch):
        if dispatch.role!='delayed-target-sparse' or len(kernel.nodes)!=1:return None
        node=nodes[kernel.nodes[0]]
        syn=model['definition']['synapses'][node.owner_index]
        code=syn['code_objects'][node.item_index]
        # Cross-lane WAR/RAW can arise between two otherwise valid pathways:
        # one reads v_pre while the other writes v_post. Reject all pre-state
        # reads, even when a more detailed field proof could permit them.
        if set(code['effects']['reads'])&set(syn['pre_state_aliases']):return None
        return syn['target_population'],syn['target_start'],syn['target_count']
    i=0
    while target_delivery and i+1<len(dispatches):
        a,b=dispatches[i:i+2];left,right=kernels[i:i+2]
        owner=target(left,a)
        if owner is not None and owner==target(right,b) and a.clock==b.clock and a.lanes==b.lanes:
            fused=_fuse(left,right,a,b,'_with_target_delivery')
            if fused is not None:
                kernels[i],dispatches[i]=fused
                dispatches[i]=replace(dispatches[i],role='fused-delayed-target-sparse')
                del kernels[i+1];del dispatches[i+1]
        i+=1
    # These are the physical barriers after verified reordering. Logical node
    # identities remain unchanged, including both nodes of a fused delivery.
    dispatches=[replace(d,dependencies=(dispatches[i-1].entry,) if i else ()) for i,d in enumerate(dispatches)]
    return kernels,dispatches
