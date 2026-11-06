"""Adjacent pre/post fusion under one target owner, without float atomics.

Post edge order may change, but each edge's pending/current order is unchanged.
Both stages' counters must use the target owner: their original shared array
mixes target and edge indices, which would race inside a fused dispatch.
"""
from dataclasses import replace
from .gpu_dispatch_fusion import _fuse
from .metal_synapses import target_owned_pathway,independent_pathway

ROLE='fused-target-pre-post'


def eligible(model,logical,left,right,a,b):
    if a.role!='target-owned-synapse-pathway' or b.role!='edge-owned-synapse-pathway':return False
    if a.clock!=b.clock or len(left.nodes)!=1 or len(right.nodes)!=1:return False
    nodes={n.id:n for n in logical.nodes}
    pre,post=nodes[left.nodes[0]],nodes[right.nodes[0]]
    if pre.owner_kind!='synapse' or post.owner_kind!='synapse' or pre.owner_index!=post.owner_index:return False
    syn=model['definition']['synapses'][pre.owner_index]
    inst=model['instance']['synapses'][pre.owner_index]
    first,second=(syn['code_objects'][n.item_index] for n in (pre,post))
    if second['kind']!='synapses_post' or not target_owned_pathway(model,syn,first) or not independent_pathway(syn,second):return False
    if a.lanes!=syn['target_count'] or not a.lanes or b.lanes!=max(1,len(inst['source'])):return False
    # These are the generated canonical pathway ABIs, not arbitrary kernels.
    if len(a.bindings)!=16 or len(b.bindings)!=15 or a.bindings[:7]!=b.bindings[:7]:return False
    if not left.entry.endswith('_target_pathway') or not right.entry.endswith('_edge_pathway'):return False
    written={syn['post_state_aliases'][n] for n in first['effects']['writes'] if n in syn['post_state_aliases']}
    if syn['source_population']==syn['target_population']:
        if any(syn['pre_state_aliases'].get(n) in written for n in second['effects']['reads']):return False
    # Counter redistribution is unobservable only while the reserved fault bit
    # and the total uint64 sum cannot overflow. Include every pathway sharing
    # this projection's counter array, including conservative pending expansion.
    count=len(inst['source']);bound=0
    for code in syn['code_objects']:
        if code['kind'] in {'synapses','synapses_post'}:
            path=next(p for p in inst['pathways'] if p['name']==code['pathway_name'])
            bound+=count*(logical.clocks[code['clock']].steps+len(path['pending']))
    return bound < 2**63


def fuse_pairs(model,logical,kernels,dispatches):
    kernels=list(kernels);dispatches=list(dispatches);i=0
    while i+1<len(dispatches):
        left,right=kernels[i:i+2];a,b=dispatches[i:i+2]
        if eligible(model,logical,left,right,a,b):
            nodes={n.id:n for n in logical.nodes};node=nodes[left.nodes[0]]
            count=len(model['instance']['synapses'][node.owner_index]['source'])
            # Only exact generated counter stores are rewritten. Native Function
            # declarations are injected later, after physical DAG fusion.
            source=right.source
            if (source.count('delivered[edge] += 1;') not in {1,2} or
                source.count('delivered[lane] |= 0x8000000000000000ul;')!=1):
                i+=1;continue
            source=source.replace('delivered[edge] += 1;','delivered[targets[edge]] += 1;')
            source=source.replace('delivered[lane] |= 0x8000000000000000ul;',
                'delivered['+('targets[lane]' if count else '0')+'] |= 0x8000000000000000ul;')
            fused=_fuse(left,replace(right,source=source),a,b,'_with_post_pathway',
                        right_edges=(a.bindings[15],a.bindings[8],count))
            if fused is not None:
                kernels[i],dispatches[i]=fused
                dispatches[i]=replace(dispatches[i],role=ROLE)
                del kernels[i+1];del dispatches[i+1]
        i+=1
    dispatches=[replace(d,dependencies=(dispatches[i-1].entry,) if i else ()) for i,d in enumerate(dispatches)]
    return kernels,dispatches
