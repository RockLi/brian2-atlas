"""Prove an active-edge prefix can precede ordered target-neuron delivery."""


def nodes(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from nodes(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from nodes(child)


def prefix_length(model, syn, code, path):
    from .metal_synapses import target_owned_pathway
    if path is None or not target_owned_pathway(model, syn, code):
        return 0
    # Amortize a second dispatch only for substantial edge work. This is a
    # bounded static heuristic, not a profile-guided throughput promise.
    count=len(model['instance']['synapses'][model['definition']['synapses'].index(syn)]['source'])
    if count < 16384 or count < 4*syn['target_count']:
        return 0
    from .gpu_prefix_pending import pending_spec
    if pending_spec(model,syn,code,path) is None:return 0
    states={s['name'] for s in syn['states'] if s['index_domain']=='synapse'}
    endpoints=set(syn['pre_state_aliases'])|set(syn['post_state_aliases'])
    length=0
    for statement in code['vector']:
        target=statement['target']
        # An assignment to a local is safe to compute in an edge lane, but its
        # value cannot cross the stage boundary without explicit storage.
        if target not in states and (target in code['effects']['writes'] or target in endpoints):
            break
        if statement.get('condition') in endpoints:
            break
        if any(n.get('op')=='load' and n.get('name') in endpoints for n in nodes(statement)):
            break
        length+=1
    if length==len(code['vector']):return 0
    # Search backwards for the longest boundary with no live local values.
    # Reading a local after redefining it also rejects the larger boundary;
    # this conservative check avoids introducing a new SSA/dataflow contract.
    for size in range(length,0,-1):
        prefix,remainder=code['vector'][:size],code['vector'][size:]
        locals_={s['target'] for s in prefix if s['target'] not in states}
        reads={n['name'] for n in nodes(remainder) if n.get('op')=='load'}
        reads.update(s['condition'] for s in remainder if s.get('condition') is not None)
        if locals_ & reads:continue
        if not any(s['target'] in states for s in prefix):continue
        if not any(n.get('op') in {'exp','log','pow','sin','cos'} for n in nodes(prefix)):continue
        return size
    return 0
