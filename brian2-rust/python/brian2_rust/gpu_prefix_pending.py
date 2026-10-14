"""Bounded proof/mask for one pre event per edge and pathway tick."""

MAX_MASK_BYTES=16*1024**2
MAX_EXPANDED_EVENTS=1024**2


def pending_spec(model,syn,code,path):
    """Return a read-only due-event bitmap recipe, or decline unsafe splitting.

    Prefix and target remainder may be separated only if a pending event never
    shares an edge/tick with another pending or potentially current event. Future
    entries beyond this activation are not consumed and need no bitmap bit.
    """
    q=model['definition']['synapses'].index(syn)
    inst=model['instance']['synapses'][q];count=len(inst['source'])
    clock=model['run']['clocks'][code['clock']];start=clock['start_tick'];end=start+clock['steps']
    active=[]
    for event in path['pending']:
        tick=event['delivery_tick']
        if tick<start:return None  # Canonical replay can drain several overdue entries together.
        if tick<end:
            active.append(event)
            if len(active)>MAX_EXPANDED_EVENTS:return None
    if not active:return dict(edges=count,span=0,words=0,bits={})
    span=max(e['delivery_tick']-start for e in active)+1;words=(span+31)//32
    if not count or words*count*4>MAX_MASK_BYTES:return None
    delays=path['delay_ticks'];uniform=bool(delays) and all(d==delays[0] for d in delays)
    by_source={}
    if uniform:
        for edge,source in enumerate(inst['source']):by_source.setdefault(source,[]).append(edge)
    bits={};expanded=0
    for event in active:
        tick=event['delivery_tick']-start
        edges=by_source.get(event['item'],()) if uniform else (event['item'],)
        for edge in edges:
            expanded+=1
            if expanded>MAX_EXPANDED_EVENTS:return None
            delay=delays[0] if uniform else delays[edge]
            # Current history can first deliver on start+delay. Do not assume
            # source spikes are absent when an imported event overlaps it.
            if tick>=delay:return None
            index=(tick//32)*count+edge;bit=1<<(tick%32)
            old=bits.get(index,0)
            if old & bit:return None
            bits[index]=old|bit
    return dict(edges=count,span=span,words=words,bits=bits)


def pending_mask(spec,max_buffer_bytes):
    import numpy as np
    size=spec['words']*spec['edges']
    if size*4>max_buffer_bytes:raise MemoryError('GPU prefix pending mask exceeds configured memory limit')
    result=np.zeros(size,np.uint32)
    for index,bits in spec['bits'].items():result[index]=bits
    return result
