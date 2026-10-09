"""Exact common on_pre kernel selection from validated instruction trees."""


def shared_additive(model,q,code):
    syn=model['definition']['synapses'][q]
    if code['kind']!='synapses' or code['scalar'] or len(code['vector'])!=1:
        return None
    instruction=code['vector'][0];target=instruction['target']
    if instruction.get('condition') is not None or instruction['dtype']!='f64' or target not in syn['post_state_aliases']:
        return None
    value=instruction['value']
    if value.get('op')!='add' or value.get('left')!={'op':'load','name':target}:
        return None
    right=value.get('right',{})
    if right.get('op')!='load':return None
    weight=next(((i,p) for i,p in enumerate(syn['parameters']) if p['name']==right['name']),None)
    if weight is None or weight[1]['dtype']!='f64':return None
    # No instructions/effects beyond the one exact target += immutable weight.
    if set(code['effects']['writes'])!={target} or set(code['effects']['reads'])!={target,right['name']}:
        return None
    pop=syn['target_population'];state=syn['post_state_aliases'][target]
    column=next(i for i,s in enumerate(model['definition']['populations'][pop]['states']) if s['name']==state)
    pathways=model['instance']['synapses'][q]['pathways']
    r,path=next((r,p) for r,p in enumerate(pathways) if p['name']==code['pathway_name'])
    if path['kind']!='pre' or path['event']!='spike':return None
    prefix=f's{q}_' if r==0 else f's{q}p{r}_'
    delays=path['delay_ticks'];uniform=bool(delays) and all(x==delays[0] for x in delays)
    delay=f'MpiDelays::Uniform({prefix}delay_ticks)' if uniform else f'MpiDelays::Edges(&{prefix}delay_ticks)'
    i,parameter=weight
    w=f'MpiWeights::Scalar(s{q}_parameter_{i})' if parameter['index_domain']=='scalar' else f'MpiWeights::Edges(&s{q}_parameter_{i})'
    source=syn['source_population'];queue=f'plan_s{q}p{r}_queue'
    return [f's{q}_delivered += mpi_additive_projection(&p{source}_fired, {syn["source_start"]}, {syn["source_count"]}, p{source}_tick, p{source}_end_tick,',
            f'    &s{q}_offsets, &s{q}_target_index, {delay}, {w}, &mut {queue},',
            f'    &mut p{pop}_state_{column}, {syn["target_start"]}, p{pop}_start);']
