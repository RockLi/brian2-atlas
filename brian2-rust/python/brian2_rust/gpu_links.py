"""Node-entry linked-input snapshots for the canonical Metal/CUDA schedule."""
import numpy as np
import re
from dataclasses import replace

from . import gpu_types as gt
from .plan import PlanValidationError


def buffer_names(population):
    return (f'population/{population}/linked_values',f'population/{population}/linked_indices')


def links_for_node(model,node):
    if node.owner_kind!='population':return []
    pop=model['definition']['populations'][node.owner_index]
    if node.operation=='code_object':
        names=pop['code_objects'][node.item_index]['effects']['reads']
    elif node.operation=='state_monitor':names=pop['monitor']['variables']
    elif node.operation=='event_monitor':names=pop['event_monitors'][node.item_index]['variables']
    else:names=[]
    return [link for link in pop.get('linked_variables',[]) if link['name'] in names]


def index_layout(pop):
    result={};offset=0
    for link in pop.get('linked_variables',[]):
        if link['index']['kind'] in {'constant','parameter'}:
            result[link['name']]=offset;offset+=pop['count']
    return result,offset


def arrays(model,p,budget):
    pop=model['definition']['populations'][p];inst=model['instance']['populations'][p]
    _,words=gt.layout(pop['linked_variables'],pop['count'])
    offsets,indices=index_layout(pop)
    if max(words*4,indices*8)>budget:
        raise MemoryError('GPU linked-input buffers exceed configured memory limit')
    table=np.zeros(max(1,indices),np.uint64)
    for link in pop['linked_variables']:
        mapping=link['index'];kind=mapping['kind']
        if kind not in {'constant','parameter'}:continue
        if kind=='constant':values=np.asarray(mapping['values'],np.uint64)
        else:
            symbol=next(s for s in pop['parameters'] if s['name']==mapping['name'])
            field=(symbol['dtype'],0,pop['count'])
            values=gt.unpack(gt.pack(inst['parameters'][symbol['name']],symbol['dtype']),field)
            # Reference materializes parameter mappings at initialization,
            # even when no code object uses this declared link. Source bounds
            # are checked when a selected lane actually reads the link.
            if values.dtype.kind=='i' and np.any(values<0):
                raise PlanValidationError('GPU linked variable index parameter is negative')
            values=values.astype(np.uint64)
        start=offsets[link['name']];table[start:start+pop['count']]=values
    return [np.zeros(max(1,words),np.float32),table]


def gather_kernel(model,logical,node,ordinal,link):
    from .metal import MetalKernel,_PRELUDE
    from .metal_event_layout import event_offset
    p=node.owner_index;pop=model['definition']['populations'][p]
    source=model['definition']['populations'][link['source_population']]
    clock=logical.clocks[node.clock];n=pop['count']
    mapping=link['index'];kind=mapping['kind']
    if kind=='state':
        field=gt.layout(pop['states'],n)[0][mapping['name']]
        ctype=gt.CTYPES[field[0]];value=gt.read('local_state',field)
        negative='index < 0 || ' if field[0].startswith('i') else ''
    else:
        ctype='ulong';negative=''
        value='ulong(i)' if kind=='identity' else f'indices[{index_layout(pop)[0][link["name"]]}+i]'
    selection=''
    if node.operation=='code_object':
        code=pop['code_objects'][node.item_index]
        if code['kind']=='reset':selection=f"if (!fired[{event_offset(pop,code['event_name'])}+i]) return;"
    elif node.operation=='state_monitor':
        selection=f"if (record_slot[i]<0 || tick<{clock.start_tick+clock.steps-pop['monitor']['window_steps']}) return;"
    elif node.operation=='event_monitor':
        monitor=pop['event_monitors'][node.item_index]
        selection=f"if (!fired[{event_offset(pop,monitor['event'])}+i]) return;"
    source_field=gt.layout(source['states'],source['count'])[0][link['source_state']]
    destination=gt.layout(pop['linked_variables'],n)[0][link['name']]
    store=gt.write('linked_values',destination,'i',gt.read('source_state',source_field,'index'))
    entry=f'stage_{ordinal}_link_{pop["linked_variables"].index(link)}'
    text=_PRELUDE+f'''
kernel void {entry}(device const float *source_state [[buffer(0)]],
    device const float *local_state [[buffer(1)]],device const ulong *indices [[buffer(2)]],
    device float *linked_values [[buffer(3)]],device long *until_tick [[buffer(4)]],
    device const uchar *fired [[buffer(5)]],device const int *record_slot [[buffer(6)]],
    constant long &tick [[buffer(7)]],uint i [[thread_position_in_grid]]) {{
    if (i>={n}u) return;
    {selection}
    {ctype} index={value};
    if ({negative}ulong(index)>={source['count']}ul) {{ until_tick[i]=-1; return; }}
    {store}
}}
'''
    return MetalKernel(p,entry,(node.id,),n,clock.start_tick,clock.steps,0,0,text)


def serial_self_links(model,node):
    if node.operation!='code_object' or node.owner_kind!='population':return []
    links=links_for_node(model,node)
    if not any(link['source_population']==node.owner_index and link['index']['kind']!='identity' for link in links):
        return []
    return [link for link in links if link['source_population']==node.owner_index]


def self_link_kernel(model,logical,node,ordinal,population,types):
    """Raw B2IR can express self mappings excluded by the Brian frontend.

    Preserve reference batching: regular/threshold/update inputs are bound per
    256-neuron batch; resets bind/commit one fired neuron at a time. One GPU lane
    executes these batches. Foreign-source gathers can run before this kernel,
    since only this population changes during its code object.
    """
    from .metal import _PRELUDE
    def helper(kernel):
        source=kernel.source.replace(_PRELUDE,'').replace('kernel void','inline void')
        return re.sub(r'\[\[(?:buffer\(\d+\)|thread_position_in_grid)\]\]','',source)
    pop=model['definition']['populations'][node.owner_index]
    code=pop['code_objects'][node.item_index]
    batch=1 if code['kind']=='reset' else 256
    gathers=[gather_kernel(model,logical,node,ordinal,link) for link in serial_self_links(model,node)]
    helpers='\n'.join(helper(gather) for gather in gathers)+'\n'+helper(population)
    # Population ABI: its 12 standard buffers followed by the linked cache.
    # The one extra buffer contains constant/parameter mapping indices.
    arguments=', '.join(f'device {t} *b{i} [[buffer({i})]]' for i,t in enumerate((*types,'const ulong')))
    calls='\n'.join(f'{gather.entry}(b0,b0,b13,b12,b8,b6,b3,tick,i);' for gather in gathers)
    entry=population.entry+'_self_links'
    source=_PRELUDE+helpers+f'''
kernel void {entry}({arguments},constant long &tick [[buffer(14)]],uint lane [[thread_position_in_grid]]) {{
    if (lane) return;
    for (uint start=0;start<{pop['count']}u;start+={batch}u) {{
        uint end=(start+{batch}u < {pop['count']}u) ? start+{batch}u : {pop['count']}u;
        for (uint i=start;i<end;++i) {{ {calls} }}
        for (uint i=start;i<end;++i) {{ {population.entry}({','.join('b'+str(i) for i in range(13))},tick,i); }}
    }}
}}
'''
    return replace(population,entry=entry,neurons=1,source=source)
