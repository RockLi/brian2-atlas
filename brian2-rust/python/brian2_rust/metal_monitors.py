"""Ordered spike-variable snapshots for shared Metal/CUDA DAG execution."""
import numpy as np

from .metal import MetalKernel, _PRELUDE, number
from .plan import PlanValidationError
from . import gpu_types as gt
from .metal_event_layout import event_offset, event_coordinates


def monitors(model):
    for p, pop in enumerate(model['definition']['populations']):
        for m, monitor in enumerate(pop.get('event_monitors', [])):
            yield p, m, pop, monitor


def buffer_names(p, m):
    return (f'population/{p}/event_monitor/{m}/flags', f'population/{p}/event_monitor/{m}/values')


def monitor_arrays(pop, monitor, budget):
    # Unlike StateMonitor's raw trace window, frozen v1 EventMonitor records
    # the whole activation; the Device applies its public window on readback.
    cells=pop['count']*pop['steps']
    dtypes={s['name']:s['dtype'] for s in pop['states']+pop['parameters']+pop.get('linked_variables',[])}
    values=cells*sum(gt.width(dtypes[name]) for name in monitor['variables'])
    if max(cells,values*4)>budget:
        raise MemoryError('GPU EventMonitor snapshot buffers exceed configured memory limit; shorten the activation or raise the buffer budget')
    return [np.zeros(max(1,cells),np.uint8),np.zeros(max(1,values),np.float32)]


def monitor_kernel(model, logical, node, ordinal):
    p=node.owner_index;pop=model['definition']['populations'][p]
    monitor=pop['event_monitors'][node.item_index]
    flag_offset=event_offset(pop,monitor['event'])
    clock=logical.clocks[node.clock];n=pop['count'];window=clock.steps
    state_layout,_=gt.layout(pop['states'],n)
    parameter_layout,_=gt.layout(pop['parameters'],n)
    symbols={name:gt.read('state',field) for name,field in state_layout.items()}
    for parameter in pop['parameters']:
        symbols[parameter['name']]=gt.read('parameters',parameter_layout[parameter['name']],'0' if parameter['index_domain']=='scalar' else 'i')
    dtypes={s['name']:s['dtype'] for s in pop['states']+pop['parameters']+pop.get('linked_variables',[])}
    fields,_=gt.layout([dict(name=name,dtype=dtypes[name]) for name in monitor['variables']],n*window)
    linked_layout,_=gt.layout(pop.get('linked_variables',[]),n)
    symbols.update({name:gt.read('linked_values',field) for name,field in linked_layout.items()})
    extra='device const float *linked_values [[buffer(5)]], ' if linked_layout else ''
    tick_binding=6 if linked_layout else 5
    body=[]
    for name in monitor['variables']:
        if name not in symbols:raise PlanValidationError(f'GPU EventMonitor cannot sample {name}')
        body.append(gt.write('values',fields[name],'at',symbols[name]))
    entry=f'stage_{ordinal}_event_monitor'
    source=_PRELUDE+f'''
kernel void {entry}(device const float *state [[buffer(0)]],
    device const float *parameters [[buffer(1)]], device const uchar *fired [[buffer(2)]],
    device uchar *flags [[buffer(3)]], device float *values [[buffer(4)]],
    {extra}constant long &tick [[buffer({tick_binding})]], uint i [[thread_position_in_grid]]) {{
    if (i>={n}u || !fired[{flag_offset}+i] || tick<{clock.start_tick+clock.steps-window}) return;
    ulong at=ulong(tick-{clock.start_tick+clock.steps-window})*{n}+i;
    flags[at]=1;
    {' '.join(body)}
}}
'''
    return MetalKernel(p,entry,(node.id,),n,clock.start_tick,clock.steps,0,0,source)


def monitor_results(model, plan, arrays, populations):
    positions={name:i for i,name in enumerate(plan.buffers)}
    for p,m,pop,monitor in monitors(model):
        flags,values=(arrays[positions[name]] for name in buffer_names(p,m))
        n=pop['count'];window=pop['steps']
        rows,indices=event_coordinates(flags[:window*n].reshape(window,n))
        if len(rows)>10000000 or len(rows)*len(monitor['variables'])>10000000:
            raise MemoryError('GPU EventMonitor recording budget exceeded')
        dtypes={s['name']:s['dtype'] for s in pop['states']+pop['parameters']+pop.get('linked_variables',[])}
        fields,_=gt.layout([dict(name=name,dtype=dtypes[name]) for name in monitor['variables']],n*window)
        sampled={name:gt.unpack(values,field).reshape(window,n)[rows,indices].copy() for name,field in fields.items()}
        if not gt.finite(sampled.values()):
            raise FloatingPointError('GPU EventMonitor produced non-finite values')
        clock=plan.logical.clocks[monitor['clock']]
        ticks=rows+clock.start_tick+clock.steps-window
        populations[p].setdefault('event_monitors',{})[monitor['name']]=dict(
            ticks=ticks,indices=indices,times=ticks*number(clock.dt),values=sampled)
