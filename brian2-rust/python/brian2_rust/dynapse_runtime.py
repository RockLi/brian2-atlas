"""Execute the stock Rockpool DynapSim and LinearJax modules on CPU."""
import hashlib
from pathlib import Path
import numpy as np


def run(plan):
    import jax
    from rockpool.devices.dynapse import DynapSim
    from rockpool.nn.modules.jax import LinearJax
    from rockpool.nn.combinators import Sequential
    import rockpool.devices.dynapse.simulation.dynapsim as implementation
    if plan.get('schema')!='atlas.dynapse.dpi-plan.v1':raise ValueError('unknown DynapSim plan')
    if jax.default_backend()!='cpu':raise ValueError('pilot requires JAX CPU')
    n=plan['neurons'];params={k:np.asarray(v) for k,v in plan['parameters'].items()}
    weights=np.diag(params.pop('weight'))
    params['Iscale']=float(params['Iscale'][0])  # SDK-wide scalar, fixed by admission
    net=Sequential(LinearJax((n,n),weight=weights,has_bias=False),
        DynapSim(n,**params,has_rec=False,percent_mismatch=None,
                 rng_key=np.array([0,1],dtype=np.uint32),dt=plan['dt_seconds']))
    output,state,record=net.evolve(np.asarray(plan['input_raster']),record=True)
    recorded=record['1_DynapSim'];final=state['1_DynapSim']
    result={k:np.asarray(recorded['spikes' if k=='q' else k])[0] for k in ('iahp','imem','isyn','q','vmem')}
    result['timer_ref_final']=np.asarray(final['timer_ref'])[0]
    result['spikes']=np.column_stack(np.nonzero(result['q']))
    result['sdk_source_sha256']=hashlib.sha256(Path(implementation.__file__).read_bytes()).hexdigest()
    return result
