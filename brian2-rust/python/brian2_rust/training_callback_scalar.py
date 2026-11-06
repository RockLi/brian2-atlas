"""Authenticated scalar callbacks share ordered native non-array draw sites."""
from .training_callback_noise import lift_callback_noise
from .training_batch_random import cached_batch_draws
from .training_equations import NormalNoise,UniformNoise,PoissonNoise
from brian2.core.functions import DEFAULT_FUNCTIONS


def lower_scalar_callback_draws(code,functions,variables,*,stream_offset=0):
    if type(stream_offset) is not int or not 0<=stream_offset<=16:
        raise ValueError('scalar callback stream offset must be an integer in 0..16')
    lifted,helpers,sites,groups=lift_callback_noise(code,functions,scalar_call=True)
    poisson={}
    rewritten,draws=cached_batch_draws(lifted,
        {**variables,**{name:DEFAULT_FUNCTIONS[site['kind']] for name,site in sites.items()}},poisson_sites=poisson,
        callback_poisson_names={name for name,site in sites.items() if site['kind']=='poisson'},
        callback_normal_sites={name:site for name,site in sites.items() if site['kind']!='poisson'})
    records=[dict(name=name,kind='poisson',stream=site['stream']+stream_offset) for name,site in poisson.items()]
    records += [dict(name=site['name'],kind=site['kind'],stream=site['stream']+stream_offset) for site in draws]
    records.sort(key=lambda site:site['stream'])
    if any(site['stream']>=16 for site in records):raise ValueError('scalar callback exceeds 16 random streams')
    parameters={**helpers,**{site['name']:(PoissonNoise if site['kind']=='poisson' else NormalNoise if site['kind']=='randn' else UniformNoise)(site['stream']) for site in records}}
    return rewritten,parameters,records,groups
