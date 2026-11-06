"""Ordered selected-array callback draws for scheduled NumPy vector blocks."""
import ast
from brian2.core.functions import DEFAULT_FUNCTIONS,Function
from .training_callback_noise import lift_callback_noise
from .training_batch_random import cached_batch_draws
from .training_batch_locals import batch_local_lifetimes
from .training_equations import NormalNoise,UniformNoise,PoissonNoise
from .training_effects import lower_state_effect_function
from .training_functions import lower_pure_function


def lower_vector_callback_draws(code,functions,variables,*,stream_offset=0,mode='array',allow_scalar=False,return_trace=False):
    if type(stream_offset) is not int or not 0<=stream_offset<=16:
        raise ValueError('vector callback stream offset must be an integer in 0..16')
    if any(node.id.startswith('__regular_whole_') or node.id=='_b2_scheduled_poisson' for node in ast.walk(ast.parse(code)) if isinstance(node,ast.Name)):
        raise ValueError('reserved scheduled callback name collision')
    lifted,helpers,sites,groups=lift_callback_noise(code,functions)
    if allow_scalar:
        from .training_batch_poisson import stage_batch_poisson
        expanded=[]
        lifted=stage_batch_poisson(lifted,statement_groups=expanded,input_statement_groups=groups,poisson_names={'poisson',*{name for name,site in sites.items() if site['kind']=='poisson'}})
        groups=expanded
    poisson={}
    caller_variables={**variables,**{name:DEFAULT_FUNCTIONS[site['kind']] for name,site in sites.items()}}
    rewritten,draws=cached_batch_draws(lifted,caller_variables,poisson_sites=poisson,
        callback_poisson_names={name for name,site in sites.items() if site['kind']=='poisson'},
        callback_normal_sites={name:site for name,site in sites.items() if site['kind']!='poisson'})
    called={node.func.id for node in ast.walk(ast.parse(rewritten)) if isinstance(node,ast.Call) and isinstance(node.func,ast.Name)}
    context={}
    for name in called:
        variable=variables.get(name)
        if type(variable) is not Function or name in DEFAULT_FUNCTIONS:continue
        try:context[name]=lower_state_effect_function(variable)
        except ValueError:context[name]=lower_pure_function(variable)
    trace=batch_local_lifetimes(rewritten,caller_variables,
        {**context,**helpers,**{name:PoissonNoise(site['stream']) for name,site in poisson.items()}},mode=mode,
        synthetic_types={site['name']:'float' for site in draws},statement_groups=groups,
        normal_domains={site['name']:site['callback_domain'] for site in draws if 'callback_domain' in site},
        callback_poisson_names={name for name,site in poisson.items() if site.get('callback')},whole_normal_domains=allow_scalar)
    if not allow_scalar and (trace['scalar_normal_sites'] or trace['scalar_poisson_sites']):
        raise ValueError('vector callback scalar random sites require once-per-block sampling')
    records=[dict(name=name,kind='poisson',stream=site['stream']+stream_offset) for name,site in poisson.items()]
    records += [dict(name=site['name'],kind=site['kind'],stream=site['stream']+stream_offset) for site in draws]
    records.sort(key=lambda site:site['stream'])
    if any(site['stream']>=16 for site in records):raise ValueError('vector callback exceeds 16 random streams')
    parameters={**helpers,**{site['name']:(PoissonNoise if site['kind']=='poisson' else NormalNoise if site['kind']=='randn' else UniformNoise)(site['stream']) for site in records}}
    if return_trace:
        for record in records:
            shapes=trace['poisson_shapes'] if record['kind']=='poisson' else trace['normal_shapes']
            record.update(shapes.get(record['name'],dict(scalar=False,length=None)))
        return rewritten,parameters,records,groups,trace
    return rewritten,parameters,records,groups
