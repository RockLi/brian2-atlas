"""Independent scalar Python semantics after authenticated callback lifting."""
import textwrap
import numpy as np
import pytest
from brian2.core.functions import DEFAULT_FUNCTIONS
from brian2_rust.training_effects import lower_state_effect_function
from brian2_rust.training_callback_noise import lift_callback_noise
from brian2_rust.training_batch_random import cached_batch_draws
import test_training_scalar_event_rng_functions as factories


@pytest.mark.parametrize('kind',['normal','uniform','poisson','mixed','pure_normal','pure_uniform','pure_poisson','twice_normal'])
@pytest.mark.parametrize('twice',[False,True])
def test_scalar_fifo_python_order(kind,twice,monkeypatch):
    captured=np.array([.217,.379])
    function=getattr(factories,kind)(captured) if kind in ('normal','uniform','poisson','mixed') else getattr(factories,kind)
    descriptor=lower_state_effect_function(function);call='f(x,mean,spread)';code='out='+call+('+'+call if twice else '+x')
    lifted,helpers,sites,groups=lift_callback_noise(code,{'f':descriptor},scalar_call=True)
    poisson={};rewritten,draws=cached_batch_draws(lifted,{name:DEFAULT_FUNCTIONS[site['kind']] for name,site in sites.items()},poisson_sites=poisson,
        callback_poisson_names={name for name,site in sites.items() if site['kind']=='poisson'},callback_normal_sites={name:site for name,site in sites.items() if site['kind']!='poisson'})
    def scalar(*args):
        assert all(np.ndim(arg)==0 for arg in args);return args[-1]
    def execute(text):
        captured[:]=[.217,.379];calls=[]
        def sample(which,*args,**kwargs):
            assert all(np.ndim(arg)==0 for arg in args)
            index=len(calls);calls.append(which)
            if which=='poisson':return index+1
            first,second=args;return float(first+(second if which=='normal' else second-first)*.2*(index+1))
        for which in ('normal','uniform','poisson'):monkeypatch.setattr(np.random,which,lambda *args,_which=which,**kwargs:sample(_which,*args,**kwargs))
        scope=dict(x=.31,mean=1.3 if 'poisson' in kind or kind=='mixed' else .04,spread=.13,f=function,_b2_scalar_rate=scalar)
        for name,helper in helpers.items():
            env={**dict(helper.captured_arrays),'_b2_scalar_noise':scalar}
            exec('def helper('+','.join(helper.arguments)+'):\n'+textwrap.indent(helper.body,'    '),env);scope[name]=env['helper']
        for entry in draws:scope[entry['name']]=.2*(entry['stream']+1)
        for name,entry in poisson.items():scope[name]=lambda rate,_stream=entry['stream']:_stream+1
        exec(text,scope);return np.asarray(scope['out']).copy(),captured.copy()
    expected=execute(code);actual=execute(rewritten)
    for left,right in zip(expected,actual):np.testing.assert_allclose(left,right)
    assert groups==[0]*len(lifted.splitlines()) and len(draws)+len(poisson)==len(sites)
