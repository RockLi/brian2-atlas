"""Independent Python scalar counts, immutable aliases and repeated-call order."""
import textwrap
import numpy as np
import pytest
from brian2.core.functions import DEFAULT_FUNCTIONS
from brian2_rust.training_effects import lower_state_effect_function
from brian2_rust.training_callback_noise import lift_callback_noise
from brian2_rust.training_batch_poisson import stage_batch_poisson
from brian2_rust.training_batch_random import cached_batch_draws
from brian2_rust.training_callback_poisson import split_scalar_callback_poisson
import test_training_callback_scalar_poisson_functions as factories


@pytest.mark.parametrize('variant',['direct','stored','response','twice','default','derived'])
@pytest.mark.parametrize('twice',[False,True])
def test_scalar_poisson_python_semantics(variant,twice,monkeypatch):
    function=getattr(factories,variant+'_poisson');descriptor=lower_state_effect_function(function)
    invocation='f(h,rate,unused)';code='out='+invocation+('+'+invocation if twice else '+h')
    lifted,getters,sites,input_groups=lift_callback_noise(code,{'f':descriptor})
    groups=[];staged=stage_batch_poisson(lifted,statement_groups=groups,input_statement_groups=input_groups,poisson_names=set(sites))
    poisson={};cached,draws=cached_batch_draws(staged,{name:DEFAULT_FUNCTIONS['poisson'] for name in sites},poisson_sites=poisson,callback_poisson_names=set(sites))
    assert not draws
    rewritten,scopes,helpers,domains,caches=split_scalar_callback_poisson(cached,set(poisson),groups)
    assert len(caches)==(2 if variant=='twice' else 1)*(2 if twice else 1)
    def execute(text):
        h=np.array([1.1,1.9]);calls=[]
        def sample(*args,**kwargs):
            rate=args[0] if args else kwargs.get('lam',1.)
            assert np.ndim(rate)==0 and float(rate)==pytest.approx(1. if variant=='default' else 2.6 if variant=='derived' else 1.3)
            calls.append(float(rate));return len(calls)+1
        monkeypatch.setattr(np.random,'poisson',sample)
        scope=dict(h=h,rate=1.3,unused=.13,f=function)
        for name,helper in {**getters,**helpers}.items():
            env={'_b2_shape_noise':lambda rate,unused,noise:noise[0]}
            exec('def helper('+','.join(helper.arguments)+'):\n'+textwrap.indent(helper.body,'    '),env)
            scope[name]=env['helper']
        for name in poisson:scope[name]=lambda rate:np.array([sample(rate),99],dtype=np.int32)
        exec(text,scope);return np.asarray(scope['out']).copy(),h.copy(),calls
    original=execute(code);actual=execute(rewritten)
    for left,right in zip(original[:2],actual[:2]):np.testing.assert_allclose(left,right)
    assert original[2]==actual[2] and scopes==[0]*len(rewritten.splitlines())
