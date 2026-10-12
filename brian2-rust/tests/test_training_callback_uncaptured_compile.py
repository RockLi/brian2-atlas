"""Python execution proves no-capture direct/pure returns and call ordering."""
import textwrap
import brian2 as b
import numpy as np
import pytest
from brian2_rust.training_effects import lower_state_effect_function
from brian2_rust.training_callback_noise import lift_callback_noise
import test_training_callback_uncaptured_functions as factories


@pytest.mark.parametrize('kind',['normal','uniform'])
@pytest.mark.parametrize('variant',['stored','direct','pure','scalar'])
@pytest.mark.parametrize('twice',[False,True])
def test_uncaptured_python_semantics(kind,variant,twice,monkeypatch):
    function=getattr(factories,variant+'_'+kind);descriptor=lower_state_effect_function(b.Function(function,arg_units=[1,1,1],return_unit=1,stateless=False))
    assert not descriptor.captured_arrays and descriptor.random_sites
    invocation='f(h,mean,spread)';code='out='+invocation+('+'+invocation if twice else '+h');lifted,helpers,sites,groups=lift_callback_noise(code,{'f':descriptor})
    def execute(text):
        h=np.array([1.1,1.9]);calls=[]
        def sample(first,second,*args,**kwargs):
            index=len(calls);calls.append(index);assert not args and kwargs.get('size') is None
            return float(first+(second if kind=='normal' else second-first)*(.2,.7)[index])
        monkeypatch.setattr(np.random,kind,sample);scope=dict(h=h,mean=.04,spread=.13,f=function)
        for name,helper in helpers.items():
            env=dict(helper.captured_arrays);env['_b2_shape_noise']=lambda first,second,noise:noise if np.broadcast_arrays(first,second)[0].ndim else noise[0]
            source='def helper('+','.join(helper.arguments)+'):\n'+textwrap.indent(helper.body,'    ');exec(source,env);scope[name]=env['helper']
        for index,(name,site) in enumerate(sites.items()):scope[name]=lambda *domains,_index=index:np.array([(.2,.7)[_index],(.91,.43)[_index]])
        exec(text,scope);return np.asarray(scope['out']).copy(),h.copy()
    original=execute(code);actual=execute(lifted)
    for left,right in zip(original,actual):np.testing.assert_allclose(left,right)
    assert groups==[0]*len(lifted.splitlines()) and len(sites)==(2 if twice else 1)
