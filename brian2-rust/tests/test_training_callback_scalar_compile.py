"""Independent Python execution of scalar samples and repeated callback order."""
import textwrap
import brian2 as b
import numpy as np
import pytest
from brian2_rust.training_effects import lower_state_effect_function
from brian2_rust.training_callback_noise import lift_callback_noise
from test_training_callback_noise_functions import scalar_normal_selected,scalar_uniform_selected,scalar_randn_selected,scalar_rand_selected,scalar_parameter_normal_selected,scalar_parameter_uniform_selected


@pytest.mark.parametrize('factory',[scalar_normal_selected,scalar_uniform_selected,scalar_randn_selected,scalar_rand_selected,scalar_parameter_normal_selected,scalar_parameter_uniform_selected])
@pytest.mark.parametrize('twice',[False,True])
def test_scalar_python_semantics(factory,twice,monkeypatch):
    captured=np.array([.017,.031]);function=factory(captured);parameterised='parameter' in factory.__name__;arity=3 if parameterised else 2
    descriptor=lower_state_effect_function(b.Function(function,arg_units=[1]*arity,return_unit=1,stateless=False))
    invocation='f(h,mean,spread)' if parameterised else 'f(h,spread)';code='out='+invocation+('+'+invocation if twice else '+h');lifted,helpers,sites,groups=lift_callback_noise(code,{'f':descriptor})
    def execute(text):
        captured[:]=[.017,.031];h=np.array([1.1,1.9]);spread=.13 if parameterised else np.array([.09,.13]);calls=[]
        def sample(kind,*args,**kwargs):
            index=len(calls);calls.append(kind);noise=(.2,.7)[index]
            if kind in ('randn','rand'):assert not args and not kwargs;return noise
            first,second=args;assert np.broadcast_arrays(first,second)[0].shape==() and kwargs.get('size') is None
            return first+(second if kind=='normal' else second-first)*noise
        for kind in ('normal','uniform','randn','rand'):monkeypatch.setattr(np.random,kind,lambda *args,_kind=kind,**kwargs:sample(_kind,*args,**kwargs))
        scope=dict(h=h,mean=.04,spread=spread,f=function)
        for name,helper in helpers.items():
            env=dict(helper.captured_arrays);env['_b2_shape_noise']=lambda first,second,noise:noise if np.broadcast_arrays(first,second)[0].ndim else noise[0]
            source='def helper('+','.join(helper.arguments)+'):\n'+textwrap.indent(helper.body,'    ');exec(source,env);scope[name]=env['helper']
        for index,(name,site) in enumerate(sites.items()):scope[name]=lambda *domains,_index=index:np.array([(.2,.7)[_index],(.91,.43)[_index]])
        exec(text,scope);return scope['out'].copy(),h.copy(),captured.copy()
    original=execute(code);actual=execute(lifted)
    for left,right in zip(original,actual):np.testing.assert_allclose(left,right)
    assert groups==[0]*len(lifted.splitlines()) and len(sites)==(2 if twice else 1)
