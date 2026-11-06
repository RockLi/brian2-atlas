"""Independent Python execution checks affine signature and keyword ordering."""
import textwrap
import brian2 as b
import numpy as np
import pytest
from brian2_rust.training_effects import lower_state_effect_function
from brian2_rust.training_callback_noise import lift_callback_noise
from test_training_callback_noise_functions import (affine_normal_selected,affine_uniform_selected,keyword_normal_selected,keyword_uniform_selected,default_normal_selected,default_uniform_selected)


@pytest.mark.parametrize('factory',[affine_normal_selected,affine_uniform_selected,keyword_normal_selected,keyword_uniform_selected,default_normal_selected,default_uniform_selected])
def test_affine_signature_python_semantics(factory,monkeypatch):
    captured=np.array([.017,.031]);function=factory(captured)
    descriptor=lower_state_effect_function(b.Function(function,arg_units=[1,1],return_unit=1,stateless=False))
    code='out=f(h,spread)+h';lifted,helpers,sites,groups=lift_callback_noise(code,{'f':descriptor})
    def execute(text):
        captured[:]=[.017,.031];h=np.array([1.1,1.9]);spread=np.array([.09,.13]);calls=[]
        def sample(kind,*args,**kwargs):
            if kind=='normal':labels=('loc','scale','size')
            else:labels=('low','high','size')
            bound=dict(zip(labels,args));bound.update(kwargs);first=np.asarray(bound.get(labels[0],0.));second=np.asarray(bound.get(labels[1],1.));size=bound['size']
            calls.append((kind,first.copy(),second.copy(),size))
            noise=np.array([.2,.7]);return first+(second if kind=='normal' else second-first)*noise
        monkeypatch.setattr(np.random,'normal',lambda *args,**kwargs:sample('normal',*args,**kwargs));monkeypatch.setattr(np.random,'uniform',lambda *args,**kwargs:sample('uniform',*args,**kwargs))
        scope=dict(h=h,spread=spread,f=function)
        for name,helper in helpers.items():
            env=dict(helper.captured_arrays);source='def helper('+','.join(helper.arguments)+'):\n'+textwrap.indent(helper.body,'    ');exec(source,env);scope[name]=env['helper']
        for name,site in sites.items():scope[name]=lambda domain:np.array([.2,.7])
        exec(text,scope);return scope['out'].copy(),h.copy(),captured.copy(),calls
    original=execute(code);actual=execute(lifted)
    for left,right in zip(original[:3],actual[:3]):np.testing.assert_allclose(left,right)
    assert groups==[0]*len(lifted.splitlines())
    expected_orders = {
        'affine_normal_selected': ('loc', 'scale', 'size'),
        'affine_uniform_selected': ('low', 'high', 'size'),
        'keyword_normal_selected': ('scale', 'size', 'loc'),
        'keyword_uniform_selected': ('high', 'size', 'low'),
        'default_normal_selected': ('size', 'loc', 'scale'),
        'default_uniform_selected': ('size', 'low', 'high'),
    }
    assert descriptor.random_argument_orders[0][1] == expected_orders[factory.__name__]


@pytest.mark.parametrize('kind',['normal','uniform'])
def test_affine_custom_rng_rejected(kind,monkeypatch):
    factory=affine_normal_selected if kind=='normal' else affine_uniform_selected
    function=b.Function(factory(np.array([.017,.031])),arg_units=[1,1],return_unit=1,stateless=False)
    monkeypatch.setattr(np.random,kind,lambda *args,**kwargs:np.zeros(2))
    with pytest.raises(ValueError,match='custom callback random'):lower_state_effect_function(function)
