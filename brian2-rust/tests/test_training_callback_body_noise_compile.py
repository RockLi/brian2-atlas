"""Independent Python array sizes, alias order and authenticated RNG identity."""
import brian2 as b
import numpy as np
import pytest
from brian2_rust.training_effects import lower_state_effect_function
from brian2_rust.training_callback_noise import lift_callback_noise
from test_training_callback_noise_functions import normal_selected,uniform_selected


@pytest.mark.parametrize('kind',['randn','rand'])
@pytest.mark.parametrize('expression',['f(h)+h','f(h)+f(h)'])
def test_noise_body_lifting_preserves_python_order(kind,expression,monkeypatch):
    captured=np.array([.017,.031]);factory=normal_selected if kind=='randn' else uniform_selected;function=factory(captured)
    descriptor=lower_state_effect_function(b.Function(function,arg_units=[1],return_unit=1,stateless=False))
    code='out='+expression;lifted,getters,sites,groups=lift_callback_noise(code,{'f':descriptor})
    def execute(text):
        captured[:]=[.017,.031];h=np.array([1.1,1.9]);calls=[]
        def sampler(size):calls.append(size);return np.array([.2,.7])
        monkeypatch.setattr(np.random,kind,sampler);scope=dict(h=h,f=function)
        for name,getter in getters.items():scope[name]=lambda getter=getter:dict(getter.captured_arrays)['array']
        for name in sites:scope[name]=lambda domain:sampler(len(domain))
        exec(text,scope);return scope['out'].copy(),h.copy(),captured.copy(),calls
    original=execute(code);actual=execute(lifted)
    for left,right in zip(original[:3],actual[:3]):np.testing.assert_array_equal(left,right)
    assert original[3]==actual[3] and groups==[0]*len(lifted.splitlines())


@pytest.mark.parametrize('kind',['randn','rand'])
def test_noise_body_custom_rng_rejected(kind,monkeypatch):
    factory=normal_selected if kind=='randn' else uniform_selected
    function=b.Function(factory(np.array([.017,.031])),arg_units=[1],return_unit=1,stateless=False)
    monkeypatch.setattr(np.random,kind,lambda size:np.zeros(size))
    with pytest.raises(ValueError,match='custom callback random'):lower_state_effect_function(function)
