"""Independent Python execution checks authenticated callback lifting."""
import brian2 as b
import numpy as np
import pytest
from brian2_rust.training_effects import lower_state_effect_function
from brian2_rust.training_callback_noise import lift_callback_noise
from test_training_callback_noise_functions import sample_selected


@pytest.mark.parametrize('expression',['f(h)','f(h)+h','f(h)+f(h)','.03*poisson(h)+f(h)'])
def test_body_lifting_preserves_python_objects_and_order(expression,monkeypatch):
    captured=np.array([.017,.031]);function=sample_selected(captured)
    descriptor=lower_state_effect_function(b.Function(function,arg_units=[1],return_unit=1,stateless=False))
    code='out='+expression;lifted,getters,sites,groups=lift_callback_noise(code,{'f':descriptor})
    def execute(text):
        captured[:]=[.017,.031];h=np.array([1.1,1.9]);calls=[]
        def poisson(rate):calls.append(np.array(rate,copy=True));return np.array([1,2],dtype=np.int64)
        monkeypatch.setattr(np.random,'poisson',poisson)
        scope=dict(h=h,f=function,poisson=poisson)
        for name,getter in getters.items():scope[name]=lambda getter=getter:dict(getter.captured_arrays)['array']
        for name in sites:scope[name]=poisson
        exec(text,scope);return scope['out'].copy(),h.copy(),captured.copy(),calls
    original=execute(code);actual=execute(lifted)
    for left,right in zip(original[:3],actual[:3]):np.testing.assert_array_equal(left,right)
    assert len(original[3])==len(actual[3])
    for left,right in zip(original[3],actual[3]):np.testing.assert_array_equal(left,right)
    assert groups==[0]*len(lifted.splitlines())


def test_callback_custom_rng_replacement_rejected(monkeypatch):
    function=b.Function(sample_selected(np.array([.017,.031])),arg_units=[1],return_unit=1,stateless=False)
    monkeypatch.setattr(np.random,'poisson',lambda rate:np.zeros_like(rate,dtype=np.int64))
    with pytest.raises(ValueError,match='custom callback random'):lower_state_effect_function(function)


def test_callback_lifting_reserved_names_rejected():
    with pytest.raises(ValueError,match='reserved callback'):lift_callback_noise('out=_b2callback_1',{})
