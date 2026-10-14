"""Independent Python original/lifted semantics for implicit sample shapes."""
import textwrap
import brian2 as b
import numpy as np
import pytest
from brian2_rust.training_effects import lower_state_effect_function
from brian2_rust.training_callback_noise import lift_callback_noise
from test_training_callback_noise_functions import inferred_normal_selected,inferred_uniform_selected,inferred_scale_normal_selected,inferred_high_uniform_selected


@pytest.mark.parametrize('factory',[inferred_normal_selected,inferred_uniform_selected,inferred_scale_normal_selected,inferred_high_uniform_selected])
def test_inferred_python_semantics(factory,monkeypatch):
    captured=np.array([.017,.031]);function=factory(captured)
    descriptor=lower_state_effect_function(b.Function(function,arg_units=[1,1],return_unit=1,stateless=False))
    code='out=f(h,spread)+h';lifted,helpers,sites,groups=lift_callback_noise(code,{'f':descriptor})
    def execute(text):
        captured[:]=[.017,.031];h=np.array([1.1,1.9]);spread=np.array([.09,.13])
        def sample(kind,*args,**kwargs):
            labels=('loc','scale','size') if kind=='normal' else ('low','high','size');bound=dict(zip(labels,args));bound.update(kwargs)
            assert bound.get('size') is None
            first,second=np.broadcast_arrays(bound.get(labels[0],0.),bound.get(labels[1],1.));assert first.shape==(2,)
            noise=np.array([.2,.7]);return first+(second if kind=='normal' else second-first)*noise
        monkeypatch.setattr(np.random,'normal',lambda *args,**kwargs:sample('normal',*args,**kwargs));monkeypatch.setattr(np.random,'uniform',lambda *args,**kwargs:sample('uniform',*args,**kwargs))
        scope=dict(h=h,spread=spread,f=function)
        for name,helper in helpers.items():
            env=dict(helper.captured_arrays)
            env['_b2_shape_noise']=lambda first,second,noise:noise if np.broadcast_arrays(first,second)[0].ndim else noise[0]
            source='def helper('+','.join(helper.arguments)+'):\n'+textwrap.indent(helper.body,'    ');exec(source,env);scope[name]=env['helper']
        for name,site in sites.items():
            def noise(first,second):
                assert np.broadcast_arrays(first,second)[0].shape==(2,)
                return np.array([.2,.7])
            scope[name]=noise
        exec(text,scope);return scope['out'].copy(),h.copy(),captured.copy()
    original=execute(code);actual=execute(lifted)
    for left,right in zip(original,actual):np.testing.assert_allclose(left,right)
    assert groups==[0]*len(lifted.splitlines())
