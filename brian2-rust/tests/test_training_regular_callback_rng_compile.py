"""Independent scalar Python semantics after authenticated callback lifting."""
import textwrap
import numpy as np
import pytest
from brian2_rust.training_effects import lower_state_effect_function
from brian2_rust.training_callback_scalar import lower_scalar_callback_draws
import test_training_scalar_event_rng_functions as factories


@pytest.mark.parametrize('kind',['normal','uniform','poisson','mixed','pure_normal','pure_uniform','pure_poisson','twice_normal'])
@pytest.mark.parametrize('twice',[False,True])
@pytest.mark.parametrize('offset',[0,3])
def test_regular_scalar_python_order(kind,twice,offset,monkeypatch):
    captured=np.array([.217,.379])
    function=getattr(factories,kind)(captured) if kind in ('normal','uniform','poisson','mixed') else getattr(factories,kind)
    descriptor=lower_state_effect_function(function);call='f(x,mean,spread)';code='out='+call+('+'+call if twice else '+x')
    rewritten,parameters,records,groups=lower_scalar_callback_draws(code,{'f':descriptor},{},stream_offset=offset)
    helpers={name:value for name,value in parameters.items() if hasattr(value,'body')}
    assert [row['stream'] for row in records]==list(range(offset,offset+len(records)))
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
        for entry in records:
            if entry['kind']=='poisson':scope[entry['name']]=lambda rate,_stream=entry['stream']:_stream-offset+1
            else:scope[entry['name']]=.2*(entry['stream']-offset+1)
        exec(text,scope);return np.asarray(scope['out']).copy(),captured.copy()
    expected=execute(code);actual=execute(rewritten)
    for left,right in zip(expected,actual):np.testing.assert_allclose(left,right)
    assert groups and set(groups)=={0} and len(records)>0


@pytest.mark.parametrize('offset',[-1,True,1.5,17,15])
def test_regular_scalar_stream_budget(offset):
    descriptor=lower_state_effect_function(factories.mixed(np.array([.217,.379])))
    with pytest.raises(ValueError,match='stream|16'):
        lower_scalar_callback_draws('out=f(x,mean,spread)',{'f':descriptor},{},stream_offset=offset)
