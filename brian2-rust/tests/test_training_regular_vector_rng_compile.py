"""Independent NumPy execution of ordered vector callback lifting."""
import textwrap
import brian2 as b
import numpy as np
import pytest
from brian2.core.functions import DEFAULT_FUNCTIONS
from brian2_rust.training_callback_vector import lower_vector_callback_draws
from brian2_rust.training_effects import lower_state_effect_function
import test_training_scalar_event_rng_functions as factories


@pytest.mark.parametrize('kind,capture',[(kind,capture) for kind in ('normal','uniform','poisson','mixed') for capture in (False,True) if capture or kind!='mixed'])
@pytest.mark.parametrize('mode',['array','vectorised'])
@pytest.mark.parametrize('repeat',[False,True])
@pytest.mark.parametrize('offset',[0,3])
def test_regular_vector_python_order(kind,capture,mode,repeat,offset,monkeypatch):
    b.start_scope();group=b.NeuronGroup(2,'u:1\nmean:1 (constant)\nspread:1 (constant)')
    captured=np.array([.59,.38]);function=getattr(factories,kind)(captured) if capture else getattr(factories,'pure_'+kind)
    descriptor=lower_state_effect_function(function);call='f(u,mean,spread)';code='out='+call+('+'+call if repeat else '+u')+'\nu+=.01*randn()'
    rewritten,parameters,records,groups=lower_vector_callback_draws(code,{'f':descriptor},{**DEFAULT_FUNCTIONS,**group.variables},mode=mode,stream_offset=offset)
    assert [row['stream'] for row in records]==list(range(offset,offset+len(records)))
    def shape(first,second,noise):
        assert np.broadcast_arrays(first,second)[0].shape==(2,);return noise
    def execute(text):
        captured[:]=[.59,.38];calls=[]
        def sample(which,*args,**kwargs):
            serial=len(calls);calls.append(which);noise=.2*(serial+1)+np.array([.01,.02])
            if which=='poisson':return np.array([serial+1,serial+2],dtype=np.int64)
            if which=='randn':return noise
            first,second=args;return first+(second if which=='normal' else second-first)*noise
        for which in ('normal','uniform','poisson'):monkeypatch.setattr(np.random,which,lambda *args,_which=which,**kwargs:sample(_which,*args,**kwargs))
        scope=dict(u=np.array([.173,.219]),mean=np.array([1.3,.7]) if kind in ('poisson','mixed') else np.array([.04,.06]),spread=np.array([.13,.11]),f=function,randn=lambda:sample('randn'))
        for name,helper in parameters.items():
            if not hasattr(helper,'body'):continue
            env={**dict(helper.captured_arrays),'_b2_shape_noise':shape}
            exec('def helper('+','.join(helper.arguments)+'):\n'+textwrap.indent(helper.body,'    '),env);scope[name]=env['helper']
        for row in records:
            serial=row['stream']-offset
            scope[row['name']]=(lambda rate,_serial=serial:np.array([_serial+1,_serial+2],dtype=np.int64)) if row['kind']=='poisson' else .2*(serial+1)+np.array([.01,.02])
        exec(text,scope);return np.asarray(scope['out']).copy(),scope['u'].copy(),captured.copy()
    expected=execute(code);actual=execute(rewritten)
    for a,c in zip(expected,actual):np.testing.assert_allclose(a,c)
    assert set(groups)=={0,1}


@pytest.mark.parametrize('kind',['normal','uniform','poisson'])
def test_regular_vector_scalar_sampler_is_not_broadcast(kind):
    b.start_scope();group=b.NeuronGroup(2,'u:1\nmean:1 (constant,shared)\nspread:1 (constant,shared)')
    descriptor=lower_state_effect_function(getattr(factories,'pure_'+kind))
    with pytest.raises(ValueError,match='once-per-block'):
        lower_vector_callback_draws('u=f(u,mean,spread)',{'f':descriptor},dict(group.variables))


def pure(x):
    return .9*x+.03


@pytest.mark.parametrize('effect',[False,True])
def test_regular_vector_context_callback(effect,monkeypatch):
    from test_training_callback_effects import curve
    b.start_scope();group=b.NeuronGroup(2,'u:1\nmean:1 (constant)\nspread:1 (constant)')
    callback=curve if effect else pure;desc=lower_state_effect_function(factories.pure_normal)
    variables={**group.variables,'other':b.Function(callback,arg_units=[1],return_unit=1,stateless=not effect)}
    code='out=f(u,mean,spread)+other(u)'
    rewritten,parameters,records,_=lower_vector_callback_draws(code,{'f':desc},variables)
    def execute(text):
        scope=dict(u=np.array([.173,.219]),mean=np.array([.04,.06]),spread=np.array([.13,.11]),f=factories.pure_normal,other=callback)
        monkeypatch.setattr(np.random,'normal',lambda first,second:first+second*np.array([.2,.3]))
        for name,value in parameters.items():
            if not hasattr(value,'body'):scope[name]=np.array([.2,.3]);continue
            env={'_b2_shape_noise':lambda first,second,noise:noise}
            exec('def helper('+','.join(value.arguments)+'):\n'+textwrap.indent(value.body,'    '),env);scope[name]=env['helper']
        exec(text,scope);return scope['out'],scope['u']
    for a,c in zip(execute(code),execute(rewritten)):np.testing.assert_allclose(a,c)
