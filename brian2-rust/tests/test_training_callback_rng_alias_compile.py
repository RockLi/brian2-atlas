"""Exact imported identities, Python behavior and shadow/replacement boundaries."""
import textwrap
import numpy as np
import pytest
from brian2_rust.training_effects import lower_state_effect_function
from brian2_rust.training_callback_noise import lift_callback_noise
import test_training_callback_rng_alias_functions as aliases
import test_training_callback_uncaptured_functions as qualified


@pytest.mark.parametrize('kind',['normal','uniform','randn','rand','poisson'])
@pytest.mark.parametrize('spelling',['module','function'])
def test_alias_python_semantics(kind,spelling,monkeypatch):
    function=getattr(aliases,spelling+'_'+kind)
    baseline=getattr(qualified,'scalar_'+kind) if kind in ('normal','uniform') else getattr(aliases,'qualified_'+kind)
    descriptor=lower_state_effect_function(function)
    assert descriptor==lower_state_effect_function(baseline)
    arguments='h,mean,spread' if kind in ('normal','uniform') else 'h'
    code='out=f('+arguments+')+f('+arguments+')+h'
    lifted,helpers,sites,groups=lift_callback_noise(code,{'f':descriptor})
    assert len(sites)==2 and groups==[0]*len(lifted.splitlines())
    def execute(text):
        h=np.array([1.1,1.9]);calls=[]
        def sample(*args,**kwargs):
            calls.append((tuple(np.array(a,copy=True) for a in args),dict(kwargs)))
            return (.2,.7)[len(calls)-1] if kind in ('normal','uniform') else np.array([.2,.7])
        monkeypatch.setattr(np.random,kind,sample)
        monkeypatch.setattr(aliases,'imported_'+kind,sample)
        scope=dict(h=h,mean=.04,spread=.13,f=function)
        for name,helper in helpers.items():
            env={'_b2_shape_noise':lambda first,second,noise:noise[0]}
            exec('def helper('+','.join(helper.arguments)+'):\n'+textwrap.indent(helper.body,'    '),env)
            scope[name]=env['helper']
        for index,(name,site) in enumerate(sites.items()):
            if kind in ('normal','uniform'):
                scope[name]=lambda *args,_index=index:np.array([((.2,.7)[_index]-.04)/.13,99.])
            else:
                scope[name]=lambda domain,_kind=kind:sample(domain if _kind=='poisson' else len(domain))
        exec(text,scope)
        return np.array(scope['out']),h.copy(),calls
    original=execute(code);actual=execute(lifted)
    for left,right in zip(original[:2],actual[:2]):np.testing.assert_allclose(left,right)
    if kind not in ('normal','uniform'):
        assert len(original[2])==len(actual[2])==2
        for (left,lk),(right,rk) in zip(original[2],actual[2]):
            assert lk==rk
            for a,c in zip(left,right):np.testing.assert_array_equal(a,c)


@pytest.mark.parametrize('kind',['normal','uniform','randn','rand','poisson'])
@pytest.mark.parametrize('spelling',['module','function'])
def test_alias_rng_replacement_rejected(kind,spelling,monkeypatch):
    monkeypatch.setattr(np.random,kind,lambda *args,**kwargs:0.)
    with pytest.raises(ValueError,match='custom callback random'):
        lower_state_effect_function(getattr(aliases,spelling+'_'+kind))


@pytest.mark.parametrize('name',['shadowed_function','shadowed_module','shadowed_numpy'])
def test_local_shadow_not_authenticated(name):
    descriptor=lower_state_effect_function(getattr(aliases,name))
    assert not descriptor.random_sites


def test_closure_rng_identity():
    assert lower_state_effect_function(aliases.closure_normal())==lower_state_effect_function(qualified.scalar_normal)


def test_rebound_import_not_authenticated(monkeypatch):
    monkeypatch.setattr(aliases,'imported_normal',lambda *args:0.)
    with pytest.raises(ValueError):lower_state_effect_function(aliases.function_normal)


@pytest.mark.parametrize('name',['keyword_poisson','keyword_none_poisson','positional_none_poisson'])
def test_poisson_rate_keywords(name):
    assert lower_state_effect_function(getattr(aliases,name))==lower_state_effect_function(aliases.qualified_poisson)


@pytest.mark.parametrize('name,reason',[('sized_poisson','explicit size'),('duplicate_poisson','invalid callback Poisson keyword')])
def test_poisson_invalid_size_and_duplicate(name,reason):
    function=getattr(aliases,name)
    if name=='duplicate_poisson':
        with pytest.raises(ValueError,match=reason):lower_state_effect_function(function)
        return
    # Explicit size=len(x) is supported. Keep this historical case identity,
    # but verify its NumPy behavior instead of requiring a removed restriction.
    descriptor=lower_state_effect_function(function)
    assert descriptor.random_argument_orders[0][1]==('lam','size')
    code='out=f(h)+f(h)+h'
    lifted,helpers,sites,_=lift_callback_noise(code,{'f':descriptor})
    assert not helpers and len(sites)==2
    saved=np.random.get_state()
    try:
        for values in ([],[1.1],[1.1,1.9,2.7]):
            results=[]
            for text in (code,lifted):
                h=np.array(values,dtype=float)
                scope=dict(h=h,f=function,
                           _b2_size_rate=lambda rate,domain:np.broadcast_to(rate,(len(domain),)))
                scope.update({site:np.random.poisson for site in sites})
                np.random.seed(137)
                exec(text,scope)
                results.append((scope['out'].copy(),h.copy(),np.random.get_state()))
            np.testing.assert_array_equal(results[0][0],results[1][0])
            np.testing.assert_array_equal(results[0][1],results[1][1])
            assert results[0][2][0]==results[1][2][0]
            np.testing.assert_array_equal(results[0][2][1],results[1][2][1])
            assert results[0][2][2:]==results[1][2][2:]
    finally:
        np.random.set_state(saved)
