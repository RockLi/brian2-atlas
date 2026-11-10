"""Ordered array calls retain borrowed objects and independent stream sites."""
import ast
import numpy as np
import pytest
from brian2.core.functions import DEFAULT_FUNCTIONS
from brian2_rust.training_batch_poisson import stage_batch_poisson
from brian2_rust.training_batch_random import cached_batch_draws


@pytest.mark.parametrize('expression',[
    'f(x)+poisson(x)', 'f(x)+poisson(f(x))',
    'minimum(f(x),poisson(f(x)))', 'poisson(x)+poisson(x)',
])
def test_staged_calls_preserve_python_argument_order(expression):
    original='out='+expression;groups=[];staged=stage_batch_poisson(original,statement_groups=groups)
    def execute(code):
        x=np.array([1.1,1.9]);calls=[]
        def f(value):
            calls.append(('f',value.copy()));value*=.7;return value
        def poisson(rate):
            calls.append(('poisson',rate.copy()));return np.array([1,2],dtype=np.int32)
        scope=dict(x=x,f=f,poisson=poisson,minimum=np.minimum);exec(code,scope)
        return scope['out'],x,calls
    a,b=execute(original),execute(staged)
    np.testing.assert_array_equal(a[0],b[0]);np.testing.assert_array_equal(a[1],b[1])
    assert [name for name,_ in a[2]]==[name for name,_ in b[2]]
    for (_,left),(_,right) in zip(a[2],b[2]):np.testing.assert_array_equal(left,right)
    assert groups==[0]*len(ast.parse(staged).body)


def test_mixed_draw_sites_keep_expression_order():
    sites={};code=stage_batch_poisson('h=poisson(rate)+randn()+poisson(rate)+rand()')
    rewritten,draws=cached_batch_draws(code,DEFAULT_FUNCTIONS,poisson_sites=sites)
    assert [site['stream'] for site in sites.values()]==[0,2]
    assert [(site['kind'],site['stream']) for site in draws]==[('randn',1),('rand',3)]
    assert 'poisson(' not in rewritten


@pytest.mark.parametrize('code,variables,message',[
    ('h=poisson()',DEFAULT_FUNCTIONS,'invalid random'),
    ('h=poisson(rate)',{'poisson':object()},'custom random'),
    ('h=_b2_batch_poisson_0+poisson(rate)',DEFAULT_FUNCTIONS,'reserved'),
    ('h='+'+'.join(['poisson(rate)']*17),DEFAULT_FUNCTIONS,'16 random'),
])
def test_poisson_site_validation(code,variables,message):
    with pytest.raises(ValueError,match=message):cached_batch_draws(code,variables,poisson_sites={})


def test_poisson_temporary_names_are_reserved():
    with pytest.raises(ValueError,match='reserved'):stage_batch_poisson('h=poisson(_b2poisson_local_0)')
