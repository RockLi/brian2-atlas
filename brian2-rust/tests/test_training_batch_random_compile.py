import pytest
from brian2.core.functions import DEFAULT_FUNCTIONS
from brian2_rust.training_batch_random import cached_batch_draws


def test_batch_draw_order_keeps_repeated_calls_independent():
    code,draws=cached_batch_draws('h=randn()+rand()\nv_post+=f(h+randn())',DEFAULT_FUNCTIONS)
    assert [(d['kind'],d['stream']) for d in draws]==[('randn',0),('rand',1),('randn',2)]
    assert code=='h = _b2_batch_random_0 + _b2_batch_random_1\nv_post += f(h + _b2_batch_random_2)'


@pytest.mark.parametrize('code,variables,message',[
 ('h=rand(2)',DEFAULT_FUNCTIONS,'invalid random'),
 ('h=rand()',{'rand':object()},'custom random'),
 ('h=_b2_batch_random_0+rand()',DEFAULT_FUNCTIONS,'reserved'),
 ('h='+'+'.join(['randn()']*17),DEFAULT_FUNCTIONS,'16 random'),
])
def test_batch_draw_validation(code,variables,message):
    with pytest.raises(ValueError,match=message):cached_batch_draws(code,variables)
