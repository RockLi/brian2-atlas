"""Semantic failures are rejected before executing callback Python bodies."""
import pytest
from brian2_rust.training_effects import StateEffectFunction, compile_state_effect_transform


def compile_effect(body, code='v=curve(v)', *, writable=('v',)):
    function=StateEffectFunction(('x',),body)
    return compile_state_effect_transform(code,states={'v':0},parameters={'curve':function},
                                          array_states={'v'},writable_states=set(writable))


@pytest.mark.parametrize('body',[
    'sin=x\nx*=.8\nreturn sin(x)',
    'x*=.8\ny=sin(x)\nsin=x\nreturn y',
    'curve=x\nx*=.8\nreturn curve(x)',
])
def test_lexically_shadowed_call_refused(body):
    with pytest.raises(ValueError,match='shadowed'):
        compile_effect(body)


def test_expression_expansion_is_bounded_before_copying():
    with pytest.raises(ValueError,match='4096 AST nodes'):
        compile_effect('x*=.8\n'+'\n'.join(['x=x+x']*25)+'\nreturn x')


def test_eager_trace_budget():
    with pytest.raises(ValueError,match='64 eager operations'):
        compile_effect('x*=.8\nreturn x','\n'.join(['v=curve(v)']*65))


def test_readonly_borrowed_storage_refused():
    with pytest.raises(ValueError,match='readonly'):
        compile_effect('x*=.8\nreturn x',writable=())


def test_unsupported_augmented_operation_refused():
    with pytest.raises(ValueError,match='augmented operation'):
        compile_effect('x**=2\nreturn x')


def test_raw_python_callback_is_not_a_descriptor():
    def must_not_run(x):
        raise AssertionError('callback Python must not execute')
    with pytest.raises(ValueError):
        compile_state_effect_transform('v=curve(v)',states={'v':0},parameters={'curve':must_not_run},
                                       array_states={'v'},writable_states={'v'})


def test_nested_descriptor_cannot_recurse():
    with pytest.raises(ValueError,match='nested'):
        compile_effect('x*=.8\nreturn curve(x)')
