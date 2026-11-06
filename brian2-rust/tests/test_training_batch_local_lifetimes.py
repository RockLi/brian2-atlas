"""Alias planning is symbolic and preserves NumPy object identity."""
import brian2 as b
import pytest
from brian2_rust.training_batch_locals import batch_local_lifetimes
from brian2_rust.training_effects import lower_state_effect_function
from test_training_capture_callbacks import add_constant_selected


def plan(code,mode):
    b.start_scope();g=b.NeuronGroup(2,'h:1\nv:1')
    fn=b.Function(add_constant_selected(g.variables['v'].get_value()),arg_units=[1],arg_names=['x'],return_unit=1,stateless=False)
    return batch_local_lifetimes(code,g.variables,{'f':lower_state_effect_function(fn)},mode=mode)


@pytest.mark.parametrize('mode',['array','vectorised'])
def test_array_borrow_and_rebind_are_distinct(mode):
    p=plan('tmp=f(h);borrow=tmp;tmp*=.8;tmp=tmp+.01;h=tmp+borrow',mode)
    s=p['stages']
    assert s[0]['outputs']['tmp']==s[1]['outputs']['borrow']==s[2]['outputs']['tmp']
    assert s[3]['outputs']['tmp']!=s[2]['outputs']['tmp']
    assert s[4]['inputs']['borrow']==s[2]['outputs']['tmp']
    assert s[3]['inputs']['tmp']==s[2]['outputs']['tmp']
    assert s[4]['inputs']['tmp']==s[3]['outputs']['tmp']
    if mode=='array':assert s[0]['outputs']['tmp']==p['initial']['h']
    else:assert s[0]['inputs']['h']!=p['initial']['h']


def test_scalar_augmented_assignment_rebinds():
    p=plan('tmp=.03;borrow=tmp;tmp*=.8', 'array');s=p['stages']
    assert not p['tokens'][s[0]['outputs']['tmp']]['array']
    assert s[0]['outputs']['tmp']==s[1]['outputs']['borrow']
    assert s[2]['outputs']['tmp']!=s[1]['outputs']['borrow']


def test_numpy_wrapper_zero_dim_array_keeps_borrow():
    p=plan('tmp=f(.03);borrow=tmp;tmp*=.8','array');s=p['stages']
    token=s[0]['outputs']['tmp']
    assert p['tokens'][token]['zero_dim']
    assert token==s[1]['outputs']['borrow']==s[2]['outputs']['tmp']


def test_array_model_assignment_keeps_old_borrow():
    p=plan('tmp=h;h=h+.1;v=tmp', 'array');s=p['stages']
    assert s[0]['outputs']['tmp']==p['initial']['h']
    assert s[1]['inputs']['h']==p['initial']['h']
    assert s[1]['outputs']['h']!=p['initial']['h']
    assert s[2]['inputs']['tmp']==p['initial']['h']


def test_invalid_statement_is_rejected():
    with pytest.raises(ValueError,match='simple assignments'):plan('print(h)','array')
