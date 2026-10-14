"""Fail closed on incomplete multistate continuation/verification evidence."""
import copy
import importlib.util
from pathlib import Path
import sys

import pytest

TOOLS=Path(__file__).resolve().parents[1]/'tools'
sys.path.insert(0,str(TOOLS))
try:
    spec=importlib.util.spec_from_file_location('native_crosshost_evidence',TOOLS/'verify_native_training_crosshost.py')
    verifier=importlib.util.module_from_spec(spec);spec.loader.exec_module(verifier)
finally:sys.path.remove(str(TOOLS))


def result():
    return dict(state=dict(step=2,weights=[[.4]],first_moment=[[.02]],second_moment=[[.003]]),
                loss=.5,spikes=[[[0.,1.]]],gradients=[[.2]],initial_gradients=[[.1,.2]],
                final_membrane=[[.3,.4]],logits=[[.5,.6]],
                final_state=[[.3,.7,2.,.4,.8,1.]],initial_state_gradients=[[.1,.9,0.,.2,.6,0.]])


def test_v4_continuation_retains_auxiliary_and_counter_without_aliasing():
    original=dict(plan=dict(state_equations=[[],[]]),initial=[[0.,0.]])
    before=copy.deepcopy(original);observed=result();snapshot=copy.deepcopy(observed)
    request=verifier.continuation_request(original,observed,4)
    assert request['initial']==[[.3,.7,2.,.4,.8,1.]] and request['plan']['mpi_ranks']==4
    request['initial'][0][1]=-1;request['state']['weights'][0][0]=-1
    assert original==before and observed==snapshot


def test_scalar_continuation_keeps_legacy_membrane():
    request=verifier.continuation_request(dict(plan={}),result())
    assert request['initial']==[[.3,.4]]


@pytest.mark.parametrize('field',['final_state','initial_state_gradients'])
def test_missing_v4_evidence_is_rejected(field):
    observed=result();observed.pop(field)
    with pytest.raises(AssertionError,match='missing'):
        verifier.compare_outputs(observed,observed,require_full_state=True)
    if field=='final_state':
        with pytest.raises(ValueError,match='final_state'):
            verifier.continuation_request(dict(plan=dict(state_equations=[[]])),observed)


@pytest.mark.parametrize('field',['final_state','initial_state_gradients'])
def test_auxiliary_only_difference_is_not_hidden_by_voltage(field):
    observed=result();observed[field][0][1]+=.1
    with pytest.raises(AssertionError):verifier.compare_outputs(result(),observed,True)


@pytest.mark.parametrize('value',[float('inf'),float('-inf'),float('nan')])
def test_identical_nonfinite_values_are_not_valid_evidence(value):
    observed=result();observed['loss']=value
    with pytest.raises(AssertionError):verifier.compare_outputs(observed,observed,True)


def test_identical_finite_evidence_and_exact_discrete_state():
    assert verifier.compare_outputs(result(),result(),True)==0
    observed=result();observed['state']['step']=2.0000000000001
    with pytest.raises(AssertionError,match='discrete'):
        verifier.compare_outputs(result(),observed,True)


def test_time_continuation_and_comparison_require_exact_tick():
    original=dict(plan=dict(clock=dict(origin=.001,dt=.0001),state_equations=[[]]),start_tick=4)
    observed=dict(result(),final_tick=7)
    assert verifier.continuation_request(original,observed)['start_tick']==7
    assert original['start_tick']==4
    for tick in [None,7.0,True,-1,2**53+1]:
        with pytest.raises(ValueError,match='final_tick'):
            verifier.continuation_request(original,dict(observed,final_tick=tick))
    for other in [result(),dict(observed,final_tick=8),dict(observed,final_tick=7.0)]:
        with pytest.raises(AssertionError):verifier.compare_outputs(observed,other)


def test_noise_continuation_requires_sequence():
    original=dict(plan=dict(state_equations=[[]],noise_streams=[1],clock=dict(origin=0,dt=.001)))
    observed=dict(result(),final_tick=7,noise_sequence=12)
    assert verifier.continuation_request(original,observed)['noise_sequence']==12
    for sequence in [None,True,-1,2**64-1,1.5]:
        with pytest.raises(ValueError,match='noise_sequence'):
            verifier.continuation_request(original,dict(observed,noise_sequence=sequence))
    for other in [dict(result(),final_tick=7),dict(observed,noise_sequence=13)]:
        with pytest.raises(AssertionError):verifier.compare_outputs(observed,other)
