"""Static early returns preserve original callback execution and native VJPs."""
import copy
import brian2 as b
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer
from brian2_rust.training_functions import lower_pure_function
from brian2_rust.training_dynamic import compile_dynamic_transform
from brian2_rust.training_equations import NeuronParameter, NormalNoise
from test_native_training import RUNNER
from test_training_integer_ir import engine, model, oracle
from test_training_poisson_zero_vjp import mpi
import test_training_pure_functions as callbacks
from test_training_pure_statements import multiple_return


def conditional_return(x):
    if 1 < 2:
        return .5*x
    x[0] = 99.
    return np.sqrt(-1.)


def for_return(x):
    y = 0.*x
    for i in range(3):
        y += .2*x
        if i == 1:
            return y
    else:
        y = np.sqrt(-1.)
    return y


def while_return(x):
    y = 0.*x
    i = 0
    while i < 4:
        i += 1
        y += .2*x
        if i == 3:
            return y
    else:
        y = np.sqrt(-1.)
    return y


def inner_else_return(x):
    for i in range(3):
        for j in range(0):
            x[0] = 99.
        else:
            return .7*x
        x = np.sqrt(-1.)
    return np.sqrt(-1.)


def straight_return(x):
    return .3*x
    x[0] = 99.
    return np.sqrt(-1.)


@pytest.mark.parametrize('function',[conditional_return,for_return,while_return,inner_else_return,straight_return,multiple_return])
def test_early_return_original_brian_forward(engine,function,monkeypatch):
    monkeypatch.setattr(callbacks,'decay',b.check_units(x=1,result=1)(function))
    callbacks.test_dynamic_brian_callback_positions_and_forward(engine,'regular')


def noisy_return(voltage, gate, gain, draw):
    y = 0.*voltage
    i = 0
    while True:
        y += .2*voltage
        i += 1
        if i == 4:
            return y+gain*gate+.05*gate*draw
    return np.sqrt(-1.)


@pytest.mark.parametrize('ranks',[None,2,8])
def test_early_return_noisy_all_bank_and_physical_initial_vjps(engine,ranks):
    mpi(ranks);p,w,x=model(noisy=True,engine=engine,ranks=ranks)
    function=lower_pure_function(noisy_return)
    for action in p['dynamic']['actions']:
        if action.get('noise_streams')!=1:continue
        p['dynamic']['program_sets'][action['program_set']]=compile_dynamic_transform(
            'v=f(v,flag,gain,draw)',states={'v':0,'flag':1},
            parameters={'f':function,'gain':NeuronParameter(0),'draw':NormalNoise(0)})['programs']
    result=NativeLIFTrainer(p,weights=w,runner=RUNNER).gradients(x,[0],noise_sequence=7)
    loss,final,spikes,anchors=oracle(p,w,noisy=True)
    np.testing.assert_array_equal(result['spikes'][0],spikes)
    np.testing.assert_allclose(result['final_state'][0],final,rtol=2e-5,atol=3e-6)
    assert result['loss']==pytest.approx(loss,abs=3e-6)
    for i in range(4):
        plus=copy.deepcopy(w);minus=copy.deepcopy(w);plus[0][i]+=1e-6;minus[0][i]-=1e-6
        fd=(oracle(p,plus,noisy=True,anchors=anchors)[0]-oracle(p,minus,noisy=True,anchors=anchors)[0])/2e-6
        assert result['gradients'][0][i]==pytest.approx(fd,rel=3e-4,abs=3e-6)
        plus=np.asarray(p['dynamic']['initial']);minus=plus.copy();plus[i]+=1e-6;minus[i]-=1e-6
        fd=(oracle(p,w,noisy=True,anchors=anchors,initial=plus)[0]-oracle(p,w,noisy=True,anchors=anchors,initial=minus)[0])/2e-6
        assert result['initial_state_gradients'][0][i]==pytest.approx(fd,rel=3e-4,abs=3e-6)
    assert (result['gpu_dispatches']>0)==(engine!='cpu')


SHADOWED = .4

def unbound_return(x):
    return SHADOWED*x
    SHADOWED = .7


def bare_return(x):
    if True:
        return
    return x


@pytest.mark.parametrize('function',[unbound_return,bare_return])
def test_early_return_keeps_unbound_and_missing_value_refusals(function):
    with pytest.raises(ValueError):lower_pure_function(function)
