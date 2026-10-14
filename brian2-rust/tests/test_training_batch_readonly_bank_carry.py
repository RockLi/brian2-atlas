"""Captured readonly banks remain live after optimizer carry and restore."""
import copy
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer
from test_training_batch_readonly_captures import model
from test_training_integer_ir import engine
from test_training_poisson_zero_vjp import mpi
from test_native_training import RUNNER


@pytest.mark.parametrize('dtype',['float','integer','boolean'])
@pytest.mark.parametrize('ranks',[None,2])
def test_readonly_bank_replacement_after_carry_is_not_cached(engine,dtype,ranks,tmp_path):
    mpi(ranks);_,_,_,_,bundle,x,bank=model(True,0,ranks,engine,dtype=dtype)
    t=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER);t.step(x[:,:2],[0])
    if dtype=='float':assert not np.array_equal(t.state['weights'][bank],bundle.weights[bank])
    t.state['weights'][bank][1]=0.
    t.evaluate(np.zeros((1,1,2)),[0],initial='carry')
    path=tmp_path/'live-bank';t.store(path);t=NativeLIFTrainer(t.plan,runner=RUNNER);t.restore(path)
    before=copy.deepcopy(dict(state=t.state,neuron=t.neuron_state,clock=t.clock_state,tick=t.clock_tick,elapsed=t.elapsed_ticks,plan=t.plan))
    with pytest.raises(ValueError,match='[Nn]on.?finite|domain'):t.step(np.array([[[1.,0.]]]),[0],initial='carry')
    assert dict(state=t.state,neuron=t.neuron_state,clock=t.clock_state,tick=t.clock_tick,elapsed=t.elapsed_ticks,plan=t.plan)==before
