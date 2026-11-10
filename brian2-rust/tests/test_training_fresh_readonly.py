"""Readonly calls must not commit initial optimizer/RNG state or checkpoints."""
import copy
import os

import numpy as np
import pytest

from brian2_rust import NativeLIFTrainer
from test_native_training import RUNNER
from test_training_dynamic import noisy_model
from test_training_stochastic import raw_noise_plan
from test_training_integer_ir import engine
from test_training_poisson_zero_vjp import mpi


def persistent(trainer):
    return copy.deepcopy({name: getattr(trainer, name) for name in (
        'plan', 'state', 'neuron_state', 'elapsed_ticks', 'clock_tick', 'clock_state',
        'poisson_state', 'noise_sequence', 'next_noise_sequence')})


@pytest.mark.parametrize('kind', ['sde', 'dynamic'])
@pytest.mark.parametrize('operation', ['evaluate', 'gradients'])
@pytest.mark.parametrize('explicit_weights', [False, True])
@pytest.mark.parametrize('ranks', [None, 2])
def test_fresh_readonly_preserves_checkpoint_and_next_train(
    engine, kind, operation, explicit_weights, ranks, tmp_path,
):
    mpi(ranks)
    if kind == 'sde':
        plan = raw_noise_plan(engine, ranks)
        weights = [[0.0]]
        inputs = np.zeros((1, 3, 1))
    else:
        plan, weights, x = noisy_model(False, 2)
        plan.update(backend=engine)
        if ranks:
            plan['mpi_ranks'] = ranks
        inputs = x[None, :3]
    kwargs = dict(runner=RUNNER)
    if explicit_weights:
        kwargs['weights'] = weights
    trainer = NativeLIFTrainer(plan, **kwargs)
    control = NativeLIFTrainer(plan, **kwargs)
    before = persistent(trainer)
    checkpoint = tmp_path/'before.json'
    trainer.store(checkpoint)
    saved = checkpoint.read_bytes()
    # Repeated default reads use the same uncommitted sequence and initial weights.
    first = getattr(trainer, operation)(inputs, [0])
    second = getattr(trainer, operation)(inputs, [0])
    assert second == first
    assert persistent(trainer) == before
    trainer.store(checkpoint)
    assert checkpoint.read_bytes() == saved
    restored = NativeLIFTrainer(plan, runner=RUNNER)
    restored.restore(checkpoint)
    assert persistent(restored) == before
    expected = control.step(inputs, [0])
    assert trainer.step(inputs, [0]) == expected
    assert restored.step(inputs, [0]) == expected
    assert persistent(trainer) == persistent(control) == persistent(restored)
    assert trainer.state['step'] == 1
    assert trainer.noise_sequence == 0 and trainer.next_noise_sequence == 1
