"""Cross-branch regression: LK synaptic runners on the explicit CPU schedule."""
from pathlib import Path
import sys
import os

import brian2 as b
from brian2.devices.device import all_devices
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
import brian2_rust  # noqa: E402,F401


@pytest.mark.parametrize('slot', ['groups', 'end'])
def test_canonical_schedule_preserves_periodic_synaptic_runner(tmp_path, slot):
    values = []
    for backend in ['numpy', 'reference', 'aot']:
        all_devices["rust_standalone"].reinit()
        b.start_scope()
        if backend == 'numpy':
            b.set_device('runtime')
            b.prefs.codegen.target = 'numpy'
        else:
            b.set_device('rust_standalone', engine=backend,
                         runner=ROOT / 'target/release/b2-runner',
                         directory=tmp_path / backend)
        neurons = b.NeuronGroup(3, 'v : 1\ntotal : 1', threshold='False',
                               reset='', dt=1*b.ms, name='merge_neurons')
        # An early population runner forces the new canonical-slot emitter.
        neurons.run_regularly('v += 1', when='before_groups', name='merge_early')
        synapses = b.Synapses(neurons, neurons,
                             'w : 1\ntotal_post = w : 1 (summed)',
                             on_pre='w += 0', clock=neurons.clock,
                             name='merge_synapses')
        synapses.connect(i=[0, 1, 2], j=[1, 1, 2])
        synapses.w = [1, 3, 2]
        regular = synapses.run_regularly('w += dt/ms + v_pre - total_post/4',
                                        dt=2*b.ms, when=slot, name='merge_periodic')
        synapses.summed_updaters['total_post']._clock = regular.clock
        b.Network(neurons, synapses).run(6*b.ms)
        values.append((synapses.w[:].copy(), neurons.total[:].copy()))
        if backend == 'aot':
            assert b.get_device().last_execution_plan.cpu.emitter == 'slot-v1'
    for actual in values[1:]:
        for result, expected in zip(actual, values[0]):
            np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize('backend', [
    pytest.param('metal', marks=pytest.mark.skipif(
        os.environ.get('B2_TEST_METAL') != '1', reason='Apple GPU opt-in')),
    pytest.param('cuda', marks=pytest.mark.skipif(
        os.environ.get('B2_TEST_CUDA') != '1', reason='CUDA opt-in')),
])
@pytest.mark.parametrize('slot', ['groups', 'end'])
def test_gpu_synaptic_runner_preserves_source_dt(tmp_path, backend, slot):
    values = []
    for engine in ['numpy', 'reference', backend]:
        all_devices["rust_standalone"].reinit()
        b.start_scope()
        if engine == 'numpy':
            b.set_device('runtime')
            b.prefs.codegen.target = 'numpy'
        else:
            b.set_device('rust_standalone', engine=engine,
                         runner=ROOT/'target/release/b2-runner', directory=tmp_path/engine,
                         **({'numeric_mode': 'float32'} if engine != 'reference' else {}))
        neurons = b.NeuronGroup(3, 'v : 1\nseen_t : 1\nseen_step : 1', threshold='False', reset='',
                               dt=b.second/1024, name='merge_gpu_neurons')
        neurons.run_regularly(
            'v += t*1024/second; seen_t=t*1024/second; seen_step=dt*1024/second',
            dt=b.second/2048, when='before_thresholds', name='merge_gpu_population_runner')
        synapses = b.Synapses(neurons, neurons, 'w : 1\nseen_dt : 1',
                             on_pre='w += 0', clock=neurons.clock,
                             name='merge_gpu_synapses')
        synapses.connect(i=[0, 1, 2], j=[1, 1, 2])
        synapses.w = [1, 3, 2]
        synapses.run_regularly('w += dt*1024/second; seen_dt = dt*1024/second',
                               dt=b.second/512, when=slot, name='merge_gpu_periodic')
        b.Network(neurons, synapses).run(6*b.second/1024)
        values.append(tuple(np.asarray(value).copy() for value in (
            synapses.w[:], synapses.seen_dt[:], neurons.v[:],
            neurons.seen_t[:], neurons.seen_step[:])))
    for output in values[1:]:
        for actual, expected in zip(output, values[0]):
            np.testing.assert_array_equal(actual, expected)


def test_uncertified_owner_clock_grid_fails_before_gpu_emission():
    from math import sqrt
    from types import SimpleNamespace
    from brian2_rust.gpu_schedule import owner_tick_expression
    from brian2_rust.plan import ClockActivation, PlanValidationError
    from brian2_rust.spec import bits

    logical = SimpleNamespace(clocks=(
        ClockActivation(0, bits(1.0), 0, 10),
        ClockActivation(1, bits(sqrt(2)), 0, 10),
    ))
    with pytest.raises(PlanValidationError, match='cannot be certified'):
        owner_tick_expression(logical, 0, 1)
