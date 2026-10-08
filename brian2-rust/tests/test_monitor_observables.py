"""Generic StateMonitor subexpression and refractory-field regression."""

import os
from pathlib import Path
import sys

import brian2 as b
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import brian2_rust  # noqa: E402


def _run(backend, directory, runner):
    b.start_scope()
    if backend == "rust":
        b.set_device("rust_standalone", engine="reference",
                     directory=directory, runner=runner)
    else:
        b.set_device("runtime")
        b.prefs.codegen.target = "numpy"
    stimulus = b.TimedArray(
        [[0.25, 1.25], [0.5, 1.5], [0.75, 1.75], [1.0, 2.0]],
        dt=5*b.ms)
    group = b.NeuronGroup(
        2,
        """dv/dt = (drive-v)/tau : 1 (unless refractory)
           I = 2*v + stimulus(t, i) : 1
           clock_value = t/(1*ms) : 1
           selected = int(i < 1) : integer
           drive : 1 (constant)""",
        threshold="v > 0.5", reset="v = 0", refractory=1 * b.ms,
        method="euler", namespace={"tau": 1 * b.ms, "stimulus": stimulus},
    )
    group.drive = [1.0, 1.2]
    state = b.StateMonitor(group, True, record=[1, 0])
    rate = b.PopulationRateMonitor(group)
    b.Network(group, state, rate).run(20 * b.ms)
    output = {
        name: np.asarray(state.variables[name].get_value()).copy()
        for name in state.record_variables
    }
    output["t"] = np.asarray(state.t).copy()
    output["rate"] = np.asarray(rate.rate).copy()
    if backend == "rust":
        b.device.reinit()
    b.set_device("runtime")
    return output


def test_derived_and_refractory_observables_match_numpy(tmp_path):
    runner = Path(os.environ.get(
        "B2_RUNNER", str(ROOT / "target" / "release" / "b2-runner")))
    assert runner.exists()
    rust = _run("rust", tmp_path / "observables", runner)
    numpy = _run("numpy", None, None)
    assert rust.keys() == numpy.keys()
    for name in rust:
        np.testing.assert_array_equal(rust[name], numpy[name], err_msg=name)


def test_queued_magicnetwork_backfills_monitors(tmp_path):
    runner = Path(os.environ.get(
        "B2_RUNNER", str(ROOT / "target" / "release" / "b2-runner")))
    b.start_scope()
    b.set_device("rust_standalone", engine="reference",
                 runner=runner, build_on_run=False)
    group = b.NeuronGroup(
        2, "dv/dt = (1-v)/(1*ms) : 1\nI = 2*v : 1",
        threshold="v > 0.5", reset="v = 0", method="euler")
    state = b.StateMonitor(group, ["v", "I"], record=[0])
    rate = b.PopulationRateMonitor(group)
    b.run(10 * b.ms)
    b.device.build(directory=tmp_path / "queued")
    assert len(rate.rate) == 100
    assert np.asarray(state.I).shape == (1, 100)
    b.device.reinit()
    b.set_device("runtime")
