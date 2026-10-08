"""Generic delayed pre-state write and eventless summed-Synapses contract."""

import os
import sys
from pathlib import Path

import brian2 as b
import numpy as np
import pytest
from brian2.devices.device import all_devices

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import brian2_rust  # noqa: F401, E402


def _run(device_name, runner=None, engine="reference"):
    b.start_scope()
    if device_name == "rust_standalone":
        all_devices[device_name].reinit()
        b.set_device(device_name, runner=runner, engine=engine)
    else:
        b.set_device("runtime")
        b.prefs.codegen.target = "numpy"
    b.defaultclock.dt = 0.1*b.ms
    source = b.NeuronGroup(
        2, "dv/dt = 0 / second : 1\n"
           "dx/dt = -x / (1*ms) : 1\n"
           "label : integer (constant)",
        threshold="v > 0", reset="v = 0", method="euler")
    source.v = [1, 0]
    source.label = [0, 0]
    auxiliary = b.NeuronGroup(1, "s : 1")
    target = b.NeuronGroup(2, "z : 1")
    pathway = b.Synapses(source, source, on_pre="x_pre += 1",
                         delay=0.5*b.ms)
    pathway.connect(j="i")
    aggregate = b.Synapses(
        source, auxiliary, "s_post = x_pre : 1 (summed)")
    aggregate.connect(j="label_pre")
    broadcast = b.Synapses(
        auxiliary, target, "z_post = s_pre : 1 (summed)")
    broadcast.connect()
    source_trace = b.StateMonitor(source, "x", record=True)
    aggregate_trace = b.StateMonitor(auxiliary, "s", record=True)
    target_trace = b.StateMonitor(target, "z", record=True)
    network = b.Network(source, auxiliary, target, pathway, aggregate,
                        broadcast, source_trace, aggregate_trace,
                        target_trace)
    network.run(3*b.ms)
    return (np.asarray(source_trace.x).copy(),
            np.asarray(aggregate_trace.s).copy(),
            np.asarray(target_trace.z).copy())


@pytest.mark.parametrize("engine", ["reference", "aot"])
def test_delayed_pre_write_and_eventless_summed_match_brian(engine):
    previous = b.get_device()
    previous_target = b.prefs.codegen.target
    runner = Path(os.environ["B2_RUNNER"])
    try:
        reference = _run("runtime")
        candidate = _run("rust_standalone", runner, engine=engine)
        for original, rust in zip(reference, candidate):
            np.testing.assert_allclose(
                original, rust, rtol=0, atol=2e-12)
        assert float(reference[0][0].max()) > 0.5
        assert float(reference[1][0].max()) > 0.5
        assert float(reference[2][0].max()) > 0.5
    finally:
        b.set_device(previous)
        b.prefs.codegen.target = previous_target
        all_devices["rust_standalone"].reinit()
        b.start_scope()
