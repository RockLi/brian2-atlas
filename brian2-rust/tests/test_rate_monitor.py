"""Generic PopulationRateMonitor semantics, independent of the NMDA fixture."""

from pathlib import Path
import os
import sys

import numpy as np
import brian2 as b

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
    group = b.NeuronGroup(
        3, "dv/dt = (drive-v)/tau : 1", threshold="v > 0.5",
        reset="v = 0", method="euler",
        namespace={"drive": 1.0, "tau": 1 * b.ms},
    )
    group.v = [0, 0.2, 0.4]
    rate = b.PopulationRateMonitor(group)
    spikes = b.SpikeMonitor(group)
    b.Network(group, rate, spikes).run(20 * b.ms)
    output = (np.asarray(rate.t).copy(), np.asarray(rate.rate).copy(),
              np.asarray(spikes.i).copy(), np.asarray(spikes.t).copy())
    if backend == "rust":
        b.device.reinit()
    b.set_device("runtime")
    return output


def test_population_rate_monitor_matches_numpy(tmp_path):
    runner = Path(os.environ.get("B2_RUNNER", str(ROOT / "target" / "release" / "b2-runner")))
    assert runner.exists()
    rust = _run("rust", tmp_path / "rate", runner)
    numpy = _run("numpy", None, None)
    for left, right in zip(rust, numpy):
        np.testing.assert_array_equal(left, right)


def _run_subgroup_monitors(backend, directory, runner):
    b.start_scope()
    if backend == "rust":
        b.set_device("rust_standalone", engine="reference",
                     directory=directory, runner=runner)
    else:
        b.set_device("runtime")
        b.prefs.codegen.target = "numpy"
    group = b.NeuronGroup(
        6, "dv/dt = drive/ms : 1\ndrive : 1 (constant)",
        threshold="v >= 1", reset="v = 0", method="euler", dt=1*b.ms)
    group.drive = [0.2, 0.3, 0.5, 0.7, 0.9, 1.1]
    left, right = group[1:4], group[4:6]
    left_spikes = b.SpikeMonitor(left, variables="v")
    right_spikes = b.SpikeMonitor(right)
    left_rate = b.PopulationRateMonitor(left)
    right_rate = b.PopulationRateMonitor(right)
    b.Network(group, left_spikes, right_spikes,
              left_rate, right_rate).run(6*b.ms)
    output = {
        "left_i": np.asarray(left_spikes.i).copy(),
        "left_t": np.asarray(left_spikes.t).copy(),
        "left_v": np.asarray(left_spikes.v).copy(),
        "right_i": np.asarray(right_spikes.i).copy(),
        "right_t": np.asarray(right_spikes.t).copy(),
        "left_rate": np.asarray(left_rate.rate).copy(),
        "right_rate": np.asarray(right_rate.rate).copy(),
    }
    if backend == "rust":
        b.device.reinit()
    b.set_device("runtime")
    return output


def test_subgroup_spike_and_rate_monitors_match_numpy(tmp_path):
    runner = Path(os.environ.get(
        "B2_RUNNER", str(ROOT / "target" / "release" / "b2-runner")))
    rust = _run_subgroup_monitors("rust", tmp_path / "subgroup-rate", runner)
    numpy = _run_subgroup_monitors("numpy", None, None)
    for name, expected in numpy.items():
        np.testing.assert_array_equal(rust[name], expected, err_msg=name)
