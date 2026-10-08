"""Real Metal execution tests are opt-in because desktop sandboxes hide GPUs."""
import copy
import json
import os
from pathlib import Path
import sys

import brian2 as b
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import brian2_rust
from brian2_rust.metal import (build_metal_plan, MetalExecutor, write_metal_results,
                              METAL_PROFILE)
from brian2_rust.plan import PlanValidationError, build_execution_plan
from brian2_rust.protocol import attach_protocol
from brian2_rust.results import load_results


@pytest.fixture
def golden():
    return json.loads((ROOT / "tests/golden/b2ir-v1/minimal-v1.json").read_text())


def test_metal_requires_explicit_precision_and_validated_model(golden):
    with pytest.raises(PlanValidationError, match="explicit"):
        build_execution_plan(golden, backend="metal")
    plan = build_execution_plan(golden, backend="metal", numeric_mode="float32")
    assert plan.numeric_profile == METAL_PROFILE
    assert "kernel void population_0" in plan.kernels[0].source
    assert "float s0" in plan.kernels[0].source
    bad = copy.deepcopy(golden)
    bad["definition"]["schedule"]["nodes"] = []
    attach_protocol(bad)
    with pytest.raises(PlanValidationError):
        build_metal_plan(bad, numeric_mode="float32")


real_metal = pytest.mark.skipif(os.environ.get("B2_TEST_METAL") != "1",
                              reason="set B2_TEST_METAL=1 with access to an Apple GPU")


@real_metal
def test_metal_golden_matches_reference_transport(golden, tmp_path):
    import subprocess
    reference = tmp_path / "reference"
    subprocess.run([str(ROOT / "target/release/b2-runner"),
                    str(ROOT / "tests/golden/b2ir-v1/minimal-v1.json"), str(reference)], check=True)
    expected = load_results(golden, reference)
    with MetalExecutor(golden, tmp_path / "metal", numeric_mode="float32") as executor:
        result = executor.run()
        repeated = executor.run()
        np.testing.assert_array_equal(result["populations"][0]["states"]["v"],
                                      repeated["populations"][0]["states"]["v"])
        assert result["device"].startswith("Apple")
        assert result["timings"][0]["gpu_seconds"] > 0
        write_metal_results(golden, result, tmp_path / "result")
    actual = load_results(golden, tmp_path / "result")
    assert actual["metadata"]["numeric_profile"] == METAL_PROFILE
    for metal, cpu in zip(actual["populations"], expected["populations"], strict=True):
        np.testing.assert_allclose(metal["states"]["v"], cpu["states"]["v"], rtol=2e-6)
        np.testing.assert_allclose(metal["trace"]["v"], cpu["trace"]["v"], rtol=2e-6)
        for field in ("spike_ticks", "indices", "counts", "last_spikes"):
            np.testing.assert_array_equal(metal[field], cpu[field])
        for field in ("lastspike", "not_refractory"):
            np.testing.assert_array_equal(metal["refractory"][field], cpu["refractory"][field])


@real_metal
def test_metal_device_multistate_segmented_and_monitors(tmp_path):
    previous = b.get_device()
    from brian2.devices.device import all_devices
    device = all_devices["rust_standalone"]
    snapshots = []
    try:
        for engine in ("aot", "metal"):
            device.reinit()
            options = {"numeric_mode": "float32"} if engine == "metal" else {}
            b.set_device("rust_standalone", engine=engine, directory=tmp_path / engine,
                         runner=ROOT / "target/release/b2-runner", **options)
            group = b.NeuronGroup(32, "dv/dt=(drive-v)/ms:1\ndx/dt=(v-x)/(2*ms):1\ndrive:1 (constant)",
                                  threshold="v>0.9", reset="v=0; x+=0.1", refractory=0.2*b.ms,
                                  dt=0.1*b.ms, method="euler")
            group.drive = np.linspace(1.1, 1.4, 32)
            monitor = b.StateMonitor(group, ["v", "x"], record=[0, 17, 31])
            spikes = b.SpikeMonitor(group)
            network = b.Network(group, monitor, spikes)
            network.run(1*b.ms)
            network.store("middle")
            network.run(1*b.ms)
            expected_replay = np.asarray(group.v[:]).copy()
            network.restore("middle")
            network.run(1*b.ms)
            np.testing.assert_array_equal(group.v[:], expected_replay)
            snapshots.append({"v": np.asarray(group.v[:]).copy(), "x": np.asarray(group.x[:]).copy(),
                              "trace_v": np.asarray(monitor.v).copy(), "trace_x": np.asarray(monitor.x).copy(),
                              "spike_i": np.asarray(spikes.i[:]).copy(), "spike_t": np.asarray(spikes.t[:]).copy()})
            if engine == "metal":
                assert "float32" in device.explain_plan()
        for name in snapshots[0]:
            if name.startswith("spike"):
                np.testing.assert_array_equal(snapshots[1][name], snapshots[0][name])
            else:
                np.testing.assert_allclose(snapshots[1][name], snapshots[0][name], rtol=3e-6, atol=1e-7)
    finally:
        device.reinit()
        b.set_device(previous)


@real_metal
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_float_storage_and_math_grid(dtype, tmp_path):
    """Independent Rust f64 interpreter oracle for the shared stable f32 math helpers."""
    previous = b.get_device()
    from brian2.devices.device import all_devices
    device = all_devices["rust_standalone"]
    values = np.concatenate((np.linspace(-80, 80, 513),
                             [-103, -100, -90, -1e-4, -1e-8, 0, 1e-8, 1e-4, 88]))
    try:
        device.reinit()
        b.set_device("rust_standalone", engine="reference", directory=tmp_path/"reference",
                     runner=ROOT/"target/release/b2-runner")
        group = b.NeuronGroup(len(values), "x:1 (constant)\ny:1\nz:1\nw:1", dtype=dtype)
        group.x = values
        group.run_regularly("y=exp(x); z=exprel(x); w=x**3", when="start")
        net = b.Network(group)
        net.run(.1*b.ms)
        model = json.loads((tmp_path/"reference/model.json").read_text())
        with MetalExecutor(model, tmp_path/"metal", numeric_mode="float32") as executor:
            actual = executor.run()["populations"][0]["states"]
            mirror = executor.run(compute="cpu-f32", workers=2)["populations"][0]["states"]
        for name in ("y", "z", "w"):
            np.testing.assert_allclose(actual[name], np.asarray(getattr(group, name)[:]), rtol=3e-6, atol=1e-7)
            np.testing.assert_array_equal(actual[name], mirror[name])
    finally:
        device.reinit()
        b.set_device(previous)


def test_refractory_checkpoint_tick_uses_reference_boundary():
    from brian2_rust.metal import _first_available_ticks
    dt = .0001
    last = np.asarray([-10000, 0, .0000999, np.nextafter(.0000999, 0),
                       np.nextafter(.0000999, 1), 123.4567])
    for period in (0, 1, 2, 100):
        ticks = _first_available_ticks(last, dt, period)
        assert np.all(((ticks*dt-last)+1e-3*dt)/dt >= period)
        positive = ticks > 0
        assert np.all((((ticks[positive]-1)*dt-last[positive])+1e-3*dt)/dt < period)
