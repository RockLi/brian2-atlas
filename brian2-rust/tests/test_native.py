"""Specialized native execution, replay validation, and fail-closed builds."""
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import brian2 as b
import numpy as np
from brian2.devices.device import all_devices

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import brian2_rust  # noqa: E402, F401
from brian2_rust.native import (  # noqa: E402
    _schedule_can_contract,
    _schedule_can_fuse_bundles,
)


class NativeTest(unittest.TestCase):
    def test_single_population_named_events_keep_complete_event_protocol(self):
        for dtype in (np.float64, np.float32):
            for record in (True, False):
                snapshots = []
                for engine in ("reference", "aot", "numpy"):
                    with self.subTest(dtype=dtype, record=record, engine=engine):
                        self.select(engine)
                        group = b.NeuronGroup(
                            1, "dv/dt=1/ms:1", events={"crossing": "v>=2"},
                            method="euler", dt=1*b.ms, dtype=dtype)
                        group.run_on_event("crossing", "v=0")
                        monitor = b.EventMonitor(group, "crossing") if record else None
                        network = b.Network(group, *([monitor] if record else []))
                        network.run(3*b.ms)
                        snapshots.append(np.asarray(group.v[:]).copy())
                        if record:
                            np.testing.assert_array_equal(monitor.i[:], [0])
                            np.testing.assert_allclose(monitor.t[:]/b.ms, [1], rtol=0, atol=1e-7)
                np.testing.assert_allclose(snapshots, [[1], [1], [1]], rtol=0, atol=1e-7)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.previous = b.get_device()
        self.target = b.prefs.codegen.target
        self.device = all_devices["rust_standalone"]
        self.runner = ROOT / "target/release/b2-runner"

    def tearDown(self):
        b.set_device(self.previous)
        b.prefs.codegen.target = self.target
        self.device.reinit()

    def select(self, engine):
        if engine == "numpy":
            b.set_device("runtime")
            b.prefs.codegen.target = "numpy"
        else:
            self.device.reinit()
            b.set_device("rust_standalone", runner=self.runner, engine=engine)

    def test_aot_fusion_requires_schedule_effect_proof(self):
        def node(identifier, reads=(), writes=(), clock=0):
            return {
                "id": identifier,
                "clock": clock,
                "effects": {"reads": list(reads), "writes": list(writes)},
            }

        first_state = node("first-state", writes=("population/0/state/v",))
        second_state = node("second-state", writes=("population/1/state/v",))
        blocker = node("blocker", writes=("population/0/state/v",))
        first_threshold = node(
            "first-threshold", reads=("population/0/state/v",),
            writes=("population/0/event/spike",))
        second_threshold = node(
            "second-threshold", reads=("population/1/state/v",),
            writes=("population/1/event/spike",))
        model = {"definition": {
            "clocks": [
                {"dt": "same"},
                {"dt": "same"},
            ],
            "schedule": {"nodes": [
                first_state, second_state, blocker, first_threshold,
                second_threshold,
            ]},
        }}

        self.assertFalse(_schedule_can_contract(
            model, first_state, first_threshold))
        self.assertFalse(_schedule_can_fuse_bundles(
            model,
            [[first_state, first_threshold],
             [second_state, second_threshold]],
        ))

        blocker["effects"]["writes"] = ["population/2/state/v"]
        self.assertTrue(_schedule_can_contract(
            model, first_state, first_threshold))
        self.assertTrue(_schedule_can_fuse_bundles(
            model,
            [[first_state, first_threshold],
             [second_state, second_threshold]],
        ))

        model["run"] = {"clocks": [
            {"start_tick": 0, "steps": 10},
            {"start_tick": 0, "steps": 10},
        ]}
        second_state["clock"] = 1
        second_threshold["clock"] = 1
        model["definition"]["clocks"][1]["dt"] = "different"
        self.assertFalse(_schedule_can_fuse_bundles(
            model,
            [[first_state, first_threshold],
             [second_state, second_threshold]],
        ))

    def test_aot_compiles_custom_schedule_with_effect_conflict(self):
        snapshots = []
        for engine in ("reference", "aot", "numpy"):
            self.select(engine)
            group = b.NeuronGroup(
                2, "dv/dt=1/ms:1", threshold="v>=1", reset="v=0",
                method="euler", dt=1*b.ms)
            monitor = b.StateMonitor(group, "v", record=True)
            spikes = b.SpikeMonitor(group)
            network = b.Network(group, monitor, spikes)
            network.schedule = [
                "start", "thresholds", "groups", "synapses", "resets", "end",
            ]
            network.run(4*b.ms)
            snapshots.append((np.asarray(group.v[:]).copy(), np.asarray(monitor.v).copy(),
                              np.asarray(spikes.i[:]).copy(), np.asarray(spikes.t[:]).copy()))
            if engine == "aot":
                self.assertEqual(self.device.last_execution_plan.cpu.emitter, "slot-v1")
        for candidate in snapshots[1:]:
            for actual, expected in zip(candidate, snapshots[0], strict=True):
                np.testing.assert_array_equal(actual, expected)

    def test_canonical_regular_slots_delays_and_segmented_execution(self):
        for regular_when in ("start", "before_thresholds", "after_synapses"):
            snapshots = []
            for engine in ("reference", "aot", "numpy"):
                with self.subTest(when=regular_when, engine=engine):
                    self.select(engine)
                    group = b.NeuronGroup(
                        3, "dv/dt=1/ms:1 (unless refractory)\nx:1", threshold="v>=2",
                        reset="v=0; x+=0.25", refractory=2*b.ms, method="euler", dt=1*b.ms)
                    group.run_regularly("x = 0.5*x + v", when=regular_when, dt=2*b.ms)
                    synapse = b.Synapses(group, group, "w:1", on_pre="x_post += w; w+=0.1",
                                        on_post="w+=0.2", clock=group.clock)
                    synapse.connect(i=[2, 0, 1, 0], j=[0, 2, 0, 1])
                    synapse.w = [1, 2, 3, 4]
                    synapse.pre.delay = [0, 1, 3, 2]*b.ms
                    synapse.post.delay = 1*b.ms
                    monitor = b.StateMonitor(group, ["v", "x"], record=True)
                    spikes = b.SpikeMonitor(group)
                    network = b.Network(group, synapse, monitor, spikes)
                    network.run(4*b.ms)
                    network.run(4*b.ms)
                    snapshots.append(tuple(np.asarray(value).copy() for value in
                        (group.v[:], group.x[:], synapse.w[:], monitor.v, monitor.x, spikes.i[:], spikes.t[:],
                         group.lastspike[:], group.not_refractory[:])))
            for candidate in snapshots[1:]:
                for actual, expected in zip(candidate, snapshots[0], strict=True):
                    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-14)

    def test_aot_accepts_observationally_equivalent_custom_schedule(self):
        self.device.reinit()
        b.set_device(
            "rust_standalone", runner=self.runner, engine="aot",
            directory=self.directory / "custom-schedule-independent-aot")
        group = b.NeuronGroup(
            2, "dv/dt=1/ms:1", method="euler", dt=1*b.ms)
        network = b.Network(group)
        network.schedule = [
            "start", "thresholds", "groups", "synapses", "resets", "end",
        ]
        network.run(2*b.ms)
        np.testing.assert_array_equal(group.v[:], [2, 2])

    def test_aot_runs_same_clock_end_run_regularly(self):
        self.device.reinit()
        b.set_device(
            "rust_standalone", runner=self.runner, engine="aot",
            directory=self.directory / "run-regularly-aot")
        group = b.NeuronGroup(
            2, "dv/dt=1/ms:1\nx:1", method="euler", dt=1*b.ms)
        group.run_regularly("x += v", when="end")
        b.Network(group).run(2*b.ms)
        np.testing.assert_allclose(group.v[:], 2, rtol=0, atol=1e-14)
        np.testing.assert_allclose(group.x[:], 3, rtol=0, atol=1e-14)

    def test_aot_runs_independent_run_regularly_clock(self):
        self.device.reinit()
        b.set_device(
            "rust_standalone", runner=self.runner, engine="aot",
            directory=self.directory / "run-regularly-independent-aot")
        group = b.NeuronGroup(
            2, "dv/dt=1/ms:1\nx:1", method="euler", dt=1*b.ms)
        group.run_regularly("x += v", dt=2*b.ms, when="end")
        b.Network(group).run(4*b.ms)
        np.testing.assert_allclose(group.v[:], 4, rtol=0, atol=1e-14)
        np.testing.assert_allclose(group.x[:], 4, rtol=0, atol=1e-14)
        generated = (self.device.last_run_directory / "native/main.rs").read_text()
        self.assertIn("let c1_active", generated)

    @staticmethod
    def rich_model(delay=.2*b.ms):
        group = b.NeuronGroup(17, """
            dv/dt=(drive-v+.1*x)/ms : 1 (unless refractory)
            dx/dt=(v-x)/(2*ms) : 1
            dz/dt=(t/ms+1.0)/second : 1
            drive : 1 (constant)
            """, threshold="v>=1", reset="offset=dt/ms; v=offset*x; x+=v",
            refractory=.3*b.ms, method="euler", dt=.1*b.ms)
        group.v = np.linspace(.1, 1.2, 17)
        group.x = np.linspace(.01, .1, 17)
        group.z = .02
        group.drive = np.linspace(1.1, 1.8, 17)
        delay_option = {} if delay is None else {"delay": delay}
        synapse = b.Synapses(group, group, "weight:1 (constant)\ngain:1 (shared)",
                            on_pre="v_post+=gain*weight; x_post+=z_pre+.1",
                            clock=group.clock, **delay_option)
        synapse.connect(i=[1, 0, 1, 0, 16, 8], j=[16, 1, 16, 16, 0, 8])
        synapse.weight = [.01, -.02, .03, .01, .04, -.01]
        synapse.gain = .5
        monitor = b.StateMonitor(group, ["x", "v", "z"], record=[16, 0, 16, 8])
        spikes = b.SpikeMonitor(group)
        return b.Network(group, synapse, monitor, spikes), group, monitor, spikes

    def test_native_reference_numpy_with_guards_delays_and_scalar_temporaries(self):
        results = []
        for engine in ["aot", "reference", "numpy"]:
            self.select(engine)
            network, group, monitor, spikes = self.rich_model()
            network.run(5*b.ms)
            results.append([np.asarray(x).copy() for x in
                            (monitor.v, monitor.x, monitor.z, group.v[:], group.x[:], group.z[:],
                             spikes.i[:], spikes.t[:]/b.second, spikes.count[:],
                             group.lastspike[:]/b.second, group.not_refractory[:])])
            if engine == "aot":
                self.assertGreater(self.device.last_build_timings["compile_seconds"], 0)
                directory = self.device.last_run_directory
                source = (directory/"native/main.rs").read_text()
                self.assertIn("tick >= refractory_until[i]", source)
                self.assertNotIn("((time - lastspike[i])", source)
                self.assertNotIn("fired: &mut Vec<usize>", source)
                binary, instance = self.device.native_artifact.values()
                output = self.directory / "replay"
                result = subprocess.run([str(binary), str(instance), str(output)],
                                        env={**os.environ, "PATH": ""}, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual((output/"results.bin").read_bytes(),
                                 (directory/"rust/results.bin").read_bytes())
                manifest = json.loads((directory/"native/manifest.json").read_text())
                self.assertEqual(manifest["engine"], "rustc-aot")
        for expected in results[1:]:
            for index, (actual, other) in enumerate(zip(results[0], expected, strict=True)):
                if index in [6, 8, 10]:
                    np.testing.assert_array_equal(actual, other)
                else:
                    np.testing.assert_allclose(actual, other, rtol=1e-12, atol=1e-14, err_msg=f"engine={engine} field={index}")

    def test_native_nonspiking_time_model_and_invalid_instances(self):
        self.select("aot")
        group = b.NeuronGroup(3, "dv/dt=t*rate : 1", namespace={"rate": 1000/b.second**2},
                              method="euler", dt=1*b.ms)
        monitor = b.StateMonitor(group, "v", record=True)
        b.Network(group, monitor).run(3*b.ms)
        np.testing.assert_allclose(group.v[:], .003, rtol=0, atol=1e-15)
        binary, instance = self.device.native_artifact.values()
        original = instance.read_bytes()
        bad_dt = bytearray(original); bad_dt[8:16] = struct.pack("<d", .5)
        nan = bytearray(original); nan[32:40] = struct.pack("<d", float("nan"))
        nan_time = bytearray(original); nan_time[-8:] = struct.pack("<d", float("nan"))
        overflow = bytearray(original); overflow[-16:-8] = struct.pack("<Q", 2**64-1)
        for index, data in enumerate([
                original[:-1], original+b"trailing", bad_dt, nan,
                nan_time, overflow]):
            path, output = self.directory/f"bad-{index}.bin", self.directory/f"bad-{index}"
            path.write_bytes(data)
            result = subprocess.run([str(binary), str(path), str(output)], capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(output.exists())

    def test_constant_parameter_monitor_uses_generic_aot_plan(self):
        snapshots = []
        for engine in ("aot", "reference", "numpy"):
            self.select(engine)
            group = b.NeuronGroup(
                3, """dv/dt=-gain*v/tau : 1
                       gain : 1 (constant)
                       tau : second (constant, shared)""",
                method="euler", dt=0.1*b.ms)
            group.v = [1, 2, 3]
            group.gain = [1, 2, 3]
            group.tau = 2*b.ms
            monitor = b.StateMonitor(
                group, ["v", "gain", "tau"], record=[2, 0])
            b.Network(group, monitor).run(0.3*b.ms)
            snapshots.append((np.asarray(monitor.v).copy(),
                              np.asarray(monitor.gain).copy(),
                              np.asarray(monitor.tau/b.second).copy()))
        for actual in snapshots[1:]:
            for value, expected in zip(actual, snapshots[0], strict=True):
                np.testing.assert_allclose(value, expected, rtol=1e-12, atol=1e-14)

    def test_math_functions_and_common_operators_match_numpy(self):
        equation = """
            dv/dt=(exp(-v)+log(v+2)+log10(v+2)+expm1(v/10)+exprel(v/10)
                   +log1p(v)+sqrt(v+1)+sin(v)+cos(v)+tan(v/10)
                   +sinh(v/10)+cosh(v/10)+tanh(v)+arcsin(v/2)
                   +arccos(v/2)+arctan(v)+abs(v-.5)+sign(v-.5)
                   +ceil(v)+floor(v)+clip(v,.1,.9)+v**2+v%0.7+v//0.2)/second : 1
        """
        results = []
        for engine in ["aot", "reference", "numpy"]:
            self.select(engine)
            group = b.NeuronGroup(
                3, equation,
                threshold=("((v >= 0) and (v <= 10) and not (v == -1) "
                           "and (v != -2)) or (v == -3)"),
                reset="v %= 10", method="euler", dt=.1*b.ms,
            )
            group.v = [.2, .4, .6]
            monitor = b.StateMonitor(group, "v", record=True)
            spikes = b.SpikeMonitor(group)
            b.Network(group, monitor, spikes).run(1*b.ms)
            results.append([
                np.asarray(monitor.v).copy(), np.asarray(group.v[:]).copy(),
                np.asarray(spikes.i[:]).copy(),
            ])
            if engine == "aot":
                source = (self.device.last_run_directory / "native/main.rs").read_text()
                for fragment in [".exp()", ".ln()", ".sqrt()", ".sin()",
                                 ".powi(2)", ".floor()", "exprel("]:
                    self.assertIn(fragment, source)
        for expected in results[1:]:
            np.testing.assert_allclose(results[0][0], expected[0], rtol=2e-12,
                                       atol=1e-14)
            np.testing.assert_allclose(results[0][1], expected[1], rtol=2e-12,
                                       atol=1e-14)
            np.testing.assert_array_equal(results[0][2], expected[2])

    def test_rk4_and_exponential_euler_match_numpy(self):
        equations = {
            "rk4": """
                dv/dt=(drive-v+0.2*w-0.1*v**2)/tau : 1
                dw/dt=(v-2*w)/tau : 1
                drive : 1 (constant)
                tau : second (constant, shared)
            """,
            "exponential_euler": """
                dv/dt=(drive-v+0.2*w)/tau : 1
                dw/dt=(v-2*w)/tau : 1
                drive : 1 (constant)
                tau : second (constant, shared)
            """,
        }
        for method, model in equations.items():
            results = []
            for engine in ["aot", "reference", "numpy"]:
                self.select(engine)
                group = b.NeuronGroup(
                    4, model, threshold="v>0.8", reset="v=0.1; w+=0.05",
                    method=method, dt=0.1*b.ms,
                )
                group.v = [0.1, 0.2, 0.3, 0.4]
                group.w = [0.05, 0.1, 0.15, 0.2]
                group.drive = [0.8, 1.0, 1.2, 1.4]
                group.tau = 2*b.ms
                monitor = b.StateMonitor(group, ["v", "w"], record=True)
                spikes = b.SpikeMonitor(group)
                b.Network(group, monitor, spikes).run(4*b.ms)
                results.append([np.asarray(value).copy() for value in (
                    monitor.v, monitor.w, group.v[:], group.w[:],
                    spikes.i[:], spikes.t[:]/b.second,
                )])
                if engine == "aot":
                    exported = json.loads(
                        (self.device.last_run_directory / "model.json").read_text()
                    )
                    statements = exported["definition"]["populations"][0]["code_objects"][0]["vector"]
                    if method == "rk4":
                        self.assertTrue(any(statement["target"].startswith("__k_")
                                            for statement in statements))
                    else:
                        source = (self.device.last_run_directory / "native/main.rs").read_text()
                        self.assertIn(".exp()", source)
            for expected in results[1:]:
                for index, (actual, other) in enumerate(zip(results[0], expected, strict=True)):
                    if index == 4:
                        np.testing.assert_array_equal(actual, other)
                    else:
                        np.testing.assert_allclose(actual, other, rtol=3e-12,
                                                   atol=1e-14,
                                                   err_msg=f"method={method} field={index}")

    def test_zero_delay_delivers_directly_without_a_queue(self):
        results = []
        for engine in ["aot", "reference"]:
            self.select(engine)
            network, group, monitor, spikes = self.rich_model(delay=0*b.ms)
            network.run(2*b.ms)
            results.append([np.asarray(value).copy() for value in
                            (group.v[:], group.x[:], monitor.v, monitor.x,
                             spikes.i[:], spikes.t[:]/b.second)])
            if engine == "aot":
                source = (self.device.last_run_directory/"native/main.rs").read_text()
                self.assertNotIn("let mut queue", source)
                self.assertIn("let active_event_count", source)
        for index, (actual, expected) in enumerate(zip(*results, strict=True)):
            if index == 4:
                np.testing.assert_array_equal(actual, expected)
            else:
                np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)

    def test_flat_and_source_delivery_batches_preserve_the_same_order(self):
        results = []
        for source_threshold in [10_000_000, 0]:
            self.select("aot")
            network, group, monitor, spikes = self.rich_model()
            with patch("brian2_rust.planner.SOURCE_BATCH_MIN_EDGES", source_threshold):
                network.run(5*b.ms)
            source = (self.device.last_run_directory/"native/main.rs").read_text()
            self.assertIn(f"pending.event_count >= {source_threshold}", source)
            self.assertIn("pending.sources.extend_from_slice(&fired)", source)
            self.assertIn("pending.edges.extend_from_slice", source)
            self.assertNotIn("Vec<bool>", source)
            results.append([np.asarray(value).copy() for value in
                            (monitor.v, monitor.x, group.v[:], group.x[:], spikes.i[:],
                             spikes.t[:]/b.second, spikes.count[:])])
        for index, (flat, source) in enumerate(zip(*results, strict=True)):
            if index in [4, 6]:
                np.testing.assert_array_equal(flat, source)
            else:
                np.testing.assert_allclose(flat, source, rtol=0, atol=0)

    def test_heterogeneous_delay_groups_preserve_flat_and_source_order(self):
        results = []
        for source_threshold in [10_000_000, 0]:
            self.select("aot")
            network, group, monitor, spikes = self.rich_model(delay=None)
            synapse = next(obj for obj in network.objects if isinstance(obj, b.Synapses))
            synapse.delay = [0, .2, .3, 0, .3, .2]*b.ms
            with patch("brian2_rust.planner.SOURCE_BATCH_MIN_EDGES", source_threshold):
                network.run(5*b.ms)
            source = (self.device.last_run_directory/"native/main.rs").read_text()
            self.assertIn("source_csr_delay", source)
            self.assertIn(f"pending.event_count >= {source_threshold}", source)
            results.append([np.asarray(value).copy() for value in
                            (monitor.v, monitor.x, group.v[:], group.x[:], spikes.i[:],
                             spikes.t[:]/b.second, spikes.count[:])])
        for index, (flat, source) in enumerate(zip(*results, strict=True)):
            if index in [4, 6]:
                np.testing.assert_array_equal(flat, source)
            else:
                np.testing.assert_allclose(flat, source, rtol=0, atol=0)

    def test_native_compilation_failure_does_not_fallback_or_publish(self):
        self.select("aot")
        network, group, monitor, _ = self.rich_model()
        initial = group.v[:].copy()
        invoke = self.device._invoke
        def fail_compile(command, **kwargs):
            if command[0] == "rustc":
                raise RuntimeError("simulated compiler failure")
            self.assertEqual(command[1], "--validate")
            return invoke(command, **kwargs)
        with patch.object(self.device, "_invoke", side_effect=fail_compile):
            with self.assertRaisesRegex(RuntimeError, "compiler failure"):
                network.run(1*b.ms)
        np.testing.assert_array_equal(group.v[:], initial)
        self.assertEqual(len(monitor.t), 0)
        self.assertFalse(self.device.has_been_run)


if __name__ == "__main__":
    unittest.main()
