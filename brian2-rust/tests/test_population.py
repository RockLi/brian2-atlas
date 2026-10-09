"""Checks that distinguish element/phase semantics from a single-neuron loop."""

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import brian2 as b
import numpy as np
from numpy import sin
from brian2.devices.device import all_devices

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import brian2_rust  # noqa: E402, F401
from brian2_rust.results import load_results  # noqa: E402
from brian2_rust.protocol import attach_protocol  # noqa: E402


@b.check_units(x=b.volt, tau=b.second, scale=b.volt,
               result=b.volt / b.second)
def portable_voltage_rhs(x, tau, scale):
    """A portable pure expression with units and a built-in function."""
    return -x / tau + scale / tau * sin(x / scale)


PORTABLE_VOLTAGE_FUNCTION = b.Function(portable_voltage_rhs, stateless=True)


@b.implementation(
    "atlasir-c-abi-v1",
    """
    #include <math.h>
    double b2_native_voltage_rhs(double x, double tau, double scale) {
        return -x / tau + scale / tau * sin(x / scale);
    }
    """,
    name="b2_native_voltage_rhs",
)
@b.check_units(x=b.volt, tau=b.second, scale=b.volt,
               result=b.volt / b.second)
def native_voltage_rhs(x, tau, scale):
    return -x / tau + scale / tau * sin(x / scale)


@b.implementation(
    "atlasir-c-abi-v1",
    """
    #include <stdint.h>
    uint8_t b2_native_odd(int64_t x) { return (uint8_t)((x % 2) != 0); }
    """,
    name="b2_native_odd",
)
@b.check_units(x=1, result=bool)
@b.declare_types(x="integer", result="boolean")
def native_odd(x):
    return x % 2 != 0


@b.implementation(
    "atlasir-metal-v1",
    "inline double b2_metal_identity(double x) { return x; }",
    name="b2_metal_identity",
)
@b.check_units(x=1, result=1)
def portable_with_metal(x):
    return x


PORTABLE_CAPTURE = 0.25


@b.implementation(
    "atlasir-c-abi-v1",
    "double b2_native_captured(double x) { return x + 0.25; }",
    name="b2_native_captured",
)
@b.check_units(x=1, result=1)
def native_captured_python_state(x):
    return x + PORTABLE_CAPTURE


@b.check_units(x=1, result=1)
def captures_python_state(x):
    return x + PORTABLE_CAPTURE


@b.check_units(x=1, limit=1, result=bool)
def portable_above(x, limit):
    return x > limit


@b.check_units(x=1, result=1)
@b.declare_types(x="integer", result="integer")
def portable_integer_step(x):
    return x * 3 + 1


@b.check_units(value=1, result=bool)
@b.declare_types(value="boolean", result="boolean")
def portable_boolean_not(value):
    return value == False


class PopulationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.previous = b.get_device()
        self.previous_target = b.prefs.codegen.target
        self.device = all_devices["rust_standalone"]
        self.device.reinit()
        b.set_device("rust_standalone", runner=ROOT / "target/release/b2-runner",
                     directory=Path(self.temp.name) / "run")

    def tearDown(self):
        b.set_device(self.previous)
        b.prefs.codegen.target = self.previous_target
        self.device.reinit()

    def test_adaptive_gsl_rkf45_matches_aot_and_absolute_error(self):
        results = []
        for engine in ("reference", "aot"):
            self.device.reinit()
            b.start_scope()
            b.set_device(
                "rust_standalone", engine=engine,
                runner=ROOT / "target/release/b2-runner",
                directory=Path(self.temp.name) / f"gsl-{engine}")
            group = b.NeuronGroup(
                2,
                """
                dv/dt = -v/(0.1*ms) : 1
                du/dt = v/(0.1*ms) : 1
                """,
                method="gsl_rkf45",
                method_options={
                    "absolute_error": 1e-9,
                    "save_failed_steps": True,
                    "save_step_count": True,
                },
                dt=1*b.ms,
            )
            group.v = [1, 2]
            b.Network(group).run(1*b.ms)
            results.append((
                np.asarray(group.v[:]).copy(),
                np.asarray(group.u[:]).copy(),
                np.asarray(group._last_timestep[:]/b.second).copy(),
                np.asarray(group._failed_steps[:]).copy(),
                np.asarray(group._step_count[:]).copy(),
            ))
        for actual, expected in zip(results[1], results[0], strict=True):
            np.testing.assert_array_equal(actual, expected)
        expected_v = np.asarray([1.0, 2.0]) * np.exp(-10.0)
        np.testing.assert_allclose(results[0][0], expected_v, rtol=0, atol=1e-9)
        np.testing.assert_allclose(results[0][1], np.asarray([1.0, 2.0]) - expected_v,
                                   rtol=0, atol=1e-9)
        self.assertTrue(np.all(results[0][2] > 0))
        self.assertTrue(np.all(results[0][3] > 0))
        self.assertTrue(np.all(results[0][4] > 1))

    def test_adaptive_gsl_rk2_matches_aot_and_official_example_lifecycle(self):
        results = []
        for engine in ("reference", "aot"):
            self.device.reinit()
            b.start_scope()
            b.set_device(
                "rust_standalone", engine=engine,
                runner=ROOT / "target/release/b2-runner",
                directory=Path(self.temp.name) / f"gsl-rk2-{engine}")
            group = b.NeuronGroup(
                4,
                "dv/dt = (5-v)/tau : 1\ntau : second",
                method="gsl_rk2",
                method_options={
                    "absolute_error": 1e-8,
                    "max_steps": 10000,
                    "save_failed_steps": True,
                    "save_step_count": True,
                },
                dt=0.1*b.ms,
            )
            group.tau = [0.5, 1, 2, 5]*b.ms
            network = b.Network(group)
            network.run(0*b.ms)
            monitor = b.StateMonitor(
                group, ["v", "tau", "_step_count"], record=True,
                dt=0.1*b.ms)
            network.add(monitor)
            network.run(1*b.ms)
            results.append((
                np.asarray(group.v[:]).copy(),
                np.asarray(monitor.v).copy(),
                np.asarray(group._failed_steps[:]).copy(),
                np.asarray(group._step_count[:]).copy(),
            ))
        for actual, expected in zip(results[1], results[0], strict=True):
            np.testing.assert_array_equal(actual, expected)
        expected = 5*(1-np.exp(-1*b.ms/([0.5, 1, 2, 5]*b.ms)))
        np.testing.assert_allclose(results[0][0], expected, rtol=0, atol=1e-7)
        self.assertTrue(np.all(results[0][2] >= 0))
        self.assertTrue(np.all(results[0][3] >= 1))

    def test_additional_gsl_methods_match_aot_and_analytic_solution(self):
        for method in ("gsl_rk4", "gsl_rkck", "gsl_rk8pd"):
            results = []
            for engine in ("reference", "aot"):
                self.device.reinit()
                b.start_scope()
                b.set_device(
                    "rust_standalone", engine=engine,
                    runner=ROOT / "target/release/b2-runner",
                    directory=Path(self.temp.name) / f"{method}-{engine}")
                group = b.NeuronGroup(
                    4,
                    "dv/dt = (5-v)/tau : 1\ntau : second",
                    method=method,
                    method_options={
                        "absolute_error": 1e-10,
                        "max_steps": 10000,
                        "save_failed_steps": True,
                        "save_step_count": True,
                    },
                    dt=0.1*b.ms,
                )
                group.tau = [0.5, 1, 2, 5]*b.ms
                b.Network(group).run(1*b.ms)
                results.append((
                    np.asarray(group.v[:]).copy(),
                    np.asarray(group._last_timestep[:]/b.second).copy(),
                    np.asarray(group._failed_steps[:]).copy(),
                    np.asarray(group._step_count[:]).copy(),
                ))
            with self.subTest(method=method):
                for actual, expected_result in zip(
                        results[1], results[0], strict=True):
                    np.testing.assert_array_equal(actual, expected_result)
                expected = 5*(1-np.exp(-1*b.ms/([0.5, 1, 2, 5]*b.ms)))
                np.testing.assert_allclose(
                    results[0][0], expected, rtol=0, atol=2e-9)
                self.assertTrue(np.all(results[0][1] > 0))
                self.assertTrue(np.all(results[0][2] >= 0))
                self.assertTrue(np.all(results[0][3] >= 1))

    def test_adaptive_gsl_respects_fixed_refractory_freeze(self):
        spikes = []
        traces = []
        for engine in ("reference", "aot"):
            self.device.reinit()
            b.start_scope()
            b.set_device(
                "rust_standalone", engine=engine,
                runner=ROOT / "target/release/b2-runner",
                directory=Path(self.temp.name) / f"gsl-refractory-{engine}")
            group = b.NeuronGroup(
                1,
                "dv/dt = 400*Hz : 1 (unless refractory)",
                threshold="v >= 1", reset="v = 0", refractory=3*b.ms,
                method="gsl", dt=1*b.ms,
            )
            monitor = b.StateMonitor(group, "v", record=True, when="end")
            spike_monitor = b.SpikeMonitor(group)
            b.Network(group, monitor, spike_monitor).run(12*b.ms)
            spikes.append(np.asarray(spike_monitor.t/b.ms).copy())
            traces.append(np.asarray(monitor.v[0]).copy())
        np.testing.assert_array_equal(spikes[1], spikes[0])
        np.testing.assert_array_equal(traces[1], traces[0])
        np.testing.assert_array_equal(spikes[0], np.asarray([2.0, 7.0]))

    def test_state_monitor_custom_slot_matches_numpy(self):
        for when in ("thresholds", "after_thresholds"):
            results = []
            for backend in ("reference", "aot", "numpy"):
                self.device.reinit()
                b.start_scope()
                if backend == "numpy":
                    b.set_device("runtime")
                    b.prefs.codegen.target = "numpy"
                else:
                    b.set_device(
                        "rust_standalone", engine=backend,
                        runner=ROOT / "target/release/b2-runner",
                        directory=Path(self.temp.name) / f"monitor-{when}-{backend}")
                group = b.NeuronGroup(
                    1, "dv/dt=1/ms : 1", threshold="v>1.5", reset="v=0",
                    method="euler", dt=1*b.ms)
                monitor = b.StateMonitor(
                    group, "v", record=True, when=when, name="zz_monitor")
                b.Network(group, monitor).run(4*b.ms)
                results.append((np.asarray(monitor.t/b.ms).copy(),
                                np.asarray(monitor.v).copy()))
            for actual in results[:2]:
                for value, expected in zip(actual, results[2], strict=True):
                    np.testing.assert_array_equal(value, expected)

    def test_state_monitors_with_distinct_slots_fail_before_build(self):
        group = b.NeuronGroup(1, "dv/dt=1/ms : 1", method="euler")
        first = b.StateMonitor(group, "v", record=True, when="start")
        second = b.StateMonitor(group, "v", record=True, when="end")
        network = b.Network(group, first, second)
        with patch.object(
                self.device, "_runner",
                side_effect=AssertionError("premature build")):
            with self.assertRaisesRegex(
                    NotImplementedError, "must use the same when/order/clock"):
                network.run(1*b.ms)
        self.assertEqual((len(first.t), len(second.t)), (0, 0))

    def test_tile_boundaries_match_numpy_with_coupling_and_refractory(self):
        for size in [255, 256, 257, 513]:
            results = []
            for backend in ["rust", "numpy"]:
                if backend == "rust":
                    self.device.reinit()
                    b.set_device("rust_standalone", runner=ROOT / "target/release/b2-runner",
                                 directory=Path(self.temp.name) / f"tile-{size}")
                else:
                    b.set_device("runtime")
                    b.prefs.codegen.target = "numpy"
                group = b.NeuronGroup(size, """
                    dv/dt=(drive-v-w+t/ms*.01)/ms : 1 (unless refractory)
                    dw/dt=(v-w)/(2*ms) : 1
                    drive : 1 (constant)
                    """, threshold="v>=1", reset="w+=.03; v=.1*w; w+=v",
                    refractory=.3*b.ms, method="euler", dt=.1*b.ms)
                group.v = np.linspace(.1, 1.2, size)
                group.w = np.linspace(.01, .1, size)
                group.drive = np.linspace(1.3, 2.1, size)
                monitor = b.StateMonitor(group, ["v", "w"], record=True)
                spikes = b.SpikeMonitor(group)
                b.Network(group, monitor, spikes).run(4*b.ms)
                results.append({name: np.asarray(value).copy() for name, value in {
                    "v": monitor.v, "w": monitor.w, "final_v": group.v[:],
                    "final_w": group.w[:], "spike_i": spikes.i[:],
                    "spike_tick": np.rint(spikes.t[:]/group.clock.dt),
                    "count": spikes.count[:], "lastspike": group.lastspike[:]/b.second,
                    "not_refractory": group.not_refractory[:],
                }.items()})
            for name, actual in results[0].items():
                with self.subTest(size=size, variable=name):
                    if name in {"spike_i", "spike_tick", "count", "not_refractory"}:
                        np.testing.assert_array_equal(actual, results[1][name])
                    else:
                        np.testing.assert_allclose(actual, results[1][name], rtol=1e-12, atol=1e-14)

    def test_coupled_euler_uses_old_values_and_preserves_record_order(self):
        g = b.NeuronGroup(2, "dv/dt=w/tau : 1\ndw/dt=v/tau : 1",
                          dt=1*b.ms, method="euler", namespace={"tau": 10*b.ms})
        g.v = [1, 3]
        g.w = [2, 4]
        m = b.StateMonitor(g, ["w", "v"], record=[1, 0, 1])
        b.Network(g, m).run(1*b.ms)
        np.testing.assert_allclose(g.v[:], [1.2, 3.4], rtol=0, atol=1e-14)
        # An in-place update of v before computing dw would give 2.12/4.34.
        np.testing.assert_allclose(g.w[:], [2.1, 4.3], rtol=0, atol=1e-14)
        np.testing.assert_array_equal(m.v, [[3], [1], [3]])
        np.testing.assert_array_equal(m.w, [[4], [2], [4]])

    def test_float32_state_has_explicit_rounding_and_typed_dump(self):
        results = []
        exported = []
        for backend in ["reference", "aot", "numpy"]:
            self.device.reinit()
            b.start_scope()
            if backend == "numpy":
                b.set_device("runtime")
                b.prefs.codegen.target = "numpy"
            else:
                b.set_device(
                    "rust_standalone",
                    runner=ROOT / "target/release/b2-runner",
                    directory=Path(self.temp.name) / f"float32-{backend}",
                    engine=backend,
                )
            group = b.NeuronGroup(
                5,
                "dv/dt=(drive-v)/(3*ms) : 1\n"
                "drive : 1 (constant)",
                method="euler", dt=.1*b.ms, dtype=np.float32,
            )
            group.v = np.linspace(.1, .5, 5, dtype=np.float32)
            group.drive = np.linspace(.7, 1.1, 5, dtype=np.float32)
            monitor = b.StateMonitor(group, ["v", "drive"], record=True)
            b.Network(group, monitor).run(.7*b.ms)
            results.append((np.asarray(monitor.v).copy(),
                            np.asarray(group.v[:]).copy()))
            if backend != "numpy":
                model = json.loads(
                    (self.device.last_run_directory / "model.json").read_text())
                loaded = load_results(model, self.device.last_run_directory / "rust")
                population = loaded["populations"][0]
                self.assertEqual(population["states"]["v"].dtype, np.dtype("<f4"))
                self.assertFalse(population["states"]["v"].flags.owndata)
                exported.append(model)
        np.testing.assert_array_equal(results[0][0], results[1][0])
        np.testing.assert_array_equal(results[0][1], results[1][1])
        for trace, final in results[:2]:
            np.testing.assert_allclose(trace, results[2][0], rtol=2e-7, atol=1e-7)
            np.testing.assert_allclose(final, results[2][1], rtol=2e-7, atol=1e-7)
        state = exported[0]["definition"]["populations"][0]["states"][0]
        self.assertEqual(state["dtype"], "f32")
        statement = next(
            code for code in exported[0]["definition"]["populations"][0]["code_objects"]
            if code["kind"] == "state_update"
        )["vector"][0]
        self.assertEqual(statement["dtype"], "f32")
        self.assertEqual(statement["value"]["op"], "f64_to_f32")

        invalid = copy.deepcopy(exported[0])
        invalid["instance"]["populations"][0]["initial_state"]["v"][0] = \
            "3ff0000000000000"
        source = Path(self.temp.name) / "invalid-f32-width.json"
        source.write_text(json.dumps(invalid))
        checked = subprocess.run(
            [str(ROOT / "target/release/b2-runner"), "--validate", str(source)],
            capture_output=True, text=True,
        )
        self.assertNotEqual(checked.returncode, 0, checked.stdout)

    def test_boolean_state_is_typed_through_reference_aot_and_dump(self):
        results = []
        loaded_result = None
        for backend in ["reference", "aot", "numpy"]:
            self.device.reinit()
            b.start_scope()
            if backend == "numpy":
                b.set_device("runtime")
                b.prefs.codegen.target = "numpy"
            else:
                b.set_device(
                    "rust_standalone",
                    runner=ROOT / "target/release/b2-runner",
                    directory=Path(self.temp.name) / f"bool-{backend}",
                    engine=backend,
                )
            group = b.NeuronGroup(
                4,
                "dv/dt=drive/ms : 1\n"
                "drive : 1 (constant)\n"
                "allow : boolean (constant, shared)\n"
                "enabled : boolean",
                threshold="v >= 1 and enabled and allow",
                reset="v = 0; enabled = False",
                method="euler", dt=1*b.ms,
            )
            group.drive = [0.4, 0.6, 1.0, 1.2]
            group.allow = True
            group.enabled = [True, False, True, False]
            group.run_regularly("enabled = not enabled", when="end")
            monitor = b.StateMonitor(group, ["v", "enabled"], record=True)
            spikes = b.SpikeMonitor(group)
            b.Network(group, monitor, spikes).run(4*b.ms)
            results.append({
                "v": np.asarray(monitor.v).copy(),
                "enabled": np.asarray(monitor.enabled).copy(),
                "final_v": np.asarray(group.v[:]).copy(),
                "final_enabled": np.asarray(group.enabled[:]).copy(),
                "spike_i": np.asarray(spikes.i[:]).copy(),
                "spike_t": np.asarray(spikes.t[:] / b.ms).copy(),
            })
            if backend == "aot":
                model = json.loads(
                    (self.device.last_run_directory / "model.json").read_text())
                bool_state = next(
                    state for state in model["definition"]["populations"][0]["states"]
                    if state["name"] == "enabled")
                self.assertEqual(bool_state["dtype"], "bool")
                bool_parameter = next(
                    parameter for parameter in
                    model["definition"]["populations"][0]["parameters"]
                    if parameter["name"] == "allow")
                self.assertEqual(bool_parameter["dtype"], "bool")
                loaded_result = load_results(
                    model, self.device.last_run_directory / "rust")
        for actual in results[1:]:
            for name in results[0]:
                np.testing.assert_array_equal(actual[name], results[0][name])
        self.assertEqual(
            loaded_result["populations"][0]["trace"]["enabled"].dtype,
            np.dtype(np.bool_),
        )
        self.assertEqual(
            loaded_result["populations"][0]["states"]["enabled"].dtype,
            np.dtype(np.bool_),
        )

    def test_integer_states_preserve_width_and_large_values(self):
        results = []
        loaded_result = None
        for backend in ["reference", "aot", "numpy"]:
            self.device.reinit()
            b.start_scope()
            if backend == "numpy":
                b.set_device("runtime")
                b.prefs.codegen.target = "numpy"
            else:
                b.set_device(
                    "rust_standalone",
                    runner=ROOT / "target/release/b2-runner",
                    directory=Path(self.temp.name) / f"integer-{backend}",
                    engine=backend,
                )
            group = b.NeuronGroup(
                2,
                "dv/dt=0*Hz : 1\n"
                "i32v : integer\n"
                "i64v : integer\n"
                "u32v : integer\n"
                "u64v : integer\n"
                "large : boolean",
                method="euler", dt=1*b.ms,
                dtype={
                    "i32v": np.int32,
                    "i64v": np.int64,
                    "u32v": np.uint32,
                    "u64v": np.uint64,
                },
            )
            group.i32v = [-7, 8]
            group.i64v = [9007199254740993, 9007199254740994]
            group.u32v = [2**31 + 1, 2**31 + 2]
            group.u64v = [2**63 + 3, 2**63 + 4]
            group.run_regularly(
                "i32v += 3\n"
                "i64v += 1\n"
                "u32v += 4\n"
                "u64v += 2\n"
                "large = i64v > 9007199254740994",
                when="end",
            )
            # Brian2's runtime DynamicArray does not support unsigned monitor
            # columns, so record signed values and verify all final arrays.
            monitor = b.StateMonitor(
                group, ["i32v", "i64v", "large"], record=True)
            b.Network(group, monitor).run(2*b.ms)
            results.append({
                "trace_i32": np.asarray(monitor.i32v).copy(),
                "trace_i64": np.asarray(monitor.i64v).copy(),
                "trace_large": np.asarray(monitor.large).copy(),
                "i32": np.asarray(group.i32v[:]).copy(),
                "i64": np.asarray(group.i64v[:]).copy(),
                "u32": np.asarray(group.u32v[:]).copy(),
                "u64": np.asarray(group.u64v[:]).copy(),
                "large": np.asarray(group.large[:]).copy(),
            })
            if backend == "aot":
                model = json.loads(
                    (self.device.last_run_directory / "model.json").read_text())
                states = {
                    state["name"]: state["dtype"]
                    for state in model["definition"]["populations"][0]["states"]
                }
                self.assertEqual(states["i32v"], "i32")
                self.assertEqual(states["i64v"], "i64")
                self.assertEqual(states["u32v"], "u32")
                self.assertEqual(states["u64v"], "u64")
                loaded_result = load_results(
                    model, self.device.last_run_directory / "rust")
        for actual in results[1:]:
            for name in results[0]:
                np.testing.assert_array_equal(actual[name], results[0][name])
        loaded_states = loaded_result["populations"][0]["states"]
        self.assertEqual(loaded_states["i32v"].dtype, np.dtype("<i4"))
        self.assertEqual(loaded_states["i64v"].dtype, np.dtype("<i8"))
        self.assertEqual(loaded_states["u32v"].dtype, np.dtype("<u4"))
        self.assertEqual(loaded_states["u64v"].dtype, np.dtype("<u8"))
        np.testing.assert_array_equal(
            loaded_states["u64v"], [2**63 + 7, 2**63 + 8])

    def test_reset_statement_order_and_simultaneous_spike_order(self):
        g = b.NeuronGroup(3, "dv/dt=0*Hz : 1\ndw/dt=0*Hz : 1",
                          threshold="v>1", reset="v=w; w=v+1",
                          dt=1*b.ms, method="euler")
        g.v = [1.1, 0.2, 1.2]
        g.w = [2, 3, 4]
        m = b.StateMonitor(g, "v", record=[2, 0])
        s = b.SpikeMonitor(g)
        b.Network(g, m, s).run(2*b.ms)
        np.testing.assert_array_equal(s.i[:], [0, 2, 0, 2])
        np.testing.assert_array_equal(s.t[:] / b.ms, [0, 0, 1, 1])
        np.testing.assert_array_equal(s.count[:], [2, 0, 2])
        np.testing.assert_array_equal(g.spikes, [0, 2])
        np.testing.assert_array_equal(g.v[:], [3, 0.2, 5])
        np.testing.assert_array_equal(g.w[:], [4, 3, 6])
        np.testing.assert_array_equal(m.v, [[1.2, 4], [1.1, 2]])

    def test_scalar_vector_parameters_and_neuron_indices(self):
        g = b.NeuronGroup(4, """
            dv/dt=(i+1.0+drive)/(N*tau) : 1
            drive : 1 (constant)
            tau : second (constant, shared)
            """, dt=1*b.ms, method="euler")
        g.drive = [0, 1, 0, 1]
        g.tau = 10*b.ms
        m = b.StateMonitor(g, "v", record=[3, 1])
        b.Network(g, m).run(1*b.ms)
        np.testing.assert_allclose(g.v[:], [0.025, 0.075, 0.075, 0.125], rtol=0, atol=1e-14)
        np.testing.assert_array_equal(g.drive[:], [0, 1, 0, 1])
        self.assertEqual(float(g.tau / b.ms), 10)

    def test_linked_variables_identity_fixed_and_dynamic_indices(self):
        results = []
        exported = []
        for backend in ["reference", "aot", "numpy"]:
            self.device.reinit()
            b.start_scope()
            if backend == "numpy":
                b.set_device("runtime")
                b.prefs.codegen.target = "numpy"
            else:
                b.set_device(
                    "rust_standalone",
                    runner=ROOT / "target/release/b2-runner",
                    directory=Path(self.temp.name) / f"linked-{backend}",
                    engine=backend,
                )
            source = b.NeuronGroup(
                4, "dx/dt=(i+1)/ms : 1", method="euler", dt=1*b.ms,
                name="a_source")
            target = b.NeuronGroup(
                4,
                "dy/dt=(direct + fixed + dynamic)/ms : 1\n"
                "direct : 1 (linked)\n"
                "fixed : 1 (linked)\n"
                "dynamic : 1 (linked)\n"
                "k : integer",
                method="euler", dt=1*b.ms, name="b_target")
            target.direct = b.linked_var(source, "x")
            target.fixed = b.linked_var(source, "x", index=[3, 1, 1, 0])
            target.k = [0, 1, 2, 3]
            target.dynamic = b.linked_var(source, "x", index="k")
            target.run_regularly("k = (k + 1) % N", when="end")
            monitor = b.StateMonitor(
                target, ["y", "direct", "fixed", "dynamic"],
                record=True)
            b.Network(source, target, monitor).run(3*b.ms)
            results.append({
                "y": np.asarray(monitor.y).copy(),
                "direct": np.asarray(monitor.direct).copy(),
                "fixed": np.asarray(monitor.fixed).copy(),
                "dynamic": np.asarray(monitor.dynamic).copy(),
                "final_y": np.asarray(target.y[:]).copy(),
                "final_k": np.asarray(target.k[:]).copy(),
            })
            if backend != "numpy":
                exported.append(json.loads(
                    (self.device.last_run_directory / "model.json").read_text()))
        for actual in results[:2]:
            for name, expected in results[2].items():
                np.testing.assert_array_equal(actual[name], expected)
        links = exported[0]["definition"]["populations"][1]["linked_variables"]
        self.assertEqual([link["name"] for link in links],
                         ["direct", "dynamic", "fixed"])
        self.assertEqual(links[0]["index"], {"kind": "identity"})
        self.assertEqual(links[1]["index"], {"kind": "state", "name": "k"})
        self.assertEqual(links[2]["index"], {
            "kind": "constant", "values": [3, 1, 1, 0]})
        schedule = exported[0]["definition"]["schedule"]["nodes"]
        source_state = next(
            node for node in schedule
            if node["owner_kind"] == "population"
            and node["owner_index"] == 0
            and node["operation"] == "code_object"
            and node["when"] == "groups")
        target_state = next(
            node for node in schedule
            if node["owner_kind"] == "population"
            and node["owner_index"] == 1
            and node["operation"] == "code_object"
            and node["when"] == "groups")
        self.assertIn(source_state["id"], target_state["dependencies"])

    def test_dynamic_linked_index_fails_closed_at_runtime(self):
        for backend in ("reference", "aot"):
            with self.subTest(backend=backend):
                self.device.reinit()
                b.start_scope()
                directory = Path(self.temp.name) / f"linked-bounds-{backend}"
                b.set_device(
                    "rust_standalone",
                    runner=ROOT / "target/release/b2-runner",
                    directory=directory, engine=backend)
                source = b.NeuronGroup(
                    2, "dx/dt=0*Hz : 1", method="euler", dt=1*b.ms)
                target = b.NeuronGroup(
                    1, "dy/dt=external/ms : 1\nexternal : 1 (linked)\nk : integer",
                    method="euler", dt=1*b.ms)
                target.external = b.linked_var(source, "x", index="k")
                target.k = 2
                with self.assertRaisesRegex(
                        RuntimeError, "linked variable index out of bounds"):
                    b.Network(source, target).run(1*b.ms)
                self.assertFalse((directory / "rust/results.bin").exists())

    def test_scalar_linked_variable_broadcast_matches_numpy(self):
        results = []
        exported = []
        for backend in ("reference", "aot", "numpy"):
            self.device.reinit()
            b.start_scope()
            if backend == "numpy":
                b.set_device("runtime")
                b.prefs.codegen.target = "numpy"
            else:
                b.set_device(
                    "rust_standalone",
                    runner=ROOT / "target/release/b2-runner",
                    directory=Path(self.temp.name) / f"linked-broadcast-{backend}",
                    engine=backend,
                )
            source = b.NeuronGroup(
                1, "dx/dt=(1-x)/(2*ms) : 1", method="euler", dt=1*b.ms,
                name="broadcast_source")
            target = b.NeuronGroup(
                4, "dy/dt=(external-y)/(3*ms) : 1\nexternal : 1 (linked)",
                method="euler", dt=1*b.ms, name="broadcast_target")
            source.x = 0.25
            target.y = [0, 1, 2, 3]
            target.external = b.linked_var(source, "x")
            monitor = b.StateMonitor(
                target, ["y", "external"], record=True)
            network = b.Network(source, target, monitor)
            if backend == "reference":
                report = brian2_rust.capability_report(network, 4*b.ms)
                self.assertTrue(report.supported, report.format_text())
            network.run(4*b.ms)
            results.append((np.asarray(monitor.y).copy(),
                            np.asarray(monitor.external).copy()))
            if backend != "numpy":
                exported.append(json.loads(
                    (self.device.last_run_directory / "model.json").read_text()))
        for actual in results[:2]:
            for value, expected in zip(actual, results[2], strict=True):
                np.testing.assert_array_equal(value, expected)
        for model in exported:
            link = model["definition"]["populations"][1]["linked_variables"][0]
            self.assertEqual(link["index"], {
                "kind": "constant", "values": [0, 0, 0, 0]})

    def test_shared_subexpression_with_linked_broadcast_matches_numpy(self):
        results = []
        for backend in ("reference", "aot", "numpy"):
            self.device.reinit()
            b.start_scope()
            if backend == "numpy":
                b.set_device("runtime")
                b.prefs.codegen.target = "numpy"
            else:
                b.set_device(
                    "rust_standalone",
                    runner=ROOT / "target/release/b2-runner",
                    directory=Path(self.temp.name) / f"shared-expression-{backend}",
                    engine=backend,
                )
            source = b.NeuronGroup(
                1, "dx/dt=(1-x)/(4*ms) : 1", method="euler", dt=1*b.ms,
                name="shared_expression_source")
            target = b.NeuronGroup(
                3,
                "dv/dt=(drive-v)/(3*ms) : 1\n"
                "drive = 2*x : 1 (shared)\n"
                "x : 1 (linked)",
                method="euler", dt=1*b.ms, name="shared_expression_target")
            source.x = 0.25
            target.v = [0, 0.5, 1]
            target.x = b.linked_var(source, "x")
            monitor = b.StateMonitor(target, ["v", "drive"], record=True)
            b.Network(source, target, monitor).run(5*b.ms)
            results.append((np.asarray(monitor.v).copy(),
                            np.asarray(monitor.drive).copy()))
        for actual in results[:2]:
            for value, expected in zip(actual, results[2], strict=True):
                np.testing.assert_array_equal(value, expected)

    def test_subgroup_state_monitor_preserves_local_record_order(self):
        results = []
        for backend in ("reference", "aot", "numpy"):
            self.device.reinit()
            b.start_scope()
            if backend == "numpy":
                b.set_device("runtime")
                b.prefs.codegen.target = "numpy"
            else:
                b.set_device(
                    "rust_standalone", runner=ROOT / "target/release/b2-runner",
                    directory=Path(self.temp.name) / f"subgroup-state-{backend}",
                    engine=backend)
            group = b.NeuronGroup(
                6, "dv/dt=(drive-v)/(2*ms) : 1\ndrive : 1 (constant)",
                method="euler", dt=1*b.ms)
            group.v = np.arange(6) / 10
            group.drive = np.arange(6) + 1
            subgroup = group[2:5]
            monitor = b.StateMonitor(subgroup, ["v", "drive"], record=[2, 0])
            b.Network(group, monitor).run(4*b.ms)
            results.append((np.asarray(monitor.v).copy(),
                            np.asarray(monitor.drive).copy()))
        for actual in results[:2]:
            for value, expected in zip(actual, results[2], strict=True):
                np.testing.assert_array_equal(value, expected)

    def test_state_monitor_independent_clock_matches_numpy(self):
        results = []
        for backend in ("reference", "aot", "numpy"):
            self.device.reinit()
            b.start_scope()
            if backend == "numpy":
                b.set_device("runtime")
                b.prefs.codegen.target = "numpy"
            else:
                b.set_device(
                    "rust_standalone", runner=ROOT / "target/release/b2-runner",
                    directory=Path(self.temp.name) / f"monitor-clock-{backend}",
                    engine=backend)
            group = b.NeuronGroup(
                2, "dv/dt=(drive-v)/(2*ms) : 1\ndrive : 1 (constant)",
                method="euler", dt=1*b.ms)
            group.v = [0, 1]
            group.drive = [1, 2]
            monitor = b.StateMonitor(
                group, ["v", "drive"], record=[1, 0], dt=2*b.ms)
            b.Network(group, monitor).run(6*b.ms)
            results.append((np.asarray(monitor.t / b.ms).copy(),
                            np.asarray(monitor.v).copy(),
                            np.asarray(monitor.drive).copy()))
        for actual in results[:2]:
            for value, expected in zip(actual, results[2], strict=True):
                np.testing.assert_array_equal(value, expected)

    def test_poisson_group_mutable_rates_monitor_matches_numpy(self):
        results = []
        for backend in ("reference", "aot", "numpy"):
            self.device.reinit()
            b.start_scope()
            if backend == "numpy":
                b.set_device("runtime")
                b.prefs.codegen.target = "numpy"
            else:
                b.set_device(
                    "rust_standalone",
                    runner=ROOT / "target/release/b2-runner",
                    directory=Path(self.temp.name) / f"poisson-rates-{backend}",
                    engine=backend)
            group = b.PoissonGroup(3, rates=0*b.Hz, dt=1*b.ms)
            group.run_regularly(
                "rates = (i+1)*Hz + int(t >= 2*ms)*3*Hz", dt=1*b.ms)
            monitor = b.StateMonitor(group, "rates", record=[2, 0], dt=2*b.ms)
            b.Network(group, monitor).run(4*b.ms)
            results.append((np.asarray(monitor.t / b.ms).copy(),
                            np.asarray(monitor.rates / b.Hz).copy(),
                            np.asarray(group.rates[:] / b.Hz).copy()))
        for actual in results[:2]:
            for value, expected in zip(actual, results[2], strict=True):
                np.testing.assert_array_equal(value, expected)

    def test_poisson_group_monitor_partial_final_interval_matches_numpy(self):
        results = []
        for backend in ("reference", "aot", "numpy"):
            self.device.reinit()
            b.start_scope()
            if backend == "numpy":
                b.set_device("runtime")
                b.prefs.codegen.target = "numpy"
            else:
                b.set_device(
                    "rust_standalone",
                    runner=ROOT / "target/release/b2-runner",
                    directory=Path(self.temp.name) / f"poisson-partial-{backend}",
                    engine=backend)
            group = b.PoissonGroup(2, rates=[3, 5] * b.Hz, dt=0.25*b.ms)
            monitor = b.StateMonitor(group, "rates", record=[1], dt=1*b.ms)
            b.Network(group, monitor).run(2.5*b.ms)
            results.append((np.asarray(monitor.t / b.ms).copy(),
                            np.asarray(monitor.rates / b.Hz).copy()))
        for actual in results[:2]:
            for value, expected in zip(actual, results[2], strict=True):
                np.testing.assert_array_equal(value, expected)

    def test_spike_monitor_additional_variables_match_numpy(self):
        results = []
        for backend in ("reference", "aot", "numpy"):
            self.device.reinit()
            b.start_scope()
            if backend == "numpy":
                b.set_device("runtime")
                b.prefs.codegen.target = "numpy"
            else:
                b.set_device(
                    "rust_standalone",
                    runner=ROOT / "target/release/b2-runner",
                    directory=Path(self.temp.name) / f"spike-values-{backend}",
                    engine=backend,
                )
            source = b.NeuronGroup(
                2, "signal : 1", dt=1*b.ms, name="spike_value_source")
            source.signal = [0.125, 0.25]
            group = b.NeuronGroup(
                2,
                "dv/dt=drive/ms : 1\n"
                "drive : 1 (constant)\nexternal : 1 (linked)",
                threshold="v >= 1", reset="v = 0", method="euler", dt=1*b.ms)
            group.v = [0.25, 0.5]
            group.drive = [0.5, 0.75]
            group.external = b.linked_var(source, "signal")
            spikes = b.SpikeMonitor(
                group, variables=["v", "drive", "external"])
            network = b.Network(source, group, spikes)
            if backend == "reference":
                report = brian2_rust.capability_report(network, 5*b.ms)
                self.assertTrue(report.supported, report.format_text())
            network.run(5*b.ms)
            results.append({
                "i": np.asarray(spikes.i[:]).copy(),
                "t": np.asarray(spikes.t[:]).copy(),
                "v": np.asarray(spikes.v[:]).copy(),
                "drive": np.asarray(spikes.drive[:]).copy(),
                "external": np.asarray(spikes.external[:]).copy(),
            })
        for actual in results[:2]:
            for name, expected in results[2].items():
                np.testing.assert_array_equal(actual[name], expected)

    def test_stochastic_xi_reference_and_aot_are_reproducible(self):
        results = []
        for backend in ("reference", "aot"):
            self.device.reinit()
            b.start_scope()
            b.set_device(
                "rust_standalone",
                runner=ROOT / "target/release/b2-runner",
                directory=Path(self.temp.name) / f"stochastic-xi-{backend}",
                engine=backend,
            )
            b.seed(1729)
            group = b.NeuronGroup(
                3,
                "dx/dt=(0.5-x)/(4*ms) + sigma*xi/sqrt(ms) : 1\n"
                "sigma : 1 (constant)",
                method="euler", dt=0.5*b.ms)
            group.x = [0.1, 0.2, 0.3]
            group.sigma = [0.02, 0.03, 0.04]
            monitor = b.StateMonitor(group, "x", record=True)
            network = b.Network(group, monitor)
            if backend == "reference":
                report = brian2_rust.capability_report(network, 20*b.ms)
                self.assertTrue(report.supported, report.format_text())
            network.run(20*b.ms)
            result = np.asarray(monitor.x).copy()
            self.assertTrue(np.isfinite(result).all())
            self.assertGreater(np.max(np.abs(np.diff(result, axis=1))), 0)
            results.append(result)
        np.testing.assert_array_equal(results[0], results[1])

    def test_dimensionful_clip_zero_in_generated_state_update_matches_numpy(self):
        results = []
        for backend in ("reference", "aot", "numpy"):
            self.device.reinit()
            b.start_scope()
            if backend == "numpy":
                b.set_device("runtime")
                b.prefs.codegen.target = "numpy"
            else:
                b.set_device(
                    "rust_standalone",
                    runner=ROOT / "target/release/b2-runner",
                    directory=Path(self.temp.name) / f"clip-zero-{backend}",
                    engine=backend,
                )
            group = b.NeuronGroup(
                2,
                "dv/dt=(baseline + clip(v-v_floor, 0*mV, inf*mV) - v)/tau : volt",
                method="euler", dt=0.1*b.ms,
                namespace={"baseline": -60*b.mV, "v_floor": -65*b.mV,
                           "tau": 5*b.ms})
            group.v = [-70, -55]*b.mV
            monitor = b.StateMonitor(group, "v", record=True)
            b.Network(group, monitor).run(1*b.ms)
            results.append((np.asarray(monitor.v/b.volt).copy(),
                            np.asarray(group.v[:]/b.volt).copy()))
        for actual in results[:2]:
            for value, expected in zip(actual, results[2], strict=True):
                np.testing.assert_allclose(value, expected, rtol=0, atol=1e-15)

    def test_multiplicative_sde_heun_and_milstein_match_engines(self):
        for method in ("heun", "milstein"):
            results = []
            for backend in ("reference", "aot"):
                self.device.reinit()
                b.start_scope()
                b.set_device(
                    "rust_standalone",
                    runner=ROOT / "target/release/b2-runner",
                    directory=Path(self.temp.name) /
                    f"multiplicative-{method}-{backend}",
                    engine=backend,
                )
                b.seed(2027)
                group = b.NeuronGroup(
                    3,
                    "dX/dt = (mu - 0.5*second*sigma**2)*X + "
                    "X*sigma*xi*second**.5 : 1",
                    method=method, dt=0.1*b.ms,
                    namespace={"mu": 0.5/b.second,
                               "sigma": 0.1/b.second})
                group.X = [0.8, 1.0, 1.2]
                monitor = b.StateMonitor(group, "X", record=True)
                network = b.Network(group, monitor)
                report = brian2_rust.capability_report(network, 5*b.ms)
                self.assertTrue(report.supported, report.format_text())
                network.run(5*b.ms)
                values = np.asarray(monitor.X).copy()
                self.assertTrue(np.isfinite(values).all())
                results.append(values)
            np.testing.assert_array_equal(results[0], results[1])

    def test_dimensionful_milstein_zero_increments_match_engines(self):
        results = []
        for backend in ("reference", "aot", "numpy"):
            self.device.reinit()
            b.start_scope()
            if backend == "numpy":
                b.set_device("runtime")
                b.prefs.codegen.target = "numpy"
            else:
                b.set_device(
                    "rust_standalone",
                    runner=ROOT / "target/release/b2-runner",
                    directory=Path(self.temp.name) /
                    f"dimensionful-milstein-{backend}",
                    engine=backend,
                )
            b.seed(2029)
            group = b.NeuronGroup(
                2,
                "dC/dt = -C/tau : mmolar\n"
                "dh/dt = -h/tau + h*xi/sqrt(tau) : 1",
                method="milstein", dt=0.1*b.ms,
                namespace={"tau": 10*b.ms})
            group.C = [1, 2]*b.mmolar
            group.h = [0.5, 0.75]
            monitor = b.StateMonitor(group, ["C", "h"], record=True)
            network = b.Network(group, monitor)
            if backend != "numpy":
                report = brian2_rust.capability_report(network, 1*b.ms)
                self.assertTrue(report.supported, report.format_text())
            network.run(1*b.ms)
            results.append((np.asarray(monitor.C/b.mmolar).copy(),
                            np.asarray(monitor.h).copy()))
        for value, expected in zip(results[0], results[1], strict=True):
            np.testing.assert_array_equal(value, expected)
        for actual in results:
            np.testing.assert_array_equal(actual[0], results[2][0])
            self.assertTrue(np.isfinite(actual[1]).all())

    def test_identity_self_link_preserves_state_update_snapshot(self):
        results = []
        for backend in ("reference", "aot", "numpy"):
            self.device.reinit()
            b.start_scope()
            if backend == "numpy":
                b.set_device("runtime")
                b.prefs.codegen.target = "numpy"
            else:
                b.set_device(
                    "rust_standalone",
                    runner=ROOT / "target/release/b2-runner",
                    directory=Path(self.temp.name) / f"self-link-{backend}",
                    engine=backend)
            group = b.NeuronGroup(
                2,
                "dx/dt=1/ms : 1\n"
                "dy/dt=alias/ms : 1\n"
                "alias : 1 (linked)",
                method="euler", dt=1*b.ms)
            group.x = [2, 3]
            group.alias = b.linked_var(group, "x")
            b.Network(group).run(2*b.ms)
            results.append((np.asarray(group.x[:]).copy(),
                            np.asarray(group.y[:]).copy()))
        for actual in results[:2]:
            for value, expected in zip(actual, results[2], strict=True):
                np.testing.assert_array_equal(value, expected)
        np.testing.assert_array_equal(results[0][0], [4, 5])
        np.testing.assert_array_equal(results[0][1], [5, 7])

    def test_portable_function_contract_matches_numpy_and_is_aot_inlined(self):
        results = []
        contracts = []
        models = []
        for backend in ["reference", "aot", "numpy"]:
            self.device.reinit()
            b.start_scope()
            if backend == "numpy":
                b.set_device("runtime")
                b.prefs.codegen.target = "numpy"
            else:
                b.set_device(
                    "rust_standalone", runner=ROOT / "target/release/b2-runner",
                    directory=Path(self.temp.name) / f"function-{backend}",
                    engine=backend)
            group = b.NeuronGroup(
                3,
                "dv/dt=portable_voltage_rhs(v, tau, scale) + "
                "portable_voltage_rhs(v, 2.0*tau, 2.0*scale) : volt\n"
                "tau : second (constant, shared)\n"
                "scale : volt (constant, shared)",
                method="euler", dt=.1*b.ms,
                namespace={"portable_voltage_rhs": PORTABLE_VOLTAGE_FUNCTION})
            group.v = [-3, 2, 7]*b.mV
            group.tau = 8*b.ms
            group.scale = 5*b.mV
            monitor = b.StateMonitor(group, "v", record=True)
            network = b.Network(group, monitor)
            network.run(2*b.ms)
            results.append((np.asarray(monitor.v/b.volt).copy(),
                            np.asarray(group.v[:]/b.volt).copy()))
            if backend != "numpy":
                model = json.loads(
                    (self.device.last_run_directory / "model.json").read_text())
                contracts.append(model["definition"]["functions"])
                models.append(model)
        for trace, final in results[:2]:
            np.testing.assert_allclose(trace, results[2][0], rtol=1e-12, atol=1e-15)
            np.testing.assert_allclose(final, results[2][1], rtol=1e-12, atol=1e-15)
        self.assertEqual(contracts[0], contracts[1])
        contract = contracts[0][0]
        self.assertEqual(contract["name"], "portable_voltage_rhs")
        self.assertEqual(contract["abi"], "b2ir-function-v1")
        self.assertEqual(contract["body"]["op"], "add")
        self.assertEqual(len(contract["implementations"]["b2ir-expression-v1"]), 64)
        corruptions = [
            ("stateful", lambda value: value["definition"]["functions"][0]
             ["effects"].__setitem__("stateful", True)),
            ("return-dimensions", lambda value: value["definition"]["functions"][0]
             ["return_dimensions"].__setitem__(2, 0.0)),
            ("implementation-hash", lambda value: value["definition"]["functions"][0]
             ["implementations"].__setitem__("b2ir-expression-v1", "invalid")),
            ("undefined-call", lambda value: value["definition"]["functions"][0]
             .__setitem__("name", "renamed_function")),
        ]
        for name, corrupt in corruptions:
            with self.subTest(corruption=name):
                invalid = copy.deepcopy(models[0])
                corrupt(invalid)
                source = Path(self.temp.name) / f"invalid-function-{name}.json"
                source.write_text(json.dumps(invalid))
                checked = subprocess.run(
                    [str(ROOT / "target/release/b2-runner"), "--validate",
                     str(source)], capture_output=True, text=True)
                self.assertNotEqual(checked.returncode, 0, checked.stdout)

    def test_native_c_function_abi_matches_portable_reference(self):
        results = []
        aot_model = None
        for backend in ("reference", "aot", "numpy"):
            self.device.reinit()
            b.start_scope()
            if backend == "numpy":
                b.set_device("runtime")
                b.prefs.codegen.target = "numpy"
            else:
                b.set_device(
                    "rust_standalone", runner=ROOT / "target/release/b2-runner",
                    directory=Path(self.temp.name) / f"native-function-{backend}",
                    engine=backend)
            group = b.NeuronGroup(
                3,
                "dv/dt=native_voltage_rhs(v, tau, scale) : volt\n"
                "tau : second (constant, shared)\n"
                "scale : volt (constant, shared)",
                method="euler", dt=.1*b.ms,
                namespace={"native_voltage_rhs": native_voltage_rhs})
            group.v = [-3, 2, 7]*b.mV
            group.tau = 8*b.ms
            group.scale = 5*b.mV
            monitor = b.StateMonitor(group, "v", record=True)
            b.Network(group, monitor).run(2*b.ms)
            results.append((np.asarray(monitor.v/b.volt).copy(),
                            np.asarray(group.v[:]/b.volt).copy()))
            if backend == "aot":
                native_dir = self.device.last_run_directory / "native"
                model = json.loads(
                    (self.device.last_run_directory / "model.json").read_text())
                aot_model = model
                contract = model["definition"]["functions"][0]
                cpu = contract["backend_implementations"]["cpu"]
                self.assertEqual(cpu["abi"],
                                 "b2ir-c-abi-v1")
                self.assertEqual(cpu["symbol"],
                                 "b2_native_voltage_rhs")
                manifest = json.loads((native_dir / "manifest.json").read_text())
                self.assertEqual(manifest["native_functions"][0]["source_sha256"],
                                 cpu["source_sha256"])
                self.assertTrue(next(native_dir.glob("function-*.o")).is_file())
        for actual in results[:2]:
            for value, expected in zip(actual, results[2], strict=True):
                np.testing.assert_allclose(value, expected, rtol=1e-12,
                                           atol=1e-15)
        invalid = copy.deepcopy(aot_model)
        invalid["definition"]["functions"][0]["backend_implementations"]["cpu"][
            "source_sha256"] = "0" * 64
        attach_protocol(invalid)
        source = Path(self.temp.name) / "invalid-native-function.json"
        source.write_text(json.dumps(invalid))
        checked = subprocess.run(
            [str(ROOT / "target/release/b2-runner"), "--validate", str(source)],
            capture_output=True, text=True)
        self.assertNotEqual(checked.returncode, 0, checked.stdout)
        self.assertIn("invalid native Function implementation", checked.stderr)

    def test_native_only_function_never_executes_python_in_reference(self):
        for backend in ("aot", "reference"):
            with self.subTest(backend=backend):
                self.device.reinit()
                b.start_scope()
                directory = Path(self.temp.name) / f"native-only-{backend}"
                b.set_device(
                    "rust_standalone", runner=ROOT / "target/release/b2-runner",
                    directory=directory, engine=backend)
                group = b.NeuronGroup(
                    1, "dv/dt=native_captured_python_state(v)/ms : 1",
                    method="euler", dt=.1*b.ms,
                    namespace={"native_captured_python_state":
                               native_captured_python_state})
                group.v = 1
                if backend == "aot":
                    b.Network(group).run(.2*b.ms)
                    np.testing.assert_allclose(group.v[:], [1.2625],
                                               rtol=0, atol=1e-15)
                    model = json.loads((directory / "model.json").read_text())
                    self.assertIsNone(
                        model["definition"]["functions"][0]["body"])
                else:
                    with self.assertRaisesRegex(
                            RuntimeError, "native-only Function"):
                        b.Network(group).run(.1*b.ms)
                    self.assertFalse((directory / "rust/results.bin").exists())

    def test_native_c_abi_preserves_i64_and_bool_boundary(self):
        results = []
        for backend in ("reference", "aot", "numpy"):
            self.device.reinit()
            b.start_scope()
            if backend == "numpy":
                b.set_device("runtime")
                b.prefs.codegen.target = "numpy"
            else:
                b.set_device(
                    "rust_standalone", runner=ROOT / "target/release/b2-runner",
                    directory=Path(self.temp.name) / f"native-types-{backend}",
                    engine=backend)
            group = b.NeuronGroup(
                2, "counter : integer\nenabled : boolean", dt=1*b.ms,
                dtype={"counter": np.int64},
                namespace={"native_odd": native_odd})
            group.counter = [9007199254740993, 9007199254740994]
            group.run_regularly("enabled = native_odd(counter)", when="end")
            b.Network(group).run(1*b.ms)
            results.append((np.asarray(group.counter[:]).copy(),
                            np.asarray(group.enabled[:]).copy()))
        for actual in results[:2]:
            for value, expected in zip(actual, results[2], strict=True):
                np.testing.assert_array_equal(value, expected)

    def test_function_contract_preserves_future_gpu_implementation(self):
        self.device.reinit()
        b.start_scope()
        directory = Path(self.temp.name) / "gpu-function-contract"
        b.set_device(
            "rust_standalone", runner=ROOT / "target/release/b2-runner",
            directory=directory, engine="reference")
        group = b.NeuronGroup(
            1, "dv/dt=portable_with_metal(v)/ms : 1", method="euler",
            namespace={"portable_with_metal": portable_with_metal})
        group.v = 1
        b.Network(group).run(.1*b.ms)
        contract = json.loads((directory / "model.json").read_text())[
            "definition"]["functions"][0]
        metal = contract["backend_implementations"]["metal"]
        self.assertEqual(metal["abi"], "b2ir-metal-v1")
        self.assertEqual(metal["symbol"], "b2_metal_identity")
        self.assertEqual(len(metal["source_sha256"]), 64)
        np.testing.assert_allclose(group.v[:], [1.1], rtol=0, atol=1e-15)

    def test_portable_function_rejects_hidden_python_state(self):
        group = b.NeuronGroup(
            1, "dv/dt=captures_python_state(v)/ms : 1",
            method="euler",
            namespace={"captures_python_state": captures_python_state})
        with self.assertRaisesRegex(NotImplementedError, "captures Python/global state"):
            b.Network(group).run(.1*b.ms)

    def test_portable_boolean_function_drives_threshold(self):
        results = []
        for backend in ["reference", "aot", "numpy"]:
            self.device.reinit()
            b.start_scope()
            if backend == "numpy":
                b.set_device("runtime")
                b.prefs.codegen.target = "numpy"
            else:
                b.set_device(
                    "rust_standalone", runner=ROOT / "target/release/b2-runner",
                    directory=Path(self.temp.name) / f"bool-function-{backend}",
                    engine=backend)
            threshold = ("v > limit" if backend == "numpy" else
                         "portable_above(v, limit)")
            group = b.NeuronGroup(
                2, "dv/dt=(i+1.0)/ms : 1", threshold=threshold,
                reset="v=0", method="euler", dt=.25*b.ms,
                namespace={"portable_above": portable_above, "limit": 1.0})
            spikes = b.SpikeMonitor(group)
            b.Network(group, spikes).run(2*b.ms)
            results.append((np.asarray(spikes.i[:]).copy(),
                            np.asarray(spikes.t[:]/b.second).copy(),
                            np.asarray(group.v[:]).copy()))
        for actual in results[:2]:
            for value, expected in zip(actual, results[2], strict=True):
                np.testing.assert_allclose(value, expected, rtol=0, atol=1e-15)

    def test_portable_integer_and_boolean_function_signatures(self):
        results = []
        contracts = None
        for backend in ("reference", "aot", "numpy"):
            self.device.reinit()
            b.start_scope()
            if backend == "numpy":
                b.set_device("runtime")
                b.prefs.codegen.target = "numpy"
            else:
                b.set_device(
                    "rust_standalone",
                    runner=ROOT / "target/release/b2-runner",
                    directory=Path(self.temp.name) / f"typed-function-{backend}",
                    engine=backend)
            group = b.NeuronGroup(
                2, "counter : integer\nenabled : boolean", dt=1*b.ms,
                dtype={"counter": np.int64},
                namespace={
                    "portable_integer_step": portable_integer_step,
                    "portable_boolean_not": portable_boolean_not,
                })
            group.counter = [9007199254740993, -7]
            group.enabled = [True, False]
            group.run_regularly(
                "counter = portable_integer_step(counter)\n"
                "enabled = portable_boolean_not(enabled)",
                when="end")
            b.Network(group).run(2*b.ms)
            results.append((np.asarray(group.counter[:]).copy(),
                            np.asarray(group.enabled[:]).copy()))
            if backend == "aot":
                model = json.loads(
                    (self.device.last_run_directory / "model.json").read_text())
                contracts = {item["name"]: item
                             for item in model["definition"]["functions"]}
        for actual in results[:2]:
            for value, expected in zip(actual, results[2], strict=True):
                np.testing.assert_array_equal(value, expected)
        self.assertEqual(
            contracts["portable_integer_step"]["arguments"][0]["dtype"],
            "i64")
        self.assertEqual(
            contracts["portable_integer_step"]["return_dtype"], "i64")
        self.assertEqual(
            contracts["portable_boolean_not"]["arguments"][0]["dtype"],
            "bool")
        self.assertEqual(
            contracts["portable_boolean_not"]["return_dtype"], "bool")

    def test_constant_over_dt_int_and_timestep_match_numpy(self):
        results = []
        for backend in ["aot", "reference", "numpy"]:
            self.device.reinit()
            b.start_scope()
            if backend == "numpy":
                b.set_device("runtime")
                b.prefs.codegen.target = "numpy"
            else:
                b.set_device(
                    "rust_standalone", runner=ROOT / "target/release/b2-runner",
                    directory=Path(self.temp.name) / f"expressions-{backend}",
                    engine=backend)
            group = b.NeuronGroup(
                3,
                """dv/dt=(held + int(v >= 0) + int(-1.75) + timestep(t, dt))/second : 1
                   held = v + t/second : 1 (constant over dt)""",
                method="euler", dt=1*b.ms)
            group.v = [-1, 0, 1]
            monitor = b.StateMonitor(group, ["v", "held"], record=True)
            b.Network(group, monitor).run(3*b.ms)
            results.append((np.asarray(monitor.v).copy(),
                            np.asarray(monitor.held).copy(),
                            np.asarray(group.v[:]).copy(),
                            np.asarray(group.held[:]).copy()))
        for actual in results[1:]:
            for left, right in zip(results[0], actual, strict=True):
                np.testing.assert_allclose(left, right, rtol=0, atol=1e-14)

    def test_mutable_per_neuron_parameter_is_monitored_reset_and_persisted(self):
        g = b.NeuronGroup(
            2, "dv/dt=drive/ms : 1\ndrive : 1",
            threshold="v>=2", reset="drive+=1; v=0",
            dt=1*b.ms, method="euler")
        g.drive = [1, 2]
        monitor = b.StateMonitor(g, ["v", "drive"], record=True)
        b.Network(g, monitor).run(3*b.ms)

        np.testing.assert_array_equal(monitor.v, [[0, 1, 0], [0, 0, 0]])
        np.testing.assert_array_equal(monitor.drive, [[1, 1, 2], [2, 3, 4]])
        np.testing.assert_array_equal(g.v[:], [0, 0])
        np.testing.assert_array_equal(g.drive[:], [3, 5])
        model = json.loads((self.device.last_run_directory / "model.json").read_text())
        self.assertEqual(
            [state["name"] for state in model["definition"]["populations"][0]["states"]],
            ["drive", "v"])
        self.assertNotIn("drive", model["instance"]["populations"][0]["parameters"])

    def test_mutable_shared_parameter_scalar_random_matches_numpy_semantics(self):
        results = []
        for backend in ["reference", "aot", "numpy"]:
            self.device.reinit()
            b.start_scope()
            b.seed(12345)
            if backend == "numpy":
                b.set_device("runtime")
                b.prefs.codegen.target = "numpy"
            else:
                b.set_device(
                    "rust_standalone", runner=ROOT / "target/release/b2-runner",
                    directory=Path(self.temp.name) / f"mutable-shared-{backend}",
                    engine=backend)
            group = b.NeuronGroup(
                513, "rates : Hz\nselected_index : integer (shared)",
                dt=1*b.ms)
            group.run_regularly(
                "selected_index = int(floor(rand()*N))\n"
                "rates = 10*Hz*int(selected_index == i)",
                when="end", name="shared_selector")
            monitor = b.StateMonitor(
                group, "rates", record=True, when="end", order=1)
            b.Network(group, monitor).run(4*b.ms)
            rates = np.asarray(monitor.rates / b.Hz).copy()
            selected = int(group.selected_index[:])
            self.assertEqual(rates.shape, (513, 4))
            np.testing.assert_array_equal(np.sum(rates == 10, axis=0), 1)
            np.testing.assert_array_equal(np.sum(rates, axis=0), 10)
            self.assertEqual(np.flatnonzero(np.asarray(group.rates / b.Hz) == 10).tolist(),
                             [selected])
            results.append((rates, selected))
        np.testing.assert_array_equal(results[0][0], results[1][0])
        self.assertEqual(results[0][1], results[1][1])

        model = json.loads(
            (Path(self.temp.name) / "mutable-shared-aot/model.json").read_text())
        population = model["definition"]["populations"][0]
        selected = next(state for state in population["states"]
                        if state["name"] == "selected_index")
        self.assertEqual(selected["index_domain"], "neuron")
        self.assertEqual(
            len(model["instance"]["populations"][0]["initial_state"]
                ["selected_index"]), 513)

    def test_mutable_shared_parameter_can_be_monitored_and_updated_between_runs(self):
        results = []
        for backend in ("reference", "aot", "numpy"):
            self.device.reinit()
            b.start_scope()
            if backend == "numpy":
                b.set_device("runtime")
                b.prefs.codegen.target = "numpy"
            else:
                b.set_device(
                    "rust_standalone", runner=ROOT / "target/release/b2-runner",
                    directory=Path(self.temp.name) / f"mutable-shared-monitor-{backend}",
                    engine=backend)
            group = b.NeuronGroup(
                3, "shared_value : 1 (shared)", dt=1*b.ms)
            group.shared_value = 1
            monitor = b.StateMonitor(group, "shared_value", record=[0])
            network = b.Network(group, monitor)
            network.run(2*b.ms)
            group.shared_value = 3
            network.run(2*b.ms)
            results.append(np.asarray(monitor.shared_value).copy())
        for values in results:
            np.testing.assert_array_equal(values, [[1, 1, 3, 3]])

    def test_mutable_shared_writer_feeds_threshold_scalar_hoists(self):
        results = []
        for backend in ("reference", "aot", "numpy"):
            self.device.reinit()
            b.start_scope()
            if backend == "numpy":
                b.set_device("runtime")
                b.prefs.codegen.target = "numpy"
            else:
                b.set_device(
                    "rust_standalone", runner=ROOT / "target/release/b2-runner",
                    directory=Path(self.temp.name) / f"shared-threshold-{backend}",
                    engine=backend)
            group = b.NeuronGroup(
                3, "v : 1\ndirection : 1 (shared)\noffset : 1 (shared)",
                threshold="v > direction + offset", reset="", dt=1*b.ms)
            group.v = 10
            group.run_regularly(
                "direction = t/ms\noffset = direction + 1", when="start")
            monitor = b.SpikeMonitor(group)
            b.Network(group, monitor).run(3*b.ms)
            results.append((np.asarray(monitor.i[:]).copy(),
                            np.asarray(monitor.t[:]/b.ms).copy(),
                            float(group.direction[:]), float(group.offset[:])))
        for actual in results[:2]:
            for value, expected in zip(actual, results[2], strict=True):
                np.testing.assert_array_equal(value, expected)

    def test_mutable_shared_parameter_rejects_stateful_writer(self):
        group = b.NeuronGroup(
            3, "rates : Hz\nselected_index : integer (shared)")
        group.run_regularly("selected_index += 1")
        with patch.object(
                self.device, "_runner",
                side_effect=AssertionError("premature build")):
            with self.assertRaisesRegex(
                    NotImplementedError, "cannot read its previous value"):
                b.Network(group).run(.1*b.ms)

    def test_dynamic_shared_subexpression_is_not_frozen_as_a_parameter(self):
        results = []
        for backend in ["reference", "aot", "numpy"]:
            self.device.reinit()
            b.start_scope()
            if backend == "numpy":
                b.set_device("runtime")
                b.prefs.codegen.target = "numpy"
            else:
                b.set_device(
                    "rust_standalone", runner=ROOT / "target/release/b2-runner",
                    directory=Path(self.temp.name) / f"shared-expression-{backend}",
                    engine=backend)
            group = b.NeuronGroup(
                3,
                "v : 1\nphase : 1 (constant, shared)\n"
                "drive = phase + t/ms : 1 (shared)",
                dt=1*b.ms)
            group.phase = 2
            group.run_regularly("v = drive", when="end")
            monitor = b.StateMonitor(
                group, "v", record=True, when="end", order=1)
            b.Network(group, monitor).run(3*b.ms)
            results.append(np.asarray(monitor.v).copy())
        for actual in results:
            np.testing.assert_array_equal(
                actual, np.tile([2, 3, 4], (3, 1)))

    def test_same_clock_populations_share_one_parallel_dispatch(self):
        self.device.reinit()
        directory = Path(self.temp.name) / "fused-populations"
        b.set_device(
            "rust_standalone", runner=ROOT / "target/release/b2-runner",
            directory=directory, engine="aot", threads=4)
        equations = "\n".join(
            f"dx{index}/dt=(1-x{index})/ms : 1" for index in range(6))
        first = b.NeuronGroup(
            32768, equations, method="euler", dt=.1*b.ms,
            name="fused_population_first")
        second = b.NeuronGroup(
            32768, equations, method="euler", dt=.1*b.ms,
            name="fused_population_second")
        first.x0 = .25
        second.x0 = .5
        b.Network(first, second).run(.2*b.ms)
        np.testing.assert_allclose(first.x0[:], .3925, rtol=0, atol=1e-14)
        np.testing.assert_allclose(second.x0[:], .595, rtol=0, atol=1e-14)
        generated = (self.device.last_run_directory / "native/main.rs").read_text()
        self.assertEqual(generated.count("// fused population clock group"), 1)
        summary = json.loads(
            (self.device.last_run_directory / "rust/summary.json").read_text())
        self.assertTrue(summary["parallel_state_update"])

    def test_memory_monitors_preserve_dense_spikes_and_duplicate_samples(self):
        size, steps = 257, 9  # Crosses multiple spike-buffer growth boundaries.
        g = b.NeuronGroup(size, "dv/dt=0*Hz:1\ndw/dt=1*Hz:1", threshold="True",
                          reset="v=v", dt=1*b.ms, method="euler")
        g.v = np.arange(size) / 10
        g.w = np.arange(size) / 100
        record = [256, 0, 256]
        m = b.StateMonitor(g, ["w", "v"], record=record)
        spikes = b.SpikeMonitor(g)
        b.Network(g, m, spikes).run(steps*b.ms)
        np.testing.assert_array_equal(spikes.i[:], np.tile(np.arange(size), steps))
        np.testing.assert_allclose(spikes.t[:]/b.ms, np.repeat(np.arange(steps), size), rtol=0, atol=1e-14)
        np.testing.assert_array_equal(spikes.count[:], np.full(size, steps))
        np.testing.assert_allclose(m.w, np.asarray(record)[:, None]/100 + np.arange(steps)*.001, rtol=0, atol=1e-14)
        np.testing.assert_array_equal(m.v, np.repeat(np.asarray(record)[:, None]/10, steps, axis=1))
        directory = self.device.last_run_directory / "rust"
        summary = json.loads((directory / "summary.json").read_text())
        self.assertEqual(summary["schema"], "b2-result-dump-v3")
        self.assertEqual(summary["dump_bytes"], (directory / "results.bin").stat().st_size)
        self.assertTrue({"final_state", "spike_counts", "populations", "refractory"}
                        .isdisjoint(summary))
        self.assertFalse((directory / "state.csv").exists())
        self.assertFalse((directory / "spikes.csv").exists())
        model = json.loads((self.device.last_run_directory / "model.json").read_text())
        loaded = load_results(model, directory)
        self.assertFalse(loaded["populations"][0]["trace"]["v"].flags.owndata)
        self.assertTrue(all(np.isfinite(value) and value >= 0 for value in summary["timings"].values()))
        self.assertIn("dump_write_seconds", summary["timings"])

    def test_scalar_temporary_is_visible_to_each_reset_lane(self):
        g = b.NeuronGroup(3, "dv/dt=0*Hz : 1\ndw/dt=0*Hz : 1",
                          threshold="v>1", reset="offset=0.25; v=offset+w; w=v+offset",
                          dt=1*b.ms, method="euler")
        g.v = [1.1, 0.2, 1.2]
        g.w = [2, 3, 4]
        m = b.StateMonitor(g, ["v", "w"], record=True)
        s = b.SpikeMonitor(g)
        b.Network(g, m, s).run(1*b.ms)
        np.testing.assert_array_equal(g.v[:], [2.25, 0.2, 4.25])
        np.testing.assert_array_equal(g.w[:], [2.5, 3, 4.5])

    def test_one_and_two_dimensional_timed_arrays_match_numpy_and_segmented_run(self):
        results = []
        for backend, segmented in (("reference", False), ("aot", False),
                                   ("aot", True), ("numpy", False)):
            self.device.reinit()
            b.start_scope()
            if backend == "numpy":
                b.set_device("runtime")
                b.prefs.codegen.target = "numpy"
            else:
                b.set_device(
                    "rust_standalone", runner=ROOT / "target/release/b2-runner",
                    directory=Path(self.temp.name) / f"timed-{backend}-{segmented}",
                    engine=backend)
            wave = b.TimedArray([0, 1, 2, 3], dt=.2*b.ms,
                                name=f"wave_{backend}_{segmented}")
            field = b.TimedArray(
                [[0, 10, 20], [1, 11, 21], [2, 12, 22], [3, 13, 23]],
                dt=.2*b.ms, name=f"field_{backend}_{segmented}")
            group = b.NeuronGroup(
                3, "dv/dt=(wave(t) + field(t, i))/ms : 1",
                method="euler", dt=.1*b.ms,
                namespace={"wave": wave, "field": field})
            monitor = b.StateMonitor(group, "v", record=True)
            network = b.Network(group, monitor)
            if segmented:
                network.run(.4*b.ms)
                network.run(.6*b.ms)
            else:
                network.run(1*b.ms)
            results.append((np.asarray(monitor.v).copy(),
                            np.asarray(group.v[:]).copy()))
        for actual in results[1:]:
            for value, expected in zip(actual, results[0], strict=True):
                np.testing.assert_allclose(value, expected, rtol=0, atol=1e-14)

    def test_constant_parameter_cannot_be_written_by_reset(self):
        g = b.NeuronGroup(3, "dv/dt=drive/second : 1\ndrive : 1 (constant)",
                          threshold="v>1", reset="drive=0", dt=1*b.ms, method="euler")
        m = b.StateMonitor(g, "v", record=True)
        s = b.SpikeMonitor(g)
        with patch.object(self.device, "_runner", side_effect=AssertionError("premature build")):
            with self.assertRaisesRegex(NotImplementedError, "read-only parameter"):
                b.Network(g, m, s).run(1*b.ms)

    def test_index_arithmetic_uses_an_explicit_float_conversion(self):
        g = b.NeuronGroup(2, "dv/dt=0*Hz : 1", method="euler", dt=1*b.ms,
                          threshold="True", reset="v=i*(i+1)")
        m = b.StateMonitor(g, "v", record=True)
        s = b.SpikeMonitor(g)
        b.Network(g, m, s).run(1*b.ms)
        np.testing.assert_array_equal(g.v[:], [0, 2])
        model = json.loads(
            (self.device.last_run_directory / "model.json").read_text())
        reset = next(
            code for code in model["definition"]["populations"][0]["code_objects"]
            if code["kind"] == "reset")
        encoded = json.dumps(reset["vector"])
        self.assertIn("index_to_f64", encoded)


if __name__ == "__main__":
    unittest.main()
