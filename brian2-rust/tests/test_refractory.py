"""Fixed refractory timing, conditional writes, result lifecycle and IR validation."""

import copy
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
from brian2.devices.device import RuntimeDevice, all_devices

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import brian2_rust  # noqa: E402, F401
from brian2_rust.export import lower_network  # noqa: E402
from brian2_rust.results import load_results  # noqa: E402
from brian2_rust.protocol import attach_protocol  # noqa: E402
from brian2_rust.schedule import build_schedule  # noqa: E402
from brian2_rust.spec import bits  # noqa: E402
from brian2_rust.native import fixed_schedule_is_semantically_equivalent, generate_source  # noqa: E402


class RefractoryTest(unittest.TestCase):
    def test_custom_schedule_cannot_reorder_implicit_refractory_writes(self):
        results = []
        for backend in ("reference", "numpy", "aot"):
            self.select(backend)
            group = b.NeuronGroup(
                1, "dv/dt=lastspike/second**2:1", threshold="True", reset="",
                refractory=1*b.ms, method="euler", dt=1*b.ms)
            network = b.Network(group)
            network.schedule = ["start", "thresholds", "groups", "synapses", "resets", "end"]
            if backend == "aot":
                model = lower_network(network, 3*b.ms)
                self.assertFalse(fixed_schedule_is_semantically_equivalent(model))
                self.assertIn("canonical node", generate_source(model))
            network.run(3*b.ms)
            results.append(np.asarray(group.v[:]).copy())
        np.testing.assert_allclose(results, [[2e-6], [2e-6], [2e-6]], rtol=0, atol=1e-18)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.previous_device = b.get_device()
        self.previous_target = b.prefs.codegen.target
        self.previous_legacy = b.prefs.legacy.refractory_timing
        b.prefs.legacy.refractory_timing = False
        self.device = all_devices["rust_standalone"]
        self.runner = ROOT / "target/release/b2-runner"
        self.selection = 0
        self.select("rust")

    def select(self, backend):
        if backend in {"rust", "reference", "aot"}:
            self.device.reinit()
            self.selection += 1
            b.set_device(
                "rust_standalone", runner=self.runner,
                directory=self.directory / f"refractory-{backend}-{self.selection}",
                engine="aot" if backend == "aot" else "reference")
        else:
            b.set_device("runtime")
            b.prefs.codegen.target = "numpy"

    def tearDown(self):
        b.set_device(self.previous_device)
        b.prefs.codegen.target = self.previous_target
        b.prefs.legacy.refractory_timing = self.previous_legacy
        self.device.reinit()

    @staticmethod
    def make_group(freeze=True, size=1, **kwargs):
        flag = " (unless refractory)" if freeze else ""
        options = dict(threshold="v>=1", reset="v=0", refractory=3*b.ms,
                       method="euler", dt=1*b.ms)
        options.update(kwargs)
        model = options.pop("model", f"dv/dt=0.5/ms : 1{flag}\ndx/dt=1/ms : 1")
        return b.NeuronGroup(size, model, **options)

    @staticmethod
    def network(group, *objects):
        state = b.StateMonitor(group, ["v", "x"], record=True)
        spikes = b.SpikeMonitor(group)
        return b.Network(group, state, spikes, *objects), state, spikes

    @staticmethod
    def snapshot(group, state, spikes):
        return [np.asarray(value).copy() for value in
                (state.v, state.x, group.v[:], group.x[:], spikes.i[:],
                 spikes.t[:] / b.second, group.lastspike[:] / b.second,
                 group.not_refractory[:])]

    def test_freeze_is_optional_and_unflagged_state_keeps_integrating(self):
        for freeze, expected_ticks in [(True, [1, 5, 9]), (False, [1, 4, 7])]:
            results = []
            for backend in ["rust", "numpy"]:
                self.select(backend)
                group = self.make_group(freeze)
                net, state, spikes = self.network(group)
                net.run(10*b.ms)
                np.testing.assert_array_equal(np.rint(spikes.t[:] / b.ms), expected_ticks)
                np.testing.assert_array_equal(state.x, [np.arange(10)])
                self.assertEqual(group.x[0], 10)
                if freeze:
                    np.testing.assert_array_equal(state.v, [[0, .5, 0, 0, 0, .5, 0, 0, 0, .5]])
                results.append(self.snapshot(group, state, spikes))
            for actual, expected in zip(*results, strict=True):
                np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-15)

    def test_rk4_and_exponential_euler_preserve_unless_refractory(self):
        for method in ["rk4", "exponential_euler"]:
            results = []
            for backend in ["rust", "numpy"]:
                self.select(backend)
                group = self.make_group(
                    method=method, dt=0.1*b.ms, refractory=0.3*b.ms,
                    threshold="v>=0.25", reset="v=0",
                    model=("dv/dt=(1-v)/ms : 1 (unless refractory)\n"
                           "dx/dt=(-x+0.2*v)/ms : 1"),
                )
                net, state, spikes = self.network(group)
                net.run(2*b.ms)
                results.append(self.snapshot(group, state, spikes))
            for index, (actual, expected) in enumerate(zip(*results, strict=True)):
                if index in {4, 7}:
                    np.testing.assert_array_equal(actual, expected)
                else:
                    np.testing.assert_allclose(actual, expected, rtol=3e-12,
                                               atol=1e-14,
                                               err_msg=f"method={method} field={index}")

    def test_zero_subtick_fractional_and_long_run_boundaries(self):
        for period, dt, count, spacing in [(0, 1, 8, 1), (.2, 1, 8, 1),
                                           (1.5, 1, 8, 1), (2.5, 1, 8, 2),
                                           (2.9995, 1, 8, 3), (.3, .1, 1000, 3)]:
            for backend in ["rust", "numpy"]:
                with self.subTest(period=period, backend=backend):
                    self.select(backend)
                    group = self.make_group(threshold="True", refractory=period*b.ms, dt=dt*b.ms)
                    net, _, spikes = self.network(group)
                    net.run(count*dt*b.ms)
                    np.testing.assert_array_equal(np.rint(spikes.t[:] / group.clock.dt),
                                                  np.arange(0, count, spacing))
                    self.assertFalse(group.not_refractory[0])

    def test_synaptic_guard_applies_per_statement_and_reset_overrides_it(self):
        for delay in [0, 1]:
            results = []
            for backend in ["rust", "numpy"]:
                self.select(backend)
                group = self.make_group(size=3, model="dv/dt=0*Hz:1 (unless refractory)\ndx/dt=0*Hz:1",
                                        threshold="v>1", reset="v=0.25")
                group.v = [1.1, 1.2, 0]
                synapse = b.Synapses(group, group, "w:1", clock=group.clock, delay=delay*b.ms,
                                    on_pre="v_post += w; x_post += v_post; v_post = 0.7")
                synapse.connect(i=[0, 0], j=[1, 2])
                synapse.w = 2
                net, state, spikes = self.network(group, synapse)
                net.run((delay+1)*b.ms)
                np.testing.assert_allclose(group.v[:], [.25, .25, .7], rtol=0, atol=1e-15)
                np.testing.assert_allclose(group.x[:], [0, 1.2 if delay == 0 else .25, 2], rtol=0, atol=1e-15)
                results.append(self.snapshot(group, state, spikes))
            for actual, expected in zip(*results, strict=True):
                np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-15)

    def test_initial_lastspike_and_public_arrays_match_numpy(self):
        results = []
        for backend in ["rust", "numpy"]:
            self.select(backend)
            group = self.make_group(size=3, threshold="True", reset="v=0; x=(t-lastspike)/ms")
            group.lastspike = [0, -1, -10_000_000]*b.ms
            group.not_refractory = False
            net, state, spikes = self.network(group)
            net.run(4*b.ms)
            np.testing.assert_array_equal(spikes.i[:], [2, 1, 0, 2])
            np.testing.assert_array_equal(np.rint(spikes.t[:] / b.ms), [0, 2, 3, 3])
            results.append(self.snapshot(group, state, spikes))
        for actual, expected in zip(*results, strict=True):
            np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-15)

    def test_duration_and_boolean_expression_refractory_match_numpy(self):
        cases = [
            ("3*ms", "", None),
            ("tau_ref", "tau_ref : second", [2, 3]*b.ms),
            ("x < 3", "", None),
        ]
        for refractory, extra, tau_values in cases:
            results = []
            for backend in ("reference", "aot", "numpy"):
                self.select(backend)
                model = "dv/dt=0.5/ms : 1 (unless refractory)\ndx/dt=1/ms : 1"
                if extra:
                    model += "\n" + extra
                group = self.make_group(
                    size=2, model=model, refractory=refractory,
                    threshold="v >= 1", reset="v = 0")
                if tau_values is not None:
                    group.tau_ref = tau_values
                net, state, spikes = self.network(group)
                if backend in {"reference", "aot"}:
                    lowered = lower_network(net, 10*b.ms)
                    definition = lowered["definition"]["populations"][0]
                    self.assertEqual(definition["refractory"]["mode"], "expression")
                    update = next(code for code in definition["code_objects"]
                                  if code["kind"] == "state_update")
                    self.assertIn("not_refractory", update["effects"]["writes"])
                net.run(10*b.ms)
                results.append(self.snapshot(group, state, spikes))
            for actual in results[:2]:
                for value, expected in zip(actual, results[2], strict=True):
                    np.testing.assert_allclose(
                        value, expected, rtol=0, atol=1e-15,
                        err_msg=f"refractory={refractory}")

    def test_masked_rhs_skips_nonfinite_arithmetic_across_tiles(self):
        # Use valid IR to isolate the statement guard from Brian's Euler
        # temporary generation. Inactive lanes have v=0, active lanes have v=2.
        size = 513
        group = self.make_group(size=size, threshold="False")
        mask = np.arange(size) % 2 == 0
        group.v = np.where(mask, 2., 0.)
        group.lastspike = np.where(mask, -10., 0.)*b.ms
        net, _, _ = self.network(group)
        model = lower_network(net, 1*b.ms)
        update = model["definition"]["populations"][0]["code_objects"][0]
        update["scalar"] = []
        update["vector"] = [
            {"target": "v", "dtype": "f64", "condition": "not_refractory",
             "dimensions": [0.0] * 7,
             "value": {"op": "div", "left": {"op": "literal", "bits": bits(1.)},
                       "right": {"op": "load", "name": "v"}}},
            {"target": "x", "dtype": "f64", "dimensions": [0.0] * 7,
             "value": {"op": "load", "name": "v"}},
        ]
        update["effects"] = {"reads": ["not_refractory", "v"], "writes": ["v", "x"]}
        model["definition"]["schedule"] = build_schedule(
            model["definition"], model["instance"],
            model["definition"]["schedule"]["base_slots"])
        attach_protocol(model)
        source = self.directory / "masked.json"
        source.write_text(json.dumps(model))
        output = self.directory / "masked"
        result = subprocess.run([str(self.runner), str(source), str(output)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        population = load_results(model, output)["populations"][0]
        for name in ["v", "x"]:
            np.testing.assert_array_equal(population["states"][name], np.where(mask, .5, 0.))

        # The same zero denominator must still fail when its lane is active,
        # even in the final, partial tile. No successful summary may be emitted.
        model["instance"]["populations"][0]["initial_state"]["v"][-1] = bits(0.)
        attach_protocol(model)
        source.write_text(json.dumps(model))
        output = self.directory / "unmasked"
        result = subprocess.run([str(self.runner), str(source), str(output)], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("non-finite numeric value", result.stderr)
        self.assertFalse((output / "summary.json").exists())

    def test_unsupported_refractory_rejected_before_build(self):
        for ref in [-1*b.ms, float("nan")*b.ms]:
            with self.subTest(refractory=ref):
                group = self.make_group(refractory=ref)
                net, state, _ = self.network(group)
                with patch.object(self.device, "_runner", side_effect=AssertionError("premature build")):
                    with self.assertRaisesRegex(NotImplementedError, "refractory"):
                        net.run(1*b.ms)
                self.assertEqual(len(state.t), 0)
        group = self.make_group()
        net, _, _ = self.network(group)
        b.prefs.legacy.refractory_timing = True
        with self.assertRaisesRegex(NotImplementedError, "legacy refractory"):
            net.run(1*b.ms)
        b.prefs.legacy.refractory_timing = False
        group.lastspike = 1*b.ms
        with self.assertRaisesRegex(NotImplementedError, "initial lastspike"):
            net.run(1*b.ms)
        group = self.make_group(reset="lastspike=t; v=0")
        net, _, _ = self.network(group)
        with self.assertRaisesRegex(NotImplementedError, "read-only"):
            net.run(1*b.ms)
        group = self.make_group(refractory=1*b.mV)
        net, _, _ = self.network(group)
        with self.assertRaises(b.DimensionMismatchError):
            net.run(1*b.ms)

    def test_python_free_replay_and_invalid_refractory_ir(self):
        group = self.make_group()
        synapse = b.Synapses(group, group, on_pre="v_post+=0.1", clock=group.clock)
        synapse.connect(i=0, j=0)
        net, _, _ = self.network(group, synapse)
        original = lower_network(net, 10*b.ms)
        update = original["definition"]["populations"][0]["code_objects"][0]
        guarded_index = next(i for i, s in enumerate(update["vector"]) if s["target"] == "v")
        unsafe_vector = copy.deepcopy(update["vector"])
        x_index = next(i for i, statement in enumerate(unsafe_vector)
                       if statement["target"] == "x")
        unsafe_vector.insert(x_index, {
            "target": "conditional_temp", "dtype": "f64",
            "value": {"op": "load", "name": "x"},
            "condition": "not_refractory",
        })
        unsafe_vector[x_index + 1]["value"] = {"op": "load", "name": "conditional_temp"}
        mutations = [
            (("instance", "populations", 0, "refractory", "period_ticks"), 4),
            (("instance", "populations", 0, "refractory", "period"), bits(-.003)),
            (("instance", "populations", 0, "refractory", "initial_lastspike"), []),
            (("instance", "populations", 0, "refractory", "initial_not_refractory"), [1]),
            (("instance", "populations", 0, "refractory", "initial_lastspike", 0), bits(1)),
            (("definition", "populations", 0, "refractory", "frozen_states"), ["missing"]),
            (("definition", "populations", 0, "refractory", "frozen_states"), ["v", "v"]),
            (("definition", "populations", 0, "refractory", "mode"), "invalid"),
            (("definition", "populations", 0, "refractory", "mode"), "expression"),
            (("definition", "populations", 0, "refractory"), None),
            (("definition", "populations", 0, "code_objects", 0, "vector", guarded_index, "condition"), None),
            (("definition", "populations", 0, "code_objects", 0, "vector"), unsafe_vector),
            (("definition", "synapses", 0, "code_objects", 0, "vector", 0, "condition"), "not_refractory_pre"),
            (("definition", "populations", 0, "code_objects", 2, "vector", 0, "condition"), "not_refractory"),
        ]
        for index, mutation in enumerate([None, *mutations]):
            model = copy.deepcopy(original)
            if mutation is not None:
                path, value = mutation
                node = model
                for key in path[:-1]:
                    node = node[key]
                node[path[-1]] = value
            source = self.directory / f"model-{index}.json"
            output = self.directory / f"run-{index}"
            source.write_text(json.dumps(model))
            result = subprocess.run([str(self.runner), str(source), str(output)],
                                    env={**os.environ, "PATH": ""}, capture_output=True, text=True)
            if mutation is None:
                self.assertEqual(result.returncode, 0, result.stderr)
                population = load_results(model, output)["populations"][0]
                np.testing.assert_array_equal(population["counts"], [3])
                np.testing.assert_array_equal(
                    population["refractory"]["not_refractory"], [False])
            else:
                self.assertNotEqual(result.returncode, 0, mutation)
                self.assertFalse(output.exists())

    def test_bad_refractory_results_do_not_publish_partial_state(self):
        group = self.make_group()
        net, state, _ = self.network(group)
        invoke = self.device._invoke

        def corrupt_result(command, **kwargs):
            result = invoke(command, **kwargs)
            path = Path(command[-1]) / "results.bin"
            data = bytearray(path.read_bytes())
            (count, steps, records, variables, states, spikes, last, flags) = \
                struct.unpack_from("<8Q", data, 40)
            self.assertEqual(flags, 1)
            available = (40 + 64 + steps*records*variables*8 + spikes*16 +
                         count*8 + last*8 + states*count*8 + count*8)
            data[available] ^= 1
            path.write_bytes(data)
            return result

        with patch.object(self.device, "_invoke", side_effect=corrupt_result), \
                patch.object(RuntimeDevice, "code_object", side_effect=AssertionError("runtime fallback")):
            with self.assertRaisesRegex(RuntimeError, "refractory state"):
                net.run(10*b.ms)
        self.assertEqual(len(state.t), 0)
        self.assertEqual(group.x[0], 0)
        self.assertEqual(group.lastspike[0], -1e4*b.second)
        self.assertTrue(group.not_refractory[0])
        self.assertFalse(self.device.has_been_run)


if __name__ == "__main__":
    unittest.main()
