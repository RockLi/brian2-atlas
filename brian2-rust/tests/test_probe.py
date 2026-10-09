"""Numerical differential checks and fail-closed capability/IR checks."""

import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from brian2 import (Hz, Network, NeuronGroup, PoissonGroup, SpikeMonitor,
                    StateMonitor, Synapses, mV, ms, prefs, second)
from brian2.units.fundamentalunits import DimensionMismatchError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from brian2_rust import CapabilityError, capability_report, export_network  # noqa: E402
from brian2_rust.protocol import attach_protocol, canonical_bytes, layer_hashes  # noqa: E402
from brian2_rust.results import load_results  # noqa: E402


class ProbeTest(unittest.TestCase):
    def setUp(self):
        prefs.codegen.target = "numpy"
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.runner = ROOT / "target" / "release" / "b2-runner"
        self.assertTrue(self.runner.exists(), "Build the Rust runner before running tests")

    def make_network(self, model="dv/dt = (drive-v)/tau : 1", initial=0, spiking=True, **kwargs):
        options = dict(method="euler", dt=0.1 * ms, namespace={"drive": 1.5, "tau": 10 * ms})
        if spiking:
            options.update(threshold="v > 1", reset="v = 0")
        options.update(kwargs)
        group = NeuronGroup(1, model, **options)
        group.v = initial
        state = StateMonitor(group, "v", record=True)
        spike = SpikeMonitor(group) if spiking else None
        network = Network(group, state, *([spike] if spike is not None else []))
        return network, group, state, spike

    def run_model(self, model, name):
        source, output = self.directory / f"{name}.json", self.directory / name
        source.write_text(json.dumps(model))
        # Only the executable and IR are used. There is no Python on PATH.
        result = subprocess.run([str(self.runner), str(source), str(output)], env={**os.environ, "PATH": ""}, text=True, capture_output=True)
        return result, output

    def test_differential_models_and_python_free_replay(self):
        cases = [
            ("lif", {}, 100 * ms),
            ("decay", dict(model="dv/dt = -v/tau : volt", initial=5 * mV, spiking=False), 10 * ms),
            ("time", dict(model="dv/dt = t*rate : 1", namespace={"rate": 10000 / second**2}, spiking=False), 10 * ms),
            ("boundary", dict(model="dv/dt = drive/tau : 1", threshold="v >= 1", namespace={"drive": 1.0, "tau": 1 * ms}, dt=1 * ms), 4 * ms),
        ]
        for name, options, duration in cases:
            with self.subTest(model=name):
                network, group, state, spike = self.make_network(**options)
                model = export_network(network, duration, self.directory / f"{name}-export.json")
                result, output = self.run_model(model, name)
                self.assertEqual(result.returncode, 0, result.stderr)
                network.run(duration)
                loaded = load_results(model, output)["populations"][0]
                np.testing.assert_allclose(loaded["trace"]["v"][:, 0],
                                           np.asarray(state.v[0]), rtol=1e-12,
                                           atol=1e-14)
                np.testing.assert_allclose(loaded["times"], state.t / second,
                                           rtol=0, atol=1e-15)
                np.testing.assert_allclose(loaded["states"]["v"][0],
                                           float(group.v[0]), rtol=1e-12,
                                           atol=1e-14)
                expected = np.rint(spike.t / group.clock.dt).astype(int) if spike is not None else []
                np.testing.assert_array_equal(
                    np.rint(loaded["spike_times"] /
                            float(group.clock.dt / second)),
                    expected)
                if name == "boundary":
                    np.testing.assert_array_equal(expected, [0, 1, 2, 3])

    def test_unsupported_brian_semantics_rejected_before_export(self):
        network, group, _, _ = self.make_network()
        group.state_updater.method_choice = "gsl_msadams"
        path = self.directory / "unsupported-method.json"
        with self.assertRaisesRegex(NotImplementedError, "population.method"):
            export_network(network, 10 * ms, path)
        self.assertFalse(path.exists())
        network, _, state, _ = self.make_network()
        state.when = "end"
        model = export_network(network, 10 * ms, self.directory / "end.json")
        node = next(node for node in model["definition"]["schedule"]["nodes"]
                    if node["operation"] == "state_monitor")
        self.assertEqual(node["when"], "end")

    def test_capability_report_aggregates_independent_model_issues(self):
        network, group, state, _ = self.make_network()
        group.state_updater.method_choice = "gsl_msadams"
        state.when = "end"
        network.add(StateMonitor(group, "v", record=True, when="start"))
        group.run_regularly("v = v", dt=0.2*ms)
        report = capability_report(network, 10 * ms)
        self.assertFalse(report.supported)
        codes = {issue.code for issue in report.issues}
        self.assertTrue({"population.method", "monitor.state.schedule"} <= codes)
        as_dict = report.to_dict()
        self.assertEqual(as_dict["schema"], "b2-capability-report-v1")
        self.assertEqual(as_dict["summary"]["populations"], 1)
        path = self.directory / "aggregate.json"
        with self.assertRaises(CapabilityError) as caught:
            export_network(network, 10 * ms, path)
        self.assertGreaterEqual(len(caught.exception.report.issues), 2)
        self.assertIn("population.method", str(caught.exception))
        self.assertFalse(path.exists())

    def test_explicit_exact_matches_numpy_and_runner(self):
        group = NeuronGroup(
            2, "dv/dt=(drive-v)/tau : volt",
            threshold="v > 1*mV", reset="v = 0*mV", method="exact",
            dt=0.1*ms, namespace={"drive": 1.5*mV, "tau": 10*ms})
        group.v = [0.5, 0.75]*mV
        state = StateMonitor(group, "v", record=True)
        spike = SpikeMonitor(group)
        network = Network(group, state, spike)
        model = export_network(
            network, 20 * ms, self.directory / "explicit-exact.json")
        result, output = self.run_model(model, "explicit-exact")
        self.assertEqual(result.returncode, 0, result.stderr)
        network.run(20 * ms)
        loaded = load_results(model, output)["populations"][0]
        np.testing.assert_allclose(
            loaded["trace"]["v"][:, 0], np.asarray(state.v[0]),
            rtol=1e-12, atol=1e-14)
        np.testing.assert_array_equal(loaded["indices"], np.asarray(spike.i))
        np.testing.assert_allclose(
            loaded["states"]["v"], np.asarray(group.v[:]),
            rtol=1e-12, atol=1e-14)

    def test_explicit_linear_independent_and_rk2_match_numpy_and_runner(self):
        for method in ("linear", "independent", "rk2"):
            with self.subTest(method=method):
                network, group, state, _ = self.make_network(
                    method=method, spiking=False, initial=.25)
                model = export_network(
                    network, 2 * ms,
                    self.directory / f"explicit-{method}.json")
                result, output = self.run_model(model, f"explicit-{method}")
                self.assertEqual(result.returncode, 0, result.stderr)
                network.run(2 * ms)
                loaded = load_results(model, output)["populations"][0]
                np.testing.assert_allclose(
                    loaded["trace"]["v"][:, 0], np.asarray(state.v[0]),
                    rtol=3e-12, atol=1e-14)
                np.testing.assert_allclose(
                    loaded["states"]["v"], np.asarray(group.v[:]),
                    rtol=3e-12, atol=1e-14)

    def test_independent_rejects_cross_state_ode_dependencies(self):
        group = NeuronGroup(
            1,
            "dv/dt=(-v+w)/(10*ms) : 1\n"
            "dw/dt=(v-w)/(20*ms) : 1",
            method="independent", dt=.1*ms)
        report = capability_report(Network(group), 1*ms)
        self.assertFalse(report.supported)
        self.assertEqual([issue.code for issue in report.issues],
                         ["population.method"])

    def test_supported_capability_report_does_not_build_or_run(self):
        network, _, state, _ = self.make_network()
        report = capability_report(network, 1 * ms)
        self.assertTrue(report.supported)
        self.assertEqual(report.issues, ())
        self.assertEqual(len(state.t), 0)

    def test_gsl_capability_accepts_supported_method_options(self):
        network, _, state, _ = self.make_network(
            method="gsl_rk2",
            method_options={
                "absolute_error": 1e-9,
                "save_failed_steps": True,
                "save_step_count": True,
            },
        )
        report = capability_report(network, 1 * ms)
        self.assertTrue(report.supported, report.format_text())
        self.assertEqual(report.issues, ())
        self.assertEqual(len(state.t), 0)

    def test_stochastic_differential_equation_is_supported(self):
        group = NeuronGroup(
            2,
            "dv/dt = -v/(10*ms) + xi/sqrt(ms) : 1",
            method="euler",
            dt=0.1 * ms,
        )
        network = Network(group)
        report = capability_report(network, 1 * ms)
        self.assertTrue(report.supported, report.format_text())

    def test_multiplicative_stochastic_equation_uses_default_heun(self):
        group = NeuronGroup(
            2,
            "dv/dt = -v/(10*ms) + v*xi/sqrt(ms) : 1",
            dt=0.1 * ms,
        )
        report = capability_report(Network(group), 1 * ms)
        self.assertTrue(report.supported, report.format_text())

    def test_poisson_source_accepts_contained_outgoing_synapse(self):
        source = PoissonGroup(2, 0 * Hz, dt=0.1 * ms, name="source")
        target = NeuronGroup(2, "v : 1", dt=0.1 * ms, name="target")
        edges = Synapses(
            source, target, on_pre="v_post += 1", clock=source.clock, name="edges"
        )
        edges.connect(j="i")
        source.contained_objects.append(edges)
        network = Network(source, target, edges)
        report = capability_report(network, 1 * ms)
        self.assertTrue(report.supported, report.format_text())

    def test_integer_summed_expression_casts_to_float_target(self):
        source = NeuronGroup(2, "x : 1", dt=0.1 * ms, name="source")
        target = NeuronGroup(1, "total : 1", dt=0.1 * ms, name="target")
        source.x = [0, 1]
        edges = Synapses(
            source,
            target,
            "total_post = int(x_pre > 0) : 1 (summed)",
            clock=source.clock,
            name="edges",
        )
        edges.connect()
        model = export_network(
            Network(source, target, edges),
            1 * ms,
            self.directory / "integer-summed.json",
        )
        summed = next(
            code
            for code in model["definition"]["synapses"][0]["code_objects"]
            if code["kind"] == "summed_variable"
        )
        self.assertEqual(summed["vector"][-1]["dtype"], "f64")

    def test_brian_default_method_selection_lowers_exact_and_euler(self):
        cases = {
            "exact": "dv/dt = -v/(2*ms) : 1",
            "euler": "dv/dt = -v**2/(2*ms) : 1",
        }
        for expected_method, equation in cases.items():
            with self.subTest(method=expected_method):
                group = NeuronGroup(2, equation, dt=0.1 * ms)
                group.v = [0.5, 1.0]
                monitor = StateMonitor(group, "v", record=True)
                network = Network(group, monitor)
                model = export_network(
                    network, 0.5 * ms,
                    self.directory / f"automatic-{expected_method}.json")
                result, output = self.run_model(model, f"automatic-{expected_method}")
                self.assertEqual(result.returncode, 0, result.stderr)
                network.run(0.5 * ms)
                loaded = load_results(model, output)["populations"][0]
                np.testing.assert_allclose(
                    loaded["trace"]["v"].T, monitor.v,
                    rtol=1e-12, atol=1e-14)
                np.testing.assert_allclose(
                    loaded["states"]["v"], group.v[:],
                    rtol=1e-12, atol=1e-14)

    def test_result_dump_is_memmapped_and_rejects_bad_framing(self):
        network, _, _, _ = self.make_network()
        model = export_network(network, 1 * ms, self.directory / "dump-export.json")
        result, output = self.run_model(model, "dump-good")
        self.assertEqual(result.returncode, 0, result.stderr)
        loaded = load_results(model, output)
        self.assertFalse(loaded["populations"][0]["trace"]["v"].flags.owndata)
        original = (output / "results.bin").read_bytes()
        metadata = (output / "summary.json").read_bytes()
        corruptions = {
            "magic": b"X" + original[1:],
            "version": original[:8] + (4).to_bytes(4, "little") + original[12:],
            "truncated": original[:-1],
            "trailing": original + b"x",
            "shape": original[:40] + (2).to_bytes(8, "little") + original[48:],
        }
        for name, data in corruptions.items():
            with self.subTest(name=name):
                directory = self.directory / f"dump-{name}"
                directory.mkdir()
                (directory / "results.bin").write_bytes(data)
                (directory / "summary.json").write_bytes(metadata)
                with self.assertRaisesRegex(RuntimeError, "inconsistent results"):
                    load_results(model, directory)

    def test_brian_unit_check_is_preserved(self):
        network, _, _, _ = self.make_network(namespace={"drive": 1.5, "tau": 10 * mV})
        with self.assertRaises(DimensionMismatchError):
            export_network(network, 10 * ms, self.directory / "bad-units.json")

    def test_invalid_ir_rejected_before_output(self):
        network, _, _, _ = self.make_network()
        original = export_network(network, 10 * ms, self.directory / "valid.json")
        mutations = [
            (("schema",), "future-version"),
            (("definition", "rng_algorithm"), "unknown-rng"),
            (("instance", "rng_seed"), -1),
            (("definition", "populations", 0, "states", 0, "dtype"), "f32"),
            (("definition", "schedule"), ["state_update", "record_start"]),
            (("instance", "populations", 0, "initial_state", "v", 0), "7ff8000000000000"),
            (("instance", "populations", 0, "parameters"), {}),
            (("definition", "populations", 0, "dt"), "0000000000000000"),
            (("definition", "populations", 0, "steps"), 1_000_001),
            (("definition", "populations", 0, "code_objects", 0, "vector", 0, "value"), {"op": "load", "name": "missing"}),
            (("definition", "populations", 0, "code_objects", 0, "vector", 0, "target"), "dt"),
            (("definition", "populations", 0, "code_objects", 1, "vector", 0, "value", "left"), {"op": "load", "name": "_v"}),
            (("definition", "populations", 0, "code_objects", 2, "vector", 0, "value"), {"op": "call_python"}),
            (("definition", "populations", 0, "code_objects", 2, "vector", 0, "value"),
             {"op": "exp", "arg": {"op": "boolean", "value": True}}),
            (("definition", "populations", 0, "code_objects", 0, "vector", 0, "dtype"), "bool"),
            (("definition", "populations", 0, "code_objects", 0, "vector", 0,
              "dimensions"), [0, 0, 1, 0, 0, 0, 0]),
            (("definition", "populations", 0, "parameters", 1,
              "dimensions"), [0, 0, 0, 0, 0, 0, 0]),
            (("definition", "populations", 0, "code_objects", 0, "effects", "writes"), []),
            (("definition", "populations", 0, "code_objects", 1, "iteration_domain"), "spiking_neurons"),
            (("definition", "populations", 0, "code_objects", 0, "scalar"), [{"target": "temp", "dtype": "f64", "value": {"op": "load", "name": "v"}}]),
            (("definition", "populations", 0, "monitor", "record"), [1]),
            (("definition", "populations", 0, "monitor", "variables"), ["missing"]),
            (("instance", "populations", 0, "initial_state", "v"), []),
            (("instance", "neuron_count"), 0),
            (("schema",), "b2ir-gate0-probe-v0"),
            (("schema",), "b2ir-gate0-probe-v1"),
            (("schema",), "b2ir-gate0-probe-v2"),
            (("schema",), "b2ir-gate0-probe-v7"),
            (("unknown_field",), True),
        ]
        for index, (path, value) in enumerate(mutations):
            with self.subTest(path=path):
                model = copy.deepcopy(original)
                node = model
                for key in path[:-1]:
                    node = node[key]
                node[path[-1]] = value
                result, output = self.run_model(model, f"invalid-{index}")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("b2-runner:", result.stderr)
                self.assertFalse(output.exists())

    def test_logical_index_and_tick_types_are_explicit_and_tamper_evident(self):
        group = NeuronGroup(
            3,
            """dv/dt=(held + int(v >= 0) + int(-1.75) +
                       timestep(t, dt))/second : 1
               held = v + i + t/second : 1 (constant over dt)""",
            method="euler", dt=1*ms)
        model = export_network(
            Network(group), 2*ms, self.directory / "logical-valid.json")
        code = model["definition"]["populations"][0]["code_objects"]

        def expressions(node):
            if isinstance(node, dict):
                if "op" in node:
                    yield node
                for value in node.values():
                    yield from expressions(value)
            elif isinstance(node, list):
                for value in node:
                    yield from expressions(value)

        nodes = list(expressions(code))
        self.assertIn(
            "tick", {statement["dtype"] for item in code
                     for statement in item["scalar"]})
        self.assertTrue(any(node["op"] == "timestep" for node in nodes))
        self.assertTrue(any(node["op"] == "tick_offset" for node in nodes))
        self.assertTrue(any(node["op"] == "index_to_f64" for node in nodes))
        self.assertTrue(any(node["op"] == "tick_to_f64" for node in nodes))

        valid = subprocess.run(
            [str(self.runner), "--validate", str(self.directory / "logical-valid.json")],
            capture_output=True, text=True)
        self.assertEqual(valid.returncode, 0, valid.stderr)
        for op in ["index_to_f64", "tick_to_f64"]:
            with self.subTest(op=op):
                invalid = copy.deepcopy(model)
                target = next(node for node in expressions(invalid) if node.get("op") == op)
                target["arg"] = {"op": "literal", "bits": "0000000000000000"}
                source = self.directory / f"logical-invalid-{op}.json"
                source.write_text(json.dumps(invalid))
                checked = subprocess.run(
                    [str(self.runner), "--validate", str(source)],
                    capture_output=True, text=True)
                self.assertNotEqual(checked.returncode, 0)
                self.assertIn(op, checked.stderr)

        invalid = copy.deepcopy(model)
        timestep_node = next(
            node for node in expressions(invalid) if node.get("op") == "timestep")
        timestep_node["dt"] = {"op": "load", "name": "N"}
        source = self.directory / "logical-invalid-timestep.json"
        source.write_text(json.dumps(invalid))
        checked = subprocess.run(
            [str(self.runner), "--validate", str(source)],
            capture_output=True, text=True)
        self.assertNotEqual(checked.returncode, 0)
        self.assertIn("timestep arguments", checked.stderr)

        invalid = copy.deepcopy(model)
        tick_offset = next(
            node for node in expressions(invalid) if node.get("op") == "tick_offset")
        tick_offset["offset"] = 2**53 + 1
        source = self.directory / "logical-invalid-tick-offset.json"
        source.write_text(json.dumps(invalid))
        checked = subprocess.run(
            [str(self.runner), "--validate", str(source)],
            capture_output=True, text=True)
        self.assertNotEqual(checked.returncode, 0)
        self.assertIn("tick_offset", checked.stderr)

    def test_schedule_graph_is_canonical_and_tamper_evident(self):
        network, _, _, _ = self.make_network()
        original = export_network(
            network, 10 * ms, self.directory / "schedule-valid.json")
        self.assertEqual(original["schema"], "b2ir-v1")
        schedule = original["definition"]["schedule"]
        self.assertEqual(
            schedule["base_slots"],
            ["start", "groups", "thresholds", "synapses", "resets", "end"],
        )
        self.assertEqual(
            schedule["slots"],
            [expanded for base in schedule["base_slots"]
             for expanded in (f"before_{base}", base, f"after_{base}")],
        )
        slot = {name: index for index, name in enumerate(schedule["slots"])}
        keys = [(slot[node["when"]], node["order"], node["name"], node["id"])
                for node in schedule["nodes"]]
        self.assertEqual(keys, sorted(keys))
        self.assertEqual(
            {node["operation"] for node in schedule["nodes"]},
            {"code_object", "state_monitor", "spike_monitor"},
        )
        threshold = next(
            node for node in schedule["nodes"]
            if node["operation"] == "code_object"
            and original["definition"]["populations"][0]["code_objects"]
            [node["item_index"]]["kind"] == "threshold")
        spike_monitor = next(
            node for node in schedule["nodes"]
            if node["operation"] == "spike_monitor")
        self.assertIn("population/0/event/spike", threshold["effects"]["writes"])
        self.assertIn(threshold["id"], spike_monitor["dependencies"])

        mutations = {}
        missing = copy.deepcopy(original)
        missing["definition"]["schedule"]["nodes"].pop()
        mutations["missing-node"] = missing
        bad_effect = copy.deepcopy(original)
        bad_threshold = next(
            node for node in bad_effect["definition"]["schedule"]["nodes"]
            if node["id"] == threshold["id"])
        bad_threshold["effects"]["writes"] = []
        mutations["forged-effect"] = bad_effect
        bad_dependency = copy.deepcopy(original)
        bad_spike_monitor = next(
            node for node in bad_dependency["definition"]["schedule"]["nodes"]
            if node["id"] == spike_monitor["id"])
        bad_spike_monitor["dependencies"] = []
        mutations["forged-dependency"] = bad_dependency
        bad_slots = copy.deepcopy(original)
        bad_slots["definition"]["schedule"]["slots"][0:3] = reversed(
            bad_slots["definition"]["schedule"]["slots"][0:3])
        mutations["forged-expanded-slots"] = bad_slots
        bad_clock = copy.deepcopy(original)
        bad_clock["definition"]["schedule"]["nodes"][0]["clock"] = len(
            bad_clock["definition"]["clocks"])
        mutations["forged-node-clock"] = bad_clock
        bad_clock_interval = copy.deepcopy(original)
        bad_clock_interval["run"]["clocks"][0]["steps"] += 1
        mutations["forged-clock-interval"] = bad_clock_interval

        for name, model in mutations.items():
            with self.subTest(mutation=name):
                result, output = self.run_model(model, f"schedule-{name}")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("b2-runner:", result.stderr)
                self.assertFalse(output.exists())

        # A threshold can intentionally omit its reset and/or SpikeMonitor, but
        # the inverse relationships remain invalid in untrusted IR.
        population = original["definition"]["populations"][0]
        for name, code_objects in {
                "monitor-without-threshold": [population["code_objects"][0]],
                "reset-without-threshold": [population["code_objects"][0],
                                            population["code_objects"][2]],
        }.items():
            with self.subTest(relationship=name):
                model = copy.deepcopy(original)
                model["definition"]["populations"][0]["code_objects"] = copy.deepcopy(
                    code_objects)
                result, output = self.run_model(model, f"invalid-{name}")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("b2-runner:", result.stderr)
                self.assertFalse(output.exists())

    def test_canonical_layer_hashes_and_v34_through_v37_migration(self):
        network, _, _, _ = self.make_network()
        model = export_network(
            network, 1 * ms, self.directory / "canonical-v1.json")
        self.assertEqual(
            model["protocol"]["canonical_encoding"],
            "b2ir-canonical-json-v1")
        self.assertEqual(model["protocol"]["layers"], layer_hashes(model))
        self.assertEqual(
            canonical_bytes(model["definition"]),
            canonical_bytes(json.loads(json.dumps(model["definition"]))))

        tampered = copy.deepcopy(model)
        tampered["instance"]["rng_seed"] += 1
        result, output = self.run_model(tampered, "canonical-tampered")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("canonical layer hash mismatch", result.stderr)
        self.assertFalse(output.exists())

        predecessor = copy.deepcopy(model)
        predecessor["schema"] = "b2ir-gate0-probe-v34"
        del predecessor["protocol"]
        for population in predecessor["definition"]["populations"]:
            population.pop("linked_variables")
        result, output = self.run_model(predecessor, "canonical-v34")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((output / "results.bin").is_file())

        predecessor = copy.deepcopy(model)
        predecessor["schema"] = "b2ir-gate0-probe-v35"
        for population in predecessor["definition"]["populations"]:
            population.pop("linked_variables")
        predecessor["protocol"]["layers"] = layer_hashes(predecessor)
        result, output = self.run_model(predecessor, "canonical-v35")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((output / "results.bin").is_file())

        predecessor = copy.deepcopy(model)
        predecessor["schema"] = "b2ir-gate0-probe-v36"
        predecessor["protocol"]["layers"] = layer_hashes(predecessor)
        result, output = self.run_model(predecessor, "canonical-v36")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((output / "results.bin").is_file())

        predecessor = copy.deepcopy(model)
        predecessor["schema"] = "b2ir-gate0-probe-v37"
        predecessor["protocol"]["layers"] = layer_hashes(predecessor)
        result, output = self.run_model(predecessor, "canonical-v37")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((output / "results.bin").is_file())

    def test_custom_network_schedule_matches_brian_reference_order(self):
        network, group, state, spike = self.make_network(
            model="dv/dt = 1/ms : 1", initial=0,
            threshold="v >= 1", reset="v = 0", dt=1*ms,
        )
        network.schedule = [
            "start", "thresholds", "groups", "synapses", "resets", "end",
        ]
        model = export_network(
            network, 4*ms, self.directory / "custom-schedule.json")
        self.assertEqual(
            model["definition"]["schedule"]["base_slots"], network.schedule)
        result, output = self.run_model(model, "custom-schedule")
        self.assertEqual(result.returncode, 0, result.stderr)
        network.run(4*ms)
        loaded = load_results(model, output)["populations"][0]
        np.testing.assert_array_equal(
            loaded["trace"]["v"][:, 0], np.asarray(state.v[0]))
        np.testing.assert_array_equal(
            loaded["indices"], np.asarray(spike.i))
        np.testing.assert_allclose(
            loaded["spike_times"], np.asarray(spike.t/second),
            rtol=0, atol=1e-15)
        np.testing.assert_array_equal(loaded["states"]["v"], np.asarray(group.v))

    def test_same_clock_run_regularly_matches_brian_and_enters_effect_dag(self):
        group = NeuronGroup(
            3, "dv/dt=(i+1.0)/ms:1\nx:1", method="euler", dt=1*ms,
            name="regular_group")
        regular = group.run_regularly(
            "x += v", when="end", order=2, name="regular_update")
        monitor = StateMonitor(group, ["v", "x"], record=True)
        network = Network(group, monitor)
        model = export_network(
            network, 4*ms, self.directory / "run-regularly.json")
        code_index = next(
            index for index, code in enumerate(
                model["definition"]["populations"][0]["code_objects"])
            if code["kind"] == "run_regularly")
        code = model["definition"]["populations"][0]["code_objects"][code_index]
        self.assertEqual(
            (code["name"], code["when"], code["order"], code["effects"]),
            (regular.name, "end", 2,
             {"reads": ["v", "x"], "writes": ["x"]}),
        )
        node = next(
            node for node in model["definition"]["schedule"]["nodes"]
            if node["owner_kind"] == "population"
            and node["item_index"] == code_index)
        self.assertEqual((node["when"], node["order"]), ("end", 2))
        self.assertIn("population/0/state/v", node["effects"]["reads"])
        self.assertIn("population/0/state/x", node["effects"]["writes"])

        result, output = self.run_model(model, "run-regularly")
        self.assertEqual(result.returncode, 0, result.stderr)
        network.run(4*ms)
        loaded = load_results(model, output)["populations"][0]
        np.testing.assert_allclose(
            loaded["trace"]["v"], np.asarray(monitor.v).T,
            rtol=0, atol=1e-14)
        np.testing.assert_allclose(
            loaded["trace"]["x"], np.asarray(monitor.x).T,
            rtol=0, atol=1e-14)
        np.testing.assert_allclose(
            loaded["states"]["v"], np.asarray(group.v), rtol=0, atol=1e-14)
        np.testing.assert_allclose(
            loaded["states"]["x"], np.asarray(group.x), rtol=0, atol=1e-14)

    def test_independent_run_regularly_clock_matches_brian(self):
        group = NeuronGroup(
            2, "dv/dt=1/ms:1\nx:1", method="euler", dt=1*ms,
            name="independent_clock_group")
        regular = group.run_regularly(
            "x += v", dt=2*ms, when="end", name="independent_regular")
        monitor = StateMonitor(group, ["v", "x"], record=True)
        network = Network(group, monitor)
        model = export_network(
            network, 4*ms, self.directory / "independent-clock.json")
        definition = model["definition"]
        self.assertEqual(len(definition["clocks"]), 2)
        code = next(code for code in definition["populations"][0]["code_objects"]
                    if code["name"] == regular.name)
        self.assertNotEqual(code["clock"], definition["populations"][0]["clock"])
        self.assertEqual(
            model["run"]["clocks"][code["clock"]],
            {"start_tick": 0, "steps": 2},
        )

        result, output = self.run_model(model, "independent-clock")
        self.assertEqual(result.returncode, 0, result.stderr)
        network.run(4*ms)
        loaded = load_results(model, output)["populations"][0]
        for name, expected in {"v": monitor.v, "x": monitor.x}.items():
            np.testing.assert_allclose(
                loaded["trace"][name], np.asarray(expected).T,
                rtol=0, atol=1e-14)
        np.testing.assert_allclose(
            loaded["states"]["x"], np.asarray(group.x), rtol=0, atol=1e-14)

    def test_subgroup_run_regularly_masks_writes_and_uses_local_indices(self):
        group = NeuronGroup(6, "x:1\ny:1", dt=1*ms, name="parent_group")
        excitatory = group[:4]
        inhibitory = group[4:]
        excitatory.run_regularly(
            "x = i + N; y += 1", name="excitatory_drive")
        inhibitory.run_regularly(
            "x = 10 + i + N; y += 3", name="inhibitory_drive")
        monitor = StateMonitor(group, ["x", "y"], record=True)
        network = Network(group, monitor)
        model = export_network(
            network, 3*ms, self.directory / "subgroup-run-regularly.json")
        definition = model["definition"]["populations"][0]
        masks = [parameter for parameter in definition["parameters"]
                 if parameter["name"].startswith("b2_subgroup_mask_")]
        indices = [parameter for parameter in definition["parameters"]
                   if parameter["name"].startswith("b2_subgroup_index_")]
        self.assertEqual((len(masks), len(indices)), (2, 2))
        regular = [code for code in definition["code_objects"]
                   if code["kind"] == "run_regularly"]
        self.assertEqual(len(regular), 2)
        for code in regular:
            guards = {statement["condition"] for statement in code["vector"]
                      if statement["target"] in {"x", "y"}}
            self.assertEqual(len(guards), 1)
            self.assertTrue(next(iter(guards)).startswith("b2_subgroup_mask_"))

        result, output = self.run_model(model, "subgroup-run-regularly")
        self.assertEqual(result.returncode, 0, result.stderr)
        network.run(3*ms)
        loaded = load_results(model, output)["populations"][0]
        for name, expected in {"x": monitor.x, "y": monitor.y}.items():
            np.testing.assert_array_equal(
                loaded["trace"][name], np.asarray(expected).T)
            np.testing.assert_array_equal(
                loaded["states"][name], np.asarray(group.variables[name].get_value()))
        np.testing.assert_array_equal(group.x[:], [4, 5, 6, 7, 12, 13])
        np.testing.assert_array_equal(group.y[:], [3, 3, 3, 3, 9, 9])

    def test_runtime_numeric_failure_does_not_write_result_files(self):
        network, _, _, _ = self.make_network(threshold="True")
        model = export_network(network, 1 * ms, self.directory / "overflow-model.json")
        huge = {"op": "literal", "bits": "7fefffffffffffff"}
        failures = {
            "overflow": {"op": "mul", "left": huge, "right": huge},
            "domain": {"op": "log", "arg": {"op": "literal",
                                               "bits": "bff0000000000000"}},
        }
        for name, expression in failures.items():
            with self.subTest(name=name):
                current = copy.deepcopy(model)
                current["definition"]["populations"][0]["code_objects"][-1]["vector"][0]["value"] = expression
                attach_protocol(current)
                result, output = self.run_model(current, name)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("non-finite numeric value", result.stderr)
                # No result dump is created until the simulation has succeeded.
                for filename in ["results.bin", "summary.json"]:
                    self.assertFalse((output / filename).exists())


if __name__ == "__main__":
    unittest.main()
