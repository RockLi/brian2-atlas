"""Static on_pre topology, event ordering, delay and standalone conformance."""

import copy
import json
import os
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
from brian2_rust.native import (  # noqa: E402
    target_parallel_pathway,
    target_parallel_plan,
)
from brian2_rust.results import load_results  # noqa: E402


class SynapsesTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.previous_device = b.get_device()
        self.previous_target = b.prefs.codegen.target
        self.device = all_devices["rust_standalone"]
        self.runner = Path(os.environ.get(
            "B2_RUNNER", str(ROOT / "target/release/b2-runner")))
        self.select_rust()

    def select_rust(self):
        self.device.reinit()
        b.set_device("rust_standalone", runner=self.runner)

    def tearDown(self):
        b.set_device(self.previous_device)
        b.prefs.codegen.target = self.previous_target
        self.device.reinit()

    @staticmethod
    def make_group(size=3, model="dv/dt=0*Hz : 1\ndx/dt=0*Hz : 1"):
        return b.NeuronGroup(size, model, threshold="v>1", reset="v=0",
                             dt=1*b.ms, method="euler")

    @staticmethod
    def network(group, synapse):
        state = b.StateMonitor(group, ["v", "x"], record=True)
        spikes = b.SpikeMonitor(group)
        return b.Network(group, synapse, state, spikes), state, spikes

    def test_target_partition_rejects_recurrent_pre_read_post_write_alias(self):
        synapse = {
            "states": [],
            "pre_state_aliases": {"v_pre": "v"},
            "post_state_aliases": {"v_post": "v", "x_post": "x"},
        }
        hazard = {"effects": {"reads": ["v_pre"], "writes": ["v_post"]}}
        independent = {"effects": {"reads": ["v_pre"], "writes": ["x_post"]}}
        self.assertFalse(target_parallel_pathway(
            synapse, hazard, same_population=True))
        self.assertTrue(target_parallel_pathway(
            synapse, hazard, same_population=False))
        self.assertTrue(target_parallel_pathway(
            synapse, independent, same_population=True))

    def test_boolean_synapse_state_matches_reference_aot_and_numpy(self):
        results = []
        for backend in ["reference", "aot", "numpy"]:
            self.device.reinit()
            b.start_scope()
            if backend == "numpy":
                b.set_device("runtime")
                b.prefs.codegen.target = "numpy"
            else:
                b.set_device(
                    "rust_standalone", runner=self.runner, engine=backend,
                    directory=self.directory / f"bool-synapse-{backend}")
            source = b.NeuronGroup(
                2, "dv/dt=0*Hz : 1", threshold="True", reset="v=v",
                method="euler", dt=1*b.ms)
            target = b.NeuronGroup(
                2, "dv/dt=0*Hz : 1", method="euler", dt=1*b.ms)
            synapse = b.Synapses(
                source, target, "enabled : boolean",
                on_pre="enabled = not enabled\nv_post += int(enabled)",
                clock=source.clock)
            synapse.connect(i=[0, 1], j=[1, 0])
            synapse.enabled = [True, False]
            b.Network(source, target, synapse).run(3*b.ms)
            results.append((np.asarray(synapse.enabled[:]).copy(),
                            np.asarray(target.v[:]).copy()))
            if backend != "numpy":
                model = json.loads(
                    (self.device.last_run_directory / "model.json").read_text())
                self.assertEqual(
                    model["definition"]["synapses"][0]["states"][0]["dtype"],
                    "bool")
                loaded = load_results(
                    model, self.device.last_run_directory / "rust")
                self.assertEqual(
                    loaded["synapses"][0]["states"]["enabled"].dtype,
                    np.dtype(np.bool_))
        for actual in results[1:]:
            np.testing.assert_array_equal(actual[0], results[0][0])
            np.testing.assert_array_equal(actual[1], results[0][1])

    def test_integer_synapse_state_preserves_i64_and_u64(self):
        results = []
        loaded_result = None
        for backend in ["reference", "aot"]:
            self.device.reinit()
            b.start_scope()
            b.set_device(
                "rust_standalone", runner=self.runner, engine=backend,
                directory=self.directory / f"integer-synapse-{backend}")
            source = b.NeuronGroup(
                2, "dv/dt=0*Hz : 1", threshold="True", reset="v=v",
                method="euler", dt=1*b.ms)
            target = b.NeuronGroup(
                2, "dv/dt=0*Hz : 1", method="euler", dt=1*b.ms)
            synapse = b.Synapses(
                source, target, "signed : integer\nunsigned : integer",
                on_pre=("signed += 1\nunsigned += 2\n"
                        "v_post += int(signed > 9007199254740994)"),
                clock=source.clock,
                dtype={"signed": np.int64, "unsigned": np.uint64},
            )
            synapse.connect(i=[0, 1], j=[1, 0])
            synapse.signed = [9007199254740993, 9007199254740994]
            synapse.unsigned = [2**63 + 3, 2**63 + 4]
            b.Network(source, target, synapse).run(3*b.ms)
            results.append({
                "signed": np.asarray(synapse.signed[:]).copy(),
                "unsigned": np.asarray(synapse.unsigned[:]).copy(),
                "target": np.asarray(target.v[:]).copy(),
            })
            if backend == "aot":
                model = json.loads(
                    (self.device.last_run_directory / "model.json").read_text())
                loaded_result = load_results(
                    model, self.device.last_run_directory / "rust")
        for actual in results[1:]:
            for name in results[0]:
                np.testing.assert_array_equal(actual[name], results[0][name])
        np.testing.assert_array_equal(
            results[0]["signed"], [9007199254740996, 9007199254740997])
        np.testing.assert_array_equal(
            results[0]["unsigned"], [2**63 + 9, 2**63 + 10])
        np.testing.assert_array_equal(results[0]["target"], [3, 2])
        loaded_states = loaded_result["synapses"][0]["states"]
        self.assertEqual(loaded_states["signed"].dtype, np.dtype("<i8"))
        self.assertEqual(loaded_states["unsigned"].dtype, np.dtype("<u8"))

    def test_shared_target_plan_applies_work_and_cross_pathway_guards(self):
        synapse = {
            "states": [],
            "pre_state_aliases": {"v_pre": "v"},
            "post_state_aliases": {"v_post": "v", "x_post": "x"},
        }
        reads_v = {"effects": {"reads": ["v_pre"], "writes": ["x_post"]}}
        writes_v = {"effects": {"reads": [], "writes": ["v_post"]}}

        small = target_parallel_plan([(synapse, reads_v, True, 2_047)])
        self.assertFalse(small.eligible)
        self.assertEqual(small.reason, "event batch too small")

        safe = target_parallel_plan([(synapse, reads_v, True, 2_048)])
        self.assertTrue(safe.eligible)
        self.assertEqual(safe.reason, "target-owned")

        dependency = target_parallel_plan([
            (synapse, reads_v, True, 1_024),
            (synapse, writes_v, True, 1_024),
        ])
        self.assertFalse(dependency.eligible)
        self.assertEqual(dependency.reason, "cross-pathway recurrent dependency")

    def test_connect_append_and_source_then_edge_order_without_runtime_fallback(self):
        with patch.object(RuntimeDevice, "code_object", side_effect=AssertionError("runtime fallback")):
            group = self.make_group()
            group.v = [1.1, 1.2, 0]
            synapse = b.Synapses(group, group, "w : 1",
                                on_pre="factor=10.0; x_post=factor*x_post+w",
                                clock=group.clock)
            synapse.connect(i=[1, 0], j=2)
            synapse.connect(i=[1, 0], j=[2, 2])
            synapse.w = [1, 2, 3, 4]
            net, _, spikes = self.network(group, synapse)
            net.run(2*b.ms)
        np.testing.assert_array_equal(synapse.i[:], [1, 0, 1, 0])
        np.testing.assert_array_equal(synapse.j[:], [2, 2, 2, 2])
        np.testing.assert_array_equal(synapse.w[:], [1, 2, 3, 4])
        np.testing.assert_array_equal(synapse.variables["N_outgoing"].get_value(), [2, 2, 0])
        np.testing.assert_array_equal(synapse.variables["N_incoming"].get_value(), [0, 0, 4])
        # Source spikes are ascending; each source retains connection creation order.
        np.testing.assert_array_equal(group.x[:], [0, 0, 2413])
        np.testing.assert_array_equal(spikes.i[:], [0, 1])

    def test_float32_population_and_synapse_storage_match_backends(self):
        results = []
        for backend in ["reference", "aot", "numpy"]:
            self.device.reinit()
            b.start_scope()
            if backend == "numpy":
                b.set_device("runtime")
                b.prefs.codegen.target = "numpy"
            else:
                b.set_device("rust_standalone", runner=self.runner, engine=backend)
            clock = b.Clock(dt=.1*b.ms)
            source = b.NeuronGroup(
                3, "dv/dt=0*Hz : 1", threshold="v>.5", reset="v=0",
                method="euler", clock=clock, dtype=np.float32,
                name=f"float32_source_{backend}")
            target = b.NeuronGroup(
                2, "dv/dt=-v/(2*ms) : 1", method="euler", clock=clock,
                dtype=np.float32, name=f"float32_target_{backend}")
            source.v = [.6, .1, .7]
            projection = b.Synapses(
                source, target, "w : 1", on_pre="v_post += w; w += .03125",
                clock=clock, dtype=np.float32,
                name=f"float32_projection_{backend}")
            projection.connect()
            projection.w = .125
            monitor = b.StateMonitor(target, "v", record=True)
            b.Network(source, target, projection, monitor).run(.3*b.ms)
            results.append((np.asarray(monitor.v).copy(),
                            np.asarray(target.v[:]).copy(),
                            np.asarray(projection.w[:]).copy()))
            if backend != "numpy":
                model = json.loads(
                    (self.device.last_run_directory / "model.json").read_text())
                synapse = model["definition"]["synapses"][0]
                self.assertEqual(synapse["states"][0]["dtype"], "f32")
                loaded = load_results(model, self.device.last_run_directory / "rust")
                self.assertEqual(
                    loaded["synapses"][0]["states"]["w"].dtype,
                    np.dtype("<f4"))
        for left, right in zip(results[0], results[1], strict=True):
            np.testing.assert_array_equal(left, right)
        for rust, numpy in zip(results[0], results[2], strict=True):
            np.testing.assert_allclose(rust, numpy, rtol=2e-7, atol=1e-7)

    def test_named_custom_event_routes_and_resets_in_global_schedule(self):
        snapshots = []
        for backend in ("reference", "aot", "numpy"):
            if backend != "numpy":
                self.device.reinit()
                b.set_device("rust_standalone", runner=self.runner,
                             engine=backend,
                             directory=self.directory / f"custom-event-{backend}")
            else:
                b.set_device("runtime")
                b.prefs.codegen.target = "numpy"
            source = b.NeuronGroup(
                2, "dv/dt=1/ms : 1", events={"crossing": "v >= 2"},
                dt=1*b.ms, method="euler", dtype=np.float32,
                name="custom_source")
            source.run_on_event("crossing", "v = 0")
            events = b.EventMonitor(
                source, "crossing", variables=["v"],
                name=f"custom_events_{backend}")
            target = b.NeuronGroup(
                2, "dy/dt=0*Hz : 1", dt=1*b.ms, method="euler",
                dtype=np.float32,
                name="custom_target")
            synapse = b.Synapses(
                source, target,
                on_pre={"crossing_path": "y_post += 1"},
                on_event={"crossing_path": "crossing"},
                clock=source.clock, name="custom_synapse")
            synapse.connect(i=[0, 1], j=[0, 1])
            monitor = b.StateMonitor(target, "y", record=True)
            network = b.Network(source, target, synapse, monitor, events)
            if backend == "reference":
                model = lower_network(network, 5*b.ms)
                population = next(item for item in model["definition"]["populations"]
                                  if item["name"] == source.name)
                self.assertEqual(population["events"], ["crossing"])
                event_codes = [code for code in population["code_objects"]
                               if code["kind"] in {"threshold", "reset"}]
                self.assertEqual([code["event_name"] for code in event_codes],
                                 ["crossing", "crossing"])
                pathway = model["instance"]["synapses"][0]["pathways"][0]
                self.assertEqual(pathway["event"], "crossing")
                self.assertEqual(population["event_monitors"], [{
                    "name": events.name, "event": "crossing", "variables": ["v"],
                    "clock": population["clock"], "when": "after_thresholds",
                    "order": 1,
                }])
            network.run(5*b.ms)
            snapshots.append((np.asarray(source.v[:]).copy(),
                              np.asarray(target.y[:]).copy(),
                              np.asarray(monitor.y[:]).copy(),
                              np.asarray(events.i[:]).copy(),
                              np.asarray(events.t[:] / b.ms).copy(),
                              np.asarray(events.v[:]).copy()))
        for rust in snapshots[:2]:
            for actual, numpy in zip(rust, snapshots[2], strict=True):
                np.testing.assert_array_equal(actual, numpy)
        np.testing.assert_array_equal(snapshots[0][1], [2, 2])
        np.testing.assert_array_equal(snapshots[0][3], [0, 1, 0, 1])
        np.testing.assert_array_equal(snapshots[0][4], [1, 1, 3, 3])
        np.testing.assert_array_equal(snapshots[0][5], [2, 2, 2, 2])
        self.assertEqual(snapshots[0][5].dtype, np.dtype(np.float32))

    def test_named_event_delay_survives_segmented_run_boundary(self):
        snapshots = []
        for backend in ("reference", "aot", "numpy"):
            if backend != "numpy":
                self.device.reinit()
                b.set_device("rust_standalone", runner=self.runner,
                             engine=backend,
                             directory=self.directory / f"seg-event-{backend}")
            else:
                b.set_device("runtime")
                b.prefs.codegen.target = "numpy"
            source = b.NeuronGroup(
                1, "dv/dt=1/ms : 1", events={"crossing": "v >= 2"},
                dt=1*b.ms, method="euler", name="seg_event_source")
            source.run_on_event("crossing", "v = 0")
            target = b.NeuronGroup(
                1, "y : 1", dt=1*b.ms, name="seg_event_target")
            pathway = b.Synapses(
                source, target, on_pre={"crossing_path": "y_post += 1"},
                on_event={"crossing_path": "crossing"},
                clock=source.clock, name="seg_event_synapse")
            pathway.connect(i=[0], j=[0])
            pathway.crossing_path.delay = 2*b.ms
            events = b.EventMonitor(source, "crossing")
            network = b.Network(source, target, pathway, events)
            if backend != "numpy":
                network.run(3*b.ms)
                network.run(3*b.ms)
            else:
                network.run(6*b.ms)
            snapshots.append((float(target.y[0]),
                              np.asarray(events.i[:]).copy(),
                              np.asarray(events.t[:] / b.ms).copy()))
        for rust in snapshots[:2]:
            self.assertEqual(rust[0], snapshots[2][0])
            np.testing.assert_array_equal(rust[1], snapshots[2][1])
            np.testing.assert_array_equal(rust[2], snapshots[2][2])
        self.assertEqual(snapshots[0][0], 2)
        np.testing.assert_array_equal(snapshots[0][2], [1, 3, 5])

    def test_aot_keeps_distinct_named_event_routes_separate(self):
        results = []
        for backend in ("reference", "aot", "numpy"):
            self.device.reinit()
            b.start_scope()
            if backend == "numpy":
                b.set_device("runtime")
                b.prefs.codegen.target = "numpy"
            else:
                b.set_device(
                    "rust_standalone", runner=self.runner, engine=backend,
                    directory=self.directory / f"two-events-{backend}")
            source = b.NeuronGroup(
                1, "dx/dt=1/ms : 1",
                events={"first": "x == 1", "second": "x == 2"},
                method="euler", dt=1*b.ms, name="event_source")
            target = b.NeuronGroup(
                1, "y : 1", dt=1*b.ms, name="event_target")
            first = b.Synapses(
                source, target, on_pre="y_post += 1", on_event="first",
                clock=source.clock, name="first_route")
            second = b.Synapses(
                source, target, on_pre="y_post += 10", on_event="second",
                clock=source.clock, name="second_route")
            first.connect()
            second.connect()
            b.Network(source, target, first, second).run(3*b.ms)
            results.append(np.asarray(target.y[:]).copy())
        for actual in results[:2]:
            np.testing.assert_array_equal(actual, results[2])
        np.testing.assert_array_equal(results[0], [11])

    def test_connect_multiplicity_expands_pairs_in_creation_order(self):
        group = self.make_group(size=2)
        group.v = [1.1, 0]
        synapse = b.Synapses(
            group, group, "w : 1", on_pre="x_post += w", clock=group.clock)
        synapse.connect(i=0, j=1, n=2)
        synapse.connect(i=0, j=1, n=3)
        synapse.w = [1, 2, 3, 4, 5]
        network, _, _ = self.network(group, synapse)
        network.run(1*b.ms)
        np.testing.assert_array_equal(synapse.i[:], [0, 0, 0, 0, 0])
        np.testing.assert_array_equal(synapse.j[:], [1, 1, 1, 1, 1])
        self.assertEqual(group.x[1], 15)

        self.select_rust()
        group = self.make_group(size=3)
        repeated = b.Synapses(
            group, group, on_pre="x_post += 1", clock=group.clock)
        repeated.connect("i != j", n=2)
        np.testing.assert_array_equal(
            repeated.i[:], [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
        np.testing.assert_array_equal(
            repeated.j[:], [1, 1, 2, 2, 0, 0, 2, 2, 0, 0, 1, 1])

    def test_multisynaptic_index_is_a_canonical_read_only_edge_input(self):
        snapshots = []
        for engine in ("reference", "aot"):
            self.device.reinit()
            b.set_device("rust_standalone", runner=self.runner, engine=engine)
            group = self.make_group(size=2)
            group.v = [1.1, 0]
            synapse = b.Synapses(
                group, group, on_pre="x_post = 10*x_post + synapse_number",
                multisynaptic_index="synapse_number", clock=group.clock)
            synapse.connect(i=0, j=1, n=3)
            network, _, _ = self.network(group, synapse)
            network.run(1*b.ms)
            snapshots.append((np.asarray(synapse.synapse_number[:]).copy(),
                              np.asarray(group.x[:]).copy()))
        for occurrence, state in snapshots:
            np.testing.assert_array_equal(occurrence, [0, 1, 2])
            np.testing.assert_array_equal(state, [0, 12])
        np.testing.assert_array_equal(snapshots[0][0], snapshots[1][0])
        np.testing.assert_array_equal(snapshots[0][1], snapshots[1][1])

    def test_dynamic_synaptic_subexpression_is_materialized_per_event(self):
        snapshots = []
        for engine in ("reference", "aot"):
            self.device.reinit()
            b.set_device("rust_standalone", runner=self.runner, engine=engine)
            group = self.make_group(size=2)
            group.v = [1.1, 0]
            synapse = b.Synapses(
                group, group,
                """w : 1
                   doubled = 2*w : 1
                   effective = doubled + 0.5*x_pre : 1""",
                on_pre="x_post += effective", clock=group.clock)
            synapse.connect(i=0, j=1)
            synapse.w = 1.25
            group.x = [2, 0]
            network, state, _ = self.network(group, synapse)
            network.run(1*b.ms)
            snapshots.append((np.asarray(group.x[:]).copy(),
                              np.asarray(state.x).copy()))
        for final, trace in snapshots:
            np.testing.assert_array_equal(final, [2, 3.5])
            np.testing.assert_array_equal(trace[:, 0], [2, 0])
        np.testing.assert_array_equal(snapshots[0][0], snapshots[1][0])

    def test_constant_over_dt_synaptic_subexpression_uses_before_start_schedule(self):
        results = []
        for backend in ["aot", "reference", "numpy"]:
            if backend == "numpy":
                b.set_device("runtime")
                b.prefs.codegen.target = "numpy"
                b.start_scope()
            else:
                self.device.reinit()
                b.set_device("rust_standalone", runner=self.runner, engine=backend)
            source = b.NeuronGroup(
                1, "v:1", threshold="v > 0", reset="v = 1",
                dt=1*b.ms, name=f"constant_source_{backend}")
            target = b.NeuronGroup(
                1, "dy/dt=0*Hz:1", method="euler", dt=1*b.ms,
                name=f"constant_target_{backend}")
            source.v = 1
            projection = b.Synapses(
                source, target,
                "w:1\neffective = w + t/(1*ms) : 1 (constant over dt)",
                on_pre="y_post += effective", clock=source.clock,
                name=f"constant_projection_{backend}")
            projection.connect()
            projection.w = 1
            state = b.StateMonitor(target, "y", record=True)
            b.Network(source, target, projection, state).run(3*b.ms)
            results.append((np.asarray(state.y).copy(),
                            np.asarray(target.y[:]).copy(),
                            np.asarray(projection.effective[:]).copy()))
            if backend == "aot":
                exported = json.loads(
                    (self.device.last_run_directory / "model.json").read_text())
                self.assertEqual(
                    exported["definition"]["synapses"][0]["code_objects"][0]["kind"],
                    "synapse_subexpression_update")
        for actual in results:
            np.testing.assert_array_equal(actual[0], [[0, 1, 3]])
            np.testing.assert_array_equal(actual[1], [6])
            np.testing.assert_array_equal(actual[2], [3])
        for actual in results[1:]:
            for left, right in zip(results[0], actual, strict=True):
                np.testing.assert_array_equal(left, right)

    def test_seeded_probability_conditions_and_string_initialization_are_reproducible(self):
        results = []
        for engine in ["aot", "reference"]:
            self.device.reinit()
            b.set_device("rust_standalone", runner=self.runner, engine=engine)
            b.seed(1729)
            source = b.NeuronGroup(
                4, "dv/dt=0*Hz:1\ndx/dt=0*Hz:1", threshold="v>2",
                reset="v=0", method="euler", dt=1*b.ms, name=f"source_{engine}")
            target = b.NeuronGroup(
                5, "dv/dt=0*Hz:1\ndx/dt=0*Hz:1", threshold="v>2",
                reset="v=0", method="euler", dt=1*b.ms, name=f"target_{engine}")
            source.v = "0.25 + 0.1*rand()"
            source.x = "randn()"
            target.v = "0.1*rand()"
            projection = b.Synapses(
                source, target, "w:1", on_pre="x_post += w",
                clock=source.clock, name=f"projection_{engine}")
            active_sources = 3
            projection.connect("i < active_sources and j != i", p=0.4)
            projection.w = "0.5 + 0.1*i + 0.01*j"
            source_state = b.StateMonitor(source, ["v", "x"], record=True)
            target_state = b.StateMonitor(target, ["v", "x"], record=True)
            network = b.Network(source, target, projection, source_state, target_state)
            initial = (source.v[:].copy(), source.x[:].copy(), target.v[:].copy())
            topology = (projection.i[:].copy(), projection.j[:].copy(),
                        projection.w[:].copy())
            network.run(2*b.ms)
            results.append((*initial, *topology, source_state.v[:].copy(),
                            source_state.x[:].copy(), target_state.v[:].copy()))
            self.assertGreater(len(projection), 0)
            self.assertTrue(np.all(projection.i[:] < active_sources))
            self.assertTrue(np.all(projection.i[:] != projection.j[:]))
            order = projection.i[:] * len(target) + projection.j[:]
            self.assertTrue(np.all(np.diff(order) > 0))
        for actual, expected in zip(results[0], results[1], strict=True):
            np.testing.assert_array_equal(actual, expected)

        # String initialization uses the same seeded NumPy frontend semantics
        # as Brian's NumPy target; it is not runtime simulation fallback.
        initialized = []
        for backend in ["rust", "numpy"]:
            if backend == "rust":
                self.select_rust()
            else:
                b.set_device("runtime")
                b.prefs.codegen.target = "numpy"
            b.seed(991)
            group = b.NeuronGroup(8, "v:1\nx:1")
            group.v = "rand()"
            group.x = "randn()"
            initialized.append((group.v[:].copy(), group.x[:].copy()))
        for actual, expected in zip(initialized[0], initialized[1], strict=True):
            np.testing.assert_array_equal(actual, expected)

    def test_probability_filters_explicit_pairs_and_default_connect_is_all_to_all(self):
        group = self.make_group(3)
        b.seed(42)
        probabilistic = b.Synapses(group, group, on_pre="x_post += 1",
                                   clock=group.clock)
        candidates_i = np.array([0, 0, 1, 1, 2, 2])
        candidates_j = np.array([0, 1, 0, 2, 1, 2])
        expected = np.random.RandomState(42).random(len(candidates_i)) < 0.5
        probabilistic.connect(i=candidates_i, j=candidates_j, p=0.5)
        np.testing.assert_array_equal(probabilistic.i[:], candidates_i[expected])
        np.testing.assert_array_equal(probabilistic.j[:], candidates_j[expected])

        complete = b.Synapses(group, group, on_pre="x_post += 1",
                              clock=group.clock)
        complete.connect()
        np.testing.assert_array_equal(complete.i[:], np.repeat(np.arange(3), 3))
        np.testing.assert_array_equal(complete.j[:], np.tile(np.arange(3), 3))

        empty = b.Synapses(group, group, on_pre="x_post += 1",
                           clock=group.clock)
        empty.connect(p=0)
        self.assertEqual(len(empty), 0)
        state = b.StateMonitor(group, ["v", "x"], record=True)
        b.Network(group, empty, state).run(1*b.ms)
        self.assertEqual(len(empty), 0)

    def test_state_dependent_condition_and_string_probability_connect(self):
        group = b.NeuronGroup(4, "x : meter")
        group.x = [0, 1, 2, 3]*b.umeter
        span = 1.1*b.umeter
        expected_i, expected_j = np.nonzero(
            np.abs(np.arange(4)[:, None] - np.arange(4)[None, :]) <= 1)

        deterministic = b.Synapses(group, group)
        deterministic.connect("sqrt((x_pre-x_post)**2) <= span")
        probabilistic = b.Synapses(group, group)
        probabilistic.connect(p="int(abs(x_pre-x_post) <= span)")

        for projection in (deterministic, probabilistic):
            np.testing.assert_array_equal(projection.i[:], expected_i)
            np.testing.assert_array_equal(projection.j[:], expected_j)

    def test_postsynaptic_generator_range_sample_and_skip_if_invalid(self):
        group = b.NeuronGroup(5, "x : 1")
        radius = 1
        neighbours = b.Synapses(group, group)
        neighbours.connect(
            j="k for k in range(i-radius, i+radius+1) if k != i",
            skip_if_invalid=True)
        np.testing.assert_array_equal(
            neighbours.i[:], [0, 1, 1, 2, 2, 3, 3, 4])
        np.testing.assert_array_equal(
            neighbours.j[:], [1, 0, 2, 1, 3, 2, 4, 3])

        for first in [True, False]:
            b.seed(731)
            sampled = b.Synapses(group, group)
            sampled.connect(
                j=("k for k in sample(0, N_post, p=0.7) "
                   "if rand() < exp(-0.1*(i-j)**2) and k != i"),
                skip_if_invalid=True)
            topology = (sampled.i[:].copy(), sampled.j[:].copy())
            if first:
                expected = topology
            else:
                np.testing.assert_array_equal(topology[0], expected[0])
                np.testing.assert_array_equal(topology[1], expected[1])
            self.assertTrue(np.all(topology[0] != topology[1]))
            self.assertEqual(
                len(set(zip(topology[0].tolist(), topology[1].tolist()))),
                len(sampled))

        invalid = b.Synapses(group, group)
        with self.assertRaises(IndexError):
            invalid.connect(j="k for k in range(i-1, i+2)")
        self.assertEqual(len(invalid), 0)

        layered = b.NeuronGroup(6, "x : 1")
        block = 2
        forward = b.Synapses(layered, layered)
        forward.connect(
            j=("k for k in range((int(i/block)+1)*block, "
               "(int(i/block)+2)*block) if i < N_pre-block"))
        np.testing.assert_array_equal(forward.i[:], [0, 0, 1, 1, 2, 2, 3, 3])
        np.testing.assert_array_equal(forward.j[:], [2, 3, 2, 3, 4, 5, 4, 5])

    def test_synapse_state_monitor_matches_numpy_with_own_clock(self):
        results = []
        for backend in ["reference", "aot", "numpy"]:
            self.device.reinit()
            b.start_scope()
            if backend == "numpy":
                b.set_device("runtime")
                b.prefs.codegen.target = "numpy"
            else:
                b.set_device(
                    "rust_standalone", runner=self.runner, engine=backend,
                    directory=self.directory / f"synapse-monitor-clock-{backend}")
            group = b.NeuronGroup(
                2, "v:1", threshold="v>0", reset="v=0", dt=1*b.ms)
            group.v = [1, 0]
            synapse = b.Synapses(
                group, group, "w:1\nx:1",
                on_pre="w += 1; x += 2", clock=group.clock)
            synapse.connect(i=[0, 0], j=[0, 1])
            monitor = b.StateMonitor(
                synapse, ["w", "x"], record=[1, 0], dt=2*b.ms)
            b.Network(group, synapse, monitor).run(4*b.ms)
            results.append(tuple(np.asarray(value).copy() for value in (
                monitor.t[:] / b.ms, monitor.w[:], monitor.x[:],
                synapse.w[:], synapse.x[:])))
        for actual_result in results[1:]:
            for actual, expected in zip(actual_result, results[0], strict=True):
                np.testing.assert_array_equal(actual, expected)

    def test_synapse_state_monitors_can_use_distinct_clocks(self):
        results = []
        for backend in ["reference", "aot", "numpy"]:
            self.device.reinit()
            b.start_scope()
            if backend == "numpy":
                b.set_device("runtime")
                b.prefs.codegen.target = "numpy"
            else:
                b.set_device(
                    "rust_standalone", runner=self.runner, engine=backend,
                    directory=self.directory / f"synapse-monitor-clocks-{backend}")
            group = b.NeuronGroup(1, "v:1", dt=1*b.ms)
            synapse = b.Synapses(
                group, group, "dw/dt = 1/ms : 1 (clock-driven)",
                method="euler", clock=group.clock)
            synapse.connect()
            fast = b.StateMonitor(synapse, "w", record=True, dt=1*b.ms)
            slow = b.StateMonitor(synapse, "w", record=True, dt=2*b.ms)
            b.Network(group, synapse, fast, slow).run(4*b.ms)
            results.append(tuple(np.asarray(value).copy() for value in (
                fast.t[:] / b.ms, fast.w[:], slow.t[:] / b.ms, slow.w[:])))
        for actual_result in results[1:]:
            for actual, expected in zip(actual_result, results[0], strict=True):
                np.testing.assert_array_equal(actual, expected)

    def test_synapse_ode_inlines_endpoint_subexpressions(self):
        results = []
        for backend in ["reference", "aot", "numpy"]:
            self.device.reinit()
            b.start_scope()
            if backend == "numpy":
                b.set_device("runtime")
                b.prefs.codegen.target = "numpy"
            else:
                b.set_device(
                    "rust_standalone", runner=self.runner, engine=backend,
                    directory=self.directory / f"endpoint-expression-{backend}")
            group = b.NeuronGroup(
                2, "du/dt = 1/ms : 1\nF = 2*u + 1 : 1",
                method="euler", dt=1*b.ms)
            group.u = [0, 1]
            synapse = b.Synapses(
                group, group,
                "dw/dt = (F_pre + F_post)/ms : 1 (clock-driven)",
                method="euler", clock=group.clock)
            synapse.connect(i=[0], j=[1])
            b.Network(group, synapse).run(3*b.ms)
            results.append(np.asarray(synapse.w[:]).copy())
        for actual in results[:2]:
            np.testing.assert_array_equal(actual, results[2])

    def test_synapse_state_monitor_records_event_driven_states(self):
        results = []
        for backend in ("reference", "aot", "numpy"):
            self.device.reinit()
            b.start_scope()
            if backend == "numpy":
                b.set_device("runtime")
                b.prefs.codegen.target = "numpy"
            else:
                b.set_device(
                    "rust_standalone", runner=self.runner, engine=backend,
                    directory=self.directory / f"synapse-monitor-event-{backend}")
            source = b.SpikeGeneratorGroup(
                1, [0, 0], [1, 3]*b.ms, dt=1*b.ms)
            target = b.NeuronGroup(1, "v : 1", dt=1*b.ms)
            synapse = b.Synapses(
                source, target,
                "du/dt=-u/(2*ms) : 1 (event-driven)",
                on_pre="u += 1", clock=source.clock)
            synapse.connect()
            monitor = b.StateMonitor(
                synapse, "u", record=True, when="after_synapses")
            b.Network(source, target, synapse, monitor).run(5*b.ms)
            results.append((np.asarray(monitor.u).copy(),
                            np.asarray(synapse.u[:]).copy()))
        for actual_result in results[1:]:
            for actual, expected in zip(actual_result, results[0], strict=True):
                np.testing.assert_array_equal(actual, expected)

    def test_synapse_state_monitor_records_postsynaptic_alias(self):
        results = []
        for backend in ["reference", "aot", "numpy"]:
            self.device.reinit()
            b.start_scope()
            if backend == "numpy":
                b.set_device("runtime")
                b.prefs.codegen.target = "numpy"
            else:
                b.set_device(
                    "rust_standalone", runner=self.runner, engine=backend,
                    directory=self.directory / f"synapse-monitor-post-{backend}")
            source = b.NeuronGroup(2, "v:1", dt=1*b.ms)
            source.v = [3, 4]
            target = b.NeuronGroup(1, "g:1", dt=1*b.ms)
            synapse = b.Synapses(
                source, target, "w:1\ng_post = w : 1 (summed)",
                clock=source.clock)
            synapse.connect()
            synapse.w = [1, 2]
            monitor = b.StateMonitor(
                synapse, ["v_pre", "g"], record=True)
            network = b.Network(source, target, synapse, monitor)
            if backend != "numpy":
                model = lower_network(network, 3*b.ms)
                monitor_def = model["definition"]["synapses"][0][
                    "state_monitors"][0]
                self.assertEqual(monitor_def["sources"], [
                    {"kind": "pre_state", "name": "v", "dtype": "f64"},
                    {"kind": "post_state", "name": "g", "dtype": "f64"},
                ])
            network.run(3*b.ms)
            if backend == "aot":
                self.assertEqual(
                    self.device.last_execution_plan.cpu.emitter, "slot-v1")
                native = (self.device.last_run_directory /
                          "native" / "main.rs").read_text()
                self.assertIn("for edge in 0..s0_edge_count {", native)
            results.append(tuple(np.asarray(value).copy() for value in (
                monitor.t[:] / b.ms, monitor.v_pre[:], monitor.g[:],
                target.g[:])))
        for actual_result in results[1:]:
            for actual, expected in zip(actual_result, results[0], strict=True):
                np.testing.assert_array_equal(actual, expected)

    def test_fixed_index_linked_subexpression_matches_numpy_and_monitor(self):
        results = []
        for backend in ["reference", "aot", "numpy"]:
            self.device.reinit()
            b.start_scope()
            if backend == "numpy":
                b.set_device("runtime")
                b.prefs.codegen.target = "numpy"
            else:
                b.set_device(
                    "rust_standalone", runner=self.runner, engine=backend,
                    directory=self.directory / f"linked-subexpression-{backend}")
            gamma = 0.5 * b.Hz
            global_factor = b.NeuronGroup(
                1,
                "G = H - gamma : Hz\n"
                "dH/dt = 1*Hz/second : Hz",
                method="euler", dt=1*b.ms, namespace={"gamma": gamma})
            global_factor.H = 2*b.Hz
            source = b.SpikeGeneratorGroup(
                1, [0, 0], [1, 3]*b.ms, dt=1*b.ms)
            target = b.NeuronGroup(1, "v : 1", dt=1*b.ms)
            synapse = b.Synapses(
                source, target, "G : Hz (linked)\nw : 1",
                on_pre="w += G*ms", clock=source.clock)
            synapse.connect()
            synapse.G = b.linked_var(global_factor, "G")
            monitor = b.StateMonitor(synapse, ["G", "w"], record=True)
            network = b.Network(
                global_factor, source, target, synapse, monitor)
            if backend != "numpy":
                model = lower_network(network, 5*b.ms)
                synapse_def = model["definition"]["synapses"][0]
                self.assertEqual(len(synapse_def["linked_variables"]), 1)
                self.assertEqual(
                    synapse_def["linked_variables"][0]["source_state"], "H")
                self.assertIn("G", synapse_def["monitor_expressions"])
                self.assertEqual(
                    synapse_def["state_monitors"][0]["output_variables"],
                    ["G", "w"])
            network.run(5*b.ms)
            results.append(tuple(np.asarray(value).copy() for value in (
                monitor.t[:] / b.ms, monitor.G[:] / b.Hz,
                monitor.w[:], synapse.w[:])))
            if backend == "aot":
                self.assertEqual(
                    self.device.last_execution_plan.cpu.emitter, "slot-v1")
        for actual_result in results[1:]:
            for actual, expected in zip(actual_result, results[0], strict=True):
                np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-14)

    def test_spike_generator_group_can_drive_post_pathway(self):
        results = []
        for backend in ["reference", "aot", "numpy"]:
            b.start_scope()
            if backend != "numpy":
                self.device.reinit()
                b.set_device(
                    "rust_standalone", runner=self.runner, engine=backend,
                    directory=self.directory / f"stateless-post-{backend}")
            else:
                b.set_device("runtime")
                b.prefs.codegen.target = "numpy"
            pre = b.SpikeGeneratorGroup(1, [0], [1]*b.ms, dt=1*b.ms)
            post = b.SpikeGeneratorGroup(1, [0], [2]*b.ms, dt=1*b.ms)
            synapse = b.Synapses(
                pre, post, "w:1", on_pre="w += 1", on_post="w += 10",
                clock=pre.clock)
            synapse.connect()
            b.Network(pre, post, synapse).run(4*b.ms)
            results.append(np.asarray(synapse.w[:]).copy())
        for actual in results:
            np.testing.assert_array_equal(actual, [11])

    def test_stateless_neuron_groups_are_synapse_endpoints(self):
        results = []
        for backend in ("reference", "aot", "numpy"):
            self.device.reinit()
            b.start_scope()
            if backend == "numpy":
                b.set_device("runtime")
                b.prefs.codegen.target = "numpy"
            else:
                b.set_device(
                    "rust_standalone", runner=self.runner, engine=backend,
                    directory=self.directory / f"stateless-neurons-{backend}")
            source = b.NeuronGroup(
                1, "", threshold="t >= 1*ms", refractory=100*b.ms,
                dt=1*b.ms)
            target = b.NeuronGroup(1, "", dt=1*b.ms)
            synapse = b.Synapses(
                source, target, "w : integer", on_pre="w += 1",
                clock=source.clock)
            synapse.connect()
            spikes = b.SpikeMonitor(source)
            b.Network(source, target, synapse, spikes).run(4*b.ms)
            results.append((np.asarray(synapse.w[:]).copy(),
                            np.asarray(spikes.t/b.ms).copy()))
        for weights, times in results:
            np.testing.assert_array_equal(weights, [1])
            np.testing.assert_array_equal(times, [1])

    def test_zero_delay_delivers_before_reset_but_after_threshold(self):
        group = self.make_group()
        group.v = [1.1, 0, 0]
        synapse = b.Synapses(group, group, "w : 1 (constant)",
                            on_pre="v_post += w", clock=group.clock)
        synapse.connect(i=[0, 0, 1], j=[0, 1, 2])
        synapse.w = [2, 1.2, 1.3]
        net, state, spikes = self.network(group, synapse)
        net.run(3*b.ms)
        np.testing.assert_array_equal(state.v, [[1.1, 0, 0], [0, 1.2, 0], [0, 0, 1.3]])
        np.testing.assert_array_equal(group.v[:], [0, 0, 0])
        np.testing.assert_array_equal(spikes.i[:], [0, 1, 2])
        np.testing.assert_array_equal(spikes.t[:] / b.ms, [0, 1, 2])
        summary = json.loads((self.device.last_run_directory / "rust/summary.json").read_text())
        self.assertEqual(summary["synaptic_events"], 3)

    def test_delayed_source_events_keep_edge_order_when_ring_wraps(self):
        for delay in [0, 2, 20]:
            self.select_rust()
            group = b.NeuronGroup(4, "dv/dt=0*Hz:1\ndx/dt=0*Hz:1",
                                  threshold="v>1", reset="v=2", method="euler", dt=1*b.ms)
            # Source 2 fires but has no outgoing edges. Sources 0/1 share a
            # target and have duplicate edges in interleaved creation order.
            group.v = [1.1, 1.2, 1.3, 0]
            synapse = b.Synapses(group, group, "w:1", on_pre="x_post=2*x_post+w",
                                delay=delay*b.ms, clock=group.clock)
            synapse.connect(i=[1, 0, 1, 0], j=[3, 3, 3, 3])
            synapse.w = [1, 2, 3, 4]
            net, state, _ = self.network(group, synapse)
            net.run(6*b.ms)
            expected, value = [], 0
            for tick in range(6):
                expected.append(value)
                if tick >= delay:
                    for weight in [2, 4, 1, 3]:
                        value = 2*value + weight
            np.testing.assert_array_equal(state.x[3], expected)
            self.assertEqual(group.x[3], value)
            summary = json.loads((self.device.last_run_directory / "rust/summary.json").read_text())
            self.assertEqual(summary["synaptic_events"], 4*max(0, 6-delay))

    def test_fixed_delay_rounding_and_events_beyond_run_end(self):
        for delay, tick in [(1.49, 2), (1.5, 3)]:
            with self.subTest(delay=delay):
                self.select_rust()
                group = self.make_group()
                group.v = [1.1, 0, 0]
                synapse = b.Synapses(group, group, "w : 1 (constant)",
                                    on_pre="v_post += w", delay=delay*b.ms,
                                    clock=group.clock)
                synapse.connect(i=[0, 1], j=[1, 2])
                synapse.w = 1.2
                net, _, spikes = self.network(group, synapse)
                net.run(4*b.ms)
                np.testing.assert_array_equal(spikes.i[:], [0, 1])
                np.testing.assert_array_equal(spikes.t[:] / b.ms, [0, tick])
                self.assertEqual(group.v[2], 1.2 if delay < 1.5 else 0)

    def test_presynaptic_state_is_read_at_delivery_time(self):
        group = self.make_group(model="dv/dt=0*Hz : 1\ndx/dt=1*Hz : 1")
        group.v = [1.1, 0, 0]
        group.x = [1.2, 0, 0]
        synapse = b.Synapses(group, group, on_pre="v_post += x_pre",
                            delay=1*b.ms, clock=group.clock)
        synapse.connect(i=0, j=1)
        net, state, spikes = self.network(group, synapse)
        net.run(3*b.ms)
        self.assertAlmostEqual(state.v[1, 2], 1.202)
        np.testing.assert_array_equal(spikes.i[:], [0, 1])
        np.testing.assert_array_equal(spikes.t[:] / b.ms, [0, 2])

    def test_numpy_differential_with_static_shared_weights_and_delays(self):
        for delay in [None, 1.49*b.ms, 1.5*b.ms]:
            results = []
            for backend in ["aot", "reference", "numpy"]:
                if backend in {"aot", "reference"}:
                    self.device.reinit()
                    b.set_device("rust_standalone", runner=self.runner,
                                 engine=backend)
                else:
                    b.set_device("runtime")
                    b.prefs.codegen.target = "numpy"
                group = self.make_group(4)
                group.v = [1.1, 1.2, 0, 0]
                options = {} if delay is None else {"delay": delay}
                synapse = b.Synapses(group, group,
                                    "w : 1\ngain : 1 (shared)",
                                    on_pre="v_post += gain*w; x_post += w",
                                    clock=group.clock, **options)
                synapse.connect(i=[1, 0, 1, 0, 2], j=[2, 1, 3, 0, 3])
                synapse.w = [1.2, 0.4, 0.2, 0.9, 1.1]
                synapse.gain = 1
                net, state, spikes = self.network(group, synapse)
                net.run(6*b.ms)
                results.append((state.v[:].copy(), state.x[:].copy(),
                                group.v[:].copy(), group.x[:].copy(),
                                spikes.i[:].copy(), (spikes.t[:]/b.ms).copy()))
                if backend == "aot":
                    source = (self.device.last_run_directory / "native/main.rs").read_text()
                    self.assertIn("let mut syn_state_0 = data.f64_vec(edge_count)?;", source)
                    self.assertIn("let syn_parameter_0 = data.f64()?;", source)
                    self.assertIn("let target_index = data.u32_vec(edge_count)?;", source)
            for expected in results[1:]:
                for actual, other in zip(results[0], expected, strict=True):
                    np.testing.assert_allclose(actual, other, rtol=0, atol=1e-14)

    def test_clock_driven_synaptic_odes_and_on_pre_state_writes(self):
        results = []
        for backend in ["aot", "reference", "numpy"]:
            if backend == "numpy":
                b.set_device("runtime")
                b.prefs.codegen.target = "numpy"
            else:
                self.device.reinit()
                b.set_device("rust_standalone", runner=self.runner, engine=backend)
            group = self.make_group()
            group.v = [1.1, 0, 0]
            synapse = b.Synapses(
                group, group,
                "da/dt=-a/(5*ms) : 1 (clock-driven)\n"
                "db/dt=(a-b)/(2*ms) : 1 (clock-driven)\n"
                "w : 1 (constant)",
                on_pre="v_post += w*a; a += 1; b += a",
                method="euler", clock=group.clock)
            synapse.connect(i=[0, 0], j=[1, 2])
            synapse.a = [.5, .25]
            synapse.b = [.1, .2]
            synapse.w = [1, 2]
            synapse.delay = [0, 1]*b.ms
            net, state, spikes = self.network(group, synapse)
            net.run(4*b.ms)
            results.append([np.asarray(value).copy() for value in
                            (state.v, group.v[:], synapse.a[:], synapse.b[:],
                             spikes.i[:], spikes.t[:]/b.ms)])
            if backend == "aot":
                source = (self.device.last_run_directory/"native/main.rs").read_text()
                self.assertIn("for edge in 0..edge_count", source)
                self.assertIn("syn_state_0[edge]", source)
        for expected in results[1:]:
            for index, (actual, other) in enumerate(zip(results[0], expected, strict=True)):
                if index == 4:
                    np.testing.assert_array_equal(actual, other)
                else:
                    np.testing.assert_allclose(actual, other, rtol=1e-12, atol=1e-14)

    def test_mutable_synaptic_state_and_on_post_match_numpy(self):
        results = []
        for backend in ["aot", "reference", "numpy"]:
            if backend == "numpy":
                b.set_device("runtime")
                b.prefs.codegen.target = "numpy"
            else:
                self.device.reinit()
                b.set_device("rust_standalone", runner=self.runner, engine=backend)
            source = b.NeuronGroup(
                1, "dx/dt=0/second : 1", threshold="True", reset="x=x",
                method="euler", dt=1*b.ms, name=f"mutable_source_{backend}")
            target = b.NeuronGroup(
                1, "dx/dt=0/second : 1", threshold="True", reset="x=x",
                method="euler", dt=1*b.ms, name=f"mutable_target_{backend}")
            synapse = b.Synapses(
                source, target, "w : 1",
                on_pre="x_post += w; w += 1",
                on_post="w *= 2", clock=source.clock,
                name=f"mutable_projection_{backend}")
            synapse.connect(i=0, j=0)
            synapse.w = 1
            source_state = b.StateMonitor(source, "x", record=True)
            target_state = b.StateMonitor(target, "x", record=True)
            network = b.Network(source, target, synapse, source_state, target_state)
            network.run(3*b.ms)
            results.append((np.asarray(target_state.x).copy(),
                            np.asarray(target.x[:]).copy(),
                            np.asarray(synapse.w[:]).copy()))
            if backend == "aot":
                generated = (self.device.last_run_directory / "native/main.rs").read_text()
                self.assertIn("target_offsets", generated)
                self.assertIn(
                    "s0_post_delivered += s0p1_edge_scratch.len()", generated)
                summary = json.loads(
                    (self.device.last_run_directory / "rust/summary.json").read_text())
                self.assertEqual(summary["synaptic_events"], 6)
        for expected in results[1:]:
            for actual, other in zip(results[0], expected, strict=True):
                np.testing.assert_allclose(actual, other, rtol=0, atol=1e-14)
        np.testing.assert_array_equal(results[0][0], [[0, 1, 5]])
        np.testing.assert_array_equal(results[0][1], [15])
        np.testing.assert_array_equal(results[0][2], [22])

    def _multiple_pathways(self, backend, segmented=False):
        if backend == "numpy":
            b.set_device("runtime")
            b.prefs.codegen.target = "numpy"
        else:
            self.device.reinit()
            b.set_device("rust_standalone", runner=self.runner, engine=backend)
        source = b.NeuronGroup(
            2, "dx/dt=0/second : 1", threshold="True", reset="x=x",
            method="euler", dt=1*b.ms, name=f"multi_source_{backend}")
        target = b.NeuronGroup(
            2, "dx/dt=0/second : 1", threshold="True", reset="x=x",
            method="euler", dt=1*b.ms, name=f"multi_target_{backend}")
        synapse = b.Synapses(
            source, target, "w : 1\nz : 1",
            on_pre={
                "slow": "x_post += 10*w; w += 2",
                "fast": "x_post += w; w += 1",
            },
            on_post={
                "back_two": "z += 2*w",
                "back_one": "z += w",
            },
            clock=source.clock, name=f"multi_projection_{backend}")
        synapse.connect()
        synapse.w = [1, 2, 3, 4]
        synapse.fast.delay = 0*b.ms
        synapse.slow.delay = [0, 1, 1, 0]*b.ms
        synapse.back_one.delay = 0*b.ms
        synapse.back_two.delay = [0, 1, 0, 1]*b.ms
        source_state = b.StateMonitor(source, "x", record=True)
        target_state = b.StateMonitor(target, "x", record=True)
        network = b.Network(source, target, synapse, source_state, target_state)
        if segmented:
            network.run(2*b.ms)
            network.run(2*b.ms)
        else:
            network.run(4*b.ms)
        model = None
        if backend != "numpy":
            model = json.loads(
                (self.device.last_run_directory / "model.json").read_text())
        return (tuple(np.asarray(value).copy() for value in (
                    source_state.x, target_state.x, target.x[:],
                    synapse.w[:], synapse.z[:])), model)

    def test_canonical_pathway_continuation_without_spike_monitor(self):
        from unittest.mock import patch
        reference, _ = self._multiple_pathways("reference")
        with patch("brian2_rust.planner.fixed_phase_eligible", return_value=False), \
             patch("brian2_rust.plan.fixed_phase_eligible", return_value=False):
            for segmented in (False, True):
                actual, _ = self._multiple_pathways("aot", segmented=segmented)
                for value, expected in zip(actual, reference, strict=True):
                    np.testing.assert_allclose(value, expected, rtol=0, atol=1e-14)

    def test_named_pre_post_pathways_and_independent_delays_match_numpy(self):
        reference, reference_model = self._multiple_pathways("reference")
        aot, aot_model = self._multiple_pathways("aot")
        segmented, _ = self._multiple_pathways("aot", segmented=True)
        numpy, _ = self._multiple_pathways("numpy")
        for actual in (aot, segmented, numpy):
            for value, expected in zip(actual, reference, strict=True):
                np.testing.assert_allclose(value, expected, rtol=0, atol=1e-14)
        for model in (reference_model, aot_model):
            pathways = model["instance"]["synapses"][0]["pathways"]
            self.assertEqual([pathway["kind"] for pathway in pathways],
                             ["pre", "pre", "post", "post"])
            self.assertEqual([pathway["name"].rsplit("_", 1)[-1]
                              for pathway in pathways],
                             ["fast", "slow", "one", "two"])
            self.assertEqual([pathway["delay_ticks"] for pathway in pathways],
                             [[0], [0, 1, 1, 0], [0], [0, 1, 0, 1]])

    def _pair_stdp(self, backend, segmented=False):
        if backend == "numpy":
            b.set_device("runtime")
            b.prefs.codegen.target = "numpy"
        else:
            self.device.reinit()
            b.set_device("rust_standalone", runner=self.runner, engine=backend)
        source = b.NeuronGroup(
            1, "dv/dt=0/second : 1", threshold="True", refractory=100*b.ms,
            method="euler", dt=1*b.ms, name=f"stdp_source_{backend}")
        target = b.NeuronGroup(
            1, "dv/dt=0/second : 1", threshold="t >= 2*ms",
            refractory=100*b.ms, method="euler", dt=1*b.ms,
            name=f"stdp_target_{backend}")
        synapse = b.Synapses(
            source, target,
            "dApre/dt = -Apre/(2*ms) : 1 (event-driven)\n"
            "dApost/dt = -Apost/(3*ms) : 1 (event-driven)\n"
            "w : 1",
            on_pre="Apre += 0.1; w = clip(w + Apost, 0, 1)",
            on_post="Apost += -0.12; w = clip(w + Apre, 0, 1)",
            delay=1*b.ms, method="euler", clock=source.clock,
            name=f"stdp_projection_{backend}")
        synapse.connect(i=0, j=0)
        synapse.w = 0.5
        source_state = b.StateMonitor(source, "v", record=True)
        target_state = b.StateMonitor(target, "v", record=True)
        network = b.Network(source, target, synapse, source_state, target_state)
        if segmented:
            network.run(1*b.ms)
            network.run(3*b.ms)
        else:
            network.run(4*b.ms)
        return tuple(np.asarray(value).copy() for value in (
            synapse.w[:], synapse.Apre[:], synapse.Apost[:],
            synapse.lastupdate[:], source_state.v, target_state.v))

    def test_event_driven_pair_stdp_matches_numpy_and_segmented_run(self):
        reference = self._pair_stdp("reference")
        aot = self._pair_stdp("aot")
        generated = (self.device.last_run_directory / "native/main.rs").read_text()
        self.assertIn("fn s0_pre_plastic", generated)
        self.assertIn("fn s0_post_plastic", generated)
        self.assertIn("parallel.for_each(active.len(), parallel_work", generated)
        self.assertIn("parallel_plasticity |= s0_pre_plastic(&parallel", generated)
        self.assertIn("parallel_plasticity |= s0_post_plastic(&parallel", generated)
        self.assertIn("let mut s0p1_edge_scratch: Vec<u32>", generated)
        self.assertIn("&s0p1_edge_scratch, time", generated)
        self.assertIn("source_csr_u32", generated)
        segmented = self._pair_stdp("aot", segmented=True)
        numpy = self._pair_stdp("numpy")
        for actual in (aot, segmented, numpy):
            for value, expected in zip(actual, reference, strict=True):
                np.testing.assert_allclose(value, expected, rtol=1e-12, atol=1e-14)
        self.assertAlmostEqual(reference[0][0], 0.5 + 0.1*np.exp(-0.5))
        self.assertAlmostEqual(reference[1][0], 0.1*np.exp(-0.5))
        self.assertAlmostEqual(reference[2][0], -0.12)
        self.assertAlmostEqual(reference[3][0], 2e-3)

    def test_parallel_plasticity_is_worker_count_independent(self):
        def run(threads):
            self.device.reinit()
            directory = self.directory / f"parallel-plasticity-{threads}"
            b.set_device(
                "rust_standalone", runner=self.runner, engine="aot",
                threads=threads,
                directory=directory)
            self.assertEqual(self.device.build_options["threads"], threads)
            source = b.NeuronGroup(
                256, "dv/dt=0/second : 1", threshold="True",
                refractory=100*b.ms, method="euler", dt=1*b.ms,
                name=f"parallel_plasticity_source_{threads}")
            target = b.NeuronGroup(
                256, "dv/dt=0/second : 1", threshold="True",
                refractory=100*b.ms, method="euler", dt=1*b.ms,
                name=f"parallel_plasticity_target_{threads}")
            synapse = b.Synapses(
                source, target,
                "dApre/dt=-Apre/(2*ms) : 1 (event-driven)\n"
                "dApost/dt=-Apost/(3*ms) : 1 (event-driven)\n"
                "w : 1",
                on_pre="Apre += 0.1; w = clip(w + Apost, 0, 1)",
                on_post="Apost -= 0.12; w = clip(w + Apre, 0, 1)",
                method="euler", clock=source.clock,
                name=f"parallel_plasticity_projection_{threads}")
            synapse.connect()
            synapse.w = 0.5
            b.Network(source, target, synapse).run(1*b.ms)
            generated = (directory / "native/main.rs").read_text()
            self.assertIn("Parallel::from_env(true)", generated)
            summary = json.loads(
                (directory / "rust/summary.json").read_text())
            return (np.asarray(synapse.w[:]).copy(),
                    np.asarray(synapse.Apre[:]).copy(),
                    np.asarray(synapse.Apost[:]).copy(), summary)

        serial = run(1)
        parallel = run(4)
        for actual, expected in zip(parallel[:3], serial[:3], strict=True):
            np.testing.assert_array_equal(actual, expected)
        self.assertFalse(serial[3]["parallel_plasticity"])
        self.assertTrue(parallel[3]["parallel_plasticity"], parallel[3])

    def test_sparse_post_trace_batch_uses_parallel_pool_and_preserves_bytes(self):
        snapshots = []
        for threads in [1, 4]:
            self.device.reinit()
            b.start_scope()
            directory = self.directory / f"sparse-post-{threads}"
            b.set_device("rust_standalone", runner=self.runner, engine="aot",
                         threads=threads, directory=directory)
            source = b.NeuronGroup(2048, 'v : 1', threshold='False', reset='', dt=.1*b.ms,
                                  name='sparse_source')
            target = b.NeuronGroup(2, 'v : 1', threshold='i == 0 and t == 0*ms',
                                  reset='', clock=source.clock, name='sparse_target')
            synapse = b.Synapses(source, target,
                'a : 1\nbb : 1\nc : 1\nw : 1', on_pre='w += 0',
                on_post='a += 0.1; bb += a; c += bb; w += c',
                clock=source.clock, name='sparse_projection')
            # Each post event gathers every other edge in source storage.
            synapse.connect(i=np.arange(2048), j=np.arange(2048) % 2)
            synapse.w = '0.25 + i*0.0001'
            b.Network(source, target, synapse).run(.2*b.ms)
            summary = json.loads((directory/'rust/summary.json').read_text())
            self.assertEqual(summary['threads'], threads)
            self.assertEqual(summary['parallel_plasticity'], threads > 1)
            snapshots.append((directory/'rust/results.bin').read_bytes())
            np.testing.assert_array_equal(np.asarray(synapse.a[:])[::2], .1)
            np.testing.assert_array_equal(np.asarray(synapse.a[:])[1::2], 0.)
        self.assertEqual(snapshots[0], snapshots[1])

    def test_deterministic_synaptic_ode_methods_match_numpy(self):
        for method in ["linear", "independent", "rk2", "rk4",
                       "exponential_euler"]:
            results = []
            for backend in ["aot", "reference", "numpy"]:
                if backend == "numpy":
                    b.set_device("runtime")
                    b.prefs.codegen.target = "numpy"
                else:
                    self.device.reinit()
                    b.set_device("rust_standalone", runner=self.runner, engine=backend)
                group = self.make_group()
                equations = (
                    "da/dt=-a/(5*ms) : 1 (clock-driven)\n"
                    "db/dt=-2*b/(4*ms) : 1 (clock-driven)"
                    if method == "independent" else
                    "da/dt=(-a+0.2*b)/(5*ms) : 1 (clock-driven)\n"
                    "db/dt=(a-2*b)/(4*ms) : 1 (clock-driven)"
                )
                synapse = b.Synapses(
                    group, group, equations,
                    on_pre="v_post += a", method=method, clock=group.clock,
                )
                synapse.connect(i=[0, 1], j=[1, 2])
                synapse.a = [0.5, 0.25]
                synapse.b = [0.1, 0.2]
                net, _, _ = self.network(group, synapse)
                net.run(4*b.ms)
                results.append([np.asarray(synapse.a[:]).copy(),
                                np.asarray(synapse.b[:]).copy()])
                if backend == "aot":
                    exported = json.loads(
                        (self.device.last_run_directory / "model.json").read_text()
                    )
                    statements = exported["definition"]["synapses"][0]["code_objects"][0]["vector"]
                    if method == "rk4":
                        self.assertTrue(any(statement["target"].startswith("__k_")
                                            for statement in statements))
                    elif method in {"linear", "independent",
                                   "exponential_euler"}:
                        source = (self.device.last_run_directory / "native/main.rs").read_text()
                        self.assertIn(".exp()", source)
            for expected in results[1:]:
                for actual, other in zip(results[0], expected, strict=True):
                    np.testing.assert_allclose(actual, other, rtol=3e-12,
                                               atol=1e-14,
                                               err_msg=f"method={method}")

    def test_multiplicative_synaptic_sde_matches_reference_and_aot(self):
        for method in ("heun", "milstein"):
            results = []
            for backend in ("reference", "aot"):
                self.device.reinit()
                b.start_scope()
                b.set_device(
                    "rust_standalone", runner=self.runner, engine=backend,
                    directory=self.directory / f"synapse-sde-{method}-{backend}")
                source = b.NeuronGroup(
                    3, "v : 1", threshold="False", method="euler",
                    dt=.1*b.ms)
                target = b.NeuronGroup(
                    3, "v : 1", method="euler", dt=.1*b.ms)
                synapse = b.Synapses(
                    source, target,
                    "dw/dt = (mu - 0.5*second*sigma**2)*w + "
                    "w*sigma*xi*second**.5 : 1 (clock-driven)",
                    method=method, clock=source.clock,
                    namespace={"mu": .5/b.second,
                               "sigma": .1/b.second})
                synapse.connect(j="i")
                synapse.w = [.8, 1, 1.2]
                b.seed(2028)
                b.Network(source, target, synapse).run(5*b.ms)
                values = np.asarray(synapse.w[:]).copy()
                self.assertTrue(np.isfinite(values).all())
                results.append(values)
            np.testing.assert_array_equal(results[0], results[1])

    def test_cross_population_synapses_match_reference_and_numpy(self):
        results = []
        for backend in ["aot", "reference", "numpy"]:
            if backend == "numpy":
                b.set_device("runtime")
                b.prefs.codegen.target = "numpy"
            else:
                self.device.reinit()
                b.set_device("rust_standalone", runner=self.runner, engine=backend)
            model = ("dv/dt=drive/ms : 1\n"
                     "dx/dt=-x/(3*ms) : 1\n"
                     "drive : 1 (constant)")
            source = b.NeuronGroup(
                2, model, threshold="v>1", reset="v=0", method="euler",
                dt=1*b.ms, name="source_group")
            target = b.NeuronGroup(
                3, model, threshold="v>1", reset="v=0", method="euler",
                dt=1*b.ms, name="target_group")
            source.v = [1.2, .2]
            source.x = [.1, .2]
            source.drive = [0, .1]
            target.v = [.1, .2, .3]
            target.x = [.4, .5, .6]
            target.drive = [.1, .2, .3]
            synapse = b.Synapses(
                source, target,
                "dtrace/dt=-log1p(trace)/(4*ms) : 1 (clock-driven)\n"
                "w : 1 (constant)",
                on_pre=("v_post += w*exp(-trace) + 0.0*N_pre + 0.0*N_post; "
                        "x_post += trace; trace += 0.1"),
                method="euler", clock=source.clock, name="feedforward")
            synapse.connect(i=[0, 0, 1], j=[2, 0, 1])
            synapse.w = [.5, .25, .75]
            synapse.trace = [1, .5, .25]
            synapse.delay = [0, 1, 0]*b.ms
            source_state = b.StateMonitor(
                source, ["v", "x"], record=[1, 0], name="source_state")
            target_state = b.StateMonitor(
                target, ["v", "x"], record=[2, 0, 2], name="target_state")
            source_spikes = b.SpikeMonitor(source, name="source_spikes")
            target_spikes = b.SpikeMonitor(target, name="target_spikes")
            network = b.Network(source, target, synapse, source_state, target_state,
                                source_spikes, target_spikes)
            network.run(5*b.ms)
            results.append([np.asarray(value).copy() for value in (
                source_state.v, source_state.x, target_state.v, target_state.x,
                source.v[:], source.x[:], target.v[:], target.x[:],
                synapse.trace[:], source_spikes.i[:], source_spikes.t[:]/b.ms,
                source_spikes.count[:], target_spikes.i[:],
                target_spikes.t[:]/b.ms, target_spikes.count[:])])
            if backend == "aot":
                exported = json.loads(
                    (self.device.last_run_directory / "model.json").read_text())
                self.assertEqual(
                    [(item["name"], item["offset"], item["count"])
                     for item in exported["definition"]["populations"]],
                    [("source_group", 0, 2), ("target_group", 2, 3)])
                self.assertEqual(exported["instance"]["synapses"][0]["source"],
                                 [0, 0, 1])
                self.assertEqual(exported["instance"]["synapses"][0]["target"],
                                 [2, 0, 1])
                generated = (self.device.last_run_directory / "native/main.rs").read_text()
                self.assertIn(".ln_1p()", generated)
                self.assertIn(".exp()", generated)
        for expected in results[1:]:
            for index, (actual, other) in enumerate(zip(results[0], expected, strict=True)):
                if index in {9, 11, 12, 14}:
                    np.testing.assert_array_equal(actual, other)
                else:
                    np.testing.assert_allclose(actual, other, rtol=1e-12, atol=1e-14)

    def test_summed_linked_reader_observes_every_tick(self):
        snapshots = []
        for engine in ("reference", "aot", "numpy"):
            self.device.reinit()
            b.start_scope()
            if engine == "numpy":
                b.set_device("runtime")
                b.prefs.codegen.target = "numpy"
            else:
                b.set_device("rust_standalone", runner=self.runner, engine=engine,
                             directory=self.directory / engine)
            source = b.NeuronGroup(1, "v:1", threshold="False", reset="",
                                   dt=b.ms, name="a_source")
            target = b.NeuronGroup(1, "total:1", dt=b.ms, name="b_target")
            reader = b.NeuronGroup(
                1, "dx/dt=external/ms:1\nexternal:1 (linked)",
                dt=b.ms, method="euler", name="c_reader")
            reader.external = b.linked_var(target, "total")
            projection = b.Synapses(
                source, target, "dw/dt=1/ms:1 (clock-driven)\n"
                "total_post=w:1 (summed)", method="euler",
                on_pre="w += 0", clock=source.clock)
            projection.connect(i=[0], j=[0])
            projection.w = 1
            monitor = b.StateMonitor(reader, "x", record=True)
            b.Network(source, target, reader, projection, monitor).run(4*b.ms)
            snapshots.append((np.asarray(monitor.x).copy(),
                              np.asarray(reader.x[:]).copy(),
                              np.asarray(target.total[:]).copy()))
        for actual in snapshots[:2]:
            for value, expected in zip(actual, snapshots[2], strict=True):
                np.testing.assert_array_equal(value, expected)

    def test_summed_variables_match_brian_schedule_clocks_and_zeroing(self):
        results = []
        for backend in ["aot", "reference", "numpy"]:
            if backend == "numpy":
                b.set_device("runtime")
                b.prefs.codegen.target = "numpy"
                b.start_scope()
            else:
                self.device.reinit()
                b.set_device("rust_standalone", runner=self.runner, engine=backend)
            source = b.NeuronGroup(
                3, "dv/dt=(0.2-v)/(3*ms):1\nsumout:1", threshold="v>2",
                reset="v=0", method="euler", dt=1*b.ms,
                name=f"summed_source_{backend}")
            target = b.NeuronGroup(
                4, "dv/dt=(gtot-v)/(2*ms):1\ngtot:1", threshold="v>9",
                reset="v=0", method="euler", dt=.5*b.ms,
                name=f"summed_target_{backend}")
            source.v = [.1, .2, .3]
            target.v = [.4, .5, .6, .7]
            projection = b.Synapses(
                source, target,
                "dg/dt=-g/(4*ms):1 (clock-driven)\n"
                "gtot_post = g + 0.1*i + 0.01*j : 1 (summed)\n"
                "sumout_pre = 2*g : 1 (summed)",
                on_pre="g += .2", method="euler", clock=source.clock,
                name=f"summed_projection_{backend}")
            projection.connect(i=[0, 0, 1], j=[0, 2, 1])
            projection.g = [1, .5, .25]
            source_state = b.StateMonitor(source, ["v", "sumout"], record=True)
            target_state = b.StateMonitor(target, ["v", "gtot"], record=True)
            b.Network(source, target, projection, source_state, target_state).run(3*b.ms)
            results.append([np.asarray(value).copy() for value in (
                source_state.v, source_state.sumout, target_state.v,
                target_state.gtot, source.sumout[:], target.gtot[:], projection.g[:])])
            # Targets without an incoming/outgoing edge are explicitly reset
            # by the summed-variable updater on every matching target tick.
            self.assertEqual(source.sumout[2], 0)
            self.assertEqual(target.gtot[3], 0)
            if backend == "aot":
                exported = json.loads(
                    (self.device.last_run_directory / "model.json").read_text())
                summed = [code for code in
                          exported["definition"]["synapses"][0]["code_objects"]
                          if code["kind"] == "summed_variable"]
                self.assertEqual(
                    [(code["summed_target"], code["summed_state"])
                     for code in summed],
                    [("post", "gtot"), ("pre", "sumout")])
                generated = (
                    self.device.last_run_directory / "native/main.rs").read_text()
                self.assertIn("p1_state_0[0..4].fill(0.0)", generated)
                # Both summed states are observed during the run (and gtot is
                # also consumed by the target ODE), so neither updater may be
                # delayed to the final endpoint tick.
                self.assertNotIn("tick + 1 == p0_end_tick", generated)
                self.assertNotIn("tick + 1 == p1_end_tick", generated)
                summary = json.loads(
                    (self.device.last_run_directory / "rust/summary.json").read_text())
                self.assertEqual(summary["final_only_summed_variable_count"], 0)
        for expected in results[1:]:
            for actual, other in zip(results[0], expected, strict=True):
                np.testing.assert_allclose(actual, other, rtol=1e-12, atol=1e-14)

    def test_summed_variable_subgroup_zeros_and_updates_only_endpoint_window(self):
        results = []
        for backend in ["aot", "reference", "numpy"]:
            if backend == "numpy":
                b.set_device("runtime")
                b.prefs.codegen.target = "numpy"
                b.start_scope()
            else:
                self.device.reinit()
                b.set_device("rust_standalone", runner=self.runner, engine=backend)
            source = self.make_group(4, "dv/dt=0*Hz:1\nsumout:1")
            target = self.make_group(5, "dv/dt=0*Hz:1\ngtot:1")
            target.gtot = [9, 7, 6, 5, 8]
            projection = b.Synapses(
                source[1:3], target[1:4],
                "g:1\ngtot_post=g + 0.1*i + 0.01*j:1 (summed)",
                on_pre="g += 1", clock=source.clock,
                name=f"subgroup_summed_{backend}")
            projection.connect()
            projection.g = np.arange(len(projection)) + 1
            state = b.StateMonitor(target, "gtot", record=True)
            b.Network(source, target, projection, state).run(1*b.ms)
            results.append((np.asarray(state.gtot).copy(),
                            np.asarray(target.gtot[:]).copy()))
        for trace, final in results:
            np.testing.assert_array_equal(trace[:, 0], [9, 7, 6, 5, 8])
            np.testing.assert_allclose(final, [9, 5.1, 7.12, 9.14, 8])
        for actual in results[1:]:
            np.testing.assert_array_equal(results[0][0], actual[0])
            np.testing.assert_allclose(results[0][1], actual[1])

    def test_target_owned_summed_variable_is_worker_count_independent(self):
        def run(threads):
            self.device.reinit()
            directory = self.directory / f"parallel-summed-{threads}"
            b.set_device(
                "rust_standalone", runner=self.runner, engine="aot",
                threads=threads, directory=directory)
            source = b.NeuronGroup(
                256, "x : 1", threshold="False", method="euler", dt=1*b.ms,
                name=f"parallel_summed_source_{threads}")
            target = b.NeuronGroup(
                256, "gtot : 1", method="euler", dt=1*b.ms,
                name=f"parallel_summed_target_{threads}")
            synapse = b.Synapses(
                source, target,
                "w : 1\n"
                "gtot_post = exp(w) + sin(w) + cos(w) + sqrt(w) : 1 (summed)",
                on_pre="w += 0",
                clock=source.clock,
                name=f"parallel_summed_projection_{threads}")
            synapse.connect()
            synapse.w = "0.001*(i + 1)"
            b.Network(source, target, synapse).run(1*b.ms)
            summary = json.loads((directory / "rust/summary.json").read_text())
            generated = (directory / "native/main.rs").read_text()
            return np.asarray(target.gtot[:]).copy(), summary, generated

        serial = run(1)
        parallel = run(4)
        np.testing.assert_array_equal(parallel[0], serial[0])
        self.assertFalse(serial[1]["parallel_summed_variable"])
        self.assertTrue(parallel[1]["parallel_summed_variable"], parallel[1])
        self.assertEqual(parallel[1]["final_only_summed_variable_count"], 1)
        # Summed updaters now use their own clock, including when it is
        # identical to the target population clock in this fixture.
        self.assertRegex(parallel[2], r"c(\d+)_tick \+ 1 == c\1_end_tick")

    def test_independent_population_schemas_clocks_spikes_and_refractory(self):
        results = []
        for backend in ["aot", "reference", "numpy"]:
            if backend == "numpy":
                b.set_device("runtime")
                b.prefs.codegen.target = "numpy"
            else:
                self.device.reinit()
                b.set_device("rust_standalone", runner=self.runner, engine=backend)
            source = b.NeuronGroup(
                2, "dv/dt=(i+1.0)/ms : 1", threshold="v>1.5", reset="v=0",
                method="euler", dt=1*b.ms, name="heterogeneous_source")
            target = b.NeuronGroup(
                3, "dx/dt=-x/(2*ms) : 1\n"
                   "dy/dt=1/ms : 1 (unless refractory)",
                threshold="y>1.5", reset="x+=1; y=.25", refractory=2*b.ms,
                method="euler", dt=.5*b.ms, name="heterogeneous_target")
            target.x = [1, 2, 3]
            synapse = b.Synapses(source, target, "w:1", on_pre="x_post+=w+0.1*v_pre",
                                 clock=source.clock)
            synapse.connect(i=[0, 1], j=[1, 2])
            synapse.w = [.2, .4]
            source_state = b.StateMonitor(source, "v", record=True)
            target_state = b.StateMonitor(target, ["x", "y"], record=True)
            source_spikes = b.SpikeMonitor(source)
            target_spikes = b.SpikeMonitor(target)
            b.Network(source, target, synapse, source_state, target_state,
                      source_spikes, target_spikes).run(4*b.ms)
            results.append([np.asarray(value).copy() for value in (
                source_state.v, target_state.x, target_state.y,
                source.v[:], target.x[:], target.y[:],
                source_spikes.i[:], source_spikes.t[:]/b.ms,
                target_spikes.i[:], target_spikes.t[:]/b.ms,
                target.lastspike[:]/b.ms, target.not_refractory[:])])
            if backend == "aot":
                exported = json.loads((self.device.last_run_directory / "model.json").read_text())
                populations = exported["definition"]["populations"]
                self.assertEqual([[state["name"] for state in pop["states"]]
                                  for pop in populations], [["v"], ["x", "y"]])
                self.assertEqual([pop["steps"] for pop in populations], [4, 8])
                self.assertIsNone(populations[0]["refractory"])
                self.assertEqual(populations[1]["refractory"]["frozen_states"], ["y"])
                self.assertNotEqual(populations[0]["code_objects"][0]["name"],
                                    populations[1]["code_objects"][0]["name"])
        for expected in results[1:]:
            for index, (actual, other) in enumerate(zip(results[0], expected, strict=True)):
                if index in {6, 8, 11}:
                    np.testing.assert_array_equal(actual, other)
                else:
                    np.testing.assert_allclose(actual, other, rtol=1e-12, atol=1e-14)

    def test_four_projection_two_population_network_matches_reference_and_numpy(self):
        results = []
        for backend in ["aot", "reference", "numpy"]:
            if backend == "numpy":
                b.set_device("runtime")
                b.prefs.codegen.target = "numpy"
            else:
                self.device.reinit()
                b.set_device("rust_standalone", runner=self.runner, engine=backend)
            model = ("dv/dt=drive/ms : 1\n"
                     "dx/dt=-x/(3*ms) : 1\n"
                     "drive : 1 (constant)")
            exc = b.NeuronGroup(
                3, model, threshold="v>1", reset="v=0", method="euler",
                dt=1*b.ms, name="exc")
            inh = b.NeuronGroup(
                2, model, threshold="v>1", reset="v=0", method="euler",
                dt=1*b.ms, name="inh")
            exc.v, exc.drive, exc.x = [1.1, .2, .3], [.6, .8, 1.0], [.1, .2, .3]
            inh.v, inh.drive, inh.x = [1.2, .4], [.7, .9], [.4, .5]

            projections = []
            layouts = [
                ("proj_ee", exc, exc, [0, 1, 2], [1, 2, 0], [.2, .3, .4], 1),
                ("proj_ei", exc, inh, [0, 2], [1, 0], [.5, .6], 1),
                ("proj_ie", inh, exc, [0, 1], [2, 1], [-.7, -.8], 2),
                ("proj_ii", inh, inh, [0, 1], [1, 0], [-.9, -1.0], 2),
            ]
            for name, source, target, pre, post, weights, delay in layouts:
                projection = b.Synapses(
                    source, target,
                    "da/dt=-a/(4*ms) : 1 (clock-driven)\nw : 1 (constant)",
                    on_pre="x_post += w+a; a += 0.1", method="euler",
                    delay=delay*b.ms, clock=source.clock, name=name)
                projection.connect(i=pre, j=post)
                projection.w = weights
                projection.a = np.arange(len(projection))*.05 + .1
                projections.append(projection)
            exc_state = b.StateMonitor(exc, ["v", "x"], record=True, name="exc_state")
            inh_state = b.StateMonitor(inh, ["v", "x"], record=True, name="inh_state")
            exc_spikes = b.SpikeMonitor(exc, name="exc_spikes")
            inh_spikes = b.SpikeMonitor(inh, name="inh_spikes")
            b.Network(exc, inh, *projections, exc_state, inh_state,
                      exc_spikes, inh_spikes).run(6*b.ms)
            results.append([np.asarray(value).copy() for value in (
                exc_state.v, exc_state.x, inh_state.v, inh_state.x,
                exc.v[:], exc.x[:], inh.v[:], inh.x[:],
                *(projection.a[:] for projection in projections),
                exc_spikes.i[:], exc_spikes.t[:]/b.ms,
                inh_spikes.i[:], inh_spikes.t[:]/b.ms)])
            if backend == "aot":
                exported = json.loads(
                    (self.device.last_run_directory / "model.json").read_text())
                self.assertEqual([item["name"] for item in
                                  exported["definition"]["synapses"]],
                                 [item[0] for item in layouts])
                summary = json.loads(
                    (self.device.last_run_directory / "rust/summary.json").read_text())
                self.assertGreater(summary["synaptic_events"], 0)
                generated = (self.device.last_run_directory / "native/main.rs").read_text()
                self.assertIn("s3_delivered", generated)
                self.assertIn("s2_state_0", generated)
                self.assertIn("let mut r0_queue", generated)
                self.assertIn("let mut r1_queue", generated)
                self.assertNotIn("let mut s0_queue", generated)
                self.assertNotIn("let mut s1_queue", generated)
        for expected in results[1:]:
            for index, (actual, other) in enumerate(zip(results[0], expected, strict=True)):
                if index in {12, 14}:
                    np.testing.assert_array_equal(actual, other)
                else:
                    np.testing.assert_allclose(actual, other, rtol=1e-12, atol=1e-14)

    def test_three_populations_and_nine_projections_match_reference_and_numpy(self):
        results = []
        for backend in ["aot", "reference", "numpy"]:
            if backend == "numpy":
                b.set_device("runtime")
                b.prefs.codegen.target = "numpy"
            else:
                self.device.reinit()
                b.set_device("rust_standalone", runner=self.runner, engine=backend)

            groups = [b.NeuronGroup(
                1, "dv/dt=1/ms : 1\ndx/dt=0*Hz : 1",
                threshold="v>=1", reset="v=0", method="euler", dt=1*b.ms,
                name=f"many_group_{index}") for index in range(3)]
            projections = []
            for source_index, source in enumerate(groups):
                for target_index, target in enumerate(groups):
                    projection = b.Synapses(
                        source, target, "w : 1 (constant)",
                        on_pre="x_post+=w", clock=source.clock,
                        name=f"many_projection_{source_index}_{target_index}")
                    projection.connect(i=0, j=0)
                    projection.w = 1 + source_index + target_index/10
                    projections.append(projection)
            monitors = [b.StateMonitor(group, ["v", "x"], record=True,
                                       name=f"many_monitor_{index}")
                        for index, group in enumerate(groups)]
            b.Network(*groups, *projections, *monitors).run(2*b.ms)
            results.append([
                *(np.asarray(monitor.v).copy() for monitor in monitors),
                *(np.asarray(monitor.x).copy() for monitor in monitors),
                *(np.asarray(group.x[:]).copy() for group in groups),
            ])
            if backend == "aot":
                exported = json.loads(
                    (self.device.last_run_directory / "model.json").read_text())
                self.assertEqual(len(exported["definition"]["populations"]), 3)
                self.assertEqual(len(exported["definition"]["synapses"]), 9)
                self.assertEqual(
                    [population["offset"]
                     for population in exported["definition"]["populations"]],
                    [0, 1, 2])
                summary = json.loads(
                    (self.device.last_run_directory / "rust/summary.json").read_text())
                self.assertEqual(summary["population_count"], 3)
                self.assertEqual(summary["synaptic_events"], 18)
        for expected in results[1:]:
            for actual, other in zip(results[0], expected, strict=True):
                np.testing.assert_allclose(actual, other, rtol=1e-12, atol=1e-14)

    def test_cross_population_subgroups_keep_local_indices_and_parent_state_mapping(self):
        results = []
        for backend in ["aot", "reference", "numpy"]:
            if backend == "numpy":
                b.set_device("runtime")
                b.prefs.codegen.target = "numpy"
            else:
                self.device.reinit()
                b.set_device("rust_standalone", runner=self.runner, engine=backend)
            source = b.NeuronGroup(
                5, "dv/dt=0*Hz : 1\ndx/dt=0*Hz : 1",
                threshold="v>1", reset="v=0", method="euler", dt=1*b.ms,
                name="subgroup_source_parent")
            target = b.NeuronGroup(
                6, "dv/dt=0*Hz : 1\ndx/dt=0*Hz : 1",
                threshold="v>10", reset="v=0", method="euler", dt=1*b.ms,
                name="subgroup_target_parent")
            source.v = [1.3, 1.1, 0, 1.2, 0]
            source.x = [.1, .2, .3, .4, .5]
            projection = b.Synapses(
                source[1:4], target[2:5],
                "da/dt=-(a+0.01*i)/(4*ms) : 1 (clock-driven)\nw : 1 (constant)",
                on_pre="x_post += w + a + 0.1*i + 0.01*j + 0.001*x_pre; a += 0.2",
                method="euler", delay=1*b.ms, clock=source.clock,
                name="subgroup_projection")
            projection.connect(i=[0, 2], j=[2, 0])
            projection.w = [1, 2]
            projection.a = [.4, .8]
            source_state = b.StateMonitor(source, ["v", "x"], record=True)
            target_state = b.StateMonitor(target, ["v", "x"], record=True)
            source_spikes = b.SpikeMonitor(source)
            target_spikes = b.SpikeMonitor(target)
            b.Network(source, target, projection, source_state, target_state,
                      source_spikes, target_spikes).run(3*b.ms)
            results.append([np.asarray(value).copy() for value in (
                source_state.v, source_state.x, target_state.x,
                source.v[:], source.x[:], target.x[:], projection.a[:],
                source_spikes.i[:], source_spikes.t[:]/b.ms,
                target_spikes.i[:], target_spikes.t[:]/b.ms)])
            if backend == "aot":
                exported = json.loads(
                    (self.device.last_run_directory / "model.json").read_text())
                synapse = exported["definition"]["synapses"][0]
                instance = exported["instance"]["synapses"][0]
                self.assertEqual(
                    (synapse["source_start"], synapse["source_count"],
                     synapse["target_start"], synapse["target_count"]),
                    (1, 3, 2, 3))
                self.assertEqual(instance["source"], [0, 2])
                self.assertEqual(instance["target"], [2, 0])
                self.assertEqual(
                    json.loads((self.device.last_run_directory /
                                "rust/summary.json").read_text())["synaptic_events"], 2)
        for expected in results[1:]:
            for index, (actual, other) in enumerate(zip(results[0], expected, strict=True)):
                if index in {7, 9}:
                    np.testing.assert_array_equal(actual, other)
                else:
                    np.testing.assert_allclose(actual, other, rtol=1e-12, atol=1e-14)

    def test_unsupported_connect_and_on_pre_semantics_fail_before_run(self):
        cases = [
            (dict(i="k for k in range(j, j + 2)"), NotImplementedError),
            (dict(i=[0], j=[1], p="0.5"), NotImplementedError),
            (dict(condition="i < limit", namespace={"limit": b.ms}),
             b.DimensionMismatchError),
            (dict(i=[2**32], j=[1]), IndexError),
        ]
        for options, error in cases:
            with self.subTest(connect=options):
                group = self.make_group()
                synapse = b.Synapses(
                    group, group, "w : 1", on_pre="v_post += w",
                    clock=group.clock)
                with self.assertRaises(error):
                    synapse.connect(**options)
                self.assertEqual(len(synapse), 0)

    def test_mixed_unqualified_and_post_state_aliases_match_numpy(self):
        results = []
        for backend in ("reference", "aot", "numpy"):
            self.device.reinit()
            b.start_scope()
            if backend == "numpy":
                b.set_device("runtime")
                b.prefs.codegen.target = "numpy"
            else:
                b.set_device(
                    "rust_standalone", runner=self.runner, engine=backend,
                    directory=self.directory / f"mixed-post-alias-{backend}")
            group = b.NeuronGroup(
                2, "dv/dt=0/second : 1", threshold="v >= 1", reset="v = 0",
                method="euler", dt=1*b.ms)
            group.v = [1, 0.25]
            synapse = b.Synapses(
                group, group, on_pre="v_post += v", clock=group.clock)
            synapse.connect(i=0, j=1)
            monitor = b.StateMonitor(group, "v", record=True)
            b.Network(group, synapse, monitor).run(2*b.ms)
            results.append((np.asarray(monitor.v).copy(),
                            np.asarray(group.v[:]).copy()))
        for actual in results[:2]:
            for value, expected in zip(actual, results[2], strict=True):
                np.testing.assert_array_equal(value, expected)

    def test_on_post_can_write_postsynaptic_state_with_mixed_aliases(self):
        results = []
        for backend in ("reference", "aot", "numpy"):
            self.device.reinit()
            b.start_scope()
            if backend == "numpy":
                b.set_device("runtime")
                b.prefs.codegen.target = "numpy"
            else:
                b.set_device(
                    "rust_standalone", runner=self.runner, engine=backend,
                    directory=self.directory / f"on-post-state-{backend}")
            group = b.NeuronGroup(
                2, "dv/dt=0/second : 1\ndx/dt=0/second : 1",
                threshold="v >= 1", reset="v = 0", method="euler", dt=1*b.ms)
            group.v = [0, 1]
            group.x = [0.125, 0.25]
            synapse = b.Synapses(
                group, group, on_post="x_post += x", clock=group.clock)
            synapse.connect(i=0, j=1)
            monitor = b.StateMonitor(group, "x", record=True)
            b.Network(group, synapse, monitor).run(3*b.ms)
            results.append((np.asarray(monitor.x).copy(),
                            np.asarray(group.x[:]).copy()))
        for actual in results[:2]:
            for value, expected in zip(actual, results[2], strict=True):
                np.testing.assert_array_equal(value, expected)

    def test_heterogeneous_delay_matches_reference_and_numpy_with_ring_wrap(self):
        results = []
        for backend in ["aot", "reference", "numpy"]:
            if backend == "numpy":
                b.set_device("runtime")
                b.prefs.codegen.target = "numpy"
            else:
                self.device.reinit()
                b.set_device("rust_standalone", runner=self.runner, engine=backend)
            group = b.NeuronGroup(4, "dv/dt=0*Hz:1\ndx/dt=0*Hz:1",
                                  threshold="v>1", reset="v=2", method="euler", dt=1*b.ms)
            group.v = [1.1, 1.2, 0, 0]
            synapse = b.Synapses(group, group, "w:1", on_pre="x_post=2*x_post+w",
                                 clock=group.clock)
            synapse.connect(i=[1, 0, 1, 0], j=[3, 3, 3, 3])
            synapse.w = [1, 2, 3, 4]
            synapse.delay = [1.5, 0, 1.49, 3.5]*b.ms
            net, state, spikes = self.network(group, synapse)
            net.run(7*b.ms)
            results.append([np.asarray(value).copy() for value in
                            (state.x, group.x[:], spikes.i[:], spikes.t[:]/b.ms)])
            if backend == "aot":
                source = (self.device.last_run_directory/"native/main.rs").read_text()
                self.assertIn("let delay_ticks = delay_values", source)
                self.assertIn("source_csr_delay", source)
                self.assertIn("pending.sources.extend_from_slice(&fired)", source)
                summary = json.loads((self.device.last_run_directory/"rust/summary.json").read_text())
                self.assertEqual(summary["synaptic_events"], 21)
        for expected in results[1:]:
            for actual, other in zip(results[0], expected, strict=True):
                np.testing.assert_allclose(actual, other, rtol=0, atol=1e-14)

    def test_target_partitioned_on_pre_and_fused_threshold_are_byte_exact(self):
        dumps, final_states = [], []
        size = 32768
        source_model = "\n".join(
            f"dx{index}/dt=0*Hz : 1" for index in range(6))
        edge_source = np.repeat(np.arange(size, dtype=np.int32), 4)
        edge_target = np.concatenate([
            (np.arange(size, dtype=np.int32) + offset) % size
            for offset in range(4)
        ]).reshape(4, size).T.reshape(-1)
        for threads in [1, 4, 8]:
            self.device.reinit()
            b.set_device(
                "rust_standalone", runner=self.runner, engine="aot",
                threads=threads,
                directory=self.directory / f"target-owner-{threads}")
            source = b.NeuronGroup(
                size, source_model, threshold="x0 > 0", reset="", method="euler",
                refractory=.2*b.ms, dt=.1*b.ms, name="parallel_source")
            target = b.NeuronGroup(
                size, "dy/dt=0*Hz : 1", method="euler", dt=.1*b.ms,
                name="parallel_target")
            source.x0 = 1
            projection = b.Synapses(
                source, target, "w : 1 (constant)", on_pre="y_post += w",
                clock=source.clock, name="a_parallel_projection")
            projection.connect(i=edge_source, j=edge_target)
            projection.w = .125
            second = b.Synapses(
                source, target, "w : 1 (constant)",
                on_pre="y_post = 2*y_post + w", clock=source.clock,
                name="b_parallel_projection")
            second.connect(i=edge_source, j=edge_target)
            second.w = .25
            source_state = b.StateMonitor(source, "x0", record=[0, size-1])
            target_state = b.StateMonitor(target, "y", record=[0, size-1])
            spikes = b.SpikeMonitor(source)
            b.Network(source, target, projection, second, source_state,
                      target_state, spikes).run(.4*b.ms)
            directory = self.device.last_run_directory
            dumps.append((directory / "rust/results.bin").read_bytes())
            final_states.append(np.asarray(target.y[:]).copy())
            generated = (directory / "native/main.rs").read_text()
            self.assertIn("struct WorkerSignal", generated)
            self.assertNotIn("shared.done.fetch_add", generated)
            self.assertIn("signal.done.store", generated)
            self.assertEqual(
                generated.count("let p1_target_owners = if"), 1)
            self.assertIn(
                "target_owner_csr_u32(&s0_offsets, &s0_edges, "
                "&s0_target_index, 32768, 0, &p1_target_owners",
                generated)
            self.assertIn(
                "target_owner_csr_u32(&s1_offsets, &s1_edges, "
                "&s1_target_index, 32768, 0, &p1_target_owners",
                generated)
            summary = json.loads((directory / "rust/summary.json").read_text())
            self.assertEqual(summary["threads"], threads)
            self.assertEqual(summary["parallel_state_update"], threads > 1)
            self.assertEqual(summary["parallel_on_pre"], threads > 1)
            self.assertEqual(summary["synaptic_events"],
                             2 * (len(projection) + len(second)))
            self.assertEqual(len(spikes.i), 2 * size)
        self.assertEqual(dumps[0], dumps[1])
        np.testing.assert_array_equal(final_states[0], final_states[1])
        np.testing.assert_array_equal(final_states[0], np.full(size, 199.75))

    def test_heterogeneous_delay_fuses_target_owned_projections(self):
        snapshots = []
        size = 4096
        edge_source = np.repeat(np.arange(size, dtype=np.int32), 4)
        edge_target = np.column_stack([
            (np.arange(size, dtype=np.int32) + offset) % size
            for offset in range(4)
        ]).reshape(-1)
        delays = (np.arange(edge_source.size) % 4) * .1 * b.ms
        for backend, threads in (("aot", 1), ("aot", 4),
                                 ("reference", 1), ("numpy", 1)):
            if backend == "numpy":
                b.set_device("runtime")
                b.prefs.codegen.target = "numpy"
            else:
                self.device.reinit()
                b.set_device(
                    "rust_standalone", runner=self.runner, engine=backend,
                    threads=threads,
                    directory=self.directory / f"heterogeneous-{backend}-{threads}")
            source = b.NeuronGroup(
                size, "v : 1", threshold="v > 0", reset="", dt=.1*b.ms,
                name=f"heterogeneous_source_{backend}_{threads}")
            target = b.NeuronGroup(
                size, "y : 1", dt=.1*b.ms,
                name=f"heterogeneous_target_{backend}_{threads}")
            source.v = 1
            first = b.Synapses(
                source, target, "w : 1 (constant)", on_pre="y_post += w",
                clock=source.clock,
                name=f"a_heterogeneous_{backend}_{threads}")
            first.connect(i=edge_source, j=edge_target)
            first.w = .125
            first.delay = delays
            second = b.Synapses(
                source, target, "w : 1 (constant)",
                on_pre="y_post = 2*y_post + w",
                clock=source.clock,
                name=f"b_heterogeneous_{backend}_{threads}")
            second.connect(i=edge_source, j=edge_target)
            second.w = .25
            second.delay = delays
            profile_phases = backend == "aot" and threads == 4
            with patch.dict(os.environ, {
                    "B2_AOT_PROFILE_PHASES": "1" if profile_phases else "0"}):
                b.Network(source, target, first, second).run(.4*b.ms)
            snapshots.append(np.asarray(target.y[:]).copy())
            if backend == "aot":
                directory = self.device.last_run_directory
                generated = (directory / "native/main.rs").read_text()
                self.assertIn("route_0_heterogeneous_target_on_pre", generated)
                self.assertIn("enqueue_owner_csr_ptr", generated)
                self.assertIn("queue_at = slot*parallel_threads+owner", generated)
                summary = json.loads(
                    (directory / "rust/summary.json").read_text())
                self.assertEqual(summary["event_route_count"], 1)
                self.assertEqual(summary["parallel_event_route_count"], 1)
                self.assertEqual(summary["parallel_on_pre"], threads > 1)
                self.assertEqual(summary["parallel_event_enqueue"], threads > 1)
                self.assertEqual(summary["synaptic_events"], 81920)
                profile = summary["phase_profile"]
                self.assertEqual(profile["enabled"], profile_phases)
                self.assertEqual(profile["ticks"], 4 if profile_phases else 0)
                phase_seconds = [value for name, value in profile.items()
                                 if name.endswith("_seconds")]
                self.assertTrue(all(value >= 0 for value in phase_seconds))
                if profile_phases:
                    self.assertGreater(profile["primary_events_seconds"], 0)
                    # Both populations have parameters only. The empty neuron
                    # update phase can finish within one timer tick and be zero.
        for snapshot in snapshots[1:]:
            np.testing.assert_array_equal(snapshots[0], snapshot)

    def test_sparse_spike_fanout_parallelizes_heterogeneous_enqueue(self):
        snapshots = []
        size, degree = 4096, 128
        edge_source = np.repeat(np.arange(size, dtype=np.int32), degree)
        edge_target = np.column_stack([
            (np.arange(size, dtype=np.int32) + offset) % size
            for offset in range(degree)
        ]).reshape(-1)
        delays = (np.arange(edge_source.size) % 2) * .1 * b.ms
        for backend, threads in (("aot", 1), ("aot", 4), ("reference", 1)):
            self.device.reinit()
            b.set_device(
                "rust_standalone", runner=self.runner, engine=backend,
                threads=threads,
                directory=self.directory / f"sparse-fanout-{backend}-{threads}")
            source = b.NeuronGroup(
                size, "v : 1", threshold="v > 0", reset="", dt=.1*b.ms,
                name=f"sparse_fanout_source_{backend}_{threads}")
            target = b.NeuronGroup(
                size, "y : 1", dt=.1*b.ms,
                name=f"sparse_fanout_target_{backend}_{threads}")
            source.v = 1
            synapses = b.Synapses(
                source, target, on_pre="y_post += 0.125",
                clock=source.clock,
                name=f"sparse_fanout_synapses_{backend}_{threads}")
            synapses.connect(i=edge_source, j=edge_target)
            synapses.delay = delays
            b.Network(source, target, synapses).run(.2*b.ms)
            snapshots.append(np.asarray(target.y[:]).copy())
            if backend == "aot":
                summary = json.loads(
                    (self.device.last_run_directory / "rust/summary.json").read_text())
                self.assertEqual(
                    summary["parallel_event_enqueue"], threads > 1)
                self.assertEqual(summary["synaptic_events"], 3 * edge_source.size // 2)
        for snapshot in snapshots[1:]:
            np.testing.assert_array_equal(snapshots[0], snapshot)
        np.testing.assert_array_equal(snapshots[0], np.full(size, 24.0))

    def test_cross_source_fused_delay_routes_preserve_projection_order(self):
        snapshots = []
        size = 4096
        indices = np.arange(size, dtype=np.int32)
        delays = (indices % 2) * .1 * b.ms
        for backend, threads in (("aot", 1), ("aot", 4),
                                 ("reference", 1)):
            self.device.reinit()
            b.set_device(
                "rust_standalone", runner=self.runner, engine=backend,
                threads=threads,
                directory=self.directory / f"cross-source-{backend}-{threads}")
            first_source = b.NeuronGroup(
                size, "v : 1", threshold="v > 0", reset="", dt=.1*b.ms,
                name=f"cross_source_a_{backend}_{threads}")
            second_source = b.NeuronGroup(
                size, "v : 1", threshold="v > 0", reset="", dt=.1*b.ms,
                name=f"cross_source_b_{backend}_{threads}")
            target = b.NeuronGroup(
                size, "y : 1", dt=.1*b.ms,
                name=f"cross_source_target_{backend}_{threads}")
            first_source.v = 1
            second_source.v = 1
            first = b.Synapses(
                first_source, target, on_pre="y_post += 1",
                clock=first_source.clock,
                name=f"a_cross_projection_{backend}_{threads}")
            first.connect(i=indices, j=indices)
            first.delay = delays
            second = b.Synapses(
                second_source, target, on_pre="y_post = 2*y_post + 1",
                clock=second_source.clock,
                name=f"b_cross_projection_{backend}_{threads}")
            second.connect(i=indices, j=indices)
            second.delay = delays
            b.Network(first_source, second_source, target, first, second).run(
                .2*b.ms)
            snapshots.append(np.asarray(target.y[:]).copy())
            if backend == "aot":
                directory = self.device.last_run_directory
                generated = (directory / "native/main.rs").read_text()
                self.assertIn(
                    "route_f0_heterogeneous_target_on_pre", generated)
                summary = json.loads(
                    (directory / "rust/summary.json").read_text())
                self.assertEqual(summary["event_route_count"], 2)
                self.assertEqual(summary["parallel_event_route_count"], 2)
                self.assertEqual(summary["fused_event_dispatch_count"], 1)
                self.assertEqual(summary["parallel_on_pre"], threads > 1)
                self.assertEqual(summary["parallel_event_enqueue"], threads > 1)
                self.assertEqual(summary["synaptic_events"], 3 * size)
        for snapshot in snapshots[1:]:
            np.testing.assert_array_equal(snapshots[0], snapshot)

    def test_on_post_pre_neuron_write_and_unit_mismatch_are_rejected(self):
        cases = [
            ({"on_pre": "v_post += 1*ms"}, "", b.DimensionMismatchError),
            ({"on_pre": "w += 0", "on_post": "v_pre += 0.1"},
             "w : 1", NotImplementedError),
        ]
        for options, model, error in cases:
            group = self.make_group()
            synapse = b.Synapses(group, group, model, clock=group.clock, **options)
            synapse.connect(i=0, j=1)
            net, _, _ = self.network(group, synapse)
            with self.assertRaises(error):
                net.run(2*b.ms)

    def test_unsupported_synaptic_equation_modes_fail_before_build(self):
        cases = [
            ("da/dt=-a/(5*ms):1", "euler", "clock-driven"),
            # Explicit RK2 is supported; a non-default fallback sequence is not.
            ("da/dt=-a/(5*ms):1 (clock-driven)", ("rk2", "euler"),
             "default method selection"),
        ]
        for model, method, message in cases:
            with self.subTest(model=model, method=method):
                group = self.make_group()
                synapse = b.Synapses(group, group, model, on_pre="v_post += a",
                                     method=method, clock=group.clock)
                synapse.connect(i=0, j=1)
                net, state, _ = self.network(group, synapse)
                with patch.object(self.device, "_runner",
                                  side_effect=AssertionError("premature build")):
                    with self.assertRaisesRegex(NotImplementedError, message):
                        net.run(2*b.ms)
                self.assertEqual(len(state.t), 0)

    def test_default_synaptic_method_selection_matches_numpy(self):
        results = []
        for backend in ("reference", "aot", "numpy"):
            self.device.reinit()
            b.start_scope()
            if backend == "numpy":
                b.set_device("runtime")
                b.prefs.codegen.target = "numpy"
            else:
                b.set_device(
                    "rust_standalone", runner=self.runner, engine=backend)
            group = self.make_group()
            group.v = [1.1, 0, 0]
            synapse = b.Synapses(
                group, group,
                "da/dt=-a/(5*ms) : 1 (clock-driven)",
                on_pre="v_post += a", clock=group.clock)
            synapse.connect(i=0, j=1)
            synapse.a = 0.5
            network, state, spikes = self.network(group, synapse)
            network.run(4*b.ms)
            results.append((
                np.asarray(synapse.a[:]).copy(),
                np.asarray(group.v[:]).copy(),
                np.asarray(state.v).copy(),
                np.asarray(spikes.t/b.ms).copy(),
            ))
        for actual in results[:2]:
            for value, expected in zip(actual, results[2], strict=True):
                np.testing.assert_allclose(value, expected, rtol=1e-12, atol=1e-14)

    def test_python_free_replay_and_invalid_synaptic_ir(self):
        group = self.make_group()
        group.v = [1.1, 0, 0]
        synapse = b.Synapses(group, group, "w : 1", on_pre="v_post += weight_scale*w",
                            delay=1*b.ms, clock=group.clock, namespace={"weight_scale": 2.0})
        synapse.connect(i=0, j=1)
        synapse.w = 0.6
        net, _, _ = self.network(group, synapse)
        original = lower_network(net, 4*b.ms)
        mutations = [
            (("instance", "synapses", 0, "source", 0), 3),
            (("instance", "synapses", 0, "target"), []),
            (("instance", "synapses", 0, "pathways", 0, "delay"), []),
            (("instance", "synapses", 0, "pathways", 0, "delay_ticks"), []),
            (("instance", "synapses", 0, "pathways", 0,
              "delay_ticks", 0), 2),
            (("instance", "synapses", 0, "parameters", "w"), []),
            (("definition", "synapses", 0, "post_state_aliases", "v_post"), "x"),
            (("definition", "synapses", 0, "parameters", 0, "index_domain"), "neuron"),
            (("definition", "synapses", 0, "source_start"), 1),
            (("definition", "synapses", 0, "name"), "not valid"),
            (("definition", "synapses", 0, "code_objects", 0, "iteration_domain"), "all_neurons"),
            (("definition", "synapses", 0, "code_objects", 0, "vector", 0, "target"), "w"),
            (("definition", "synapses", 0, "code_objects", 0, "effects", "writes"), []),
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
                loaded = load_results(model, output)
                self.assertEqual(loaded["synaptic_events"], 1)
                np.testing.assert_array_equal(
                    loaded["populations"][0]["counts"], [1, 1, 0])
            else:
                self.assertNotEqual(result.returncode, 0, mutation)
                self.assertFalse(output.exists())

    def test_synapse_tick_budget_remains_bounded(self):
        group = self.make_group()
        synapse = b.Synapses(
            group, group, "da/dt = -a/(5*ms) : 1 (clock-driven)",
            on_pre="v_post += a", method="euler", clock=group.clock)
        synapse.connect(i=0, j=1)
        net, _, _ = self.network(group, synapse)
        with patch("brian2_rust.export.MAX_SYNAPSE_TICKS", 4):
            lower_network(net, 4*b.ms)
            with self.assertRaisesRegex(NotImplementedError, "total synapse-ticks"):
                lower_network(net, 5*b.ms)

    def test_fixed_total_topology_is_deferred_and_matches_both_engines(self):
        snapshots = []
        for engine, threads in (("reference", 1), ("aot", 8)):
            self.device.reinit()
            b.set_device(
                "rust_standalone", runner=self.runner, engine=engine,
                threads=threads)
            source = b.NeuronGroup(
                4, "v : 1", threshold="v > 0.5", reset="v = 0",
                dt=1*b.ms)
            target = b.NeuronGroup(3, "x : 1", dt=1*b.ms)
            source.v = 1
            synapse = b.Synapses(
                source, target, "w : 1 (constant)", on_pre="x_post += w",
                clock=source.clock)
            brian2_rust.connect_fixed_total(
                synapse, 65536, seed=0x1234,
                initializers={
                    "w": brian2_rust.ClippedNormal(1.0, 0.1, minimum=0.0)},
                delay_initializer=brian2_rust.ClippedNormal(
                    2*b.ms, 0.5*b.ms, minimum=1*b.ms))
            self.assertEqual(len(synapse), 65536)
            self.assertEqual(
                len(synapse.variables["_synaptic_pre"].get_value()), 0)
            network = b.Network(source, target, synapse)
            model = lower_network(network, 1*b.ms)
            instance = model["instance"]["synapses"][0]
            self.assertEqual(model["schema"], "b2ir-v1")
            self.assertEqual(instance["source"], [])
            self.assertEqual(instance["target"], [])
            self.assertEqual(instance["topology"], {
                "kind": "fixed_total", "edge_count": 65536,
                "seed": 0x1234,
                "initializers": {
                    "w": {
                        "kind": "clipped_normal",
                        "mean": "3ff0000000000000",
                        "std": "3fb999999999999a",
                        "minimum": "0000000000000000",
                        "maximum": None,
                        "stream": 2,
                    }}})
            self.assertIsNotNone(instance["pathways"][0]["delay_initializer"])
            network.run(10*b.ms)
            snapshots.append(np.asarray(target.x[:]).copy())
            if engine == "aot":
                source_text = (self.device.last_run_directory /
                               "native/main.rs").read_text()
                self.assertIn("fn automatic_cpu_order()", source_text)
                self.assertIn("fixed_total_topology(&parallel", source_text)
                self.assertIn("materialize_clipped_normal_ticks(", source_text)
                summary = json.loads((self.device.last_run_directory /
                                      "rust/summary.json").read_text())
                self.assertIsInstance(summary["thread_affinity"], bool)
                self.assertIsInstance(summary["thread_cpus"], list)
        np.testing.assert_array_equal(snapshots[0], snapshots[1])
        self.assertGreater(float(snapshots[0].sum()), 58000.0)
        self.assertLess(float(snapshots[0].sum()), 72000.0)

    def test_fixed_total_rejects_per_edge_state_before_build(self):
        group = self.make_group(size=2)
        synapse = b.Synapses(
            group, group, "w : 1", on_pre="x_post += w", clock=group.clock)
        brian2_rust.connect_fixed_total(synapse, 10, seed=7)
        network, _, _ = self.network(group, synapse)
        with patch.object(
                self.device, "_runner",
                side_effect=AssertionError("premature build")):
            with self.assertRaisesRegex(
                    NotImplementedError, "mutable per-edge state"):
                network.run(1*b.ms)

    def test_fixed_indegree_is_deferred_and_matches_both_engines(self):
        snapshots = []
        for engine, threads in (("reference", 1), ("aot", 4)):
            self.device.reinit()
            b.start_scope()
            b.set_device(
                "rust_standalone", runner=self.runner, engine=engine,
                threads=threads)
            source = b.NeuronGroup(
                5, "v : 1", threshold="v > 0.5", reset="v = 0", dt=1*b.ms)
            target = b.NeuronGroup(7, "x : 1", dt=1*b.ms)
            source.v = 1
            synapse = b.Synapses(
                source, target, on_pre="x_post += 1", clock=source.clock)
            brian2_rust.connect_fixed_indegree(synapse, 3, seed=0x5678)
            self.assertEqual(len(synapse), 21)
            network = b.Network(source, target, synapse)
            model = lower_network(network, 1*b.ms)
            self.assertEqual(model["instance"]["synapses"][0]["topology"], {
                "kind": "fixed_indegree", "edge_count": 21,
                "indegree": 3, "seed": 0x5678, "initializers": {},
            })
            network.run(1*b.ms)
            snapshots.append(np.asarray(target.x[:]).copy())
            if engine == "aot":
                generated = (self.device.last_run_directory /
                             "native/main.rs").read_text()
                self.assertIn("fixed_indegree_topology(&parallel", generated)
        for snapshot in snapshots:
            np.testing.assert_array_equal(snapshot, np.full(7, 3.0))
        np.testing.assert_array_equal(snapshots[0], snapshots[1])

    def test_procedural_uniform_delays_match_both_engines(self):
        snapshots = []
        for engine, threads in (("reference", 1), ("aot", 4)):
            self.device.reinit()
            b.start_scope()
            b.set_device(
                "rust_standalone", runner=self.runner, engine=engine,
                threads=threads)
            source = b.NeuronGroup(
                5, "v : 1", threshold="True", reset="v = 0", dt=1*b.ms)
            target = b.NeuronGroup(7, "x : 1", dt=1*b.ms)
            synapse = b.Synapses(
                source, target, on_pre="x_post += 1", clock=source.clock)
            brian2_rust.connect_fixed_indegree(
                synapse, 3, seed=0x9ABC,
                delay_initializer=brian2_rust.Uniform(0*b.ms, 3*b.ms))
            network = b.Network(source, target, synapse)
            model = lower_network(network, 5*b.ms)
            descriptor = model["instance"]["synapses"][0]["pathways"][0][
                "delay_initializer"]
            self.assertEqual(descriptor["kind"], "uniform")
            self.assertEqual(descriptor["minimum"], "0000000000000000")
            network.run(5*b.ms)
            snapshots.append(np.asarray(target.x[:]).copy())
            if engine == "aot":
                generated = (self.device.last_run_directory /
                             "native/main.rs").read_text()
                self.assertIn("materialize_uniform_ticks(", generated)
        np.testing.assert_array_equal(snapshots[0], snapshots[1])
        self.assertGreater(float(snapshots[0].sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
