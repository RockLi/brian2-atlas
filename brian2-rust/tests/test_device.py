"""Exercise the registered Device via Brian's normal run/result APIs."""

import json
import os
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from weakref import WeakKeyDictionary

import brian2 as b
import numpy as np
from brian2.devices.device import Device, RuntimeDevice, all_devices

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from brian2_rust import (RustStandaloneDevice, capability_report,
                         open_monitor_stream)  # noqa: E402


class DeviceTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.runner = Path(os.environ.get(
            "B2_RUNNER", str(ROOT / "target" / "release" / "b2-runner")))
        self.assertTrue(self.runner.is_file(), "Build the Rust runner first")
        self.previous_target = b.prefs.codegen.target
        self.previous_device = b.get_device()
        self.device = all_devices["rust_standalone"]
        self.device.reinit()
        b.start_scope()
        b.set_device("rust_standalone", runner=self.runner)

    def tearDown(self):
        b.set_device(self.previous_device)
        b.prefs.codegen.target = self.previous_target
        self.device.reinit()
        b.start_scope()

    def make_network(self, spiking=True, **kwargs):
        options = dict(method="euler", dt=0.1 * b.ms,
                       namespace={"drive": 1.5, "tau": 10 * b.ms})
        if spiking:
            options.update(threshold="v > 1", reset="v = 0")
        options.update(kwargs)
        model = options.pop("model", "dv/dt = (drive-v)/tau : 1")
        initial = options.pop("initial", 0)
        group = b.NeuronGroup(1, model, **options)
        group.v = initial
        state = b.StateMonitor(group, "v", record=True)
        spike = b.SpikeMonitor(group) if spiking else None
        return b.Network(group, state, *([spike] if spiking else [])), group, state, spike

    def test_runner_can_be_selected_by_environment(self):
        runner = self.directory / "installed-runner"
        runner.write_text("runner placeholder")
        device = RustStandaloneDevice()
        device.build_options = {}
        with patch.dict(os.environ, {"B2_RUNNER": str(runner)}):
            self.assertEqual(device._runner(), runner.resolve())

    def test_registered_device_runs_and_populates_public_api(self):
        self.assertEqual(RustStandaloneDevice.__bases__, (Device,))
        self.assertIs(b.get_device(), self.device)
        net, group, state, spikes = self.make_network()
        with patch.object(RuntimeDevice, "code_object", side_effect=AssertionError("runtime fallback")):
            net.run(100 * b.ms)
        self.assertEqual(state.v.shape, (1, 1000))
        self.assertEqual(spikes.num_spikes, 9)
        np.testing.assert_array_equal(spikes.count[:], [9])
        np.testing.assert_array_equal(spikes.i[:], np.zeros(9, dtype=int))
        ticks = np.rint(spikes.t / group.clock.dt).astype(int)
        np.testing.assert_array_equal(ticks, np.arange(109, 1000, 110))
        self.assertEqual(float(net.t / b.ms), 100)
        self.assertEqual(float(group.clock.t / b.ms), 100)
        self.assertAlmostEqual(group.v[0], 0.14342688748679328)
        np.testing.assert_array_equal(state[0].v, state.v[0])
        self.assertTrue(self.device.has_been_run)
        path = self.device.last_run_directory
        self.assertTrue((path / "model.json").is_file())
        self.assertEqual(
            json.loads((path / "rust" / "summary.json").read_text())["spike_count"],
            9)

        b.set_device("runtime")
        b.prefs.codegen.target = "numpy"
        reference, ref_group, ref_state, ref_spikes = self.make_network()
        reference.run(100 * b.ms)
        # Old result objects retain their Rust-owned arrays after switching.
        np.testing.assert_allclose(state.v, ref_state.v, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(group.v[:], ref_group.v[:], rtol=1e-12, atol=1e-14)
        np.testing.assert_array_equal(spikes.t[:], ref_spikes.t[:])

    def test_magic_run_defaultclock_and_implicit_namespace(self):
        b.defaultclock.dt = 0.1 * b.ms
        tau = 10 * b.ms
        drive = 1.5
        group = b.NeuronGroup(1, "dv/dt=(drive-v)/tau : 1", threshold="v>1", reset="v=0", method="euler")
        state = b.StateMonitor(group, "v", record=True)
        spikes = b.SpikeMonitor(group)
        b.run(100 * b.ms)
        self.assertEqual(state.v.shape, (1, 1000))
        self.assertEqual(spikes.num_spikes, 9)
        self.assertEqual(float(b.defaultclock.t / b.ms), 100)

    def test_magic_run_accepts_multiple_all_new_models(self):
        b.defaultclock.dt = 1*b.ms

        def run_once(initial):
            group = b.NeuronGroup(
                1, "dv/dt=1/ms : 1", method="euler")
            group.v = initial
            state = b.StateMonitor(group, "v", record=True)
            b.run(2*b.ms)
            return np.asarray(state.v).copy(), float(b.magic_network.t/b.ms)

        first, first_time = run_once(0)
        second, second_time = run_once(10)

        np.testing.assert_array_equal(first, [[0, 1]])
        np.testing.assert_array_equal(second, [[10, 11]])
        self.assertEqual((first_time, second_time), (2.0, 2.0))

    def test_explicit_run_namespace_and_single_tick_with_units(self):
        net, group, state, _ = self.make_network(
            spiking=False, model="dv/dt=-v/tau : volt", initial=5 * b.mV, namespace={})
        net.run(0.1 * b.ms, namespace={"tau": 10 * b.ms})
        self.assertEqual(state.v.shape, (1, 1))
        np.testing.assert_allclose(state.v / b.mV, [[5]])
        np.testing.assert_allclose(group.v[:] / b.mV, [4.95])

    def test_empty_spikes_are_valid_results(self):
        net, _, state, spikes = self.make_network(namespace={"drive": 0.5, "tau": 10 * b.ms})
        net.run(1 * b.ms)
        self.assertEqual(state.v.shape, (1, 10))
        self.assertEqual(spikes.num_spikes, 0)
        self.assertEqual(len(spikes.t), 0)
        np.testing.assert_array_equal(spikes.count[:], [0])

    def test_count_only_spike_monitors_accumulate_without_event_arrays(self):
        self.device.build_options["directory"] = self.directory / "count-only"
        neurons = b.NeuronGroup(
            2, "x : 1", threshold="i == 0", reset="", dt=1*b.ms,
            name="count_neurons")
        generated = b.SpikeGeneratorGroup(
            2, [0, 1, 1], [0, 1, 2]*b.ms, dt=1*b.ms,
            name="count_generated")
        poisson = b.PoissonGroup(
            2, rates=0*b.Hz, dt=1*b.ms, name="count_poisson")
        monitors = [
            b.SpikeMonitor(source, record=False, name=f"{source.name}_counts")
            for source in (neurons, generated, poisson)
        ]
        network = b.Network(neurons, generated, poisson, *monitors)
        report = capability_report(network, 2*b.ms)
        self.assertTrue(report.supported, report.format_text())

        network.run(2*b.ms)
        expected_after_first = ([2, 0], [1, 1], [0, 0])
        for monitor, expected in zip(monitors, expected_after_first, strict=True):
            np.testing.assert_array_equal(monitor.count[:], expected)
            self.assertEqual(monitor.num_spikes, sum(expected))
            self.assertNotIn("i", monitor.variables)
            self.assertNotIn("t", monitor.variables)

        network.run(1*b.ms)
        expected_after_second = ([3, 0], [1, 2], [0, 0])
        for monitor, expected in zip(monitors, expected_after_second, strict=True):
            np.testing.assert_array_equal(monitor.count[:], expected)
            self.assertEqual(monitor.num_spikes, sum(expected))

    def test_optional_and_multiple_state_monitors_share_one_physical_trace(self):
        group = b.NeuronGroup(
            4, """dv/dt = -gain*v/tau : 1
                  x : 1
                  gain : 1 (constant)
                  tau : second (constant, shared)""", method="euler",
            dt=0.1 * b.ms)
        group.v = [1, 2, 3, 4]
        group.x = [10, 20, 30, 40]
        group.gain = [1, 2, 3, 4]
        group.tau = 2 * b.ms
        left = b.StateMonitor(group, "v", record=[2, 0, 2], name="left")
        right = b.StateMonitor(
            group, ["x", "v", "gain", "tau"], record=[3, 2], name="right")
        empty = b.StateMonitor(group, "x", record=False, name="empty")
        network = b.Network(group, left, right, empty)
        network.run(0.3 * b.ms)

        self.assertEqual(left.v.shape, (3, 3))
        self.assertEqual(right.v.shape, (2, 3))
        self.assertEqual(right.x.shape, (2, 3))
        self.assertEqual(empty.x.shape, (0, 3))
        np.testing.assert_allclose(left.v[:, 0], [3, 1, 3])
        np.testing.assert_allclose(right.v[:, 0], [4, 3])
        np.testing.assert_allclose(right.x, [[40, 40, 40], [30, 30, 30]])
        np.testing.assert_allclose(right.gain, [[4, 4, 4], [3, 3, 3]])
        np.testing.assert_allclose(right.tau / b.ms, 2)
        np.testing.assert_allclose(empty.t / b.ms, [0, 0.1, 0.2])

        # A stateful population no longer needs a StateMonitor merely to run.
        self.device.reinit()
        self.device.activate(runner=self.runner)
        unmonitored = b.NeuronGroup(
            1, "dv/dt = -v/(1*ms) : 1", method="euler", dt=0.1 * b.ms)
        unmonitored.v = 1
        b.Network(unmonitored).run(0.2 * b.ms)
        np.testing.assert_allclose(unmonitored.v[:], [0.81])

    def test_foreign_arrays_are_rejected(self):
        b.set_device("runtime")
        net, group, state, _ = self.make_network()
        b.set_device("rust_standalone", runner=self.runner)
        with self.assertRaisesRegex(RuntimeError, "before constructing"):
            net.run(1 * b.ms)
        self.assertIsNone(self.device.last_run_directory)
        self.assertEqual(len(state.t), 0)
        self.assertEqual(group.v[0], 0)

    def test_reinit_rejects_old_model_and_allows_new_model(self):
        net, _, _, _ = self.make_network()
        net.run(1 * b.ms)
        net.run(1 * b.ms)
        self.assertEqual(float(net.t / b.ms), 2)
        self.device.reinit()
        self.device.activate(runner=self.runner)
        with self.assertRaisesRegex(RuntimeError, "old initialization"):
            net.run(1 * b.ms)
        new_net, _, state, _ = self.make_network()
        new_net.run(1 * b.ms)
        self.assertEqual(state.v.shape, (1, 10))

    def test_successive_fresh_explicit_networks_share_one_device(self):
        for engine in ("reference", "aot"):
            with self.subTest(engine=engine):
                self.device.reinit()
                b.start_scope()
                b.set_device(
                    "rust_standalone", runner=self.runner, engine=engine,
                    directory=self.directory / f"fresh-networks-{engine}")
                results = []
                monitors = []
                for slope in (1, 2):
                    group = b.NeuronGroup(
                        1, "dv/dt=slope/ms : 1", method="euler", dt=1*b.ms,
                        namespace={"slope": slope})
                    monitor = b.StateMonitor(group, "v", record=True)
                    network = b.Network(group, monitor)
                    network.run(3*b.ms)
                    results.append(np.asarray(monitor.v).copy())
                    monitors.append(monitor)

                np.testing.assert_array_equal(results[0], [[0, 1, 2]])
                np.testing.assert_array_equal(results[1], [[0, 2, 4]])
                np.testing.assert_array_equal(monitors[0].v, results[0])

                reused = b.NeuronGroup(
                    1, "dv/dt=0/ms : 1", method="euler", dt=1*b.ms)
                first_owner = b.Network(reused)
                second_owner = b.Network(reused)
                first_owner.run(1*b.ms)
                with self.assertRaisesRegex(RuntimeError, "every object.*fresh"):
                    second_owner.run(1*b.ms)

    def test_segmented_run_preserves_time_monitors_and_delayed_events(self):
        snapshots = []
        for backend in ["reference", "aot", "numpy"]:
            self.device.reinit()
            b.start_scope()
            if backend == "numpy":
                b.set_device("runtime")
                b.prefs.codegen.target = "numpy"
            else:
                b.set_device("rust_standalone", runner=self.runner, engine=backend)
            source = b.NeuronGroup(
                1, "dv/dt = t/(10*ms**2) : 1", threshold="v > 1",
                reset="v = 0", refractory=3*b.ms, method="euler", dt=1*b.ms)
            target = b.NeuronGroup(
                1, "dx/dt = 0*Hz : 1", method="euler", dt=1*b.ms)
            source.v = 2
            source_state = b.StateMonitor(source, "v", record=True)
            target_state = b.StateMonitor(target, "x", record=True)
            spikes = b.SpikeMonitor(source)
            synapse = b.Synapses(source, target, on_pre="x_post += 1",
                                 delay=2*b.ms, clock=source.clock)
            synapse.connect()
            synapse_2 = b.Synapses(source, target, on_pre="x_post += 2",
                                   delay=2*b.ms, clock=source.clock)
            synapse_2.connect()
            network = b.Network(source, target, source_state, target_state,
                                spikes, synapse, synapse_2)
            network.run(1*b.ms)
            self.assertEqual(source_state.v.shape, (1, 1))
            self.assertEqual(float(network.t/b.ms), 1)
            network.run(4*b.ms)
            self.assertEqual(source_state.v.shape, (1, 5))
            self.assertEqual(target_state.x.shape, (1, 5))
            self.assertEqual(float(network.t/b.ms), 5)
            snapshots.append([
                np.asarray(source_state.t/b.ms).copy(), source_state.v[:].copy(),
                target_state.x[:].copy(), np.asarray(spikes.t/b.ms).copy(),
                spikes.i[:].copy(), source.v[:].copy(), target.x[:].copy(),
                np.asarray(source.lastspike[:]/b.ms).copy(),
                source.not_refractory[:].copy(),
            ])
        for expected in snapshots[1:]:
            for actual, other in zip(snapshots[0], expected, strict=True):
                np.testing.assert_allclose(actual, other, rtol=1e-12, atol=1e-14)
        np.testing.assert_array_equal(snapshots[0][0], np.arange(5))
        np.testing.assert_array_equal(snapshots[0][2], [[0, 0, 0, 3, 3]])

    def test_partial_clock_intervals_match_numpy_across_continuations(self):
        results = {}
        segments = [10, 10, 30, 1, 49, 50, 25]
        for backend in ("reference", "aot", "numpy"):
            self.device.reinit()
            b.start_scope()
            if backend == "numpy":
                b.set_device("runtime")
                b.prefs.codegen.target = "numpy"
            else:
                b.set_device(
                    "rust_standalone", runner=self.runner, engine=backend,
                    directory=self.directory / f"partial-clocks-{backend}")

            slow = b.NeuronGroup(1, "x : 1", dt=50*b.ms)
            slow.run_regularly("x += 1")
            fast = b.NeuronGroup(1, "y : 1", dt=10*b.ms)
            fast.run_regularly("y += 1")
            slow_state = b.StateMonitor(slow, "x", record=True, dt=10*b.ms)

            generated = b.SpikeGeneratorGroup(
                1, [0, 0, 0], [0, 100, 150]*b.ms, dt=50*b.ms)
            generated_spikes = b.SpikeMonitor(generated)
            target = b.NeuronGroup(1, "z : 1", dt=50*b.ms)
            pathway = b.Synapses(
                generated, target, on_pre="z_post += 1", delay=50*b.ms,
                clock=generated.clock)
            pathway.connect()
            target_state = b.StateMonitor(target, "z", record=True,
                                          dt=10*b.ms)
            network = b.Network(
                slow, fast, slow_state, generated, generated_spikes,
                target, pathway, target_state)

            snapshots = []
            for duration in segments:
                network.run(duration*b.ms)
                snapshots.append((
                    np.asarray([
                        float(network.t/b.ms), float(slow.clock.t/b.ms),
                        int(slow.clock.variables["timestep"].get_value().item()),
                        float(slow.x[0]), float(fast.y[0]), float(target.z[0]),
                    ]),
                    np.asarray(slow_state.t/b.ms).copy(),
                    np.asarray(slow_state.x[0]).copy(),
                    np.asarray(generated_spikes.t/b.ms).copy(),
                    np.asarray(generated_spikes.i).copy(),
                    np.asarray(target_state.t/b.ms).copy(),
                    np.asarray(target_state.z[0]).copy(),
                ))
            results[backend] = snapshots

        for backend in ("reference", "aot"):
            for actual, expected in zip(
                    results[backend], results["numpy"], strict=True):
                for actual_array, expected_array in zip(
                        actual, expected, strict=True):
                    np.testing.assert_array_equal(actual_array, expected_array)

        final = results["numpy"][-1]
        np.testing.assert_allclose(final[3], [0, 100, 150], rtol=0, atol=1e-12)
        # The spike at 150 ms is still pending for its 200 ms delivery tick.
        self.assertEqual(final[0][-1], 2)

    def test_named_stochastic_subexpression_matches_engines(self):
        results = []
        for backend in ("reference", "aot"):
            self.device.reinit()
            b.start_scope()
            b.set_device(
                "rust_standalone", runner=self.runner, engine=backend,
                directory=self.directory / f"named-noise-{backend}")
            b.seed(20260927)
            group = b.NeuronGroup(
                3,
                """dv/dt = -v/(10*ms) + noise/(250*pF) : volt
                   noise = sigma*sqrt(noise_dt)*xi_pop : amp
                   sigma : amp (shared)
                   noise_dt = 1*ms : second (shared)""",
                method="euler", dt=.1*b.ms)
            group.v = [-70, -65, -60]*b.mV
            group.sigma = 100*b.pA
            state = b.StateMonitor(group, "v", record=True)
            network = b.Network(group, state)
            network.run(5*b.ms)
            values = np.asarray(state.v/b.mV).copy()
            self.assertTrue(np.isfinite(values).all())
            results.append(values)

        np.testing.assert_array_equal(results[0], results[1])
        self.assertFalse(np.array_equal(results[0][0], results[0][1]))

    def test_zero_duration_is_a_noop_before_continuation(self):
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
                    directory=self.directory / f"zero-duration-{backend}")
            group = b.NeuronGroup(
                1, "dx/dt = 1/ms : 1", method="euler", dt=1*b.ms)
            state = b.StateMonitor(group, "x", record=True)
            callbacks = []

            @b.network_operation(dt=1*b.ms)
            def operation():
                callbacks.append(float(group.clock.t/b.ms))

            network = b.Network(group, state, operation)
            network.run(0*b.ms)
            self.assertEqual(float(network.t/b.ms), 0)
            self.assertEqual(float(group.x[0]), 0)
            self.assertEqual(state.x.shape, (1, 0))
            self.assertEqual(callbacks, [])
            network.run(2*b.ms)
            results.append((
                np.asarray(state.t/b.ms).copy(), state.x[:].copy(),
                group.x[:].copy(), np.asarray(callbacks)))

        for actual in results[:2]:
            for value, expected in zip(actual, results[2], strict=True):
                np.testing.assert_array_equal(value, expected)

    def test_shared_synaptic_subexpressions_match_engines(self):
        results = []
        for backend in ("reference", "aot"):
            self.device.reinit()
            b.start_scope()
            b.set_device(
                "rust_standalone", runner=self.runner, engine=backend,
                directory=self.directory / f"shared-synapse-{backend}")
            source = b.NeuronGroup(
                2, "v : 1", threshold="v > 0", reset="v = 0", dt=1*b.ms)
            target = b.NeuronGroup(
                2, "dv/dt = current/(1*nA*ms) : 1\ncurrent : amp",
                method="euler", dt=1*b.ms)
            source.v = 1
            synapse = b.Synapses(
                source, target,
                """dg/dt = -g/tau_s : 1 (clock-driven)
                   current_post = J*g : amp (summed)
                   J = 2*nA : amp (shared)
                   tau_s = 2*ms : second (shared)""",
                on_pre="g += 1", method="exact", clock=source.clock)
            synapse.connect(j="i")
            state = b.StateMonitor(target, ["v", "current"], record=True)
            b.Network(source, target, synapse, state).run(5*b.ms)
            results.append((np.asarray(state.v).copy(),
                            np.asarray(state.current/b.nA).copy()))

        for actual, expected in zip(results[1], results[0], strict=True):
            np.testing.assert_array_equal(actual, expected)
        self.assertGreater(results[0][0][0, -1], 0)

    def test_store_restore_preserves_rng_monitors_and_pending_events(self):
        self.device.reinit()
        b.set_device(
            "rust_standalone", runner=self.runner, engine="aot",
            directory=self.directory / "checkpoint")
        b.seed(20260906)
        source = b.NeuronGroup(
            1, "dv/dt=0*Hz : 1", threshold="v > 1", reset="v = 0",
            method="euler", dt=1*b.ms, name="checkpoint_source")
        target = b.NeuronGroup(
            1, "dx/dt=rand()/second : 1", method="euler", dt=1*b.ms,
            name="checkpoint_target")
        source.v = 2
        state = b.StateMonitor(target, "x", record=True,
                               name="checkpoint_state")
        spikes = b.SpikeMonitor(source, name="checkpoint_spikes")
        synapse = b.Synapses(
            source, target, on_pre="x_post += 1", delay=2*b.ms,
            clock=source.clock, name="checkpoint_projection")
        synapse.connect()
        network = b.Network(source, target, synapse, state, spikes)
        network.run(1*b.ms)
        checkpoint = self.directory / "state.pkl"
        network.store("memory")
        network.store("disk", filename=checkpoint)
        self.assertTrue(checkpoint.is_file())

        def continuation():
            network.run(4*b.ms)
            return (np.asarray(state.t/b.ms).copy(), state.x[:].copy(),
                    np.asarray(spikes.t/b.ms).copy(), target.x[:].copy())

        expected = continuation()
        network.restore("memory", restore_random_state=True)
        self.assertEqual(float(network.t/b.ms), 1)
        self.assertEqual(state.x.shape, (1, 1))
        memory = continuation()
        network.restore("disk", filename=checkpoint, restore_random_state=True)
        disk = continuation()
        for actual in (memory, disk):
            for value, reference in zip(actual, expected, strict=True):
                np.testing.assert_array_equal(value, reference)
        np.testing.assert_array_equal(expected[0], np.arange(5))
        self.assertGreater(expected[1][0, 3], 1)

    def test_aot_large_timed_array_uses_binary_include(self):
        values = np.sin(
            np.arange(10_000, dtype=np.float64).reshape(5000, 2) / 17.0)
        results = []
        for backend in ("aot", "numpy"):
            self.device.reinit()
            b.start_scope()
            if backend == "numpy":
                b.set_device("runtime")
                b.prefs.codegen.target = "numpy"
            else:
                b.set_device(
                    "rust_standalone", runner=self.runner, engine="aot",
                    directory=self.directory / "large-timed-array-aot")
            stimulus = b.TimedArray(values, dt=1*b.ms)
            group = b.NeuronGroup(
                2, "dv/dt = stimulus(t, i)/ms : 1", method="euler",
                dt=1*b.ms, namespace={"stimulus": stimulus})
            state = b.StateMonitor(group, "v", record=True)
            b.Network(group, state).run(4*b.ms)
            results.append((np.asarray(state.v).copy(),
                            np.asarray(group.v[:]).copy()))
            if backend == "aot":
                native = self.device.last_run_directory / "native"
                blobs = list(native.glob("timed-array-*.bin"))
                self.assertEqual(len(blobs), 1)
                self.assertEqual(blobs[0].stat().st_size, values.nbytes)
                source = (native / "main.rs").read_text()
                self.assertIn("include_bytes!", source)
                self.assertLess(max(map(len, source.splitlines())), 100_000)
        for actual, expected in zip(results[0], results[1], strict=True):
            np.testing.assert_array_equal(actual, expected)

    def test_recording_window_bounds_state_and_spike_monitors_across_runs(self):
        for engine in ("reference", "aot"):
            self.device.reinit()
            b.start_scope()
            b.set_device(
                "rust_standalone", runner=self.runner, engine=engine,
                recording_window_steps=3,
                directory=self.directory / f"window-{engine}")
            group = b.NeuronGroup(
                1, "dv/dt=0*Hz : 1", threshold="v > 0", reset="v = 1",
                method="euler", dt=1*b.ms, name=f"window_group_{engine}")
            group.v = 1
            state = b.StateMonitor(
                group, "v", record=True, name=f"window_state_{engine}")
            spikes = b.SpikeMonitor(group, name=f"window_spikes_{engine}")
            network = b.Network(group, state, spikes)
            network.run(5*b.ms)
            np.testing.assert_array_equal(state.t/b.ms, [2, 3, 4])
            np.testing.assert_array_equal(spikes.t/b.ms, [2, 3, 4])
            np.testing.assert_array_equal(spikes.count[:], [3])
            network.run(5*b.ms)
            np.testing.assert_array_equal(state.t/b.ms, [7, 8, 9])
            np.testing.assert_array_equal(spikes.t/b.ms, [7, 8, 9])
            np.testing.assert_array_equal(spikes.count[:], [3])
            self.assertEqual(spikes.num_spikes, 3)

        self.device.reinit()
        b.start_scope()
        b.set_device("rust_standalone", runner=self.runner,
                     recording_window_steps=3)
        source = b.NeuronGroup(
            1, "dv/dt=0*Hz : 1", threshold="v > 0", reset="v = 1",
            method="euler", dt=1*b.ms)
        target = b.NeuronGroup(1, "dx/dt=0*Hz : 1", method="euler",
                               dt=1*b.ms)
        source.v = 1
        pathway = b.Synapses(source, target, on_pre="x_post += 1",
                             delay=4*b.ms, clock=source.clock)
        pathway.connect()
        with self.assertRaisesRegex(NotImplementedError, "maximum delay"):
            b.Network(source, target, pathway).run(5*b.ms)

    def test_run_artifacts_can_be_bounded_without_losing_latest_results(self):
        root = self.directory / "bounded-runs"
        self.device.reinit()
        b.start_scope()
        b.set_device(
            "rust_standalone", runner=self.runner, engine="aot",
            directory=root, retain_run_artifacts=False)
        group = b.NeuronGroup(
            1, "dv/dt = 1/ms : 1", method="euler", dt=1*b.ms)
        state = b.StateMonitor(group, "v", record=True)
        network = b.Network(group, state)

        network.run(2*b.ms)
        first = self.device.last_run_directory
        self.assertTrue((first / "rust/summary.json").is_file())
        network.run(2*b.ms)
        second = self.device.last_run_directory

        self.assertEqual(second, root.resolve() / "run-0002")
        self.assertFalse((root / "model.json").exists())
        self.assertTrue((second / "model.json").is_file())
        np.testing.assert_array_equal(state.t/b.ms, [0, 1, 2, 3])
        np.testing.assert_array_equal(group.v[:], [4])

    def test_monitor_streaming_preserves_full_history_in_bounded_chunks(self):
        for engine in ("reference", "aot"):
            with self.subTest(engine=engine):
                self.device.reinit()
                b.start_scope()
                root = self.directory / f"monitor-stream-{engine}"
                b.set_device(
                    "rust_standalone", runner=self.runner, engine=engine,
                    directory=root, monitor_streaming_steps=2)
                group = b.NeuronGroup(
                    1, "dv/dt=0*Hz : 1", threshold="v > 0",
                    reset="v = 1", method="euler", dt=1*b.ms)
                group.v = 1
                state = b.StateMonitor(
                    group, "v", record=True, name="streamed_state")
                spikes = b.SpikeMonitor(group, name="streamed_spikes")
                network = b.Network(group, state, spikes)

                network.run(3*b.ms)
                network.run(2*b.ms)

                stream = open_monitor_stream(root / "monitor-stream")
                self.assertTrue(stream.complete)
                self.assertEqual(
                    stream.monitors, ("streamed_spikes", "streamed_state"))
                state_chunks = list(stream.iter_chunks("streamed_state"))
                spike_chunks = list(stream.iter_chunks("streamed_spikes"))
                self.assertEqual(len(state_chunks), 3)
                np.testing.assert_array_equal(
                    np.concatenate([
                        chunk["arrays"]["t"] for chunk in state_chunks]) / 1e-3,
                    [0, 1, 2, 3, 4])
                np.testing.assert_array_equal(
                    np.concatenate([
                        chunk["arrays"]["v"] for chunk in state_chunks], axis=0),
                    np.ones((5, 1)))
                np.testing.assert_array_equal(
                    np.concatenate([
                        chunk["arrays"]["t"] for chunk in spike_chunks]) / 1e-3,
                    [0, 1, 2, 3, 4])
                np.testing.assert_array_equal(state.t/b.ms, [3, 4])
                np.testing.assert_array_equal(spikes.t/b.ms, [3, 4])
                self.assertEqual(self.device.last_monitor_stream,
                                 (root / "monitor-stream").resolve())
                manifest = json.loads(
                    (root / "monitor-stream/manifest.json").read_text())
                stored = manifest["chunks"][0]["monitors"]["streamed_state"]
                chunk_file = (root / "monitor-stream" /
                              manifest["chunks"][0]["directory"] /
                              stored["file"])
                damaged = bytearray(chunk_file.read_bytes())
                damaged[-1] ^= 1
                chunk_file.write_bytes(damaged)
                with self.assertRaisesRegex(RuntimeError, "sha256 mismatch"):
                    list(open_monitor_stream(
                        root / "monitor-stream").iter_chunks("streamed_state"))

    def test_segmented_run_preserves_heterogeneous_pending_edges(self):
        snapshots = []
        for backend in ["reference", "aot", "numpy"]:
            self.device.reinit()
            b.start_scope()
            if backend == "numpy":
                b.set_device("runtime")
                b.prefs.codegen.target = "numpy"
            else:
                b.set_device("rust_standalone", runner=self.runner, engine=backend)
            source = b.NeuronGroup(
                1, "dv/dt = 0*Hz : 1", threshold="v > 1", reset="v = 0",
                method="euler", dt=1*b.ms)
            target = b.NeuronGroup(2, "dx/dt = 0*Hz : 1", method="euler",
                                   dt=1*b.ms)
            source.v = 2
            source_state = b.StateMonitor(source, "v", record=True)
            target_state = b.StateMonitor(target, "x", record=True)
            spikes = b.SpikeMonitor(source)
            synapse = b.Synapses(source, target, on_pre="x_post += 1",
                                 clock=source.clock)
            synapse.connect(i=[0, 0], j=[0, 1])
            synapse.delay = [2, 4]*b.ms
            network = b.Network(source, target, source_state, target_state,
                                spikes, synapse)
            network.run(1*b.ms)
            network.run(5*b.ms)
            snapshots.append((target_state.x[:].copy(), target.x[:].copy(),
                              np.asarray(spikes.t/b.ms).copy()))
        for expected in snapshots[1:]:
            for actual, other in zip(snapshots[0], expected, strict=True):
                np.testing.assert_allclose(actual, other, rtol=0, atol=1e-14)
        np.testing.assert_array_equal(
            snapshots[0][0], [[0, 0, 0, 1, 1, 1], [0, 0, 0, 0, 0, 1]])

    def test_report_boundaries_and_coarse_profiling_use_brian_api(self):
        reports = []
        net, _, state, _ = self.make_network()
        net.run(1*b.ms, report=lambda elapsed, completed, start, duration:
                reports.append((float(elapsed/b.second), completed,
                                float(start/b.ms), float(duration/b.ms))),
                report_period=.1*b.second, profile=True)

        self.assertEqual(len(reports), 2)
        self.assertEqual(reports[0], (0.0, 0.0, 0.0, 1.0))
        self.assertEqual(reports[1][1:], (1.0, 0.0, 1.0))
        self.assertGreaterEqual(reports[1][0], 0.0)
        profiling = dict(net.profiling_info)
        self.assertEqual(set(profiling),
                         {"rust_simulation_and_recording", "rust_result_dump"})
        self.assertTrue(all(value >= 0*b.second for value in profiling.values()))
        self.assertIn("rust_simulation_and_recording", str(b.profiling_summary(net)))
        self.assertEqual(state.v.shape, (1, 10))

    def test_start_network_operation_can_stop_between_native_segments(self):
        for engine in ("reference", "aot"):
            with self.subTest(engine=engine):
                self.device.reinit()
                b.start_scope()
                b.set_device(
                    "rust_standalone", runner=self.runner, engine=engine,
                    directory=self.directory / f"network-operation-{engine}")
                group = b.NeuronGroup(
                    1, "dv/dt=2/ms : 1", threshold="v >= 1", reset="v = 0",
                    method="euler", dt=1*b.ms)
                spikes = b.SpikeMonitor(group)
                observed = []

                @b.network_operation(dt=2*b.ms, when="start", order=-1)
                def check_spikes(t):
                    observed.append((float(t/b.ms), spikes.num_spikes))
                    if spikes.num_spikes >= 4:
                        b.stop()

                network = b.Network(group, spikes, check_spikes)
                network.run(20*b.ms)

                self.assertEqual(
                    observed, [(0.0, 0), (2.0, 2), (4.0, 4)])
                self.assertEqual(spikes.num_spikes, 4)
                self.assertEqual(float(network.t/b.ms), 4.0)
                self.assertAlmostEqual(
                    self.device._last_run_completed_fraction, 0.2)
                root = (self.directory / f"network-operation-{engine}").resolve()
                self.assertEqual(
                    self.device.last_run_directory, root / "run-0002")
                self.assertTrue(
                    (self.device.last_run_directory / "model.json").is_file())
                self.assertFalse((root / "model.json").exists())
                self.assertEqual(
                    sorted(path.name for path in root.glob("run-*")),
                    ["run-0002"])

    def test_network_operation_accepts_partial_final_callback_interval(self):
        for engine in ("reference", "aot"):
            with self.subTest(engine=engine):
                self.device.reinit()
                b.start_scope()
                b.set_device(
                    "rust_standalone", runner=self.runner, engine=engine,
                    directory=self.directory / f"network-operation-tail-{engine}")
                group = b.NeuronGroup(
                    1, "dv/dt=1/ms : 1", method="euler", dt=0.5*b.ms)
                monitor = b.StateMonitor(group, "v", record=[0])
                observed = []

                @b.network_operation(dt=1*b.ms, when="start", order=-1)
                def callback(t):
                    observed.append(float(t/b.ms))

                network = b.Network(group, monitor, callback)
                network.run(2.5*b.ms)

                np.testing.assert_array_equal(observed, [0, 1, 2])
                np.testing.assert_array_equal(
                    monitor.t/b.ms, [0, 0.5, 1, 1.5, 2])
                np.testing.assert_array_equal(
                    monitor.v, [[0, 0.5, 1, 1.5, 2]])
                self.assertEqual(float(group.v[0]), 2.5)
                self.assertEqual(float(network.t/b.ms), 2.5)
                self.assertEqual(self.device._last_run_completed_fraction, 1.0)

    def test_multiple_start_network_operations_keep_clock_and_schedule_order(self):
        for engine in ("reference", "aot"):
            with self.subTest(engine=engine):
                self.device.reinit()
                b.start_scope()
                b.set_device(
                    "rust_standalone", runner=self.runner, engine=engine,
                    directory=self.directory / f"network-operations-{engine}")
                group = b.NeuronGroup(
                    1, "dv/dt=drive/ms : 1\ndrive : 1 (shared)",
                    method="euler", dt=1*b.ms)
                monitor = b.StateMonitor(group, "drive", record=[0])
                observed = []

                @b.network_operation(
                    dt=2*b.ms, when="start", order=-2, name="fast_callback")
                def fast(t):
                    observed.append(("fast", float(t/b.ms), float(group.v[0])))
                    group.drive = 1 + t/b.ms

                @b.network_operation(
                    dt=3*b.ms, when="start", order=-1, name="slow_callback")
                def slow(t):
                    observed.append(("slow", float(t/b.ms), float(group.v[0])))

                network = b.Network(group, monitor, slow, fast)
                network.run(6*b.ms)

                self.assertEqual(
                    observed,
                    [("fast", 0.0, 0.0), ("slow", 0.0, 0.0),
                     ("fast", 2.0, 2.0), ("slow", 3.0, 5.0),
                     ("fast", 4.0, 8.0)])
                np.testing.assert_array_equal(
                    monitor.drive, [[1, 1, 3, 3, 5, 5]])
                self.assertEqual(float(network.t/b.ms), 6.0)
                self.assertEqual(float(group.v[0]), 18.0)
                root = (self.directory / f"network-operations-{engine}").resolve()
                self.assertEqual(
                    self.device.last_run_directory, root / "run-0004")
                self.assertEqual(
                    sorted(path.name for path in root.glob("run-*")),
                    ["run-0004"])

    def test_start_and_end_network_operations_preserve_tick_boundaries(self):
        for engine in ("reference", "aot"):
            with self.subTest(engine=engine):
                self.device.reinit()
                b.start_scope()
                b.set_device(
                    "rust_standalone", runner=self.runner, engine=engine,
                    directory=self.directory / f"network-operation-end-{engine}")
                group = b.NeuronGroup(
                    1, "dv/dt=drive/ms : 1\ndrive : 1 (shared)",
                    method="euler", dt=1*b.ms)
                monitor = b.StateMonitor(
                    group, "v", record=[0], when="end", order=0)
                observed = []

                @b.network_operation(
                    dt=2*b.ms, when="start", order=-1,
                    name="start_callback")
                def at_start(t):
                    group.drive = 1 + t/b.ms
                    observed.append(("start", float(t/b.ms), float(group.v[0])))

                @b.network_operation(
                    dt=2*b.ms, when="end", order=1,
                    name="end_callback")
                def at_end(t):
                    observed.append((
                        "end", float(t/b.ms), float(group.v[0]),
                        float(monitor.v[0, -1])))
                    if t >= 2*b.ms:
                        b.stop()

                network = b.Network(group, monitor, at_start, at_end)
                network.run(10*b.ms)

                self.assertEqual(observed, [
                    ("start", 0.0, 0.0), ("end", 0.0, 1.0, 1.0),
                    ("start", 2.0, 2.0), ("end", 2.0, 5.0, 5.0),
                ])
                np.testing.assert_array_equal(monitor.t/b.ms, [0, 1, 2])
                np.testing.assert_array_equal(monitor.v, [[1, 2, 5]])
                self.assertEqual(float(network.t/b.ms), 3.0)
                self.assertEqual(float(group.clock.t/b.ms), 3.0)
                self.assertEqual(float(at_end.clock.t/b.ms), 4.0)
                self.assertAlmostEqual(
                    self.device._last_run_completed_fraction, 0.3)

    def test_network_operation_capability_failure_precedes_callback(self):
        group = b.NeuronGroup(
            1, "dv/dt=-v/(1*ms) : 1", method="euler", dt=1*b.ms)
        group.state_updater.method_choice = "gsl_msadams"
        calls = []

        @b.network_operation(dt=1*b.ms, when="start", order=-1)
        def callback():
            calls.append(True)

        network = b.Network(group, callback)
        with self.assertRaisesRegex(NotImplementedError, "population.method"):
            network.run(1*b.ms)
        self.assertEqual(calls, [])

    def test_aot_network_operation_reuses_continuation_binary(self):
        self.device.reinit()
        b.start_scope()
        root = (self.directory / "network-operation-reuse").resolve()
        b.set_device(
            "rust_standalone", runner=self.runner, engine="aot",
            directory=root)
        group = b.NeuronGroup(
            1, "dv/dt=1/ms : 1", method="euler", dt=1*b.ms)
        observed = []

        @b.network_operation(dt=1*b.ms, when="start", order=-1)
        def callback(t):
            observed.append(float(t/b.ms))

        from brian2_rust.protocol import verify_protocol

        with patch("brian2_rust.device.write_canonical",
                   wraps=sys.modules["brian2_rust.device"].write_canonical) as writes:
            b.Network(group, callback).run(4*b.ms)

        np.testing.assert_array_equal(observed, [0, 1, 2, 3])
        self.assertEqual(self.device.last_run_directory, root / "run-0004")
        manifest = json.loads(
            (self.device.last_run_directory / "native/manifest.json").read_text())
        self.assertTrue(manifest["compile_reused"])
        self.assertTrue(manifest["validation_reused"])
        self.assertEqual(writes.call_count, 2)
        verify_protocol(json.loads(
            (self.device.last_run_directory / "model.json").read_text()))
        self.assertEqual(len(self.device._aot_binary_cache), 2)
        self.assertEqual(
            sorted(path.name for path in root.glob("run-*")), ["run-0004"])

    def test_unsupported_configuration_and_semantics_fail(self):
        with self.assertRaisesRegex(NotImplementedError, "boolean"):
            self.device.activate(build_on_run="later")
        with self.assertRaisesRegex(NotImplementedError, "options"):
            self.device.activate(unrecognized=True)
        with self.assertRaisesRegex(NotImplementedError, "threads must"):
            self.device.activate(threads=0)
        with self.assertRaisesRegex(NotImplementedError, "requires engine='aot'"):
            self.device.activate(threads=2, engine="reference")
        with self.assertRaisesRegex(NotImplementedError, "thread_affinity must"):
            self.device.activate(thread_affinity="sometimes")
        with self.assertRaisesRegex(NotImplementedError, "retain_run_artifacts"):
            self.device.activate(retain_run_artifacts="sometimes")
        with self.assertRaisesRegex(NotImplementedError, "recording_window_steps"):
            self.device.activate(recording_window_steps=0)
        with self.assertRaisesRegex(NotImplementedError, "monitor_streaming_steps"):
            self.device.activate(monitor_streaming_steps=0)
        with self.assertRaisesRegex(NotImplementedError, "cannot be combined"):
            self.device.activate(
                directory=self.directory / "invalid-stream",
                monitor_streaming_steps=2, recording_window_steps=2)
        net, group, _, _ = self.make_network()
        group.v = "0.5"
        np.testing.assert_array_equal(group.v[:], np.full(len(group), 0.5))
        with self.assertRaisesRegex(NotImplementedError, "profile must be boolean"):
            net.run(1 * b.ms, profile="yes")
        with self.assertRaisesRegex(NotImplementedError, "report must"):
            net.run(1 * b.ms, report="custom native code")
        net.run(0.15 * b.ms)
        self.assertEqual(net.t, 0.15 * b.ms)
        with self.assertRaises(b.DimensionMismatchError):
            net.run(10)
        self.assertIsNotNone(self.device.last_run_directory)

    def test_unsupported_model_is_checked_before_build_or_execution(self):
        net, _, state, _ = self.make_network(refractory=-1*b.ms)
        with patch.object(self.device, "_runner", side_effect=AssertionError("premature build")):
            with self.assertRaisesRegex(NotImplementedError, "refractory"):
                net.run(1 * b.ms)
        self.assertEqual(len(state.t), 0)

    def test_spatial_neuron_branched_cable_matches_numpy(self):
        b.set_device(
            "rust_standalone", runner=self.runner,
            directory=self.directory / "spatial-reference")

        def make_model():
            morphology = b.Soma(20*b.um)
            morphology.left = b.Cylinder(
                length=100*b.um, diameter=2*b.um, n=3)
            morphology.right = b.Cylinder(
                length=80*b.um, diameter=1*b.um, n=2)
            spatial = b.SpatialNeuron(
                morphology=morphology,
                model="""Im = -gL*(v-EL) : amp/meter**2
                         I : amp (point current)""",
                Cm=1*b.uF/b.cm**2, Ri=100*b.ohm*b.cm,
                method="exponential_euler",
                namespace={"gL": 0.1*b.siemens/b.meter**2,
                           "EL": -70*b.mV},
                dt=0.02*b.ms)
            spatial.v = -70*b.mV
            spatial.I = 0*b.amp
            spatial.I[4] = 0.2*b.nA
            monitor = b.StateMonitor(spatial, ["v", "Ic"], record=True)
            return b.Network(spatial, monitor), spatial, monitor

        network, spatial, monitor = make_model()
        report = capability_report(network, 0.2*b.ms)
        self.assertTrue(report.supported, report.format_text())
        self.assertEqual(report.summary["spatial_neurons"], 1)
        network.run(0.2*b.ms)
        rust_v = np.asarray(monitor.v/b.volt).copy()
        rust_ic = np.asarray(monitor.Ic/(b.amp/b.meter**2)).copy()
        rust_final = np.asarray(spatial.v/b.volt).copy()

        b.set_device("runtime")
        b.prefs.codegen.target = "numpy"
        b.start_scope()
        reference, ref_spatial, ref_monitor = make_model()
        reference.run(0.2*b.ms)
        np.testing.assert_allclose(
            rust_v, ref_monitor.v/b.volt, rtol=2e-13, atol=2e-15)
        np.testing.assert_allclose(
            rust_ic, ref_monitor.Ic/(b.amp/b.meter**2),
            rtol=2e-12, atol=2e-13)
        np.testing.assert_allclose(
            rust_final, ref_spatial.v/b.volt, rtol=2e-13, atol=2e-15)

    def test_synaptic_endpoint_event_writes_edge_state(self):
        self.device.activate(
            runner=self.runner, directory=self.directory / "edge-endpoint")
        neurons = b.NeuronGroup(
            2, "v : 1", threshold="v > 1", reset="v = 0")
        primary = b.Synapses(
            neurons, neurons, "w : 1", on_pre="v_post += w")
        primary.connect(i=[0, 1], j=[1, 0])
        primary.w = [2, 3]
        dopamine = b.SpikeGeneratorGroup(1, [0], [0]*b.ms)
        modulator = b.Synapses(
            dopamine, primary, "gain : 1", on_pre="w_post += gain")
        modulator.connect(i=[0, 0], j=[0, 1])
        modulator.gain = [10, 20]
        network = b.Network(neurons, primary, dopamine, modulator)

        report = capability_report(network, 1*b.ms)

        self.assertEqual(report.issues, ())
        network.run(1*b.ms)
        np.testing.assert_array_equal(primary.w[:], [12, 23])
        model = json.loads(
            (self.device.last_run_directory / "model.json").read_text())
        definitions = model["definition"]["synapses"]
        modulator_definition = next(
            item for item in definitions if item["name"] == modulator.name)
        self.assertEqual(
            definitions[modulator_definition["target_synapse"]]["name"],
            primary.name)

    def test_synaptic_endpoint_summed_variables_are_bidirectional(self):
        self.device.activate(
            runner=self.runner, directory=self.directory / "edge-summed")
        neurons = b.NeuronGroup(
            2, "v : 1", threshold="False", reset="v = 0")
        primary = b.Synapses(
            neurons, neurons, "y : 1\nz : 1", on_pre="y += 0")
        primary.connect(i=[0, 1], j=[1, 0])
        primary.y = [2, 3]
        astrocyte = b.NeuronGroup(1, "a : 1")
        edge_to_population = b.Synapses(
            primary, astrocyte, "a_post = y_pre : 1 (summed)")
        edge_to_population.connect(i=[0, 1], j=[0, 0])
        population_to_edge = b.Synapses(
            astrocyte, primary, "z_post = a_pre : 1 (summed)")
        population_to_edge.connect(i=[0, 0], j=[0, 1])
        network = b.Network(
            neurons, primary, astrocyte,
            edge_to_population, population_to_edge)

        self.assertEqual(capability_report(network, 0.1*b.ms).issues, ())
        network.run(0.1*b.ms)

        np.testing.assert_array_equal(astrocyte.a[:], [5])
        np.testing.assert_array_equal(primary.z[:], [5, 5])

    def test_synaptic_endpoint_summed_variables_are_bilateral(self):
        self.device.activate(
            runner=self.runner, directory=self.directory / "edge-to-edge-summed")

        def make_model():
            neurons = b.NeuronGroup(
                3, "v : 1", threshold="False", reset="v = 0",
                name="edge_neurons")
            left = b.Synapses(
                neurons, neurons, "y : 1\nx : 1", on_pre="y += 0",
                name="b_left_edges")
            left.connect(i=[0, 1, 2], j=[1, 2, 0])
            left.y = [2, 3, 5]
            right = b.Synapses(
                neurons, neurons, "z : 1", on_pre="z += 0",
                name="c_right_edges")
            right.connect(i=[0, 2], j=[2, 1])
            bridge = b.Synapses(
                left, right, "z_post = y_pre : 1 (summed)",
                name="a_edge_bridge")
            bridge.connect(i=[0, 1, 2, 2], j=[1, 0, 1, 1])
            self_bridge = b.Synapses(
                left, left, "x_post = y_pre : 1 (summed)",
                name="d_self_edge_bridge")
            self_bridge.connect(i=[0, 2], j=[1, 0])
            return (b.Network(
                neurons, left, right, bridge, self_bridge),
                left, right, bridge)

        network, left, right, bridge = make_model()
        self.assertEqual(capability_report(network, 0.1*b.ms).issues, ())
        network.run(0.1*b.ms)
        rust_left_x = np.asarray(left.x[:]).copy()
        rust_right_z = np.asarray(right.z[:]).copy()
        np.testing.assert_array_equal(rust_left_x, [5, 2, 0])
        np.testing.assert_array_equal(rust_right_z, [3, 12])
        model = json.loads(
            (self.device.last_run_directory / "model.json").read_text())
        definitions = model["definition"]["synapses"]
        bridge_definition = next(
            item for item in definitions if item["name"] == bridge.name)
        self.assertEqual(
            definitions[bridge_definition["source_synapse"]]["name"],
            left.name)
        self.assertEqual(
            definitions[bridge_definition["target_synapse"]]["name"],
            right.name)

        b.set_device("runtime")
        b.prefs.codegen.target = "numpy"
        b.start_scope()
        reference, ref_left, ref_right, _ = make_model()
        reference.run(0.1*b.ms)
        np.testing.assert_array_equal(rust_left_x, ref_left.x[:])
        np.testing.assert_array_equal(rust_right_z, ref_right.z[:])

    def test_nested_synaptic_edge_endpoint_fails_capability_check(self):
        neurons = b.NeuronGroup(
            2, "a : 1", threshold="False", reset="a = 0")
        primary = b.Synapses(
            neurons, neurons, "y : 1", on_pre="y += 0")
        primary.connect(i=[0], j=[1])
        first_level = b.Synapses(
            primary, neurons,
            "u : 1\na_post = y_pre : 1 (summed)")
        first_level.connect(i=[0], j=[0])
        nested = b.Synapses(
            first_level, neurons, "a_post = u_pre : 1 (summed)")
        nested.connect(i=[0], j=[1])

        report = capability_report(
            b.Network(neurons, primary, first_level, nested), 0.1*b.ms)

        self.assertFalse(report.supported)
        self.assertIn(
            "synapse.endpoint.synapses",
            {issue.code for issue in report.issues})

    def test_summed_variable_can_run_after_groups(self):
        self.device.activate(
            runner=self.runner, directory=self.directory / "summed-after-groups")
        clock = b.Clock(dt=1*b.ms)
        source = b.NeuronGroup(
            2, """dx/dt = 1/ms : 1
                  p : 1 (constant)
                  q = p*x : 1""", method="euler", clock=clock)
        source.p = [2, 3]
        target = b.NeuronGroup(1, "y : 1", clock=clock)
        projection = b.Synapses(
            source, target, "y_post = q_pre : 1 (summed)", clock=clock)
        projection.connect(i=[0, 1], j=[0, 0])
        projection.summed_updaters["y_post"].when = "after_groups"
        network = b.Network(source, target, projection)

        self.assertEqual(capability_report(network, 1*b.ms).issues, ())
        network.run(1*b.ms)

        np.testing.assert_array_equal(source.x[:], [1, 1])
        np.testing.assert_array_equal(target.y[:], [5])

    def test_summed_variable_uses_its_own_clock(self):
        self.device.activate(
            runner=self.runner, directory=self.directory / "summed-own-clock")
        source = b.NeuronGroup(1, "x : 1", dt=1*b.ms)
        target = b.NeuronGroup(1, "y : 1", dt=0.5*b.ms)
        projection = b.Synapses(
            source, target, "y_post = t/ms : 1 (summed)")
        projection.connect()
        network = b.Network(source, target, projection)

        self.assertEqual(capability_report(network, 1*b.ms).issues, ())
        network.run(1*b.ms)

        np.testing.assert_array_equal(target.y[:], [0.5])

    def test_summed_variable_can_read_incoming_degree(self):
        self.device.activate(
            runner=self.runner, directory=self.directory / "summed-degree")
        source = b.NeuronGroup(3, "x : 1")
        target = b.NeuronGroup(2, "y : 1")
        source.x = [2, 4, 9]
        projection = b.Synapses(
            source, target, "y_post = x_pre/N_incoming : 1 (summed)")
        projection.connect(i=[0, 1, 2], j=[0, 0, 1])
        network = b.Network(source, target, projection)

        self.assertEqual(capability_report(network, 0.1*b.ms).issues, ())
        network.run(0.1*b.ms)

        np.testing.assert_array_equal(target.y[:], [3, 9])

    def test_poisson_input_accepts_integral_float_count(self):
        self.device.activate(
            runner=self.runner, directory=self.directory / "poisson-float-count")
        group = b.NeuronGroup(1, "x : 1")
        poisson = b.PoissonInput(
            group, "x", N=1000.0, rate=0*b.Hz, weight=1)
        network = b.Network(group, poisson)

        self.assertEqual(capability_report(network, 0.1*b.ms).issues, ())
        network.run(0.1*b.ms)

        np.testing.assert_array_equal(group.x[:], [0])

    def test_rust_failure_does_not_publish_state_or_fallback(self):
        net, group, state, spikes = self.make_network(namespace={"drive": 1.5, "tau": 0 * b.ms})
        with patch.object(RuntimeDevice, "code_object", side_effect=AssertionError("runtime fallback")):
            with self.assertRaisesRegex(RuntimeError, "non-finite"):
                net.run(1 * b.ms)
        self.assertEqual(group.v[0], 0)
        self.assertEqual(len(state.t), 0)
        self.assertEqual(spikes.num_spikes, 0)
        self.assertEqual(net.t, 0 * b.ms)
        self.assertFalse(self.device.has_been_run)

    def test_existing_directory_is_preserved(self):
        marker = self.directory / "keep.txt"
        marker.write_text("keep")
        self.device.activate(runner=self.runner, directory=self.directory)
        net, _, state, _ = self.make_network()
        with self.assertRaises(FileExistsError):
            net.run(1 * b.ms)
        self.assertEqual(marker.read_text(), "keep")
        self.assertFalse((self.directory / "model.json").exists())
        self.assertEqual(len(state.t), 0)

    def test_reinit_appends_to_complete_device_owned_directory(self):
        output = self.directory / "reused-device-directory"
        self.device.activate(runner=self.runner, engine="aot", directory=output)
        first, _, first_state, _ = self.make_network(spiking=False)
        first.run(1*b.ms)
        first_shape = first_state.v.shape
        first_model = (output / "model.json").read_bytes()

        self.device.reinit()
        b.start_scope()
        b.set_device("rust_standalone", runner=self.runner, engine="aot",
                     directory=output)
        second, _, second_state, _ = self.make_network(spiking=False)
        second.run(1*b.ms)

        self.assertEqual(
            self.device.last_run_directory, output.resolve() / "run-0002")
        self.assertEqual((output / "model.json").read_bytes(), first_model)
        self.assertEqual(first_shape, (1, 10))
        self.assertEqual(second_state.v.shape, (1, 10))

    def test_explicit_build_executes_multiple_queued_runs_once(self):
        self.device.reinit()
        b.start_scope()
        output = self.directory / "queued"
        b.set_device("rust_standalone", build_on_run=False, runner=self.runner,
                     engine="aot", directory=output)
        net, group, state, spikes = self.make_network()
        net.run(1*b.ms)
        net.run(2*b.ms)
        self.assertEqual(float(net.t/b.ms), 3)
        self.assertEqual(len(state.t), 0)
        self.assertEqual(spikes.num_spikes, 0)
        self.device.build()
        self.assertEqual(state.v.shape, (1, 30))
        self.assertEqual(float(group.clock.t/b.ms), 3)
        self.assertTrue((output / "native" / "b2-native").is_file())
        with self.assertRaisesRegex(NotImplementedError, "already been built"):
            self.device.build()

    def test_explicit_build_preserves_namespace_parameters_between_runs(self):
        self.device.reinit()
        b.start_scope()
        output = self.directory / "queued-parameters"
        b.set_device("rust_standalone", build_on_run=False, runner=self.runner,
                     engine="reference", directory=output)
        group = b.NeuronGroup(
            1, "dv/dt=(drive-v)/tau : 1\n"
               "tau : second (constant, shared)",
            method="exact", dt=1*b.ms)
        group.v = 0
        group.tau = 1*b.ms
        state = b.StateMonitor(group, "v", record=True)
        network = b.Network(group, state)

        network.run(2*b.ms, namespace={"drive": 1})
        network.run(2*b.ms, namespace={"drive": 0})
        self.assertEqual(len(state.t), 0)
        self.device.build()

        expected = np.asarray([
            0,
            1 - np.exp(-1),
            1 - np.exp(-2),
            (1 - np.exp(-2))*np.exp(-1),
        ])
        np.testing.assert_allclose(state.v[0], expected, rtol=1e-12, atol=0)
        self.assertAlmostEqual(group.v[0], (1 - np.exp(-2))*np.exp(-2))
        self.assertEqual(float(network.t/b.ms), 4)
        self.assertTrue((output / "run-0002/model.json").is_file())

    def test_queued_parameter_segments_preserve_continuation_state(self):
        snapshots = {}
        for backend in ("reference", "aot", "numpy"):
            with self.subTest(backend=backend):
                self.device.reinit()
                b.start_scope()
                if backend == "numpy":
                    b.set_device("runtime")
                    b.prefs.codegen.target = "numpy"
                else:
                    b.set_device(
                        "rust_standalone", build_on_run=False,
                        runner=self.runner, engine=backend,
                        directory=self.directory / f"queued-state-{backend}")
                source = b.NeuronGroup(
                    1, "dv/dt=drive/ms : 1 (unless refractory)",
                    threshold="v >= 1", reset="v = 0", refractory=2*b.ms,
                    method="euler", dt=1*b.ms, name="queued_source")
                target = b.NeuronGroup(
                    1, "x : 1", dt=1*b.ms, name="queued_target")
                projection = b.Synapses(
                    source, target,
                    "dw/dt=1/ms : 1 (clock-driven)",
                    on_pre="x_post += w", delay=1*b.ms,
                    method="euler", clock=source.clock,
                    name="queued_projection")
                projection.connect()
                projection.w = 0.5
                source_state = b.StateMonitor(
                    source, "v", record=True, name="queued_source_state")
                target_state = b.StateMonitor(
                    target, "x", record=True, name="queued_target_state")
                synapse_state = b.StateMonitor(
                    projection, "w", record=True,
                    name="queued_synapse_state")
                spikes = b.SpikeMonitor(source, name="queued_spikes")
                network = b.Network(
                    source, target, projection, source_state, target_state,
                    synapse_state, spikes)

                network.run(3*b.ms, namespace={"drive": 1})
                network.run(4*b.ms, namespace={"drive": 0})
                if backend != "numpy":
                    self.device.build()
                snapshots[backend] = {
                    "source": np.asarray(source_state.v).copy(),
                    "target": np.asarray(target_state.x).copy(),
                    "synapse": np.asarray(synapse_state.w).copy(),
                    "spikes": np.asarray(spikes.t/b.second).copy(),
                    "final_source": np.asarray(source.v[:]).copy(),
                    "final_target": np.asarray(target.x[:]).copy(),
                    "final_synapse": np.asarray(projection.w[:]).copy(),
                }

        expected = snapshots["numpy"]
        for backend in ("reference", "aot"):
            with self.subTest(comparison=backend):
                actual = snapshots[backend]
                for key in expected:
                    np.testing.assert_allclose(
                        actual[key], expected[key], rtol=1e-12, atol=1e-14)

    def test_explicit_build_can_replay_independent_run_args(self):
        self.device.reinit()
        b.start_scope()
        output = self.directory / "run-args"
        b.set_device("rust_standalone", build_on_run=False, runner=self.runner,
                     engine="aot", directory=output)
        group = b.NeuronGroup(
            1, "dv/dt=-v/tau : 1\ntau : second (constant, shared)",
            method="exact", dt=1*b.ms)
        group.v = 1
        state = b.StateMonitor(group, "v", record=True)
        network = b.Network(group, state)
        network.run(3*b.ms)

        self.device.build(run=False)
        self.assertEqual(len(state.t), 0)
        self.assertTrue((output / "native/b2-native").is_file())

        self.device.run(
            results_directory="fast", run_args={group.tau: 1*b.ms})
        fast = np.asarray(state.v).copy()
        self.assertEqual(fast.shape, (1, 3))
        self.device.run(
            results_directory="slow", run_args={group.tau: 2*b.ms})
        slow = np.asarray(state.v).copy()
        self.assertEqual(slow.shape, (1, 3))
        np.testing.assert_allclose(fast[0], np.exp(-np.arange(3)), rtol=1e-12)
        np.testing.assert_allclose(
            slow[0], np.exp(-np.arange(3)/2), rtol=1e-12)
        self.assertFalse(np.array_equal(fast, slow))
        manifest = json.loads((output / "slow/native/manifest.json").read_text())
        self.assertTrue(manifest["compile_reused"])

        before_failure = np.asarray(state.v).copy()
        with self.assertRaisesRegex(ValueError, "must be finite"):
            self.device.run(
                results_directory="invalid-nan",
                run_args={group.tau: np.nan*b.ms})
        np.testing.assert_array_equal(state.v, before_failure)

        with self.assertRaises(b.DimensionMismatchError):
            self.device.run(run_args={group.tau: 1*b.volt})
        with self.assertRaisesRegex(TypeError, "Incorrect size"):
            self.device.run(run_args={group.tau: [1, 2]*b.ms})

    def test_explicit_build_device_is_pickleable_for_workers(self):
        self.device.reinit()
        b.start_scope()
        output = self.directory / "pickled-run-args"
        b.set_device("rust_standalone", build_on_run=False, runner=self.runner,
                     engine="aot", directory=output)
        group = b.NeuronGroup(
            1, "dv/dt=-v/tau : 1\ntau : second (constant, shared)",
            method="exact", dt=1*b.ms, name="worker_group")
        group.v = 1
        state = b.StateMonitor(group, "v", record=True, name="worker_state")
        b.Network(group, state).run(2*b.ms)
        self.device.build(run=False)

        restored = pickle.loads(pickle.dumps(self.device))
        self.assertIsInstance(restored.arrays, WeakKeyDictionary)
        restored_group = next(
            obj for obj in restored._queued_roots if obj.name == "worker_group")
        restored_state = next(
            obj for obj in restored._queued_roots if obj.name == "worker_state")
        restored.run(
            results_directory="worker", run_args={restored_group.tau: 2*b.ms})
        np.testing.assert_allclose(restored_state.v[0], [1, np.exp(-0.5)])


class CompilerGateTest(unittest.TestCase):
    @staticmethod
    def rustc_result(release="1.98.1", host="aarch64-apple-darwin"):
        return SimpleNamespace(stdout=(
            f"rustc {release}\n"
            f"release: {release}\n"
            f"host: {host}\n"))

    def test_pinned_rustc_accepts_exact_native_release(self):
        with (patch.object(RustStandaloneDevice, "_invoke",
                           return_value=self.rustc_result()),
              patch("brian2_rust.device.platform.machine", return_value="arm64")):
            verbose, host = RustStandaloneDevice._pinned_rustc()
        self.assertIn("release: 1.98.1", verbose)
        self.assertEqual(host, "aarch64-apple-darwin")

    def test_pinned_rustc_rejects_older_release(self):
        with (patch.object(RustStandaloneDevice, "_invoke",
                           return_value=self.rustc_result(release="1.86.0")),
              patch("brian2_rust.device.platform.machine", return_value="arm64")):
            with self.assertRaisesRegex(NotImplementedError,
                                        "requires rustc 1.98.1"):
                RustStandaloneDevice._pinned_rustc()

    def test_pinned_rustc_rejects_non_native_architecture(self):
        with (patch.object(RustStandaloneDevice, "_invoke",
                           return_value=self.rustc_result(
                               host="x86_64-apple-darwin")),
              patch("brian2_rust.device.platform.machine", return_value="arm64")):
            with self.assertRaisesRegex(NotImplementedError,
                                        "requires native aarch64 rustc"):
                RustStandaloneDevice._pinned_rustc()


if __name__ == "__main__":
    unittest.main()
