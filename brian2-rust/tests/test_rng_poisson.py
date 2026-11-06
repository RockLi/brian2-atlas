"""Counter-based runtime RNG and constant-rate PoissonGroup conformance."""

import copy
import json
import subprocess
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

import brian2 as b
import numpy as np
from brian2.devices.device import all_devices
from brian2.input.binomial import BinomialFunction

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import brian2_rust  # noqa: E402,F401
from brian2_rust.spec import bits  # noqa: E402


class RngPoissonTest(unittest.TestCase):
    def test_binomial_parameter_cache_invalidates_without_changing_draws(self):
        from brian2_rust.native import RUNTIME

        # Compare the cached vector sampler against the unchanged scalar
        # sampler across parameter/approximation changes and normal-cache use.
        structures = RUNTIME[RUNTIME.index('#[derive(Default)]\nstruct NormalCache'):
                             RUNTIME.index('fn parse_cpu_list')]
        functions = RUNTIME[RUNTIME.index('fn mix64('):
                            RUNTIME.index('fn log_gamma_positive(')]
        source = structures + functions + r'''
fn main() {
    let mut cache=NormalCache::default();
    let cases=[(0,0.0),(0,0.3),(1,0.0),(1,1.0),(1000,0.00045),
               (1000,0.0008),(1000,0.99955),(100,0.1),(100,0.9),
               (100,0.05),(100,0.050000001),(2,0.5),(1000,0.49)];
    for tick in 0..4u64 {
        for &(n,p) in &cases {
            for approximate in [false,true,false] {
                for index in 0..128u64 {
                    let expected=counter_binomial(19,3,tick,index,n,p,approximate);
                    let actual=counter_binomial_cached(19,3,tick,index,n,p,approximate,&mut cache);
                    assert_eq!(actual.to_bits(),expected.to_bits(),"n={n},p={p},i={index}");
                    if index%7==0 {
                        assert_eq!(counter_normal_cached(5,8,tick,index,&mut cache).to_bits(),
                                   counter_normal(5,8,tick,index).to_bits());
                    }
                }
            }
        }
    }
}
'''
        directory = Path(self.temp.name)
        path = directory/'cache.rs'
        path.write_text(source)
        binary = directory/'cache'
        subprocess.run(['rustc', '--edition=2021', '-O', str(path), '-o', str(binary)],
                       check=True, capture_output=True)
        subprocess.run([str(binary)], check=True, capture_output=True, timeout=30)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.previous = b.get_device()
        self.previous_target = b.prefs.codegen.target
        self.device = all_devices["rust_standalone"]

    def tearDown(self):
        b.set_device(self.previous)
        b.prefs.codegen.target = self.previous_target
        self.device.reinit()
        b.start_scope()

    def select(self, backend, name, threads=1):
        self.device.reinit()
        b.start_scope()
        b.set_device(
            "rust_standalone",
            runner=ROOT / "target/release/b2-runner",
            directory=Path(self.temp.name) / name,
            engine=backend,
            threads=threads,
        )

    def poisson(self, backend, name, segmented=False, seed=1729):
        self.select(backend, name)
        b.seed(seed)
        source = b.PoissonGroup(
            64,
            np.linspace(0, 1000, 64) * b.Hz,
            dt=0.1 * b.ms,
            name="poisson_source",
        )
        spikes = b.SpikeMonitor(source, name="poisson_spikes")
        network = b.Network(source, spikes)
        if segmented:
            network.run(2 * b.ms)
            network.run(3 * b.ms)
        else:
            network.run(5 * b.ms)
        return (
            np.asarray(spikes.i[:]).copy(),
            np.rint(spikes.t[:] / source.clock.dt).astype(np.int64),
            np.asarray(spikes.count[:]).copy(),
        )

    def test_reference_aot_and_segmented_runs_are_bitwise_reproducible(self):
        continuous = self.poisson("reference", "reference-continuous")
        aot = self.poisson("aot", "aot-continuous")
        segmented = self.poisson("aot", "aot-segmented", segmented=True)
        for actual in (aot, segmented):
            for actual_array, expected_array in zip(actual, continuous, strict=True):
                np.testing.assert_array_equal(actual_array, expected_array)
        self.assertGreater(len(continuous[0]), 0)

    def test_seed_changes_stream_and_probability_extremes_are_exact(self):
        first = self.poisson("reference", "seed-one", seed=1)
        second = self.poisson("reference", "seed-two", seed=2)
        self.assertFalse(np.array_equal(first[0], second[0]) and
                         np.array_equal(first[1], second[1]))

        self.select("aot", "extremes")
        b.seed(9)
        source = b.PoissonGroup(
            2, [0, 10_000] * b.Hz, dt=0.1 * b.ms,
            name="extreme_source")
        spikes = b.SpikeMonitor(source)
        b.Network(source, spikes).run(1 * b.ms)
        np.testing.assert_array_equal(spikes.count[:], [0, 10])
        np.testing.assert_array_equal(spikes.i[:], np.ones(10, dtype=np.int32))

    def test_poisson_group_drives_neuron_group_synapses(self):
        results = []
        for backend in ("reference", "aot"):
            self.select(backend, f"drive-{backend}")
            b.seed(20260904)
            source = b.PoissonGroup(
                8, np.arange(1, 9) * 100 * b.Hz, dt=0.1 * b.ms,
                name="a_source")
            target = b.NeuronGroup(
                8, "dv/dt=0/second : 1", method="euler", dt=0.1 * b.ms,
                name="z_target")
            synapses = b.Synapses(
                source, target, on_pre="v_post += 1", delay=0 * b.ms,
                clock=source.clock, name="poisson_to_target")
            synapses.connect(i=np.arange(8), j=np.arange(8))
            state = b.StateMonitor(target, "v", record=True)
            spikes = b.SpikeMonitor(source)
            b.Network(source, target, synapses, state, spikes).run(3 * b.ms)
            results.append((np.asarray(target.v[:]).copy(),
                            np.asarray(state.v).copy(),
                            np.asarray(spikes.i[:]).copy(),
                            np.rint(spikes.t[:] / source.clock.dt).astype(np.int64)))
        for actual, expected in zip(results[1], results[0], strict=True):
            np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(results[0][0], np.bincount(
            results[0][2], minlength=8))

    def spike_generator(self, backend, name, segmented=False):
        if backend == "numpy":
            self.device.reinit()
            b.start_scope()
            b.set_device("runtime")
            b.prefs.codegen.target = "numpy"
        else:
            self.select(backend, name)
        source = b.SpikeGeneratorGroup(
            3, [2, 0, 1], [2, 0, 1] * b.ms, period=4 * b.ms,
            dt=1 * b.ms, name="generated_source")
        target = b.NeuronGroup(
            3, "dv/dt=0/second : 1", method="euler", dt=1 * b.ms,
            name="generated_target")
        projection = b.Synapses(
            source, target, on_pre="v_post += 1", clock=source.clock,
            name="generated_projection")
        projection.connect(i=np.arange(3), j=np.arange(3))
        source_spikes = b.SpikeMonitor(source)
        target_state = b.StateMonitor(target, "v", record=True)
        network = b.Network(source, target, projection, source_spikes,
                            target_state)
        if segmented:
            network.run(3 * b.ms)
            network.run(6 * b.ms)
        else:
            network.run(9 * b.ms)
        return tuple(np.asarray(value).copy() for value in (
            source_spikes.i[:], source_spikes.t[:] / b.ms,
            source_spikes.count[:], target_state.v, target.v[:]))

    def test_spike_generator_periodic_segmented_and_synaptic_drive_match_numpy(self):
        reference = self.spike_generator(
            "reference", "generator-reference")
        aot = self.spike_generator("aot", "generator-aot")
        segmented = self.spike_generator(
            "aot", "generator-segmented", segmented=True)
        numpy = self.spike_generator("numpy", "generator-numpy")
        for actual in (aot, segmented, numpy):
            for actual_array, expected_array in zip(
                    actual, reference, strict=True):
                np.testing.assert_array_equal(actual_array, expected_array)
        np.testing.assert_array_equal(reference[0], [0, 1, 2, 0, 1, 2, 0])
        np.testing.assert_array_equal(reference[1], [0, 1, 2, 4, 5, 6, 8])

    def test_spike_generator_set_spikes_between_runs_matches_numpy(self):
        results = []
        for backend in ("reference", "aot", "numpy"):
            if backend == "numpy":
                self.device.reinit()
                b.start_scope()
                b.set_device("runtime")
                b.prefs.codegen.target = "numpy"
            else:
                self.select(backend, f"generator-update-{backend}")
            source = b.SpikeGeneratorGroup(
                2, [0], [0] * b.ms, dt=1 * b.ms)
            spikes = b.SpikeMonitor(source)
            network = b.Network(source, spikes)
            network.run(2 * b.ms)
            source.set_spikes([1, 0], [2, 3] * b.ms)
            network.run(2 * b.ms)
            results.append((np.asarray(spikes.i[:]).copy(),
                            np.asarray(spikes.t[:] / b.ms).copy()))
        for actual in results[1:]:
            for value, expected in zip(actual, results[0], strict=True):
                np.testing.assert_array_equal(value, expected)
        np.testing.assert_array_equal(results[0][0], [0, 1, 0])
        np.testing.assert_array_equal(results[0][1], [0, 2, 3])

    def test_poisson_population_matches_declared_rate_statistically(self):
        self.select("reference", "statistics")
        b.seed(314159)
        neurons, steps, probability = 1024, 1000, 0.01
        source = b.PoissonGroup(
            neurons, 100 * b.Hz, dt=0.1 * b.ms, name="statistical_source")
        spikes = b.SpikeMonitor(source)
        b.Network(source, spikes).run(steps * source.clock.dt)
        expected = neurons * steps * probability
        sigma = np.sqrt(neurons * steps * probability * (1 - probability))
        self.assertLess(abs(spikes.num_spikes - expected), 6 * sigma)

    def test_runtime_rand_in_neuron_equation_uses_same_counter_contract(self):
        final = []
        for backend in ("reference", "aot"):
            self.select(backend, f"equation-{backend}")
            b.seed(77)
            group = b.NeuronGroup(
                16, "dv/dt=rand()/ms : 1\ndw/dt=rand()/ms : 1",
                method="euler", dt=0.1 * b.ms)
            state = b.StateMonitor(group, ["v", "w"], record=True)
            b.Network(group, state).run(1 * b.ms)
            final.append((np.asarray(state.v).copy(), np.asarray(state.w).copy(),
                          np.asarray(group.v[:]).copy(), np.asarray(group.w[:]).copy()))
        for actual, expected in zip(final[1], final[0], strict=True):
            np.testing.assert_array_equal(actual, expected)

    def normal(self, backend, name, segmented=False):
        self.select(backend, name)
        b.seed(8128)
        group = b.NeuronGroup(
            256, "dv/dt=randn()/ms : 1\ndw/dt=randn()/ms : 1",
            method="euler", dt=0.1 * b.ms, name="normal_group")
        state = b.StateMonitor(group, ["v", "w"], record=True)
        network = b.Network(group, state)
        if segmented:
            network.run(4 * b.ms)
            network.run(6 * b.ms)
        else:
            network.run(10 * b.ms)
        return (np.asarray(state.v).copy(), np.asarray(state.w).copy(),
                np.asarray(group.v[:]).copy(), np.asarray(group.w[:]).copy())

    def test_runtime_randn_is_reproducible_and_statistically_normal(self):
        reference = self.normal("reference", "normal-reference")
        aot = self.normal("aot", "normal-aot")
        segmented = self.normal("aot", "normal-segmented", segmented=True)
        for actual in (aot, segmented):
            for actual_array, expected_array in zip(actual, reference, strict=True):
                np.testing.assert_array_equal(actual_array, expected_array)
        draws = np.diff(np.concatenate(
            (np.zeros((256, 1)), reference[0], reference[2][:, None]), axis=1),
            axis=1).ravel() / 0.1
        self.assertLess(abs(draws.mean()), 0.03)
        self.assertLess(abs(draws.var() - 1), 0.05)

    def poisson_input(self, backend, name, segmented=False):
        self.select(backend, name)
        b.seed(271828)
        group = b.NeuronGroup(
            32, "dv/dt=0/second : 1", method="euler", dt=0.1 * b.ms,
            name="input_target")
        first = b.PoissonInput(
            group, "v", N=10, rate=500 * b.Hz,
            weight="0.1 + 0.001*i")
        second = b.PoissonInput(
            group, "v", N=100, rate=1000 * b.Hz,
            weight=0.01)
        state = b.StateMonitor(group, "v", record=True)
        network = b.Network(group, first, second, state)
        if segmented:
            network.run(2 * b.ms)
            network.run(3 * b.ms)
        else:
            network.run(5 * b.ms)
        return np.asarray(state.v).copy(), np.asarray(group.v[:]).copy()

    def test_poisson_input_exact_and_approximate_paths_match_all_backends(self):
        reference = self.poisson_input("reference", "input-reference")
        aot = self.poisson_input("aot", "input-aot")
        segmented = self.poisson_input("aot", "input-segmented", segmented=True)
        for actual in (aot, segmented):
            for actual_array, expected_array in zip(actual, reference, strict=True):
                np.testing.assert_array_equal(actual_array, expected_array)
        self.assertGreater(reference[1].sum(), 0)

    def test_poisson_input_contiguous_subgroup_masks_parent_writes(self):
        outputs = []
        for backend in ("reference", "aot"):
            self.select(backend, f"input-subgroup-{backend}")
            group = b.NeuronGroup(
                5, "dv/dt=0/second : 1", method="euler", dt=1*b.ms,
                name="subgroup_input_target")
            subgroup = group[1:4]
            poisson_input = b.PoissonInput(
                subgroup, "v", N=1, rate=1000*b.Hz, weight=1)
            state = b.StateMonitor(group, "v", record=True)
            b.Network(group, poisson_input, state).run(4*b.ms)
            outputs.append((np.asarray(state.v).copy(),
                            np.asarray(group.v[:]).copy()))
        for actual, expected in zip(outputs[1], outputs[0], strict=True):
            np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(outputs[0][1], [0, 4, 4, 4, 0])

    def test_poisson_input_probability_is_a_hot_instance_parameter(self):
        self.select("aot", "input-hot-probability")
        group = b.NeuronGroup(
            4, "dv/dt=0/second : 1", method="euler", dt=0.1 * b.ms,
            name="hot_input_target")
        poisson_input = b.PoissonInput(
            group, "v", N=3, rate=0 * b.Hz, weight=1)
        state = b.StateMonitor(group, "v", record=[0], name="hot_state")
        b.Network(group, poisson_input, state).run(0.1 * b.ms)

        directory = self.device.last_run_directory
        model = json.loads((directory / "model.json").read_text())
        population = model["definition"]["populations"][0]
        binomials = []

        def visit(value):
            if isinstance(value, dict):
                if value.get("op") == "binomial":
                    binomials.append(value)
                for child in value.values():
                    visit(child)
            elif isinstance(value, list):
                for child in value:
                    visit(child)

        visit(population["code_objects"])
        self.assertEqual(len(binomials), 1)
        probability = binomials[0]["p"]
        self.assertEqual(probability["op"], "load")
        probability_name = probability["name"]
        self.assertIn(
            probability_name,
            {parameter["name"] for parameter in population["parameters"]})

        changed = copy.deepcopy(model)
        changed["instance"]["populations"][0]["parameters"][probability_name] = [
            bits(1.0)]
        run = brian2_rust.run_compatible_instance(
            changed,
            directory / "native",
            Path(self.temp.name) / "hot-probability.bin",
            Path(self.temp.name) / "hot-probability-output",
        )
        np.testing.assert_array_equal(
            run["results"]["populations"][0]["states"]["v"],
            np.full(4, 3.0),
        )

        invalid = copy.deepcopy(changed)
        invalid["instance"]["populations"][0]["parameters"][probability_name] = [
            bits(1.01)]
        invalid_output = Path(self.temp.name) / "invalid-probability-output"
        with self.assertRaisesRegex(RuntimeError, "AOT instance execution failed"):
            brian2_rust.run_compatible_instance(
                invalid,
                directory / "native",
                Path(self.temp.name) / "invalid-probability.bin",
                invalid_output,
            )
        self.assertFalse(invalid_output.exists())

    def test_poisson_input_matches_declared_binomial_rate_statistically(self):
        self.select("reference", "input-statistics")
        b.seed(1618033)
        neurons, steps, inputs, probability = 512, 1000, 20, 0.01
        group = b.NeuronGroup(
            neurons, "dv/dt=0/second : 1", method="euler", dt=0.1 * b.ms)
        poisson_input = b.PoissonInput(
            group, "v", N=inputs, rate=100 * b.Hz, weight=1)
        state = b.StateMonitor(group, "v", record=[0])
        b.Network(group, poisson_input, state).run(steps * group.clock.dt)
        expected = neurons * steps * inputs * probability
        sigma = np.sqrt(neurons * steps * inputs * probability * (1 - probability))
        self.assertLess(abs(np.asarray(group.v[:]).sum() - expected), 6 * sigma)

    def test_poisson_input_parallel_range_and_normal_pair_cache_are_exact(self):
        dumps, states = [], []
        size = 8192
        for threads in (1, 4, 8):
            self.select("aot", f"input-parallel-{threads}", threads=threads)
            b.seed(424242)
            group = b.NeuronGroup(
                size, "dv/dt=0/second : 1", method="euler", dt=.1*b.ms,
                name="parallel_input_target")
            poisson_input = b.PoissonInput(
                group, "v", N=100, rate=1000*b.Hz, weight=.01)
            monitor = b.StateMonitor(group, "v", record=[0, size//2, size-1])
            b.Network(group, poisson_input, monitor).run(1*b.ms)
            directory = self.device.last_run_directory
            dumps.append((directory / "rust/results.bin").read_bytes())
            states.append((np.asarray(group.v[:]).copy(),
                           np.asarray(monitor.v).copy()))
            summary = json.loads((directory / "rust/summary.json").read_text())
            self.assertEqual(summary["threads"], threads)
            self.assertEqual(summary["parallel_poisson_input"], threads > 1)
            if threads == 8:
                generated = (directory / "native/main.rs").read_text()
                self.assertIn("counter_normal_cached", generated)
                self.assertIn("counter_binomial_cached", generated)
                self.assertIn("parallel.for_each(8192", generated)
        for dump in dumps[1:]:
            self.assertEqual(dumps[0], dump)
        for actual in states[1:]:
            for value, expected in zip(actual, states[0], strict=True):
                np.testing.assert_array_equal(value, expected)

    def test_poisson_input_runs_after_on_pre_and_before_reset(self):
        for backend in ("reference", "aot"):
            self.select(backend, f"input-schedule-{backend}")
            source = b.NeuronGroup(
                1, "dx/dt=0/second : 1", threshold="True",
                method="euler", dt=0.1 * b.ms, name="a_source")
            target = b.NeuronGroup(
                1, "dv/dt=0/second : 1", method="euler",
                dt=0.1 * b.ms, name="z_target")
            target.v = 1
            synapses = b.Synapses(
                source, target, on_pre="v_post *= 2", clock=source.clock)
            synapses.connect()
            poisson_input = b.PoissonInput(
                target, "v", N=1, rate=10_000 * b.Hz, weight=1)
            source_state = b.StateMonitor(source, "x", record=True)
            target_state = b.StateMonitor(target, "v", record=True)
            b.Network(source, target, synapses, poisson_input,
                      source_state, target_state).run(0.1 * b.ms)
            self.assertEqual(target.v[0], 3)

    def test_poisson_input_respects_unless_refractory(self):
        results = []
        for backend in ("reference", "aot", "numpy"):
            if backend == "numpy":
                self.device.reinit()
                b.start_scope()
                b.set_device("runtime")
                b.prefs.codegen.target = "numpy"
            else:
                self.select(backend, f"input-refractory-{backend}")
            group = b.NeuronGroup(
                1, "dv/dt=0/second : 1 (unless refractory)",
                threshold="v > 0.5", reset="v = 0", refractory=0.3*b.ms,
                method="euler", dt=0.1*b.ms, name="refractory_target")
            group.v = 1
            poisson_input = b.PoissonInput(
                group, "v", N=1, rate=10_000*b.Hz, weight=1)
            state = b.StateMonitor(group, "v", record=True)
            spikes = b.SpikeMonitor(group)
            b.Network(group, poisson_input, state, spikes).run(0.5*b.ms)
            results.append((
                np.asarray(state.v).copy(),
                np.rint(spikes.t[:] / group.clock.dt).astype(np.int64),
                np.asarray(group.v[:]).copy(),
            ))
        for actual in results[1:]:
            for value, expected in zip(actual, results[0], strict=True):
                np.testing.assert_array_equal(value, expected)
        np.testing.assert_array_equal(results[0][1], [0, 4])

    def test_poisson_input_probability_outside_unit_interval_fails_closed(self):
        self.select("reference", "input-probability")
        group = b.NeuronGroup(
            4, "dv/dt=0/second : 1", method="euler", dt=1 * b.ms)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            poisson_input = b.PoissonInput(
                group, "v", N=2, rate=2000 * b.Hz, weight=1)
        state = b.StateMonitor(group, "v", record=True)
        with self.assertRaisesRegex(NotImplementedError, r"probability in \[0, 1\]"):
            b.Network(group, poisson_input, state).run(1 * b.ms)

    def test_expression_and_timed_array_poisson_rates_use_runtime_values(self):
        expected_indices = np.array([0, 1, 0, 1])
        expected_ticks = np.array([0, 1, 2, 2])
        for backend in ("reference", "aot"):
            self.select(backend, f"dynamic-rates-{backend}")
            stimulus = b.TimedArray(
                [[10_000, 0], [0, 10_000], [10_000, 10_000], [0, 0]] * b.Hz,
                dt=0.1 * b.ms)
            source = b.PoissonGroup(2, "stimulus(t, i)", dt=0.1 * b.ms)
            spikes = b.SpikeMonitor(source)
            b.Network(source, spikes).run(0.4 * b.ms)
            np.testing.assert_array_equal(spikes.i[:], expected_indices)
            np.testing.assert_array_equal(
                np.rint(spikes.t[:] / source.clock.dt), expected_ticks)

        self.select("reference", "index-rates")
        source = b.PoissonGroup(4, "(1.0*i + 1.0)*100*Hz", dt=0.1 * b.ms)
        spikes = b.SpikeMonitor(source)
        b.Network(source, spikes).run(1 * b.ms)

    def test_direct_runtime_poisson_is_integer_and_reproducible(self):
        results = []
        for backend in ("reference", "aot"):
            self.select(backend, f"direct-poisson-{backend}")
            b.seed(1729)
            group = b.NeuronGroup(
                64,
                "dv/dt=poisson(2.0 + i/N)/second : 1",
                method="euler", dt=1*b.ms)
            state = b.StateMonitor(group, "v", record=True)
            b.Network(group, state).run(200*b.ms)
            values = np.asarray(group.v[:]).copy() * 1000
            self.assertTrue(np.allclose(values, np.rint(values), atol=1e-10))
            self.assertTrue(2.3 < values.mean()/200 < 2.7)
            results.append((np.asarray(state.v).copy(), values))
        for left, right in zip(results[0], results[1], strict=True):
            np.testing.assert_array_equal(left, right)

    def test_namespace_binomial_function_is_reproducible(self):
        results = []
        for backend in ("reference", "aot"):
            self.select(backend, f"direct-binomial-{backend}")
            b.seed(1729)
            exact = BinomialFunction(
                100, 0.1, approximate=False, name=f"exact_{backend}")
            approximate = BinomialFunction(
                100, 0.1, approximate=True, name=f"approximate_{backend}")
            group = b.NeuronGroup(
                64, "dv/dt=(exact() + approximate())/second : 1",
                method="euler", dt=1*b.ms,
                namespace={"exact": exact, "approximate": approximate})
            b.Network(group).run(100*b.ms)
            results.append(np.asarray(group.v[:]).copy())
        np.testing.assert_array_equal(results[0], results[1])

    def test_direct_poisson_rejects_negative_runtime_lambda(self):
        for backend in ("reference", "aot"):
            self.select(backend, f"invalid-direct-poisson-{backend}")
            group = b.NeuronGroup(
                1, "dv/dt=poisson(v)/second : 1",
                method="euler", dt=1*b.ms)
            group.v = -1
            with self.assertRaisesRegex(RuntimeError, "invalid poisson lambda"):
                b.Network(group).run(1*b.ms)


if __name__ == "__main__":
    unittest.main()
