"""Conductance-based and Hodgkin-Huxley expression conformance."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import brian2 as b
import numpy as np
from brian2.devices.device import all_devices

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import brian2_rust  # noqa: E402,F401
from brian2_rust.native import parallel_task_limit  # noqa: E402


class CobaHhTest(unittest.TestCase):
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

    def select(self, backend, name, threads=1):
        if backend == "numpy":
            b.set_device("runtime")
            b.prefs.codegen.target = "numpy"
        else:
            self.device.reinit()
            b.set_device("rust_standalone", runner=ROOT / "target/release/b2-runner",
                         directory=Path(self.temp.name) / f"{name}-{backend}",
                         engine=backend, threads=threads)

    def compare(self, actual, expected):
        self.assertEqual(set(actual), set(expected))
        for name in actual:
            if name.endswith(("_i", "_count", "_available")):
                np.testing.assert_array_equal(actual[name], expected[name], err_msg=name)
            else:
                np.testing.assert_allclose(actual[name], expected[name], rtol=3e-12,
                                           atol=1e-15, err_msg=name)

    def test_coba_conductance_states_and_synaptic_updates(self):
        results = []
        for backend in ["aot", "reference", "numpy"]:
            self.select(backend, "coba")
            source = b.NeuronGroup(
                3, "dx/dt=drive/ms : 1\ndrive : 1 (constant)",
                threshold="x>1", reset="x=0", method="euler", dt=.1*b.ms,
                name="coba_source")
            target = b.NeuronGroup(
                4, """
                dv/dt=(gl*(El-v)+ge*(Ee-v)+gi*(Ei-v))/Cm : volt
                dge/dt=-ge/taue : siemens
                dgi/dt=-gi/taui : siemens
                """, method="exponential_euler", dt=.1*b.ms,
                namespace={"gl": 10*b.nS, "El": -60*b.mV, "Ee": 0*b.mV,
                           "Ei": -80*b.mV, "Cm": 200*b.pF,
                           "taue": 5*b.ms, "taui": 10*b.ms},
                name="coba_target")
            source.drive = [6, 8, 10]
            target.v = [-65, -60, -55, -50]*b.mV
            target.ge = [1, 2, 3, 4]*b.nS
            target.gi = [4, 3, 2, 1]*b.nS
            synapse = b.Synapses(
                source, target, "we : siemens (constant)\nwi : siemens (constant)",
                on_pre="ge_post += we; gi_post += wi", delay=.2*b.ms,
                clock=source.clock, name="coba_synapse")
            synapse.connect(i=[0, 0, 1, 2], j=[1, 3, 2, 0])
            synapse.we = [1, 2, 3, 4]*b.nS
            synapse.wi = [4, 3, 2, 1]*b.nS
            source_state = b.StateMonitor(source, "x", record=True)
            target_state = b.StateMonitor(target, ["v", "ge", "gi"], record=True)
            source_spikes = b.SpikeMonitor(source)
            b.Network(source, target, synapse, source_state, target_state,
                      source_spikes).run(2*b.ms)
            results.append({
                "source_x": np.asarray(source_state.x).copy(),
                "target_v": np.asarray(target_state.v/b.volt).copy(),
                "target_ge": np.asarray(target_state.ge/b.siemens).copy(),
                "target_gi": np.asarray(target_state.gi/b.siemens).copy(),
                "final_v": np.asarray(target.v[:]/b.volt).copy(),
                "final_ge": np.asarray(target.ge[:]/b.siemens).copy(),
                "final_gi": np.asarray(target.gi[:]/b.siemens).copy(),
                "source_i": np.asarray(source_spikes.i[:]).copy(),
                "source_tick": np.rint(source_spikes.t[:]/source.clock.dt).copy(),
                "source_count": np.asarray(source_spikes.count[:]).copy(),
            })
        for expected in results[1:]:
            self.compare(results[0], expected)

    def test_hh_subexpressions_exponential_euler_and_threshold_without_reset(self):
        area = 20_000*b.umetre**2
        namespace = {
            "Cm": 1*b.ufarad*b.cm**-2*area,
            "gl": 5e-5*b.siemens*b.cm**-2*area,
            "El": -60*b.mV, "EK": -90*b.mV, "ENa": 50*b.mV,
            "g_na": 100*b.msiemens*b.cm**-2*area,
            "g_kd": 30*b.msiemens*b.cm**-2*area,
            "VT": -63*b.mV, "taue": 5*b.ms, "taui": 10*b.ms,
            "Ee": 0*b.mV, "Ei": -80*b.mV,
        }
        equations = b.Equations("""
            dv/dt=(gl*(El-v)+ge*(Ee-v)+gi*(Ei-v)-g_na*m**3*h*(v-ENa)
                   -g_kd*n**4*(v-EK))/Cm : volt
            dm/dt=alpha_m*(1-m)-beta_m*m : 1
            dn/dt=alpha_n*(1-n)-beta_n*n : 1
            dh/dt=alpha_h*(1-h)-beta_h*h : 1
            dge/dt=-ge/taue : siemens
            dgi/dt=-gi/taui : siemens
            alpha_m=0.32*(mV**-1)*4*mV/exprel((13*mV-v+VT)/(4*mV))/ms : Hz
            beta_m=0.28*(mV**-1)*5*mV/exprel((v-VT-40*mV)/(5*mV))/ms : Hz
            alpha_h=0.128*exp((17*mV-v+VT)/(18*mV))/ms : Hz
            beta_h=4/(1+exp((40*mV-v+VT)/(5*mV)))/ms : Hz
            alpha_n=0.032*(mV**-1)*5*mV/exprel((15*mV-v+VT)/(5*mV))/ms : Hz
            beta_n=0.5*exp((10*mV-v+VT)/(40*mV))/ms : Hz
        """)
        results = []
        for backend in ["aot", "reference", "numpy"]:
            self.select(backend, "hh")
            group = b.NeuronGroup(
                5, equations, threshold="v>-20*mV", refractory=.3*b.ms,
                method="exponential_euler", dt=.05*b.ms, namespace=namespace,
                name="hh_population")
            group.v = [-15, -62, -58, -54, -50]*b.mV
            group.m = [.05, .06, .07, .08, .09]
            group.n = [.31, .32, .33, .34, .35]
            group.h = [.55, .57, .59, .61, .63]
            group.ge = [0, 1, 2, 3, 4]*b.nS
            group.gi = [4, 3, 2, 1, 0]*b.nS
            monitor = b.StateMonitor(group, ["v", "m", "n", "h", "ge", "gi"],
                                     record=True)
            b.Network(group, monitor).run(1*b.ms)
            results.append({
                "trace_v": np.asarray(monitor.v/b.volt).copy(),
                "trace_m": np.asarray(monitor.m).copy(),
                "trace_n": np.asarray(monitor.n).copy(),
                "trace_h": np.asarray(monitor.h).copy(),
                "trace_ge": np.asarray(monitor.ge/b.siemens).copy(),
                "trace_gi": np.asarray(monitor.gi/b.siemens).copy(),
                "final_v": np.asarray(group.v[:]/b.volt).copy(),
                "final_m": np.asarray(group.m[:]).copy(),
                "final_n": np.asarray(group.n[:]).copy(),
                "final_h": np.asarray(group.h[:]).copy(),
                "hh_lastspike": np.asarray(group.lastspike[:]/b.second).copy(),
                "hh_available": np.asarray(group.not_refractory[:]).copy(),
            })
            if backend == "aot":
                exported = json.loads(
                    (self.device.last_run_directory / "model.json").read_text())
                generated = (
                    self.device.last_run_directory / "native/main.rs").read_text()
                self.assertIn(".powi(3)", generated)
                self.assertIn(".powi(4)", generated)
                # The serial fallback and parallel lane both lower the same
                # state-update expressions. Count the four semantic uses at
                # minimum instead of assuming code is emitted only once.
                self.assertGreaterEqual(generated.count("exprel("), 4)
                self.assertNotIn("reset", [code["kind"] for code in
                                            exported["definition"]["populations"]
                                            [0]["code_objects"]])
                self.assertIsNone(
                    exported["definition"]["populations"][0]["spike_monitor"])
        self.assertTrue(np.any(results[0]["hh_lastspike"] >= 0))
        for expected in results[1:]:
            self.compare(results[0], expected)

    def test_seeded_cobahh_style_subgroups_probability_and_random_initialization(self):
        results = []
        for backend in ["aot", "reference"]:
            self.select(backend, "seeded-cobahh")
            b.seed(20260904)
            group = b.NeuronGroup(
                20, """
                dv/dt=(ge + gi - (v-El))/taum : volt (unless refractory)
                dge/dt=-ge/taue : volt
                dgi/dt=-gi/taui : volt
                """, threshold="v>Vt", reset="v=Vr", refractory=.5*b.ms,
                method="exponential_euler", dt=.1*b.ms,
                namespace={"El": -49*b.mV, "Vt": -50*b.mV,
                           "Vr": -60*b.mV, "taum": 20*b.ms,
                           "taue": 5*b.ms, "taui": 10*b.ms},
                name="seeded_population")
            excitatory, inhibitory = group[:16], group[16:]
            exc = b.Synapses(excitatory, group, on_pre="ge += 1.62*mV",
                             clock=group.clock, name="seeded_exc")
            inh = b.Synapses(inhibitory, group, on_pre="gi += -9*mV",
                             clock=group.clock, name="seeded_inh")
            exc.connect(p=.2)
            inh.connect(p=.2)
            group.v = "-60*mV + rand() * 10*mV"
            group.ge = "randn() * 0.1*mV"
            group.gi = "randn() * 0.2*mV"
            monitor = b.StateMonitor(group, ["v", "ge", "gi"], record=[1, 10])
            initial = (group.v[:].copy(), group.ge[:].copy(), group.gi[:].copy())
            topology = (
                exc.variables["_synaptic_pre"].get_value().copy(),
                exc.variables["_synaptic_post"].get_value().copy(),
                inh.variables["_synaptic_pre"].get_value().copy() - 16,
                inh.variables["_synaptic_post"].get_value().copy(),
            )
            b.Network(group, exc, inh, monitor).run(2*b.ms)
            results.append({
                "exc_i": topology[0], "exc_j": topology[1],
                "inh_i": topology[2], "inh_j": topology[3],
                "initial_v": np.asarray(initial[0]/b.volt),
                "initial_ge": np.asarray(initial[1]/b.volt),
                "initial_gi": np.asarray(initial[2]/b.volt),
                "trace_v": np.asarray(monitor.v).copy(),
                "trace_ge": np.asarray(monitor.ge).copy(),
                "trace_gi": np.asarray(monitor.gi).copy(),
            })
            self.assertGreater(len(exc), 0)
            self.assertGreater(len(inh), 0)
        self.compare(results[0], results[1])

    def test_hh_parallel_state_update_is_deterministic(self):
        size = 4000
        self.assertGreaterEqual(parallel_task_limit(size, 380), 16)
        namespace = {"tau": 1*b.ms}
        equations = "\n".join([
            f"dx{index}/dt=(exp(-x{index})+sin(x{index})+cos(x{index})-x{index}"
            f"{'+0.01*randn()' if index == 0 else ''})/tau : 1"
            for index in range(12)
        ])
        dumps, states = [], []
        for threads in [1, 4, 8]:
            self.select("aot", f"parallel-{threads}", threads=threads)
            b.seed(20260904)
            group = b.NeuronGroup(
                size, equations, method="euler", dt=.1*b.ms,
                namespace=namespace, name="parallel_hh")
            initial = np.linspace(.01, .5, len(group))
            for index in range(12):
                setattr(group, f"x{index}", initial + index*.01)
            monitor = b.StateMonitor(
                group, ["x0", "x11"], record=[0, size//2, size-1])
            b.Network(group, monitor).run(.5*b.ms)
            states.append([np.asarray(getattr(group, f"x{index}")[:]).copy()
                           for index in range(12)] +
                          [np.asarray(monitor.x0).copy(), np.asarray(monitor.x11).copy()])
            directory = self.device.last_run_directory
            dumps.append((directory / "rust/results.bin").read_bytes())
            summary = json.loads((directory / "rust/summary.json").read_text())
            self.assertEqual(summary["threads"], threads)
            self.assertEqual(summary["parallel_state_update"], threads > 1)
            if threads == 8:
                generated = (directory / "native/main.rs").read_text()
                self.assertIn("fn task_count(&self,n:usize,work:usize)", generated)
                self.assertNotIn("(n+1023)/1024", generated)
                self.assertIn("std::thread::park()", generated)
                self.assertIn("worker.thread().unpark()", generated)
        for dump in dumps[1:]:
            self.assertEqual(dumps[0], dump)
        for other in states[1:]:
            for actual, expected in zip(states[0], other, strict=True):
                np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
