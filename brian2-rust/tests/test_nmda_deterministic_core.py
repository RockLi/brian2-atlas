"""Deterministic regression for general clock-driven NMDA synapse semantics."""

import os
from pathlib import Path
import sys

import brian2 as b
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import brian2_rust  # noqa: E402


def _run(backend, directory, runner):
    b.start_scope()
    if backend == "rust":
        b.set_device("rust_standalone", engine="aot", runner=runner,
                     directory=directory)
    else:
        b.set_device("runtime")
        b.prefs.codegen.target = "numpy"
    b.defaultclock.dt = 0.1 * b.ms
    source = b.SpikeGeneratorGroup(
        1, indices=[0, 0, 0], times=[1, 10, 25] * b.ms,
        name="source")
    target = b.NeuronGroup(
        1,
        """dV/dt = (-gLeak*(V-El) - I_NMDA)/Cm : volt
           I_NMDA = gNMDA*s_NMDA_tot*(V-V_E)/(1+exp(-0.062*V/mV)*(C/mmole/3.57)) : amp
           s_NMDA_tot : 1""",
        method="rk4",
        namespace={"gLeak": 25*b.nS, "El": -70*b.mV,
                   "gNMDA": 0.165*b.nS, "V_E": 0*b.mV,
                   "Cm": 0.5*b.nF, "C": 1*b.mmole},
        name="target")
    target.V = -70 * b.mV
    syn = b.Synapses(
        source, target,
        """s_NMDA_tot_post = s_NMDA : 1 (summed)
           ds_NMDA/dt = -s_NMDA/(100*ms) + (0.5*kHz)*x*(1-s_NMDA) : 1 (clock-driven)
           dx/dt = -x/(2*ms) : 1 (clock-driven)""",
        on_pre="x += 1", delay=0.5*b.ms, method="rk4",
        name="nmda")
    syn.connect()
    state = b.StateMonitor(target, ["V", "s_NMDA_tot"], record=True)
    b.Network(source, target, syn, state).run(100*b.ms)
    output = {
        "V": np.asarray(state.V).copy(),
        "s_NMDA_tot": np.asarray(state.s_NMDA_tot).copy(),
        "x_final": np.asarray(syn.x[:]).copy(),
        "s_NMDA_final": np.asarray(syn.s_NMDA[:]).copy(),
    }
    if backend == "rust":
        b.device.reinit()
    b.set_device("runtime")
    return output


def test_general_nmda_ode_summed_and_delay_match_numpy(tmp_path):
    runner = Path(os.environ.get(
        "B2_RUNNER", str(ROOT / "target" / "release" / "b2-runner")))
    assert runner.exists()
    rust = _run("rust", tmp_path / "nmda", runner)
    numpy = _run("numpy", None, None)
    # IEEE f64 roundoff over 1000 RK4 steps: ~10^4 eps times the state scale.
    np.testing.assert_allclose(rust["V"], numpy["V"], rtol=0, atol=2e-13)
    np.testing.assert_allclose(
        rust["s_NMDA_tot"], numpy["s_NMDA_tot"], rtol=0, atol=2e-12)
    np.testing.assert_allclose(
        rust["x_final"], numpy["x_final"], rtol=0, atol=2e-12)
    np.testing.assert_allclose(
        rust["s_NMDA_final"], numpy["s_NMDA_final"], rtol=0, atol=2e-12)
