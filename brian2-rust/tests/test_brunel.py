"""Fast checks for the Brunel example and its offline analysis."""

import importlib.util
import sys
from pathlib import Path

import brian2 as b
import numpy as np
import pytest
from brian2.devices.device import all_devices


EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
SPEC = importlib.util.spec_from_file_location(
    "brunel_device", EXAMPLES / "brunel_device.py")
brunel = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = brunel
SPEC.loader.exec_module(brunel)
SWEEP_SPEC = importlib.util.spec_from_file_location(
    "brunel_sweep", EXAMPLES / "brunel_sweep.py")
sweep = importlib.util.module_from_spec(SWEEP_SPEC)
sys.modules[SWEEP_SPEC.name] = sweep
SWEEP_SPEC.loader.exec_module(sweep)


def test_figure_8_parameters_are_canonical():
    assert (brunel.REGIMES["sr"]["g"], brunel.REGIMES["sr"]["eta"]) == (3.0, 2.0)
    assert (brunel.REGIMES["si_fast"]["g"],
            brunel.REGIMES["si_fast"]["eta"]) == (6.0, 4.0)
    assert (brunel.REGIMES["ai"]["g"], brunel.REGIMES["ai"]["eta"]) == (5.0, 2.0)
    assert (brunel.REGIMES["si_slow"]["g"],
            brunel.REGIMES["si_slow"]["eta"]) == (4.5, 0.9)
    assert (brunel.REGIMES["ar"]["g"], brunel.REGIMES["ar"]["eta"],
            brunel.REGIMES["ar"]["delay_distribution"]) == (3.0, 2.0, "uniform")


def test_full_network_shape_and_synapse_count():
    n_e, n_i, c_e = brunel.network_shape(1.0)
    assert (n_e, n_i, c_e) == (10_000, 2_500, 1_000)
    assert round(0.1 * n_e * (n_e + n_i)) == 12_500_000
    assert round(0.1 * n_i * (n_e + n_i)) == 3_125_000


def test_analysis_recovers_regular_rate_cv_and_frequency():
    # Ten neurons fire regularly at 20 Hz and in phase for one second.
    times = np.arange(0.025, 1.0, 0.05)
    spike_t = np.tile(times, 10)
    spike_i = np.repeat(np.arange(10), times.size)
    order = np.argsort(spike_t, kind="stable")
    summary, arrays = brunel.analyse_spikes(
        spike_i[order], spike_t[order], 10, 0.0, 1.0)
    assert summary["mean_rate_hz"] == pytest.approx(20.0)
    assert summary["isi_cv_mean"] == pytest.approx(0.0, abs=1e-12)
    assert summary["peak_frequency_hz"] == pytest.approx(20.0)
    assert arrays["population_rate_hz"].shape == (10_000,)


@pytest.mark.parametrize("scale", [0, -0.1, 1.1, float("nan")])
def test_invalid_network_scale_is_rejected(scale):
    with pytest.raises(ValueError, match="network scale"):
        brunel.network_shape(scale)


@pytest.mark.parametrize("regime", tuple(brunel.REGIMES))
def test_every_named_regime_lowers_for_rust(regime):
    previous = b.get_device()
    device = all_devices["rust_standalone"]
    try:
        device.reinit()
        b.start_scope()
        b.set_device("rust_standalone", runner=brunel.ROOT / "target/release/b2-runner")
        parameters = brunel.REGIMES[regime]
        network, _, _, _, _ = brunel.make_network(
            "rust", parameters["g"], parameters["eta"], 0.01, 1234, 5,
            parameters.get("delay_distribution", "fixed"))
        report = brunel.brian2_rust.capability_report(network, 10 * b.ms)
        assert report.supported, report.format_text()
    finally:
        b.set_device(previous)
        device.reinit()
        b.start_scope()


def test_g_and_eta_change_only_brunel_instance_data():
    previous = b.get_device()
    device = all_devices["rust_standalone"]
    try:
        device.reinit()
        b.start_scope()
        b.set_device("rust_standalone", runner=brunel.ROOT / "target/release/b2-runner")
        b.seed(1234)
        network, _, _, _, metadata = brunel.make_network(
            "rust", 5.0, 2.0, 0.01, 1234, 5)
        from brian2_rust.export import lower_network

        baseline = lower_network(network, 10 * b.ms)
        changed, probability = sweep.model_for_point(
            baseline, 5.5, 3.0, metadata["nu_threshold_hz"],
            float(brunel.DT / b.ms))
        assert probability == pytest.approx(0.3)
        assert changed["definition"] == baseline["definition"]
        assert sweep.brian2_rust.compatible_source_sha256(changed) == \
            sweep.brian2_rust.compatible_source_sha256(baseline)
        population_index, name = sweep.poisson_probability_parameter(changed)
        assert (changed["instance"]["populations"][population_index]
                ["parameters"][name] !=
                baseline["instance"]["populations"][population_index]
                ["parameters"][name])
    finally:
        b.set_device(previous)
        device.reinit()
        b.start_scope()
