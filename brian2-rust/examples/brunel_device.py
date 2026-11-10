#!/usr/bin/env python3
"""Brunel (2000) Figure 8 network and broad-delay AR experiment.

The first four named regimes are the simulation points in Figure 8. Every
target receives the paper's exact recurrent indegree without multapses. The
Rust backend materializes this topology procedurally; NumPy and C++ use
Brian's ``sample(..., size=...)`` generator. Random streams differ across
backends, so the scientific comparison is statistical rather than
spike-for-spike.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path

import brian2 as b
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import brian2_rust  # noqa: E402


DT = 0.1 * b.ms
TAU = 20 * b.ms
THRESHOLD = 20 * b.mV
RESET = 10 * b.mV
REFRACTORY = 2 * b.ms
J = 0.1 * b.mV
DELAY = 1.5 * b.ms
CONNECTION_PROBABILITY = 0.1
FULL_EXCITATORY = 10_000
FULL_INHIBITORY = 2_500

# Parameters and display windows from Figure 8 / the Brian2 reference example.
REGIMES = {
    "sr": {
        "label": "synchronous regular",
        "g": 3.0,
        "eta": 2.0,
        "duration_ms": 600.0,
        "discard_ms": 500.0,
    },
    "si_fast": {
        "label": "synchronous irregular, fast",
        "g": 6.0,
        "eta": 4.0,
        "duration_ms": 1200.0,
        "discard_ms": 1000.0,
    },
    "ai": {
        "label": "asynchronous irregular",
        "g": 5.0,
        "eta": 2.0,
        "duration_ms": 1200.0,
        "discard_ms": 1000.0,
    },
    "si_slow": {
        "label": "synchronous irregular, slow",
        "g": 4.5,
        "eta": 0.9,
        "duration_ms": 1200.0,
        "discard_ms": 1000.0,
    },
    "ar": {
        "label": "asynchronous regular, broad delays",
        "g": 3.0,
        "eta": 2.0,
        "duration_ms": 1200.0,
        "discard_ms": 1000.0,
        "delay_distribution": "uniform",
    },
}


def network_shape(scale: float) -> tuple[int, int, int]:
    """Return E neurons, I neurons and recurrent excitatory indegree."""
    if not math.isfinite(scale) or not 0 < scale <= 1:
        raise ValueError("network scale must be in (0, 1]")
    excitatory = max(1, round(FULL_EXCITATORY * scale))
    inhibitory = max(1, round(FULL_INHIBITORY * scale))
    excitatory_indegree = max(1, round(CONNECTION_PROBABILITY * excitatory))
    return excitatory, inhibitory, excitatory_indegree


def analyse_spikes(
    spike_indices: np.ndarray,
    spike_times_s: np.ndarray,
    neuron_count: int,
    start_s: float,
    stop_s: float,
    *,
    rate_bin_s: float = 0.0001,
) -> tuple[dict, dict[str, np.ndarray]]:
    """Calculate rate, ISI-CV, binned activity and a population spectral peak."""
    if neuron_count <= 0 or not 0 <= start_s < stop_s or rate_bin_s <= 0:
        raise ValueError("invalid analysis interval, bin width, or neuron count")
    indices = np.asarray(spike_indices, dtype=np.int64)
    times = np.asarray(spike_times_s, dtype=np.float64)
    if indices.shape != times.shape:
        raise ValueError("spike indices and times must have matching shapes")
    selected = (times >= start_s) & (times < stop_s)
    indices, times = indices[selected], times[selected]
    if indices.size and (indices.min() < 0 or indices.max() >= neuron_count):
        raise ValueError("spike index is outside the population")

    interval_s = stop_s - start_s
    spike_counts = np.bincount(indices, minlength=neuron_count)
    rates_hz = spike_counts / interval_s
    cvs = []
    for neuron in np.flatnonzero(spike_counts >= 3):
        isi = np.diff(times[indices == neuron])
        if isi.size >= 2 and np.mean(isi) > 0:
            cvs.append(float(np.std(isi) / np.mean(isi)))
    isi_cv = np.asarray(cvs, dtype=np.float64)

    def bin_edges(width: float) -> np.ndarray:
        bins = max(1, int(np.ceil(interval_s / width - 1e-12)))
        result = start_s + np.arange(bins + 1) * width
        result[-1] = stop_s
        return result

    edges = bin_edges(rate_bin_s)
    population_counts, edges = np.histogram(times, bins=edges)
    bin_widths = np.diff(edges)
    population_rate_hz = population_counts / (neuron_count * bin_widths)
    population_rate_time_s = edges[:-1] + bin_widths / 2

    # A 1 ms series is less noisy and still resolves the ~180 Hz fast SI peak.
    spectral_bin_s = 0.001
    spectral_edges = bin_edges(spectral_bin_s)
    spectral_counts, _ = np.histogram(times, bins=spectral_edges)
    centered = spectral_counts.astype(np.float64) - np.mean(spectral_counts)
    power = np.abs(np.fft.rfft(centered)) ** 2
    actual_spectral_bin_s = interval_s / centered.size
    frequencies_hz = np.fft.rfftfreq(centered.size, actual_spectral_bin_s)
    band = (frequencies_hz >= 5.0) & (frequencies_hz <= 400.0)
    if np.any(band) and np.any(power[band] > 0):
        band_indices = np.flatnonzero(band)
        peak_power = float(np.max(power[band]))
        # An ideal pulse train gives equal-power harmonics. Report the
        # fundamental (the lowest tied peak) instead of an arbitrary harmonic.
        tied = band_indices[power[band] >= peak_power * (1 - 1e-12)]
        peak_index = int(tied[0])
        peak_frequency_hz = float(frequencies_hz[peak_index])
        spectral_peak_fraction = float(power[peak_index] / np.sum(power[band]))
    else:
        peak_frequency_hz = None
        spectral_peak_fraction = 0.0

    mean_count = float(np.mean(population_counts))
    population_fano = (float(np.var(population_counts) / mean_count)
                       if mean_count > 0 else 0.0)
    summary = {
        "measurement_seconds": interval_s,
        "spike_count": int(indices.size),
        "mean_rate_hz": float(np.mean(rates_hz)),
        "rate_std_hz": float(np.std(rates_hz)),
        "active_fraction": float(np.mean(spike_counts > 0)),
        "isi_cv_mean": float(np.mean(isi_cv)) if isi_cv.size else None,
        "isi_cv_median": float(np.median(isi_cv)) if isi_cv.size else None,
        "isi_cv_neurons": int(isi_cv.size),
        "population_count_fano_0_1ms": population_fano,
        "peak_frequency_hz": peak_frequency_hz,
        "spectral_peak_fraction_5_400hz": spectral_peak_fraction,
    }
    arrays = {
        "rates_hz": rates_hz,
        "isi_cv": isi_cv,
        "population_rate_time_s": population_rate_time_s,
        "population_rate_hz": population_rate_hz,
        "spectrum_frequency_hz": frequencies_hz,
        "spectrum_power": power,
    }
    return summary, arrays


def make_network(
    backend: str,
    g: float,
    eta: float,
    scale: float,
    seed: int,
    record_neurons: int,
    delay_distribution: str = "fixed",
):
    """Build model A from Brunel (2000)."""
    n_e, n_i, c_e = network_shape(scale)
    neuron_count = n_e + n_i
    if not math.isfinite(g) or g <= 0 or not math.isfinite(eta) or eta <= 0:
        raise ValueError("g and eta must be finite and positive")
    if delay_distribution not in {"fixed", "uniform"}:
        raise ValueError("delay distribution must be fixed or uniform")
    nu_threshold = THRESHOLD / (J * c_e * TAU)
    nu_external = eta * nu_threshold
    if float(nu_external * DT) > 1:
        raise ValueError(
            "scaled network makes nu_external * dt > 1; increase --network-scale")

    neurons = b.NeuronGroup(
        neuron_count,
        "dv/dt = -v/tau : volt (unless refractory)",
        threshold="v > threshold",
        reset="v = reset",
        refractory=REFRACTORY,
        dt=DT,
        # Brian's default deterministic choice selects the exact linear solver.
        namespace={"tau": TAU, "threshold": THRESHOLD, "reset": RESET},
        name="neurons",
    )
    excitatory = neurons[:n_e]
    inhibitory = neurons[n_e:]
    projections = []
    for name, source, indegree, weight, topology_seed in (
        ("excitatory", excitatory, c_e, J, seed + 1),
        ("inhibitory", inhibitory,
         max(1, round(CONNECTION_PROBABILITY * n_i)), -g * J, seed + 2),
    ):
        synapse_options = ({"delay": DELAY}
                           if delay_distribution == "fixed" else {})
        synapses = b.Synapses(
            source,
            neurons,
            "weight : volt (constant, shared)",
            on_pre="v_post += weight",
            clock=neurons.clock,
            name=f"{name}_synapses",
            **synapse_options,
        )
        if backend == "rust":
            brian2_rust.connect_fixed_indegree(
                synapses, indegree, seed=topology_seed,
                delay_initializer=(
                    brian2_rust.Uniform(0*b.ms, 3*b.ms)
                    if delay_distribution == "uniform" else None))
        else:
            synapses.connect(
                i=f"k for k in sample(N_pre, size={indegree})")
            if delay_distribution == "uniform":
                synapses.delay = "rand() * 3*ms"
        synapses.weight = weight
        projections.append(synapses)

    external = b.PoissonInput(
        neurons, "v", N=c_e, rate=nu_external, weight=J)
    spikes = b.SpikeMonitor(neurons, name="spikes")
    record_count = min(record_neurons, neuron_count)
    voltage = b.StateMonitor(
        neurons, "v", record=np.arange(record_count), name="voltage")
    network = b.Network(neurons, *projections, external, spikes, voltage)
    metadata = {
        "neuron_count": neuron_count,
        "excitatory_neurons": n_e,
        "inhibitory_neurons": n_i,
        "excitatory_indegree": c_e,
        "recurrent_synapses": [
            round(CONNECTION_PROBABILITY * n_e * neuron_count),
            round(CONNECTION_PROBABILITY * n_i * neuron_count),
        ],
        "nu_threshold_hz": float(nu_threshold / b.Hz),
        "nu_external_hz": float(nu_external / b.Hz),
        "delay_distribution": {
            "kind": delay_distribution,
            "minimum_ms": (0.0 if delay_distribution == "uniform" else 1.5),
            "maximum_ms": (3.0 if delay_distribution == "uniform" else 1.5),
        },
    }
    return network, neurons, spikes, voltage, metadata


def select_backend(backend: str, output: Path, threads: int) -> None:
    if backend == "rust":
        runner = ROOT / "target/release/b2-runner"
        if not runner.is_file():
            raise FileNotFoundError(
                f"missing {runner}; build brian2-rust with cargo build --release")
        b.set_device(
            "atlas", runner=runner, directory=output / "project",
            engine="aot", threads=threads, profile=True)
    elif backend == "cpp":
        b.prefs.devices.cpp_standalone.openmp_threads = 0 if threads == 1 else threads
        b.prefs.devices.cpp_standalone.extra_make_args_unix = [f"-j{threads}"]
        b.set_device("cpp_standalone", build_on_run=False)
    else:
        b.set_device("runtime")
        b.prefs.codegen.target = "numpy"


def execute(args: argparse.Namespace) -> dict:
    configuration = REGIMES[args.regime]
    g = configuration["g"] if args.g is None else args.g
    eta = configuration["eta"] if args.eta is None else args.eta
    duration_ms = (configuration["duration_ms"] if args.duration_ms is None
                   else args.duration_ms)
    discard_ms = (configuration["discard_ms"] if args.discard_ms is None
                  else args.discard_ms)
    delay_distribution = (configuration.get("delay_distribution", "fixed")
                          if args.delay_distribution is None
                          else args.delay_distribution)
    if not 0 <= discard_ms < duration_ms:
        raise ValueError("discard-ms must be in [0, duration-ms)")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    select_backend(args.backend, output, args.threads)
    b.seed(args.seed)
    network, neurons, spikes, voltage, model = make_network(
        args.backend, g, eta, args.network_scale, args.seed,
        args.record_neurons, delay_distribution)
    if args.backend == "rust":
        report = brian2_rust.capability_report(network, duration_ms * b.ms)
        if not report.supported:
            raise RuntimeError(report.format_text())

    started = time.perf_counter()
    network.run(duration_ms * b.ms, profile=args.backend == "rust")
    if args.backend == "cpp":
        b.get_device().build(
            directory=str(output / "project"), compile=True, run=True,
            with_output=False)
    wall_seconds = time.perf_counter() - started

    spike_i = np.asarray(spikes.i[:], dtype=np.int32)
    spike_t_s = np.asarray(spikes.t[:] / b.second, dtype=np.float64)
    statistics, arrays = analyse_spikes(
        spike_i, spike_t_s, len(neurons), discard_ms / 1000,
        duration_ms / 1000)
    np.savez_compressed(
        output / "activity.npz",
        spike_i=spike_i,
        spike_t_s=spike_t_s,
        voltage_v=np.asarray(voltage.v[:] / b.volt),
        voltage_t_s=np.asarray(voltage.t[:] / b.second),
        **arrays,
    )

    native = None
    if args.backend == "rust":
        native_path = b.get_device().last_run_directory / "rust" / "summary.json"
        native = json.loads(native_path.read_text())
    result = {
        "schema": "brunel-device-result-v1",
        "backend": args.backend,
        "regime": args.regime,
        "regime_label": configuration["label"],
        "g": g,
        "eta": eta,
        "duration_ms": duration_ms,
        "discard_ms": discard_ms,
        "dt_ms": float(DT / b.ms),
        "delay_ms": float(DELAY / b.ms),
        "delay_distribution": model["delay_distribution"],
        "connection_probability": CONNECTION_PROBABILITY,
        "network_scale": args.network_scale,
        "seed": args.seed,
        "threads": args.threads,
        "model": model,
        "statistics": statistics,
        "wall_seconds": wall_seconds,
        "simulation_and_recording_seconds": (
            None if native is None else
            native["timings"]["simulation_and_recording_seconds"]),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "brian2": b.__version__,
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return result


def parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument("--backend", choices=("rust", "cpp", "numpy"),
                                 default="rust")
    argument_parser.add_argument("--regime", choices=tuple(REGIMES), default="ai")
    argument_parser.add_argument("--g", type=float)
    argument_parser.add_argument("--eta", type=float,
                                 help="external rate divided by threshold rate")
    argument_parser.add_argument("--duration-ms", type=float)
    argument_parser.add_argument("--discard-ms", type=float)
    argument_parser.add_argument(
        "--delay-distribution", choices=("fixed", "uniform"),
        help="override the regime's recurrent delay distribution")
    argument_parser.add_argument("--network-scale", type=float, default=1.0,
                                 help="smoke-test scaling only; scientific runs use 1")
    argument_parser.add_argument("--record-neurons", type=int, default=50)
    argument_parser.add_argument("--seed", type=int, default=20260906)
    argument_parser.add_argument("--threads", type=int, default=4)
    argument_parser.add_argument("--output", type=Path, required=True)
    return argument_parser


if __name__ == "__main__":
    execute(parser().parse_args())
