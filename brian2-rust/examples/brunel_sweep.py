#!/usr/bin/env python3
"""Reuse one Brunel AOT artifact for a two-dimensional ``g``/``eta`` sweep."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import brian2_rust  # noqa: E402
from brian2_rust.spec import bits  # noqa: E402
from brian2_rust.protocol import attach_protocol  # noqa: E402
from brunel_device import J, analyse_spikes  # noqa: E402


def model_for_g(base_model: dict, g: float) -> dict:
    """Copy a lowered Brunel model and replace its shared inhibitory weight."""
    if not np.isfinite(g) or g <= 0:
        raise ValueError("g must be finite and positive")
    model = copy.deepcopy(base_model)
    matches = [
        index for index, definition in enumerate(model["definition"]["synapses"])
        if definition["name"] == "inhibitory_synapses"
    ]
    if len(matches) != 1:
        raise ValueError("artifact is not a supported Brunel model")
    index = matches[0]
    parameter = model["instance"]["synapses"][index]["parameters"].get("weight")
    if not isinstance(parameter, list) or len(parameter) != 1:
        raise ValueError("Brunel inhibitory weight is not an instance scalar")
    model["instance"]["synapses"][index]["parameters"]["weight"] = [
        bits(-g * float(J))]
    return model


def poisson_probability_parameter(model: dict) -> tuple[int, str]:
    """Locate the Instance scalar read by the Brunel PoissonInput."""
    matches = []

    def visit(value, population_index):
        if isinstance(value, dict):
            if value.get("op") == "binomial":
                probability = value.get("p")
                if (isinstance(probability, dict) and
                        probability.get("op") == "load"):
                    matches.append((population_index, probability.get("name")))
            for child in value.values():
                visit(child, population_index)
        elif isinstance(value, list):
            for child in value:
                visit(child, population_index)

    for population_index, population in enumerate(
            model["definition"]["populations"]):
        visit(population["code_objects"], population_index)
    if len(matches) != 1:
        raise ValueError("artifact must contain exactly one dynamic PoissonInput")
    population_index, name = matches[0]
    parameters = model["instance"]["populations"][population_index]["parameters"]
    if not isinstance(name, str) or not isinstance(parameters.get(name), list) or \
            len(parameters[name]) != 1:
        raise ValueError("PoissonInput probability is not an instance scalar")
    return population_index, name


def model_for_point(
    base_model: dict,
    g: float,
    eta: float,
    nu_threshold_hz: float,
    dt_ms: float,
) -> tuple[dict, float]:
    """Replace both phase-diagram coordinates without changing native source."""
    if not np.isfinite(eta) or eta <= 0:
        raise ValueError("eta must be finite and positive")
    probability = eta * nu_threshold_hz * dt_ms / 1000.0
    if not np.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("eta gives a PoissonInput probability outside [0, 1]")
    model = model_for_g(base_model, g)
    population_index, name = poisson_probability_parameter(model)
    model["instance"]["populations"][population_index]["parameters"][name] = [
        bits(probability)]
    return attach_protocol(model), probability


def execute(args: argparse.Namespace) -> dict:
    source = args.artifact.resolve()
    project = source / "project"
    base_result = json.loads((source / "result.json").read_text())
    base_model = json.loads((project / "model.json").read_text())
    if (base_result.get("schema") != "brunel-device-result-v1" or
            base_result.get("backend") != "rust"):
        raise ValueError("--artifact must be a Rust brunel_device output")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    g_values = np.linspace(args.g_min, args.g_max, args.g_count)
    eta_values = np.linspace(args.eta_min, args.eta_max, args.eta_count)
    point_count = len(g_values) * len(eta_values)
    rows = []
    sweep_started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="brunel-phase-sweep-") as temporary:
        temporary = Path(temporary)
        for position, (eta, g) in enumerate(
                (eta, g) for eta in eta_values for g in g_values):
            model, probability = model_for_point(
                base_model,
                float(g),
                float(eta),
                base_result["model"]["nu_threshold_hz"],
                base_result["dt_ms"],
            )
            point = temporary / f"point-{position:04d}"
            instance_path = temporary / f"instance-{position:04d}.bin"
            started = time.perf_counter()
            try:
                run = brian2_rust.run_compatible_instance(
                    model,
                    project / "native",
                    instance_path,
                    point,
                    threads=args.threads,
                )
                wall_seconds = time.perf_counter() - started
                population = run["results"]["populations"][0]
                statistics, _ = analyse_spikes(
                    population["indices"], population["spike_times"],
                    base_result["model"]["neuron_count"],
                    base_result["discard_ms"] / 1000,
                    base_result["duration_ms"] / 1000,
                )
                timing = run["results"]["metadata"]["timings"]
                row = {
                    "g": float(g),
                    "eta": float(eta),
                    "poisson_probability_per_step": probability,
                    "mean_rate_hz": statistics["mean_rate_hz"],
                    "isi_cv_mean": statistics["isi_cv_mean"],
                    "population_count_fano_0_1ms":
                        statistics["population_count_fano_0_1ms"],
                    "peak_frequency_hz": statistics["peak_frequency_hz"],
                    "spectral_peak_fraction_5_400hz":
                        statistics["spectral_peak_fraction_5_400hz"],
                    "simulation_and_recording_seconds":
                        timing["simulation_and_recording_seconds"],
                    "wall_seconds": wall_seconds,
                    "instance_sha256": run["instance"]["instance_sha256"],
                }
            finally:
                # Full-scale result dumps are tens of megabytes per point.
                # Statistics are already in memory, so retaining 400 dumps
                # only consumes temporary disk without adding sweep output.
                shutil.rmtree(point, ignore_errors=True)
                instance_path.unlink(missing_ok=True)
            rows.append(row)
            if not args.quiet:
                print(
                    f"[{position + 1}/{point_count}] eta={eta:.4g} g={g:.4g} "
                    f"rate={row['mean_rate_hz']:.3f} Hz "
                    f"loop={row['simulation_and_recording_seconds']:.3f}s",
                    flush=True,
                )

    report = {
        "schema": "brunel-phase-sweep-v2",
        "artifact": str(source),
        "source_sha256": brian2_rust.compatible_source_sha256(base_model),
        "g_values": g_values.tolist(),
        "eta_values": eta_values.tolist(),
        "grid_shape_eta_by_g": [len(eta_values), len(g_values)],
        "duration_ms": base_result["duration_ms"],
        "discard_ms": base_result["discard_ms"],
        "network_scale": base_result["network_scale"],
        "seed": base_result["seed"],
        "threads": args.threads,
        "point_count": point_count,
        "total_wall_seconds": time.perf_counter() - sweep_started,
        "points": rows,
    }
    (output / "sweep.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez(
        output / "sweep.npz",
        g_values=g_values,
        eta_values=eta_values,
        grid_shape_eta_by_g=np.asarray(
            [len(eta_values), len(g_values)], dtype=np.int64),
        **{
            key: np.asarray([row[key] for row in rows], dtype=np.float64)
            for key in (
                "g", "eta", "poisson_probability_per_step",
                "mean_rate_hz", "isi_cv_mean",
                "population_count_fano_0_1ms", "peak_frequency_hz",
                "spectral_peak_fraction_5_400hz",
                "simulation_and_recording_seconds", "wall_seconds",
            )
        },
    )
    summary = {key: value for key, value in report.items() if key != "points"}
    summary["output"] = str(output)
    print(json.dumps(summary, indent=2), flush=True)
    return report


def parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument(
        "--artifact", type=Path, required=True,
        help="a completed Rust brunel_device output directory")
    argument_parser.add_argument("--g-min", type=float, default=1.0)
    argument_parser.add_argument("--g-max", type=float, default=8.0)
    argument_parser.add_argument("--g-count", type=int, default=20)
    argument_parser.add_argument("--eta-min", type=float, default=0.5)
    argument_parser.add_argument("--eta-max", type=float, default=4.0)
    argument_parser.add_argument("--eta-count", type=int, default=20)
    argument_parser.add_argument("--threads", type=int, default=4)
    argument_parser.add_argument(
        "--quiet", action="store_true", help="suppress per-point progress")
    argument_parser.add_argument("--output", type=Path, required=True)
    return argument_parser


if __name__ == "__main__":
    arguments = parser().parse_args()
    if arguments.g_count < 1 or arguments.eta_count < 1:
        raise ValueError("g-count and eta-count must be positive")
    execute(arguments)
