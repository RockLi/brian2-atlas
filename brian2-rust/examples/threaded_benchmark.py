"""Compare true intra-simulation AOT threads with Brian2 C++ OpenMP."""

import argparse
from datetime import datetime
import hashlib
import json
import os
import platform
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tempfile
import time

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WORKLOADS = {
    "cuba100k": {"script": "benchmark.py", "arguments":
                 ["--scenario", "cuba-100000", "--repeats", "1"],
                 "child_creates_output": True,
                 "description":
                 "CUBA: 100,000 neurons, fixed fan-out 8, 100 simulated ms"},
    "cobahh": {"script": "cobahh_throughput.py", "description":
               "COBAHH: 4,000 neurons, p=0.02, 1 simulated second"},
    "cuba_skew100k": {"script": "benchmark.py", "arguments":
                      ["--scenario", "cuba-skew-100000", "--repeats", "1"],
                      "child_creates_output": True,
                      "description":
                      "Skewed CUBA: 100,000 neurons, 800k edges targeting first 1/8"},
    "poisson_input": {"script": "poisson_input_throughput.py",
                      "comparison": "distribution",
                      "description":
                      "PoissonInput: 100,000 neurons, N=100, 1,000 ticks"},
}
DEFAULT_WORKLOADS = ["cuba100k", "cobahh", "poisson_input"]
PINNED_ENVIRONMENT = {
    "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
}


def default_cxx():
    libomp = Path("/opt/homebrew/opt/libomp")
    if sys.platform == "darwin" and Path("/usr/bin/clang++").exists() and libomp.exists():
        return "apple-clang-libomp"
    return shutil.which("g++-15") or os.environ.get("CXX") or "c++"


def prepare_cxx(output, requested):
    """Make Apple clang's intentionally separate libomp flags usable by make."""
    if requested != "apple-clang-libomp":
        return requested, requested
    libomp = Path("/opt/homebrew/opt/libomp")
    compiler = Path("/usr/bin/clang++")
    if not compiler.exists() or not (libomp / "lib/libomp.dylib").exists():
        raise FileNotFoundError("Apple clang + Homebrew libomp is unavailable")
    wrapper = output / "clang-openmp-wrapper"
    wrapper.write_text(
        "#!/bin/bash\n"
        "compile=0\n"
        "for arg in \"$@\"; do [[ \"$arg\" == -c ]] && compile=1; done\n"
        "args=()\n"
        "for arg in \"$@\"; do\n"
        "  if [[ \"$arg\" == -fopenmp ]]; then\n"
        f"    if [[ $compile == 1 ]]; then args+=(-Xpreprocessor -fopenmp -I{libomp}/include); "
        f"else args+=(-L{libomp}/lib -Wl,-rpath,{libomp}/lib -lomp); fi\n"
        "  else args+=(\"$arg\"); fi\n"
        "done\n"
        f"exec {compiler} \"${{args[@]}}\"\n")
    wrapper.chmod(0o755)
    version = subprocess.check_output([compiler, "--version"], text=True).splitlines()[0]
    return str(wrapper), f"{version} + {libomp / 'lib/libomp.dylib'}"


def parse_levels(value):
    try:
        levels = sorted(set(int(item) for item in value.split(",")))
    except ValueError as error:
        raise argparse.ArgumentTypeError("levels must be comma-separated integers") from error
    if not levels or levels[0] != 1 or levels[-1] > 256:
        raise argparse.ArgumentTypeError("levels must include 1 and stay within 1..256")
    return levels


def default_levels():
    logical_cpus = os.cpu_count() or 1
    return [level for level in (1, 2, 4, 8) if level <= logical_cpus]


def build(workload, root, levels, cxx):
    definition = WORKLOADS[workload]
    environment = {**os.environ, **PINNED_ENVIRONMENT,
                   "OMP_NUM_THREADS": "1", "CXX": cxx}
    projects = {}
    for backend, threads in [("aot", 1), *(("cpp", level) for level in levels)]:
        label = "aot" if backend == "aot" else f"cpp-t{threads}"
        output = root / label
        if not definition.get("child_creates_output"):
            output.mkdir(parents=True)
        command = [sys.executable, str(HERE / definition["script"]),
                   "--backend", backend, "--output", str(output),
                   "--threads", str(threads), *definition.get("arguments", [])]
        with (root / f"build-{label}.log").open("w") as log:
            subprocess.run(command, cwd=ROOT, env=environment, check=True,
                           stdout=log, stderr=subprocess.STDOUT, text=True,
                           timeout=900)
        projects[label] = output / "project"
    return projects


def compare_npz(actual_path, expected_path):
    with np.load(actual_path) as actual, np.load(expected_path) as expected:
        if set(actual.files) != set(expected.files):
            raise AssertionError("result fields differ")
        for name in actual.files:
            if name.endswith(("_i", "_count")) or name == "edges":
                np.testing.assert_array_equal(actual[name], expected[name], err_msg=name)
            elif name.endswith("_t"):
                np.testing.assert_allclose(actual[name], expected[name], rtol=0,
                                           atol=1e-15, err_msg=name)
            else:
                np.testing.assert_allclose(actual[name], expected[name], rtol=3e-11,
                                           atol=1e-14, err_msg=name)


def run_native(backend, project, output, threads, timeout_seconds):
    environment = {**os.environ, **PINNED_ENVIRONMENT,
                   "B2_NUM_THREADS": str(threads),
                   "OMP_NUM_THREADS": str(threads), "OMP_DYNAMIC": "FALSE"}
    if backend == "aot":
        command = [project / "native/b2-native", project / "native/instance.bin", output]
        cwd = None
    else:
        output.mkdir()
        command = [project / "main", "--results_dir", str(output) + os.sep]
        cwd = project
    started = time.perf_counter()
    subprocess.run(command, cwd=cwd, env=environment, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
                   timeout=timeout_seconds)
    wall = time.perf_counter() - started
    if backend == "aot":
        summary = json.loads((output / "summary.json").read_text())
        if summary["threads"] != threads:
            raise AssertionError("AOT did not use the requested thread count")
        loop = summary["timings"]["simulation_and_recording_seconds"]
        parallel_state_update = summary["parallel_state_update"]
        parallel_poisson_input = summary.get("parallel_poisson_input", False)
        parallel_on_pre = summary["parallel_on_pre"]
    else:
        loop = float((output / "last_run_info.txt").read_text().split()[0])
        parallel_state_update = None
        parallel_poisson_input = None
        parallel_on_pre = None
    return {"loop_seconds": loop, "wall_seconds": wall,
            "parallel_state_update": parallel_state_update,
            "parallel_poisson_input": parallel_poisson_input,
            "parallel_on_pre": parallel_on_pre}


def measure_workload(workload, output, levels, repeats, cxx, timeout_seconds):
    output.mkdir()
    projects = build(workload, output, levels, cxx)
    comparison = WORKLOADS[workload].get("comparison", "exact")
    if comparison == "exact":
        reference_npz = output / "aot/results.npz"
        for level in levels:
            compare_npz(output / f"cpp-t{level}/results.npz", reference_npz)
    else:
        # Distribution workloads validate every build-time run in their child
        # script.  Their C++ and counter-based Rust random streams intentionally
        # differ, so equality would be the wrong conformance criterion.
        for label in ["aot", *(f"cpp-t{level}" for level in levels)]:
            result = json.loads((output / label / "result.json").read_text())
            if not np.isfinite(result["total_input"]):
                raise AssertionError(f"non-finite {workload} result for {label}")

    samples = {level: {"aot": [], "cpp": []} for level in levels}
    hashes = {}
    with tempfile.TemporaryDirectory(prefix=f"b2-threaded-{workload}-") as temporary:
        temporary = Path(temporary)
        reference_dump = None
        for level in levels:
            for backend in ("aot", "cpp"):
                project = projects["aot" if backend == "aot" else f"cpp-t{level}"]
                warm = temporary / f"warm-t{level}-{backend}"
                run_native(backend, project, warm, level, timeout_seconds)
                if backend == "aot":
                    digest = hashlib.sha256((warm / "results.bin").read_bytes()).hexdigest()
                    hashes[str(level)] = digest
                    reference_dump = reference_dump or digest
                    if digest != reference_dump:
                        raise AssertionError(
                            f"AOT result dump changed at {level} threads")
        # Rotate thread levels as well as backend order. This spreads changing
        # host load and thermal conditions across the whole scaling curve.
        for iteration in range(repeats):
            level_order = levels[iteration % len(levels):] + levels[:iteration % len(levels)]
            for level in level_order:
                order = ("aot", "cpp") if iteration % 2 == 0 else ("cpp", "aot")
                for backend in order:
                    project = projects["aot" if backend == "aot" else f"cpp-t{level}"]
                    samples[level][backend].append(run_native(
                        backend, project,
                        temporary / f"t{level}-r{iteration}-{backend}",
                        level, timeout_seconds))

    results, baselines = {}, {}
    for level in levels:
        entry = {"threads": level, "backends": {}}
        for backend in ("aot", "cpp"):
            rows = samples[level][backend]
            loop = [row["loop_seconds"] for row in rows]
            wall = [row["wall_seconds"] for row in rows]
            entry["backends"][backend] = {
                "loop_median_ms": statistics.median(loop) * 1000,
                "loop_min_ms": min(loop) * 1000, "loop_max_ms": max(loop) * 1000,
                "loop_relative_spread": ((max(loop) - min(loop)) /
                                         statistics.median(loop)),
                "wall_median_ms": statistics.median(wall) * 1000,
                "samples": rows,
            }
            if backend == "aot":
                active = {row["parallel_state_update"] for row in rows}
                if len(active) != 1:
                    raise AssertionError("inconsistent AOT parallel phase activation")
                entry["backends"][backend]["parallel_state_update"] = active.pop()
                active = {row["parallel_poisson_input"] for row in rows}
                if len(active) != 1:
                    raise AssertionError("inconsistent AOT PoissonInput phase activation")
                entry["backends"][backend]["parallel_poisson_input"] = active.pop()
                active = {row["parallel_on_pre"] for row in rows}
                if len(active) != 1:
                    raise AssertionError("inconsistent AOT on_pre activation")
                entry["backends"][backend]["parallel_on_pre"] = active.pop()
            if level == 1:
                baselines[backend] = statistics.median(loop)
            entry["backends"][backend]["speedup_vs_one_thread"] = (
                baselines[backend] / statistics.median(loop))
            entry["backends"][backend]["parallel_efficiency"] = (
                entry["backends"][backend]["speedup_vs_one_thread"] / level)
        aot = entry["backends"]["aot"]["loop_median_ms"]
        cpp = entry["backends"]["cpp"]["loop_median_ms"]
        entry["aot_speedup_over_cpp"] = cpp / aot
        entry["aot_not_slower"] = aot <= cpp
        results[str(level)] = entry
    best_threads = {
        backend: min(
            levels,
            key=lambda level: results[str(level)]["backends"][backend]
            ["loop_median_ms"],
        )
        for backend in ("aot", "cpp")
    }
    stable = all(
        backend["loop_relative_spread"] <= .15
        for entry in results.values()
        for backend in entry["backends"].values()
    )
    return {"description": WORKLOADS[workload]["description"],
            "comparison": comparison,
            "correctness_passed": True,
            "results_equal": comparison == "exact",
            "aot_thread_dumps_sha256": hashes,
            "levels": results, "best_threads": best_threads,
            "measurement_stable": stable,
            "aot_performance_gate_passed": all(
                entry["aot_not_slower"] for entry in results.values())}


def report(output, levels, repeats, workloads, cxx, timeout_seconds):
    output.mkdir(parents=True, exist_ok=False)
    measurement_started_at = datetime.now().astimezone().isoformat()
    load_average_before = os.getloadavg()
    cxx_command, cxx_label = prepare_cxx(output, cxx)
    results = {}
    for workload in workloads:
        print(f"[{workload}] building AOT and C++ OpenMP variants", flush=True)
        results[workload] = measure_workload(
            workload, output / workload, levels, repeats, cxx_command, timeout_seconds)
    result = {
        "schema": "b2-intra-simulation-threading-v2",
        "measurement_started_at": measurement_started_at,
        "environment": {"platform": platform.platform(), "machine": platform.machine(),
                        "logical_cpus": os.cpu_count(), "cxx": cxx_label,
                        "cxx_command": cxx_command,
                        "load_average_before": load_average_before,
                        "load_average_after": os.getloadavg()},
        "scope": "threads inside one simulation process",
        "aot_parallel_phases": [
            "independent per-neuron state updates",
            "independent per-neuron PoissonInput sampling",
            "threshold fused into eligible state-update worker lanes",
            "target-owner partitioned uniform-delay on_pre",
        ],
        "serial_phases": ["monitor append", "spike-lane merge",
                          "ineligible/small-batch event routing", "on_post", "reset",
                          "result dump"],
        "levels": levels, "repeats": repeats, "workloads": results,
        "aot_performance_gate_passed": all(
            workload["aot_performance_gate_passed"] for workload in results.values()),
    }
    (output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = ["# Intra-simulation threading: Rust AOT vs C++ OpenMP", "",
             f"{repeats} alternating native replays after warm-up; compiler: `{cxx_label}`.", "",
             "| Workload | Threads | AOT (ms) | C++ OpenMP (ms) | AOT speedup | "
             "C++ speedup | AOT efficiency | C++ efficiency | C++ / AOT |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, workload in results.items():
        for level in levels:
            entry = workload["levels"][str(level)]
            aot, cpp = entry["backends"]["aot"], entry["backends"]["cpp"]
            lines.append(
                f"| {name} | {level} | {aot['loop_median_ms']:.3f} | "
                f"{cpp['loop_median_ms']:.3f} | {aot['speedup_vs_one_thread']:.2f}× | "
                f"{cpp['speedup_vs_one_thread']:.2f}× | {aot['parallel_efficiency']:.0%} | "
                f"{cpp['parallel_efficiency']:.0%} | {entry['aot_speedup_over_cpp']:.2f}× |")
        lines.append(
            f"\n{name}: best AOT level is {workload['best_threads']['aot']} threads; "
            f"best C++ OpenMP level is {workload['best_threads']['cpp']} threads.")
        if not workload["measurement_stable"]:
            lines.append(
                "Measurement warning: at least one min/max spread exceeded 15%; "
                "repeat on an otherwise idle host before treating scaling as a baseline.")
    lines += ["", "AOT result dumps are SHA-256 identical across thread counts. Exact workloads "
              "compare C++ OpenMP outputs against the AOT reference with strict field-specific "
              "tolerances; distribution workloads validate every build-time run statistically.", ""]
    document = "\n".join(lines)
    (output / "report.md").write_text(document)
    print(document, end="")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--levels", type=parse_levels, default=default_levels())
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--workload", choices=WORKLOADS, action="append")
    parser.add_argument("--cxx", default=default_cxx())
    parser.add_argument("--timeout-seconds", type=float, default=900)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.repeats <= 20 or args.timeout_seconds <= 0:
        parser.error("repeats must be 1..20 and timeout must be positive")
    result = report(args.output.resolve(), args.levels, args.repeats,
                    args.workload or DEFAULT_WORKLOADS, args.cxx,
                    args.timeout_seconds)
    if args.strict and not result["aot_performance_gate_passed"]:
        raise SystemExit("Rust AOT is slower than C++ OpenMP in one or more cases")


if __name__ == "__main__":
    main()
