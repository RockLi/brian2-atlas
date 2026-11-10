"""100-neuron/two-ODE conformance harness: Rust, NumPy and C++ standalone."""

import argparse
import json
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

import brian2 as b
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import brian2_atlas  # noqa: E402, F401


def make_network():
    group = b.NeuronGroup(
        100,
        """
        dv/dt = (drive - v - w) / tau : 1
        dw/dt = (0.1*v - w) / tau_w : 1
        drive : 1 (constant)
        """,
        threshold="v > 1", reset="w += 0.02; v = 0",
        method="euler", dt=0.1 * b.ms,
        namespace={"tau": 10 * b.ms, "tau_w": 5 * b.ms},
        name="population",
    )
    group.v = np.linspace(0, 0.8, 100)
    group.w = np.linspace(0, 0.1, 100)
    group.drive = np.linspace(1.1, 1.8, 100)
    state = b.StateMonitor(group, ["v", "w"], record=True, name="state")
    spikes = b.SpikeMonitor(group, name="spikes")
    return b.Network(group, state, spikes), group, state, spikes


def run_backend(backend, output):
    output.mkdir(parents=True, exist_ok=False)
    if backend == "rust":
        b.set_device("atlas", directory=output / "project")
    elif backend == "cpp":
        # Conformance comparison: use strict arithmetic, not Brian's default
        # fast-math flags. This is not a performance benchmark.
        if sys.platform == "win32":
            b.prefs.codegen.cpp.extra_compile_args = ["/O2", "/fp:strict", "/std:c++17"]
        else:
            b.prefs.codegen.cpp.extra_compile_args = ["-O2", "-std=c++17", "-fno-fast-math", "-ffp-contract=off"]
            b.prefs.devices.cpp_standalone.extra_make_args_unix = ["-j2"]
        b.prefs.devices.cpp_standalone.openmp_threads = 0
        b.set_device("cpp_standalone", directory=str(output / "project"))
    else:
        b.set_device("runtime")
        b.prefs.codegen.target = "numpy"
    network, group, state, spikes = make_network()
    network.run(100 * b.ms)
    np.savez(
        output / "results.npz", v=state.v, w=state.w, t=state.t / b.second,
        spike_i=spikes.i[:], spike_t=spikes.t / b.second,
        spike_tick=np.rint(spikes.t / group.clock.dt).astype(np.int64),
        count=spikes.count[:], final_v=group.v[:], final_w=group.w[:],
        network_t=float(network.t / b.second), clock_t=float(group.clock.t / b.second),
    )
    metadata = {
        "backend": backend, "brian2": b.__version__, "python": platform.python_version(),
        "platform": platform.platform(), "neurons": len(group), "samples": len(state.t),
        "spikes": int(spikes.num_spikes),
        "cpp_flags": b.prefs.codegen.cpp.extra_compile_args if backend == "cpp" else None,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"{backend}: {state.v.shape}, {spikes.num_spikes} spikes", flush=True)


def compare(output):
    report = {"neurons": 100, "states": ["v", "w"], "steps": 1000,
              "state_tolerance": {"rtol": 1e-12, "atol": 1e-14}, "comparisons": {}}
    with np.load(output / "rust" / "results.npz") as rust:
        for backend in ("numpy", "cpp"):
            with np.load(output / backend / "results.npz") as reference:
                for name in ("v", "w", "final_v", "final_w"):
                    np.testing.assert_allclose(rust[name], reference[name], rtol=1e-12, atol=1e-14)
                for name in ("spike_i", "spike_tick", "count"):
                    np.testing.assert_array_equal(rust[name], reference[name])
                for name in ("t", "spike_t", "network_t", "clock_t"):
                    np.testing.assert_allclose(rust[name], reference[name], rtol=0, atol=1e-15)
                report["comparisons"][backend] = {
                    "max_abs_v_error": float(np.max(np.abs(rust["v"] - reference["v"]))),
                    "max_abs_w_error": float(np.max(np.abs(rust["w"] - reference["w"]))),
                    "spikes": len(rust["spike_i"]), "spike_order_equal": True, "passed": True,
                }
    (output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["all", "rust", "numpy", "cpp"], default="all")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.backend != "all":
        if args.output is None:
            parser.error("--output is required with an individual backend")
        run_backend(args.backend, args.output.resolve())
        return
    if args.output is None:
        (ROOT / "output").mkdir(exist_ok=True)
        output = Path(tempfile.mkdtemp(prefix="population-", dir=ROOT / "output"))
    else:
        output = args.output.resolve()
        output.mkdir(parents=True, exist_ok=False)
    # Fresh processes give each backend its own Device, clocks and model state.
    for backend in ("rust", "numpy", "cpp"):
        with (output / f"{backend}.log").open("w") as log:
            result = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--backend", backend,
                 "--output", str(output / backend)], stdout=log, stderr=subprocess.STDOUT,
            )
        if result.returncode:
            raise RuntimeError(f"{backend} failed; inspect {output / (backend + '.log')}")
        print(f"{backend} completed", flush=True)
    compare(output)
    print(f"Artifacts: {output}")


if __name__ == "__main__":
    main()
