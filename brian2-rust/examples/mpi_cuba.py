"""Small static E/I network: genuine MPI AOT vs independent Rust f64 reference.

This is a correctness/reproducibility experiment; v0 replicates input storage.
Run the Python script once. It launches MPI itself (do not mpiexec this script).
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import brian2 as b
from brian2.devices.device import all_devices
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import brian2_rust
from brian2_rust.export import lower_network
from brian2_rust.distributed import write_mpi_project, compile_mpi_project, run_mpi_project
from brian2_rust.results import load_results


def make_model(neurons, steps, degree):
    clock = b.Clock(dt=0.1*b.ms, name="mpi_clock")
    equation = "dv/dt=(drive-v+current)/(10*ms):1 (unless refractory)\ndcurrent/dt=-current/(5*ms):1\ndrive:1 (constant)"
    ne = neurons * 4 // 5
    exc = b.NeuronGroup(ne, equation, threshold="v>1", reset="v=0", refractory=0.2*b.ms,
                        method="euler", clock=clock, name="exc")
    inh = b.NeuronGroup(neurons-ne, equation, threshold="v>1", reset="v=0", refractory=0.2*b.ms,
                        method="euler", clock=clock, name="inh")
    initial = np.linspace(0, 0.9, neurons)
    drive = np.linspace(1.1, 1.7, neurons)
    exc.v, inh.v = initial[:ne], initial[ne:]
    exc.drive, inh.drive = drive[:ne], drive[ne:]
    source = np.repeat(np.arange(neurons), degree)
    target = (source + np.tile(np.arange(1, degree+1), neurons) * 7) % neurons
    projections = []
    for sa, (sg, sb) in enumerate(((exc, 0), (inh, ne))):
        for ta, (tg, tb) in enumerate(((exc, 0), (inh, ne))):
            mask = (source >= sb) & (source < sb+len(sg)) & (target >= tb) & (target < tb+len(tg))
            syn = b.Synapses(sg, tg, "w:1 (constant)", on_pre="current_post += w",
                             clock=clock, name=f"projection_{sa}_{ta}")
            syn.connect(i=source[mask]-sb, j=target[mask]-tb)
            syn.w = (0.04 if sa == 0 else -0.16)
            syn.delay = (1 + target[mask] % 4)*clock.dt
            projections.append(syn)
    monitors = [b.StateMonitor(exc, ["v", "current"], record=[0, ne-1]),
                b.StateMonitor(inh, ["v", "current"], record=True),
                b.SpikeMonitor(exc), b.SpikeMonitor(inh)]
    return lower_network(b.Network(exc, inh, *projections, *monitors), steps*clock.dt, rng_seed=1729)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--neurons", type=int, default=100)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--degree", type=int, default=8)
    parser.add_argument("--ranks", type=int, nargs="+", default=[1, 2, 4])
    args = parser.parse_args()
    if args.neurons < 5 or args.steps < 1 or not 1 <= args.degree < args.neurons:
        parser.error("require neurons>=5, steps>=1, and 1<=degree<neurons")
    if any(not 1 <= ranks <= 256 for ranks in args.ranks) or len(set(args.ranks)) != len(args.ranks):
        parser.error("rank counts must be unique and in 1..256")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    previous = b.get_device()
    device = all_devices["rust_standalone"]
    try:
        device.reinit()
        b.set_device("rust_standalone", runner=ROOT/"target/release/b2-runner")
        model = make_model(args.neurons, args.steps, args.degree)
        model_path = args.output/"model.json"
        model_path.write_text(json.dumps(model, indent=2) + "\n")
        subprocess.run([str(ROOT/"target/release/b2-runner"), str(model_path),
                        str(args.output/"reference")], check=True, capture_output=True)
        expected = load_results(model, args.output/"reference")
        output_files = ["results.bin", "events.bin"]
        reference_hashes = {name: digest(args.output/"reference"/name) for name in output_files}
        report = {"schema": "b2-mpi-comparison-v0", "platform": platform.platform(),
                  "neurons": args.neurons, "edges": args.neurons*args.degree, "steps": args.steps,
                  "reference_sha256": reference_hashes,
                  "reference_synaptic_events": expected["metadata"]["synaptic_events"], "runs": [],
                  "timing_scope": "run_wall includes MPI launch, initialization, simulation, final collection, and output; compilation separately reported",
                  "limitation": "one host; replicated arrays; no distributed-memory scaling or speedup claim"}
        for ranks in args.ranks:
            project = args.output/f"rank-{ranks}"
            plan = write_mpi_project(model, project, ranks=ranks)
            started = time.perf_counter()
            compile_mpi_project(project)
            compile_seconds = time.perf_counter()-started
            started = time.perf_counter()
            observed = run_mpi_project(project, project/"result")
            wall_seconds = time.perf_counter()-started
            load_results(model, project/"result")
            actual_hashes = {name: digest(project/"result"/name) for name in output_files}
            row = {"ranks": ranks, "compile_seconds": compile_seconds, "run_wall_seconds": wall_seconds,
                   "result_sha256": actual_hashes, "exact_reference_match": actual_hashes == reference_hashes,
                   "plan_sha256": plan.sha256, "runtime": observed}
            report["runs"].append(row)
            (args.output/"report.json").write_text(json.dumps(report, indent=2) + "\n")
            if not row["exact_reference_match"]:
                raise RuntimeError(f"MPI {ranks} ranks differs from independent reference")
            print(f"{ranks} ranks: exact results/events match; {observed['rank_work']}")
        print(f"Evidence: {args.output/'report.json'}")
    finally:
        device.reinit()
        b.set_device(previous)


if __name__ == "__main__":
    main()
