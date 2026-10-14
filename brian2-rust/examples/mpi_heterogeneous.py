"""Run once with Python; the Device launches CPU/GPU MPI ranks itself."""
import argparse
import json
from pathlib import Path
import sys

import brian2 as b
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import brian2_atlas  # noqa: F401


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank-backends", required=True,
                        help="comma-separated rank devices, e.g. cpu,metal or cpu,cuda:0,cuda:1")
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    backends = args.rank_backends.split(",")
    b.set_device("atlas", engine="mpi", ranks=len(backends),
                 rank_backends=backends, directory=args.directory,
                 numeric_mode="mixed-f32" if any(v != "cpu" for v in backends) else "reference-f64")
    clock = b.Clock(dt=b.second / 1024)
    neurons = b.NeuronGroup(16, "dv/dt=drive*1024*Hz:1 (unless refractory)\ndrive:1 (constant)",
        threshold="v>1", reset="v=0", refractory=2*clock.dt, method="euler", clock=clock)
    neurons.drive = np.tile([0.125, 0.25, 0.5, 0.125], 4)
    synapses = b.Synapses(neurons, neurons, on_pre="v_post += 0.125", clock=clock)
    synapses.connect(i=np.arange(16), j=(np.arange(16)+8) % 16)
    synapses.delay = (1+np.arange(16) % 3)*clock.dt
    spikes = b.SpikeMonitor(neurons)
    b.Network(neurons, synapses, spikes).run(32*clock.dt)
    report = json.loads((args.directory / "rust" / "mpi-runtime.json").read_text())
    print(json.dumps(report, indent=2))
    print(f"Recorded {spikes.num_spikes} spikes")


if __name__ == "__main__":
    main()
