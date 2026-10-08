"""Run one LIF neuron through the Rust Device and compare Brian's public API."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from brian2 import Network, NeuronGroup, SpikeMonitor, StateMonitor, get_device, ms, prefs, second, set_device

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import brian2_rust  # noqa: E402, F401 -- registers rust_standalone


def make_network():
    group = NeuronGroup(
        1, "dv/dt = (drive - v) / tau : 1",
        threshold="v > 1", reset="v = 0", method="euler", dt=0.1 * ms,
        namespace={"drive": 1.5, "tau": 10 * ms},
    )
    group.v = 0
    state = StateMonitor(group, "v", record=True)
    spikes = SpikeMonitor(group)
    return Network(group, state, spikes), group, state, spikes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="new output directory (default: output/device-<unique suffix>)")
    parser.add_argument("--runner", type=Path, help="prebuilt runner (default: build with Cargo)")
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error("output already exists; choose a new directory")

    set_device("rust_standalone", directory=args.output, runner=args.runner)
    rust_network, rust_group, rust_state, rust_spikes = make_network()
    rust_network.run(100 * ms)
    output = get_device().last_run_directory
    rust_ticks = np.rint(np.asarray(rust_spikes.t / rust_group.clock.dt)).astype(np.int64)
    print(f"Device: {type(get_device()).__name__}")
    print(f"StateMonitor.v.shape: {rust_state.v.shape}; SpikeMonitor.num_spikes: {rust_spikes.num_spikes}")

    # Construct a separate reference model after switching back to runtime.
    set_device("runtime")
    prefs.codegen.target = "numpy"
    network, group, state, spikes = make_network()
    network.run(100 * ms)
    brian_ticks = np.rint(np.asarray(spikes.t / group.clock.dt)).astype(np.int64)
    np.testing.assert_allclose(rust_state.t / second, state.t / second, rtol=0, atol=1e-15)
    np.testing.assert_allclose(rust_state.v, state.v, rtol=1e-12, atol=1e-14)
    np.testing.assert_array_equal(rust_ticks, brian_ticks)
    np.testing.assert_array_equal(rust_spikes.i, spikes.i)
    np.testing.assert_array_equal(rust_spikes.count, spikes.count)
    np.testing.assert_allclose(rust_spikes.t / second, spikes.t / second, rtol=0, atol=1e-15)
    np.testing.assert_allclose(rust_group.v[:], group.v[:], rtol=1e-12, atol=1e-14)
    np.testing.assert_equal(rust_network.t, network.t)
    comparison = {
        "device": "rust_standalone",
        "reference": "Brian2 NumPy", "samples": len(state.t),
        "spikes": int(rust_spikes.num_spikes), "spike_ticks": rust_ticks.tolist(),
        "max_abs_state_error": float(np.max(np.abs(rust_state.v - state.v))),
        "final_state": float(rust_group.v[0]), "passed": True,
    }
    (output / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    print(json.dumps(comparison, indent=2))
    print(f"Artifacts: {output.resolve()}")


if __name__ == "__main__":
    main()
