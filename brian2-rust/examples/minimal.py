"""Smallest complete example: select the Rust Device, create a model, run it."""

import sys
from pathlib import Path

import brian2 as b

# For this source checkout; external scripts can instead set PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import brian2_rust  # noqa: E402, F401 -- registers the optional Device

b.set_device("rust_standalone")
b.defaultclock.dt = 0.1 * b.ms

tau = 10 * b.ms
drive = 1.5
neurons = b.NeuronGroup(
    1, "dv/dt = (drive - v) / tau : 1",
    threshold="v > 1", reset="v = 0", method="euler",
)
neurons.v = 0
state = b.StateMonitor(neurons, "v", record=True)
spikes = b.SpikeMonitor(neurons)

b.run(100 * b.ms)  # Device builds Rust, runs it, and fills Brian's result arrays.

print(f"Device: {type(b.get_device()).__name__}")
print(f"StateMonitor.v.shape: {state.v.shape}")
print(f"Spikes: {spikes.num_spikes}; times (ms): {spikes.t[:] / b.ms}")
print(f"Final v: {neurons.v[0]}; clock: {b.defaultclock.t}")
print(f"Artifacts: {b.get_device().last_run_directory}")
