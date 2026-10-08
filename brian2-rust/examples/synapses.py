"""Smallest delayed chain through Brian2's independent Rust Device."""

import sys
from pathlib import Path

from brian2 import Network, NeuronGroup, SpikeMonitor, StateMonitor, Synapses, ms, set_device

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import brian2_rust  # noqa: E402, F401

set_device("rust_standalone")
group = NeuronGroup(3, "dv/dt = 0*Hz : 1", threshold="v > 1", reset="v = 0",
                    method="euler", dt=1*ms)
group.v = [1.1, 0, 0]
synapses = Synapses(group, group, "w : 1", on_pre="v_post += w",
                    delay=1*ms, clock=group.clock)
synapses.connect(i=[0, 1], j=[1, 2])
synapses.w = 1.2
state = StateMonitor(group, "v", record=True)
spikes = SpikeMonitor(group)
Network(group, synapses, state, spikes).run(5*ms)

print("spike neuron indices:", spikes.i[:])  # [0, 1, 2]
print("spike times (ms):", spikes.t[:] / ms)  # [0, 2, 4]
print("state shape:", state.v.shape)  # (3, 5)
