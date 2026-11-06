"""Fixed refractory with frozen voltage and a continuously updated second state."""

import sys
from pathlib import Path

from brian2 import Network, NeuronGroup, SpikeMonitor, StateMonitor, ms, set_device

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import brian2_atlas  # noqa: E402, F401

set_device("atlas")
group = NeuronGroup(
    1, "dv/dt=0.5/ms : 1 (unless refractory)\ndx/dt=1/ms : 1",
    threshold="v>=1", reset="v=0", refractory=3*ms, method="euler", dt=1*ms,
)
state = StateMonitor(group, ["v", "x"], record=True)
spikes = SpikeMonitor(group)
Network(group, state, spikes).run(10*ms)

print("spike times (ms):", spikes.t[:] / ms)  # [1, 5, 9]
print("voltage:", state.v[0])  # [0, .5, 0, 0, 0, .5, 0, 0, 0, .5]
print("unfrozen x:", state.x[0])  # [0, 1, ..., 9]
print("lastspike (ms):", group.lastspike[:] / ms)  # [9]
print("not_refractory:", group.not_refractory[:])  # [False]
