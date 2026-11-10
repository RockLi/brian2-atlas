"""Minimal two-population Brian2 network executed by the Rust AOT Device."""

import sys
from pathlib import Path

import brian2 as b

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import brian2_atlas  # noqa: E402, F401


b.set_device("atlas", engine="aot")
equations = """
dv/dt = drive/ms : 1
drive : 1 (constant)
"""
source = b.NeuronGroup(
    2, equations, threshold="v>1", reset="v=0", method="euler",
    dt=1*b.ms, name="source_group")
target = b.NeuronGroup(
    3, equations, threshold="v>1", reset="v=0", method="euler",
    dt=1*b.ms, name="target_group")
source.v = [1.2, .2]
source.drive = [0, .1]
target.v = [.1, .2, .3]
target.drive = [.1, .2, .3]

connections = b.Synapses(
    source, target,
    "dtrace/dt=-trace/(4*ms) : 1 (clock-driven)\nw : 1 (constant)",
    on_pre="v_post += w*trace; trace += 0.1", method="euler",
    clock=source.clock, name="feedforward")
connections.connect(i=[0, 0, 1], j=[2, 0, 1])
connections.w = [.5, .25, .75]
connections.trace = [1, .5, .25]
connections.delay = [0, 1, 0]*b.ms

source_state = b.StateMonitor(source, "v", record=True, name="source_state")
target_state = b.StateMonitor(target, "v", record=True, name="target_state")
source_spikes = b.SpikeMonitor(source, name="source_spikes")
target_spikes = b.SpikeMonitor(target, name="target_spikes")
network = b.Network(source, target, connections, source_state, target_state,
                    source_spikes, target_spikes)
network.run(5*b.ms)

print("source final:", source.v[:])
print("target final:", target.v[:])
print("synaptic trace:", connections.trace[:])
print("source spikes:", list(zip(source_spikes.i[:], source_spikes.t[:]/b.ms)))
print("target spikes:", list(zip(target_spikes.i[:], target_spikes.t[:]/b.ms)))
