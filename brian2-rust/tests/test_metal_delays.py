"""Delay chronology, bounded rings and Device pending/checkpoint semantics."""
import json
import os
import copy
from pathlib import Path
import subprocess

import brian2 as b
import brian2_rust
import numpy as np
import pytest

from brian2_rust.metal import MetalExecutor, build_metal_plan
from brian2_rust.plan import PlanValidationError
from brian2_rust.protocol import attach_protocol
from brian2_rust.results import load_results

ROOT = Path(__file__).resolve().parents[1]
real_metal = pytest.mark.skipif(os.environ.get("B2_TEST_METAL") != "1", reason="requires an Apple GPU")


@pytest.fixture
def device():
    previous = b.get_device()
    from brian2.devices.device import all_devices
    dev = all_devices["rust_standalone"]
    dev.reinit()
    try:
        yield dev
    finally:
        dev.reinit()
        b.set_device(previous)


def compare(actual, expected):
    for a, e in zip(actual["populations"], expected["populations"], strict=True):
        for field in ("states", "trace"):
            for name in a[field]:
                np.testing.assert_array_equal(a[field][name], e[field][name])
        for field in ("spike_ticks", "indices", "counts", "last_spikes"):
            np.testing.assert_array_equal(a[field], e[field])
        for name in a["event_streams"]:
            for field in ("ticks", "indices"):
                np.testing.assert_array_equal(a["event_streams"][name][field], e["event_streams"][name][field])
    assert [s["events"] for s in actual["synapses"]] == [s["events"] for s in expected["synapses"]]


@real_metal
def test_delay_collision_chronology_and_plan_boundary(device, tmp_path):
    b.set_device("rust_standalone", engine="reference", directory=tmp_path/"reference",
                 runner=ROOT/"target/release/b2-runner")
    source = b.NeuronGroup(35, "v:1\nonset:second (constant)", threshold="v>0.5")
    source.onset = ([2, 0, 1]+[100]*32)*b.defaultclock.dt
    source.run_regularly("v=int(t>=onset)*int(t<onset+0.05*ms)")
    target = b.NeuronGroup(2, "v:1")
    syn = b.Synapses(source, target, "w:1", on_pre="v_post+=w", clock=source.clock)
    # Quiet filler edges force the compact sorting path (3 arrivals / 35 edges).
    syn.connect(i=[0, 2, 1]+list(range(3,35)), j=np.zeros(35, int))
    syn.w = [1, -2**24, 2**24]+[0]*32
    syn.delay = ([0, 1, 2]+[n%3 for n in range(32)])*b.defaultclock.dt
    monitor = b.StateMonitor(target, "v", record=True)
    b.Network(source, target, syn, monitor).run(.6*b.ms)
    model = json.loads((tmp_path/"reference/model.json").read_text())
    expected = load_results(model, tmp_path/"reference/rust")
    target_index = next(p for p, pop in enumerate(model["definition"]["populations"]) if pop["count"] == 2)
    np.testing.assert_array_equal(expected["populations"][target_index]["states"]["v"], [1, 0])
    for route in ("scan", "sparse"):
        with MetalExecutor(model, tmp_path/route, numeric_mode="float32", event_delivery=route) as executor:
            assert f"delayed-target-{route}" in [s.role for s in executor.plan.dispatches]
            if route == "sparse":
                assert "population-delay-source-enqueue" in [s.role for s in executor.plan.dispatches]
            assert tuple(n for k in executor.plan.kernels for n in k.nodes) == tuple(
                n.id for n in executor.plan.logical.nodes if n.id not in executor.plan.elided_nodes)
            compare(executor.run(compute="cpu-f32", workers=3), expected)
            for _ in range(2):
                compare(executor.run(), expected)
            with pytest.raises(MemoryError, match="delay"):
                # Direct allocator isolates ring capacity from recording limits.
                from brian2_rust.metal_delays import delay_arrays
                delay_arrays(model["definition"]["synapses"][0], model["instance"]["synapses"][0],
                             model["instance"]["synapses"][0]["pathways"][0], executor.plan.logical.clocks[0], 1)
    from brian2_rust.schedule import build_schedule
    model["definition"]["schedule"] = build_schedule(model["definition"], model["instance"],
        ["start", "groups", "synapses", "thresholds", "resets", "end"])
    attach_protocol(model)
    changed = tmp_path / "early-model.json"
    changed.write_text(json.dumps(model))
    subprocess.run([str(ROOT/"target/release/b2-runner"),str(changed),str(tmp_path/"early-ref")],check=True,capture_output=True)
    expected = load_results(model,tmp_path/"early-ref")
    with MetalExecutor(model,tmp_path/"early-metal",numeric_mode="float32") as executor:
        compare(executor.run(),expected)


@real_metal
@pytest.mark.parametrize("uniform", [False, True])
def test_delay_segmented_restore_ring_wrap_and_subgroups(device, tmp_path, uniform):
    snapshots = []
    for engine, segmented in (("reference", False), ("reference", True), ("metal", True)):
        device.reinit()
        options = {"numeric_mode": "float32", "event_delivery": "sparse"} if engine == "metal" else {}
        b.set_device("rust_standalone", engine=engine, directory=tmp_path/f"{engine}-{segmented}",
                     recording_window_steps=9, runner=ROOT/"target/release/b2-runner", **options)
        pop = b.NeuronGroup(7, "v:1\nx:1", threshold="v>0.5")
        pop.v = [1, 1, 0, 1, 1, 0, 0]
        syn = b.Synapses(pop[1:6], pop[2:7], "w:1", on_pre={"a":"x_post=w", "b":"x_post+=w"}, clock=pop.clock)
        syn.connect(i=[3, 0, 2, 0], j=[1, 1, 1, 1])
        syn.w = [1, 2, 3, 4]
        syn.a.delay = (np.full(4, 5) if uniform else np.array([0, 1, 4, 9]))*b.defaultclock.dt
        syn.b.delay = 2*b.defaultclock.dt
        monitor = b.StateMonitor(pop, "x", record=True)
        net = b.Network(pop, syn, monitor)
        if segmented:
            net.run(.2*b.ms)  # Delays exceed this run; future events live in Device.
            net.run(.3*b.ms)
            net.store("pending")
            saved_pending = copy.deepcopy(device._pending_events)
            net.run(1.2*b.ms)  # Multiple wraps; rolling event history covers max delay.
            replay = np.asarray(pop.x[:]).copy()
            net.restore("pending")
            assert device._pending_events == saved_pending
            net.run(1.2*b.ms)
            np.testing.assert_array_equal(pop.x[:], replay)
        else:
            net.run(1.7*b.ms)
        snapshots.append((np.asarray(pop.x[:]).copy(), np.asarray(monitor.x).copy(), device._pending_events.copy()))
    for snapshot in snapshots[1:]:
        for a, e in zip(snapshot[:2], snapshots[0][:2], strict=True):
            np.testing.assert_array_equal(a, e)
        # Names differ between independently created networks; compare pathway values.
        assert list(snapshot[2].values()) == list(snapshots[0][2].values())


@real_metal
@pytest.mark.parametrize("uniform", [False, True])
def test_seeded_pending_order_duplicates_and_current_events(device, tmp_path, uniform):
    b.set_device("rust_standalone", engine="reference", directory=tmp_path/"export",
                 runner=ROOT/"target/release/b2-runner")
    source = b.NeuronGroup(3, "v:1", threshold="v>0.5"); source.v=1
    target = b.NeuronGroup(2, "v:1")
    syn = b.Synapses(source, target, "w:1", on_pre="v_post=0.5*v_post+w", clock=source.clock)
    syn.connect(i=[2, 0, 2, 1], j=[0, 0, 0, 0]); syn.w=[1, 2, 3, 4]
    syn.delay = (np.full(4, 2) if uniform else np.array([0, 2, 1, 2]))*b.defaultclock.dt
    monitor = b.StateMonitor(target, "v", record=True)
    b.Network(source, target, syn, monitor).run(.5*b.ms)
    model = json.loads((tmp_path/"export/model.json").read_text())
    path = model["instance"]["synapses"][0]["pathways"][0]
    path["pending"] = [{"delivery_tick":tick,"item":item} for tick,item in [(0,2),(0,0),(0,2),(2,1)]]
    attach_protocol(model)
    model_path=tmp_path/"pending.json"; model_path.write_text(json.dumps(model))
    subprocess.run([str(ROOT/"target/release/b2-runner"),str(model_path),str(tmp_path/"reference")],check=True)
    expected=load_results(model,tmp_path/"reference")
    with MetalExecutor(model,tmp_path/"metal",numeric_mode="float32",event_delivery="sparse") as executor:
        compare(executor.run(),expected)
        compare(executor.run(compute="cpu-f32",workers=4),expected)


@real_metal
@pytest.mark.parametrize("active_sources", [0, 3, 17, 65])
def test_sparse_delay_groups_capacity_bursts_and_quiet_reuse(device, tmp_path, active_sources):
    b.set_device("rust_standalone", engine="reference", directory=tmp_path/"reference",
                 runner=ROOT/"target/release/b2-runner")
    source = b.NeuronGroup(65, "v:1\ndrive:1 (constant)", threshold="v>0.5", reset="")
    source.drive = np.arange(65) < active_sources
    source.run_regularly("v=drive*int(t<0.15*ms)")
    target = b.NeuronGroup(3, "v:1")
    syn = b.Synapses(source,target,"w:1",on_pre={"a":"v_post+=w","b":"v_post+=2*w"},clock=source.clock)
    ids = np.repeat(np.arange(65),4)
    shuffle = np.random.default_rng(171).permutation(len(ids))
    syn.connect(i=ids[shuffle],j=np.zeros(len(ids),int))
    syn.w = ((np.arange(len(ids))%7)[shuffle]-3)/64
    delays = np.tile([0,0,1,3],65)[shuffle]
    syn.a.delay = delays*b.defaultclock.dt
    syn.b.delay = (0 if active_sources == 17 else 2)*b.defaultclock.dt
    empty = b.Synapses(source,target,"w:1",on_pre="v_post+=w",clock=source.clock)
    empty.connect(i=np.empty(0,int),j=np.empty(0,int)); empty.delay=3*b.defaultclock.dt
    monitor = b.StateMonitor(target,"v",record=True)
    b.Network(source,target,syn,empty,monitor).run(1.2*b.ms)
    model = json.loads((tmp_path/"reference/model.json").read_text())
    expected = load_results(model,tmp_path/"reference/rust")
    assert sum(s["events"] for s in expected["synapses"]) == active_sources*4*2*2
    with MetalExecutor(model,tmp_path/"sparse",numeric_mode="float32",event_delivery="sparse") as executor:
        compare(executor.run(compute="cpu-f32",workers=8),expected)
        for _ in range(3):
            compare(executor.run(),expected)
        # A caller may mutate results; later replays must still start from the
        # validated snapshot. Exercise both compute paths and a tighter budget.
        previous = executor.run()
        assert previous["storage_preparation_seconds"] == 0
        for pop in previous["populations"]:
            for values in pop["states"].values(): values.fill(-123)
            for values in pop["trace"].values(): values.fill(-456)
        for projection in previous["synapses"]:
            for values in projection["states"].values(): values.fill(-789)
        compare(executor.run(compute="cpu-f32",workers=3),expected)
        compare(executor.run(),expected)
        with pytest.raises(MemoryError):
            executor.run(max_buffer_bytes=1)
    with MetalExecutor(model,tmp_path/"scan",numeric_mode="float32") as executor:
        compare(executor.run(),expected)
