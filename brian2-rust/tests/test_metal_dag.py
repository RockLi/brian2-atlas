"""Coupled Metal schedule semantics, independent oracle and capability boundaries."""
import copy
import json
import os
from pathlib import Path

import brian2 as b
import brian2_rust
import numpy as np
import pytest

from brian2_rust.metal import MetalExecutor, build_metal_plan, write_metal_results
from brian2_rust.plan import PlanValidationError
from brian2_rust.protocol import attach_protocol
from brian2_rust.results import load_results

ROOT = Path(__file__).resolve().parents[1]
real_metal = pytest.mark.skipif(os.environ.get("B2_TEST_METAL") != "1", reason="requires an Apple GPU")


@pytest.fixture
def coupled(tmp_path):
    previous = b.get_device()
    from brian2.devices.device import all_devices
    device = all_devices["rust_standalone"]
    device.reinit()
    b.set_device("rust_standalone", engine="reference", directory=tmp_path/"reference",
                 runner=ROOT/"target/release/b2-runner")
    try:
        source = b.NeuronGroup(8, "dv/dt=drive/ms:1\ndrive:1 (constant)", threshold="v>=0.95",
                               reset="v=0", refractory=.2*b.ms, method="euler")
        target = b.NeuronGroup(9, "v:1\nx:1", threshold="v>=0.95", reset="v=0")
        source.drive = np.arange(1,9)*1.25
        synapse = b.Synapses(source, target, "w:1\nx_post=w*v_pre:1 (summed)",
                            on_pre={"first": "v_post+=w", "second": "v_post+=2*w"}, clock=source.clock)
        # Duplicate edges, unsorted source order, a disconnected target.
        synapse.connect(i=[7,0,3,0,2,7,1,4], j=[2,2,2,2,4,5,5,7])
        synapse.w = np.arange(1,9)/32
        mon = b.StateMonitor(target, ["v","x"], record=True)
        spikes = b.SpikeMonitor(target)  # Source deliberately has no SpikeMonitor.
        b.Network(source,target,synapse,mon,spikes).run(1*b.ms)
        model = json.loads((tmp_path/"reference/model.json").read_text())
        expected = load_results(model, tmp_path/"reference/rust")
        yield model, expected
    finally:
        device.reinit()
        b.set_device(previous)


def test_coupled_plan_and_fail_closed_boundaries(coupled):
    model, _ = coupled
    plan = build_metal_plan(model, numeric_mode="float32", event_delivery="sparse")
    assert plan.strategy == "canonical-target-owned-dag"
    from brian2_rust.plan import build_execution_plan
    assert build_execution_plan(model,backend="metal",numeric_mode="float32",event_delivery="sparse") == plan
    assert build_metal_plan(model,numeric_mode="float32").event_delivery == "scan"
    with pytest.raises(PlanValidationError,match="event_delivery"):
        build_execution_plan(model,event_delivery="sparse")
    with pytest.raises(PlanValidationError,match="event_delivery"):
        build_metal_plan(model,numeric_mode="float32",event_delivery="invalid")
    assert plan.event_delivery == "sparse"
    assert tuple(n for k in plan.kernels for n in k.nodes) == tuple(n.id for n in plan.logical.nodes if n.id not in plan.elided_nodes)
    assert len(plan.buffers) == 12*len(model["definition"]["populations"])+11
    for n, stage in enumerate(plan.dispatches):
        assert stage.dependencies == ((plan.dispatches[n-1].entry,) if n else ())
    for change, message in (("custom_event", "declared"),):
        bad = copy.deepcopy(model)
        # Exercise the planner's bounded-subset check after a trusted derivation;
        # public entry validation and plan substitution are tested separately.
        bad["instance"]["synapses"][0]["pathways"][0]["event"] = "custom"
        from brian2_rust.metal_dag import derive_dag
        with pytest.raises(PlanValidationError, match=message):
            derive_dag(bad, plan.logical)


@real_metal
@pytest.mark.parametrize("before_threshold", [False, True])
def test_coupled_gpu_reference_and_float32_mirror(coupled, tmp_path, before_threshold):
    model, expected = coupled
    if before_threshold:
        # Re-export canonical slots with pathway before threshold, so it observes
        # the previous emission. Reordering raw schedule JSON would be invalid.
        from brian2_rust.schedule import build_schedule
        slots = ["start", "groups", "synapses", "thresholds", "resets", "end"]
        model["definition"]["schedule"] = build_schedule(model["definition"], model["instance"], slots)
        attach_protocol(model)
        import subprocess
        path = tmp_path/"before.json"
        path.write_text(json.dumps(model))
        subprocess.run([str(ROOT/"target/release/b2-runner"),str(path),str(tmp_path/"before")],check=True)
        expected = load_results(model,tmp_path/"before")
    with MetalExecutor(model,tmp_path/"metal",numeric_mode="float32",event_delivery="sparse") as executor:
        result = executor.run()
        mirror = executor.run(compute="cpu-f32",workers=3)
        replay = executor.run()
        with MetalExecutor(model,tmp_path/"scan",numeric_mode="float32",event_delivery="scan") as baseline:
            scanned = baseline.run()
        with pytest.raises(MemoryError):
            executor.run(max_buffer_bytes=1)
    for p, (actual, control, reference) in enumerate(zip(result["populations"],mirror["populations"],expected["populations"],strict=True)):
        for field in ("states","trace"):
            for name in actual[field]:
                np.testing.assert_array_equal(actual[field][name],control[field][name])
                np.testing.assert_array_equal(actual[field][name],scanned["populations"][p][field][name])
                np.testing.assert_array_equal(actual[field][name],replay["populations"][p][field][name])
                np.testing.assert_allclose(actual[field][name],reference[field][name],rtol=2e-6,atol=1e-7)
        for field in ("spike_ticks","indices","counts","last_spikes"):
            np.testing.assert_array_equal(actual[field],reference[field])
        for name in ("ticks","indices"):
            np.testing.assert_array_equal(actual["event_streams"]["spike"][name],reference["event_streams"]["spike"][name])
    assert result["synapses"][0]["events"] == expected["synapses"][0]["events"]
    write_metal_results(model,result,tmp_path/"transport")
    transported = load_results(model,tmp_path/"transport")
    assert transported["synapses"][0]["events"] == expected["synapses"][0]["events"]
    for p in range(len(result["populations"])):
        np.testing.assert_array_equal(transported["populations"][p]["event_streams"]["spike"]["ticks"],result["populations"][p]["event_streams"]["spike"]["ticks"])


@real_metal
def test_coupled_device_subgroups_segmented_restore_and_empty_edges(tmp_path):
    previous = b.get_device()
    from brian2.devices.device import all_devices
    device = all_devices["rust_standalone"]
    snapshots = []
    try:
        for engine in ("reference", "metal"):
            device.reinit()
            options = {"numeric_mode": "float32", "event_delivery": "sparse"} if engine == "metal" else {}
            b.set_device("rust_standalone",engine=engine,directory=tmp_path/engine,
                         runner=ROOT/"target/release/b2-runner",recording_window_steps=2,**options)
            pop = b.NeuronGroup(12,"v:1\nx:1",threshold="v>0.9",reset="v=0",refractory=.2*b.ms)
            pop.run_regularly("v+=0.125")
            syn = b.Synapses(pop[1:10],pop[2:11],"w:1\nx_post=w:1 (summed)",on_pre="v_post+=w",clock=pop.clock)
            syn.connect(i=[8,0,2,0],j=[2,2,0,2]); syn.w=.0625
            empty = b.Synapses(pop,pop,"w:1",on_pre="v_post+=w",clock=pop.clock)
            empty.connect(i=np.array([],int),j=np.array([],int))
            monitor = b.StateMonitor(pop,["v","x"],record=True)
            spike = b.SpikeMonitor(pop)
            net = b.Network(pop,syn,empty,monitor,spike)
            net.run(.5*b.ms); net.store("middle"); net.run(1*b.ms)
            replay = np.asarray(pop.v[:]).copy()
            net.restore("middle"); net.run(1*b.ms)
            np.testing.assert_array_equal(pop.v[:],replay)
            snapshots.append((np.asarray(pop.v[:]).copy(),np.asarray(monitor.v).copy(),
                              np.asarray(spike.i[:]).copy(),np.asarray(spike.t[:]).copy(),np.asarray(syn.w[:]).copy()))
        for gpu, reference in zip(snapshots[1],snapshots[0],strict=True):
            np.testing.assert_array_equal(gpu,reference)
    finally:
        device.reinit(); b.set_device(previous)


@real_metal
def test_event_and_summed_use_distinct_reference_orders(tmp_path):
    previous = b.get_device()
    from brian2.devices.device import all_devices
    device = all_devices["rust_standalone"]
    try:
        device.reinit()
        b.set_device("rust_standalone",engine="metal",numeric_mode="float32",event_delivery="sparse",directory=tmp_path/"metal",
                     runner=ROOT/"target/release/b2-runner")
        source = b.NeuronGroup(3,"v:1",threshold="v>1")
        target = b.NeuronGroup(2,"v:1\nx:1")
        source.v=2
        syn = b.Synapses(source,target,"w:1\nx_post=w:1 (summed)",on_pre="v_post+=w",clock=source.clock)
        # Creation order yields ((1+2**24)-2**24)==0 in f32;
        # source order yields ((2**24-2**24)+1)==1.
        syn.connect(i=[2,0,1],j=[0,0,0]);syn.w=[1,2**24,-2**24]
        b.Network(source,target,syn).run(.1*b.ms)
        np.testing.assert_array_equal(target.v[:],[1,0])
        np.testing.assert_array_equal(target.x[:],[0,0])
    finally:
        device.reinit(); b.set_device(previous)


@real_metal
@pytest.mark.parametrize("active_sources", [0, 3, 64, 257])
def test_sparse_queue_collisions_order_and_reuse(tmp_path, active_sources):
    """Many writers to one target, duplicate edges, sparse sorting and bursts."""
    previous = b.get_device()
    from brian2.devices.device import all_devices
    device = all_devices["rust_standalone"]
    try:
        device.reinit()
        b.set_device("rust_standalone",engine="reference",directory=tmp_path/"reference",
                     runner=ROOT/"target/release/b2-runner")
        source = b.NeuronGroup(257,"v:1\nseed:1 (constant)",threshold="v>1",reset="")
        source.run_regularly("v=seed*int(t<0.15*ms)")
        # Always include the cancellation triplet, then add other sources.
        chosen = ([2,129,255] + [i for i in range(257) if i not in (2,129,255)])[:active_sources]
        initial_drive = np.zeros(257);initial_drive[chosen]=2;source.seed=initial_drive
        target = b.NeuronGroup(3,"v:1")
        syn = b.Synapses(source,target,"w:1",on_pre={"a":"v_post+=w","b":"v_post+=w"},clock=source.clock)
        ids = np.repeat(np.arange(257),2)
        order = np.random.default_rng(71).permutation(len(ids))
        syn.connect(i=ids[order],j=np.zeros(len(ids),int))
        weights = np.zeros(257);weights[[2,129,255]]=[2**24,-2**24,1]
        syn.w=weights[ids[order]]
        b.Network(source,target,syn).run(.4*b.ms)
        model=json.loads((tmp_path/"reference/model.json").read_text())
        expected=load_results(model,tmp_path/"reference/rust")
        with MetalExecutor(model,tmp_path/"sparse",numeric_mode="float32",event_delivery="sparse") as executor:
            control=executor.run(compute="cpu-f32",workers=8)
            for _ in range(3):
                actual=executor.run()
                for p in range(2):
                    for name, value in actual["populations"][p]["states"].items():
                        np.testing.assert_array_equal(value,control["populations"][p]["states"][name])
                        if model["definition"]["populations"][p]["count"] == 257:
                            np.testing.assert_array_equal(value,expected["populations"][p]["states"][name])
                        else:
                            # Every subsequent cancellation pass loses the old
                            # residual in f32; f64 retains it. Do not relax tolerance.
                            np.testing.assert_array_equal(value,[2 if active_sources else 0,0,0])
                            np.testing.assert_array_equal(expected["populations"][p]["states"][name],[8 if active_sources else 0,0,0])
                            assert actual["populations"][p]["event_streams"] == {}
                assert actual["synapses"][0]["events"] == active_sources*2*2*2
        with MetalExecutor(model,tmp_path/"scan",numeric_mode="float32",event_delivery="scan") as executor:
            scanned=executor.run()
            for p in range(2):
                for name,value in actual["populations"][p]["states"].items():
                    np.testing.assert_array_equal(value,scanned["populations"][p]["states"][name])
    finally:
        device.reinit(); b.set_device(previous)


@real_metal
def test_inert_synaptic_network_keeps_identity_results(tmp_path):
    previous=b.get_device()
    from brian2.devices.device import all_devices
    device=all_devices["rust_standalone"]
    try:
        device.reinit()
        b.set_device("rust_standalone",engine="metal",numeric_mode="float32",event_delivery="sparse",directory=tmp_path/"metal",
                     runner=ROOT/"target/release/b2-runner")
        pop=b.NeuronGroup(3,"v:1",threshold="False",reset="");pop.v=[.125,.25,.5]
        syn=b.Synapses(pop,pop,"w:1",on_pre="v_post+=w",clock=pop.clock)
        syn.connect(i=[0],j=[2]);syn.w=.125
        b.Network(pop,syn).run(.2*b.ms)
        np.testing.assert_array_equal(pop.v[:],[.125,.25,.5])
        np.testing.assert_array_equal(syn.w[:],[.125])
        assert device.last_execution_plan.elided_nodes
        assert device.last_execution_plan.event_delivery == "sparse"
    finally:
        device.reinit();b.set_device(previous)
