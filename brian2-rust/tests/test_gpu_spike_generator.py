"""Integer external spike schedules on both native GPU platforms and CPU control."""
import copy
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import brian2 as b
import numpy as np
import pytest

from brian2_rust.cuda import CudaExecutor, build_cuda_plan, write_cuda_results
from brian2_rust.metal import MetalExecutor, build_metal_plan, write_metal_results, population_arrays
from brian2_rust.protocol import attach_protocol
from brian2_rust.results import load_results
from brian2_rust.spec import bits
from test_metal_delays import device, ROOT
from test_metal_plasticity import equivalent

BACKENDS = ["cpu-f32", pytest.param("metal", marks=pytest.mark.skipif(
    os.environ.get("B2_TEST_METAL") != "1", reason="requires Apple GPU")),
    pytest.param("cuda", marks=pytest.mark.skipif(
    os.environ.get("B2_TEST_CUDA") != "1", reason="requires NVIDIA GPU"))]


def execute(model, path, backend, route="scan"):
    path.mkdir()
    if backend == "cpu-f32":
        control = SimpleNamespace(model=model, directory=path,
            plan=build_cuda_plan(model, numeric_mode="float32", event_delivery=route))
        return CudaExecutor._cpu_control(control, 512*1024**2, 3)
    cls, writer = ((MetalExecutor, write_metal_results) if backend == "metal"
                   else (CudaExecutor, write_cuda_results))
    with cls(model, path, numeric_mode="float32", event_delivery=route) as executor:
        result = executor.run()
        equivalent(result, executor.run(), exact=True)
        writer(model, result, path/"transport")
        equivalent(load_results(model, path/"transport"), result, exact=True)
        return result


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("large_tick", [False, True])
def test_independent_periodic_empty_multiple_clocks(device, tmp_path, backend, large_tick):
    b.set_device("rust_standalone", engine="reference", directory=tmp_path/"ref",
                 runner=ROOT/"target/release/b2-runner")
    dt = b.second/1024
    source = b.SpikeGeneratorGroup(5, [2, 0, 1, 0], [2, 0, 1, 3]*dt,
                                   period=4*dt, dt=dt, name="input")
    empty = b.SpikeGeneratorGroup(3, [], []*b.second, dt=2*dt, name="empty")
    monitor, quiet = b.SpikeMonitor(source), b.SpikeMonitor(empty)
    net = b.Network(source, empty, monitor, quiet)
    net.run(4*dt)
    net.run(8*dt)
    model = json.loads((device.last_run_directory/"model.json").read_text())
    if large_tick:
        shift = 2**25
        model["run"]["start"] = bits((4+shift)/1024)
        for clock, inst in zip(model["definition"]["clocks"], model["run"]["clocks"], strict=True):
            from brian2_rust.metal import number
            inst["start_tick"] += int(shift/(1024*number(clock["dt"])))
        for inst in model["instance"]["populations"]:
            inst["spike_generator"]["spike_ticks"] = [t+shift for t in inst["spike_generator"]["spike_ticks"]]
        attach_protocol(model)
    path = tmp_path/"input.json"; path.write_text(json.dumps(model))
    subprocess.run([str(ROOT/"target/release/b2-runner"), str(path), str(tmp_path/"oracle")], check=True)
    expected = load_results(model, tmp_path/"oracle")
    actual = execute(model, tmp_path/backend, backend)
    equivalent(actual, expected, exact=True)
    p = next(i for i, pop in enumerate(model["definition"]["populations"]) if pop["name"] == "input")
    np.testing.assert_array_equal(actual["populations"][p]["indices"], [0,1,2,0,0,1,2,0])
    np.testing.assert_array_equal(actual["populations"][p]["spike_ticks"], np.arange(4,12)+(2**25 if large_tick else 0))
    plan = build_metal_plan(model, numeric_mode="float32")
    # Sparse schedule memory is O(neurons + events), not neurons * duration.
    arrays, _ = population_arrays(model, p, plan.kernels[p], 512*1024**2)
    assert arrays[1].dtype == np.int64
    assert arrays[1].size == 5+1+8
    with pytest.raises(MemoryError, match="spike schedule"):
        population_arrays(model, p, plan.kernels[p], 8)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("route", ["scan", "sparse"])
@pytest.mark.parametrize("mutable", [False, True])
def test_generator_delays_subgroups_without_monitor(device, tmp_path, backend, route, mutable):
    b.set_device("rust_standalone", engine="reference", directory=tmp_path/"ref",
                 runner=ROOT/"target/release/b2-runner")
    dt = b.second/1024
    source = b.SpikeGeneratorGroup(6, [3,0,2,1], [2,0,1,2]*dt, period=4*dt, dt=dt)
    target = b.NeuronGroup(5, "v:1", clock=source.clock)
    syn = b.Synapses(source, target[1:4], "w:1",
                     on_pre="v_post += w"+("; w += 0.125" if mutable else ""), clock=source.clock)
    syn.connect(i=[3,1,2,1,4], j=[1,1,2,1,0]); syn.w=[.25,.5,.75,1,2]
    syn.delay=[0,1,4,2,3]*dt
    monitor = b.StateMonitor(target,"v",record=True)
    b.Network(source,target,syn,monitor).run(10*dt)
    model=json.loads((device.last_run_directory/"model.json").read_text())
    expected=load_results(model,device.last_run_directory/"rust")
    actual=execute(model,tmp_path/backend,backend,route)
    equivalent(actual,expected,exact=True)
    for a,e in zip(actual["populations"],expected["populations"],strict=True):
        for name,stream in a["event_streams"].items():
            for field in ("ticks","indices"):
                np.testing.assert_array_equal(stream[field],e["event_streams"][name][field])


@pytest.mark.parametrize("backend", BACKENDS[1:])
@pytest.mark.parametrize("queued", [False, True])
def test_generator_device_pending_restore_and_queued_build(device, tmp_path, backend, queued):
    snapshots=[]
    for engine in ("reference",backend):
        device.reinit()
        options={"numeric_mode":"float32","event_delivery":"sparse"} if engine != "reference" else {}
        b.set_device("rust_standalone",engine=engine,directory=tmp_path/engine,
                     runner=ROOT/"target/release/b2-runner",build_on_run=not queued,**options)
        dt=b.second/1024
        source=b.SpikeGeneratorGroup(3,[2,0,1],[2,0,1]*dt,period=4*dt,dt=dt,name="input")
        target=b.NeuronGroup(2,"v:1",clock=source.clock,name="target")
        syn=b.Synapses(source,target,"w:1",on_pre="v_post+=w",clock=source.clock,name="projection")
        syn.connect(i=[2,0,1],j=[0,0,1]);syn.w=[.25,.5,1];syn.delay=[5,0,2]*dt
        spikes=b.SpikeMonitor(source);monitor=b.StateMonitor(target,"v",record=True)
        net=b.Network(source,target,syn,spikes,monitor)
        net.run(3*dt)
        if not queued:
            net.store("pending")
            pending=copy.deepcopy(device._pending_events)
            net.run(7*dt)
            expected=np.asarray(target.v[:]).copy()
            net.restore("pending")
            assert device._pending_events==pending
            net.run(7*dt)
            np.testing.assert_array_equal(target.v[:],expected)
        else:
            net.run(7*dt)
            device.build()
        snapshots.append((np.asarray(target.v[:]).copy(),np.asarray(monitor.v).copy(),
                          np.asarray(spikes.t[:]).copy(),np.asarray(spikes.i[:]).copy(),
                          copy.deepcopy(device._pending_events)))
    for a,e in zip(snapshots[1][:4],snapshots[0][:4],strict=True):np.testing.assert_array_equal(a,e)
    assert snapshots[1][4]==snapshots[0][4]
