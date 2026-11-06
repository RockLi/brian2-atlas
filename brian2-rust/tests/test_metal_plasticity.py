"""Canonical GPU plasticity, state integration and typed result lifecycle."""
import json
import numpy as np
import brian2 as b
import pytest

from test_metal_delays import device, real_metal, ROOT
from brian2_rust.metal import MetalExecutor, write_metal_results
from brian2_rust.results import load_results


def equivalent(actual, expected, *, exact=False):
    for a,e in zip(actual["populations"],expected["populations"],strict=True):
        for category in ("states","trace"):
            for name,value in a[category].items():
                if exact: np.testing.assert_array_equal(value,e[category][name])
                else: np.testing.assert_allclose(value,e[category][name],rtol=2e-5,atol=1e-7)
        for name in ("spike_ticks","indices","counts","last_spikes"):
            np.testing.assert_array_equal(a[name],e[name])
    for a,e in zip(actual["synapses"],expected["synapses"],strict=True):
        assert a["events"]==e["events"]
        for name,value in a["states"].items():
            if exact: np.testing.assert_array_equal(value,e["states"][name])
            else: np.testing.assert_allclose(value,e["states"][name],rtol=2e-5,atol=1e-7)


@real_metal
@pytest.mark.parametrize("integrate",[False,True])
def test_gpu_plasticity_and_clock_driven_integration(device,tmp_path,integrate):
    plasticity_case(device,tmp_path,integrate,MetalExecutor,write_metal_results)


def plasticity_case(device,tmp_path,integrate,executor_class,write_results):
    b.set_device("rust_standalone",engine="reference",directory=tmp_path/"reference",runner=ROOT/"target/release/b2-runner")
    pre=b.NeuronGroup(4,"v:1\nu:1",threshold="v>0.5",reset="");pre.v=1
    post=b.NeuronGroup(5,"v:1\nu:1",threshold="v>0.5",reset="");post.v=1
    if integrate:
        syn=b.Synapses(pre,post,"da/dt=b/ms:1 (clock-driven)\ndb/dt=a/ms:1 (clock-driven)\nz=a+b:1 (constant over dt)\nu_pre=a:1 (summed)\nu_post=b+u_pre:1 (summed)",
                       on_pre="a+=0.125",on_post="b+=0.25",method="euler",clock=pre.clock)
    else:
        syn=b.Synapses(pre,post,"dApre/dt=-Apre/(10*ms):1 (event-driven)\ndApost/dt=-Apost/(10*ms):1 (event-driven)\nw:1",
                       on_pre="Apre+=0.125; w=clip(w+Apost,0,1); v_post+=w/64",
                       on_post="Apost-=0.0625; w=clip(w+Apre,0,1)",clock=pre.clock)
    syn.connect(i=[3,0,2,0,1,3],j=[1,1,2,1,3,0])
    syn.pre.delay=[0,1,3,0,2,1]*b.defaultclock.dt
    syn.post.delay=2*b.defaultclock.dt
    if integrate:syn.a=.5;syn.b=.25
    else:syn.w=.5
    monitor=b.StateMonitor(post,["v","u"],record=True)
    b.Network(pre,post,syn,monitor).run(.8*b.ms)
    model=json.loads((tmp_path/"reference/model.json").read_text())
    reference=load_results(model,tmp_path/"reference/rust")
    for route in ("scan","sparse"):
        with executor_class(model,tmp_path/route,numeric_mode="float32",event_delivery=route) as executor:
            assert any(stage.role==("endpoint-owned-summed" if integrate else "target-owned-synapse-pathway") for stage in executor.plan.dispatches)
            result=executor.run();mirror=executor.run(compute="cpu-f32",workers=3)
            equivalent(result,reference)
            equivalent(result,mirror,exact=True)
            equivalent(executor.run(),result,exact=True)
            write_results(model,result,tmp_path/f"{route}-transport")
            equivalent(load_results(model,tmp_path/f"{route}-transport"),result,exact=True)


@real_metal
def test_gpu_mutable_synapse_segmented_pending_restore(device,tmp_path):
    segmented_case(device,tmp_path,"metal")


def segmented_case(device,tmp_path,gpu_engine):
    snapshots=[]
    for engine in ("reference",gpu_engine):
        device.reinit()
        options={"numeric_mode":"float32","event_delivery":"sparse"} if engine==gpu_engine else {}
        b.set_device("rust_standalone",engine=engine,directory=tmp_path/engine,runner=ROOT/"target/release/b2-runner",**options)
        pop=b.NeuronGroup(6,"v:1",threshold="v>0.5",reset="");pop.v=1
        syn=b.Synapses(pop[1:5],pop[2:6],"w:1\nApre:1\nApost:1",on_pre="Apre+=0.125; w=clip(w+Apost,0,1)",
                       on_post="Apost-=0.0625; w=clip(w+Apre,0,1)",clock=pop.clock)
        syn.connect(i=[3,0,2,0],j=[1,1,2,1]);syn.w=.5
        syn.pre.delay=[0,1,4,2]*b.defaultclock.dt;syn.post.delay=3*b.defaultclock.dt
        net=b.Network(pop,syn);net.run(.2*b.ms);net.store("pending");net.run(.8*b.ms)
        expected=np.asarray(syn.w[:]).copy()
        net.restore("pending");net.run(.8*b.ms)
        np.testing.assert_array_equal(syn.w[:],expected)
        snapshots.append([np.asarray(getattr(syn,n)[:]).copy() for n in ("w","Apre","Apost")])
    for a,e in zip(*snapshots,strict=True):np.testing.assert_array_equal(a,e)
