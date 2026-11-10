"""History fusion preserves source lanes and all remaining cross-lane barriers."""
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
import json,sys
import numpy as np
import pytest
from brian2_rust import metal_dag
from brian2_rust.gpu_dispatch_fusion import fuse_dispatches
from brian2_rust.metal import build_metal_plan,MetalExecutor
from brian2_rust.cuda import CudaExecutor
from test_metal_delays import device,real_metal
from test_cuda import real_cuda
from test_cuda_graphs import result_exact
from test_gpu_workgroup import save
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'examples'))
from gpu_stdp_compare import build_brian,DT
from brian2_rust.export import lower_network

@contextmanager
def policy(enabled):
    original=metal_dag.fuse_dispatches
    metal_dag.fuse_dispatches=lambda m,l,k,d:fuse_dispatches(m,l,k,d,pathway_history=enabled)
    try:yield
    finally:metal_dag.fuse_dispatches=original

def model_at(tmp_path):
    import brian2 as b
    b.set_device('rust_standalone',engine='reference',directory=tmp_path)
    return lower_network(build_brian(513,8)[0],16*DT*b.second)

def test_four_dispatches_keep_population_and_edge_parallelism(device,tmp_path):
    model=model_at(tmp_path/'ref')
    with policy(False):old=build_metal_plan(model,numeric_mode='float32',event_delivery='sparse')
    new=build_metal_plan(model,numeric_mode='float32',event_delivery='sparse')
    assert len(old.dispatches)==6 and len(new.dispatches)==4
    assert old.logical==new.logical and old.buffers==new.buffers
    assert [d.lanes for d in new.dispatches]==[513,513,4104,513]
    assert [n for k in old.kernels for n in k.nodes]==[n for k in new.kernels for n in k.nodes]
    assert new.dispatches[0].role=='population-pathway-history'
    assert all(d.dependencies==((new.dispatches[i-1].entry,) if i else ()) for i,d in enumerate(new.dispatches))

@pytest.mark.parametrize('hazard',['clock','range','read-after-write','write-after-read','write-after-write','unproven-owner'])
def test_history_never_crosses_unproven_barrier(device,tmp_path,hazard):
    model=model_at(tmp_path/'ref')
    with policy(False):p=build_metal_plan(model,numeric_mode='float32')
    # Exercise the second history moving across the first target pathway.
    k=list(p.kernels);d=list(p.dispatches);h=d[3];middle=d[2]
    if hazard=='clock':d[2]=replace(middle,clock=h.clock+1)
    elif hazard=='range':k[3]=replace(k[3],neurons=k[3].neurons-1);d[3]=replace(h,lanes=h.lanes-1)
    elif hazard=='unproven-owner':d[0]=replace(d[0],role='canonical-linked-population')
    else:
        slot=h.bindings[0 if hazard=='read-after-write' else 1]
        typ='const uchar' if hazard=='write-after-read' else 'uchar'
        d[2]=replace(middle,bindings=middle.bindings+(slot,),types=middle.types+(typ,))
    _,after=fuse_dispatches(model,p.logical,k,d)
    assert any(x.entry==h.entry for x in after)


def full_model(path,variant):
    import brian2 as b
    b.set_device('rust_standalone',engine='reference',directory=path)
    dt=DT*b.second
    pre=b.NeuronGroup(257,'v:1',threshold='v>0.5',reset='v=0',events={'burst':'v>0.25'},dt=dt,name='pre')
    post=b.NeuronGroup(263,'v:1',threshold='v>0.5',reset='v=0',dt=2*dt,name='post')
    pre.run_regularly('v+=0.25');post.run_regularly('v+=0.125')
    syn=b.Synapses(pre,post,'w:1',on_pre='v_post+=w; w+=0.0009765625',on_post='w*=0.5',
        on_event={'pre':'burst' if variant=='named' else 'spike','post':'spike'},clock=pre.clock,name='projection')
    edges=np.arange(1028);syn.connect(i=edges%257,j=edges%263);syn.w=1/128
    syn.pre.delay=(edges%3)*dt;syn.post.delay=(edges%2)*2*dt
    net=b.Network(pre,post,syn,b.EventMonitor(pre,'burst' if variant=='named' else 'spike'),b.SpikeMonitor(post),b.StateMonitor(post,'v',record=[0,262]))
    if variant=='early':net.schedule=['start','groups','synapses','thresholds','resets','end']
    model=lower_network(net,16*dt)
    with policy(False):old=build_metal_plan(model,numeric_mode='float32')
    new=build_metal_plan(model,numeric_mode='float32')
    if variant=='early':assert new.sha256==old.sha256  # Clock boundary keeps the original history slots.
    else:assert len(new.dispatches)<len(old.dispatches)
    return model

@pytest.mark.parametrize('backend',['cpu-f32',pytest.param('metal',marks=real_metal),pytest.param('cuda',marks=real_cuda)])
@pytest.mark.parametrize('variant',['stdp','early','named','subgroup','hazard','empty'])
def test_full_results_repeat_and_early_named_subgroups(device,tmp_path,backend,variant):
    from test_gpu_edge_pathway import model_at as edge_model
    from test_gpu_target_pathway import model_at as target_model
    if variant=='stdp':model=model_at(tmp_path/'ref')
    elif variant=='hazard':model=target_model(tmp_path/'ref',hazard=True)
    elif variant=='empty':model=edge_model(tmp_path/'ref',edges=0,pending=False)
    elif variant=='subgroup':model=edge_model(tmp_path/'ref')
    else:model=full_model(tmp_path/'ref',variant)
    # edge_model uses source/target subgroups, so this also verifies rejection.
    cls=CudaExecutor if backend=='cuda' else MetalExecutor
    results=[]
    for enabled in (False,True):
        with policy(enabled):
            if backend=='cpu-f32':
                from types import SimpleNamespace
                folder=tmp_path/str(enabled);folder.mkdir()
                ex=SimpleNamespace(model=model,plan=build_metal_plan(model,numeric_mode='float32',event_delivery='sparse'),directory=folder,compile_seconds=0,device_name='CPU f32')
            else:ex=cls(model,tmp_path/str(enabled),numeric_mode='float32',event_delivery='sparse')
        try:
            run=(lambda:metal_dag.run_dag(ex,max_buffer_bytes=512*1024**2,compute='cpu-f32',workers=3)) if backend=='cpu-f32' else ex.run
            first=run();second=run();result_exact(second,first);results.append(second)
        finally:
            if backend!='cpu-f32':ex.close()
    result_exact(results[1],results[0]);save(tmp_path/'history-results.npz',results[1],results[0])
