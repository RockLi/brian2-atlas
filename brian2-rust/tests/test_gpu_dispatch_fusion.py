"""Fusion proof boundaries and full-result conformance on real runtimes."""
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import sys
import brian2 as b
import numpy as np
import pytest

from brian2_rust import metal_dag
from brian2_rust.gpu_dispatch_fusion import fuse_dispatches,_arguments
from brian2_rust.metal import build_metal_plan,MetalExecutor
from brian2_rust.cuda import CudaExecutor
from test_metal_delays import device,real_metal
from test_cuda import real_cuda
from test_cuda_graphs import result_exact

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'examples'))
from cuda_dag_benchmark import workload


@contextmanager
def policy(name):
    original=metal_dag.fuse_dispatches
    metal_dag.fuse_dispatches=(lambda m,l,k,d:(k,d)) if name=='previous' else (
        lambda m,l,k,d:fuse_dispatches(m,l,k,d,target_delivery=name=='full'))
    try:yield
    finally:metal_dag.fuse_dispatches=original


@pytest.fixture
def model(device,tmp_path):
    return workload(64,64,8,tmp_path/'model')


def plan(model,name='full'):
    with policy(name):return build_metal_plan(model,numeric_mode='float32',event_delivery='sparse')


def test_lane_local_fusion_reduces_five_dispatches_to_three(model):
    before=plan(model,'previous');source=plan(model,'source');after=plan(model)
    assert [len(p.dispatches) for p in (before,source,after)]==[5,4,3]
    assert after.logical==before.logical and after.buffers==before.buffers
    assert [len(d.bindings) for d in after.dispatches]==[30,28,12]
    assert [n for k in before.kernels for n in k.nodes]==[n for k in after.kernels for n in k.nodes]
    for i,d in enumerate(after.dispatches):
        assert d.dependencies==((after.dispatches[i-1].entry,) if i else ())
        assert len(set(d.bindings))==len(d.bindings)


@pytest.mark.parametrize('hazard',['read-after-write','write-after-read','write-after-write','clock','subgroup'])
def test_enqueue_cannot_cross_alias_hazard_or_clock_or_subgroup(model,hazard):
    before=plan(model,'previous');kernels=list(before.kernels);dispatches=list(before.dispatches)
    enqueue=dispatches[2]
    if hazard=='clock':dispatches[1]=replace(dispatches[1],clock=enqueue.clock+1)
    elif hazard=='subgroup':
        kernels[2]=replace(kernels[2],neurons=kernels[2].neurons-1)
        dispatches[2]=replace(enqueue,lanes=enqueue.lanes-1)
    else:
        binding=enqueue.bindings[0 if hazard=='read-after-write' else 1]
        dtype='const uchar' if hazard=='write-after-read' else 'uchar'
        old=dispatches[1];dispatches[1]=replace(old,bindings=old.bindings+(binding,),types=old.types+(dtype,))
    _,result=fuse_dispatches(model,before.logical,kernels,dispatches)
    assert len(result)==5 and result[2].role=='delay-source-group-enqueue'


def test_binding_budget_and_type_aliases_are_enforced(model):
    p=plan(model,'previous');a,b=p.dispatches[0],p.dispatches[2]
    assert len(_arguments(a,b)[0])==30
    assert _arguments(a,replace(b,bindings=b.bindings+(999,),types=b.types+('const uint',))) is None
    bad=replace(b,types=('const float',)+b.types[1:])
    assert _arguments(a,bad) is None
    bindings,types,_=_arguments(a,b)
    assert types[bindings.index(b.bindings[0])]=='uchar'  # mutable producer + const consumer


def test_source_state_read_prevents_target_fusion(model):
    before=plan(model,'previous')
    # The pass must conservatively reject a pre-state dependency, even if a
    # field-sensitive analysis might prove this particular field independent.
    alias=next(iter(model['definition']['synapses'][0]['pre_state_aliases']))
    model['definition']['synapses'][0]['code_objects'][0]['effects']['reads'].append(alias)
    _,result=fuse_dispatches(model,before.logical,before.kernels,before.dispatches)
    assert len(result)==4 and sum(d.role=='delayed-target-sparse' for d in result)==2


def control(model,name,directory,workers=1):
    directory.mkdir()
    ex=SimpleNamespace(model=model,plan=plan(model,name),directory=directory,compile_seconds=0,device_name='CPU f32')
    return metal_dag.run_dag(ex,max_buffer_bytes=512*1024**2,compute='cpu-f32',workers=workers)


@pytest.mark.parametrize('workers',[1,3])
def test_compiled_cpu_fusion_keeps_complete_results(model,tmp_path,workers):
    expected=control(model,'previous',tmp_path/'previous',workers)
    for name in ('source','full'):
        result_exact(control(model,name,tmp_path/name,workers),expected)


@pytest.mark.parametrize('backend',[pytest.param('metal',marks=real_metal),pytest.param('cuda',marks=real_cuda)])
def test_real_gpu_fusion_and_replay_match_previous_plan(model,tmp_path,backend):
    expected=control(model,'previous',tmp_path/'control',3)
    for name in ('previous','source','full'):
        with policy(name):
            ex=(MetalExecutor if backend=='metal' else CudaExecutor)(model,tmp_path/name,numeric_mode='float32',event_delivery='sparse')
        with ex:
            for _ in range(2):result_exact(ex.run(),expected)
            if backend=='cuda':
                for mode in ('direct','graph','chunked'):result_exact(ex.run(dag_execution=mode),expected)


def variant_model(directory,kind):
    from brian2_rust.export import lower_network
    b.set_device('rust_standalone',engine='reference',directory=directory,
        runner=Path(__file__).resolve().parents[1]/'target/release/b2-runner')
    dt=.1*b.ms
    pop=b.NeuronGroup(16,'v:1\nx:1',threshold='v>0.5',reset='v=.75*v+.25',dt=dt,
        events={'burst':'v>0.3'} if kind=='named' else None,name='population')
    pop.v=(np.arange(16)+1)/16
    objects=[pop,b.SpikeMonitor(pop)]
    if kind=='named':objects.append(b.EventMonitor(pop,'burst',variables=['x']))
    for q in range(1 if kind=='shared-pathways' else 2):
        source=pop[1:15] if kind=='subgroup' else pop
        if kind=='clock' and q==1:
            source=b.NeuronGroup(16,'v:1',threshold='v>0.5',dt=2*dt,name='slow_source')
            source.v=1;objects.extend([source,b.SpikeMonitor(source)])
        target=pop[2:14] if kind=='subgroup' else pop[q*8:(q+1)*8] if kind=='target-range' else pop
        code=('x_post+=w*v_pre' if q==0 else 'v_post+=w') if kind=='pre-read' else 'x_post+=w'
        if kind=='shared-pathways':code={'a':'x_post+=w','b':'x_post-=w'}
        options={'clock':source.clock}
        syn=b.Synapses(source,target,'w:1 (constant)',on_pre=code,
            on_event='burst' if kind=='named' else 'spike',name=f'projection_{q}',**options)
        edges=np.arange(48);syn.connect(i=edges%len(source),j=(edges*7+q)%len(target))
        syn.w=.02 if q==0 else -.01
        if kind=='shared-pathways':syn.a.delay=dt;syn.b.delay=2*dt
        else:syn.delay=(edges%3 if kind=='heterogeneous' else np.ones(48))*dt
        objects.append(syn)
    return lower_network(b.Network(*objects),24*dt)


@pytest.mark.parametrize('kind',['pre-read','subgroup','clock','heterogeneous','named','shared-pathways','target-range'])
@pytest.mark.parametrize('backend',['cpu-f32',pytest.param('metal',marks=real_metal),pytest.param('cuda',marks=real_cuda)])
def test_fusion_boundaries_with_real_model_semantics(device,tmp_path,kind,backend):
    model=variant_model(tmp_path/'model',kind)
    previous=plan(model,'previous');after=plan(model)
    if kind in {'pre-read','target-range'}:
        assert len(after.dispatches)==len(previous.dispatches)-1
        assert not any(d.role=='fused-delayed-target-sparse' for d in after.dispatches)
    if kind in {'subgroup','clock','shared-pathways'}:assert after.sha256==previous.sha256
    expected=control(model,'previous',tmp_path/'control',3)
    if backend=='cpu-f32':result_exact(control(model,'full',tmp_path/'full',3),expected)
    else:
        with (MetalExecutor if backend=='metal' else CudaExecutor)(model,tmp_path/'gpu',numeric_mode='float32',event_delivery='sparse') as ex:
            for _ in range(2):result_exact(ex.run(),expected)
