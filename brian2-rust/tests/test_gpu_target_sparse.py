"""Ordered sparse plasticity: queue layout, pending chronology and native replay."""
from copy import deepcopy
from types import SimpleNamespace
import json
import numpy as np
import pytest
from brian2_rust.metal import MetalExecutor,build_metal_plan
from brian2_rust.cuda import CudaExecutor,build_cuda_plan
from brian2_rust.metal_dag import run_dag,_prepare_dag_storage
from brian2_rust.protocol import attach_protocol
from brian2_rust import gpu_target_sparse as sparse
from test_metal_delays import device
from test_gpu_prefix_pending import pending_model
from test_gpu_synapse_prefix import model_at,has_prefix
from test_gpu_synapse_parallel import control
from test_gpu_spike_generator import BACKENDS
from test_cuda_graphs import result_exact
from test_gpu_workgroup import save


@pytest.mark.parametrize('backend',BACKENDS)
@pytest.mark.parametrize('queued',[False,True])
def test_event_monitor_buffers_coexist_with_target_queues(device,tmp_path,backend,queued):
    from test_gpu_monitors import setup,monitored_network,DT,event_equivalent
    from brian2_rust.results import load_results
    setup(tmp_path/'reference')
    network,_,_=monitored_network(True)
    network.run(8*DT)
    model=json.loads((device.last_run_directory/'model.json').read_text())
    expected=load_results(model,device.last_run_directory/'rust')
    if backend=='cpu-f32':
        directory=tmp_path/'control';directory.mkdir()
        plan=build_metal_plan(model,numeric_mode='float32',synapse_sparse=queued)
        ex=SimpleNamespace(model=model,plan=plan,directory=directory,compile_seconds=0,device_name='CPU f32')
        actual=run_dag(ex,max_buffer_bytes=512*1024**2,compute='cpu-f32',workers=3)
    else:
        cls=MetalExecutor if backend=='metal' else CudaExecutor
        with cls(model,tmp_path/backend,numeric_mode='float32',synapse_sparse=queued) as ex:
            plan=ex.plan;actual=ex.run()
            event_equivalent(ex.run(),actual)
    assert any(d.role=='event-monitor' for d in plan.dispatches)
    assert any(d.role==sparse.ROLE for d in plan.dispatches)==queued
    event_equivalent(actual,expected)
    save(tmp_path/'monitor-queue-results.npz',actual,expected)
    (tmp_path/'monitor-queue-plan.json').write_text(plan.to_json()+'\n')


def cpu(model,directory,**options):
    directory.mkdir()
    ex=SimpleNamespace(model=model,plan=build_metal_plan(model,numeric_mode='float32',event_delivery='sparse',synapse_sparse=True,**options),directory=directory,compile_seconds=0,device_name='CPU f32')
    return run_dag(ex,max_buffer_bytes=512*1024**2,compute='cpu-f32',workers=3)


def fixture(path,case):
    if case in {'uniform','heterogeneous','duplicate'}:
        m=pending_model(path,uniform=case=='uniform')
        if case=='duplicate':
            route=m['instance']['synapses'][0]['pathways'][0]
            route['pending']=[dict(delivery_tick=0,item=0),dict(delivery_tick=0,item=0),*route['pending']]
    else:m=model_at(path,early=case=='early',empty=case=='empty',temporary_tail=True)
    if case=='low':
        values=m['instance']['populations'][0]['initial_state']['v']
        m['instance']['populations'][0]['initial_state']['v']=[v if i<4 else '0000000000000000' for i,v in enumerate(values)]
    attach_protocol(m);return m


def test_queue_groups_preserve_order_capacity_and_original_storage(device,tmp_path):
    m=fixture(tmp_path/'ref','heterogeneous')
    a=build_metal_plan(m,numeric_mode='float32',event_delivery='sparse')
    b=build_metal_plan(m,numeric_mode='float32',event_delivery='sparse',synapse_sparse=True)
    assert a.logical==b.logical and b.buffers[:len(a.buffers)]==a.buffers
    assert b.buffers[len(a.buffers):]==sparse.buffer_names(0,0)
    before,after=(_prepare_dag_storage(SimpleNamespace(model=m,plan=p),512*1024**2)[0] for p in (a,b))
    for x,y in zip(before,after[:len(before)],strict=True):np.testing.assert_array_equal(x,y)
    source_groups,delays,offsets,ranks,targets,active,counts=after[len(before):]
    inst=m['instance']['synapses'][0];path=inst['pathways'][0];edges=len(inst['source'])
    base=a.buffers.index('synapse/0/pathway/0/delays');ordered=before[base+1]
    seen=[]
    for owner in range(len(source_groups)-1):
        for group in range(source_groups[owner],source_groups[owner+1]):
            for rank in ranks[offsets[group]:offsets[group+1]]:
                edge=ordered[rank];seen.append(int(edge))
                assert inst['source'][edge]==owner and path['delay_ticks'][edge]==delays[group]
                assert inst['target'][edge]==targets[rank]
    assert sorted(seen)==list(range(edges))
    assert active.size==edges and counts.size==m['definition']['synapses'][0]['target_count']
    assert not np.any(active) and not np.any(counts)
    consumer=next(d for d in b.dispatches if d.role==sparse.ROLE)
    assert consumer.types[-2:]==('uint','atomic_uint') and consumer.bindings[-2:]==(len(b.buffers)-2,len(b.buffers)-1)
    with pytest.raises(MemoryError,match='target sparse queue'):sparse.arrays(m,b,before,0,0,active.nbytes-1)
    (tmp_path/'target-sparse-model.json').write_text(json.dumps(m)+'\n')
    (tmp_path/'target-sparse-plan.json').write_text(b.to_json())


@pytest.mark.parametrize('backend',BACKENDS)
@pytest.mark.parametrize('case',['uniform','heterogeneous','duplicate','early','empty','low'])
def test_sparse_pending_current_and_empty_native_replay(device,tmp_path,backend,case):
    m=fixture(tmp_path/'ref',case);expected=control(m,tmp_path/'baseline',3)
    prefix=case in {'uniform','heterogeneous','duplicate'}
    if backend=='cpu-f32':actual=cpu(m,tmp_path/'cpu',synapse_prefix=prefix)
    else:
        cls=MetalExecutor if backend=='metal' else CudaExecutor
        with cls(m,tmp_path/'native',numeric_mode='float32',synapse_sparse=True,synapse_prefix=prefix) as ex:
            assert any(d.role==sparse.ROLE for d in ex.plan.dispatches)
            if case=='duplicate':assert not has_prefix(ex.plan)
            for _ in range(2):actual=ex.run();result_exact(actual,expected)
            result_exact(ex.run(dag_execution='workgroup'),expected)
    result_exact(actual,expected)
    assert [s['events'] for s in actual['synapses']]==[s['events'] for s in expected['synapses']]
    save(tmp_path/'target-sparse-results.npz',actual,expected)


@pytest.mark.parametrize('backend',BACKENDS)
def test_empty_current_queue_keeps_scalar_faults(device,tmp_path,backend):
    from test_gpu_refractory import refresh_code
    from brian2_rust.spec import bits
    m=fixture(tmp_path/'ref','empty');code=next(c for c in m['definition']['synapses'][0]['code_objects'] if c['kind']=='synapses')
    value=dict(op='floor_div',left=dict(op='literal',bits=bits(1.)),right=dict(op='literal',bits=bits(0.)))
    code['scalar'].insert(0,dict(target='_checked',dtype='f64',dimensions=[0.]*7,condition=None,value=value));refresh_code(m,code)
    with pytest.raises(FloatingPointError):control(m,tmp_path/'baseline',1)
    if backend=='cpu-f32':
        with pytest.raises(FloatingPointError):cpu(m,tmp_path/'cpu')
    else:
        cls=MetalExecutor if backend=='metal' else CudaExecutor
        with cls(m,tmp_path/'native',numeric_mode='float32',synapse_sparse=True) as ex:
            for _ in range(2):
                with pytest.raises(FloatingPointError):ex.run()


def test_unsafe_target_aliases_keep_canonical_plan_and_options_validate(device,tmp_path):
    m=model_at(tmp_path/'ref',recurrent=True,hazard=True)
    for build in (build_metal_plan,build_cuda_plan):
        assert build(m,numeric_mode='float32').to_json()==build(m,numeric_mode='float32',synapse_sparse=True).to_json()
        with pytest.raises(ValueError,match='synapse_sparse'):build(m,numeric_mode='float32',synapse_sparse='yes')


@pytest.mark.parametrize('backend',BACKENDS[1:])
@pytest.mark.parametrize('queued',[False,True])
def test_sparse_device_pending_restore_and_queued_runs(device,tmp_path,backend,queued):
    import brian2 as b
    from test_gpu_target_pathway import network
    from test_gpu_monitors import DT
    from test_metal_delays import ROOT
    all_records=[]
    for enabled in (False,True):
        device.reinit()
        b.set_device('rust_standalone',engine=backend,numeric_mode='float32',event_delivery='sparse',
            directory=tmp_path/str(enabled),runner=ROOT/'target/release/b2-runner',
            build_on_run=not queued,gpu_synapse_sparse=enabled,gpu_buffer_reuse=True,gpu_compile_reuse=True)
        b.seed(1729)
        net,syn,post=network(edge_count=16384)
        net.run(4*DT)
        if not queued:
            assert any(d.role==sparse.ROLE for d in device.last_execution_plan.dispatches)==enabled
            net.store('pending')
        net.run(4*DT)
        if not queued:
            assert any(d.role==sparse.ROLE for d in device.last_execution_plan.dispatches)==enabled
            saved=np.asarray(syn.w[:]).copy()
            net.restore('pending',restore_random_state=True);net.run(4*DT)
            np.testing.assert_array_equal(syn.w[:],saved)
        net.run(4*DT)
        if queued:device.build()
        record={name:np.asarray(getattr(syn,name)[:]).copy() for name in ('w','Apre','Apost','hits')}
        for name in ('v','x'):record['post/'+name]=np.asarray(getattr(post,name)[:]).copy()
        for obj in net.objects:
            if isinstance(obj,b.SpikeMonitor):
                for name in ('t','i','count'):record['spike/'+name]=np.asarray(getattr(obj,name)[:]).copy()
            elif isinstance(obj,b.StateMonitor):
                for name in ('v','x','t'):record['monitor/'+name]=np.asarray(getattr(obj,name)[:]).copy()
        all_records.append(record);device.close_gpu()
    assert all_records[0].keys()==all_records[1].keys()
    for k,v in all_records[0].items():np.testing.assert_array_equal(v,all_records[1][k],err_msg=k)
    np.savez_compressed(tmp_path/'target-sparse-device.npz',**{label+'/'+k:v for label,r in zip(('reference','actual'),all_records) for k,v in r.items()})
