"""One-activation chunk capture: exact schedules, cursor guards and conformance."""
from dataclasses import replace
from types import SimpleNamespace
import json

import numpy as np
import pytest

from brian2_rust.cuda import CudaExecutor,build_cuda_plan
from brian2_rust.cuda_chunks import build_schedule,MAX_TABLE_LAUNCHES
from brian2_rust.gpu_schedule import dispatch_ticks
from brian2_rust.plan import ClockActivation,PlanValidationError,verify_execution_plan
from brian2_rust.spec import bits
from brian2_rust.export import lower_network
from test_cuda import real_cuda
from test_cuda_graphs import result_exact
from test_metal_delays import device
from test_gpu_monitors import setup,monitored_network,DT


def test_chunk_schedule_preserves_irregular_clocks_empty_lanes_and_large_ticks():
    clocks=(ClockActivation(0,bits(1/1024),3*2**25+1,257),
            ClockActivation(1,bits(3/2048),2*2**25+1,173),
            ClockActivation(2,bits(1/4096),0,0))
    stages=tuple(SimpleNamespace(clock=c,lanes=n) for c,n in [(1,3),(0,4),(2,3),(0,0),(1,2),(0,1)])
    expected=[(s,t) for s,t in dispatch_ticks(clocks,tuple(d.clock for d in stages)) if stages[s].lanes]
    schedule=build_schedule(clocks,stages,len(expected))
    ordinals=[s for pattern in schedule.sequence for s in schedule.patterns[pattern]]
    assert list(zip(ordinals,schedule.ticks.tolist(),strict=True))==expected
    assert schedule.ticks.dtype==np.int64 and not schedule.ticks.flags.writeable
    assert schedule.capture_launches<=65*64
    assert schedule.buffer_bytes==8*len(expected)+12
    assert build_schedule(clocks,stages,len(expected)).sha256==schedule.sha256
    with pytest.raises(RuntimeError,match='count mismatch'):build_schedule(clocks,stages,len(expected)+1)


def test_chunk_limits_and_pattern_reuse(monkeypatch):
    import brian2_rust.cuda_chunks as chunks
    clocks=(ClockActivation(0,bits(.1),0,4096),)
    stages=(SimpleNamespace(clock=0,lanes=1),)
    schedule=build_schedule(clocks,stages,4096)
    assert len(schedule.patterns)==1 and len(schedule.sequence)==64 and schedule.worthwhile
    assert not build_schedule((replace(clocks[0],steps=2),),stages,2).worthwhile
    with pytest.raises(ValueError,match='nonempty'):build_schedule(clocks,stages,MAX_TABLE_LAUNCHES+1)
    monkeypatch.setattr(chunks,'MAX_PATTERNS',1)
    with pytest.raises(ValueError,match='distinct'):build_schedule((replace(clocks[0],steps=65),),stages,65)


def test_chunk_abi_and_generated_sources_are_plan_bound(device,tmp_path):
    setup(tmp_path/'ref');model=lower_network(monitored_network(True)[0],8*DT)
    plan=build_cuda_plan(model,numeric_mode='float32')
    assert plan.chunk_abi=='b2-cuda-chunk-ticks-v0'
    assert 'void b2_cuda_chunk_advance(' in plan.kernels[0].source
    for kernel in plan.kernels:
        assert 'void '+kernel.entry+'(' in kernel.source
        assert 'void '+kernel.entry+'_chunk(' in kernel.source
        assert 'b2_chunk_offset >= b2_chunk_length-b2_chunk_begin' in kernel.source
    with pytest.raises(PlanValidationError):verify_execution_plan(replace(plan,chunk_abi=None),model)


@real_cuda
@pytest.mark.parametrize('route',['scan','sparse'])
def test_chunked_first_activation_replay_modes_and_budget(device,tmp_path,route):
    setup(tmp_path/'ref');model=lower_network(monitored_network(True)[0],1024*DT)
    with CudaExecutor(model,tmp_path/'cuda',numeric_mode='float32',event_delivery=route) as ex:
        direct=ex.run(dag_execution='direct')
        first=ex.run();result_exact(first,direct)
        metadata=first['cuda_runtime']['dag_execution']
        assert metadata['selected']=='chunked' and not metadata['graph_reused']
        assert metadata['graph_build_seconds']>0
        assert metadata['host_graph_launches']*16<metadata['kernel_launches']
        cache=ex._resident_dag;chunk=cache.chunks
        for _ in range(2):
            replay=ex.run(dag_execution='chunked');result_exact(replay,direct)
            assert replay['cuda_runtime']['dag_execution']['graph_reused']
            assert int(chunk.cursor.get()[0])==len(chunk.schedule.ticks)
        result_exact(ex.run(dag_execution='resident'),direct)
        result_exact(ex.run(dag_execution='graph'),direct)
        result_exact(ex.run(dag_execution='chunked'),direct)
        budget=cache.data_bytes
        with pytest.raises(MemoryError,match='chunk schedule'):ex.run(max_buffer_bytes=budget,dag_execution='chunked')
        assert ex._resident_dag is None and chunk.graphs==[] and chunk.cursor is None
        fallback=ex.run(max_buffer_bytes=budget);result_exact(fallback,direct)
        assert fallback['cuda_runtime']['dag_execution']['selected']=='resident'
        assert fallback['cuda_runtime']['dag_execution']['chunk_fallback_reason']=='chunk-buffer-budget'
        result_exact(ex.run(dag_execution='chunked'),direct)
        chunk=ex._resident_dag.chunks
    assert chunk.graphs==[] and chunk.cursor is None


@real_cuda
def test_chunk_cursor_fault_rejects_publication_and_reset_recovers(device,tmp_path,monkeypatch):
    setup(tmp_path/'ref');model=lower_network(monitored_network(True)[0],8*DT)
    with CudaExecutor(model,tmp_path/'cuda',numeric_mode='float32',dag_execution='chunked') as ex:
        expected=ex.run();chunk=ex._resident_dag.chunks
        reset=chunk.reset;invalid=np.asarray([2**64-1],np.uint64)
        def invalid_cursor():
            reset();chunk.cursor.set(invalid,stream=ex.stream)
        monkeypatch.setattr(chunk,'reset',invalid_cursor)
        with pytest.raises(RuntimeError,match='cursor invariant'):ex.run()
        assert ex._resident_dag is None
        result_exact(ex.run(),expected)


@real_cuda
def test_chunk_capture_failure_clears_partial_graphs(device,tmp_path):
    setup(tmp_path/'ref');model=lower_network(monitored_network(True)[0],8*DT)
    with CudaExecutor(model,tmp_path/'cuda',numeric_mode='float32',dag_execution='chunked') as ex:
        saved=ex.chunk_advance
        def fail(*args,**kwargs):raise ValueError('injected chunk failure')
        ex.chunk_advance=fail
        with pytest.raises(ValueError,match='injected chunk'):ex.run()
        assert ex._resident_dag is None
        ex.chunk_advance=saved
        result_exact(ex.run(),ex.run(dag_execution='direct'))


@pytest.fixture
def forced_chunks(monkeypatch):
    original=CudaExecutor.run
    def run(executor,*args,**kwargs):
        kwargs.setdefault('dag_execution','chunked')
        result=original(executor,*args,**kwargs)
        if executor.plan.dispatches and kwargs.get('compute','cuda')=='cuda':
            assert result['cuda_runtime']['dag_execution']['selected']=='chunked'
        return result
    monkeypatch.setattr(CudaExecutor,'run',run)


@real_cuda
@pytest.mark.parametrize('case',['large-pending','typed','linked','random','timed-input','early-event','restore','tick-fault'])
def test_chunked_existing_semantic_contracts(device,tmp_path,forced_chunks,case):
    common=dict(device=device,tmp_path=tmp_path,backend='cuda')
    if case=='large-pending':
        from test_gpu_multiclock import test_generator_coupled_large_absolute_ticks_and_pending as check
        check(**common)
    elif case=='typed':
        from test_gpu_typed_storage import test_typed_delayed_pre_post_plasticity_and_subgroups as check
        check(**common,route='sparse')
    elif case=='linked':
        from test_gpu_links import test_linked_source_with_sparse_delay_fusion_and_feedback as check
        check(**common,route='sparse')
    elif case=='random':
        from test_gpu_random import test_synaptic_counter_uses_edge_and_delivery_tick as check
        check(**common)
    elif case=='timed-input':
        from test_gpu_timed_array import test_delayed_synaptic_time_and_subgroup_columns as check
        check(**common,route='scan')
    elif case=='early-event':
        from test_gpu_pathway_order import test_early_delayed_pathways_match_reference as check
        check(**common,route='sparse',mutable=True,event='pulse')
    elif case=='restore':
        from test_gpu_pathway_order import test_early_pending_continuation_restore_and_queued as check
        check(**common,queued=False,event='pulse')
    else:
        from test_gpu_ticks import test_synaptic_scalar_timestep_fault_even_without_events as check
        check(**common,empty=True)



def test_async_cursor_reset_keeps_host_sources_alive_until_close():
    # CuPy set() hands a raw host pointer to an asynchronous transfer. Model a
    # stream which has not consumed that pointer when reset() returns.
    import weakref
    from brian2_rust.cuda_chunks import ChunkGraphs
    pending=[]
    class DeviceArray:
        def set(self,host,stream):pending.append(weakref.ref(host))
    cp=SimpleNamespace(asarray=lambda host:DeviceArray())
    chunks=ChunkGraphs(SimpleNamespace(ticks=np.zeros(1,np.int64)),cp,object())
    chunks.reset()
    assert len(pending)==2 and all(source() is not None for source in pending)
    for source in pending:np.testing.assert_array_equal(source(),0)
    chunks.close()
    assert all(source() is None for source in pending)
