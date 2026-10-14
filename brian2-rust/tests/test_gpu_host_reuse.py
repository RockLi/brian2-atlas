"""Private recording reuse must not alias public results, inputs or other executors."""
from types import SimpleNamespace
import numpy as np
import pytest
from brian2_rust import gpu_readback


def fixture():
    return SimpleNamespace(buffers=('population/0/4','population/0/5','state')), [np.zeros(24,np.int64),np.zeros(3,np.uint32),np.arange(4,dtype=np.float32)]


def test_private_cache_reuses_only_fresh_record_slots_and_discards_changed_layout():
    plan,initial=fixture();cache={};write={0,1,2}
    first,a=gpu_readback.fresh_dag_arrays(plan,initial,write,'cuda',spike_cache=cache)
    first[0].fill(-991);first[1].fill(8);first[2].fill(42)
    second,b=gpu_readback.fresh_dag_arrays(plan,initial,write,'cuda',spike_cache=cache)
    assert second[0] is first[0] and np.all(second[0]==-991)
    assert set(cache)=={0} and a['reused_spike_bytes']==0
    assert b['reused_spike_bytes']==b['retained_spike_bytes']==initial[0].nbytes
    assert a['allocated_bytes']-b['allocated_bytes']==initial[0].nbytes
    assert a['copied_bytes']==b['copied_bytes']==initial[1].nbytes+initial[2].nbytes
    for i in [1,2]:
        assert second[i] is not first[i] and not np.shares_memory(second[i],initial[i])
        np.testing.assert_array_equal(second[i],initial[i])
    changed=[np.zeros(30,np.int64),*initial[1:]]
    third,c=gpu_readback.fresh_dag_arrays(plan,changed,write,'cuda',spike_cache=cache)
    assert third[0] is not first[0] and c['reused_spike_bytes']==0
    initial[1][0]=1
    fourth,d=gpu_readback.fresh_dag_arrays(plan,initial,write,'cuda',spike_cache=cache)
    assert cache=={} and d['reused_spike_bytes']==d['retained_spike_bytes']==0
    np.testing.assert_array_equal(fourth[0],initial[0])


@pytest.mark.parametrize('bad',['initial-alias','readonly','strided','wrong-dtype'])
def test_invalid_cached_layout_cannot_alias_snapshot(bad):
    plan,initial=fixture();a=initial[0]
    cached={'initial-alias':a,'readonly':a.copy(),'strided':np.zeros(48,np.int64)[::2],
            'wrong-dtype':a.astype(np.float64)}[bad]
    if bad=='readonly':cached.setflags(write=False)
    cache={0:cached};arrays,stats=gpu_readback.fresh_dag_arrays(plan,initial,{0,1,2},'metal',spike_cache=cache)
    assert arrays[0] is not cached and stats['reused_spike_bytes']==0
    assert arrays[0].flags.writeable and not np.shares_memory(arrays[0],a)


def test_cache_is_executor_local_and_cpu_path_does_not_use_it():
    x,y=SimpleNamespace(),SimpleNamespace()
    assert gpu_readback.host_spike_cache(x,'metal') is None
    a=gpu_readback.host_spike_cache(x,'metal',enabled=True);b=gpu_readback.host_spike_cache(y,'cuda')
    assert a is not b and gpu_readback.host_spike_cache(x,'metal',enabled=True) is a
    assert gpu_readback.host_spike_cache(x,'cpu-f32') is None
    a[0]=np.zeros(8);gpu_readback.release_host_spike_cache(x)
    assert a=={} and b=={}
    gpu_readback.release_host_spike_cache(SimpleNamespace())


from brian2_rust.cuda import CudaExecutor
from brian2_rust.metal import MetalExecutor
from brian2_rust.export import lower_network
from test_metal_delays import device
from test_gpu_spike_generator import BACKENDS
from test_gpu_composed_models import setup,network,DT
from test_gpu_expression_contract import oracle
from test_gpu_custom_events import compare
from test_cuda_graphs import result_exact


def result_arrays(value,path=()):
    if isinstance(value,np.ndarray):yield path,value
    elif isinstance(value,dict):
        for k,v in value.items():yield from result_arrays(v,path+(k,))
    elif isinstance(value,list):
        for i,v in enumerate(value):yield from result_arrays(v,path+(i,))


@pytest.mark.parametrize('backend',BACKENDS[1:])
def test_retained_public_results_and_reset_survive_dirty_cache_and_failed_replay(device,tmp_path,monkeypatch,backend):
    setup(tmp_path/'ref');net,*_=network(42);model=lower_network(net,24*DT)
    reference=oracle(model,tmp_path/'oracle')
    cls=MetalExecutor if backend=='metal' else CudaExecutor
    modes=('direct','resident','indirect') if backend=='metal' else ('direct','resident','graph','chunked')
    original_cache=gpu_readback.host_spike_cache
    monkeypatch.setattr(gpu_readback,'host_spike_cache',lambda ex,compute:original_cache(ex,compute,enabled=True))
    with cls(model,tmp_path/backend,numeric_mode='float32',event_delivery='sparse') as ex:
        with monkeypatch.context() as patch:
            patch.setattr(gpu_readback,'host_spike_cache',lambda *a:None)
            control=ex.run()
        compare(control,reference)
        first=ex.run();result_exact(first,control)
        retained={p:a.copy() for p,a in result_arrays(first)}
        cache=ex._dag_host_spike_cache;identities={i:id(a) for i,a in cache.items()}
        assert cache and first['host_storage']['reused_spike_bytes']==0
        for _,a in result_arrays(first):assert all(not np.shares_memory(a,b) for b in cache.values())
        initial=[a.copy() for a in ex._dag_initial_storage[0]]
        # Interrupted execution may dirty the cache. The next replay must reset
        # counters/state and republish only newly written recording prefixes.
        native=ex._execute_dag
        def fail(arrays,**kwargs):
            for i in cache:arrays[i].fill(-7654321)
            raise RuntimeError('injected replay failure')
        with monkeypatch.context() as patch:
            patch.setattr(ex,'_execute_dag',fail)
            with pytest.raises(RuntimeError,match='injected replay failure'):ex.run()
        for mode in modes:
            for full_transfer in [False,True]:
                for a in cache.values():a.fill(-7654321)
                with monkeypatch.context() as patch:
                    if full_transfer:patch.setattr(gpu_readback,'spike_prefix_bindings',lambda *a:{})
                    actual=ex.run(dag_execution=mode)
                result_exact(actual,control)
                assert actual['host_storage']['reused_spike_bytes']==sum(a.nbytes for a in cache.values())>0
                assert {i:id(a) for i,a in cache.items()}==identities
                for p,a in result_arrays(first):np.testing.assert_array_equal(a,retained[p])
                for a,b in zip(initial,ex._dag_initial_storage[0],strict=True):np.testing.assert_array_equal(a,b)
        # Mutating already returned arrays must not modify future runs either.
        for _,a in result_arrays(first):a.fill(0)
        result_exact(ex.run(),control)
    assert cache=={}


def test_ablation_policy_is_restored_on_failure(monkeypatch):
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'examples'))
    from gpu_host_reuse_compare import host_reuse_mode
    original=gpu_readback.host_spike_cache
    with pytest.raises(RuntimeError):
        with host_reuse_mode('fresh'):
            assert gpu_readback.host_spike_cache(SimpleNamespace(),'cuda') is None
            raise RuntimeError('failure')
    assert gpu_readback.host_spike_cache is original
    with pytest.raises(ValueError):
        with host_reuse_mode('unknown'):pytest.fail('accepted unknown mode')
