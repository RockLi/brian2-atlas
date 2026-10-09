"""Uninitialized recording tails must not enter observable results or snapshots."""
from types import SimpleNamespace
import numpy as np
import pytest
from brian2_rust import gpu_readback


@pytest.mark.parametrize('compute',['metal','cuda','cpu-f32','unknown'])
@pytest.mark.parametrize('count',[0,1])
def test_only_fresh_native_recording_storage_skips_copy(compute,count):
    plan=SimpleNamespace(buffers=('population/0/4','population/0/5','population/0/8','constant'))
    initial=[np.full(24,-99,np.int64),np.full(3,count,np.uint32),np.arange(4,dtype=np.int64),np.ones(2)]
    writable={0,1,2}
    arrays,stats=gpu_readback.fresh_dag_arrays(plan,initial,writable,compute)
    saved=initial[0].nbytes if compute in {'metal','cuda'} and count==0 else 0
    assert stats==dict(allocated_bytes=236,copied_bytes=236-saved,omitted_spike_copy_bytes=saved,reused_spike_bytes=0,retained_spike_bytes=0)
    assert arrays[3] is initial[3]
    for i in writable:
        assert not np.shares_memory(arrays[i],initial[i])
        if i!=0 or not saved:np.testing.assert_array_equal(arrays[i],initial[i])
        arrays[i].fill(7)
    assert np.all(initial[0]==-99) and np.all(initial[1]==count)
    np.testing.assert_array_equal(initial[2],np.arange(4))


@pytest.mark.parametrize('change',['unknown','missing-count','readonly-count','wrong-dtype','strided'])
def test_unrecognized_or_unreset_record_layout_keeps_copy(change):
    names=['population/0/4','population/0/5'];arrays=[np.arange(24,dtype=np.int64),np.zeros(3,np.uint32)];writable={0,1}
    if change=='unknown':names[0]='future/record'
    if change=='missing-count':names[1]='other/count'
    if change=='readonly-count':writable={0}
    if change=='wrong-dtype':arrays[0]=arrays[0].astype(np.float64)
    if change=='strided':arrays[0]=arrays[0][::2]
    plan=SimpleNamespace(buffers=names)
    fresh,stats=gpu_readback.fresh_dag_arrays(plan,arrays,writable,'metal')
    assert stats['omitted_spike_copy_bytes']==0
    np.testing.assert_array_equal(fresh[0],arrays[0])


from brian2_rust.cuda import CudaExecutor
from brian2_rust.metal import MetalExecutor
from brian2_rust.export import lower_network
from test_metal_delays import device
from test_gpu_spike_generator import BACKENDS
from test_gpu_composed_models import setup,network,DT
from test_gpu_expression_contract import oracle
from test_gpu_custom_events import compare
from test_cuda_graphs import result_exact


@pytest.mark.parametrize('backend',BACKENDS[1:])
def test_dirty_host_records_match_copied_snapshot_across_native_modes(device,tmp_path,monkeypatch,backend):
    setup(tmp_path/'ref');net,*_=network(42);model=lower_network(net,24*DT)
    expected=oracle(model,tmp_path/'oracle')
    cls=MetalExecutor if backend=='metal' else CudaExecutor
    modes=('direct','resident','indirect') if backend=='metal' else ('direct','resident','graph','chunked')
    original=gpu_readback.fresh_dag_arrays
    def poisoned(*args,**kwargs):
        arrays,stats=original(*args,**kwargs)
        plan,initial,writable,compute=args
        for i in gpu_readback.spike_host_copy_omissions(plan,initial,writable,compute):arrays[i].fill(-987654321)
        return arrays,stats
    for mode in modes:
        with cls(model,tmp_path/mode,numeric_mode='float32',event_delivery='sparse',dag_execution=mode) as ex:
            with monkeypatch.context() as patch:
                patch.setattr(gpu_readback,'spike_host_copy_omissions',lambda *a:frozenset())
                full=ex.run()
            compare(full,expected)
            initial=[a.copy() for a in ex._dag_initial_storage[0]]
            for full_transfer in [False,True]:
                with monkeypatch.context() as patch:
                    patch.setattr(gpu_readback,'fresh_dag_arrays',poisoned)
                    if full_transfer:
                        patch.setattr(gpu_readback,'spike_prefix_bindings',lambda *a:{})
                    actual=ex.run()
                result_exact(actual,full)
                runtime=actual['metal_runtime'] if backend=='metal' else actual['cuda_runtime']['dag_execution']
                assert (runtime['omitted_spike_upload_bytes']==0)==full_transfer
                stats=actual['host_storage'];base=full['host_storage']
                assert stats['omitted_spike_copy_bytes']>0 and base['omitted_spike_copy_bytes']==0
                assert stats['allocated_bytes']+stats['reused_spike_bytes']==base['allocated_bytes']+base['reused_spike_bytes']
                assert stats['copied_bytes']==base['copied_bytes']-stats['omitted_spike_copy_bytes']
                for a,b in zip(initial,ex._dag_initial_storage[0],strict=True):np.testing.assert_array_equal(a,b)
