"""Append-only recording storage may start dirty; every observable result must agree."""
import ctypes
from types import SimpleNamespace
import numpy as np
import pytest
from brian2_rust import gpu_readback


def buffers():
    arrays=[np.full(24,-99,np.int64),np.zeros(3,np.uint32),np.zeros(4,np.int64)]
    plan=SimpleNamespace(buffers=('population/0/4','population/0/5','population/0/8'))
    return plan,arrays


def test_only_zero_count_prefix_outputs_can_omit_upload():
    plan,arrays=buffers()
    prefixes=gpu_readback.spike_prefix_bindings(plan,arrays,(0,1,2))
    assert gpu_readback.spike_upload_omissions(arrays,prefixes)=={0}
    arrays[1][1]=1
    assert not gpu_readback.spike_upload_omissions(arrays,prefixes)
    arrays[1].fill(0)
    assert not gpu_readback.spike_upload_omissions(arrays,{})


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
def test_skipped_tick_upload_matches_full_upload_across_modes(device,tmp_path,monkeypatch,backend):
    setup(tmp_path/'ref');net,*_=network(42);model=lower_network(net,24*DT)
    expected=oracle(model,tmp_path/'oracle')
    cls=MetalExecutor if backend=='metal' else CudaExecutor
    modes=('direct','resident','indirect') if backend=='metal' else ('direct','resident','graph','chunked')
    for mode in modes:
        with cls(model,tmp_path/mode,numeric_mode='float32',event_delivery='sparse',dag_execution=mode) as ex:
            with monkeypatch.context() as patch:
                patch.setattr(gpu_readback,'spike_upload_omissions',lambda *a:frozenset())
                full=ex.run()
            compare(full,expected)
            for _ in range(2):
                if backend=='cuda' and ex._resident_dag is not None:
                    # The optimized path must tolerate dirty record slots from
                    # prior use; counters and all other writable inputs reset.
                    with ex.device,ex.stream:
                        for i,name in enumerate(ex.plan.buffers):
                            if name.endswith('/4') and name.startswith('population/'):
                                ex._resident_dag.gpu[i].fill(-987654321)
                actual=ex.run();result_exact(actual,full)
                stats=lambda r:r['metal_runtime'] if backend=='metal' else r['cuda_runtime']['dag_execution']
                a,f=stats(actual),stats(full)
                field='uploaded_bytes' if backend=='metal' else 'upload_bytes'
                assert a['omitted_spike_upload_bytes']>0 and f['omitted_spike_upload_bytes']==0
                assert a[field]<f[field] and a['readback_bytes']==f['readback_bytes']
