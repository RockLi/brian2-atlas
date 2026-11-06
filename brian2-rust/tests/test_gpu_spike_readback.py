"""Selective transfer must use fresh counts and retain every observable result."""
import ctypes
from types import SimpleNamespace
import numpy as np
import pytest
from brian2_rust import gpu_readback


def buffers():
    arrays=[np.full(24,-99,np.int64),np.zeros(3,np.uint32),np.zeros(4,np.int64)]
    plan=SimpleNamespace(buffers=('population/0/4','population/0/5','population/0/8'))
    return plan,arrays


def test_only_known_contiguous_tick_count_pairs_are_selected():
    plan,a=buffers()
    assert gpu_readback.spike_prefix_bindings(plan,a,(0,1,2))=={0:1}
    assert gpu_readback.spike_prefix_bindings(plan,a,(0,2))=={}
    for bad in [np.zeros(3,np.int64),np.zeros((3,1),np.uint32),np.zeros(6,np.uint32)[::2]]:
        assert gpu_readback.spike_prefix_bindings(plan,[a[0],bad,a[2]],(0,1,2))=={}
    plan.buffers=('future/ticks','population/0/5','population/0/8')
    assert gpu_readback.spike_prefix_bindings(plan,a,(0,1,2))=={}


@pytest.mark.parametrize('recorded',[[0,0,0],[1,3,0],[8,0,8]])
def test_cuda_transfer_uses_fresh_device_counts_and_preserves_row_pitch(recorded):
    _,a=buffers();a[1].fill(7)
    device=[np.arange(24,dtype=np.int64),np.array(recorded,np.uint32),np.array([-1,4,5,6],np.int64)]
    calls=[]
    class Buffer:
        def __init__(self,i):self.i=i;self.data=SimpleNamespace(ptr=device[i].ctypes.data)
        def get(self,*,out,stream,blocking):
            assert blocking;calls.append(('get',self.i));np.copyto(out,device[self.i])
    def copy(dst,dpitch,src,spitch,width,height,kind):
        assert dpitch==spitch==64 and width==max(recorded)*8 and height==3 and kind==2
        calls.append(('2d',width))
        for row in range(height):ctypes.memmove(dst+row*dpitch,src+row*spitch,width)
    cp=SimpleNamespace(cuda=SimpleNamespace(runtime=SimpleNamespace(memcpy2D=copy,memcpyDeviceToHost=2)))
    copied=gpu_readback.cuda_readback(cp,object(),[Buffer(i) for i in range(3)],a,(0,1,2),{0:1})
    width=max(recorded)
    assert calls[0]==('get',1) and calls.count(('get',1))==1
    assert copied==a[1].nbytes+a[2].nbytes+width*3*8
    np.testing.assert_array_equal(a[0].reshape(3,8)[:,:width],device[0].reshape(3,8)[:,:width])
    assert np.all(a[0].reshape(3,8)[:,width:]==-99)
    np.testing.assert_array_equal(a[2],device[2])  # fault/refractory words included


def test_corrupt_device_count_fails_before_any_tick_transfer():
    _,a=buffers();calls=[]
    class Buffer:
        def get(self,*,out,**kw):calls.append('counts');out[:]=[0,9,0]
    cp=SimpleNamespace(cuda=SimpleNamespace(runtime=SimpleNamespace(memcpy2D=lambda *a:pytest.fail('unsafe copy'))))
    with pytest.raises(RuntimeError,match='capacity invariant'):
        gpu_readback.cuda_readback(cp,None,[None,Buffer(),None],a,(0,1,2),{0:1})
    assert calls==['counts']


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
def test_prefix_and_full_transfer_match_all_results_across_modes(device,tmp_path,monkeypatch,backend):
    setup(tmp_path/'ref');net,*_=network(42);model=lower_network(net,24*DT)
    expected=oracle(model,tmp_path/'oracle')
    cls=MetalExecutor if backend=='metal' else CudaExecutor
    modes=('direct','resident','indirect') if backend=='metal' else ('direct','resident','graph','chunked')
    for mode in modes:
        with cls(model,tmp_path/mode,numeric_mode='float32',event_delivery='sparse',dag_execution=mode) as ex:
            with monkeypatch.context() as patch:
                patch.setattr(gpu_readback,'spike_prefix_bindings',lambda *a:{})
                full=ex.run()
            compare(full,expected)
            for _ in range(2):
                actual=ex.run();result_exact(actual,full)
                stats=lambda r:r['metal_runtime'] if backend=='metal' else r['cuda_runtime']['dag_execution']
                assert stats(actual)['readback_bytes']<stats(full)['readback_bytes']
