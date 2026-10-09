"""Capacity-independent spike decoding must preserve every valid recorded event."""
import numpy as np
import pytest
from brian2_rust.metal_event_layout import spike_coordinates


@pytest.mark.parametrize('n,capacity',[(0,0),(0,9),(1,0),(1,1),(7,5),(13,64)])
@pytest.mark.parametrize('density',[0.,.02,.249,.25,.9,1.])
def test_prefix_decode_matches_independent_coordinate_list(n,capacity,density):
    rng=np.random.default_rng(1729)
    counts=np.minimum(rng.binomial(capacity,density,n),capacity).astype(np.uint32)
    ticks=rng.integers(-5,50,max(n*capacity,1),dtype=np.int64)
    expected=sorted((int(ticks[i*capacity+j]),i) for i in range(n) for j in range(int(counts[i])))
    ticks.setflags(write=False);counts.setflags(write=False)
    t,i=spike_coordinates(ticks,counts,capacity)
    assert t.dtype==i.dtype==np.int64
    assert list(zip(t.tolist(),i.tolist()))==expected
    assert not np.shares_memory(t,ticks) and not np.shares_memory(i,counts)


def test_sparse_decode_does_not_allocate_capacity_mask(monkeypatch):
    n,capacity=7,4096
    counts=np.array([0,2,0,1,0,0,2],np.uint32)
    ticks=np.full(n*capacity,-999,np.int64)
    ticks[capacity:capacity+2]=[8,4];ticks[3*capacity]=8;ticks[6*capacity:6*capacity+2]=[4,4]
    original=np.arange
    def checked(*args,**kwargs):
        assert args[0]!=capacity,'constructed capacity-sized mask range'
        return original(*args,**kwargs)
    monkeypatch.setattr(np,'arange',checked)
    t,i=spike_coordinates(ticks,counts,capacity)
    assert t.tolist()==[4,4,4,8,8] and i.tolist()==[1,6,6,1,3]


@pytest.mark.parametrize('counts',[np.array([-1,0]),np.array([4,0]),np.array([2**32-1,0],np.uint32)])
def test_invalid_counts_never_read_recording_storage(counts):
    with pytest.raises(RuntimeError,match='capacity invariant'):
        spike_coordinates(np.zeros(6,np.int64),counts,3)


def test_malformed_layout_rejected():
    with pytest.raises(ValueError):spike_coordinates(np.zeros(4,np.int64),np.zeros((2,2),np.uint32),1)
    with pytest.raises(ValueError):spike_coordinates(np.zeros(4,np.int64),np.zeros(2,np.float32),2)
    with pytest.raises(ValueError):spike_coordinates(np.zeros(3,np.int64),np.zeros(2,np.uint32),2)
    with pytest.raises(ValueError):spike_coordinates(np.zeros(4,np.int64),np.zeros(2,np.uint32),-2)


def test_ablation_restores_decoder_after_failure(monkeypatch):
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'examples'))
    from gpu_spike_decode_compare import decode_mode
    from brian2_rust import metal_event_layout as module
    original=module.spike_coordinates
    with pytest.raises(RuntimeError):
        with decode_mode('capacity-mask'):
            module.spike_coordinates(np.zeros(2,np.int64),np.array([3],np.uint32),2)
    assert module.spike_coordinates is original
    with pytest.raises(ValueError):
        with decode_mode('unknown'):pytest.fail('accepted unknown mode')
    assert module.spike_coordinates is original


def test_ablation_creates_control_and_gpu_directories(monkeypatch,tmp_path):
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'examples'))
    import gpu_spike_decode_compare as experiment
    import gpu_stdp_precompiled as worker
    import gpu_stdp_compare as model
    monkeypatch.setattr(model,'configuration',lambda *a,**k:{'test':'directory setup'})
    monkeypatch.setattr(model,'oracle',lambda *a,**k:{'v':np.zeros(2,np.float32)})
    events=[]
    class Replay:
        def __init__(self,backend,n,steps,degree,output,**options):
            assert output.is_dir()
            events.append(backend)
            if backend=='metal':raise RuntimeError('GPU setup reached')
            self.bootstrap={'v':np.zeros(2,np.float32)}
        def close(self):events.append('control closed')
    monkeypatch.setattr(worker,'Replay',Replay)
    with pytest.raises(RuntimeError,match='GPU setup reached'):
        experiment.compare('metal',tmp_path/'trial')
    assert events==['cpu-f32','control closed','metal']
