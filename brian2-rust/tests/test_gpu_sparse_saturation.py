"""Saturation may discard queue entries only after dense fallback is inevitable."""
from types import SimpleNamespace
import json
import numpy as np
import pytest
from brian2_rust.metal import MetalExecutor,build_metal_plan
from brian2_rust.cuda import CudaExecutor
from brian2_rust.metal_dag import run_dag
from brian2_rust.export import lower_network
from brian2_rust import gpu_readback
from gpu_sparse_saturation_compare import original_queue,bounded_queue
from test_metal_delays import device,ROOT
from test_gpu_spike_generator import BACKENDS
from test_gpu_synapse_parallel import control
from test_cuda_graphs import result_exact
from test_gpu_workgroup import save


def test_all_small_stale_read_reservation_interleavings_preserve_fallback():
    overshoots=0
    for degree in range(1,14):
        limit=(degree+3)//4
        for events in range(degree+1):
            # unread events, accepted stale reads awaiting fetch_add, count,
            # skipped events. Writes use unique fetch_add return values.
            stack=[(events,0,0,0)];seen=set()
            while stack:
                state=stack.pop()
                if state in seen:continue
                seen.add(state);unread,ready,count,skipped=state
                assert count+ready+unread+skipped==events
                if not unread and not ready:
                    assert (4*count<degree)==(4*events<degree)
                    if 4*count<degree:assert count==events
                    assert min(count,limit)<=limit
                    overshoots+=count>limit
                if unread:
                    stack.append((unread-1,ready+(count<limit),count,skipped+(count>=limit)))
                if ready:stack.append((unread,ready-1,count+1,skipped))
    assert overshoots>0
    # The generated division/remainder expression avoids uint32 degree+3 wrap.
    for degree in (0,1,3,4,5,2**32-4,2**32-3,2**32-2,2**32-1):
        assert degree//4+int(degree%4!=0)==(degree+3)//4


def boundary_model(path):
    import brian2 as b
    dt=b.second/1024
    degrees=np.asarray([0,1,2,3,4,5,7,8,9,13,16,17,32,33,257],np.int64)
    b.set_device('rust_standalone',engine='reference',directory=path,runner=ROOT/'target/release/b2-runner')
    drive=np.zeros((4,257));drive[0]=1;drive[1,:1]=1;drive[3,:8]=1
    table=b.TimedArray(drive,dt=dt)
    pre=b.NeuronGroup(257,'v:1',threshold='v>0.5',reset='',dt=dt,name='pre',namespace={'drive':table})
    pre.run_regularly('v=drive(t,i)')
    post=b.NeuronGroup(len(degrees),'v:1',threshold='v>1',reset='v=0',dt=dt,name='post')
    syn=b.Synapses(pre,post,'w:1\nhits:integer',on_pre='v_post=v_post*0.5+w; w+=0.125; hits+=1',clock=pre.clock,name='plastic')
    syn.connect(i=np.concatenate([np.arange(d) for d in degrees]),j=np.repeat(np.arange(len(degrees)),degrees))
    syn.w=np.arange(len(syn))%13/64
    # Zero delay makes each tick independently switch dense / sparse / quiet.
    spikes=b.SpikeMonitor(post);trace=b.StateMonitor(post,'v',record=True)
    return lower_network(b.Network(pre,post,syn,spikes,trace),4*dt),degrees


@pytest.mark.parametrize('backend',BACKENDS)
def test_native_saturation_never_writes_beyond_sparse_prefix(device,tmp_path,monkeypatch,backend):
    model,degrees=boundary_model(tmp_path/'ref');expected=control(model,tmp_path/'control',3)
    sentinel=np.uint32(0xffffffff);captures={};results={};plans={}
    fresh=gpu_readback.fresh_dag_arrays;readback=gpu_readback.readback_bindings
    for mode in ('original','bounded'):
        captured=[]
        def poisoned(plan,*args,**kwargs):
            arrays,stats=fresh(plan,*args,**kwargs)
            index=plan.buffers.index('synapse/0/pathway/0/target_sparse_active_ranks')
            arrays[index].fill(sentinel);captured.append(arrays[index])
            return arrays,stats
        def include_queue(plan):
            index=plan.buffers.index('synapse/0/pathway/0/target_sparse_active_ranks')
            return tuple(sorted(set(readback(plan))|{index}))
        with original_queue() if mode=='original' else bounded_queue():
            if backend=='cpu-f32':
                path=tmp_path/mode;path.mkdir()
                ex=SimpleNamespace(model=model,plan=build_metal_plan(model,numeric_mode='float32',synapse_sparse=True),directory=path,compile_seconds=0,device_name='CPU f32')
            else:
                cls=MetalExecutor if backend=='metal' else CudaExecutor
                ex=cls(model,tmp_path/mode,numeric_mode='float32',synapse_sparse=True)
        plans[mode]=ex.plan.to_dict()
        try:
            with monkeypatch.context() as patch:
                patch.setattr(gpu_readback,'fresh_dag_arrays',poisoned)
                patch.setattr(gpu_readback,'readback_bindings',include_queue)
                actual=run_dag(ex,max_buffer_bytes=512*1024**2,compute='cpu-f32',workers=3) if backend=='cpu-f32' else ex.run(dag_execution='direct')
            assert len(captured)==1
            captures[mode]=captured[0].copy();results[mode]=actual
            result_exact(actual,expected)
        finally:
            if backend!='cpu-f32':ex.close()
    result_exact(results['bounded'],results['original'])
    offsets=np.r_[0,np.cumsum(degrees)];limits=(degrees+3)//4
    outside=np.zeros(int(degrees.sum()),bool)
    for target,limit in enumerate(limits):outside[offsets[target]+limit:offsets[target+1]]=True
    assert np.all(captures['bounded'][outside]==sentinel)
    assert np.any(captures['original'][outside]!=sentinel)
    assert plans['bounded']['logical']==plans['original']['logical']
    assert plans['bounded']['buffers']==plans['original']['buffers']
    assert plans['bounded']['dispatches']==plans['original']['dispatches']
    np.savez_compressed(tmp_path/'saturation-queue.npz',bounded=captures['bounded'],original=captures['original'],outside=outside,offsets=offsets,limits=limits)
    save(tmp_path/'saturation-results.npz',results['bounded'],expected)
    (tmp_path/'saturation-model.json').write_text(json.dumps(model)+'\n')
    (tmp_path/'saturation-plans.json').write_text(json.dumps(plans)+'\n')
