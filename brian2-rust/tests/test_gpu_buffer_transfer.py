"""Cross-activation allocations preserve fresh input, ticks and pending events."""
from copy import deepcopy
import json

import brian2 as b
import numpy as np
import pytest
import importlib

from brian2_rust.cuda import CudaExecutor
from brian2_rust.metal import MetalExecutor
from brian2_rust.gpu_buffer_transfer import adopt_buffers,refresh_indices,compatible_indices
from brian2_rust.protocol import attach_protocol
from test_metal_delays import device,ROOT
from test_gpu_spike_generator import BACKENDS
from test_gpu_summed_parallel import model_at,control
from test_cuda_graphs import result_exact
from test_gpu_multiclock import network,DT


def test_refresh_accounts_for_old_writes_changed_constants_and_exact_bits():
    old=[np.array([0.],np.float32),np.array([1],np.int64),np.array([2],np.int64),np.array([3],np.int64)]
    new=[np.array([-0.],np.float32),old[1].copy(),old[2].copy(),np.array([4],np.int64)]
    assert refresh_indices(old,new,{1},{2})==[0,1,2,3]
    assert refresh_indices(old,[a.copy() for a in old],set(),set())==[]
    assert refresh_indices(old,new[:-1],set(),set()) == [0]
    assert compatible_indices(old,new[:-1])==[0,1,2]
    new[1]=new[1].astype(np.float64)
    assert refresh_indices(old,new,set(),set())==[0,1,3]
    assert compatible_indices(old,new)==[0,2,3]


def test_device_option_rejects_cpu_and_nonboolean(device):
    with pytest.raises(NotImplementedError,match='GPU engine'):
        device.activate(engine='reference',gpu_buffer_reuse=True)
    with pytest.raises(NotImplementedError,match='boolean'):
        device.activate(engine='metal',numeric_mode='float32',gpu_buffer_reuse=1)
    with pytest.raises(NotImplementedError,match='positive integer'):
        device.activate(engine='metal',numeric_mode='float32',gpu_max_buffer_bytes=0)
    with pytest.raises(NotImplementedError,match='GPU engine'):
        device.activate(engine='reference',gpu_max_buffer_bytes=1024)


def test_device_accepts_explicit_gpu_buffer_budget(device):
    device.activate(engine='metal',numeric_mode='float32',gpu_max_buffer_bytes=2**30)
    assert device.build_options['gpu_max_buffer_bytes'] == 2**30


def runtime(result,backend):
    m=result[backend+'_runtime']
    return m if backend=='metal' else m['dag_execution']


def live_route_model(path):
    """Float-only immutable edges actually bind the generic sparse queues.

    The mutable summed fixture is canonical and no longer allocates its unused
    generic queues after projection pruning; it cannot test route layout growth.
    """
    from test_gpu_composed_models import setup
    from brian2_rust.export import lower_network
    setup(path)
    pre=b.NeuronGroup(17,'v:1',threshold='v>=1',reset='v=0',dt=DT,name='float_source')
    post=b.NeuronGroup(19,'v:1',threshold='v>=1',reset='v=0',dt=DT,name='float_target')
    pre.v=np.arange(17)%8/8;pre.run_regularly('v+=0.25')
    syn=b.Synapses(pre,post,'w:1 (constant)',on_pre='v_post+=w',clock=pre.clock,name='immutable_float')
    syn.connect(i=np.arange(17),j=np.arange(17));syn.w=.125
    monitor=b.StateMonitor(post,'v',record=True);spikes=b.SpikeMonitor(post)
    return lower_network(b.Network(pre,post,syn,monitor,spikes),8*DT)


@pytest.mark.parametrize('backend',BACKENDS[1:])
@pytest.mark.parametrize('change',['same','topology','grow','shrink','route-grow','route-shrink'])
def test_transfer_changed_input_and_ownership(device,tmp_path,backend,change):
    first=live_route_model(tmp_path/'ref') if change.startswith('route-') else model_at(tmp_path/'ref',edges=273,steps=4)
    second=deepcopy(first)
    if change=='topology':
        inst=second['instance']['synapses'][0]
        inst['target']=list(reversed(inst['target']))
    elif change in {'grow','shrink'}:second=model_at(tmp_path/'new-ref',edges=274 if change=='grow' else 272,steps=4)
    attach_protocol(second)
    expected=control(second,tmp_path/'cpu')
    cls=MetalExecutor if backend=='metal' else CudaExecutor
    with cls(first,tmp_path/'first',numeric_mode='float32',event_delivery='scan' if change=='route-grow' else 'sparse',dag_execution='resident') as old:
        old.run()
        with cls(second,tmp_path/'second',numeric_mode='float32',event_delivery='scan' if change=='route-shrink' else 'sparse',dag_execution='resident') as new:
            with pytest.raises(ValueError,match='positive integer'):
                adopt_buffers(new,old,backend=backend,max_buffer_bytes=0)
            moved=adopt_buffers(new,old,backend=backend)
            assert moved
            if moved:
                assert not (old._resident_dag_bytes if backend=='metal' else old._resident_dag)
            old.close()
            result=new.run();result_exact(result,expected)
            m=runtime(result,backend);assert m.get('buffers_adopted',False)==moved
            if moved:assert m['buffers_reused' if backend=='metal' else 'buffer_reused']
            total=sum(a.nbytes for a in new._dag_initial_storage[0])
            assert m['reused_buffer_bytes']+m['allocated_bytes']==total
            assert m['reused_buffer_count']>0
            if change in {'grow','shrink'}:assert 0<m['allocated_bytes']<total
            elif change=='same' or change=='topology':assert m['allocated_bytes']==0
            else:
                assert len(old._dag_initial_storage[0])!=len(new._dag_initial_storage[0])
                queues={f'synapse/0/{kind}' for kind in ('source_offsets','source_ranks','rank_targets','active_ranks','active_counts')}
                sparse,scan=(new.plan,old.plan) if change=='route-grow' else (old.plan,new.plan)
                assert set(sparse.buffers)-set(scan.buffers)==queues
                assert queues & {sparse.buffers[i] for d in sparse.dispatches for i in d.bindings}
            result_exact(new.run(),expected)
            assert not runtime(new.run(),backend).get('buffers_adopted',False)
            payload={}
            for label,res in [('actual',result),('reference',expected)]:
                for i,p in enumerate(res['populations']):
                    for name,value in p['states'].items():payload[f'{label}/population/{i}/{name}']=value
                    for name in ('spike_ticks','indices','counts'):payload[f'{label}/population/{i}/{name}']=p[name]
                for i,s in enumerate(res['synapses']):
                    for name,value in s['states'].items():payload[f'{label}/synapse/{i}/{name}']=value
            np.savez_compressed(tmp_path/'transfer-results.npz',**payload)
            (tmp_path/'transfer-runtime.json').write_text(json.dumps(m,indent=2)+'\n')


@pytest.mark.parametrize('backend',BACKENDS[1:])
def test_device_segment_restore_mutation_cleanup_and_fresh_control(device,tmp_path,backend):
    snapshots=[];reports=[]
    for reuse in (False,True):
        device.reinit()
        b.set_device('rust_standalone',engine=backend,numeric_mode='float32',event_delivery='sparse',
                     gpu_buffer_reuse=reuse,directory=tmp_path/str(reuse),runner=ROOT/'target/release/b2-runner',
                     **({'cuda_dag_execution':'graph'} if backend=='cuda' else {}))
        net,pre,post,syn,monitors,spikes=network(True)
        state=[];metadata=[]
        for step in range(4):
            if step==1:net.store('checkpoint')
            if step==2:net.restore('checkpoint')
            if step==3:
                pre.v=[.125,.25,.5];syn.w=np.asarray(syn.w[:])+.125
            net.run(6*DT)
            state.append(dict(pre=np.asarray(pre.v[:]).copy(),post=np.asarray(post.v[:]).copy(),
                w=np.asarray(syn.w[:]).copy(),traces=[np.asarray(m.v[:]).copy() for m in monitors],
                spikes=[np.asarray(s.t[:]).copy() for s in spikes]))
            metadata.append(json.loads((device.last_run_directory/'rust/summary.json').read_text())[backend+'_runtime'])
            if backend=='cuda':assert not metadata[-1]['dag_execution']['graph_reused']
        snapshots.append(state);reports.append(metadata)
        retained=device._gpu_executor
        assert (retained is not None)==reuse
        device.close_gpu();device.close_gpu();assert device._gpu_executor is None
        if retained is not None:assert not retained.handles if backend=='metal' else retained.closed
        # Releasing allocation ownership must not erase frontend continuation.
        net.run(6*DT)
        device.reinit();assert device._gpu_executor is None
    for a,e in zip(snapshots[0],snapshots[1],strict=True):
        for field in ('pre','post','w'):np.testing.assert_array_equal(a[field],e[field])
        for field in ('traces','spikes'):
            for x,y in zip(a[field],e[field],strict=True):np.testing.assert_array_equal(x,y)
    assert not any(m['activation_buffer_reuse']['adopted'] for m in reports[0])
    assert all(m['activation_buffer_reuse']['adopted'] for m in reports[1][1:])
    (tmp_path/'activation-reports.json').write_text(json.dumps(reports,indent=2)+'\n')
    arrays={}
    for label,states in zip(('fresh','reuse'),snapshots,strict=True):
        for step,state in enumerate(states):
            for field,value in state.items():
                if isinstance(value,list):
                    for i,v in enumerate(value):arrays[f'{label}/{step}/{field}/{i}']=v
                else:arrays[f'{label}/{step}/{field}']=value
    np.savez_compressed(tmp_path/'activation-results.npz',**arrays)


@pytest.mark.parametrize('backend',BACKENDS[1:])
def test_failed_result_publication_releases_both_executors(device,tmp_path,backend,monkeypatch):
    b.set_device('rust_standalone',engine=backend,numeric_mode='float32',event_delivery='sparse',
                 gpu_buffer_reuse=True,directory=tmp_path/'run',runner=ROOT/'target/release/b2-runner')
    net,*_=network(True);net.run(6*DT);previous=device._gpu_executor
    module=importlib.import_module('brian2_rust.'+backend)
    def failed(*args,**kwargs):raise OSError('injected result publication failure')
    monkeypatch.setattr(module,'write_'+backend+'_results',failed)
    with pytest.raises(OSError,match='injected result publication failure'):net.run(6*DT)
    assert device._gpu_executor is None
    assert not previous.handles if backend=='metal' else previous.closed


def test_metal_transfer_rejects_different_bridge_before_reading_handles():
    from types import SimpleNamespace
    plan=SimpleNamespace(dispatches=(True,))
    def owner(name,retained):
        return SimpleNamespace(plan=plan,dag_execution='indirect',handles=[1],_resident_dag_bytes=retained,
            bridge=SimpleNamespace(_name=name))
    current=owner('/new/metal-bridge-new-layout.dylib',0)
    previous=owner('/old/metal-bridge-old-layout.dylib',128)
    assert not adopt_buffers(current,previous,backend='metal')
    assert previous._resident_dag_bytes==128
