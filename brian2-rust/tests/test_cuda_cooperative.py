"""Canonical cross-block barriers, numerical faults and Device continuation."""
import json
import os
from types import SimpleNamespace
import numpy as np
import pytest
from brian2_rust.cuda import CudaExecutor,build_cuda_plan
from brian2_rust.cuda_cooperative import prepare,grid_size
from brian2_rust.metal_dag import _prepare_dag_storage
from test_metal_delays import device,ROOT
from test_gpu_summed_parallel import model_at,control
from test_cuda_graphs import result_exact
from test_gpu_workgroup import save

native=pytest.mark.skipif(os.environ.get('B2_TEST_CUDA')!='1',reason='requires NVIDIA GPU')


def test_capability_and_residency_bounds():
    good=dict(cooperativeLaunch=1,multiProcessorCount=80)
    assert grid_size(good,2,273)==3
    assert grid_size(good,2,32768)==80
    assert grid_size(good,1,0)==1
    with pytest.raises(ValueError,match='support'):grid_size(dict(good,cooperativeLaunch=0),2,273)
    with pytest.raises(ValueError,match='capacity'):grid_size(good,0,273)


def test_canonical_source_schedule_and_budget(device,tmp_path,monkeypatch):
    from brian2_rust import cuda_cooperative as module
    from brian2_rust.gpu_schedule import dispatch_ticks
    from brian2_rust.cuda_graphs import select_mode,validate_mode
    model=model_at(tmp_path/'ref',edges=273,steps=8)
    owner=SimpleNamespace(model=model,plan=build_cuda_plan(model,numeric_mode='float32',event_delivery='sparse'))
    arrays,*_=_prepare_dag_storage(owner,512*1024**2);p=prepare(owner,arrays)
    schedule=[(s,t) for s,t in dispatch_ticks(owner.plan.logical.clocks,tuple(d.clock for d in owner.plan.dispatches)) if owner.plan.dispatches[s].lanes]
    assert list(zip(p.stages.tolist(),p.ticks.tolist()))==schedule
    assert 'grid.sync();' in p.source and '__syncthreads()' not in p.source
    assert 'addresses[' in p.source and 'arena+offsets' not in p.source
    assert 'blockIdx.x*blockDim.x+threadIdx.x' in p.source
    assert p.manifest['total_buffer_bytes']==sum(a.nbytes for a in arrays)+p.manifest['metadata_bytes']
    assert validate_mode('cooperative')=='cooperative'
    assert select_mode('auto',100)[0]=='resident'
    monkeypatch.setattr(module,'MAX_LANE_VISITS',1)
    with pytest.raises(ValueError,match='budget'):prepare(owner,arrays)


@native
@pytest.mark.parametrize('route',['scan','sparse'])
@pytest.mark.parametrize('flavor',['plain','random','timed'])
def test_multiblock_replay_full_results_and_release(device,tmp_path,route,flavor):
    model=model_at(tmp_path/'ref',edges=273,steps=16,flavor=flavor)
    expected=control(model,tmp_path/'control')
    with CudaExecutor(model,tmp_path/'gpu',numeric_mode='float32',event_delivery=route) as ex:
        result_exact(ex.run(),expected)
        for i in range(2):
            actual=ex.run(dag_execution='cooperative');result_exact(actual,expected)
            runtime=actual['cuda_runtime']['dag_execution']
            assert runtime['workgroups']>1 and runtime['kernel_launches']==1
            assert runtime['buffer_reused']==bool(i)
            save(tmp_path/f'cooperative-results-{i}.npz',actual,expected)
            (tmp_path/f'cooperative-runtime-{i}.json').write_text(json.dumps(runtime,indent=2)+'\n')
        handle=ex._cooperative_dag
        with pytest.raises(MemoryError):ex.run(dag_execution='cooperative',max_buffer_bytes=1)
        result_exact(ex.run(dag_execution='resident'),expected)
        assert ex._cooperative_dag is None and handle.resident is None and handle.module is None
        result_exact(ex.run(dag_execution='cooperative'),expected)
        ex.close();assert ex._cooperative_dag is None


@native
def test_typed_results_and_monitors(device,tmp_path):
    from test_gpu_typed_storage import synaptic_network
    from test_gpu_monitors import setup,DT
    from brian2_rust.export import lower_network
    setup(tmp_path/'ref');model=lower_network(synaptic_network()[0],8*DT)
    expected=control(model,tmp_path/'control')
    with CudaExecutor(model,tmp_path/'gpu',numeric_mode='float32',event_delivery='sparse') as ex:
        actual=ex.run(dag_execution='cooperative');result_exact(actual,expected)
        save(tmp_path/'cooperative-typed.npz',actual,expected)


@native
@pytest.mark.parametrize('empty',[False,True])
def test_fault_in_last_lane_or_empty_stage_is_not_published(device,tmp_path,empty):
    from test_gpu_refractory import refresh_code
    from brian2_rust.spec import bits
    model=model_at(tmp_path/'ref',edges=0 if empty else 273,steps=4)
    code=next(c for c in model['definition']['synapses'][0]['code_objects'] if c['kind']=='summed_variable')
    bad=dict(op='floor_div',left=dict(op='literal',bits=bits(1.)),right=dict(op='literal',bits=bits(0.)))
    if not empty:
        model['instance']['synapses'][0]['initial_state']['k']=['00000001']*272+['00000000']
        bad=dict(op='floor_div',left=dict(op='integer',dtype='i32',value='1'),right=dict(op='load',name='k'))
        code['effects']['reads']=sorted(set(code['effects']['reads'])|{'k'})
    code['scalar' if empty else 'vector'].insert(0,dict(target='_fault',dtype='f64' if empty else 'i32',dimensions=[0.]*7,condition=None,value=bad))
    refresh_code(model,code)
    with CudaExecutor(model,tmp_path/'gpu',numeric_mode='float32',event_delivery='sparse') as ex:
        with pytest.raises(FloatingPointError):ex.run(dag_execution='cooperative')


@native
def test_device_pending_store_restore_and_mode_option(device,tmp_path):
    import brian2 as b
    from test_gpu_multiclock import network,DT
    records=[]
    for mode in ('direct','cooperative'):
        device.reinit()
        b.set_device('rust_standalone',engine='cuda',numeric_mode='float32',event_delivery='sparse',
            directory=tmp_path/mode,runner=ROOT/'target/release/b2-runner',gpu_compile_reuse=True,
            cuda_dag_execution=mode)
        net,pre,post,syn,monitors,spikes=network(True);states=[]
        for i in range(3):
            if i==1:net.store('saved')
            if i==2:net.restore('saved')
            net.run(6*DT)
            states.append([np.asarray(x).copy() for x in [pre.v[:],post.v[:],syn.w[:],syn.x[:],
                *[m.v[:] for m in monitors],*[s.t[:] for s in spikes],*[s.i[:] for s in spikes]]])
        records.append(states);ex=device._gpu_executor;device.close_gpu()
        if mode=='cooperative':assert ex._cooperative_dag is None
    for a,e in zip(*records,strict=True):
        for x,y in zip(a,e,strict=True):np.testing.assert_array_equal(x,y)
    np.savez_compressed(tmp_path/'cooperative-device.npz',**{f'{mode}/{i}/{j}':v for mode,states in zip(('reference','actual'),records) for i,values in enumerate(states) for j,v in enumerate(values)})


def test_close_releases_both_runtimes_when_stream_sync_fails():
    from contextlib import nullcontext
    calls=[]
    def fail_sync():raise RuntimeError('injected asynchronous CUDA failure')
    runtime=SimpleNamespace(close=lambda:calls.append('cooperative'))
    owner=SimpleNamespace(device=nullcontext(),stream=SimpleNamespace(synchronize=fail_sync),
        _cooperative_dag=runtime,_release_resident_dag=lambda:calls.append('ordinary'))
    with pytest.raises(RuntimeError,match='asynchronous'):
        CudaExecutor.close(owner)
    assert calls==['cooperative','ordinary']
    assert owner._cooperative_dag is None and owner.closed
    assert owner.kernels==owner.modules==owner.chunk_kernels==[]
    assert owner.chunk_advance is None and owner._compiled_binaries=={}
