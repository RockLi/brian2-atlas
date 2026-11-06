"""Ordered current-event bitmaps preserve pending multiplicity and mixed lifecycles."""
from types import SimpleNamespace
import json
import numpy as np
import pytest
from gpu_target_bitset_compare import target_bitset
from brian2_rust.metal import MetalExecutor,build_metal_plan
from brian2_rust.cuda import CudaExecutor
from brian2_rust.metal_dag import run_dag
from brian2_rust import gpu_readback
from test_metal_delays import device
from test_gpu_spike_generator import BACKENDS
from test_gpu_target_sparse import fixture
from test_gpu_synapse_parallel import control
from test_gpu_sparse_saturation import boundary_model
from test_gpu_expression_contract import oracle
from test_gpu_composed_policies import full_compare
from test_gpu_workgroup import save
from test_metal_plasticity import equivalent
from test_gpu_composed_policies import (
    test_combined_queues_prefixes_and_full_observation,
    test_combined_policy_continuation_restore_and_reuse)


@pytest.fixture(autouse=True)
def bitset_policy():
    with target_bitset():yield


def executor(model,path,backend):
    options=dict(numeric_mode='float32',synapse_sparse=True,synapse_prefix=True)
    if backend=='cpu-f32':
        path.mkdir()
        return SimpleNamespace(model=model,plan=build_metal_plan(model,**options),directory=path,
            compile_seconds=0,device_name='CPU f32')
    return (MetalExecutor if backend=='metal' else CudaExecutor)(model,path,**options)


def run(ex,backend,mode='auto'):
    if backend=='cpu-f32':return run_dag(ex,max_buffer_bytes=512*1024**2,compute='cpu-f32',workers=3)
    return ex.run(dag_execution=mode)


@pytest.mark.parametrize('backend',BACKENDS)
@pytest.mark.parametrize('case',['uniform','heterogeneous','duplicate','early','empty','low'])
def test_bitset_pending_multiplicity_and_replay(device,tmp_path,backend,case):
    model=fixture(tmp_path/'ref',case);expected=control(model,tmp_path/'control',3)
    ex=executor(model,tmp_path/'native',backend)
    try:
        for mode in ('direct','auto','workgroup'):
            actual=run(ex,backend,mode);full_compare(actual,expected)
        save(tmp_path/'bitset-results.npz',actual,expected)
        (tmp_path/'bitset-model.json').write_text(json.dumps(model)+'\n')
        (tmp_path/'bitset-plan.json').write_text(ex.plan.to_json())
    finally:
        if backend!='cpu-f32':ex.close()


@pytest.mark.parametrize('backend',BACKENDS)
def test_bitset_word_boundaries_clear_and_leave_unused_storage(device,tmp_path,backend,monkeypatch):
    model,degrees=boundary_model(tmp_path/'ref');reference=oracle(model,tmp_path/'oracle')
    expected=control(model,tmp_path/'control',3)
    ex=executor(model,tmp_path/'native',backend)
    offsets=np.r_[0,np.cumsum(degrees)];used=np.zeros(int(degrees.sum()),bool)
    for base,degree in zip(offsets,degrees):used[base:base+(degree+31)//32]=True
    sentinel=np.uint32(0xdeadbeef);captures=[];fresh=gpu_readback.fresh_dag_arrays
    def poison(plan,*args,**kwargs):
        values,stats=fresh(plan,*args,**kwargs)
        index=plan.buffers.index('synapse/0/pathway/0/target_sparse_active_ranks')
        values[index].fill(sentinel);values[index][used]=0;captures.append(values[index])
        return values,stats
    try:
        with monkeypatch.context() as patch:
            patch.setattr(gpu_readback,'fresh_dag_arrays',poison)
            first=run(ex,backend,'direct');actual=run(ex,backend,'workgroup')
        full_compare(first,expected);full_compare(actual,expected)
        # Repeated affine accumulation rounds in f32 even with dyadic inputs.
        # Keep the compiled-f32 gate exact and retain the independent f64 gate
        # under the existing plasticity tolerance, with exact spike coordinates.
        equivalent(actual,reference)
        assert len(captures)==2
        for queue in captures:
            assert np.all(queue[used]==0) and np.all(queue[~used]==sentinel)
        np.savez_compressed(tmp_path/'bitset-queue.npz',first=captures[0],second=captures[1],used=used,degrees=degrees,offsets=offsets)
        save(tmp_path/'bitset-boundary-results.npz',actual,expected)
        save(tmp_path/'bitset-boundary-reference.npz',actual,reference)
        (tmp_path/'bitset-boundary-model.json').write_text(json.dumps(model)+'\n')
        (tmp_path/'bitset-boundary-plan.json').write_text(ex.plan.to_json())
    finally:
        if backend!='cpu-f32':ex.close()


@pytest.mark.parametrize('backend',BACKENDS)
def test_bitset_quiet_scalar_fault_is_not_skipped(device,tmp_path,backend):
    from test_gpu_target_sparse import test_empty_current_queue_keeps_scalar_faults
    test_empty_current_queue_keeps_scalar_faults(device,tmp_path,backend)
