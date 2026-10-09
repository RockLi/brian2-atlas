"""Compact target word offsets, pending multiplicity, and immutable reuse inputs."""
from types import SimpleNamespace
import json
import numpy as np
import pytest
from brian2_rust.metal import MetalExecutor,build_metal_plan
from brian2_rust.cuda import CudaExecutor
from brian2_rust.metal_dag import run_dag,_prepare_dag_storage
from brian2_rust.gpu_buffer_transfer import writable
from brian2_rust import gpu_readback
from test_metal_delays import device
from test_gpu_spike_generator import BACKENDS
from test_gpu_target_sparse import fixture
from test_gpu_synapse_parallel import control
from test_gpu_sparse_saturation import boundary_model
from test_gpu_composed_policies import full_compare
from test_gpu_workgroup import save


def executor(model,path,backend):
    options=dict(numeric_mode='float32',synapse_sparse='bitset',synapse_prefix=True)
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
def test_compact_pending_multiplicity_and_replay(device,tmp_path,backend,case):
    model=fixture(tmp_path/'ref',case);expected=control(model,tmp_path/'control',3)
    ex=executor(model,tmp_path/'native',backend)
    try:
        for mode in ('direct','auto','workgroup'):
            actual=run(ex,backend,mode);full_compare(actual,expected)
        save(tmp_path/'compact-results.npz',actual,expected)
        (tmp_path/'compact-model.json').write_text(json.dumps(model)+'\n')
        (tmp_path/'compact-plan.json').write_text(ex.plan.to_json())
    finally:
        if backend!='cpu-f32':ex.close()


@pytest.mark.parametrize('backend',BACKENDS)
def test_compact_boundary_words_and_read_only_offsets(device,tmp_path,backend,monkeypatch):
    model,degrees=boundary_model(tmp_path/'ref');expected=control(model,tmp_path/'control',3)
    ex=executor(model,tmp_path/'native',backend)
    words=np.r_[0,np.cumsum((degrees+31)//32)].astype(np.uint32)
    captures=[];fresh=gpu_readback.fresh_dag_arrays;readback=gpu_readback.readback_bindings
    q=ex.plan.buffers.index('synapse/0/pathway/0/target_sparse_bitmap_words');o=q+1
    assert q in writable(ex.plan) and o not in writable(ex.plan)
    assert o not in readback(ex.plan)
    def guarded(plan,*args,**kwargs):
        arrays,stats=fresh(plan,*args,**kwargs)
        assert arrays[q].size==int(words[-1])
        np.testing.assert_array_equal(arrays[o],words)
        # Append guard words without changing any address seen by native kernels.
        arrays[q]=np.r_[arrays[q],np.full(16,0xdeadbeef,np.uint32)]
        arrays[o]=np.r_[arrays[o],np.full(16,0xabcddcba,np.uint32)]
        captures.append((arrays[q],arrays[o]))
        return arrays,stats
    try:
        with monkeypatch.context() as patch:
            patch.setattr(gpu_readback,'fresh_dag_arrays',guarded)
            patch.setattr(gpu_readback,'readback_bindings',lambda plan:tuple(sorted(set(readback(plan))|{o})))
            for mode in ('direct','workgroup'):
                actual=run(ex,backend,mode);full_compare(actual,expected)
        assert len(captures)==2
        for bitmap,offsets in captures:
            assert np.all(bitmap[:words[-1]]==0)
            assert np.all(bitmap[words[-1]:]==np.uint32(0xdeadbeef))
            np.testing.assert_array_equal(offsets[:len(words)],words)
            assert np.all(offsets[len(words):]==np.uint32(0xabcddcba))
        np.savez_compressed(tmp_path/'compact-guards.npz',degrees=degrees,word_offsets=words,
            bitmap0=captures[0][0],bitmap1=captures[1][0],offsets0=captures[0][1],offsets1=captures[1][1])
        save(tmp_path/'compact-boundary-results.npz',actual,expected)
        (tmp_path/'compact-boundary-model.json').write_text(json.dumps(model)+'\n')
        (tmp_path/'compact-boundary-plan.json').write_text(ex.plan.to_json())
    finally:
        if backend!='cpu-f32':ex.close()


def test_compact_storage_exact_capacity_and_non_bitmap_layout_unchanged(device,tmp_path):
    from gpu_target_bitset_compare import target_bitset
    model,degrees=boundary_model(tmp_path/'ref')
    plans={};arrays={}
    for name,mode in [('rank',True),('compact','bitset')]:
        plan=build_metal_plan(model,numeric_mode='float32',synapse_sparse=mode)
        plans[name]=plan;arrays[name]=_prepare_dag_storage(SimpleNamespace(model=model,plan=plan),512*1024**2)[0]
    with target_bitset():plan=build_metal_plan(model,numeric_mode='float32',synapse_sparse=True)
    plans['legacy']=plan;arrays['legacy']=_prepare_dag_storage(SimpleNamespace(model=model,plan=plan),512*1024**2)[0]
    q=plan.buffers.index('synapse/0/pathway/0/target_sparse_active_ranks')
    for i,(rank,legacy,compact) in enumerate(zip(arrays['rank'],arrays['legacy'],arrays['compact'],strict=True)):
        np.testing.assert_array_equal(rank,legacy)
        if i not in (q,q+1):np.testing.assert_array_equal(legacy,compact)
    assert arrays['compact'][q].size==sum((degrees+31)//32)
    assert arrays['compact'][q+1].size==len(degrees)+1
    saved=sum(a.nbytes for a in arrays['legacy'])-sum(a.nbytes for a in arrays['compact'])
    assert saved==4*(degrees.sum()-sum((degrees+31)//32)-1)>0
    (tmp_path/'compact-storage.json').write_text(json.dumps(dict(bytes_saved=int(saved),
        buffers={k:[dict(name=n,shape=list(a.shape),dtype=str(a.dtype),bytes=a.nbytes) for n,a in zip(plans[k].buffers,v,strict=True)] for k,v in arrays.items()}))+'\n')
    (tmp_path/'compact-storage-model.json').write_text(json.dumps(model)+'\n')
