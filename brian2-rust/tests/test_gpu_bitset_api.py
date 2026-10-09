"""Public bitmap selection, plan validation and reuse across queue representations."""
from dataclasses import replace
import json
import platform
from types import SimpleNamespace
import numpy as np
import pytest
from brian2_rust.metal import build_metal_plan,MetalExecutor
from brian2_rust.cuda import build_cuda_plan,CudaExecutor
from brian2_rust.plan import build_execution_plan,verify_execution_plan,explain_plan
from brian2_rust.gpu_buffer_transfer import adopt_buffers
from brian2_rust import gpu_target_sparse
from test_metal_delays import device
from test_gpu_spike_generator import BACKENDS
from test_gpu_sparse_saturation import boundary_model
from test_gpu_target_sparse import fixture
from test_gpu_synapse_parallel import control
from test_gpu_workgroup import save
import test_gpu_composed_policies as mixed


@pytest.mark.parametrize('backend',BACKENDS)
@pytest.mark.parametrize('prefix',[False,True])
@pytest.mark.parametrize('case',['seed-7','seed-42','named-pathways','trace-42'])
def test_public_bitmap_mixed_model(device,tmp_path,backend,prefix,case):
    mixed.test_combined_queues_prefixes_and_full_observation(device,tmp_path,backend,prefix,case,queue_mode='bitset')


@pytest.mark.parametrize('backend',BACKENDS[1:])
@pytest.mark.parametrize('queued',[False,True])
@pytest.mark.parametrize('seed',[7,42])
def test_public_bitmap_device_lifecycle(device,tmp_path,backend,queued,seed):
    mixed.test_combined_policy_continuation_restore_and_reuse(device,tmp_path,backend,queued,seed,queue_mode='bitset')


def test_public_plan_matches_experiment_and_rejects_wrong_selection(device,tmp_path):
    from gpu_target_bitset_compare import target_bitset
    model=fixture(tmp_path/'ref','heterogeneous');records=[]
    for backend,build,cls in [('metal',build_metal_plan,MetalExecutor),('cuda',build_cuda_plan,CudaExecutor)]:
        for prefix in (False,True):
            options=dict(numeric_mode='float32',event_delivery='sparse',synapse_prefix=prefix)
            rank=build(model,**options,synapse_sparse=True)
            public=build_execution_plan(model,backend=backend,**options,synapse_sparse='bitset')
            with target_bitset():experimental=build(model,**options,synapse_sparse=True)
            assert public.logical==experimental.logical
            assert len(public.kernels)==len(experimental.kernels)
            expected_buffers=tuple(n.replace('target_sparse_active_ranks','target_sparse_bitmap_words').replace('target_sparse_active_counts','target_sparse_bitmap_offsets') for n in experimental.buffers)
            assert public.buffers==expected_buffers
            normalized=tuple(replace(d,role=gpu_target_sparse.ROLE if d.role==gpu_target_sparse.BITSET_ROLE else d.role,types=e.types) for d,e in zip(public.dispatches,experimental.dispatches,strict=True))
            assert normalized==experimental.dispatches
            assert public.sha256!=rank.sha256 and public.sha256!=experimental.sha256
            verify_execution_plan(public,model,synapse_sparse='bitset',synapse_prefix=prefix)
            verify_execution_plan(rank,model,synapse_sparse=True,synapse_prefix=prefix)
            assert 'ordered bitmaps' in explain_plan(public) and 'ordered bitmaps' not in explain_plan(rank)
            for wrong,mode in [(rank,'bitset'),(public,True),(experimental,'bitset')]:
                with pytest.raises(ValueError,match='plan does not match'):
                    verify_execution_plan(wrong,model,synapse_sparse=mode,synapse_prefix=prefix)
                destination=tmp_path/f'{backend}-{prefix}-{mode}'
                if backend=='metal' and platform.system()!='Darwin':
                    with pytest.raises(RuntimeError,match='Apple Metal requires macOS'):
                        cls(model,destination,**options,synapse_sparse=mode,plan=wrong)
                else:
                    with pytest.raises(ValueError,match='plan does not match'):
                        cls(model,destination,**options,synapse_sparse=mode,plan=wrong)
                assert not destination.exists()
            records.append(dict(backend=backend,prefix=prefix,rank=rank.to_dict(),public=public.to_dict(),experimental=experimental.to_dict()))
    (tmp_path/'api-plan-proofs.json').write_text(json.dumps(records)+'\n')
    (tmp_path/'api-plan-model.json').write_text(json.dumps(model)+'\n')


def test_invalid_public_selection_is_rejected_before_device_setup(device,tmp_path):
    import brian2 as b
    model=fixture(tmp_path/'ref','empty')
    for value in (None,0,1,[],{},'yes','BITSET',np.bool_(True)):
        for build in (build_metal_plan,build_cuda_plan):
            with pytest.raises(ValueError,match='synapse_sparse'):build(model,numeric_mode='float32',synapse_sparse=value)
        for engine in ('metal','cuda'):
            device.reinit()
            with pytest.raises(NotImplementedError,match='gpu_synapse_sparse'):
                b.set_device('rust_standalone',engine=engine,numeric_mode='float32',gpu_synapse_sparse=value,directory=tmp_path/'invalid')
    for engine in ('reference','aot'):
        device.reinit()
        with pytest.raises(NotImplementedError,match='GPU engine'):
            b.set_device('rust_standalone',engine=engine,gpu_synapse_sparse='bitset')
    with pytest.raises(ValueError,match='GPU backend'):build_execution_plan(model,synapse_sparse='bitset')
    assert not (tmp_path/'invalid').exists()


@pytest.mark.parametrize('backend',BACKENDS[1:])
def test_public_queue_switch_refreshes_storage_and_recompiles_changed_kernels(device,tmp_path,backend):
    model,_=boundary_model(tmp_path/'ref');expected=control(model,tmp_path/'control',3)
    cls=MetalExecutor if backend=='metal' else CudaExecutor
    previous=None;records=[]
    try:
        for i,mode in enumerate((True,'bitset',True,'bitset')):
            current=cls(model,tmp_path/str(i),numeric_mode='float32',synapse_sparse=mode,compile_reuse=True,reuse_from=previous,
                dag_execution='auto')
            adopted=adopt_buffers(current,previous,backend=backend) if previous is not None else False
            if previous is not None:previous.close()
            previous=current
            queue=current.plan.buffers.index('synapse/0/pathway/0/target_sparse_'+('bitmap_words' if mode=='bitset' else 'active_ranks'))
            uploads=list(getattr(current,'_activation_upload_indices',()))
            if adopted:assert queue in uploads
            actual=current.run();mixed.full_compare(actual,expected)
            mixed.full_compare(current.run(),actual)
            if i and backend=='cuda':assert adopted and current.compilation_report['kernels_reused']>0
            if i==1:assert current.compilation_report['kernels_compiled']>=2
            save(tmp_path/f'api-switch-{i}.npz',actual,expected)
            records.append(dict(mode=mode,adopted=adopted,activation_upload_indices=uploads,compilation=current.compilation_report.copy(),
                plan=current.plan.to_dict(),runtime=actual.get('cuda_runtime',actual.get('metal_runtime'))))
    finally:
        if previous is not None:previous.close()
    (tmp_path/'api-switch-records.json').write_text(json.dumps(records)+'\n')
    (tmp_path/'api-switch-model.json').write_text(json.dumps(model)+'\n')
