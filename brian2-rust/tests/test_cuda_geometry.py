"""Exact resident-grid bounds and numerical replay with the same CUDA cubin."""
import hashlib
import json
from types import SimpleNamespace
import pytest
from brian2_rust.cuda import CudaExecutor
from brian2_rust.cuda_cooperative import launch_binding,CooperativeDag
from brian2_rust.protocol import canonical_bytes
from test_cuda_cooperative import native
from test_metal_delays import device
from test_gpu_summed_parallel import model_at,control
from test_cuda_graphs import result_exact
from test_gpu_workgroup import save


@pytest.mark.parametrize('blocks',[0,-1,True,1.0,'1',17,None])
def test_invalid_grid_is_not_clamped_or_stored(blocks):
    owner=SimpleNamespace(function=object(),program=SimpleNamespace(sha256='plan'),binary_sha256='binary',sms=2,active_blocks_per_sm=8,blocks=2)
    with pytest.raises(ValueError,match='residency'):
        CooperativeDag.configure_grid(owner,blocks)
    assert owner.blocks==2


def test_launch_identity_binds_grid_and_binary():
    a=launch_binding('plan','binary',1,2,8)
    b=launch_binding('plan','binary',16,2,8)
    c=launch_binding('plan','other-binary',16,2,8)
    assert len({x['sha256'] for x in (a,b,c)})==3
    for value in (a,b,c):
        assert value['sha256']==hashlib.sha256(canonical_bytes({k:v for k,v in value.items() if k!='sha256'})).hexdigest()


@native
@pytest.mark.parametrize('route',['scan','sparse'])
def test_geometry_changes_preserve_full_result_and_compiled_program(device,tmp_path,route):
    from gpu_stdp_precompiled import forbid_compilation,artifact_hashes
    model=model_at(tmp_path/'ref',edges=273,steps=16,flavor='random')
    expected=control(model,tmp_path/'control');records=[]
    with CudaExecutor(model,tmp_path/'gpu',numeric_mode='float32',event_delivery=route,dag_execution='cooperative') as ex:
        result_exact(ex.run(),expected)
        runtime=ex._cooperative_dag;identity=runtime.binary_sha256;function=runtime.function
        before=artifact_hashes(tmp_path/'gpu');pointers=tuple(a.data.ptr for a in runtime.resident.gpu)
        grids=sorted({1,4,runtime.sms,runtime.sms*runtime.active_blocks_per_sm})
        for blocks in grids:
            binding=runtime.configure_grid(blocks)
            with forbid_compilation():actual=ex.run()
            result_exact(actual,expected)
            observed=actual['cuda_runtime']['dag_execution']
            assert observed['launch_binding']==binding and observed['workgroups']==blocks
            assert runtime.function is function and runtime.binary_sha256==identity
            assert tuple(a.data.ptr for a in runtime.resident.gpu)==pointers
            save(tmp_path/f'geometry-{blocks}.npz',actual,expected);records.append(observed)
        with pytest.raises(ValueError):runtime.configure_grid(runtime.sms*runtime.active_blocks_per_sm+1)
        with forbid_compilation():result_exact(ex.run(),expected)
        assert artifact_hashes(tmp_path/'gpu')==before
        ex.close()
        with pytest.raises(RuntimeError,match='closed'):runtime.configure_grid(1)
    (tmp_path/'geometry-runtimes.json').write_text(json.dumps(records,indent=2)+'\n')
