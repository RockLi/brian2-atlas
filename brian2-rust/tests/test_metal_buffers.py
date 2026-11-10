"""Real Metal storage lifetime, bounded retention and exact replay controls."""
from copy import deepcopy
import json

import numpy as np
import pytest

from brian2_rust.metal import MetalExecutor, write_metal_results
from brian2_rust.metal_buffers import validate_mode
from test_cuda_graphs import result_exact
from test_gpu_summed_parallel import model_at, control
from test_metal import real_metal
from test_metal_delays import device


def test_modes_reject_invalid_values():
    for valid in ('auto','resident','direct'):assert validate_mode(valid)==valid
    for invalid in (None,[],1,True,'graph','typo'):
        with pytest.raises(ValueError):validate_mode(invalid)


@real_metal
def test_default_remains_direct_until_workload_selects_reuse(device,tmp_path):
    model=model_at(tmp_path/'ref',edges=0)
    with MetalExecutor(model,tmp_path/'metal',numeric_mode='float32') as ex:
        first=ex.run();second=ex.run();result_exact(first,second)
        for result in (first,second):
            m=result['metal_runtime']
            assert m['dag_execution']=='direct' and not m['buffers_reused']
            assert m['allocated_bytes']==m['total_buffer_bytes']>0
        assert ex._resident_dag_bytes==0


@real_metal
@pytest.mark.parametrize('route',['scan','sparse'])
@pytest.mark.parametrize('flavor',['plain','random','timed'])
def test_repeated_mutable_multiclock_dag_resets_and_avoids_constant_transfers(device,tmp_path,route,flavor):
    model=model_at(tmp_path/'ref',flavor=flavor)
    expected=control(model,tmp_path/'cpu')
    with MetalExecutor(model,tmp_path/'metal',numeric_mode='float32',event_delivery=route,dag_execution='auto') as ex:
        original_plan=ex.plan.sha256
        direct=ex.run(dag_execution='direct');result_exact(direct,expected)
        m=direct['metal_runtime'];total=m['total_buffer_bytes']
        assert m['allocated_bytes']==total
        assert m['uploaded_bytes']==total-m['omitted_spike_upload_bytes']
        readback=m['readback_bytes'];assert 0<readback<m['writable_buffer_bytes']
        assert m['resident_bytes']==0 and not m['buffers_reused']
        saved=deepcopy(direct)
        initial=[a.tobytes() for a in ex._dag_initial_storage[0]]
        first=ex.run(dag_execution='resident');result_exact(first,saved)
        m=first['metal_runtime'];writable=m['writable_buffer_bytes']
        assert 0<writable<total
        assert m['allocated_bytes']==m['resident_bytes']==total
        assert m['uploaded_bytes']==total-m['omitted_spike_upload_bytes']
        assert m['readback_bytes']==readback and not m['buffers_reused']
        for policy in ('resident','auto'):
            replay=ex.run(dag_execution=policy);result_exact(replay,saved)
            m=replay['metal_runtime']
            assert m['allocated_bytes']==0 and m['buffers_reused']
            assert m['uploaded_bytes']==writable-m['omitted_spike_upload_bytes'] and m['readback_bytes']==readback
            assert m['resident_bytes']==total
        # Mutating a returned array cannot corrupt cached initial or GPU data.
        replay['populations'][0]['states']['v'][:]=123
        result_exact(ex.run(),saved)
        assert initial==[a.tobytes() for a in ex._dag_initial_storage[0]]
        result_exact(ex.run(compute='cpu-f32',workers=2),saved)
        result_exact(ex.run(),saved)
        direct_again=ex.run(dag_execution='direct');result_exact(direct_again,saved)
        assert ex._resident_dag_bytes==0
        fresh=ex.run();result_exact(fresh,saved)
        assert not fresh['metal_runtime']['buffers_reused']
        write_metal_results(model,fresh,tmp_path/'transport')
        assert json.loads((tmp_path/'transport/summary.json').read_text())['metal_runtime']==fresh['metal_runtime']
        assert ex.plan.sha256==original_plan
    assert ex._resident_dag_bytes==0 and ex._metal_dag_metadata is None
    ex.close()
    with pytest.raises(RuntimeError,match='closed'):ex.run()


@real_metal
def test_retention_budget_auto_fallback_and_explicit_failure(device,tmp_path):
    model=model_at(tmp_path/'ref',edges=0)
    with MetalExecutor(model,tmp_path/'metal',numeric_mode='float32',dag_execution='auto') as ex:
        first=ex.run();total=ex._resident_dag_bytes
        cap=max(a.nbytes for a in ex._dag_initial_storage[0])
        assert 0<cap<total
        with pytest.raises(MemoryError,match='total memory'):ex.run(max_buffer_bytes=cap,dag_execution='resident')
        assert ex._resident_dag_bytes==0
        direct=ex.run(max_buffer_bytes=cap);result_exact(first,direct)
        assert direct['metal_runtime']['dag_execution']=='direct' and ex._resident_dag_bytes==0
        result_exact(first,ex.run())
        assert ex._resident_dag_bytes==total
        for invalid in (0,-1,True,1.5):
            with pytest.raises(ValueError,match='max_buffer_bytes'):ex.run(max_buffer_bytes=invalid)


@real_metal
def test_shape_failure_releases_gpu_storage(device,tmp_path):
    from brian2_rust.metal_dag import _prepare_dag_storage
    model=model_at(tmp_path/'ref')
    with MetalExecutor(model,tmp_path/'metal',numeric_mode='float32',dag_execution='auto') as ex:
        result=ex.run();assert ex._resident_dag_bytes>0
        arrays=_prepare_dag_storage(ex,512*1024**2)[0]
        arrays[0]=np.zeros(1,np.float32)
        with pytest.raises(ValueError,match='shape changed'):ex._execute_dag(arrays,max_buffer_bytes=512*1024**2)
        assert ex._resident_dag_bytes==0
        result_exact(ex.run(),result)
        counts=ex._metal_dag_metadata[5]
        old=counts[0];counts[0]=31
        try:
            with pytest.raises(RuntimeError,match='Invalid Metal DAG stage'):ex.run()
            assert ex._resident_dag_bytes==0
        finally:counts[0]=old
        result_exact(ex.run(),result)
