"""Reduced host copies preserve complete results and device lifecycle state."""
import json
from types import SimpleNamespace
import pytest
from brian2_rust import gpu_readback
from brian2_rust.cuda import CudaExecutor
from brian2_rust.metal import MetalExecutor
from brian2_rust.export import lower_network
from test_metal_delays import device
from test_gpu_spike_generator import BACKENDS
from test_gpu_composed_models import setup, network, DT
from test_gpu_expression_contract import oracle
from test_gpu_custom_events import compare
from test_cuda_graphs import result_exact
from test_gpu_workgroup import save


def test_unknown_buffers_and_faults_are_conservatively_read_back():
    names=('population/0/8','population/0/11','population/0/linked_values',
           'synapse/0/delivered','synapse/0/pathway/0/history',
           'synapse/0/pathway/0/new_fault','future/kind','population/0/1')
    plan=SimpleNamespace(buffers=names,dispatches=[SimpleNamespace(
        bindings=tuple(range(len(names))),types=('long',)*7+('const float',))])
    assert gpu_readback.readback_bindings(plan)==(0,1,3,5,6)


@pytest.mark.parametrize('backend',BACKENDS[1:])
def test_omitted_scratch_does_not_hide_overwritten_dag_fault(device,tmp_path,backend):
    from test_gpu_expression_contract import base,statement,unary,lit,refresh_code
    model,code=base(tmp_path/'ref',dag=True)
    code['vector']=[statement('y',unary('exp',lit(90))),statement('y',lit(0))]
    refresh_code(model,code)
    cls=MetalExecutor if backend=='metal' else CudaExecutor
    modes=('direct','resident','indirect') if backend=='metal' else ('direct','resident','graph','chunked')
    for mode in modes:
        with cls(model,tmp_path/mode,numeric_mode='float32',dag_execution=mode) as ex:
            for _ in range(2):
                with pytest.raises(FloatingPointError):ex.run()


@pytest.mark.parametrize('backend',BACKENDS[1:])
@pytest.mark.parametrize('route',['scan','sparse'])
@pytest.mark.parametrize('seed',[7,42])
def test_full_results_with_scratch_readback_omitted(device,tmp_path,monkeypatch,backend,route,seed):
    setup(tmp_path/'ref');net,*_=network(seed);model=lower_network(net,24*DT)
    (tmp_path/'model.json').write_text(json.dumps(model,sort_keys=True)+'\n')
    expected=oracle(model,tmp_path/'oracle')
    cls=MetalExecutor if backend=='metal' else CudaExecutor
    modes=('direct','resident','indirect') if backend=='metal' else ('direct','resident','graph','chunked')
    reports=[]
    for mode in modes:
        with cls(model,tmp_path/mode,numeric_mode='float32',event_delivery=route,dag_execution=mode) as ex:
            with monkeypatch.context() as patch:
                patch.setattr(gpu_readback,'readback_bindings',lambda plan:tuple(range(len(plan.buffers))))
                full=ex.run()
            compare(full,expected)
            for repeat in range(2):
                actual=ex.run();result_exact(actual,full)
                metadata=actual['metal_runtime'] if backend=='metal' else actual['cuda_runtime']['dag_execution']
                baseline=full['metal_runtime'] if backend=='metal' else full['cuda_runtime']['dag_execution']
                assert metadata['readback_bytes']<baseline['readback_bytes']
                reports.append(dict(mode=mode,repeat=repeat,full=baseline,reduced=metadata))
            save(tmp_path/(mode+'-readback-results.npz'),actual,expected)
    (tmp_path/'readback-report.json').write_text(json.dumps(reports,indent=2)+'\n')
