"""Native scalar Functions survive workgroup source packing and exact replay."""
from types import SimpleNamespace
import json
import brian2 as b
import numpy as np
import pytest
from brian2_rust.cuda import CudaExecutor,build_cuda_plan
from brian2_rust.metal import MetalExecutor,build_metal_plan
from brian2_rust.metal_dag import _prepare_dag_storage
from brian2_rust.gpu_workgroup import prepare
from brian2_rust.protocol import attach_protocol
from test_metal_delays import device,ROOT
from test_gpu_native_functions import GPU,fixture,descriptor,typed_step,typed_odd
from test_gpu_expression_contract import oracle
from test_gpu_custom_events import compare
from test_cuda_graphs import result_exact
from test_gpu_workgroup import save
from test_gpu_regular_clock import setup,DT
from brian2_rust.export import lower_network


def native_model(path):
    model,reference=fixture(path,dag=True)
    # These strings are valid comment text in a user's self-contained source;
    # workgroup transforms and builtin detection must never process them.
    for backend,native in model['definition']['functions'][0]['backend_implementations'].items():
        text=native['source']+'\n// kernel void user_marker [[buffer(0)]] constant long &tick\n// __global__ threadgroup_barrier( using namespace metal;\n'
        model['definition']['functions'][0]['backend_implementations'][backend]=descriptor(backend,text)
    attach_protocol(model)
    return model,reference


@pytest.mark.parametrize('backend',['metal','cuda'])
def test_native_source_is_verbatim_inside_every_packed_stage(device,tmp_path,backend):
    model,_=native_model(tmp_path/'ref')
    plan=(build_metal_plan if backend=='metal' else build_cuda_plan)(model,numeric_mode='float32',event_delivery='sparse')
    owner=SimpleNamespace(plan=plan,model=model)
    arrays,*_=_prepare_dag_storage(owner,512*1024**2)
    program=prepare(owner,arrays,backend)
    native=model['definition']['functions'][0]['backend_implementations'][backend]['source']
    assert program.source.count(native)==len(plan.kernels)
    assert program.manifest['logical_launches']>0 and program.manifest['workgroups']==1
    (tmp_path/'workgroup-source.txt').write_text(program.source)


@pytest.mark.parametrize('backend',GPU)
@pytest.mark.parametrize('route',['scan','sparse'])
def test_native_float_workgroup_matches_portable_reference_and_other_modes(device,tmp_path,backend,route):
    model,reference=native_model(tmp_path/'ref');expected=oracle(reference,tmp_path/'oracle')
    (tmp_path/'model.json').write_text(json.dumps(model,sort_keys=True)+'\n')
    cls=MetalExecutor if backend=='metal' else CudaExecutor
    with cls(model,tmp_path/backend,numeric_mode='float32',event_delivery=route) as ex:
        original=ex.run();compare(original,expected)
        for _ in range(2):actual=ex.run(dag_execution='workgroup');result_exact(actual,original)
        compare(actual,expected)
        packed=ex._workgroup;assert packed.replays==2
        result_exact(ex.run(dag_execution='direct'),original)
        save(tmp_path/'native-workgroup-results.npz',actual,expected)
    assert packed.handle is None and packed.function is None


@pytest.mark.parametrize('backend',GPU)
def test_native_i64_bool_workgroup_with_independent_clock(device,tmp_path,backend):
    setup(tmp_path/'ref')
    pop=b.NeuronGroup(259,'counter:integer\nenabled:boolean',dtype={'counter':np.int64},dt=DT,
        namespace={'typed_step':typed_step,'typed_odd':typed_odd},name='population')
    initial=np.arange(259,dtype=np.int64)+2**54;pop.counter=initial;pop.enabled=True
    pop.run_regularly('counter=typed_step(counter,enabled); enabled=typed_odd(counter)',dt=1.5*DT,when='groups',name='update')
    monitor=b.StateMonitor(pop,['counter','enabled'],record=[0,128,258],name='monitor')
    model=lower_network(b.Network(pop,monitor),6*DT)
    (tmp_path/'model.json').write_text(json.dumps(model,sort_keys=True)+'\n')
    expected=initial.copy();active=np.ones(259,bool);trace=[];flags=[]
    for tick in range(12):
        if tick%2==0:trace.append(expected[[0,128,258]].copy());flags.append(active[[0,128,258]].copy())
        if tick%3==0:expected+=active;active=expected%2!=0
    cls=MetalExecutor if backend=='metal' else CudaExecutor
    with cls(model,tmp_path/backend,numeric_mode='float32') as ex:
        original=ex.run();actual=ex.run(dag_execution='workgroup');result_exact(actual,original)
        p=actual['populations'][0]
        np.testing.assert_array_equal(p['states']['counter'],expected)
        np.testing.assert_array_equal(p['states']['enabled'],active)
        np.testing.assert_array_equal(p['trace']['counter'],np.array(trace))
        np.testing.assert_array_equal(p['trace']['enabled'],np.array(flags))
        save(tmp_path/'native-workgroup-results.npz',actual,original)
        np.savez_compressed(tmp_path/'native-workgroup-integer-oracle.npz',initial=initial,counter=p['states']['counter'],enabled=p['states']['enabled'],trace=p['trace']['counter'],flags=p['trace']['enabled'])


@pytest.mark.parametrize('backend',GPU)
def test_native_workgroup_nonfinite_return_rejects_result_and_closes(device,tmp_path,backend):
    model,_=native_model(tmp_path/'ref');f=model['definition']['functions'][0]
    prefix='__device__ ' if backend=='cuda' else ''
    f['backend_implementations'][backend]=descriptor(backend,prefix+'float curve(float x) { return x>0.0f ? INFINITY : 0.0f; }')
    attach_protocol(model)
    cls=MetalExecutor if backend=='metal' else CudaExecutor
    with cls(model,tmp_path/backend,numeric_mode='float32') as ex:
        with pytest.raises(FloatingPointError):ex.run(dag_execution='workgroup')
        # Numeric validation happens after packed execution. It may retain
        # immutable compiled code; all writable data is fresh on each replay.
        program=ex._workgroup
        with pytest.raises(FloatingPointError):ex.run(dag_execution='direct')
    if program is not None:assert program.handle is None and program.function is None
