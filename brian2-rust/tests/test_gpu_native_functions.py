"""Native-only scalar Functions: source identity, real calls and fail-closed ABI."""
from copy import deepcopy
import hashlib
import os
import json
from types import SimpleNamespace

import brian2 as b
import numpy as np
import pytest

from brian2_rust.cuda import CudaExecutor, build_cuda_plan
from brian2_rust.metal import MetalExecutor, build_metal_plan
from brian2_rust.plan import PlanValidationError
from brian2_rust.protocol import attach_protocol
from test_gpu_expression_contract import base, portable, binary, load, lit, statement, oracle
from test_gpu_refractory import refresh_code
from test_gpu_spike_generator import execute as _execute
from test_metal_delays import device
from test_metal_plasticity import equivalent
from test_gpu_monitors import setup, DT

GPU = [pytest.param('metal', marks=pytest.mark.skipif(os.environ.get('B2_TEST_METAL') != '1', reason='requires Apple GPU')),
       pytest.param('cuda', marks=pytest.mark.skipif(os.environ.get('B2_TEST_CUDA') != '1', reason='requires NVIDIA GPU'))]


def execute(model, path, backend, route='scan'):
    (path.parent/(path.name+'-model.json')).write_text(json.dumps(model,sort_keys=True)+'\n')
    return _execute(model,path,backend,route)


def descriptor(backend, source, symbol='curve'):
    return dict(abi={'metal':'b2ir-metal-v1','cuda':'b2ir-cuda-device-v1'}[backend],
                symbol=symbol, source=source, source_sha256=hashlib.sha256(source.encode()).hexdigest())


def fixture(path, dag=False):
    model, code = base(path, count=9, dag=dag)
    portable(model, 'curve', binary('add', binary('mul', load('arg'), load('arg')), lit(.5)))
    code['vector'] = [statement('y', dict(op='call', function='curve', arguments=[load('x')]))]
    refresh_code(model, code)
    reference = deepcopy(model)
    function = model['definition']['functions'][0]
    function['body'] = None; function['implementations'] = {}
    for backend in ('metal','cuda'):
        qualifier = '__device__ ' if backend == 'cuda' else ''
        # Identical helper names in different contracts must remain private.
        source = f'{qualifier}float helper(float x) {{ return x*x; }}\n{qualifier}float curve(float x) {{ return helper(x)+0.5f; }}'
        function['backend_implementations'][backend] = descriptor(backend, source)
    attach_protocol(model)
    return model, reference


@pytest.mark.parametrize('backend', ['metal','cuda'])
def test_plan_native_source_identity_and_missing_backend(device,tmp_path,backend):
    model, reference = fixture(tmp_path/'ref', dag=True)
    build = build_metal_plan if backend == 'metal' else build_cuda_plan
    plan = build(model,numeric_mode='float32')
    (tmp_path/(backend+'-plan.json')).write_text(plan.to_json())
    native = model['definition']['functions'][0]['backend_implementations'][backend]
    assert all(native['source'] in k.source for k in plan.kernels)
    assert all('b2_invoke' in k.source for k in plan.kernels)
    assert plan.definition_sha256 == model['protocol']['layers']['definition']
    changed = deepcopy(model)
    changed['definition']['functions'][0]['backend_implementations'][backend] = descriptor(backend,native['source']+'\n// changed source identity\n')
    attach_protocol(changed)
    assert build(changed,numeric_mode='float32').sha256 != plan.sha256
    bad = deepcopy(model);bad['definition']['functions'][0]['backend_implementations'][backend]['source'] += ' '
    attach_protocol(bad)
    with pytest.raises(PlanValidationError): build(bad,numeric_mode='float32')
    del model['definition']['functions'][0]['backend_implementations'][backend];attach_protocol(model)
    with pytest.raises(PlanValidationError,match=backend): build(model,numeric_mode='float32')
    # Portable bodies take precedence and do not require a native descriptor.
    before = build(reference,numeric_mode='float32')
    reference['definition']['functions'][0]['backend_implementations'][backend] = descriptor(backend,'invalid unused native source')
    attach_protocol(reference)
    after = build(reference,numeric_mode='float32')
    assert [k.source for k in before.kernels] == [k.source for k in after.kernels]


def test_native_gpu_does_not_claim_a_cpu_f32_mirror(device,tmp_path):
    model, _ = fixture(tmp_path/'ref')
    control = SimpleNamespace(model=model)
    with pytest.raises(PlanValidationError,match='no CPU mirror'):
        CudaExecutor._cpu_control(control,512*1024**2,1)
    with pytest.raises(PlanValidationError,match='no CPU mirror'):
        MetalExecutor._cpu_mirror(control)


@pytest.mark.parametrize('backend',GPU)
@pytest.mark.parametrize('dag',[False,True])
def test_native_population_matches_independent_portable_reference(device,tmp_path,backend,dag):
    model, reference = fixture(tmp_path/'ref',dag)
    result = execute(model,tmp_path/backend,backend)
    expected = oracle(reference,tmp_path/'oracle')
    equivalent(result,expected,exact=True)
    from test_gpu_workgroup import save
    save(tmp_path/'native-population-results.npz',result,expected)
    np.testing.assert_array_equal(result['populations'][0]['states']['y'],np.linspace(-1,1,9)**2+.5)


@pytest.mark.parametrize('backend',GPU)
@pytest.mark.parametrize('fault',['signature','nonfinite','syntax'])
def test_native_invalid_signature_source_and_result_fail(device,tmp_path,backend,fault):
    model, _ = fixture(tmp_path/'ref')
    qualifier = '__device__ ' if backend == 'cuda' else ''
    source = {'signature':f'{qualifier}float curve(int x) {{ return float(x); }}',
              'nonfinite':f'{qualifier}float curve(float x) {{ return INFINITY; }}',
              'syntax':'not valid GPU source'}[fault]
    model['definition']['functions'][0]['backend_implementations'][backend] = descriptor(backend,source)
    attach_protocol(model)
    with pytest.raises(FloatingPointError if fault == 'nonfinite' else RuntimeError,
                       match='signature must match' if fault == 'signature' else None):
        execute(model,tmp_path/backend,backend)
    assert not (tmp_path/backend/'transport/summary.json').exists()


@b.implementation('atlasir-metal-v1', 'long typed_step(long x, bool active) { return active ? x+1 : x; }', name='typed_step')
@b.implementation('atlasir-cuda-device-v1', '__device__ long typed_step(long x, bool active) { return active ? x+1 : x; }', name='typed_step')
@b.check_units(x=1, active=1, result=1)
@b.declare_types(x='integer', active='boolean', result='integer')
def typed_step(x, active):
    if active:
        return x+1
    return x


@b.implementation('atlasir-metal-v1', 'bool typed_odd(long x) { return (x%2)!=0; }', name='typed_odd')
@b.implementation('atlasir-cuda-device-v1', '__device__ bool typed_odd(long x) { return (x%2)!=0; }', name='typed_odd')
@b.check_units(x=1, result=bool)
@b.declare_types(x='integer', result='boolean')
def typed_odd(x):
    if x%2:
        return True
    return False


@pytest.mark.parametrize('backend',GPU)
@pytest.mark.parametrize('queued',[False,True])
def test_native_typed_frontend_device_continuation_restore_and_queued(device,tmp_path,backend,queued):
    setup(tmp_path/backend,backend,build_on_run=not queued,numeric_mode='float32')
    p = b.NeuronGroup(4,'counter:integer\nenabled:boolean',dtype={'counter':np.int64},dt=DT,
                      namespace={'typed_step':typed_step,'typed_odd':typed_odd})
    start = np.array([2**54,2**54+1,-2**54,-2**54+1],np.int64)
    p.counter = start;p.enabled = True
    p.run_regularly('counter=typed_step(counter, enabled)\nenabled=typed_odd(counter)',when='groups')
    monitor = b.StateMonitor(p,['counter','enabled'],record=True)
    net = b.Network(p,monitor)
    net.run(2*DT)
    if not queued:net.store('middle')
    net.run(2*DT)
    if queued:device.build()
    else:
        saved = np.asarray(p.counter[:]).copy();net.restore('middle');net.run(2*DT)
        np.testing.assert_array_equal(p.counter[:],saved)
    expected=start.copy();active=np.ones(4,bool);traces=[]
    for _ in range(4):
        traces.append(expected.copy());expected += active;active=(expected%2)!=0
    np.testing.assert_array_equal(p.counter[:],expected)
    np.testing.assert_array_equal(p.enabled[:],active)
    np.testing.assert_array_equal(monitor.counter[:],np.array(traces).T)
    np.savez_compressed(tmp_path/'typed-results.npz',counter=np.asarray(p.counter[:]),
        enabled=np.asarray(p.enabled[:]),trace=np.asarray(monitor.counter[:]),
        expected=expected,expected_trace=np.array(traces).T,initial=start)


@pytest.mark.parametrize('backend',GPU)
@pytest.mark.parametrize('route',['scan','sparse'])
def test_native_delayed_synapse_function_and_namespace_isolation(device,tmp_path,backend,route):
    from brian2_rust.export import lower_network
    setup(tmp_path/'ref')
    p=b.NeuronGroup(4,'v:1',threshold='v>=0',reset='',dt=DT)
    syn=b.Synapses(p,p,'w:1',on_pre='w+=0.125;v_post+=w',clock=p.clock)
    syn.connect(i=[0,1,2,3],j=[1,2,3,0]);syn.w=.25;syn.delay=np.array([0,1,2,1])*DT
    model=lower_network(b.Network(p,syn),6*DT)
    portable(model,'curve',binary('add',load('arg'),lit(.125)))
    portable(model,'other',binary('mul',load('arg'),lit(.5)))
    code=next(c for c in model['definition']['synapses'][0]['code_objects'] if c['kind']=='synapses')
    code['vector'][0]['value']=dict(op='call',function='curve',arguments=[load('w')])
    code['vector'][-1]['value']=binary('add',load('v_post'),dict(op='call',function='other',arguments=[load('w')]))
    refresh_code(model,code);reference=deepcopy(model)
    for function,operation in zip(model['definition']['functions'],('x+0.125f','x*0.5f'),strict=True):
        function['body']=None;function['implementations']={}
        qualifier='__device__ ' if backend=='cuda' else ''
        # Both sources define helper, but their namespaces and functions differ.
        source=f'{qualifier}float helper(float x) {{ return {operation}; }}\n{qualifier}float {function["name"]}(float x) {{ return helper(x); }}'
        function['backend_implementations'][backend]=descriptor(backend,source,function['name'])
    attach_protocol(model)
    equivalent(execute(model,tmp_path/backend,backend,route),oracle(reference,tmp_path/'oracle'),exact=True)


@pytest.mark.parametrize('backend',GPU)
def test_native_false_mask_suppresses_call_and_overwrite_cannot_hide_fault(device,tmp_path,backend):
    from test_gpu_refractory import make_model
    model=make_model(device,tmp_path,'masked')
    code=next(c for c in model['definition']['populations'][0]['code_objects'] if c['kind']=='state_update')
    target=next(s for s in code['vector'] if s['target']=='v')
    assert target['condition']=='not_refractory'
    portable(model,'curve',load('arg'))
    function=model['definition']['functions'][0]
    function['body']=None;function['implementations']={}
    qualifier='__device__ ' if backend=='cuda' else ''
    function['backend_implementations'][backend]=descriptor(backend,f'{qualifier}float curve(float x) {{ return INFINITY; }}')
    target['value']=dict(op='call',function='curve',arguments=[lit(0)])
    refresh_code(model,code)
    # Every selected lane is refractory: an invalid return must never execute.
    result=execute(model,tmp_path/'masked',backend)
    assert np.isfinite(result['populations'][0]['states']['v']).all()
    active, c=base(tmp_path/'active-ref')
    active['definition']['functions']=[deepcopy(function)]
    c['vector']=[statement('y',dict(op='call',function='curve',arguments=[lit(0)])),statement('y',lit(0))]
    refresh_code(active,c)
    with pytest.raises(FloatingPointError):execute(active,tmp_path/'overwritten',backend)
