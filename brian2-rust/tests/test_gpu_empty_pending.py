"""Activation-entry pending specialization retains live delays and continuation."""
from contextlib import contextmanager
from copy import deepcopy
from types import SimpleNamespace
import json
import numpy as np
import pytest
import brian2 as b
from brian2_rust import metal_synapses
from brian2_rust.protocol import attach_protocol
from brian2_rust.metal import build_metal_plan,MetalExecutor
from brian2_rust.cuda import CudaExecutor,build_cuda_plan
from brian2_rust.metal_dag import _prepare_dag_storage
from brian2_rust.plan import PlanValidationError
from test_metal_delays import device,ROOT
from test_gpu_spike_generator import BACKENDS
from test_gpu_synapse_parallel import control
from test_gpu_synapse_fusion import model_at as owned_model,network,DT
from test_gpu_target_pathway import model_at as target_model
from test_cuda_graphs import result_exact
from test_gpu_workgroup import save


@contextmanager
def original_pending():
    previous=metal_synapses.pending_code
    metal_synapses.pending_code=lambda path,source:source
    try:yield
    finally:metal_synapses.pending_code=previous


def model_at(path,canonical=False,pending=False):
    model=target_model(path,recurrent=True,hazard=True) if canonical else owned_model(path)
    for pathway in model['instance']['synapses'][0]['pathways']:
        pathway['pending']=([dict(delivery_tick=t,item=i) for t,i in ((0,0),(0,0),(2,7))] if pending else [])
    attach_protocol(model);return model


@pytest.mark.parametrize('canonical',[False,True])
@pytest.mark.parametrize('pending',[False,True])
def test_plan_specializes_only_empty_entry_queue(device,tmp_path,canonical,pending):
    model=model_at(tmp_path/'ref',canonical,pending)
    with original_pending():before=build_metal_plan(model,numeric_mode='float32')
    after=build_metal_plan(model,numeric_mode='float32')
    assert before.logical==after.logical and before.buffers==after.buffers and before.dispatches==after.dispatches
    if pending:assert before==after
    else:
        assert before!=after
        assert not any('pending_ticks[' in k.source or 'cursor[' in k.source for k in after.kernels)
        assert any('pending_ticks[' in k.source for k in before.kernels)
    for a,e in zip(*[_prepare_dag_storage(SimpleNamespace(model=model,plan=p),512*1024**2)[0] for p in (before,after)],strict=True):
        np.testing.assert_array_equal(a,e)
    for label,p in [('baseline',before),('specialized',after)]:
        (tmp_path/f'pending-{label}-plan.json').write_text(p.to_json())
    (tmp_path/'pending-model.json').write_text(json.dumps(model)+'\n')


@pytest.mark.parametrize('backend',BACKENDS)
@pytest.mark.parametrize('canonical',[False,True])
@pytest.mark.parametrize('pending',[False,True])
def test_delayed_events_duplicates_and_replay(device,tmp_path,backend,canonical,pending):
    model=model_at(tmp_path/'ref',canonical,pending)
    with original_pending():expected=control(model,tmp_path/'baseline',1)
    if backend=='cpu-f32':
        actual=control(model,tmp_path/'specialized',3);result_exact(actual,expected)
    else:
        cls=MetalExecutor if backend=='metal' else CudaExecutor
        with cls(model,tmp_path/'gpu',numeric_mode='float32',event_delivery='sparse') as ex:
            for _ in range(2):actual=ex.run();result_exact(actual,expected)
            result_exact(ex.run(dag_execution='workgroup'),expected)
    assert actual['synapses'][0]['events']==expected['synapses'][0]['events']>0
    save(tmp_path/'pending-results.npz',actual,expected)


@pytest.mark.parametrize('backend',BACKENDS[1:])
def test_pending_instance_cannot_reuse_empty_plan(device,tmp_path,backend):
    build,executor=(build_metal_plan,MetalExecutor) if backend=='metal' else (build_cuda_plan,CudaExecutor)
    model=model_at(tmp_path/'ref');plan=build(model,numeric_mode='float32')
    other=deepcopy(model);other['instance']['synapses'][0]['pathways'][0]['pending']=[dict(delivery_tick=0,item=0)]
    attach_protocol(other)
    with pytest.raises(PlanValidationError):executor(other,tmp_path/'stale',numeric_mode='float32',plan=plan)


@pytest.mark.parametrize('backend',BACKENDS[1:])
def test_device_empty_pending_empty_restore_and_compile_reuse(device,tmp_path,backend):
    records=[]
    for old in (True,False):
        device.reinit()
        b.set_device('rust_standalone',engine=backend,numeric_mode='float32',event_delivery='sparse',
            gpu_buffer_reuse=True,gpu_compile_reuse=True,directory=tmp_path/str(old),runner=ROOT/'target/release/b2-runner')
        net,pop,syn,spikes,monitor=network();net.store('initial');states=[];patterns=[]
        def run():
            net.run(8*DT)
            states.append([np.asarray(x).copy() for x in (pop.v[:],syn.w[:],syn.k[:],spikes.t[:],spikes.i[:],monitor.v[:])])
            patterns.append(any('pending_ticks[' in k.source for k in device._gpu_executor.plan.kernels))
        from contextlib import nullcontext
        with original_pending() if old else nullcontext():
            run();run();net.restore('initial');run()
        assert patterns==([True,True,True] if old else [False,True,False])
        device.close_gpu();records.append(states)
    for a,e in zip(*records,strict=True):
        for x,y in zip(a,e,strict=True):np.testing.assert_array_equal(x,y)
    np.savez_compressed(tmp_path/'pending-device.npz',**{f'{mode}/{i}/{j}':v for mode,states in zip(('reference','actual'),records) for i,values in enumerate(states) for j,v in enumerate(values)})
