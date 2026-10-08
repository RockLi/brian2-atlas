"""Local liveness at an edge-prefix/ordered-target stage boundary."""
from copy import deepcopy
from types import SimpleNamespace
import json
import numpy as np
import pytest
from brian2_rust.gpu_synapse_prefix import prefix_length,nodes
from brian2_rust.protocol import attach_protocol
from brian2_rust.metal import build_metal_plan,MetalExecutor
from brian2_rust.cuda import CudaExecutor
from brian2_rust.metal_dag import run_dag
from gpu_prefix_locals_compare import with_prefix_local
from test_metal_delays import device
from test_gpu_synapse_prefix import model_at,has_prefix
from test_gpu_spike_generator import BACKENDS
from test_gpu_synapse_parallel import control
from test_gpu_refractory import refresh_code
from test_cuda_graphs import result_exact
from test_gpu_workgroup import save


def parts(model):
    syn=model['definition']['synapses'][0]
    code=next(c for c in syn['code_objects'] if c['kind']=='synapses')
    path=next(p for p in model['instance']['synapses'][0]['pathways'] if p['name']==code['pathway_name'])
    return syn,code,path


def typed_locals(model):
    model=with_prefix_local(model);syn,code,path=parts(model)
    index=next(i for i,s in enumerate(code['vector']) if s['target']=='hits')
    statement=code['vector'][index]
    cached={**deepcopy(statement),'target':'_cached_hits'}
    statement['value']={'op':'load','name':'_cached_hits'}
    code['vector'].insert(index,cached)
    flag=dict(target='_prefix_mask',dtype='bool',dimensions=[0.]*7,condition=None,value=dict(op='boolean',value=True))
    code['vector'].insert(index,flag)
    statement['value']=dict(op='mul',left=statement['value'],right=dict(op='cast',dtype='i64',arg=dict(op='bool_to_f64',arg=dict(op='load',name='_prefix_mask'))))
    refresh_code(model,code)
    return model


@pytest.mark.parametrize('hazard',['value','condition','endpoint','pending'])
def test_live_locals_endpoint_reads_and_pending_do_not_cross(device,tmp_path,hazard):
    model=with_prefix_local(model_at(tmp_path/'ref'));syn,code,path=parts(model)
    boundary=next(i for i,s in enumerate(code['vector']) if s['target'] in syn['post_state_aliases'])
    if hazard=='value':code['vector'][boundary]['value']={'op':'load','name':'_b2_prefix_cached_value'}
    elif hazard=='condition':
        code['vector'].insert(0,dict(target='_live_mask',dtype='bool',dimensions=[0.]*7,condition=None,
            value=dict(op='boolean',value=True)))
        code['vector'][boundary+1]['condition']='_live_mask'
    elif hazard=='endpoint':code['vector'][0]['value']={'op':'load','name':'v_post'}
    else:path['pending']=[dict(delivery_tick=0,item=0)]
    refresh_code(model,code)
    assert prefix_length(model,syn,code,path)==0
    if hazard=='condition':
        # Frozen B2IR only permits its declared refractory guards. The raw
        # liveness check also stays conservative for this unsupported guard.
        from brian2_rust.plan import PlanValidationError
        with pytest.raises(PlanValidationError,match='conditional write does not match refractory definition'):
            build_metal_plan(model,numeric_mode='float32',synapse_prefix=True)
    else:assert not has_prefix(build_metal_plan(model,numeric_mode='float32',synapse_prefix=True))


def test_longest_safe_boundary_can_stop_before_live_local(device,tmp_path):
    model=model_at(tmp_path/'ref');syn,code,path=parts(model)
    original=prefix_length(model,syn,code,path);assert original>0
    boundary=next(i for i,s in enumerate(code['vector']) if s['target'] in syn['post_state_aliases'])
    local={**deepcopy(code['vector'][0]),'target':'_live_local'}
    code['vector'].insert(boundary,local)
    code['vector'][boundary+1]['value']={'op':'load','name':'_live_local'}
    refresh_code(model,code)
    assert prefix_length(model,syn,code,path)==original


@pytest.mark.parametrize('backend',BACKENDS)
@pytest.mark.parametrize('flavor',['float','typed'])
def test_prefix_locals_all_results_without_scratch_storage(device,tmp_path,backend,flavor):
    original=model_at(tmp_path/'ref',temporary_tail=True)
    model=typed_locals(original) if flavor=='typed' else with_prefix_local(original)
    expected=control(original,tmp_path/'original',3)
    result_exact(control(model,tmp_path/'unfused',3),expected)
    plan=build_metal_plan(model,numeric_mode='float32',event_delivery='sparse',synapse_prefix=True)
    assert has_prefix(plan)
    before=build_metal_plan(model,numeric_mode='float32',event_delivery='sparse')
    assert plan.buffers==before.buffers and plan.logical==before.logical
    assert len(plan.dispatches)==len(before.dispatches)+1
    syn,code,path=parts(model);size=prefix_length(model,syn,code,path)
    assert '_b2_prefix_cached_value' in {s['target'] for s in code['vector'][:size]}
    if flavor=='typed':assert {'_cached_hits','_prefix_mask'}<={s['target'] for s in code['vector'][:size]}
    if backend=='cpu-f32':
        directory=tmp_path/'split';directory.mkdir()
        ex=SimpleNamespace(model=model,plan=plan,directory=directory,compile_seconds=0,device_name='CPU f32')
        actual=run_dag(ex,max_buffer_bytes=512*1024**2,compute='cpu-f32',workers=3)
    else:
        cls=MetalExecutor if backend=='metal' else CudaExecutor
        with cls(model,tmp_path/'gpu',numeric_mode='float32',event_delivery='sparse',synapse_prefix=True) as ex:
            for _ in range(2):actual=ex.run();result_exact(actual,expected)
            result_exact(ex.run(dag_execution='workgroup'),expected)
    result_exact(actual,expected)
    assert actual['synapses'][0]['events']==expected['synapses'][0]['events']
    save(tmp_path/'locals-results.npz',actual,expected)
    (tmp_path/'locals-model.json').write_text(json.dumps(model)+'\n')
    (tmp_path/'locals-plan.json').write_text(plan.to_json())
