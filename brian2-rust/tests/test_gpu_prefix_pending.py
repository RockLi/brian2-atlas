"""Single-event pending prefixes preserve per-edge chronology and target order."""
from copy import deepcopy
from types import SimpleNamespace
import json
import numpy as np
import pytest
from brian2_rust import gpu_prefix_pending as pending
from brian2_rust.gpu_synapse_prefix import prefix_length
from brian2_rust.protocol import attach_protocol
from brian2_rust.metal import MetalExecutor,build_metal_plan
from brian2_rust.cuda import CudaExecutor
from brian2_rust.metal_dag import run_dag,_prepare_dag_storage
from test_metal_delays import device
from test_gpu_prefix_locals import parts,typed_locals
from test_gpu_synapse_prefix import model_at,has_prefix
from test_gpu_spike_generator import BACKENDS
from test_gpu_synapse_parallel import control
from test_cuda_graphs import result_exact
from test_gpu_workgroup import save


def pending_model(path,uniform=False):
    model=typed_locals(model_at(path,uniform=uniform,temporary_tail=True))
    syn,code,route=parts(model)
    if uniform:route['pending']=[dict(delivery_tick=0,item=0),dict(delivery_tick=1,item=7)]
    else:route['pending']=[dict(delivery_tick=0,item=1),dict(delivery_tick=1,item=2),dict(delivery_tick=2,item=3)]
    attach_protocol(model);return model


def split_control(model,path):
    path.mkdir();ex=SimpleNamespace(model=model,plan=build_metal_plan(model,numeric_mode='float32',event_delivery='sparse',synapse_prefix=True),directory=path,compile_seconds=0,device_name='CPU f32')
    return run_dag(ex,max_buffer_bytes=512*1024**2,compute='cpu-f32',workers=3)


@pytest.mark.parametrize('uniform',[False,True])
def test_pending_bitmap_matches_expansion_and_keeps_one_new_binding(device,tmp_path,uniform):
    model=pending_model(tmp_path/'ref',uniform);syn,code,path=parts(model)
    spec=pending.pending_spec(model,syn,code,path);assert spec['span']>0
    mask=pending.pending_mask(spec,512*1024**2)
    inst=model['instance']['synapses'][0]
    actual={(edge,t) for t in range(spec['span']) for edge in range(spec['edges']) if mask[(t//32)*spec['edges']+edge] & (1<<(t%32))}
    expected={(edge,e['delivery_tick']) for e in path['pending'] for edge,source in enumerate(inst['source']) if (source if uniform else edge)==e['item']}
    assert actual==expected
    before=build_metal_plan(model,numeric_mode='float32',event_delivery='sparse')
    after=build_metal_plan(model,numeric_mode='float32',event_delivery='sparse',synapse_prefix=True)
    assert after.logical==before.logical and after.buffers[:-1]==before.buffers
    assert after.buffers[-1].endswith('/prefix_pending_mask') and has_prefix(after)
    d=next(d for d in after.dispatches if d.role=='edge-synapse-prefix')
    assert d.bindings[-1]==len(after.buffers)-1 and d.types[-1]=='const uint'
    a,b=(_prepare_dag_storage(SimpleNamespace(model=model,plan=p),512*1024**2)[0] for p in (before,after))
    for x,y in zip(a,b[:-1],strict=True):np.testing.assert_array_equal(x,y)
    np.testing.assert_array_equal(b[-1],mask)
    (tmp_path/'pending-prefix-model.json').write_text(json.dumps(model)+'\n')
    (tmp_path/'pending-prefix-plan.json').write_text(after.to_json())
    np.savez_compressed(tmp_path/'pending-prefix-mask.npz',actual=b[-1],reference=mask)


@pytest.mark.parametrize('hazard',['duplicate','overlap','overdue','mask-limit','expansion-limit'])
def test_proof_declines_ambiguous_or_oversized_pending(device,tmp_path,monkeypatch,hazard):
    model=pending_model(tmp_path/'ref');syn,code,path=parts(model)
    if hazard=='duplicate':path['pending'].append(deepcopy(path['pending'][0]))
    elif hazard=='overlap':path['pending'][0]['delivery_tick']=path['delay_ticks'][path['pending'][0]['item']]
    elif hazard=='overdue':path['pending'][0]['delivery_tick']=-1
    elif hazard=='mask-limit':monkeypatch.setattr(pending,'MAX_MASK_BYTES',1)
    else:monkeypatch.setattr(pending,'MAX_EXPANDED_EVENTS',1)
    assert pending.pending_spec(model,syn,code,path) is None
    assert prefix_length(model,syn,code,path)==0


def test_mask_word_boundaries_and_preallocation_budget(device,tmp_path,monkeypatch):
    model=pending_model(tmp_path/'ref');syn,code,path=parts(model)
    model['run']['clocks'][code['clock']]['steps']=96
    path['delay_ticks']=[65]*len(model['instance']['synapses'][0]['source'])
    path['pending']=[dict(delivery_tick=t,item=0) for t in (0,31,32,63,64)]
    spec=pending.pending_spec(model,syn,code,path);assert spec['span']==65 and spec['words']==3
    mask=pending.pending_mask(spec,512*1024**2).reshape(3,-1)
    edges=np.flatnonzero(np.asarray(model['instance']['synapses'][0]['source'])==0)
    np.testing.assert_array_equal(mask[:,edges],np.broadcast_to(np.asarray([0x80000001,0x80000001,1],np.uint32)[:,None],(3,len(edges))))
    with monkeypatch.context() as patch:
        patch.setattr(np,'zeros',lambda *a,**kw:pytest.fail('Allocated mask before budget check'))
        with pytest.raises(MemoryError,match='prefix pending mask'):pending.pending_mask(spec,mask.nbytes-1)


@pytest.mark.parametrize('backend',BACKENDS)
@pytest.mark.parametrize('uniform',[False,True])
def test_pending_prefix_native_results_and_replay(device,tmp_path,backend,uniform):
    model=pending_model(tmp_path/'ref',uniform);expected=control(model,tmp_path/'baseline',3)
    if backend=='cpu-f32':actual=split_control(model,tmp_path/'prefix')
    else:
        cls=MetalExecutor if backend=='metal' else CudaExecutor
        with cls(model,tmp_path/'gpu',numeric_mode='float32',event_delivery='sparse',synapse_prefix=True) as ex:
            assert has_prefix(ex.plan)
            for _ in range(2):actual=ex.run();result_exact(actual,expected)
            result_exact(ex.run(dag_execution='workgroup'),expected)
    result_exact(actual,expected)
    assert actual['synapses'][0]['events']==expected['synapses'][0]['events']
    save(tmp_path/'pending-prefix-results.npz',actual,expected)


def test_overlapping_pending_and_current_events_have_a_counterexample(device,tmp_path,monkeypatch):
    model=pending_model(tmp_path/'ref');syn,code,path=parts(model)
    path['pending']=[dict(delivery_tick=0,item=0)];attach_protocol(model)
    assert pending.pending_spec(model,syn,code,path) is None
    expected=control(model,tmp_path/'baseline',3)
    result_exact(split_control(model,tmp_path/'safe'),expected)
    original=pending.pending_spec
    def unsafe(m,s,c,p):
        if p is path:return dict(edges=len(m['instance']['synapses'][0]['source']),span=1,words=1,bits={0:1})
        if p['name']==path['name']:return dict(edges=len(m['instance']['synapses'][0]['source']),span=1,words=1,bits={0:1})
        return original(m,s,c,p)
    with monkeypatch.context() as patch:
        patch.setattr(pending,'pending_spec',unsafe)
        actual=split_control(model,tmp_path/'unsafe')
    safe=expected['synapses'][0]['states']['hits'];bad=actual['synapses'][0]['states']['hits']
    assert not np.array_equal(safe,bad)
    np.savez_compressed(tmp_path/'pending-prefix-counterexample.npz',safe_hits=safe,unsafe_hits=bad)
