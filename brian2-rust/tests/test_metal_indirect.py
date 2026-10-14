"""Indirect replay keeps stage barriers, exact state resets and resource ownership."""
import json
from dataclasses import replace
import numpy as np
import pytest
import brian2 as b
from brian2_rust.metal import MetalExecutor
from brian2_rust.export import lower_network
from test_metal import real_metal
from test_metal_delays import device,ROOT
from test_cuda_graphs import result_exact
from test_gpu_workgroup import save
from test_gpu_summed_parallel import model_at,control
from test_gpu_native_workgroup import native_model
from test_gpu_expression_contract import oracle
from test_gpu_custom_events import compare


@real_metal
@pytest.mark.parametrize('flavor',['plain','random','timed','native','composed'])
@pytest.mark.parametrize('route',['scan','sparse'])
def test_indirect_replay_and_mode_changes(device,tmp_path,flavor,route):
    if flavor=='native':model,reference=native_model(tmp_path/'ref');expected=oracle(reference,tmp_path/'oracle')
    elif flavor=='composed':
        from test_gpu_composed_models import setup,network,DT
        setup(tmp_path/'ref');net,*_=network(7);model=lower_network(net,24*DT);expected=oracle(model,tmp_path/'oracle')
    else:model=model_at(tmp_path/'ref',edges=273,steps=8,flavor=flavor);expected=control(model,tmp_path/'control')
    (tmp_path/'model.json').write_text(json.dumps(model,sort_keys=True)+'\n')
    with MetalExecutor(model,tmp_path/'metal',numeric_mode='float32',event_delivery=route) as ex:
        initial=ex.run();compare(initial,expected)
        first=ex.run(dag_execution='indirect');result_exact(first,initial)
        meta=first['metal_runtime'];assert meta['indirect_commands_encoded']==meta['dispatches']>0
        assert not meta['indirect_commands_reused'] and meta['indirect_bytes']>0
        for _ in range(2):
            actual=ex.run(dag_execution='indirect');result_exact(actual,initial)
            m=actual['metal_runtime'];assert m['indirect_commands_reused'] and m['indirect_commands_encoded']==0
            assert m['dispatches']==m['explicit_barriers'] and m['resident_bytes']==m['total_buffer_bytes']+m['indirect_bytes']
        save(tmp_path/'indirect-results.npz',actual,expected)
        for mode in ('resident','direct'):
            result_exact(ex.run(dag_execution=mode),initial)
            fresh=ex.run(dag_execution='indirect');result_exact(fresh,initial)
            assert not fresh['metal_runtime']['indirect_commands_reused']
    assert ex._resident_dag_bytes==0 and not ex.handles


@real_metal
def test_indirect_reduced_budget_and_synchronization_release(device,tmp_path):
    model=model_at(tmp_path/'ref',edges=0)
    with MetalExecutor(model,tmp_path/'metal',numeric_mode='float32') as ex:
        first=ex.run(dag_execution='indirect');m=first['metal_runtime']
        with pytest.raises(RuntimeError,match='memory budget'):ex.run(dag_execution='indirect',max_buffer_bytes=m['resident_bytes']-1)
        assert ex._resident_dag_bytes==0
        result_exact(ex.run(dag_execution='indirect'),first)
        with pytest.raises(ValueError,match='explicit synchronization'):ex.run(dag_execution='indirect',dag_synchronization='tracked')
        assert ex._resident_dag_bytes==0
        result_exact(ex.run(dag_execution='direct'),first)
        original=ex.plan
        from brian2_rust.metal_buffers import MAX_INDIRECT_DISPATCHES
        steps=MAX_INDIRECT_DISPATCHES//len(original.dispatches)+1
        ex.plan=replace(original,logical=replace(original.logical,clocks=tuple(replace(c,steps=steps) for c in original.logical.clocks)))
        with pytest.raises(ValueError,match=str(MAX_INDIRECT_DISPATCHES)):ex.run(dag_execution='indirect')
        ex.plan=original
        result_exact(ex.run(dag_execution='indirect'),first)


@real_metal
def test_indirect_numeric_fault_rejected(device,tmp_path):
    from test_gpu_native_functions import descriptor
    from brian2_rust.protocol import attach_protocol
    model,_=native_model(tmp_path/'ref')
    model['definition']['functions'][0]['backend_implementations']['metal']=descriptor('metal','float curve(float x) { return INFINITY; }')
    attach_protocol(model)
    with MetalExecutor(model,tmp_path/'metal',numeric_mode='float32') as ex:
        for _ in range(2):
            with pytest.raises(FloatingPointError):ex.run(dag_execution='indirect')
    assert not ex.handles and ex._resident_dag_bytes==0


@real_metal
def test_indirect_device_continuation_restore_and_pipeline_reuse(device,tmp_path):
    from test_gpu_multiclock import network,DT
    records=[]
    for mode in ('direct','indirect'):
        device.reinit()
        b.set_device('rust_standalone',engine='metal',numeric_mode='float32',event_delivery='sparse',directory=tmp_path/mode,
            runner=ROOT/'target/release/b2-runner',gpu_compile_reuse=True,gpu_buffer_reuse=True,metal_dag_execution=mode)
        net,pre,post,syn,monitors,spikes=network(True);states=[]
        for i in range(3):
            if i==1:net.store('saved')
            if i==2:net.restore('saved')
            net.run(6*DT)
            states.append([np.asarray(x).copy() for x in [pre.v[:],post.v[:],syn.w[:],syn.x[:],*[m.v[:] for m in monitors],*[s.t[:] for s in spikes],*[s.i[:] for s in spikes]]])
        records.append(states);ex=device._gpu_executor;device.close_gpu();assert ex._resident_dag_bytes==0 and not ex.handles
    for a,e in zip(*records):
        for x,y in zip(a,e):np.testing.assert_array_equal(x,y)
    np.savez_compressed(tmp_path/'indirect-device-results.npz',**{f'{mode}/{i}/{j}':value for mode,states in zip(('reference','actual'),records) for i,values in enumerate(states) for j,value in enumerate(values)})
