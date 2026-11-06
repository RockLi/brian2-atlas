"""Independent edge prefixes retain ordered target updates and pending fallback."""
from copy import deepcopy
import json
import numpy as np
import pytest
from brian2_rust.export import lower_network
from brian2_rust.protocol import attach_protocol
from brian2_rust.metal import MetalExecutor,build_metal_plan
from brian2_rust.cuda import CudaExecutor,build_cuda_plan
from test_metal_delays import device
from test_gpu_monitors import setup,DT
from test_gpu_target_pathway import network
from test_gpu_synapse_parallel import control
from test_gpu_spike_generator import BACKENDS
from test_cuda_graphs import result_exact
from test_gpu_workgroup import save


def model_at(path,**options):
    setup(path);net,*_=network(edge_count=16384,**options)
    return lower_network(net,12*DT)


def has_prefix(plan):return any(d.role=='edge-synapse-prefix' for d in plan.dispatches)


@pytest.mark.parametrize('case',['pending','endpoint-read','live-local','small','hazard','no-exp'])
def test_split_proof_falls_back_without_rejecting_the_model(device,tmp_path,case):
    model=model_at(tmp_path/'ref',recurrent=case=='hazard',hazard=case=='hazard',**({'empty':True} if case=='small' else {}))
    syn=model['definition']['synapses'][0];code=next(c for c in syn['code_objects'] if c['kind']=='synapses')
    if case=='pending':
        model['instance']['synapses'][0]['pathways'][0]['pending']=[dict(delivery_tick=0,item=0)]
    elif case=='endpoint-read':
        code['vector'][0]['value']={'op':'load','name':'v_post'}
        from test_gpu_refractory import refresh_code
        refresh_code(model,code)
    elif case=='live-local':
        # A local used after the target write cannot cross the boundary.
        removed=code['vector'][0]['target']
        code['vector'][0]['target']='local_only'
        if not any(s['target']==removed for s in code['vector']):code['effects']['writes'].remove(removed)
        index=next(i for i,s in enumerate(code['vector']) if s['target'] in syn['post_state_aliases'])
        code['vector'].insert(index+1,{**deepcopy(code['vector'][index]),'value':{'op':'load','name':'local_only'}})
        from test_gpu_refractory import refresh_code
        refresh_code(model,code)
    elif case=='no-exp':
        from test_gpu_expression_contract import lit
        for statement in code['vector'][:2]:statement['value']=lit(0)
        from test_gpu_refractory import refresh_code
        refresh_code(model,code)
    attach_protocol(model)
    for build in (build_metal_plan,build_cuda_plan):
        plan=build(model,numeric_mode='float32',synapse_prefix=True)
        assert not has_prefix(plan)
        assert plan.to_json()==build(model,numeric_mode='float32').to_json()


@pytest.mark.parametrize('backend',BACKENDS[1:])
@pytest.mark.parametrize('early',[False,True])
@pytest.mark.parametrize('route',['scan','sparse'])
def test_prefix_replay_typed_subgroups_rng_timed_input(device,tmp_path,backend,early,route):
    model=model_at(tmp_path/'ref',early=early,temporary_tail=True)
    (tmp_path/'model.json').write_text(json.dumps(model,sort_keys=True)+'\n')
    expected=control(model,tmp_path/'cpu',3)
    cls=MetalExecutor if backend=='metal' else CudaExecutor
    mode='resident' if backend=='metal' else 'chunked'
    with cls(model,tmp_path/'previous',numeric_mode='float32',event_delivery=route,dag_execution=mode) as baseline:
        assert not has_prefix(baseline.plan)
        previous=baseline.run();result_exact(previous,expected)
    with cls(model,tmp_path/'prefix',numeric_mode='float32',event_delivery=route,dag_execution=mode,synapse_prefix=True) as ex:
        assert has_prefix(ex.plan)
        for _ in range(2):
            actual=ex.run();result_exact(actual,previous)
        save(tmp_path/'prefix-results.npz',actual,previous)
        assert ex.plan.buffers==baseline.plan.buffers
        assert ex.plan.logical==baseline.plan.logical
        (tmp_path/'prefix-plan.json').write_text(ex.plan.to_json())
        (tmp_path/'previous-plan.json').write_text(baseline.plan.to_json())


@pytest.mark.parametrize('backend',BACKENDS[1:])
def test_prefix_fault_rejects_overwritten_state(device,tmp_path,backend):
    from test_gpu_expression_contract import unary,lit
    from test_gpu_refractory import refresh_code
    model=model_at(tmp_path/'ref')
    code=next(c for c in model['definition']['synapses'][0]['code_objects'] if c['kind']=='synapses')
    code['vector'][0]['value']=unary('exp',lit(90))
    code['vector'].insert(1,{**deepcopy(code['vector'][0]),'value':lit(0)})
    refresh_code(model,code)
    cls=MetalExecutor if backend=='metal' else CudaExecutor
    with cls(model,tmp_path/'prefix',numeric_mode='float32',synapse_prefix=True,dag_execution='resident') as ex:
        assert has_prefix(ex.plan)
        for _ in range(2):
            with pytest.raises(FloatingPointError):ex.run()


@pytest.mark.parametrize('backend',BACKENDS[1:])
@pytest.mark.parametrize('queued',[False,True])
def test_device_pending_prefix_restore_and_queued_runs(device,tmp_path,backend,queued):
    import brian2 as b
    from test_metal_delays import ROOT
    all_records=[]
    for enabled in (False,True):
        device.reinit()
        b.set_device('rust_standalone',engine=backend,numeric_mode='float32',event_delivery='sparse',
            directory=tmp_path/str(enabled),runner=ROOT/'target/release/b2-runner',
            build_on_run=not queued,gpu_synapse_prefix=enabled,gpu_buffer_reuse=True,gpu_compile_reuse=True)
        b.seed(1729)
        net,syn,post=network(edge_count=16384)
        net.run(4*DT)
        if not queued:
            assert has_prefix(device.last_execution_plan)==enabled
            net.store('pending')
        net.run(4*DT)
        if not queued:
            assert has_prefix(device.last_execution_plan)==enabled
            saved=np.asarray(syn.w[:]).copy()
            net.restore('pending',restore_random_state=True);net.run(4*DT)
            np.testing.assert_array_equal(syn.w[:],saved)
        net.run(4*DT)
        if queued:device.build()
        record={name:np.asarray(getattr(syn,name)[:]).copy() for name in ('w','Apre','Apost','hits')}
        for name in ('v','x'):record['post/'+name]=np.asarray(getattr(post,name)[:]).copy()
        for obj in net.objects:
            if isinstance(obj,b.SpikeMonitor):
                for name in ('t','i','count'):record['spike/'+name]=np.asarray(getattr(obj,name)[:]).copy()
            elif isinstance(obj,b.StateMonitor):
                for name in ('v','x','t'):record['monitor/'+name]=np.asarray(getattr(obj,name)[:]).copy()
        all_records.append(record);device.close_gpu()
    assert all_records[0].keys()==all_records[1].keys()
    for k,v in all_records[0].items():np.testing.assert_array_equal(v,all_records[1][k],err_msg=k)
    np.savez_compressed(tmp_path/'prefix-device.npz',**{label+'/'+k:v for label,r in zip(('reference','actual'),all_records) for k,v in r.items()})
