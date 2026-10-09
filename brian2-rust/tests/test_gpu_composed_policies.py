"""GPU policies must preserve mixed-feature networks and Device lifecycles."""
from types import SimpleNamespace
import json
import numpy as np
import pytest
from brian2_rust.export import lower_network
from brian2_rust.metal import MetalExecutor,build_metal_plan
from brian2_rust.cuda import CudaExecutor
from brian2_rust.metal_dag import run_dag
from brian2_rust import gpu_target_sparse
from test_metal_delays import device
from test_gpu_spike_generator import BACKENDS
from test_gpu_composed_models import setup,network,snapshot,DT
from test_gpu_custom_events import network as named_network,compare
from test_gpu_expression_contract import oracle
from test_gpu_synapse_prefix import has_prefix
from test_gpu_workgroup import save


def mixed(seed,*,traced=False):
    objects=network(seed,edges=16384 if traced else 1028,traced=traced)
    # This declared fixture updates local edge state before the target write.
    # Both the independent reference and every policy run this exact program.
    objects[3].pre.code=('eligibility+=0.0625; ' if traced else '')+'w=clip(w+1.0/4096,0,1.0/16); hits+=1; v_post+=w'
    return objects


def named():
    objects=named_network(True);syn=objects[3]
    for path,scale in [('spike_path',1),('alpha_path',2),('zeta_path',4)]:
        getattr(syn,path).code=f'w+=0.0625; v_post+={scale}*w'
    return objects


def full_compare(actual,expected,*,trace_tolerance=False):
    def arrays(result):
        out={}
        def visit(value,path):
            if isinstance(value,np.ndarray):out[path]=value
            elif isinstance(value,dict):
                for key,item in value.items():visit(item,(*path,str(key)))
            elif isinstance(value,(tuple,list)):
                for i,item in enumerate(value):visit(item,(*path,str(i)))
        for group in ('populations','synapses'):visit(result[group],(group,))
        return out
    a,e=arrays(actual),arrays(expected)
    # The raw compute API returns integer coordinates; the transport reader
    # additionally derives these seconds arrays. Compare them whenever both
    # sides provide them, and never omit any other array.
    def derived(path):
        return path[0]=='populations' and (
            len(path)==3 and path[-1] in ('times','spike_times') or
            len(path)==5 and path[2]=='event_streams' and path[-1]=='times')
    assert all(derived(path) for path in a.keys() ^ e.keys())
    for path in a.keys() & e.keys():
        assert a[path].shape==e[path].shape,path
        if e[path].dtype.kind in 'biu':assert a[path].dtype==e[path].dtype,path
        if trace_tolerance and path[0]=='synapses' and path[-2:]==('states','eligibility'):
            np.testing.assert_allclose(a[path],e[path],rtol=2e-5,atol=2e-6,err_msg='/'.join(path))
        else:np.testing.assert_array_equal(a[path],e[path],err_msg='/'.join(path))
    for group in ('populations','synapses'):assert len(actual[group])==len(expected[group])
    for a,e in zip(actual['synapses'],expected['synapses'],strict=True):assert a['events']==e['events']


def run_model(model,path,backend,prefix,expected_prefix,queue_mode=True):
    options=dict(numeric_mode='float32',event_delivery='sparse',synapse_sparse=queue_mode,synapse_prefix=prefix)
    if backend=='cpu-f32':
        path.mkdir();plan=build_metal_plan(model,**options)
        ex=SimpleNamespace(model=model,plan=plan,directory=path,compile_seconds=0,device_name='CPU f32')
        first=run_dag(ex,max_buffer_bytes=512*1024**2,compute='cpu-f32',workers=3)
        second=run_dag(ex,max_buffer_bytes=512*1024**2,compute='cpu-f32',workers=3)
        runtime={}
    else:
        cls=MetalExecutor if backend=='metal' else CudaExecutor
        with cls(model,path,**options) as ex:
            plan=ex.plan;first=ex.run();second=ex.run()
            runtime=second.get('cuda_runtime',second.get('metal_runtime',{}))
            from brian2_rust.metal import write_metal_results
            from brian2_rust.cuda import write_cuda_results
            from brian2_rust.results import load_results
            writer=write_metal_results if backend=='metal' else write_cuda_results
            writer(model,first,path/'transport')
            transported=load_results(model,path/'transport')
            full_compare(transported,first);first=transported
    full_compare(second,first)
    role=gpu_target_sparse.BITSET_ROLE if queue_mode=='bitset' else gpu_target_sparse.ROLE
    assert any(d.role==role for d in plan.dispatches)
    assert has_prefix(plan)==expected_prefix
    assert len(plan.buffers)==len(set(plan.buffers))
    (path/'policy-plan.json').write_text(plan.to_json()+'\n')
    (path/'policy-runtime.json').write_text(json.dumps(runtime,indent=2)+'\n')
    return first


@pytest.mark.parametrize('backend',BACKENDS)
@pytest.mark.parametrize('prefix',[False,True])
@pytest.mark.parametrize('case',['seed-7','seed-42','named-pathways','trace-42'])
def test_combined_queues_prefixes_and_full_observation(device,tmp_path,backend,prefix,case,queue_mode=True):
    setup(tmp_path/'reference')
    traced=case=='trace-42'
    objects=named() if case=='named-pathways' else mixed(int(case.split('-')[-1]),traced=traced)
    model=lower_network(objects[0],(12 if case=='named-pathways' else 24)*DT)
    (tmp_path/'model.json').write_text(json.dumps(model,sort_keys=True)+'\n')
    expected=oracle(model,tmp_path/'oracle')
    baseline=build_metal_plan(model,numeric_mode='float32',event_delivery='sparse')
    assert not has_prefix(baseline) and not any(d.role==gpu_target_sparse.ROLE for d in baseline.dispatches)
    (tmp_path/'baseline-plan.json').write_text(baseline.to_json()+'\n')
    actual=run_model(model,tmp_path/backend,backend,prefix,prefix and traced,queue_mode)
    full_compare(actual,expected,trace_tolerance=traced)
    assert any(len(p['indices']) for p in expected['populations'])
    save(tmp_path/'composed-policy-results.npz',actual,expected)


@pytest.mark.parametrize('backend',BACKENDS[1:])
@pytest.mark.parametrize('queued',[False,True])
@pytest.mark.parametrize('seed',[7,42])
def test_combined_policy_continuation_restore_and_reuse(device,tmp_path,backend,queued,seed,queue_mode=True):
    records=[];plans=[];models=[]
    for engine in ('reference',backend):
        device.reinit()
        options={} if engine=='reference' else dict(gpu_synapse_sparse=queue_mode,gpu_synapse_prefix=True,
            gpu_buffer_reuse=True,gpu_compile_reuse=True)
        setup(tmp_path/engine,engine,build_on_run=not queued,**options)
        net,pre,post,syn,monitors,spikes,events=mixed(seed,traced=seed==42)
        checkpoints={}
        def capture(label):
            checkpoints.update({label+'/'+k:v for k,v in snapshot(pre,post,syn,monitors,spikes,events).items()})
            if engine!='reference':
                plan=device.last_execution_plan
                assert len(plan.buffers)==len(set(plan.buffers))
                role=gpu_target_sparse.BITSET_ROLE if queue_mode=='bitset' else gpu_target_sparse.ROLE
                assert any(d.role==role for d in plan.dispatches)
                plans.append(dict(checkpoint=label,plan=plan.to_dict()))
                model=json.loads((device.last_run_directory/'model.json').read_text())
                models.append(dict(checkpoint=label,model=model))
                binding=device.last_run_directory/'runtime-binding.json'
                if binding.is_file():plans[-1]['runtime_binding']=json.loads(binding.read_text())
        net.run(6*DT)
        if not queued:capture('first');net.store('pending')
        net.run(6*DT)
        if not queued:
            capture('second')
            before=snapshot(pre,post,syn,monitors,spikes,events)
            net.restore('pending');net.run(6*DT)
            capture('restored')
            restored=snapshot(pre,post,syn,monitors,spikes,events)
            for key in before:np.testing.assert_array_equal(restored[key],before[key],err_msg=key)
            syn.w=np.asarray(syn.w[:])+1/4096
        net.run(12*DT)
        if queued:device.build()
        capture('final');records.append(checkpoints)
        if engine!='reference':
            if seed==42:assert any(d['role']=='edge-synapse-prefix' for row in plans for d in row['plan']['dispatches'])
            pending=any(p['pending'] for row in models for s in row['model']['instance']['synapses'] for p in s['pathways'])
            if queued:
                # Contiguous queued calls build one 24-tick activation.
                assert not pending and len(models)==1
                assert sorted(c['steps'] for c in models[0]['model']['run']['clocks'])==[12,16,24]
            else:assert pending and len(models)==4
            device.close_gpu()
    assert records[0].keys()==records[1].keys()
    for key in records[0]:
        if records[0][key].dtype.kind in 'biu':assert records[1][key].dtype==records[0][key].dtype
        if key.endswith('/syn/eligibility'):np.testing.assert_allclose(records[1][key],records[0][key],rtol=2e-5,atol=2e-6,err_msg=key)
        else:np.testing.assert_array_equal(records[1][key],records[0][key],err_msg=key)
    np.savez_compressed(tmp_path/'composed-policy-device.npz',**{
        label+'/'+k:v for label,d in zip(('reference','actual'),records,strict=True) for k,v in d.items()})
    (tmp_path/'composed-policy-device-plans.json').write_text(json.dumps(plans)+'\n')
    (tmp_path/'composed-policy-device-models.json').write_text(json.dumps(models)+'\n')
