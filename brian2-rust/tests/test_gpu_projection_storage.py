"""Prune only unbound queues, retain live immutable routes, and switch layouts."""
from types import SimpleNamespace
import json
import numpy as np
import pytest
from brian2_rust.metal import MetalExecutor,build_metal_plan
from brian2_rust.cuda import CudaExecutor,build_cuda_plan
from brian2_rust.metal_dag import _prepare_dag_storage,run_dag
from brian2_rust.gpu_buffer_transfer import adopt_buffers
from test_gpu_spike_generator import BACKENDS
from test_metal_delays import device
from test_gpu_composed_models import setup,network,DT
from brian2_rust.export import lower_network
from test_gpu_composed_policies import full_compare
from test_gpu_synapse_parallel import control
from test_gpu_workgroup import save
from gpu_projection_storage_compare import legacy_projection_storage


def mixed_model(path):
    import brian2 as b
    setup(path);net,*_=network(42)
    # The typed endpoints in the original fixture make both its projections
    # canonical. Add float-only endpoints to exercise a live generic queue too.
    a=b.NeuronGroup(17,'v:1',threshold='v>=1',reset='v=0',dt=DT,name='float_source')
    z=b.NeuronGroup(19,'v:1',threshold='v>=1',reset='v=0',dt=DT,name='float_target')
    a.v=np.arange(17)%8/8;a.run_regularly('v+=0.25')
    edge=b.Synapses(a,z,'w:1 (constant)',on_pre='v_post+=w',clock=a.clock,name='immutable_float')
    edge.connect(i=np.arange(17),j=np.arange(17));edge.w=.125
    monitor=b.StateMonitor(z,'v',record=True);spikes=b.SpikeMonitor(z)
    net.add(a,z,edge,monitor,spikes)
    model=lower_network(net,24*DT)
    # Lowering sorts names; identify routes by their declared synapse names.
    return model


def inventories(model,build,mode):
    opts=dict(numeric_mode='float32',event_delivery='sparse',synapse_sparse=mode)
    with legacy_projection_storage():old=build(model,**opts)
    new=build(model,**opts)
    assert old.logical==new.logical and old.kernels==new.kernels and old.elided_nodes==new.elided_nodes
    assert old.sha256!=new.sha256
    removed=set(old.buffers)-set(new.buffers)
    bound={old.buffers[i] for d in old.dispatches for i in d.bindings}
    assert not removed&bound
    live=next(q for q,s in enumerate(model['definition']['synapses']) if s['name']=='immutable_float')
    dead=[q for q in range(len(model['definition']['synapses'])) if q!=live]
    kinds=('source_offsets','source_ranks','rank_targets','active_ranks','active_counts')
    assert removed=={f'synapse/{q}/{k}' for q in dead for k in kinds}
    assert all(f'synapse/{live}/{k}' in new.buffers for k in kinds)
    assert any(f'synapse/{live}/{k}' in bound for k in kinds)
    for a,b in zip(old.dispatches,new.dispatches,strict=True):
        assert (a.role,a.clock,a.lanes,a.types)==(b.role,b.clock,b.lanes,b.types)
        assert tuple(old.buffers[i] for i in a.bindings)==tuple(new.buffers[i] for i in b.bindings)
    arrays={}
    for name,plan in [('legacy',old),('pruned',new)]:
        arrays[name]=dict(zip(plan.buffers,_prepare_dag_storage(SimpleNamespace(model=model,plan=plan),512*1024**2)[0],strict=True))
    for name,v in arrays['pruned'].items():np.testing.assert_array_equal(v,arrays['legacy'][name])
    expected=sum(4*(3*len(model['instance']['synapses'][q]['source'])+model['definition']['synapses'][q]['source_count']+model['definition']['synapses'][q]['target_count']+1) for q in dead)
    assert sum(arrays['legacy'][n].nbytes for n in removed)==expected
    return old,new,dict(removed=sorted(removed),bytes_saved=expected,
        legacy=[dict(name=n,bytes=v.nbytes) for n,v in arrays['legacy'].items()],
        pruned=[dict(name=n,bytes=v.nbytes) for n,v in arrays['pruned'].items()])


@pytest.mark.parametrize('mode',[False,True,'bitset'])
def test_pruned_inventory_matches_legacy_kernel_bindings(device,tmp_path,mode):
    model=mixed_model(tmp_path/'ref')
    records=[]
    for build in (build_metal_plan,build_cuda_plan):
        old,new,storage=inventories(model,build,mode)
        records.append(dict(old=old.to_dict(),new=new.to_dict(),storage=storage))
    (tmp_path/'projection-inventory.json').write_text(json.dumps(records)+'\n')
    (tmp_path/'projection-model.json').write_text(json.dumps(model)+'\n')


@pytest.mark.parametrize('backend',BACKENDS)
@pytest.mark.parametrize('mode',[False,True,'bitset'])
def test_native_mixed_projection_routes_match_and_replay(device,tmp_path,backend,mode):
    model=mixed_model(tmp_path/'ref');expected=control(model,tmp_path/'control',3)
    cls=MetalExecutor if backend=='metal' else CudaExecutor
    plan=(build_cuda_plan if backend=='cuda' else build_metal_plan)(model,numeric_mode='float32',event_delivery='sparse',synapse_sparse=mode)
    if backend=='cpu-f32':
        path=tmp_path/'cpu';path.mkdir();ex=SimpleNamespace(model=model,plan=plan,directory=path,compile_seconds=0,device_name='CPU f32')
    else:ex=cls(model,tmp_path/'native',plan=plan,numeric_mode='float32',event_delivery='sparse',synapse_sparse=mode)
    try:
        for route in ('direct','auto','workgroup'):
            actual=run_dag(ex,max_buffer_bytes=512*1024**2,compute='cpu-f32',workers=3) if backend=='cpu-f32' else ex.run(dag_execution=route)
            full_compare(actual,expected)
        save(tmp_path/'projection-results.npz',actual,expected)
        (tmp_path/'projection-plan.json').write_text(plan.to_json())
        (tmp_path/'projection-model.json').write_text(json.dumps(model)+'\n')
    finally:
        if backend!='cpu-f32':ex.close()


@pytest.mark.parametrize('backend',BACKENDS[1:])
def test_cross_inventory_transfer_refreshes_shifted_bindings(device,tmp_path,backend):
    model=mixed_model(tmp_path/'ref');expected=control(model,tmp_path/'control',3)
    cls=MetalExecutor if backend=='metal' else CudaExecutor
    previous=None;records=[]
    try:
        for i,legacy in enumerate((True,False,True,False)):
            with legacy_projection_storage() if legacy else __import__('contextlib').nullcontext():
                current=cls(model,tmp_path/str(i),numeric_mode='float32',event_delivery='sparse',synapse_sparse='bitset',compile_reuse=True,reuse_from=previous,dag_execution='auto')
            adopted=adopt_buffers(current,previous,backend=backend) if previous is not None else False
            if previous is not None:previous.close()
            previous=current
            uploads=list(getattr(current,'_activation_upload_indices',()))
            first=current.run();full_compare(first,expected);full_compare(current.run(),first)
            if i:assert adopted
            records.append(dict(legacy=legacy,adopted=adopted,uploads=uploads,plan=current.plan.to_dict(),compilation=current.compilation_report.copy()))
            save(tmp_path/f'projection-switch-{i}.npz',first,expected)
    finally:
        if previous is not None:previous.close()
    (tmp_path/'projection-switch.json').write_text(json.dumps(records)+'\n')
    (tmp_path/'projection-model.json').write_text(json.dumps(model)+'\n')
