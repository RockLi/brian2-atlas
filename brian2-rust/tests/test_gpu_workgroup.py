"""Single-group stage barriers preserve all results, clocks, faults and lifecycle."""
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import pytest
from brian2_rust.cuda import CudaExecutor,build_cuda_plan
from brian2_rust.metal import MetalExecutor,build_metal_plan
from brian2_rust.gpu_workgroup import prepare,layout_of
from brian2_rust.metal_dag import _prepare_dag_storage
from test_metal_delays import device,ROOT
from test_gpu_spike_generator import BACKENDS
from test_gpu_summed_parallel import model_at,control
from test_cuda_graphs import result_exact


def save(path,actual,expected):
    values={}
    for label,result in [('actual',actual),('reference',expected)]:
        for group in ('populations','synapses'):
            def add(value,key):
                if isinstance(value,dict):
                    for k,v in value.items():add(v,key+'/'+k)
                elif isinstance(value,(tuple,list)):
                    for k,v in enumerate(value):add(v,key+'/'+str(k))
                elif isinstance(value,np.ndarray):values[key]=value
            add(result[group],label+'/'+group)
    np.savez_compressed(path,**values)


@pytest.mark.parametrize('backend',BACKENDS[1:])
@pytest.mark.parametrize('route',['scan','sparse'])
@pytest.mark.parametrize('flavor',['plain','random','timed'])
def test_workgroup_exact_barriers_replay_modes_and_release(device,tmp_path,backend,route,flavor):
    model=model_at(tmp_path/'ref',edges=273,steps=8,flavor=flavor);expected=control(model,tmp_path/'control')
    cls=MetalExecutor if backend=='metal' else CudaExecutor
    with cls(model,tmp_path/backend,numeric_mode='float32',event_delivery=route) as ex:
        result_exact(ex.run(),expected)
        for _ in range(2):
            actual=ex.run(dag_execution='workgroup');result_exact(actual,expected)
        wg=ex._workgroup;assert wg.replays==2
        assert wg.program.manifest['workgroups']==1 and wg.program.manifest['logical_launches']>1
        save(tmp_path/'workgroup-results.npz',actual,expected)
        runtime=actual['cuda_runtime']['dag_execution'] if backend=='cuda' else actual['metal_runtime']
        (tmp_path/'workgroup-runtime.json').write_text(json.dumps(runtime,indent=2)+'\n')
        result_exact(ex.run(dag_execution='resident'),expected)
        with pytest.raises(MemoryError):ex.run(dag_execution='workgroup',max_buffer_bytes=1)
        # The normal host limit may reject before entering the workgroup runtime;
        # explicit close must still destroy the compiled group.
        ex.close();assert ex._workgroup is None
        assert wg.handle is None and wg.function is None


def test_packed_layout_bounds_and_tick_schedule_are_explicit(device,tmp_path,monkeypatch):
    from brian2_rust.gpu_schedule import dispatch_ticks
    import brian2_rust.gpu_workgroup as w
    model=model_at(tmp_path/'ref',edges=273,steps=8)
    for backend,build in [('metal',build_metal_plan),('cuda',build_cuda_plan)]:
        plan=build(model,numeric_mode='float32',event_delivery='sparse')
        owner=SimpleNamespace(plan=plan,model=model)
        arrays,*_=_prepare_dag_storage(owner,512*1024**2);program=prepare(owner,arrays,backend)
        assert all(offset%16==0 for offset in program.offsets)
        expected=[(s,t) for s,t in dispatch_ticks(plan.logical.clocks,tuple(d.clock for d in plan.dispatches)) if plan.dispatches[s].lanes]
        assert list(zip(program.stages.tolist(),program.ticks.tolist()))==expected
        assert program.layout==layout_of(arrays)
        assert ('threadgroup_barrier(mem_flags::mem_device)' if backend=='metal' else '__syncthreads()') in program.source
    monkeypatch.setattr(w,'MAX_LANE_VISITS',1)
    with pytest.raises(ValueError,match='budget'):prepare(owner,arrays,backend)


@pytest.mark.parametrize('backend',BACKENDS[1:])
def test_typed_synapses_and_monitors_preserve_packed_bits(device,tmp_path,backend):
    from test_gpu_typed_storage import synaptic_network
    from test_gpu_monitors import setup,DT
    from brian2_rust.export import lower_network
    setup(tmp_path/'ref');model=lower_network(synaptic_network()[0],8*DT)
    expected=control(model,tmp_path/'control');cls=MetalExecutor if backend=='metal' else CudaExecutor
    with cls(model,tmp_path/backend,numeric_mode='float32',event_delivery='sparse') as ex:
        actual=ex.run(dag_execution='workgroup');result_exact(actual,expected)
        save(tmp_path/'workgroup-results.npz',actual,expected)


@pytest.mark.parametrize('backend',BACKENDS[1:])
@pytest.mark.parametrize('empty',[False,True])
def test_workgroup_scalar_and_last_lane_faults_not_published(device,tmp_path,backend,empty):
    from test_gpu_refractory import refresh_code
    from brian2_rust.spec import bits
    model=model_at(tmp_path/'ref',edges=0 if empty else 273,steps=4)
    code=next(c for c in model['definition']['synapses'][0]['code_objects'] if c['kind']=='summed_variable')
    bad=dict(op='floor_div',left=dict(op='literal',bits=bits(1.)),right=dict(op='literal',bits=bits(0.)))
    if not empty:
        model['instance']['synapses'][0]['initial_state']['k']=['00000001']*272+['00000000']
        bad=dict(op='floor_div',left=dict(op='integer',dtype='i32',value='1'),right=dict(op='load',name='k'))
        code['effects']['reads']=sorted(set(code['effects']['reads'])|{'k'})
    stmt=dict(target='_fault',dtype='f64' if empty else 'i32',dimensions=[0.]*7,condition=None,value=bad)
    code['scalar' if empty else 'vector'].insert(0,stmt);refresh_code(model,code)
    cls=MetalExecutor if backend=='metal' else CudaExecutor
    with cls(model,tmp_path/backend,numeric_mode='float32',event_delivery='sparse') as ex:
        with pytest.raises(FloatingPointError):ex.run(dag_execution='workgroup')


@pytest.mark.parametrize('backend',BACKENDS[1:])
def test_device_workgroup_pending_restore_and_compile_reuse(device,tmp_path,backend):
    import brian2 as b
    from test_gpu_multiclock import network,DT
    records=[]
    for mode in ('direct','workgroup'):
        device.reinit()
        b.set_device('rust_standalone',engine=backend,numeric_mode='float32',event_delivery='sparse',
            directory=tmp_path/mode,runner=ROOT/'target/release/b2-runner',gpu_compile_reuse=True,gpu_buffer_reuse=True,
            **{backend+'_dag_execution':mode})
        net,pre,post,syn,monitors,spikes=network(True);states=[]
        for i in range(3):
            if i==1:net.store('saved')
            if i==2:net.restore('saved')
            net.run(6*DT)
            states.append([np.asarray(x).copy() for x in [pre.v[:],post.v[:],syn.w[:],syn.x[:],
                *[m.v[:] for m in monitors],*[s.t[:] for s in spikes],*[s.i[:] for s in spikes]]])
        records.append(states);ex=device._gpu_executor;device.close_gpu();assert ex._workgroup is None if mode=='workgroup' else device._gpu_executor is None
    for a,e in zip(*records):
        for x,y in zip(a,e):np.testing.assert_array_equal(x,y)
    np.savez_compressed(tmp_path/'workgroup-device-results.npz',**{f'{mode}/{i}/{j}':value for mode,states in zip(('reference','actual'),records) for i,values in enumerate(states) for j,value in enumerate(values)})
