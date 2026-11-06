"""Effect proof, counter ownership and complete pre/post fusion results."""
from dataclasses import replace
from types import SimpleNamespace
import json
import subprocess
import brian2 as b
import numpy as np
import pytest
from brian2_rust.metal import build_metal_plan,MetalExecutor
from brian2_rust.cuda import CudaExecutor
from brian2_rust.export import lower_network
from brian2_rust.metal_dag import run_dag,_prepare_dag_storage
from brian2_rust import gpu_synapse_fusion as fusion
from test_metal_delays import device,ROOT
from test_gpu_spike_generator import BACKENDS
from test_cuda_graphs import result_exact
from test_gpu_workgroup import save

DT=b.second/1024


def network(flavor='plain',edges=273):
    pop=b.NeuronGroup(73,'v:1',threshold='v>0.5',reset='v=0.25',dt=DT,name='population')
    pop.v=(np.arange(73)%4)/4;pop.run_regularly('v+=0.125')
    drive=b.TimedArray(np.arange(16)/1024,dt=DT)
    extra={'plain':'','typed':'','random':'+rand()/128','timed':'+drive(t)',
           'own-read':'+v_post/1024','source-read':'+v_pre/1024'}[flavor]
    pre='v_post+=w/128; w+=0.125; k+=1'
    post='w-=0.0625'+extra+'; k+=2'
    if flavor=='typed':pre+='; flag=not flag';post+='; flag=not flag'
    syn=b.Synapses(pop,pop,'w:1\nk:integer\nflag:boolean\ndivisor:integer (constant)',on_pre=pre,on_post=post,
        clock=pop.clock,namespace={'drive':drive},name='plastic')
    # Deliberately decorrelate edge indices and target owners: raw shared
    # delivered[edge]/delivered[target] would collide under naive fusion.
    syn.connect(i=(np.arange(edges)*17)%73,j=(np.arange(edges)*31+19)%73)
    syn.w=(np.arange(edges)%11)/64;syn.k=np.arange(edges)%3;syn.flag=np.arange(edges)%2;syn.divisor=1
    syn.pre.delay=(np.arange(edges)%4)*DT;syn.post.delay=(np.arange(edges)%3)*DT
    spikes=b.SpikeMonitor(pop);monitor=b.StateMonitor(pop,'v',record=True)
    return b.Network(pop,syn,spikes,monitor),pop,syn,spikes,monitor


def model_at(path,flavor='plain',edges=273,steps=16):
    b.set_device('rust_standalone',engine='reference',directory=path,runner=ROOT/'target/release/b2-runner')
    net,*_=network(flavor,edges)
    return lower_network(net,steps*DT)


def control(model,path,enabled=False,workers=3):
    path.mkdir()
    ex=SimpleNamespace(model=model,plan=build_metal_plan(model,numeric_mode='float32',event_delivery='sparse',synapse_fusion=enabled),
        directory=path,compile_seconds=0,device_name='CPU f32')
    result=run_dag(ex,max_buffer_bytes=512*1024**2,compute='cpu-f32',workers=workers)
    return result


def pair(plan):
    i=next(i for i,d in enumerate(plan.dispatches) if d.role=='target-owned-synapse-pathway')
    assert plan.dispatches[i+1].role=='edge-owned-synapse-pathway'
    return (*plan.kernels[i:i+2],*plan.dispatches[i:i+2])


def test_plan_inventory_storage_and_counter_mapping(device,tmp_path):
    model=model_at(tmp_path/'ref');before=build_metal_plan(model,numeric_mode='float32',event_delivery='sparse')
    after=build_metal_plan(model,numeric_mode='float32',event_delivery='sparse',synapse_fusion=True)
    assert len(after.dispatches)==len(before.dispatches)-1
    assert before.logical==after.logical and before.buffers==after.buffers
    assert [n for k in before.kernels for n in k.nodes]==[n for k in after.kernels for n in k.nodes]
    for a,e in zip(*[_prepare_dag_storage(SimpleNamespace(model=model,plan=p),512*1024**2)[0] for p in (before,after)],strict=True):
        np.testing.assert_array_equal(a,e)
    kernel=next(k for k,d in zip(after.kernels,after.dispatches) if d.role==fusion.ROLE)
    assert 'delivered[targets[edge]] += 1;' in kernel.source
    assert 'delivered[targets[lane]] |= 0x8000000000000000ul;' in kernel.source
    assert 'if(i>=73u) return;' in kernel.source
    for i,d in enumerate(after.dispatches):
        assert d.dependencies==((after.dispatches[i-1].entry,) if i else ())
        assert len(set(d.bindings))==len(d.bindings) and len(d.bindings)<=30
    (tmp_path/'fusion-plan.json').write_text(json.dumps(after.to_dict(),indent=2)+'\n')


@pytest.mark.parametrize('hazard',['source-read','clock','counter-bound','different-owner','scalar-write'])
def test_pair_proof_rejects_cross_lane_or_counter_hazards(device,tmp_path,hazard):
    model=model_at(tmp_path/'ref');plan=build_metal_plan(model,numeric_mode='float32',event_delivery='sparse')
    left,right,a,d=pair(plan);logical=plan.logical
    syn=model['definition']['synapses'][0];post=next(c for c in syn['code_objects'] if c['kind']=='synapses_post')
    if hazard=='source-read':post['effects']['reads'].append('v_pre')
    elif hazard=='clock':d=replace(d,clock=a.clock+1)
    elif hazard=='counter-bound':logical=replace(logical,clocks=tuple(replace(c,steps=2**63) for c in logical.clocks))
    elif hazard=='different-owner':
        logical=replace(logical,nodes=tuple(replace(n,owner_index=1) if n.id==right.nodes[0] else n for n in logical.nodes))
    else:
        state=next(s for s in syn['states'] if s['name']=='k');state['index_domain']='scalar'
    assert not fusion.eligible(model,logical,left,right,a,d)


def test_recurrent_source_read_has_a_real_counterexample(device,tmp_path,monkeypatch):
    model=model_at(tmp_path/'ref','source-read')
    expected=control(model,tmp_path/'baseline',workers=1)
    safe=control(model,tmp_path/'safe',enabled=True,workers=1);result_exact(safe,expected)
    with monkeypatch.context() as patch:
        patch.setattr(fusion,'eligible',lambda m,l,left,right,a,d: a.role=='target-owned-synapse-pathway' and d.role=='edge-owned-synapse-pathway')
        unsafe=control(model,tmp_path/'unsafe',enabled=True,workers=1)
    assert not np.array_equal(unsafe['synapses'][0]['states']['w'],expected['synapses'][0]['states']['w'])
    np.savez_compressed(tmp_path/'fusion-hazard-counterexample.npz',safe_w=expected['synapses'][0]['states']['w'],unsafe_w=unsafe['synapses'][0]['states']['w'])


@pytest.mark.parametrize('backend',BACKENDS)
@pytest.mark.parametrize('flavor',['plain','typed','random','timed','own-read'])
def test_fused_public_arrays_event_totals_and_replay(device,tmp_path,backend,flavor):
    model=model_at(tmp_path/'ref',flavor);expected=control(model,tmp_path/'baseline')
    if backend=='cpu-f32':
        actual=control(model,tmp_path/'fused',True);result_exact(actual,expected)
        save(tmp_path/'fusion-results.npz',actual,expected)
    else:
        cls=MetalExecutor if backend=='metal' else CudaExecutor
        with cls(model,tmp_path/'gpu',numeric_mode='float32',event_delivery='sparse',synapse_fusion=True) as ex:
            assert any(d.role==fusion.ROLE for d in ex.plan.dispatches)
            for i in range(2):
                actual=ex.run();result_exact(actual,expected)
                save(tmp_path/f'fusion-results-{i}.npz',actual,expected)
            result_exact(ex.run(dag_execution='workgroup'),expected)
            (tmp_path/'fusion-runtime.json').write_text(json.dumps(actual.get('cuda_runtime',actual.get('metal_runtime')),indent=2)+'\n')
    assert actual['synapses'][0]['events']==expected['synapses'][0]['events']>0


@pytest.mark.parametrize('backend',BACKENDS)
@pytest.mark.parametrize('empty',[False,True])
def test_post_scalar_fault_and_last_edge_fault_survive_counter_remap(device,tmp_path,backend,empty):
    from test_gpu_refractory import refresh_code
    from brian2_rust.spec import bits
    model=model_at(tmp_path/'ref',edges=0 if empty else 273)
    code=next(c for c in model['definition']['synapses'][0]['code_objects'] if c['kind']=='synapses_post')
    bad=dict(op='floor_div',left=dict(op='literal',bits=bits(1.)),right=dict(op='literal',bits=bits(0.)))
    if not empty:
        model['instance']['synapses'][0]['parameters']['divisor']=['00000001']*272+['00000000']
        bad=dict(op='floor_div',left=dict(op='integer',dtype='i32',value='1'),right=dict(op='load',name='divisor'))
        code['effects']['reads']=sorted(set(code['effects']['reads'])|{'divisor'})
    code['scalar' if empty else 'vector'].insert(0,dict(target='_fault',dtype='f64' if empty else 'i32',dimensions=[0.]*7,condition=None,value=bad))
    refresh_code(model,code)
    if backend=='cpu-f32':
        with pytest.raises(FloatingPointError):control(model,tmp_path/'fused',True)
    else:
        cls=MetalExecutor if backend=='metal' else CudaExecutor
        with cls(model,tmp_path/'gpu',numeric_mode='float32',event_delivery='sparse',synapse_fusion=True) as ex:
            assert any(d.role==fusion.ROLE for d in ex.plan.dispatches)
            with pytest.raises(FloatingPointError):ex.run()


@pytest.mark.parametrize('backend',BACKENDS[1:])
def test_device_pending_restore_and_reuse(device,tmp_path,backend):
    records=[]
    for enabled in (False,True):
        device.reinit()
        b.set_device('rust_standalone',engine=backend,numeric_mode='float32',event_delivery='sparse',
            gpu_synapse_fusion=enabled,gpu_buffer_reuse=True,gpu_compile_reuse=True,
            directory=tmp_path/str(enabled),runner=ROOT/'target/release/b2-runner')
        net,pop,syn,spikes,monitor=network();states=[]
        for i in range(3):
            if i==1:net.store('saved')
            if i==2:net.restore('saved')
            net.run(8*DT)
            states.append([np.asarray(x).copy() for x in (pop.v[:],syn.w[:],syn.k[:],spikes.t[:],spikes.i[:],monitor.v[:])])
        if enabled:assert any(d.role==fusion.ROLE for d in device._gpu_executor.plan.dispatches)
        device.close_gpu();records.append(states)
    for a,e in zip(*records,strict=True):
        for x,y in zip(a,e,strict=True):np.testing.assert_array_equal(x,y)
    np.savez_compressed(tmp_path/'fusion-device.npz',**{f'{mode}/{i}/{j}':v for mode,states in zip(('reference','actual'),records) for i,values in enumerate(states) for j,v in enumerate(values)})
