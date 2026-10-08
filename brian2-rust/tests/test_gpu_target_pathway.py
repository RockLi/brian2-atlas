"""Ordered target lanes for mutable pre/post pathways with neuron writes."""
from contextlib import contextmanager
from copy import deepcopy
import numpy as np
import brian2 as b
import pytest
from brian2_rust import metal_synapses
from brian2_rust.export import lower_network
from brian2_rust.metal import build_metal_plan
from brian2_rust.protocol import attach_protocol
from test_metal_delays import device
from test_gpu_monitors import setup,DT
from test_gpu_synapse_parallel import control
from test_cuda_graphs import result_exact
from test_gpu_spike_generator import BACKENDS,execute
from test_cuda import real_cuda


@contextmanager
def canonical_targets():
    old=metal_synapses.target_owned_pathway;metal_synapses.target_owned_pathway=lambda model,syn,code:False
    try:yield
    finally:metal_synapses.target_owned_pathway=old


def network(*,uniform=False,early=False,recurrent=False,hazard=False,empty=False,edge_count=521,temporary_tail=False):
    pre=b.NeuronGroup(133,'v:1\nx:1',threshold='v>0.5',events={'pulse':'v>0.75'},reset='',dt=DT,name='pre');pre.v=1
    post=pre if recurrent else b.NeuronGroup(135,'v:1\nx:1',threshold='v>0.5',reset='',dt=2*DT,name='post')
    post.v=1
    drive=b.TimedArray(np.arange(4*131).reshape(4,131)/2**18,dt=2*DT)
    code='Apre+=0.125; w=clip(w+Apost,0,2); hits+=1; v_post=(v_post+w)/2+drive(t,j)+rand()/128; x_post=(x_post+w)/2'
    if temporary_tail:code+='; scratch=w/2; x_post=(x_post+scratch)/2'
    if hazard:code+='; w+=v_pre/256'
    syn=b.Synapses(pre[1:132],post[1:132],
        'dApre/dt=-Apre/(8*tau):1 (event-driven)\ndApost/dt=-Apost/(8*tau):1 (event-driven)\nw:1\nhits:integer',
        on_pre={'learn':code},on_post='Apost-=0.0625; w=clip(w+Apre,0,2)',
        on_event={'learn':'pulse','post':'spike'},namespace={'tau':DT,'drive':drive},clock=pre.clock,dtype={'hits':np.int64},name='projection')
    edges=np.arange(0 if empty else edge_count)
    syn.connect(i=(edges*7)%131,j=(edges*17)%131);syn.w=(edges%7+1)/16;syn.hits=2**40
    syn.learn.delay=(np.full(len(edges),2) if uniform else edges%4)*DT
    syn.post.delay=(np.full(len(edges),1) if uniform else edges%3)*post.clock.dt
    net=b.Network(pre,syn,b.SpikeMonitor(pre),b.StateMonitor(post,['v','x'],record=[1,17,131]))
    if not recurrent:net.add(post)
    if early:net.schedule=['start','groups','synapses','thresholds','resets','end']
    return net,syn,post


def model_at(path,**kwargs):
    setup(path);b.seed(1729);net,*_=network(**kwargs);model=lower_network(net,12*DT)
    if not kwargs.get('empty',False):
        for p in model['instance']['synapses'][0]['pathways']:
            items=[7,0,7,2] if kwargs.get('uniform',False) else [520,0,520,7]
            p['pending']=[dict(delivery_tick=t,item=i) for t,i in zip([0,0,0,2],items)]
    attach_protocol(model);return model


@pytest.mark.parametrize('backend',BACKENDS)
@pytest.mark.parametrize('case',['heterogeneous','uniform','early','recurrent','hazard','empty'])
def test_target_order_typed_rng_traces_and_pending(device,tmp_path,backend,case):
    kwargs={} if case=='heterogeneous' else {case:True}
    if case=='hazard':kwargs['recurrent']=True
    model=model_at(tmp_path/'ref',**kwargs)
    with canonical_targets():expected=control(model,tmp_path/'old',1)
    result_exact(control(model,tmp_path/'cpu',3),expected)
    result_exact(execute(model,tmp_path/backend,backend,'sparse'),expected)
    plan=build_metal_plan(model,numeric_mode='float32')
    owned=[d for d in plan.dispatches if d.role=='target-owned-synapse-pathway']
    assert len(owned)==(0 if case=='hazard' else 1)
    assert all(d.lanes==131 for d in owned)
    if case=='hazard':assert any(d.role=='canonical-synapse' for d in plan.dispatches)


def test_proof_rejects_shared_source_stores_and_cross_target_reads(device,tmp_path):
    model=model_at(tmp_path/'ref',recurrent=True);syn=model['definition']['synapses'][0];code=next(c for c in syn['code_objects'] if c['kind']=='synapses')
    assert metal_synapses.target_owned_pathway(model,syn,code)
    for bad in ['v_pre','unknown']:
        copy=deepcopy(code);copy['effects']['writes'].append(bad);assert not metal_synapses.target_owned_pathway(model,syn,copy)
    copy=deepcopy(code);copy['effects']['reads'].append('v_pre');assert not metal_synapses.target_owned_pathway(model,syn,copy)
    copy=deepcopy(code);copy['scalar']=[{'op':'rand','stream':0}];assert not metal_synapses.target_owned_pathway(model,syn,copy)
    copy=deepcopy(model)
    for p in copy['definition']['populations']:
        for state in p['states']:state['index_domain']='scalar'
    assert not metal_synapses.target_owned_pathway(copy,syn,code)


@pytest.mark.parametrize('backend',['reference',*BACKENDS[1:]])
@pytest.mark.parametrize('queued',[False,True])
def test_target_device_restore_pending_and_queued(device,tmp_path,backend,queued):
    values=[]
    for split in [False,True]:
        device.reinit();opts=dict(numeric_mode='float32') if backend!='reference' else {}
        setup(tmp_path/str(split),engine=backend,build_on_run=not queued,**opts);b.seed(1729)
        net,syn,post=network(uniform=True,early=True)
        if split:
            net.run(6*DT)
            if not queued:net.store('pending')
            net.run(6*DT)
            if not queued:
                saved=np.asarray(post.v[:]).copy();net.restore('pending',restore_random_state=True);net.run(6*DT);np.testing.assert_array_equal(post.v[:],saved)
        else:net.run(12*DT)
        if queued:device.build()
        values.append([np.asarray(x[:]).copy() for x in [post.v,post.x,syn.w,syn.Apre,syn.Apost,syn.hits]])
    for a,e in zip(*values,strict=True):np.testing.assert_array_equal(a,e)


@real_cuda
def test_target_cuda_graph_resets_history_cursors_and_populations(device,tmp_path):
    from brian2_rust.cuda import CudaExecutor
    model=model_at(tmp_path/'ref',early=True)
    with canonical_targets():expected=control(model,tmp_path/'canonical',1)
    with CudaExecutor(model,tmp_path/'cuda',numeric_mode='float32') as ex:
        for mode in ['direct','resident','graph','chunked','graph','chunked']:
            result=ex.run(dag_execution=mode);result_exact(result,expected)
            assert result['cuda_runtime']['dag_execution']['selected']==mode
            if mode!='direct':
                with ex.device,ex.stream:
                    for i in ex._resident_dag.writable:ex._resident_dag.gpu[i].fill(99)


@pytest.mark.parametrize('backend',BACKENDS)
@pytest.mark.parametrize('empty',[False,True])
def test_target_fault_in_final_lane_or_empty_projection(device,tmp_path,backend,empty):
    from test_gpu_refractory import refresh_code
    from brian2_rust.spec import bits
    model=model_at(tmp_path/'ref',empty=empty)
    code=next(c for c in model['definition']['synapses'][0]['code_objects'] if c['kind']=='synapses')
    if empty:
        value=dict(op='floor_div',left=dict(op='literal',bits=bits(1.)),right=dict(op='literal',bits=bits(0.)))
    else:
        inst=model['instance']['synapses'][0]
        inst['initial_state']['hits']=['0000000000000000' if j==130 else '0000000000000001' for j in inst['target']]
        value=dict(op='floor_div',left=dict(op='integer',dtype='i64',value='1'),right=dict(op='load',name='hits'))
    code['scalar' if empty else 'vector'].insert(0,dict(target='_checked',dtype='f64' if empty else 'i64',dimensions=[0.]*7,condition=None,value=value));refresh_code(model,code)
    with canonical_targets():
        with pytest.raises(FloatingPointError):control(model,tmp_path/'canonical',1)
    with pytest.raises(FloatingPointError):execute(model,tmp_path/backend,backend,'sparse')


@pytest.mark.parametrize('uniform',[False,True])
def test_target_csr_filters_canonical_order_without_reassociation(device,tmp_path,uniform):
    model=model_at(tmp_path/'ref',uniform=uniform)
    with canonical_targets():before=build_metal_plan(model,numeric_mode='float32')
    after=build_metal_plan(model,numeric_mode='float32');assert before.logical==after.logical and before.buffers==after.buffers
    assert len(after.dispatches)==len(before.dispatches)+1
    i=next(i for i,d in enumerate(after.dispatches) if d.role=='target-owned-synapse-pathway')
    assert after.dispatches[i-1].role=='target-pathway-history'
    assert after.dispatches[i].dependencies==(after.dispatches[i-1].entry,)
    syn=model['definition']['synapses'][0];inst=model['instance']['synapses'][0];path=inst['pathways'][0]
    clock=after.logical.clocks[after.dispatches[i].clock]
    old=metal_synapses.canonical_delay_arrays(syn,inst,path,clock,1<<20)
    new=metal_synapses.canonical_delay_arrays(syn,inst,path,clock,1<<20,target_owned=True)
    targets=np.asarray(inst['target']);offsets=np.r_[0,np.cumsum(np.bincount(targets,minlength=131))]
    for lane in range(131):
        np.testing.assert_array_equal(new[1][offsets[lane]:offsets[lane+1]],old[1][targets[old[1]]==lane])
        for index in [4,5]:np.testing.assert_array_equal(new[index][new[3][lane]:new[3][lane+1]],old[index][targets[old[4]]==lane])
