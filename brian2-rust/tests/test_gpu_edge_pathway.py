"""Per-edge pre/post delivery preserves pending chronology and scalar faults."""
from contextlib import contextmanager
from copy import deepcopy
import brian2 as b
import numpy as np
import pytest
from brian2_rust import metal_synapses
from brian2_rust.export import lower_network
from brian2_rust.metal import build_metal_plan
from brian2_rust.protocol import attach_protocol
from brian2_rust.spec import bits
from test_cuda_graphs import result_exact
from test_gpu_spike_generator import BACKENDS,execute
from test_gpu_synapse_parallel import control
from test_gpu_monitors import setup,DT
from test_gpu_refractory import refresh_code
from test_metal_delays import device
from test_cuda import real_cuda


@contextmanager
def canonical_pathways():
    old=metal_synapses.independent_pathway;old_target=metal_synapses.target_owned_pathway
    metal_synapses.independent_pathway=lambda syn,code:False
    metal_synapses.target_owned_pathway=lambda model,syn,code:False
    try:yield
    finally:
        metal_synapses.independent_pathway=old;metal_synapses.target_owned_pathway=old_target


def network(uniform=False,early=False,named=False,edges=129,population_write=False):
    pre=b.NeuronGroup(9,'v:1',threshold='v>0.5',reset='',events={'burst':'v>1.5'},dt=DT,name='pre');pre.v=2
    post=b.NeuronGroup(7,'v:1',threshold='v>0.5',reset='',dt=2*DT,name='post');post.v=1
    precode='Apre+=0.125; w=clip(w+Apost+v_pre/128,0,16); hits+=1'
    if population_write:precode+='; v_post+=w/128'
    syn=b.Synapses(pre[1:8],post[1:6],
        'dApre/dt=-Apre/(8*tau):1 (event-driven)\ndApost/dt=-Apost/(8*tau):1 (event-driven)\nw:1\nhits:integer',
        on_pre={'learn':precode},on_post='Apost-=0.0625; w=clip(w+Apre+v_post/128,0,16); hits+=1',
        on_event={'learn':'burst' if named else 'spike','post':'spike'},namespace={'tau':DT},clock=pre.clock,dtype={'hits':np.int64},name='projection')
    syn.connect(i=np.arange(edges)%7,j=(np.arange(edges)*3)%5);syn.w=.5
    syn.learn.delay=(np.full(edges,2) if uniform else np.arange(edges)%4)*DT
    syn.post.delay=(np.full(edges,1) if uniform else np.arange(edges)%3)*2*DT
    net=b.Network(pre,post,syn,b.EventMonitor(pre,'burst' if named else 'spike',variables='v',when='end'))
    if early:net.schedule=['start','groups','synapses','thresholds','resets','end']
    return net,syn


def model_at(path,uniform=False,early=False,named=False,edges=129,population_write=False,pending=True):
    setup(path);net,syn=network(uniform,early,named,edges,population_write);model=lower_network(net,12*DT)
    if pending:
        for path in model['instance']['synapses'][0]['pathways']:
            items=[2,0,2,1] if uniform else [128,0,128,5]
            path['pending']=[dict(delivery_tick=t,item=i) for t,i in zip([0,0,0,2],items)]
        attach_protocol(model)
    return model


@pytest.mark.parametrize('uniform',[False,True])
def test_pending_buckets_and_owned_memory_are_in_the_plan(device,tmp_path,uniform):
    model=model_at(tmp_path/'ref',uniform=uniform)
    with canonical_pathways():old=build_metal_plan(model,numeric_mode='float32',event_delivery='sparse')
    new=build_metal_plan(model,numeric_mode='float32',event_delivery='sparse')
    assert old.logical==new.logical and old.buffers==new.buffers and len(new.dispatches)==len(old.dispatches)+2
    assert sum(d.role=='edge-owned-synapse-pathway' for d in new.dispatches)==2
    for i,d in enumerate(new.dispatches):
        if d.role=='edge-owned-synapse-pathway':
            assert d.lanes==129 and new.dispatches[i-1].role=='edge-pathway-history'
            assert d.dependencies==(new.dispatches[i-1].entry,)
    syn=model['definition']['synapses'][0];inst=model['instance']['synapses'][0]
    for path in inst['pathways']:
        code=next(c for c in syn['code_objects'] if c.get('pathway_name')==path['name'])
        clock=new.logical.clocks[code['clock']]
        serial=metal_synapses.canonical_delay_arrays(syn,inst,path,clock,1<<20)
        parallel=metal_synapses.canonical_delay_arrays(syn,inst,path,clock,1<<20,edge_owned=True)
        assert len(parallel[3])==130 and len(parallel[6])==129
        for edge in range(129):
            np.testing.assert_array_equal(parallel[5][parallel[3][edge]:parallel[3][edge+1]],serial[5][serial[4]==edge])
        with pytest.raises(MemoryError):metal_synapses.canonical_delay_arrays(syn,inst,path,clock,129*4,edge_owned=True)


def test_ownership_rejects_population_shared_scalar_rng_and_wrong_domain(device,tmp_path):
    model=model_at(tmp_path/'ref',pending=False);syn=model['definition']['synapses'][0]
    code=next(c for c in syn['code_objects'] if c['kind']=='synapses')
    assert metal_synapses.independent_pathway(syn,code)
    for key,value in [('kind','summed_variable'),('iteration_domain','all_synapses'),('scalar',[{'op':'rand','stream':0}])]:
        copy=deepcopy(code);copy[key]=value;assert not metal_synapses.independent_pathway(syn,copy)
    for alias in ['v_pre','v_post','absent']:
        copy=deepcopy(code);copy['effects']['writes'].append(alias);assert not metal_synapses.independent_pathway(syn,copy)
    copy=deepcopy(syn)
    for state in copy['states']:state['index_domain']='scalar'
    assert not metal_synapses.independent_pathway(copy,code)


@pytest.mark.parametrize('backend',BACKENDS)
@pytest.mark.parametrize('variant',['heterogeneous','uniform','early-named','mixed-ownership'])
def test_pending_trace_integer_and_repeated_results(device,tmp_path,backend,variant):
    model=model_at(tmp_path/'ref',uniform=variant=='uniform',early=variant=='early-named',named=variant=='early-named',population_write=variant=='mixed-ownership')
    with canonical_pathways():expected=control(model,tmp_path/'canonical',1)
    result_exact(control(model,tmp_path/'cpu-3',3),expected)
    result_exact(execute(model,tmp_path/backend,backend,'sparse'),expected)
    if variant=='mixed-ownership':
        plan=build_metal_plan(model,numeric_mode='float32')
        assert sum(d.role=='edge-owned-synapse-pathway' for d in plan.dispatches)==1
        assert any(d.role=='target-owned-synapse-pathway' for d in plan.dispatches)


@pytest.mark.parametrize('backend',BACKENDS)
@pytest.mark.parametrize('empty',[False,True])
def test_scalar_fault_runs_without_arrivals_and_empty_edges(device,tmp_path,backend,empty):
    model=model_at(tmp_path/'ref',edges=0 if empty else 129,pending=False)
    for pop in model['instance']['populations']:pop['initial_state']['v']=[bits(0.) for _ in pop['initial_state']['v']]
    code=next(c for c in model['definition']['synapses'][0]['code_objects'] if c['kind']=='synapses')
    code['scalar'].insert(0,dict(target='_bad',dtype='f64',dimensions=[0.]*7,condition=None,
        value=dict(op='floor_div',left=dict(op='literal',bits=bits(1.)),right=dict(op='literal',bits=bits(0.)))))
    refresh_code(model,code)
    with canonical_pathways():
        with pytest.raises(FloatingPointError):control(model,tmp_path/'old',1)
    with pytest.raises(FloatingPointError):execute(model,tmp_path/backend,backend,'sparse')


@pytest.mark.parametrize('backend',['reference',*BACKENDS[1:]])
@pytest.mark.parametrize('queued',[False,True])
def test_device_pending_restore_and_queued_build(device,tmp_path,backend,queued):
    values=[]
    for split in [False,True]:
        device.reinit();opts=dict(numeric_mode='float32',event_delivery='sparse') if backend!='reference' else {}
        setup(tmp_path/str(split),engine=backend,build_on_run=not queued,**opts)
        net,syn=network(uniform=True,early=True,named=True)
        if split:
            net.run(6*DT)
            if not queued:net.store('pending')
            net.run(6*DT)
            if not queued:
                expect=np.asarray(syn.w[:]).copy();net.restore('pending');net.run(6*DT);np.testing.assert_array_equal(syn.w[:],expect)
        else:net.run(12*DT)
        if queued:device.build()
        values.append([np.asarray(getattr(syn,name)[:]).copy() for name in ['w','Apre','Apost','hits']])
    for a,before in zip(*values,strict=True):np.testing.assert_array_equal(a,before)


@pytest.mark.parametrize('backend',BACKENDS)
def test_last_edge_vector_fault_is_not_lost(device,tmp_path,backend):
    model=model_at(tmp_path/'ref',pending=False)
    syn=model['instance']['synapses'][0]
    syn['initial_state']['hits']=['0000000000000001']*129;syn['initial_state']['hits'][-1]='0000000000000000'
    code=next(c for c in model['definition']['synapses'][0]['code_objects'] if c['kind']=='synapses')
    code['vector'].insert(0,dict(target='_checked',dtype='i64',dimensions=[0.]*7,condition=None,
        value=dict(op='floor_div',left=dict(op='integer',dtype='i64',value='1'),right=dict(op='load',name='hits'))))
    refresh_code(model,code)
    with canonical_pathways():
        with pytest.raises(FloatingPointError):control(model,tmp_path/'old',1)
    with pytest.raises(FloatingPointError):execute(model,tmp_path/backend,backend,'sparse')


@pytest.mark.parametrize('backend',BACKENDS)
def test_owned_rng_timed_lookup_and_duplicate_pending(device,tmp_path,backend):
    setup(tmp_path/'ref');b.seed(129)
    pop=b.NeuronGroup(9,'v:1',threshold='v>0.5',reset='',dt=DT);pop.v=1
    drive=b.TimedArray(np.arange(36).reshape(4,9)/128,dt=2*DT)
    syn=b.Synapses(pop,pop,'w:1\nhits:integer',on_pre='w+=rand()+drive(t,i); hits+=1',
        namespace={'drive':drive},clock=pop.clock,dtype={'hits':np.int64})
    syn.connect(i=np.arange(129)%9,j=np.arange(129)*3%9);syn.delay=np.arange(129)%4*DT;syn.hits=2**40
    model=lower_network(b.Network(pop,syn),12*DT)
    model['instance']['synapses'][0]['pathways'][0]['pending']=[dict(delivery_tick=0,item=i) for i in [128,0,128]]
    attach_protocol(model)
    with canonical_pathways():expected=control(model,tmp_path/'canonical',1)
    result_exact(execute(model,tmp_path/backend,backend,'sparse'),expected)
    assert min(expected['synapses'][0]['states']['hits'])>2**40


@real_cuda
def test_cuda_modes_reset_per_edge_cursors_counters_and_history(device,tmp_path):
    from brian2_rust.cuda import CudaExecutor
    model=model_at(tmp_path/'ref',early=True,named=True)
    with canonical_pathways():expected=control(model,tmp_path/'canonical',1)
    with CudaExecutor(model,tmp_path/'cuda',numeric_mode='float32',event_delivery='sparse') as ex:
        for mode in ['direct','resident','graph','chunked','graph','chunked']:
            result=ex.run(dag_execution=mode);result_exact(result,expected)
            assert result['cuda_runtime']['dag_execution']['selected']==mode
            if mode!='direct':
                with ex.device,ex.stream:
                    for i in ex._resident_dag.writable:ex._resident_dag.gpu[i].fill(99)
