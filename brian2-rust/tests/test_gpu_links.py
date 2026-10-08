"""Linked-input snapshots, exact mapping indices and selected-lane faults."""
import json
import subprocess

import brian2 as b
import numpy as np
import pytest

from brian2_rust.cuda import build_cuda_plan
from brian2_rust.export import lower_network
from brian2_rust.results import load_results
from brian2_rust.protocol import attach_protocol
from brian2_rust.schedule import build_schedule
from test_gpu_spike_generator import BACKENDS,execute
from test_gpu_monitors import setup,event_equivalent,DT
from test_metal_delays import device,ROOT


def refresh(model):
    d=model['definition'];d['schedule']=build_schedule(d,model['instance'],d['schedule']['base_slots'])
    attach_protocol(model)


def reference(model,path,*,error=None):
    path.mkdir();source=path/'model.json';source.write_text(json.dumps(model))
    result=subprocess.run([str(ROOT/'target/release/b2-runner'),str(source),str(path/'result')],capture_output=True,text=True)
    if error:
        assert result.returncode!=0 and error in result.stderr,result.stderr
        return
    assert result.returncode==0,result.stderr
    return load_results(model,path/'result')


def linked_network(feedback=False):
    source=b.NeuronGroup(4,'dx/dt=(i+1)/second:1\nwide:integer\nflag:boolean',dt=DT,method='euler',
                         dtype=dict(wide=np.int64),name='source')
    source.x=[.25,.5,1,2];source.wide=np.array([2**53+1,2**53+2,2**53+3,2**53+4],np.int64)
    source.run_regularly('wide+=1; flag=not flag',when='groups')
    target=b.NeuronGroup(4,'''dy/dt=(direct+fixed+dynamic)/second:1
        direct:1 (linked)
        fixed:1 (linked)
        dynamic:1 (linked)
        wide_link:integer (linked)
        flag_link:boolean (linked)
        seen:integer
        choice:boolean
        k:integer
        mapping:integer (constant)''',threshold='k%2==0',reset='',dt=2*DT,method='euler',
        dtype=dict(seen=np.int64,wide_link=np.int64,mapping=np.int64),name='target')
    target.k=[0,1,2,3];target.mapping=[3,1,1,0]
    target.direct=b.linked_var(source,'x');target.fixed=b.linked_var(source,'x',index='mapping')
    target.dynamic=b.linked_var(source,'x',index='k')
    target.wide_link=b.linked_var(source,'wide',index='k');target.flag_link=b.linked_var(source,'flag')
    target.run_regularly('seen=wide_link; choice=flag_link; k=(k+1)%N',when='end')
    variables=['y','direct','fixed','dynamic','wide_link','flag_link','k','seen','choice']
    monitors=[b.StateMonitor(target,variables,record=[3,0,2]),
              b.EventMonitor(target,'spike',variables=variables,when='end',order=1)]
    objects=[source,target,*monitors]
    if feedback:
        syn=b.Synapses(target,source,'w:1 (constant)',on_pre='x_post+=w',clock=target.clock)
        syn.connect(i=[0,1,2,3],j=[1,2,3,0]);syn.w=.125;syn.delay=np.array([0,1,2,3])*2*DT
        objects.append(syn)
    return b.Network(*objects),target,monitors


@pytest.mark.parametrize('backend',BACKENDS)
@pytest.mark.parametrize('parameter_index',[False,True])
def test_link_mappings_typed_monitors_and_coupled_clocks(device,tmp_path,backend,parameter_index):
    setup(tmp_path/'ref');net,target,monitors=linked_network()
    model=lower_network(net,8*DT)
    if parameter_index:
        pop=next(p for p in model['definition']['populations'] if p['name']=='target')
        next(link for link in pop['linked_variables'] if link['name']=='fixed')['index']=dict(kind='parameter',name='mapping')
        refresh(model)
    expected=reference(model,tmp_path/'expected')
    actual=execute(model,tmp_path/backend,backend)
    event_equivalent(actual,expected)
    if backend!='cpu-f32':event_equivalent(load_results(model,tmp_path/backend/'transport'),actual)


@pytest.mark.parametrize('backend',BACKENDS[1:])
@pytest.mark.parametrize('queued',[False,True])
def test_linked_device_segment_restore_and_queued(device,tmp_path,backend,queued):
    outputs=[]
    for engine in ('reference',backend):
        device.reinit();setup(tmp_path/engine,engine,build_on_run=not queued,
            **(dict(numeric_mode='float32',event_delivery='sparse') if engine!='reference' else {}))
        net,target,monitors=linked_network(feedback=True);net.run(4*DT)
        if not queued:net.store('snapshot')
        net.run(4*DT)
        if queued:device.build()
        else:
            before=np.asarray(target.seen[:]).copy();net.restore('snapshot');net.run(4*DT)
            np.testing.assert_array_equal(target.seen[:],before)
        outputs.append([np.asarray(target.y[:]).copy(),np.asarray(target.seen[:]).copy(),
                        np.asarray(monitors[0].dynamic[:]).copy(),np.asarray(monitors[1].wide_link[:]).copy()])
    for a,e in zip(*outputs,strict=True):np.testing.assert_array_equal(a,e)


@pytest.mark.parametrize('backend',BACKENDS)
@pytest.mark.parametrize('case',['snapshot','separate_blocks','masked','inactive_reset','selected_reset','unrecorded','unfired_monitor','unused'])
def test_link_input_snapshot_and_lane_selection(device,tmp_path,backend,case):
    setup(tmp_path/'ref')
    source=b.NeuronGroup(2,'x:1',dt=DT,name='source');source.x=[1,2]
    equation='dy/dt=external/second:1 (unless refractory)' if case=='masked' else 'y:1'
    options=dict(threshold='k==0',reset='y=external' if case in {'inactive_reset','selected_reset'} else '',refractory='True') if case in {'masked','inactive_reset','selected_reset','unfired_monitor'} else {}
    target=b.NeuronGroup(2,equation+'\nexternal:1 (linked)\nk:integer',clock=source.clock,method='euler',name='target',**options)
    target.k=[0,1] if case in {'snapshot','separate_blocks'} else ([0,2] if case in {'unrecorded','unfired_monitor','selected_reset'} else [2,2])
    target.external=b.linked_var(source,'x',index='k');objects=[source,target]
    if case=='snapshot':target.run_regularly('k=1-k; y=external',when='groups')
    if case=='separate_blocks':
        target.run_regularly('k=1-k',when='groups',order=-1)
        target.run_regularly('y=external',when='groups',order=1)
    if case=='unrecorded':objects.append(b.StateMonitor(target,'external',record=[0]))
    if case=='unfired_monitor':objects.append(b.EventMonitor(target,'spike',variables='external'))
    model=lower_network(b.Network(*objects),DT)
    if case=='masked':
        p=next(i for i,pop in enumerate(model['definition']['populations']) if pop['name']=='target')
        model['instance']['populations'][p]['refractory']['initial_not_refractory']=[False,False];refresh(model)
        reference(model,tmp_path/'expected',error='linked variable index out of bounds')
        with pytest.raises(FloatingPointError,match='linked index'):execute(model,tmp_path/backend,backend)
    else:
        expected=reference(model,tmp_path/'expected');actual=execute(model,tmp_path/backend,backend)
        event_equivalent(actual,expected)
        if case in {'snapshot','separate_blocks'}:
            p=next(i for i,pop in enumerate(model['definition']['populations']) if pop['name']=='target')
            np.testing.assert_array_equal(actual['populations'][p]['states']['y'],[1,2] if case=='snapshot' else [2,1])


@pytest.mark.parametrize('backend',BACKENDS)
@pytest.mark.parametrize('dtype,value',[(np.int64,-1),(np.uint64,2**64-1)])
def test_dynamic_index_never_rounds_or_wraps_to_valid_lane(device,tmp_path,backend,dtype,value):
    setup(tmp_path/'ref');source=b.NeuronGroup(1,'x:1',dt=DT)
    target=b.NeuronGroup(1,'y:1\nexternal:1 (linked)\nk:integer',clock=source.clock,dtype=dict(k=dtype))
    target.k=np.array([value],dtype=dtype);target.external=b.linked_var(source,'x',index='k')
    target.run_regularly('y=external',when='groups')
    model=lower_network(b.Network(source,target),DT)
    reference(model,tmp_path/'expected',error='conversion' if value<0 else 'index')
    with pytest.raises(FloatingPointError,match='linked index'):execute(model,tmp_path/backend,backend)


@pytest.mark.parametrize('backend',BACKENDS)
@pytest.mark.parametrize('reset',[False,True])
@pytest.mark.parametrize('dtype',[np.float64,np.int64])
def test_raw_self_mapping_preserves_reference_batch_and_reset_order(device,tmp_path,backend,reset,dtype):
    setup(tmp_path/'ref')
    options=dict(threshold='True',reset='x=external+1') if reset else {}
    kind='integer' if dtype==np.int64 else '1'
    pop=b.NeuronGroup(260,f'x:{kind}\nexternal:{kind} (linked)',dt=DT,dtype=dict(x=dtype,external=dtype),**options)
    pop.x=np.arange(260,dtype=dtype)+(2**53 if dtype==np.int64 else 0);pop.external=b.linked_var(pop,'x')
    if not reset:pop.run_regularly('x=external+1',when='groups')
    model=lower_network(b.Network(pop),2*DT)
    model['definition']['populations'][0]['linked_variables'][0]['index']=dict(kind='constant',values=[259]+list(range(259)))
    refresh(model)
    expected=reference(model,tmp_path/'expected');actual=execute(model,tmp_path/backend,backend)
    event_equivalent(actual,expected)
    plan=build_cuda_plan(model,numeric_mode='float32')
    assert any(stage.role=='canonical-linked-population' and stage.lanes==1 for stage in plan.dispatches)


@pytest.mark.parametrize('backend',BACKENDS)
def test_more_link_sources_than_metal_argument_slots(device,tmp_path,backend):
    setup(tmp_path/'ref')
    sources=[b.NeuronGroup(1,'x:1',dt=DT,name=f'source_{i:02}') for i in range(33)]
    pop=b.NeuronGroup(1,'y:1\n'+'\n'.join(f'link_{i}:1 (linked)' for i in range(33)),dt=DT,name='target')
    for i,source in enumerate(sources):
        source.x=(i+1)/4;setattr(pop,f'link_{i}',b.linked_var(source,'x'))
    pop.run_regularly('y='+'+'.join(f'link_{i}' for i in range(33)),when='groups')
    model=lower_network(b.Network(*sources,pop),DT)
    event_equivalent(execute(model,tmp_path/backend,backend),reference(model,tmp_path/'expected'))
    plan=build_cuda_plan(model,numeric_mode='float32')
    assert max(len(stage.bindings) for stage in plan.dispatches)<=14


@pytest.mark.parametrize('backend',BACKENDS)
@pytest.mark.parametrize('route',['scan','sparse'])
def test_linked_source_with_sparse_delay_fusion_and_feedback(device,tmp_path,backend,route):
    setup(tmp_path/'ref')
    base=b.NeuronGroup(3,'x:1',dt=DT,name='base');base.x=[.125,.25,.5]
    emitter=b.NeuronGroup(3,'v:1\nexternal:1 (linked)',clock=base.clock,threshold='v>=.5',reset='v=0',name='emitter')
    emitter.external=b.linked_var(base,'x');emitter.run_regularly('v+=external',when='groups')
    syn=b.Synapses(emitter,base,'w:1 (constant)',on_pre='x_post+=w',clock=emitter.clock)
    syn.connect(i=[0,1,2,2],j=[1,2,0,1]);syn.w=.125;syn.delay=np.array([0,1,2,1])*DT
    monitor=b.EventMonitor(emitter,'spike',variables=['v','external'],when='end')
    state=b.StateMonitor(emitter,['v','external'],record=True)
    model=lower_network(b.Network(base,emitter,syn,monitor,state),8*DT)
    actual=execute(model,tmp_path/backend,backend,route)
    event_equivalent(actual,reference(model,tmp_path/'expected'))
    if backend!='cpu-f32':event_equivalent(load_results(model,tmp_path/backend/'transport'),actual)
    plan=build_cuda_plan(model,numeric_mode='float32',event_delivery=route)
    if route=='sparse':assert any(stage.role=='population-delay-source-enqueue' for stage in plan.dispatches)


def test_parameter_indices_initialize_even_when_unused_and_cache_budget(device,tmp_path):
    from brian2_rust import gpu_links
    from brian2_rust.plan import PlanValidationError
    setup(tmp_path/'ref')
    source=b.NeuronGroup(2,'x:1',dt=DT,name='source')
    target=b.NeuronGroup(2,'y:1\nexternal:1 (linked)\nmapping:integer (constant)',clock=source.clock,
                         dtype=dict(mapping=np.int64),name='target')
    target.mapping=[0,1];target.external=b.linked_var(source,'x',index='mapping')
    model=lower_network(b.Network(source,target),DT)
    p=next(i for i,pop in enumerate(model['definition']['populations']) if pop['name']=='target')
    with pytest.raises(MemoryError,match='linked-input buffers'):gpu_links.arrays(model,p,12)
    model['definition']['populations'][p]['linked_variables'][0]['index']=dict(kind='parameter',name='mapping')
    model['instance']['populations'][p]['parameters']['mapping'][0]='ffffffffffffffff'
    refresh(model);reference(model,tmp_path/'expected',error='conversion')
    with pytest.raises(PlanValidationError,match='negative'):gpu_links.arrays(model,p,1024)


@pytest.mark.parametrize('backend',BACKENDS)
@pytest.mark.parametrize('zero_divisor',[False,True])
def test_floating_floor_operations_used_by_dynamic_index_expressions(device,tmp_path,backend,zero_divisor):
    setup(tmp_path/'ref')
    pop=b.NeuronGroup(4,'a:1 (constant)\nbb:1 (constant)\nrem:1\nquot:1',dt=DT)
    pop.a=[-7,7,-7,7];pop.bb=[0 if zero_divisor else 3,-3,-3,3]
    pop.run_regularly('rem=a%bb; quot=a//bb',when='groups')
    model=lower_network(b.Network(pop),DT)
    if zero_divisor:
        reference(model,tmp_path/'expected',error='non-finite')
        with pytest.raises(FloatingPointError):execute(model,tmp_path/backend,backend)
    else:event_equivalent(execute(model,tmp_path/backend,backend),reference(model,tmp_path/'expected'))


@pytest.mark.parametrize('backend',BACKENDS)
def test_identity_self_link_reads_state_update_snapshot(device,tmp_path,backend):
    setup(tmp_path/'ref')
    pop=b.NeuronGroup(3,'dx/dt=1/second:1\ndy/dt=external/second:1\nexternal:1 (linked)',dt=DT,method='euler')
    pop.x=[2,3,4];pop.external=b.linked_var(pop,'x')
    monitor=b.StateMonitor(pop,['x','y','external'],record=True)
    model=lower_network(b.Network(pop,monitor),4*DT)
    event_equivalent(execute(model,tmp_path/backend,backend),reference(model,tmp_path/'expected'))
