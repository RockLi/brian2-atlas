"""Endpoint-owned summed reductions preserve canonical edge order and faults."""
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import subprocess
import brian2 as b
import numpy as np
import pytest
from brian2_rust import metal_synapses
from brian2_rust.export import lower_network
from brian2_rust.metal import build_metal_plan
from brian2_rust.metal_dag import run_dag
from brian2_rust.results import load_results
from brian2_rust.spec import bits
from test_cuda_graphs import result_exact
from test_gpu_monitors import setup, DT
from test_gpu_refractory import refresh_code
from test_gpu_spike_generator import BACKENDS, execute
from test_metal_delays import device, ROOT


@contextmanager
def serial_policy():
    old=metal_synapses.independent_summed
    metal_synapses.independent_summed=lambda model,syn,code:False
    try:yield
    finally:metal_synapses.independent_summed=old


def model_at(path,edges=273,flavor='plain',steps=8):
    setup(path)
    pre=b.NeuronGroup(73,'v:1\nu:1',threshold='v>0.5',reset='',dt=DT,name='pre');pre.v=1;pre.u=-9
    post=b.NeuronGroup(69,'v:1\nu:1',threshold='v>0.5',reset='',dt=2*DT,name='post');post.v=1;post.u=-7
    drive=b.TimedArray(np.arange(8)/64,dt=2*DT)
    extra={'plain':'','random':'+rand()','timed':'+drive(t)'}[flavor]
    syn=b.Synapses(pre[1:72],post[2:69],f'w:1\nk:integer\nu_pre=w{extra}:1 (summed)\nu_post=w+u_pre/64{extra}:1 (summed)',
                   on_pre='w+=0.125',on_post='w-=0.0625',namespace={'drive':drive},clock=pre.clock,name='projection')
    syn.connect(i=(np.arange(edges)*13)%70,j=(np.arange(edges)*19)%66)
    syn.w=(np.arange(edges)%17)/64;syn.k=1
    syn.pre.delay=(np.arange(edges)%4)*DT;syn.post.delay=2*DT
    monitors=[b.StateMonitor(p,'u',record=True,name='trace_'+p.name) for p in (pre,post)]
    return lower_network(b.Network(pre,post,syn,*monitors),steps*DT)


def control(model,path):
    path.mkdir()
    ex=SimpleNamespace(model=model,plan=build_metal_plan(model,numeric_mode='float32',event_delivery='sparse'),directory=path,compile_seconds=0,device_name='CPU f32 control')
    return run_dag(ex,max_buffer_bytes=512*1024**2,compute='cpu-f32',workers=3)


@pytest.mark.parametrize('edges',[0,1,273])
def test_plan_owns_both_endpoint_reductions_and_preserves_inventory(device,tmp_path,edges):
    model=model_at(tmp_path/'ref',edges)
    with serial_policy():old=build_metal_plan(model,numeric_mode='float32',event_delivery='sparse')
    new=build_metal_plan(model,numeric_mode='float32',event_delivery='sparse')
    assert old.logical==new.logical and old.buffers==new.buffers
    stages=[s for s in new.dispatches if s.role=='endpoint-owned-summed']
    assert sorted(s.lanes for s in stages)==[67,71]
    assert all(s.types[5]=='atomic_uint' and s.types[-2:]==('const uint','const uint') for s in stages)


@pytest.mark.parametrize('backend',BACKENDS)
@pytest.mark.parametrize('edges',[0,1,273])
def test_delayed_mutable_subgroup_multiclock_reductions(device,tmp_path,backend,edges):
    model=model_at(tmp_path/'ref',edges)
    with serial_policy():expected=control(model,tmp_path/'serial')
    result_exact(execute(model,tmp_path/backend,backend,'sparse'),expected)
    # Untargeted subgroup endpoints reset; neurons outside each subgroup retain state.
    q=model['definition']['synapses'][0]
    pre=expected['populations'][q['source_population']]['states']['u']
    post=expected['populations'][q['target_population']]['states']['u']
    assert pre[0]==-9
    assert pre[72]==-9
    assert post[0]==-7
    assert post[1]==-7
    assert pre[71]==0
    assert post[68]==0


@pytest.mark.parametrize('backend',BACKENDS)
@pytest.mark.parametrize('flavor',['random','timed'])
def test_edge_rng_identity_and_timed_table_are_preserved(device,tmp_path,backend,flavor):
    model=model_at(tmp_path/'ref',flavor=flavor)
    with serial_policy():expected=control(model,tmp_path/'serial')
    result_exact(execute(model,tmp_path/backend,backend),expected)


def test_proof_rejects_destination_reads_and_scalar_rng(device,tmp_path):
    model=model_at(tmp_path/'ref');syn=model['definition']['synapses'][0]
    for original in [c for c in syn['code_objects'] if c['kind']=='summed_variable']:
        assert metal_synapses.independent_summed(model,syn,original)
        code=deepcopy(original);code['effects']['reads'].append('u_'+code['summed_target'])
        assert not metal_synapses.independent_summed(model,syn,code)
        code=deepcopy(original);code['scalar']=[{'op':'rand','stream':0}]
        assert not metal_synapses.independent_summed(model,syn,code)
    recurrent=deepcopy(syn);recurrent['target_population']=recurrent['source_population']
    code=deepcopy(next(c for c in syn['code_objects'] if c['kind']=='summed_variable' and c['summed_target']=='post'))
    assert 'u_pre' in code['effects']['reads']
    assert not metal_synapses.independent_summed(model,recurrent,code)


@pytest.mark.parametrize('backend',BACKENDS)
@pytest.mark.parametrize('where',['last-edge','empty-scalar'])
def test_checked_reduction_faults_remain_visible(device,tmp_path,backend,where):
    model=model_at(tmp_path/'ref',0 if where=='empty-scalar' else 273)
    code=next(c for c in model['definition']['synapses'][0]['code_objects'] if c['kind']=='summed_variable')
    bad=dict(op='floor_div',left=dict(op='literal',bits=bits(1.)),right=dict(op='literal',bits=bits(0.)))
    if where=='last-edge':
        model['instance']['synapses'][0]['initial_state']['k']=['00000001']*273
        model['instance']['synapses'][0]['initial_state']['k'][-1]='00000000'
        bad=dict(op='floor_div',left=dict(op='integer',dtype='i32',value='1'),right=dict(op='load',name='k'))
        code['effects']['reads']=sorted(set(code['effects']['reads'])|{'k'})
    stmt=dict(target='_fault',dtype='i32' if where=='last-edge' else 'f64',dimensions=[0.]*7,condition=None,value=bad)
    code['scalar' if where=='empty-scalar' else 'vector'].insert(0,stmt);refresh_code(model,code)
    with serial_policy():
        with pytest.raises(FloatingPointError):control(model,tmp_path/'serial')
    with pytest.raises(FloatingPointError):execute(model,tmp_path/backend,backend)


@pytest.mark.parametrize('backend',BACKENDS)
def test_original_edge_order_cancellation_and_independent_reference(device,tmp_path,backend):
    model=model_at(tmp_path/'ref',3,steps=2)
    for pop,inst_pop in zip(model['definition']['populations'],model['instance']['populations']):
        inst_pop['initial_state']['v']=[bits(0.)]*pop['count']
    inst=model['instance']['synapses'][0];inst['source']=[0,0,0];inst['target']=[0,0,0]
    inst['initial_state']['w']=[bits(x) for x in (2**24,1.,-2**24)]
    from brian2_rust.protocol import attach_protocol
    attach_protocol(model)
    with serial_policy():expected=control(model,tmp_path/'serial')
    actual=execute(model,tmp_path/backend,backend)
    result_exact(actual,expected)
    # f32 must retain ((2**24 + 1) - 2**24) == 0, never tree-reassociate to 1.
    q=model['definition']['synapses'][0]
    assert actual['populations'][q['source_population']]['states']['u'][1]==0
    assert actual['populations'][q['target_population']]['states']['u'][2]==0
    # The independent f64 engine deliberately sees the extra unit instead.
    path=tmp_path/'model.json';path.write_text(json.dumps(model))
    subprocess.run([str(ROOT/'target/release/b2-runner'),str(path),str(tmp_path/'oracle')],check=True,capture_output=True)
    reference=load_results(model,tmp_path/'oracle')
    assert reference['populations'][q['source_population']]['states']['u'][1]==1


@pytest.mark.parametrize('backend',BACKENDS)
def test_recurrent_destination_dependency_keeps_global_order(device,tmp_path,backend):
    setup(tmp_path/'ref')
    pop=b.NeuronGroup(4,'v:1\nu:1',threshold='v>0.5',reset='',dt=DT,name='recurrent')
    pop.u=9
    syn=b.Synapses(pop,pop,'w:1\nu_post=w+u_pre:1 (summed)',on_pre='w+=0.125',clock=pop.clock)
    syn.connect(i=[0,1,0],j=[1,2,3]);syn.w=1
    model=lower_network(b.Network(pop,syn),DT)
    plan=build_metal_plan(model,numeric_mode='float32')
    assert any(s.role=='canonical-synapse' for s in plan.dispatches)
    assert not any(s.role=='endpoint-owned-summed' for s in plan.dispatches)
    actual=execute(model,tmp_path/backend,backend)
    np.testing.assert_array_equal(actual['populations'][0]['states']['u'],[0,1,2,1])
