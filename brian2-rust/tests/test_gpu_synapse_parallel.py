"""Edge-owned state updates: order, typed storage, faults and real GPU replay."""
from contextlib import contextmanager
from copy import deepcopy
import json
from types import SimpleNamespace
import brian2 as b
import numpy as np
import pytest
from brian2_rust import metal_synapses
from brian2_rust.metal import build_metal_plan
from brian2_rust.metal_dag import run_dag
from brian2_rust.export import lower_network
from brian2_rust.spec import bits
from test_cuda_graphs import result_exact
from test_gpu_spike_generator import BACKENDS,execute
from test_gpu_monitors import setup,DT
from test_gpu_refractory import refresh_code
from test_metal_delays import device


@contextmanager
def serial_policy():
    old=metal_synapses.independent_state_update
    metal_synapses.independent_state_update=lambda syn,code:False
    try:yield
    finally:metal_synapses.independent_state_update=old


def model_at(path,edges=135,flavor="plain"):
    z={"plain":"a+b","random":"a+b+rand()","timed":"a+b+drive(t,i)"}[flavor]
    drive=b.TimedArray(np.arange(44).reshape(4,11)/64,dt=2*DT)
    setup(path)
    pre=b.NeuronGroup(11,'v:1',threshold='v>0.5',reset='',dt=DT);pre.v=1
    post=b.NeuronGroup(7,'v:1',threshold='v>0.5',reset='',dt=2*DT);post.v=1
    syn=b.Synapses(pre,post,f'da/dt=b/(8*tau):1 (clock-driven)\ndb/dt=a/(16*tau):1 (clock-driven)\nz={z}:1 (constant over dt)\nk:integer',
        on_pre='a+=0.125',on_post='b+=0.25',method='euler',namespace={'tau':DT,'drive':drive},dtype={'k':np.int64},clock=pre.clock)
    syn.connect(i=np.arange(edges)%11,j=(np.arange(edges)*3)%7);syn.a=.5;syn.b=.25
    syn.pre.delay=(np.arange(edges)%4)*DT;syn.post.delay=2*DT
    model=lower_network(b.Network(pre,post,syn),16*DT)
    # Include exact multiword stores in the subexpression node; the frozen
    # validator admits any own synapse state here, including integer states.
    code=next(c for c in model['definition']['synapses'][0]['code_objects'] if c['kind']=='synapse_subexpression_update')
    code['vector'].append(dict(target='k',dtype='i64',dimensions=[0.]*7,condition=None,
        value=dict(op='add',left=dict(op='load',name='k'),right=dict(op='integer',dtype='i64',value='1'))))
    code['effects']['writes']=sorted([*code['effects']['writes'],'k'])
    refresh_code(model,code)
    return model


def control(model,path,workers):
    path.mkdir();ex=SimpleNamespace(model=model,plan=build_metal_plan(model,numeric_mode='float32',event_delivery='sparse'),directory=path,compile_seconds=0,device_name='CPU f32 control')
    return run_dag(ex,max_buffer_bytes=512*1024**2,compute='cpu-f32',workers=workers)


@pytest.mark.parametrize('edges',[0,1,135])
def test_plan_owns_only_edge_state_and_preserves_other_stages(device,tmp_path,edges):
    model=model_at(tmp_path/'ref',edges)
    with serial_policy():before=build_metal_plan(model,numeric_mode='float32',event_delivery='sparse')
    after=build_metal_plan(model,numeric_mode='float32',event_delivery='sparse')
    assert before.logical==after.logical and before.buffers==after.buffers
    selected=[d for d in after.dispatches if d.role=='parallel-synapse-state']
    assert len(selected)==2 and all(d.lanes==max(1,edges) for d in selected)
    for a,before_kernel,after_kernel in zip(after.dispatches,before.kernels,after.kernels,strict=True):
        if a.role=='parallel-synapse-state':
            assert a.types[0:2]==('const float','const float') and a.types[5]=='atomic_uint'
        else:assert before_kernel==after_kernel


def test_proof_rejects_population_shared_event_and_scalar_rng_writes(device,tmp_path):
    model=model_at(tmp_path/'ref');syn=model['definition']['synapses'][0]
    original=next(c for c in syn['code_objects'] if c['kind']=='synapse_state_update')
    assert metal_synapses.independent_state_update(syn,original)
    for field,value in [('kind','synapses'),('iteration_domain','active_synapses'),('scalar',[{'op':'rand','stream':0}])]:
        code=deepcopy(original);code[field]=value;assert not metal_synapses.independent_state_update(syn,code)
    code=deepcopy(original);code['effects']['writes'].append('v_post');assert not metal_synapses.independent_state_update(syn,code)
    shared=deepcopy(syn)
    for s in shared['states']:
        if s['name'] in original['effects']['writes']:s['index_domain']='scalar'
    assert not metal_synapses.independent_state_update(shared,original)


@pytest.mark.parametrize('backend',BACKENDS)
def test_exact_coupled_delayed_typed_state_and_subexpression_replay(device,tmp_path,backend):
    model=model_at(tmp_path/'ref')
    with serial_policy():expected=control(model,tmp_path/'old',1)
    result_exact(control(model,tmp_path/'parallel-cpu',3),expected)
    result_exact(execute(model,tmp_path/backend,backend,'sparse'),expected)
    assert all(v==16 for v in expected['synapses'][0]['states']['k'])


@pytest.mark.parametrize('backend',BACKENDS)
@pytest.mark.parametrize('where',['last-edge','all-edges','empty-scalar'])
def test_parallel_checked_faults_are_never_lost(device,tmp_path,backend,where):
    model=model_at(tmp_path/'ref',0 if where=='empty-scalar' else 135)
    code=next(c for c in model['definition']['synapses'][0]['code_objects'] if c['kind']=='synapse_subexpression_update')
    bad=dict(op='floor_div',left=dict(op='literal',bits=bits(1.)),right=dict(op='literal',bits=bits(0.)))
    condition=None
    if where=='last-edge':
        # Only the final edge's counter becomes zero. This catches a last-warp
        # fault dropped by a non-atomic reduction; no new public buffer needed.
        syn=model['instance']['synapses'][0];syn['initial_state']['k']=['0000000000000001']*135;syn['initial_state']['k'][-1]='0000000000000000'
        bad=dict(op='floor_div',left=dict(op='integer',dtype='i64',value='1'),right=dict(op='load',name='k'))
    statement=dict(target='_checked',dtype='i64' if where=='last-edge' else 'f64',dimensions=[0.]*7,condition=condition,value=bad)
    code['scalar' if where=='empty-scalar' else 'vector'].insert(0,statement);refresh_code(model,code)
    with serial_policy():
        with pytest.raises(FloatingPointError):control(model,tmp_path/'old',1)
    with pytest.raises(FloatingPointError):execute(model,tmp_path/backend,backend,'sparse')


@pytest.mark.parametrize('backend',BACKENDS)
@pytest.mark.parametrize('flavor',['random','timed'])
def test_edge_counter_rng_and_timed_population_lookup(device,tmp_path,backend,flavor):
    model=model_at(tmp_path/'ref',flavor=flavor)
    with serial_policy():expected=control(model,tmp_path/'old',1)
    result_exact(execute(model,tmp_path/backend,backend,'sparse'),expected)
