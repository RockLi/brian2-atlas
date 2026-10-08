"""Tracked serial Metal dispatches preserve cross-lane/tick data dependencies."""
import json

import brian2 as b
import numpy as np
import pytest

from brian2_rust.export import lower_network
from brian2_rust.metal import MetalExecutor
from brian2_rust.metal_buffers import validate_synchronization
from test_cuda_graphs import result_exact
from test_gpu_summed_parallel import model_at,control
from test_gpu_links import reference
from test_gpu_monitors import setup,DT,event_equivalent
from test_metal import real_metal
from test_metal_delays import device


def test_synchronization_policy_validation():
    for p in ('tracked','explicit'):assert validate_synchronization(p)==p
    for p in (None,[],1,True,'concurrent','off'):
        with pytest.raises(ValueError):validate_synchronization(p)


def check_runtime(result,plan,policy):
    m=result['metal_runtime']
    expected=sum(plan.logical.clocks[d.clock].steps for d in plan.dispatches)
    assert m['synchronization']==policy
    assert m['dispatch_type']=='serial' and m['hazard_tracking']=='tracked'
    assert m['dispatches']==expected>0
    assert m['explicit_barriers']==(expected if policy=='explicit' else 0)


@real_metal
@pytest.mark.parametrize('route',['scan','sparse'])
@pytest.mark.parametrize('edges',[0,273])
def test_delayed_mutable_multiclock_graph_across_command_chunks(device,tmp_path,route,edges):
    model=model_at(tmp_path/'ref',edges=edges,steps=130)
    expected=control(model,tmp_path/'cpu')
    (tmp_path/'model.json').write_text(json.dumps(model,sort_keys=True)+'\n')
    with MetalExecutor(model,tmp_path/'metal',numeric_mode='float32',event_delivery=route) as ex:
        plan_sha=ex.plan.sha256
        # Paired timings do not establish a general win, so preserve the old default.
        check_runtime(ex.run(),ex.plan,'explicit')
        for memory in ('direct','resident'):
            for policy in ('explicit','tracked','tracked','explicit'):
                r=ex.run(dag_execution=memory,dag_synchronization=policy)
                result_exact(r,expected);check_runtime(r,ex.plan,policy)
        assert ex.plan.sha256==plan_sha


def linked_model(path):
    setup(path);n=769
    source=b.NeuronGroup(n,'counter:integer',dt=DT,dtype={'counter':np.int64},name='a_source')
    initial=np.arange(n,dtype=np.int64)*17+2**54
    source.counter=initial;source.run_regularly('counter+=1',when='groups')
    target=b.NeuronGroup(n,'remote:integer (linked)\nseen:integer\nmapping:integer (constant)',
                        dt=DT,dtype={'remote':np.int64,'seen':np.int64,'mapping':np.int64},name='z_target')
    mapping=np.arange(n-1,-1,-1,dtype=np.int64)
    target.mapping=mapping;target.remote=b.linked_var(source,'counter',index='mapping')
    target.run_regularly('seen=remote',when='groups')
    monitor=b.StateMonitor(target,['remote','seen'],record=[0,256,768])
    model=lower_network(b.Network(source,target,monitor),130*DT)
    return model,initial,mapping


def check_integer_dependency(model,result,initial,mapping):
    by_name={p['name']:v for p,v in zip(model['definition']['populations'],result['populations'],strict=True)}
    np.testing.assert_array_equal(by_name['a_source']['states']['counter'],initial+130)
    np.testing.assert_array_equal(by_name['z_target']['states']['seen'],initial[mapping]+130)
    return by_name['z_target']['states']['seen']


def test_integer_dependency_reference_oracle(device,tmp_path):
    model,initial,mapping=linked_model(tmp_path/'ref')
    expected=reference(model,tmp_path/'oracle')
    check_integer_dependency(model,expected,initial,mapping)


@real_metal
def test_integer_link_gather_observes_previous_dispatch_across_workgroups(device,tmp_path):
    model,initial,mapping=linked_model(tmp_path/'ref')
    (tmp_path/'model.json').write_text(json.dumps(model,sort_keys=True)+'\n')
    expected=reference(model,tmp_path/'oracle')
    records={}
    with MetalExecutor(model,tmp_path/'metal',numeric_mode='float32',dag_execution='resident') as ex:
        for policy in ('explicit','tracked'):
            r=ex.run(dag_synchronization=policy);event_equivalent(r,expected);check_runtime(r,ex.plan,policy)
            records[policy]=check_integer_dependency(model,r,initial,mapping)
    np.savez_compressed(tmp_path/'integer-results.npz',initial=initial,mapping=mapping,**records)
