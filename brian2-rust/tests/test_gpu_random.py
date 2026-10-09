"""Counter identities, distribution checks and native GPU RNG lifecycle."""
import copy
from dataclasses import replace
import json
import subprocess

import brian2 as b
from brian2.input.binomial import BinomialFunction
import numpy as np
import pytest

from brian2_rust.export import lower_network
from brian2_rust.metal import build_metal_plan,write_metal_results
from brian2_rust.cuda import build_cuda_plan,write_cuda_results
from brian2_rust.metal_random import RNG_PROFILE
from brian2_rust.plan import bind_execution_plan,explain_plan,verify_execution_plan,PlanValidationError
from brian2_rust.results import load_results
from brian2_rust.protocol import attach_protocol
from brian2_rust.spec import bits
from test_gpu_spike_generator import BACKENDS,execute
from test_metal_delays import device,ROOT
from test_metal_plasticity import equivalent

SEED=1729
MASK=(1<<64)-1


def uniform(stream,tick,index,draw=0):
    x=SEED ^ ((stream*0x9e3779b97f4a7c15)&MASK) ^ ((tick*0xd1b54a32d192ed03)&MASK)
    x^=((index*0x94d049bb133111eb)&MASK)^((draw*0x369dea0f31a53f85)&MASK)
    x=((x^(x>>30))*0xbf58476d1ce4e5b9)&MASK
    x=((x^(x>>27))*0x94d049bb133111eb)&MASK
    return np.float32(((x^(x>>31))>>40)/2**24)


def random_nodes(tree):
    if isinstance(tree,dict):
        if tree.get('op') in {'rand','randn','binomial'}:yield tree
        for value in tree.values():yield from random_nodes(value)
    elif isinstance(tree,list):
        for value in tree:yield from random_nodes(value)


def setup(device,tmp_path):
    b.set_device('rust_standalone',engine='reference',directory=tmp_path/'ref',runner=ROOT/'target/release/b2-runner')
    b.seed(SEED)


def lower(net,steps=1):return lower_network(net,steps*b.second/1024,rng_seed=SEED)


@pytest.mark.parametrize('backend',BACKENDS)
def test_uniform_counter_projection_and_profile(device,tmp_path,backend):
    setup(device,tmp_path)
    pop=b.NeuronGroup(257,'u:1\nv:1',dt=b.second/1024)
    pop.run_regularly('u=rand(); v=rand()')
    model=lower(b.Network(pop),9)
    # Preserve integer counter distinctions beyond f32's exact integer range.
    model['run']['start']=bits(2**25/1024);model['run']['clocks'][0]['start_tick']=2**25
    attach_protocol(model)
    plan=build_cuda_plan(model,numeric_mode='float32')
    assert plan.rng_profile==RNG_PROFILE and RNG_PROFILE in explain_plan(plan)
    with pytest.raises(PlanValidationError):verify_execution_plan(replace(plan,rng_profile=None),model)
    actual=execute(model,tmp_path/backend,backend)
    assert actual['rng_profile']==RNG_PROFILE
    nodes=list(random_nodes(model['definition']))
    assert len(nodes)==2
    for name,node in zip(('u','v'),nodes,strict=True):
        expected=np.array([uniform(node['stream'],2**25+8,i) for i in range(257)])
        np.testing.assert_array_equal(actual['populations'][0]['states'][name],expected)
        assert np.all((expected>=0)&(expected<1))
    if backend!='cpu-f32':
        metadata=load_results(model,tmp_path/backend/'transport')['metadata']
        assert metadata['rng_profile']==RNG_PROFILE
        native_plan=build_metal_plan(model,numeric_mode='float32') if backend=='metal' else plan
        bind_execution_plan(native_plan,metadata)
        with pytest.raises(PlanValidationError):bind_execution_plan(native_plan,{**metadata,'rng_profile':None})
        writer=write_metal_results if backend=='metal' else write_cuda_results
        with pytest.raises(PlanValidationError,match='RNG profile'):
            writer(model,{**actual,'rng_profile':None},tmp_path/'invalid-profile')
        assert not (tmp_path/'invalid-profile').exists()


@pytest.mark.parametrize('backend',BACKENDS)
def test_normal_paired_stream_and_statistics(device,tmp_path,backend):
    setup(device,tmp_path)
    pop=b.NeuronGroup(8192,'z:1',dt=b.second/1024);pop.run_regularly('z=randn()')
    model=lower(b.Network(pop));actual=execute(model,tmp_path/backend,backend)['populations'][0]['states']['z']
    stream=next(random_nodes(model['definition']))['stream'];expected=[]
    for pair in range(4096):
        for draw in range(0,8192,2):
            a=np.float32(2*uniform(stream,0,pair,draw)-1);c=np.float32(2*uniform(stream,0,pair,draw+1)-1)
            radius=np.float32(a*a+c*c)
            if 0<radius<1:
                factor=np.float32(np.sqrt(np.float32(-2*np.log(radius)/radius)))
                expected.extend([np.float32(factor*a),np.float32(factor*c)]);break
    np.testing.assert_allclose(actual,expected,rtol=3e-6,atol=3e-6)
    assert abs(actual.mean())<.06
    assert abs(actual.var()-1)<.08


@pytest.mark.parametrize('backend',BACKENDS)
def test_poisson_group_dyadic_probabilities_match_reference(device,tmp_path,backend):
    setup(device,tmp_path)
    probabilities=np.resize([0,.25,.5,1],64)
    pop=b.PoissonGroup(64,probabilities*1024*b.Hz,dt=b.second/1024)
    monitor=b.SpikeMonitor(pop);model=lower(b.Network(pop,monitor),32)
    path=tmp_path/'model.json';path.write_text(json.dumps(model))
    subprocess.run([str(ROOT/'target/release/b2-runner'),str(path),str(tmp_path/'oracle')],check=True)
    expected=load_results(model,tmp_path/'oracle')
    actual=execute(model,tmp_path/backend,backend)
    equivalent(actual,expected,exact=True)


@pytest.mark.parametrize('backend',BACKENDS)
def test_poisson_input_extremes_and_binomial_statistics(device,tmp_path,backend):
    setup(device,tmp_path)
    pop=b.NeuronGroup(4096,'a:1\nb:1\nc:1\nd:1',dt=b.second/1024)
    inputs=[b.PoissonInput(pop,name,n,p*1024*b.Hz,weight=1) for name,n,p in [('a',8,.25),('b',100,.2),('c',8,0),('d',8,1)]]
    model=lower(b.Network(pop,*inputs));actual=execute(model,tmp_path/backend,backend)['populations'][0]['states']
    np.testing.assert_array_equal(actual['c'],0);np.testing.assert_array_equal(actual['d'],8)
    assert np.all((actual['a']>=0)&(actual['a']<=8)&(actual['a']==np.floor(actual['a'])))
    assert abs(actual['a'].mean()-2)<.12 and abs(actual['a'].var()-1.5)<.2
    assert abs(actual['b'].mean()-20)<.4 and abs(actual['b'].var()-16)<2


@pytest.mark.parametrize('backend',BACKENDS)
def test_synaptic_counter_uses_edge_and_delivery_tick(device,tmp_path,backend):
    setup(device,tmp_path)
    source=b.SpikeGeneratorGroup(4,[0,1,2,3],[0,0,0,0]*b.second,period=b.second/1024,dt=b.second/1024)
    target=b.NeuronGroup(5,'v:1',clock=source.clock)
    syn=b.Synapses(source,target,'w:1',on_pre='w=rand(); v_post+=w',clock=source.clock)
    syn.connect(i=np.arange(257)%4,j=np.arange(257)%5);syn.delay=(np.arange(257)%3)*source.clock.dt
    model=lower(b.Network(source,target,syn),8)
    stream=next(random_nodes(model['definition']['synapses']))['stream']
    for route in ('scan','sparse'):
        actual=execute(model,tmp_path/(backend+route),backend,route)
        np.testing.assert_array_equal(actual['synapses'][0]['states']['w'],[uniform(stream,7,e) for e in range(257)])
        control=execute(model,tmp_path/('control'+route),'cpu-f32',route)
        equivalent(actual,control,exact=True)
    invalid=copy.deepcopy(model)
    node=next(random_nodes(invalid['definition']['synapses']))
    node.update(op='binomial',n=8,p={'op':'literal','bits':bits(-1)},approximate=False)
    attach_protocol(invalid)
    for route in ('scan','sparse'):
        path=tmp_path/('invalid-'+backend+route)
        with pytest.raises(FloatingPointError,match='sampler'):
            execute(invalid,path,backend,route)
        assert not (path/'transport').exists()


@pytest.mark.parametrize('backend',BACKENDS)
def test_exact_binomial_and_underflow_rejection_sampling(device,tmp_path,backend):
    setup(device,tmp_path)
    sample=BinomialFunction(16,.5,approximate=False,name='draw_small')
    pop=b.NeuronGroup(4096,'x:1',dt=b.second/1024,namespace={'sample':sample});pop.run_regularly('x=sample()')
    model=lower(b.Network(pop));actual=execute(model,tmp_path/backend,backend)['populations'][0]['states']['x']
    assert abs(actual.mean()-8)<.2 and abs(actual.var()-4)<.5
    node=next(random_nodes(model['definition']));node['n']=10000;attach_protocol(model)
    large=execute(model,tmp_path/'underflow',backend)['populations'][0]['states']['x']
    assert np.all((large>=0)&(large<=10000)&(large==np.floor(large)))
    assert abs(large.mean()-5000)<5 and abs(large.var()-2500)<250


@pytest.mark.parametrize('backend',BACKENDS)
def test_invalid_binomial_probability_is_not_published(device,tmp_path,backend):
    setup(device,tmp_path)
    sample=BinomialFunction(8,.25,approximate=False,name='draw_bad')
    pop=b.NeuronGroup(2,'x:1',dt=b.second/1024,namespace={'sample':sample});pop.run_regularly('x=sample()')
    model=lower(b.Network(pop));next(random_nodes(model['definition']))['p']={'op':'literal','bits':bits(-1)};attach_protocol(model)
    with pytest.raises(FloatingPointError,match='sampler'):execute(model,tmp_path/backend,backend)
    assert not (tmp_path/backend/'transport').exists()


@pytest.mark.parametrize('backend',BACKENDS[1:])
def test_poisson_device_segmentation_seed_restore(device,tmp_path,backend):
    results=[]
    for segmented in (False,True):
        device.reinit();b.set_device('rust_standalone',engine=backend,numeric_mode='float32',
          directory=tmp_path/str(segmented),runner=ROOT/'target/release/b2-runner');b.seed(SEED)
        source=b.PoissonGroup(64,np.resize([0,.25,.5,1],64)*1024*b.Hz,dt=b.second/1024,name='source')
        mon=b.SpikeMonitor(source);net=b.Network(source,mon)
        if segmented:
            net.run(3*source.clock.dt);net.store('seed');net.run(9*source.clock.dt)
            expected=np.asarray(mon.t[:]).copy();b.seed(999)
            net.restore('seed',restore_random_state=True);net.run(9*source.clock.dt)
            np.testing.assert_array_equal(np.asarray(mon.t[:]),expected)
        else:net.run(12*source.clock.dt)
        results.append((np.asarray(mon.i[:]).copy(),np.asarray(mon.t[:]).copy()))
    for a,e in zip(*results,strict=True):np.testing.assert_array_equal(a,e)
