"""Canonical host topology initialization followed by native GPU simulation."""
import copy
from dataclasses import replace
import json
import math
import subprocess

import brian2 as b
import brian2_rust as rust
import numpy as np
import pytest

from brian2_rust.cuda import build_cuda_plan
from brian2_rust.metal import build_metal_plan
from brian2_rust.export import lower_network
from brian2_rust.gpu_initialization import prepare_model
from brian2_rust.plan import bind_execution_plan, verify_execution_plan, explain_plan, PlanValidationError
from brian2_rust.protocol import attach_protocol, canonical_bytes
from brian2_rust.results import load_results
from brian2_rust.spec import bits
from test_gpu_spike_generator import BACKENDS, execute
from test_metal_delays import device, ROOT
from test_metal_plasticity import equivalent

DT=b.second/1024
SEED=0x12345678
MASK=(1<<64)-1


def draw(stream,index,attempt=0):
    x=(SEED ^ ((stream*0xd2b74407b1ce6e93)&MASK) ^ ((index*0x9e3779b97f4a7c15)&MASK)
       ^ ((attempt*0xca5a826395121157)&MASK))
    x=(x+0x9e3779b97f4a7c15)&MASK
    x=((x^(x>>30))*0xbf58476d1ce4e5b9)&MASK
    x=((x^(x>>27))*0x94d049bb133111eb)&MASK
    return x^(x>>31)


def uniform(stream,index,attempt=0):return ((draw(stream,index,attempt)>>11)+.5)/2**53


def normal(stream,index,mean,std,low,high):
    for attempt in range(1000):
        z=math.sqrt(-2*math.log(uniform(stream,index,2*attempt)))*math.cos(math.tau*uniform(stream+1,index,2*attempt+1))
        value=mean+std*z
        if low<=value<=high:return value
    raise AssertionError('unexpected rejection exhaustion')


def endpoints(kind):
    pairs=[]
    if kind=='total':
        pairs=[((draw(0,e)*8)>>64,(draw(1,e)*5)>>64) for e in range(257)]
    else:
        for target in range(5):
            selected=set()
            for candidate in range(5,8):
                value=(draw(0,target*8+candidate)*(candidate+1))>>64
                source=candidate if value in selected else value
                selected.add(source);pairs.append((source,target))
    return np.asarray(sorted(pairs,key=lambda pair:pair[0]),dtype=np.uint32)


def make_model(device,tmp_path,kind,distribution):
    b.set_device('rust_standalone',engine='reference',directory=tmp_path/'ref',runner=ROOT/'target/release/b2-runner')
    source=b.SpikeGeneratorGroup(8,np.arange(8),np.zeros(8)*DT,period=DT,dt=DT)
    target=b.NeuronGroup(7,'v:1',clock=source.clock)
    syn=b.Synapses(source,target[1:6],'w:1 (constant)',on_pre='v_post+=w',clock=source.clock)
    weight=rust.Uniform(.0625,.25) if distribution=='uniform' else rust.ClippedNormal(.125,.03,.0625,.25)
    delay=rust.Uniform(0*DT,3*DT) if distribution=='uniform' else rust.ClippedNormal(1.5*DT,.5*DT,0*DT,3*DT)
    fn=rust.connect_fixed_total if kind=='total' else rust.connect_fixed_indegree
    fn(syn,257 if kind=='total' else 3,seed=SEED,initializers={'w':weight},delay_initializer=delay)
    monitor=b.StateMonitor(target,'v',record=True)
    return lower_network(b.Network(source,target,syn,monitor),8*DT)


@pytest.mark.parametrize('backend',BACKENDS)
@pytest.mark.parametrize('kind,distribution',[('total','uniform'),('total','normal'),('indegree','uniform'),('indegree','normal')])
def test_procedural_edges_initializers_and_gpu_delivery(device,tmp_path,backend,kind,distribution):
    model=make_model(device,tmp_path,kind,distribution)
    original=canonical_bytes(model)
    prepared=prepare_model(model)
    assert canonical_bytes(model)==original
    inst=prepared['instance']['synapses'][0];expected_edges=endpoints(kind)
    np.testing.assert_array_equal(inst['source'],expected_edges[:,0]);np.testing.assert_array_equal(inst['target'],expected_edges[:,1])
    count=len(expected_edges)
    weights=np.array([.0625+.1875*uniform(2,e) if distribution=='uniform'
                      else normal(2,e,.125,.03,.0625,.25) for e in range(count)])
    delays=np.array([3*uniform(4,e) if distribution=='uniform'
                     else normal(4,e,1.5,.5,0,3) for e in range(count)])
    np.testing.assert_allclose(inst['parameters']['w'],weights,rtol=2e-15,atol=0)
    ticks=np.floor(delays+.5).astype(int)
    np.testing.assert_array_equal(inst['pathways'][0]['delay_ticks'],ticks)
    expected=np.zeros(7,np.float32);trace=[];events=0
    order=sorted(range(count),key=lambda e:(-int(ticks[e]),int(expected_edges[e,0]),e))
    for tick in range(8):
        trace.append(expected.copy())
        for edge in order:
            if tick>=ticks[edge]:
                target=int(expected_edges[edge,1])+1
                expected[target]=np.float32(expected[target]+np.float32(weights[edge]));events+=1
    for route in ('scan','sparse'):
        path=tmp_path/(backend+route)
        actual=execute(model,path,backend,route)
        target=actual['populations'][model['definition']['synapses'][0]['target_population']]
        np.testing.assert_array_equal(target['states']['v'],expected)
        np.testing.assert_array_equal(target['trace']['v'],np.asarray(trace))
        assert actual['synapses'][0]['events']==events
        plan=(build_metal_plan if backend=='metal' else build_cuda_plan)(model,numeric_mode='float32',event_delivery=route)
        assert plan.initializations[0].edge_count==count and 'rust-host-f64-v0' in explain_plan(plan)
        with pytest.raises(PlanValidationError):verify_execution_plan(replace(plan,initializations=()),model)
        if backend!='cpu-f32':
            metadata=load_results(model,path/'transport')['metadata'];bind_execution_plan(plan,metadata)
            assert metadata['initialization_seconds']>0
            with pytest.raises(PlanValidationError):bind_execution_plan(plan,{**metadata,'initializations':[]})


@pytest.mark.parametrize('backend',BACKENDS)
def test_binary_csr_host_loading_and_native_gpu_execution(device,tmp_path,backend):
    from brian2_rust.binary_topology import HEADER, MAGIC
    b.set_device('rust_standalone',engine='reference',directory=tmp_path/'ref',runner=ROOT/'target/release/b2-runner')
    path=tmp_path/'connections.b2csr'
    with path.open('wb') as output:
        output.write(HEADER.pack(MAGIC,4,4,5,1))
        output.write(np.array([0,2,3,4,5],dtype='<u8').tobytes())
        output.write(np.array([0,2,3,1,0],dtype='<u4').tobytes())
        output.write(np.array([1,3,2,4,5],dtype='<f8').tobytes())
    source=b.SpikeGeneratorGroup(4,np.arange(4),np.zeros(4)*DT,period=DT,dt=DT)
    target=b.NeuronGroup(4,'v:1',clock=source.clock)
    syn=b.Synapses(source,target,'w:1 (constant)',on_pre='v_post+=w',delay=DT,clock=source.clock)
    rust.connect_binary_csr(syn,path,parameters={'w':0})
    model=lower_network(b.Network(source,target,syn),4*DT)
    actual=execute(model,tmp_path/backend,backend,'sparse')
    np.testing.assert_array_equal(actual['populations'][model['definition']['synapses'][0]['target_population']]['states']['v'],[18,12,9,6])
    model_path=tmp_path/'model.json';model_path.write_text(json.dumps(model))
    subprocess.run([str(ROOT/'target/release/b2-runner'),str(model_path),str(tmp_path/'oracle')],check=True)
    equivalent(actual,load_results(model,tmp_path/'oracle'),exact=True)
    data=bytearray(path.read_bytes());data[-1]^=1;path.write_bytes(data)
    with pytest.raises(PlanValidationError):build_cuda_plan(model,numeric_mode='float32')


def test_initialization_limits_and_failed_sampler_do_not_publish(device,tmp_path):
    model=make_model(device,tmp_path,'total','uniform')
    with pytest.raises(PlanValidationError,match='byte budget'):prepare_model(model,max_bytes=1)
    invalid=copy.deepcopy(model)
    invalid['instance']['synapses'][0]['topology']['initializers']['w']={
        'kind':'clipped_normal','mean':bits(0),'std':bits(0),'minimum':bits(1),'maximum':bits(2),'stream':2}
    attach_protocol(invalid)
    with pytest.raises(PlanValidationError,match='rejection limit'):build_cuda_plan(invalid,numeric_mode='float32')


@pytest.mark.parametrize('backend',BACKENDS)
def test_procedural_delay_rounding_preserves_reference_upper_boundary(device,tmp_path,backend):
    model=make_model(device,tmp_path,'indegree','uniform')
    delay=model['instance']['synapses'][0]['pathways'][0]['delay_initializer']
    delay['minimum']=delay['maximum']=bits(1000000.75*float(DT))
    attach_protocol(model)
    prepared=prepare_model(model)
    np.testing.assert_array_equal(prepared['instance']['synapses'][0]['pathways'][0]['delay_ticks'],1000001)
    actual=execute(model,tmp_path/backend,backend)
    path=tmp_path/'model.json';path.write_text(json.dumps(model))
    subprocess.run([str(ROOT/'target/release/b2-runner'),str(path),str(tmp_path/'oracle')],check=True)
    equivalent(actual,load_results(model,tmp_path/'oracle'),exact=True)
    assert actual['synapses'][0]['events']==0


@pytest.mark.parametrize('backend',BACKENDS[1:])
@pytest.mark.parametrize('queued',[False,True])
def test_device_procedural_single_activation_contract(device,tmp_path,backend,queued):
    results=[]
    for engine in ('reference',backend):
        device.reinit()
        opts={'numeric_mode':'float32'} if engine!='reference' else {}
        b.set_device('rust_standalone',engine=engine,directory=tmp_path/engine,runner=ROOT/'target/release/b2-runner',
                     build_on_run=not queued,**opts)
        source=b.SpikeGeneratorGroup(5,np.arange(5),np.zeros(5)*DT,period=DT,dt=DT)
        target=b.NeuronGroup(7,'v:1',clock=source.clock)
        syn=b.Synapses(source,target,on_pre='v_post+=1',clock=source.clock)
        rust.connect_fixed_indegree(syn,3,seed=SEED,delay_initializer=rust.Uniform(0*DT,3*DT))
        mon=b.StateMonitor(target,'v',record=True);net=b.Network(source,target,syn,mon)
        net.run(8*DT)
        if queued:device.build()
        results.append((np.asarray(target.v[:]).copy(),np.asarray(mon.v).copy()))
        error,pattern=(RuntimeError,'already been built') if queued else (NotImplementedError,'one run|one .*activation')
        with pytest.raises(error,match=pattern):net.run(DT)
    for actual,expected in zip(*results,strict=True):np.testing.assert_array_equal(actual,expected)
