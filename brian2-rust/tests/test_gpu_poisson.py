"""Native Poisson expressions: counter identity, probability stability, lifecycle."""
import ctypes
from dataclasses import replace
import json
import math
import platform
import subprocess

import brian2 as b
import mpmath as mp
import numpy as np
import pytest

from brian2_rust.metal import _CPU_PRELUDE,_PRELUDE
from brian2_rust.cuda import build_cuda_plan
from brian2_rust.plan import PlanValidationError,verify_execution_plan
from brian2_rust.results import load_results
from brian2_rust.protocol import attach_protocol
from brian2_rust.spec import bits
from brian2_rust.metal_random import RNG_PROFILE
from test_gpu_spike_generator import BACKENDS,execute
from test_gpu_random import setup,lower,uniform,SEED
from test_gpu_monitors import event_equivalent,DT
from test_gpu_refractory import refresh_code,make_model
from test_metal_delays import device,ROOT


def poisson_nodes(tree):
    if isinstance(tree,dict):
        if tree.get('op')=='poisson':yield tree
        for value in tree.values():yield from poisson_nodes(value)
    elif isinstance(tree,list):
        for value in tree:yield from poisson_nodes(value)


def product_oracle(rate,stream,tick,index):
    # Independent host f64 calculation fed the documented U24 counter values.
    product=1.0;limit=math.exp(-float(rate))
    for draw in range(8192):
        product*=float(uniform(stream,tick,index,draw))
        if product<=limit:return draw
    raise AssertionError('oracle draw bound exhausted')


@pytest.mark.parametrize('backend',BACKENDS)
def test_poisson_small_counters_large_ticks_profile_and_transport(device,tmp_path,backend):
    setup(device,tmp_path)
    pop=b.NeuronGroup(257,'x:1\ny:1\nrate:1 (constant)',dt=DT)
    pop.rate=np.resize([0,.125,.5,1,2,4,8,9.5],257)
    pop.run_regularly('x=poisson(rate); y=poisson(rate)')
    model=lower(b.Network(pop),3)
    model['run']['start']=bits(2**25/1024);model['run']['clocks'][0]['start_tick']=2**25;attach_protocol(model)
    actual=execute(model,tmp_path/backend,backend)
    assert actual['rng_profile']==RNG_PROFILE
    plan=build_cuda_plan(model,numeric_mode='float32')
    assert plan.rng_profile==RNG_PROFILE
    with pytest.raises(PlanValidationError):verify_execution_plan(replace(plan,rng_profile=None),model)
    for name,node in zip(('x','y'),poisson_nodes(model['definition']),strict=True):
        expected=[product_oracle(rate,node['stream'],2**25+2,i) for i,rate in enumerate(pop.rate[:])]
        np.testing.assert_array_equal(actual['populations'][0]['states'][name],expected)
    if backend!='cpu-f32':
        loaded=load_results(model,tmp_path/backend/'transport')
        assert loaded['metadata']['rng_profile']==RNG_PROFILE
        event_equivalent(loaded,actual)


@pytest.mark.parametrize('backend',BACKENDS)
def test_poisson_distribution_across_parameter_regimes(device,tmp_path,backend):
    setup(device,tmp_path)
    rates=np.asarray([.01,.5,2,9.5,10,10.125,16,32,100,1e4,2**24,1e8,1e12],np.float32)
    count=8192
    pop=b.NeuronGroup(len(rates)*count,'x:1\nrate:1 (constant)',dt=DT)
    pop.rate=np.repeat(rates,count);pop.run_regularly('x=poisson(rate)')
    model=lower(b.Network(pop));samples=execute(model,tmp_path/backend,backend)['populations'][0]['states']['x'].astype(np.float64).reshape(len(rates),count)
    diagnostics=[]
    for rate,values in zip(rates,samples,strict=True):
        rate=float(rate)
        assert np.all(np.isfinite(values)&(values>=0)&(values==np.floor(values)))
        z=(values-rate)/math.sqrt(rate)
        assert abs(z.mean())<7/math.sqrt(count)
        # Poisson standardized fourth moment is 3+1/lambda.
        assert abs(z.var()-1)<7*math.sqrt((2+1/rate)/count)
        if rate<=32:
            mass=math.exp(-rate);cdf=mass
            for k in range(int(rate+6*math.sqrt(rate))+1):
                if k:mass*=rate/k;cdf+=mass
                assert abs(np.mean(values<=k)-cdf)<.025
        diagnostics.append(dict(rate=rate,mean=float(values.mean()),variance=float(values.var()),z_mean=float(z.mean()),z_variance=float(z.var())))
    (tmp_path/(backend+'-distribution.json')).write_text(json.dumps(diagnostics,indent=2)+'\n')


@pytest.fixture
def probability_kernel(tmp_path):
    source=_CPU_PRELUDE+_PRELUDE.replace('#include <metal_stdlib>','').replace('using namespace metal;','').replace('thread ','').replace('device ','')
    source+='''\nextern "C" float log_mass(long k,float rate) {
        return b2_poisson_log_mass(k,rate,long(rate),rate-float(long(rate)));
    }\n'''
    path=tmp_path/'probability.cpp';path.write_text(source)
    library=tmp_path/('probability.dylib' if platform.system()=='Darwin' else 'probability.so')
    subprocess.run(['clang++','-std=c++17','-O2','-ffp-contract=off','-fno-fast-math',
                    '-dynamiclib' if platform.system()=='Darwin' else '-shared','-fPIC',str(path),'-o',str(library)],check=True,capture_output=True,text=True)
    native=ctypes.CDLL(str(library));native.log_mass.argtypes=[ctypes.c_int64,ctypes.c_float];native.log_mass.restype=ctypes.c_float
    return native.log_mass


def test_log_probability_against_70_digit_oracle(probability_kernel,tmp_path):
    rows=[]
    with mp.workdps(70):
        for rate in map(lambda v:float(np.float32(v)),[10,15.5,16,100,1e4,2**24,1e8,1e12]):
            candidates={0,1,2,7,15,16,int(.8749*rate),int(.8751*rate),int(1.1249*rate),int(1.1251*rate)}
            candidates.update(max(0,int(rate+z*math.sqrt(rate))) for z in (-12,-6,-2,-1,0,1,2,6,12))
            for k in sorted(candidates):
                expected=float(-mp.mpf(rate)+k*mp.log(rate)-mp.loggamma(k+1))
                actual=float(probability_kernel(k,rate))
                assert abs(actual-expected)<=1e-4+2e-5*abs(expected),(rate,k,actual,expected)
                if abs(k-rate)<12*math.sqrt(rate) and rate>=1e8:
                    assert abs(actual-expected)<2e-5,(rate,k,actual,expected)
                rows.append(dict(rate=rate,k=k,expected=expected,actual=actual,absolute_error=abs(actual-expected)))
    (tmp_path/'log-mass-diagnostics.json').write_text(json.dumps(rows,indent=2)+'\n')


def synaptic_network():
    source=b.NeuronGroup(4,'v:1',threshold='True',reset='',dt=DT,name='input')
    target=b.NeuronGroup(5,'v:1',threshold='v>=4',reset='v=0',dt=2*DT,name='target')
    syn=b.Synapses(source[1:],target[1:],'w:1\np:1 (constant)',on_pre='w=poisson(p); v_post+=w',
                   on_post='w+=poisson(p)',clock=source.clock,name='projection')
    syn.connect(i=[2,0,1,0,2],j=[1,1,2,1,0]);syn.p=[0,.5,2,16,32]
    syn.pre.delay=[0,1,2,0,3]*DT;syn.post.delay=np.array([0,1,2,1,3])*2*DT
    monitor=b.EventMonitor(target,'spike',variables='v',when='end')
    return b.Network(source,target,syn,monitor),target,syn,monitor


@pytest.mark.parametrize('backend',BACKENDS)
@pytest.mark.parametrize('route',['scan','sparse'])
def test_poisson_delayed_synaptic_edge_counter_and_multiclock(device,tmp_path,backend,route):
    setup(device,tmp_path);net,target,syn,monitor=synaptic_network();model=lower(net,12)
    actual=execute(model,tmp_path/backend,backend,route)
    control=execute(model,tmp_path/'control','cpu-f32',route);event_equivalent(actual,control)
    assert build_cuda_plan(model,numeric_mode='float32',event_delivery=route).rng_profile==RNG_PROFILE
    device.reinit();setup(device,tmp_path/'edge')
    source=b.SpikeGeneratorGroup(3,[0,1,2],[0,0,0]*b.second,period=DT,dt=DT)
    target=b.NeuronGroup(4,'v:1',clock=source.clock)
    syn=b.Synapses(source,target,'w:1',on_pre='w=poisson(2.0); v_post+=w',clock=source.clock)
    syn.connect(i=np.arange(37)%3,j=np.arange(37)%4);syn.delay=(np.arange(37)%3)*DT
    model=lower(b.Network(source,target,syn),8);stream=next(poisson_nodes(model['definition']))['stream']
    result=execute(model,tmp_path/'edge-result',backend,route)
    np.testing.assert_array_equal(result['synapses'][0]['states']['w'],[product_oracle(2,stream,7,e) for e in range(37)])


@pytest.mark.parametrize('backend',BACKENDS[1:])
@pytest.mark.parametrize('queued',[False,True])
def test_poisson_device_segments_queued_seed_and_pending_restore(device,tmp_path,backend,queued):
    outputs=[]
    for segmented in (False,True):
        device.reinit();b.set_device('rust_standalone',engine=backend,numeric_mode='float32',event_delivery='sparse',
            directory=tmp_path/str(segmented),runner=ROOT/'target/release/b2-runner',build_on_run=not queued);b.seed(SEED)
        net,target,syn,monitor=synaptic_network()
        if segmented:
            net.run(4*DT)
            if not queued:net.store('pending')
            net.run(8*DT)
            if not queued:
                before=np.asarray(syn.w[:]).copy();b.seed(999);net.restore('pending',restore_random_state=True);net.run(8*DT)
                np.testing.assert_array_equal(syn.w[:],before)
        else:net.run(12*DT)
        if queued:device.build()
        outputs.append([np.asarray(syn.w[:]).copy(),np.asarray(target.v[:]).copy(),np.asarray(monitor.i[:]).copy(),np.asarray(monitor.v[:]).copy()])
    for a,e in zip(*outputs,strict=True):np.testing.assert_array_equal(a,e)


@pytest.mark.parametrize('backend',BACKENDS)
@pytest.mark.parametrize('masked',[False,True])
def test_poisson_invalid_rate_faults_and_statement_masks(device,tmp_path,backend,masked):
    model=make_model(device,tmp_path,'masked');model['instance']['rng_seed']=SEED
    p=model['definition']['populations'][0]
    code=next(c for c in p['code_objects'] if c['kind']=='state_update')
    statement=next(s for s in code['vector'] if s['target']=='v')
    statement['value']=dict(op='poisson',stream=0,**{'lambda':dict(op='literal',bits=bits(-1))})
    model['instance']['populations'][0]['refractory']['initial_not_refractory']=[not masked]*p['count'];refresh_code(model,code)
    if masked:
        result=execute(model,tmp_path/backend,backend)
        np.testing.assert_array_equal(result['populations'][0]['states']['v'],.5)
    else:
        with pytest.raises(FloatingPointError,match='sampler'):execute(model,tmp_path/backend,backend)
        assert not (tmp_path/backend/'transport').exists()


@pytest.mark.parametrize('backend',BACKENDS)
def test_poisson_rate_upper_bound_rejected(device,tmp_path,backend):
    setup(device,tmp_path);pop=b.NeuronGroup(2,'x:1',dt=DT);pop.run_regularly('x=poisson(1e13)')
    model=lower(b.Network(pop))
    with pytest.raises(FloatingPointError,match='sampler'):execute(model,tmp_path/backend,backend)
