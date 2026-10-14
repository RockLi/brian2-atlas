"""Numerical Poisson core: real Cython, high precision probabilities, Metal.

These are sampler/score tests; SSA/frontend and loss-score integration remain
separate acceptance requirements, and a likelihood score is not a sample VJP.
"""
import ctypes as c
import os
from pathlib import Path
import subprocess
from unittest.mock import patch
import mpmath as mp
import numpy as np
import pytest
import brian2 as b
from scipy.stats import poisson as poisson_distribution
from test_training_linked import cython_cache

ROOT=Path(__file__).resolve().parents[1]
MASK=(1<<64)-1

def mix(x):
    x=((x^(x>>30))*0xbf58476d1ce4e5b9)&MASK
    x=((x^(x>>27))*0x94d049bb133111eb)&MASK
    return x^(x>>31)

def key(entity,*,sequence=0,tick=0,stream=0,event=0,seed=7123):
    value=mix(seed^0x4232504f49533031)
    for field in [sequence,0,1,entity,event,tick,stream]:value=mix(value^mix((field+0x9e3779b97f4a7c15)&MASK))
    return value

def uniform(k,draw):
    bits=mix(k^mix((draw+0x9e3779b97f4a7c15)&MASK))
    return (((bits>>12)<<1)|1)/2**53

def reference_logp(rate,count):
    if not rate:return 0. if not count else -np.inf
    with mp.workdps(70):
        x=mp.mpf(float(rate));k=mp.mpf(int(count));return float(-x+k*mp.log(x)-mp.loggamma(k+1))

@pytest.fixture(scope='module')
def cpu(tmp_path_factory):
    path=tmp_path_factory.mktemp('poisson-rust')/'sampler.dylib'
    subprocess.run(['rustc','--edition=2021','-O','--crate-type','cdylib',str(ROOT/'tests/poisson_core_probe.rs'),'-o',str(path)],check=True,capture_output=True,text=True)
    lib=c.CDLL(str(path));lib.poisson_batch.argtypes=[c.c_size_t,*([c.c_void_p]*7)];lib.poisson_batch.restype=None
    lib.poisson_logp.argtypes=[c.c_double,c.c_double];lib.poisson_logp.restype=c.c_double
    lib.poisson_uniform.argtypes=[c.c_uint64,c.c_uint64];lib.poisson_uniform.restype=c.c_double
    def sample(rates,keys):
        rates=np.asarray(rates,dtype=np.float64);keys=np.asarray(keys,dtype=np.uint64);assert rates.shape==keys.shape and rates.ndim==1
        output=[np.zeros(len(rates),dtype=t) for t in (np.int32,np.uint32,np.uint32,np.float64,np.float64)]
        lib.poisson_batch(len(rates),rates.ctypes.data,keys.ctypes.data,*[a.ctypes.data for a in output]);return output
    sample.lib=lib;return sample

@pytest.fixture(scope='module',params=['cpu','metal'])
def sampler(request,cpu,tmp_path_factory):
    if request.param=='cpu':return cpu,False
    if os.environ.get('B2_TEST_GPU')!='1':pytest.skip('actual Metal hardware required')
    path=tmp_path_factory.mktemp('poisson-metal')/'sampler.dylib'
    subprocess.run(['clang','-O2','-ffp-contract=off','-fobjc-arc','-dynamiclib','-framework','Foundation','-framework','Metal',str(ROOT/'tests/poisson_metal_probe.m'),'-o',str(path)],check=True,capture_output=True,text=True)
    lib=c.CDLL(str(path));lib.poisson_metal_batch.argtypes=[c.c_char_p,c.c_uint64,*([c.c_void_p]*7),c.c_void_p,c.c_size_t];lib.poisson_metal_batch.restype=c.c_int
    shader='#include <metal_stdlib>\nusing namespace metal;\n'+(ROOT/'python/brian2_rust/training_poisson.metal').read_text()+'''
kernel void poisson_probe(device const float* rates [[buffer(0)]],device const ulong* keys [[buffer(1)]],
 device int* counts [[buffer(2)]],device uint* draws [[buffer(3)]],device uint* errors [[buffer(4)]],
 device float* scores [[buffer(5)]],device float* logps [[buffer(6)]],device const ulong* size [[buffer(7)]],
 uint i [[thread_position_in_grid]]) {
 if(i>=size[0])return;uint used=0,error=0;int value=atlas_dynamic_poisson(rates[i],keys[i],used,error);
 counts[i]=value;draws[i]=used;errors[i]=error;
 scores[i]=error?NAN:atlas_dynamic_poisson_score(rates[i],value);
 logps[i]=error?NAN:atlas_dynamic_poisson_log_probability(value,rates[i]);
}
'''
    def sample(rates,keys):
        rates=np.asarray(rates,dtype=np.float32);keys=np.asarray(keys,dtype=np.uint64);assert rates.shape==keys.shape and rates.ndim==1
        output=[np.zeros(len(rates),dtype=t) for t in (np.int32,np.uint32,np.uint32,np.float32,np.float32)]
        message=c.create_string_buffer(4096)
        status=lib.poisson_metal_batch(shader.encode(),len(rates),rates.ctypes.data,keys.ctypes.data,*[a.ctypes.data for a in output],message,len(message))
        assert status==0,message.value.decode();return output
    return sample,True

def test_counter_uniform_reference_and_open_endpoints(cpu):
    for seed in [0,1,MASK]:
        for entity in [0,1,17,MASK]:
            for draw in [0,1,99999,MASK]:
                k=key(entity,seed=seed);actual=cpu.lib.poisson_uniform(k,draw)
                assert actual==uniform(k,draw) and 0<actual<1

@pytest.mark.parametrize('rate',[10.,16.,100.,1e3,1e6,1e9,float(2**31-1)])
def test_stable_log_probability_against_high_precision(cpu,rate):
    for count in [0,1,15,16,*[max(0,round(rate+d*np.sqrt(rate))) for d in [-10,-3,-.1,0,.1,3,10]]]:
        expected=reference_logp(rate,count)
        assert cpu.lib.poisson_logp(count,rate)==pytest.approx(expected,rel=3e-13,abs=3e-12)

@pytest.mark.parametrize('rate',[0.,.05,1.,9.99,10.,10.01,20.,1e3,1e6,1e9])
def test_distribution_score_and_exact_large_integer_payload(sampler,rate):
    sample,gpu=sampler;rate=float(np.float32(rate)) if gpu else rate;n=40000
    keys=[key(i) for i in range(n)];counts,draws,errors,scores,logps=sample([rate]*n,keys)
    assert not errors.any();assert np.all(counts>=0);assert np.all(draws<=100000)
    if not rate:
        assert not counts.any() and not draws.any();assert np.isnan(scores).all();return
    delta=counts.astype(float)-rate;variance=rate+2*rate*rate
    assert abs(delta.mean())<7*np.sqrt(rate/n)
    assert abs((delta**2).mean()-rate)<7*np.sqrt(variance/n)
    for quantile in [.001,.01,.1,.5,.9,.99,.999]:
        threshold=poisson_distribution.ppf(quantile,rate)
        probability=poisson_distribution.cdf(threshold,rate)
        assert abs(np.mean(counts<=threshold)-probability)<7*np.sqrt(probability*(1-probability)/n)+1/n
    np.testing.assert_allclose(scores,delta/rate,rtol=5e-7 if gpu else 2e-15,atol=1e-37)
    # score * (K - baseline) estimates d E[K]/d rate = 1.
    gradient=(delta*scores.astype(float)).mean()
    assert abs(gradient-1)<7*np.sqrt((2+1/rate)/n)
    # The same score with the K² loss estimates 2*rate+1.
    centered_loss=2*rate*delta+delta**2-rate
    second=(centered_loss*scores.astype(float)).mean()
    assert abs(second-(2*rate+1))<7*np.sqrt((8*rate**2+46*rate+26+1/rate)/n)
    if rate>2**24:
        assert abs((counts%2).mean()-.5)<.02
        assert np.mean(counts%64!=0)>.95
    for i in [0,1,17,999,n-1]:
        assert logps[i]==pytest.approx(reference_logp(rate,counts[i]),rel=2e-6 if gpu else 3e-13,abs=8e-6 if gpu else 3e-12)

def test_replay_and_batch_partition(sampler):
    sample,gpu=sampler;rates=np.tile([.1,1.,10.,1000.,1e9],101);keys=np.array([key(i,sequence=17,tick=31) for i in range(len(rates))],np.uint64)
    whole=sample(rates,keys);order=np.random.default_rng(17).permutation(len(rates));restored=sample(rates[order],keys[order])
    for a,c_ in zip(whole,restored):np.testing.assert_array_equal(a[order],c_)
    parts=[sample(rates[:97],keys[:97]),sample(rates[97:],keys[97:])]
    for j,a in enumerate(whole):np.testing.assert_array_equal(a,np.concatenate([p[j] for p in parts]))
    changed=sample(rates,[key(i,sequence=18,tick=31) for i in range(len(rates))])[0]
    assert np.mean(changed!=whole[0])>.5

def test_tiny_positive_rate_keeps_rare_draw_on_actual_device(sampler):
    sample,_=sampler
    # SplitMix(0)=0. This key makes the first odd uniform numerator exactly 1;
    # it exercises the rare event directly, without searching or mocking RNG.
    k=mix(0x9e3779b97f4a7c15)
    assert uniform(k,0)==2.**-53
    count,draws,error,_,_=sample([0.,1e-12,2.**-52,2.**-54],[k]*4)
    np.testing.assert_array_equal(error,0)
    np.testing.assert_array_equal(count,[0,1,1,0])
    np.testing.assert_array_equal(draws,[0,2,2,1])

def test_subnormal_positive_rate_is_distinct_from_zero(sampler):
    sample,_=sampler
    tiny=np.array([1,0x007fffff,0x00800000],dtype=np.uint32).view(np.float32)
    count,draws,error,score,_=sample(tiny,[key(i) for i in range(3)])
    np.testing.assert_array_equal(error,0)
    np.testing.assert_array_equal(count,0)
    np.testing.assert_array_equal(draws,1)
    np.testing.assert_array_equal(score,-1)
    _,_,error,_,_=sample(-tiny,[key(i) for i in range(3)])
    np.testing.assert_array_equal(error,1)

def test_invalid_rate_and_accepted_overflow_are_errors(sampler):
    sample,gpu=sampler
    _,_,error,_,_=sample([-1.,np.nan,np.inf,2.**31],[key(i) for i in range(4)])
    np.testing.assert_array_equal(error,1)
    rate=float(2**31-128);count,_,error,_,_=sample([rate]*128,[key(i) for i in range(128)])
    assert set(error)=={0,4};assert np.all(count[error==0]>=0)

def test_actual_brian_cython_poisson_with_identical_uniforms(cpu):
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='cython'
    rates=np.tile([0.,1e-10,.1,2.,9.9,10.,10.1,1000.,1e9],8);keys=[key(i) for i in range(len(rates))]
    counts,used,errors,_,_=cpu(rates,keys);assert not errors.any()
    draws=[(1-uniform(k,j)) if rate<10 else uniform(k,j) for rate,k,n in zip(rates,keys,used) for j in range(int(n))]
    group=b.NeuronGroup(len(rates),'k:integer\nrate:1',name='poisson_reference');group.rate=rates
    group.run_regularly('k=poisson(rate)',dt=.1*b.ms,when='start');net=b.Network(group);net.run(0*b.ms)
    device=b.get_device();device.rand_buffer_index[:]=0;calls=[]
    def refill(n):
        assert n==20000 and not calls;calls.append(n);values=np.full(n,.5);values[:len(draws)]=draws;return values
    with patch('numpy.random.rand',refill):net.run(.1*b.ms)
    np.testing.assert_array_equal(group.k[:],counts)
    assert calls==[20000] and device.rand_buffer_index[0]==len(draws);device.rand_buffer_index[:]=0
