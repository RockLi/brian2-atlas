"""Stable large-binomial probabilities, sampling and native lifecycle."""
import ctypes
import json
import math
import platform
import subprocess

import brian2 as b
from brian2.input.binomial import BinomialFunction
import mpmath as mp
import numpy as np
import pytest

from brian2_rust.metal import _CPU_PRELUDE,_PRELUDE
from test_gpu_spike_generator import BACKENDS,execute
from test_gpu_random import setup,lower,SEED
from test_gpu_monitors import event_equivalent,DT
from test_metal_delays import device,ROOT


@pytest.fixture
def probability_kernel(tmp_path):
    source=_CPU_PRELUDE+_PRELUDE.replace('#include <metal_stdlib>','').replace('using namespace metal;','').replace('thread ','').replace('device ','')
    source=source.replace('return float(b2_mix64(value) >> 40) * 0x1p-24f;', 'return 0.0f;')
    source+='''\nextern "C" float binomial_mass(unsigned long n,long k,float p) {
        float mean=float(n)*p; long center=long(mean);
        float fraction=(mean-float(center))+b2_binomial_mean_tail(n,p,mean);
        return b2_binomial_log_mass(n,k,p,center,fraction,mean);
    }
    extern "C" int binomial_exhaustion() {
        bool fault=false;
        b2_binomial(0ul,0ul,0ul,0ul,10000ul,0.5f,false,&fault);
        return fault ? 1 : 0;
    }\n'''
    path=tmp_path/'probability.cpp';path.write_text(source)
    library=tmp_path/('probability.dylib' if platform.system()=='Darwin' else 'probability.so')
    subprocess.run(['clang++','-std=c++17','-O2','-ffp-contract=off','-fno-fast-math',
                    '-dynamiclib' if platform.system()=='Darwin' else '-shared','-fPIC',str(path),'-o',str(library)],check=True,capture_output=True,text=True)
    native=ctypes.CDLL(str(library));native.binomial_mass.argtypes=[ctypes.c_uint64,ctypes.c_int64,ctypes.c_float];native.binomial_mass.restype=ctypes.c_float
    native.binomial_exhaustion.restype=ctypes.c_int
    return native


def test_binomial_probability_against_70_digit_oracle(probability_kernel,tmp_path):
    rows=[]
    with mp.workdps(70):
        for n in (127,512,10000,2**24+1,2**31-1):
            for p in map(lambda x:float(np.float32(x)),(.000001,.01,.123456789,.4999,.5)):
                mean=n*p
                if n*math.log1p(-p)>-87:continue
                sigma=math.sqrt(mean*(1-p))
                candidates={0,1,2,15,16,n-1,n}
                candidates.update(max(0,min(n,int(mean+z*sigma))) for z in (-12,-6,-2,-1,0,1,2,6,12))
                candidates.update(max(0,min(n,int(mean*r))) for r in (.8749,.8751,1.1249,1.1251))
                for k in sorted(candidates):
                    expected=float(mp.loggamma(n+1)-mp.loggamma(k+1)-mp.loggamma(n-k+1)+k*mp.log(p)+(n-k)*mp.log1p(-mp.mpf(p)))
                    actual=float(probability_kernel.binomial_mass(n,k,p));error=abs(actual-expected)
                    assert error<1e-4+2e-5*abs(expected),(n,p,k,actual,expected)
                    if abs(k-mean)<12*sigma:assert error<2e-4,(n,p,k,actual,expected)
                    rows.append(dict(n=n,p=p,k=k,actual=actual,expected=expected,absolute_error=error))
    (tmp_path/'binomial-probability-diagnostics.json').write_text(json.dumps(rows,indent=2)+'\n')


def test_binomial_rejection_exhaustion_is_bounded(probability_kernel):
    # Every uniform is forced to zero, so every proposal has gap=0.
    assert probability_kernel.binomial_exhaustion()==1


@pytest.mark.parametrize('backend',BACKENDS)
@pytest.mark.parametrize('n',[126,127,512,10000,2**24+1,2**31-1])
def test_binomial_distribution_across_inversion_boundary_and_large_counts(device,tmp_path,backend,n):
    setup(device,tmp_path)
    rates=np.asarray([.01,.25,.5,.75,.99],np.float32);count=8192
    sample=BinomialFunction(n,.5,approximate=False,name='draw_binomial')
    pop=b.NeuronGroup(count*len(rates),'x:1\np:1 (constant)',dt=DT,namespace={'sample':sample})
    pop.p=np.repeat(rates,count);pop.run_regularly('x=sample()')
    model=lower(b.Network(pop))
    # The portable binomial AST supports a dynamic probability expression.
    from test_gpu_random import random_nodes
    from test_gpu_refractory import refresh_code
    node=next(random_nodes(model['definition']));node['p']={'op':'load','name':'p'}
    code=next(c for c in model['definition']['populations'][0]['code_objects'] if c['kind']=='run_regularly')
    code['effects']['reads'].append('p');refresh_code(model,code)
    samples=execute(model,tmp_path/backend,backend)['populations'][0]['states']['x'].astype(np.float64).reshape(len(rates),count)
    rows=[]
    for p,values in zip(rates,samples,strict=True):
        p=float(p);mean=n*p;variance=mean*(1-p);z=(values-mean)/math.sqrt(variance)
        assert np.all((values>=0)&(values<=float(np.float32(n)))&(values==np.floor(values)))
        assert abs(z.mean())<7/math.sqrt(count)
        assert abs(z.var()-1)<7*math.sqrt((2+(1-6*p*(1-p))/variance)/count)
        if n<=512:
            prob=min(p,1-p);observed=n-values if p>.5 else values
            mass=(1-prob)**n;cdf=mass
            for k in range(min(n,int(n*prob+8*math.sqrt(n*prob*(1-prob))))+1):
                if k:mass*=(n-k+1)*prob/(k*(1-prob));cdf+=mass
                assert abs(np.mean(observed<=k)-cdf)<.025
        rows.append(dict(n=n,p=p,mean=float(values.mean()),variance=float(values.var()),z_mean=float(z.mean()),z_variance=float(z.var())))
    (tmp_path/(backend+'-binomial-distribution.json')).write_text(json.dumps(rows,indent=2)+'\n')


def network():
    sample=BinomialFunction(512,.5,approximate=False,name='draw_large')
    source=b.SpikeGeneratorGroup(3,[0,1,2],[0,0,0]*DT,period=DT,dt=DT)
    target=b.NeuronGroup(4,'v:1',threshold='v>=512',reset='v=0',dt=2*DT)
    syn=b.Synapses(source,target,'w:1',on_pre='w=sample(); v_post+=w',on_post='w+=sample()',
                   namespace={'sample':sample},clock=source.clock)
    syn.connect(i=[2,0,1,0],j=[1,1,2,0]);syn.pre.delay=[0,1,3,2]*DT;syn.post.delay=np.asarray([0,1,2,3])*2*DT
    monitor=b.EventMonitor(target,'spike',variables='v',when='end')
    return b.Network(source,target,syn,monitor),target,syn,monitor


@pytest.mark.parametrize('backend',BACKENDS)
def test_large_binomial_delayed_edge_rng_replay_and_transport(device,tmp_path,backend):
    setup(device,tmp_path);net,*_=network();model=lower(net,12)
    control=execute(model,tmp_path/'control','cpu-f32','sparse')
    event_equivalent(execute(model,tmp_path/backend,backend,'sparse'),control)


@pytest.mark.parametrize('backend',BACKENDS[1:])
@pytest.mark.parametrize('queued',[False,True])
def test_large_binomial_segments_and_seeded_restore(device,tmp_path,backend,queued):
    outputs=[]
    for split in (False,True):
        device.reinit();b.set_device('rust_standalone',engine=backend,numeric_mode='float32',event_delivery='sparse',
            directory=tmp_path/str(split),runner=ROOT/'target/release/b2-runner',build_on_run=not queued);b.seed(SEED)
        net,target,syn,monitor=network()
        if split:
            net.run(6*DT)
            if not queued:net.store('saved')
            net.run(6*DT)
            if not queued:
                saved=np.asarray(syn.w[:]).copy();b.seed(999);net.restore('saved',restore_random_state=True);net.run(6*DT)
                np.testing.assert_array_equal(syn.w[:],saved)
        else:net.run(12*DT)
        if queued:device.build()
        outputs.append([np.asarray(target.v[:]).copy(),np.asarray(syn.w[:]).copy(),np.asarray(monitor.i[:]).copy(),np.asarray(monitor.v[:]).copy()])
    for a,e in zip(*outputs,strict=True):np.testing.assert_array_equal(a,e)
