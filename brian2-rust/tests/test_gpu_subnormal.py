"""Record subnormal arithmetic/threshold boundaries without relaxing f32 gates."""
import hashlib,json
import brian2 as b
import numpy as np
import pytest
from brian2_rust.export import lower_network
from test_gpu_expression_contract import oracle
from test_gpu_spike_generator import BACKENDS,execute
from test_gpu_monitors import setup,DT
from test_metal_delays import device

VALUES=np.array([2.**-125,2.**-126,2.**-127,2.**-149,
                 -2.**-125,-2.**-126,-2.**-127,-2.**-149])
STEPS=3


def flush(values):
    result=np.asarray(values,np.float32).copy()
    sub=(np.abs(result)<np.finfo(np.float32).tiny)&(result!=0)
    result[sub]=np.copysign(np.float32(0),result[sub])
    return result


def reference(operation,profile):
    dtype=np.float64 if profile=='f64' else np.float32
    x=VALUES.astype(dtype);v=x.copy();trace=[];ticks=[];indices=[]
    factor=dtype(.5 if operation=='shrink' else 2**24)
    for tick in range(STEPS):
        trace.append(v.copy())
        if operation=='copy':v=x.copy()
        else:
            with np.errstate(under='ignore'):
                v=np.multiply(flush(v) if profile=='ftz-f32' else v,factor,dtype=dtype)
            if profile=='ftz-f32':v=flush(v)
        compared=flush(v) if profile=='ftz-f32' else v
        fired=np.flatnonzero(compared>0).astype(np.int64)
        ticks.extend([tick]*len(fired));indices.extend(fired.tolist())
    indices=np.asarray(indices,np.int64)
    return dict(v=v,trace=np.asarray(trace),ticks=np.asarray(ticks,np.int64),indices=indices,
                counts=np.bincount(indices,minlength=len(x)).astype(np.int64),last_spikes=fired)


def arrays(result):
    p=result['populations'][0]
    return dict(v=p['states']['v'],trace=p['trace']['v'],ticks=p['spike_ticks'],
                indices=p['indices'],counts=p['counts'],last_spikes=p['last_spikes'])


def exact(actual,expected):
    return all(np.array_equal(a,expected[k]) and (a.dtype.kind!='f' or
               np.array_equal(np.signbit(a),np.signbit(expected[k]))) for k,a in actual.items())


@pytest.mark.parametrize('backend',BACKENDS)
@pytest.mark.parametrize('domain',['population','dag'])
@pytest.mark.parametrize('operation',['copy','shrink','grow'])
def test_subnormal_storage_arithmetic_and_threshold_contract(device,tmp_path,backend,domain,operation):
    setup(tmp_path/'ref')
    p=b.NeuronGroup(len(VALUES),'v:1\nx:1 (constant)\nfactor:1 (constant, shared)\nsink:1',
                    threshold='v>0',reset='',dt=DT,name='population')
    p.v=VALUES;p.x=VALUES;p.factor=.5 if operation=='shrink' else 2**24
    p.run_regularly('v=x' if operation=='copy' else 'v=v*factor',when='groups')
    objects=[p,b.SpikeMonitor(p),b.StateMonitor(p,'v',record=True)]
    if domain=='dag':
        s=b.Synapses(p,p,'w:1',on_pre='sink_post+=w',clock=p.clock,name='projection')
        s.connect(i=np.array([],np.int32),j=np.array([],np.int32));objects.append(s)
    model=lower_network(b.Network(*objects),STEPS*DT)
    expected={profile:reference(operation,profile) for profile in ['f64','ieee-f32','ftz-f32']}
    rust=arrays(oracle(model,tmp_path/'oracle'));assert exact(rust,expected['f64'])
    result=execute(model,tmp_path/backend,backend,'sparse');actual=arrays(result)
    profile='ftz-f32' if backend=='metal' else 'ieee-f32'
    if backend=='metal':assert 'Apple' in result['device'], 'This boundary fixture targets Apple GPUs'
    assert exact(actual,expected[profile])
    matches={k:exact(actual,v) for k,v in expected.items()}
    float_close=all(np.allclose(actual[k],expected['ieee-f32'][k],rtol=2e-5,atol=2e-6) for k in ['v','trace'])
    spikes_equal=all(np.array_equal(actual[k],expected['ieee-f32'][k]) for k in ['ticks','indices','counts','last_spikes'])
    assert float_close
    assert spikes_equal==(backend!='metal')
    saved={'actual/'+k:v for k,v in actual.items()}
    saved.update({'rust/'+k:v for k,v in rust.items()})
    saved.update({profile+'/'+k:v for profile,a in expected.items() for k,v in a.items()})
    (tmp_path/'subnormal-model.json').write_text(json.dumps(model,sort_keys=True)+'\n')
    np.savez_compressed(tmp_path/'subnormal-results.npz',**saved)
    (tmp_path/'subnormal-checks.json').write_text(json.dumps(dict(backend=backend,device=result['device'],domain=domain,
        operation=operation,expected_profile=profile,matches=matches,float_close_to_ieee=float_close,
        spikes_equal_to_ieee=spikes_equal,portable_f32_timing_eligible=matches['ieee-f32'],
        arrays={k:dict(dtype=str(v.dtype),shape=list(v.shape),sha256=hashlib.sha256(v.tobytes()).hexdigest()) for k,v in saved.items()}),indent=2)+'\n')
