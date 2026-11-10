"""Non-ring graphs retain declared identities, edge mapping and numerical gates."""
from pathlib import Path
import json
import numpy as np
import pytest
from test_gpu_stdp_parameters import stdp


@pytest.mark.parametrize('seed',[0,42,2**32-1])
def test_version_independent_target_ranks_and_edge_identity(stdp,seed):
    n,k=17,7
    source,target,delay=stdp.topology(n,k,delay_span=8,topology_kind='random-fixed-outdegree',topology_seed=seed)
    mask=2**64-1
    def score(key):
        z=(key+0x9e3779b97f4a7c15)&mask
        z=((z^(z>>30))*0xbf58476d1ce4e5b9)&mask
        z=((z^(z>>27))*0x94d049bb133111eb)&mask
        return z^(z>>31)
    expected=[t for s in range(n) for t in sorted(sorted((t for t in range(n) if t!=s),key=lambda t:score(s*n+t+seed))[:k])]
    np.testing.assert_array_equal(target,expected)
    np.testing.assert_array_equal(source,np.repeat(np.arange(n),k))
    np.testing.assert_array_equal(delay,np.arange(n*k)%8)
    assert not np.any(source==target) and len(set(zip(source,target)))==n*k
    # GeNN CSR and homogeneous-delay partitions must restore original edge ids.
    restored=np.full(n*k,-1)
    for d in reversed(range(8)):
        edges=np.flatnonzero(delay==d);order=np.lexsort((target[edges],source[edges]))
        restored[edges[order]]=edges[order]
    np.testing.assert_array_equal(restored,np.arange(n*k))


@pytest.mark.parametrize('options',[
    dict(topology_kind='bad'),dict(topology_seed=True),dict(topology_seed=-1),
    dict(topology_seed=2**32),dict(topology_seed=1.5),dict(topology_seed=42),
])
def test_invalid_topology_options_rejected(stdp,options):
    with pytest.raises(ValueError):stdp.configuration(17,7,48,**options)


def test_identity_and_unchanged_precision_gate(stdp):
    ring=stdp.configuration(1024,8,256)
    a=stdp.configuration(1024,8,256,topology_kind='random-fixed-outdegree',topology_seed=42)
    b=stdp.configuration(1024,8,256,topology_kind='random-fixed-outdegree',topology_seed=43)
    assert len({r['topology_sha256'] for r in (ring,a,b)})==3
    assert a['acceptance']==b['acceptance']==ring['acceptance']
    assert a['workload']['topology_seed']==42
    assert a['topology']['indegree_min']<8<a['topology']['indegree_max']
    assert a['topology']['indegree_mean']==8 and a['topology']['indegree_std']>0


@pytest.mark.parametrize('seed',[0,42])
@pytest.mark.parametrize('backend',['rust','cpu-f32'])
@pytest.mark.parametrize('split',[False,True])
def test_random_full_results_and_delay_group_mapping(stdp,tmp_path,seed,backend,split):
    opts=dict(drive=.0625,delay_span=8,post_delay=3,topology_kind='random-fixed-outdegree',topology_seed=seed)
    folder=tmp_path/'run';folder.mkdir()
    actual,_=stdp.brian_run(backend,17,7,48,folder,split,**opts)
    expected=stdp.oracle(17,7,48,**opts);gate=stdp.checks(actual,expected)
    assert gate['passed'],gate
    assert len(actual['ticks']) and np.any(actual['w']!=.25)
    (tmp_path/'topology-gate.json').write_text(json.dumps(dict(options=opts,backend=backend,split=split,gate=gate),indent=2)+'\n')
    np.savez_compressed(tmp_path/'topology-results.npz',**{'actual/'+k:v for k,v in actual.items()},**{'reference/'+k:v for k,v in expected.items()})
