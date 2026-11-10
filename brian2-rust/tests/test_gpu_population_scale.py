"""Bounded population scaling retains topology identity and comparison gates."""
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


@pytest.mark.parametrize('n',[4096,16384])
def test_scaled_ring_has_exact_creation_order_and_delay_groups(n):
    from gpu_stdp_compare import topology,configuration
    source,target,delay=topology(n,8,delay_span=8)
    assert len(source)==len(target)==len(delay)==n*8
    np.testing.assert_array_equal(source,np.arange(n*8)//8)
    np.testing.assert_array_equal(target,(np.arange(n*8)//8+np.arange(n*8)%8+1)%n)
    np.testing.assert_array_equal(delay,np.arange(n*8)%8)
    assert not np.any(source==target)
    config=configuration(n,8,256,drive=1/16,delay_span=8,post_delay=3)
    assert config['acceptance']['spikes']=='exact ticks and neuron indices'
    assert config['acceptance']['lastupdate']=='exact'


@pytest.mark.parametrize('n,kind',[(16385,'ring'),(4097,'random-fixed-outdegree')])
def test_large_invalid_topology_rejects_before_allocation(n,kind,monkeypatch):
    from gpu_stdp_compare import topology
    def forbidden(*a,**k):raise AssertionError('allocation before bound check')
    monkeypatch.setattr(np,'repeat',forbidden)
    with pytest.raises(ValueError):topology(n,8,topology_kind=kind)


def test_worker_rejects_quadratic_large_random_case_before_output(tmp_path):
    script=Path(__file__).resolve().parents[1]/'examples/gpu_stdp_precompiled.py'
    output=tmp_path/'no-output'
    p=subprocess.run([sys.executable,str(script),'--backend','cuda','--neurons','16384',
        '--topology-kind','random-fixed-outdegree','--output',str(output)],capture_output=True,text=True)
    assert p.returncode==2 and 'Rank-based random topology' in p.stderr
    assert not output.exists()


def test_scale_scenario_preserves_failed_backends_and_declared_precision(monkeypatch):
    import modal_f32_stdp_compare as runner
    import modal_stdp_precompiled as precompiled
    calls=[]
    def compare(n,steps,degree,repeats,**kwargs):
        calls.append((n,steps,degree,repeats,kwargs))
        return dict(status='completed-with-gate-failures',nvidia_smi='same allocation',
            payloads={'failed-bootstrap.npz':b'retained'},excluded_by_gate={'genn':{'passed':False}})
    monkeypatch.setattr(precompiled,'compare',compare)
    result=runner.verify(scenario='population-scale')
    assert [c[:4] for c in calls]==[(4096,256,8,5),(16384,256,8,5)]
    assert all(c[4]['continue_on_gate_failure'] and c[4]['numeric_contract']=='explicit-f32-v1' for c in calls)
    assert result['status']=='completed-with-gate-failures' and not result['passed']
    assert len(result['payloads'])==2
    assert all(c['excluded_by_gate']['genn']['passed'] is False for c in result['cases'].values())
