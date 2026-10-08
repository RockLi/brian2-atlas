"""Independent STDP oracle, delay-group mapping, and complete acceptance gates."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec=importlib.util.spec_from_file_location('stdp_comparison',Path(__file__).resolve().parents[1]/'examples/gpu_stdp_compare.py')
stdp=importlib.util.module_from_spec(spec);spec.loader.exec_module(stdp)


@pytest.mark.parametrize('backend',['rust','cpu-f32'])
@pytest.mark.parametrize('split',[False,True])
def test_delayed_stdp_full_outputs_match_independent_oracle(backend,split,tmp_path,monkeypatch):
    import sys
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'examples'))
    output=tmp_path/backend;output.mkdir()
    actual,_=stdp.brian_run(backend,17,7,48,output,split)
    expected=stdp.oracle(17,7,48)
    gate=stdp.checks(actual,expected)
    assert gate['passed'],gate
    assert len(actual['ticks'])>0
    assert np.any(actual['w']!=.25)
    if backend=='rust':
        for key in ('ticks','indices','lastupdate'):assert np.array_equal(actual[key],expected[key])


@pytest.mark.parametrize('field',['v','w','Apre','Apost','lastupdate','trace','ticks','indices'])
def test_gate_rejects_each_corrupted_field(field):
    expected=stdp.oracle(17,7,48)
    actual={k:v.copy() for k,v in expected.items()}
    actual[field].flat[-1]+=1
    assert not stdp.checks(actual,expected)['passed']
    assert not stdp.checks(actual,expected)['fields'][field]['passed']


def test_gate_requires_full_trace_and_all_edge_states():
    expected=stdp.oracle(17,7,48)
    for field in expected:
        actual=dict(expected);actual.pop(field)
        assert not stdp.checks(actual,expected)['passed']
    actual=dict(expected);actual['trace']=actual['trace'][:-1]
    assert not stdp.checks(actual,expected)['passed']


def test_reset_discards_same_tick_transmission():
    # Neuron 15 starts at .9375 and crosses threshold at tick zero. Its reset
    # must win over any incoming event, while the trace samples after drive.
    result=stdp.oracle(17,7,1)
    assert 15 in result['indices']
    assert result['v'][15]==0
    np.testing.assert_array_equal(result['trace'][0],np.array([.125,.125]))


def test_delay_group_mapping_is_bijective():
    source,target,delay=stdp.topology(17,7)
    groups=[np.flatnonzero(delay==d) for d in (3,2,1,0)]
    np.testing.assert_array_equal(np.sort(np.concatenate(groups)),np.arange(119))
    restored=np.empty(119,np.int64)
    for edges in groups:
        order=np.lexsort((target[edges],source[edges]))
        # Same edge-index restoration used after GeNN CSR readback.
        restored[edges[order]]=edges[order]
    np.testing.assert_array_equal(restored,np.arange(119))
