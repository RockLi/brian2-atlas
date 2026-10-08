"""Parameter sweeps preserve exact delay semantics and predeclared gates."""
from pathlib import Path
import numpy as np
import pytest

@pytest.fixture
def stdp(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'examples'))
    import gpu_stdp_compare
    return gpu_stdp_compare

@pytest.mark.parametrize('options',[
    dict(drive=.0625,delay_span=8,post_delay=3),
    dict(drive=.25,delay_span=1,post_delay=0),
    dict(drive=.015625,delay_span=16,post_delay=16),
    dict(drive=.125,delay_span=16,post_delay=0),
])
@pytest.mark.parametrize('backend',['rust','cpu-f32'])
@pytest.mark.parametrize('split',[False,True])
def test_variable_activity_and_delays_against_ordered_oracle(stdp,tmp_path,options,backend,split):
    output=tmp_path/'run';output.mkdir()
    actual,_=stdp.brian_run(backend,17,7,48,output,split,**options)
    expected=stdp.oracle(17,7,48,**options)
    gate=stdp.checks(actual,expected);assert gate['passed'],gate
    if options['drive']==.015625:
        assert actual['ticks'].size==0
        np.testing.assert_array_equal(actual['w'],np.full(119,.25))
    else:
        assert actual['ticks'].size>0 and np.any(actual['w']!=.25)
    import json
    (tmp_path/'gate.json').write_text(json.dumps(dict(options=options,backend=backend,split=split,gate=gate),indent=2)+'\n')
    np.savez_compressed(tmp_path/'full-results.npz',**{f'actual/{k}':v for k,v in actual.items()},**{f'oracle/{k}':v for k,v in expected.items()})

@pytest.mark.parametrize('options',[
    dict(drive=float('nan')),dict(drive=float('inf')),dict(drive=True),dict(drive=-.1),dict(drive=.6),
    dict(delay_span=0),dict(delay_span=17),dict(delay_span=True),dict(post_delay=-1),dict(post_delay=17),dict(post_delay=2.5),
])
def test_invalid_sweep_parameters_rejected(stdp,options):
    with pytest.raises(ValueError):stdp.configuration(17,7,48,**options)


def test_configuration_binds_drive_delay_and_topology(stdp):
    base=stdp.configuration(17,7,48)
    assert base['case']=='recurrent-delayed-stdp-v0'
    low=stdp.configuration(17,7,48,drive=.0625,delay_span=8,post_delay=3)
    assert low['case']=='recurrent-delayed-stdp-v1'
    assert low['topology_sha256']!=base['topology_sha256']
    assert low['workload']==dict(drive=.0625,delay_span=8,post_delay=3)
    assert low['acceptance']==base['acceptance']
    _,_,delay=stdp.topology(17,7,delay_span=8)
    assert sorted(set(delay))==list(range(8))


def test_activity_counts_only_deliveries_inside_run(stdp):
    opts=dict(drive=.0625,delay_span=8,post_delay=3)
    result=stdp.oracle(17,7,48,**opts);counts=stdp.activity(17,7,48,result,**opts)
    source,target,delay=stdp.topology(17,7,delay_span=8)
    pre=post=0
    for tick,neuron in zip(result['ticks'],result['indices']):
        pre+=int(np.count_nonzero((source==neuron)&(tick+delay<48)))
        post+=int(np.count_nonzero(target==neuron)) if tick+3<48 else 0
    assert counts['delivered_pre_events']==pre and counts['delivered_post_events']==post
    assert counts['spikes']==len(result['ticks'])
