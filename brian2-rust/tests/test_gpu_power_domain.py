"""Dynamic signed-base powers keep finite values, signed zero and error gates."""
import json
import math
from copy import deepcopy

import brian2 as b
import numpy as np
import pytest

from brian2_rust.export import lower_network
from test_gpu_expression_contract import base,binary,load,statement,oracle,lit
from test_gpu_refractory import refresh_code
from test_gpu_spike_generator import BACKENDS,execute
from test_gpu_monitors import setup,DT
from test_metal_delays import device

CASES=[(-2,3),(-2,33),(-.5,-3),(-1,16777215),(-1,16777216),(-1,-16777215),
       (-0.,3),(-0.,2),(0.,0.),(-0.,0.),(-2,-33),(-1,0.),(-2,2),(-2,-2),(2,3)]


def assert_power_values(actual,expected):
    """Native pow follows the existing float-function contract, not libm bits.

    Use the pre-existing expression-grid relative tolerance; deliberately use
    zero absolute tolerance so tiny results cannot disappear. Signs, exact zero
    and unit results (including parity and exponent-zero cases) remain exact.
    """
    a=np.asarray(actual);e=np.asarray(expected)
    assert a.shape==e.shape
    assert np.all(np.isfinite(a)) and np.all(np.isfinite(e))
    np.testing.assert_array_equal(np.signbit(a),np.signbit(e))
    np.testing.assert_array_equal(a==0,e==0)
    special=(e==0)|(np.abs(e)==1)
    np.testing.assert_array_equal(a[special],e[special])
    np.testing.assert_allclose(a,e,rtol=2e-5,atol=0)


def compare_power_results(actual,expected,target):
    # Only the declared power output and its monitor use the float-function
    # gate. All other arrays, event coordinates and integer state stay exact.
    from test_gpu_composed_policies import full_compare
    normalized=deepcopy(actual)
    for group in ('populations','synapses'):
        for a,e in zip(normalized[group],expected[group],strict=True):
            for field in ('states','trace'):
                if target in e.get(field,{}):
                    assert_power_values(a[field][target],e[field][target])
                    a[field][target]=e[field][target].copy()
    for a,e in zip(normalized['populations'],expected['populations'],strict=True):
        if 'event_streams' not in a:
            assert set(e['event_streams'])=={'spike'}
            a['event_streams']={'spike':dict(ticks=a['spike_ticks'],indices=a['indices'])}
    full_compare(normalized,expected)


@pytest.mark.parametrize('actual,expected',[(0.,1e-10),(-0.,0.),(-1.,1.),(.25,.5),(1.000001,1.),(float('nan'),.5)])
def test_power_contract_rejects_semantic_errors(actual,expected):
    with pytest.raises(AssertionError):assert_power_values(np.array([actual]),np.array([expected]))


@pytest.mark.parametrize('backend',BACKENDS)
@pytest.mark.parametrize('domain',['population','dag','synapse'])
def test_dynamic_signed_power_matches_full_reference(device,tmp_path,backend,domain):
    setup(tmp_path/'ref')
    p=b.NeuronGroup(len(CASES),'x:1\nexponent:1\ny:1\nsink:1',
        threshold='True' if domain=='synapse' else 'False',reset='',dt=DT,name='population')
    p.x=[x for x,y in CASES];p.exponent=[y for x,y in CASES];objects=[p]
    if domain=='synapse':
        syn=b.Synapses(p,p,'answer:1',on_pre='answer=x_pre**exponent_pre',clock=p.clock,name='projection')
        syn.connect(i=np.arange(len(CASES)),j=np.arange(len(CASES)));objects.append(syn)
    else:
        p.run_regularly('y=x**exponent');objects.append(b.StateMonitor(p,'y',record=True))
        if domain=='dag':
            syn=b.Synapses(p,p,'w:1',on_pre='sink_post+=w',clock=p.clock,name='projection')
            syn.connect(i=np.array([],np.int32),j=np.array([],np.int32));objects.append(syn)
    model=lower_network(b.Network(*objects),2*DT)
    expected=oracle(model,tmp_path/'oracle');actual=execute(model,tmp_path/backend,backend,'sparse')
    key='synapses' if domain=='synapse' else 'populations'
    target='answer' if domain=='synapse' else 'y'
    values=actual[key][0]['states'][target];reference=expected[key][0]['states'][target]
    independent=np.asarray([math.pow(x,y) for x,y in CASES],np.float32)
    np.testing.assert_array_equal(reference.astype(np.float32).view(np.uint32),independent.view(np.uint32))
    from gpu_autotune_benchmark import save_observables
    save_observables(tmp_path/'power-actual',actual);save_observables(tmp_path/'power-reference',expected)
    (tmp_path/'power-model.json').write_text(json.dumps(model,sort_keys=True)+'\n')
    np.savez_compressed(tmp_path/'power-domain.npz',cases=np.asarray(CASES),actual=values,reference=reference)
    assert_power_values(values,independent)
    compare_power_results(actual,expected,target)


@pytest.mark.parametrize('backend',BACKENDS)
@pytest.mark.parametrize('x,exponent,f64_finite',[(-2,.5,False),(0.,-1,False),(2.,128,True)])
def test_dynamic_power_faults_cannot_be_overwritten(device,tmp_path,backend,x,exponent,f64_finite):
    model,code=base(tmp_path/'ref')
    for key,value in [('x',x),('lo',exponent)]:
        model['instance']['populations'][0]['initial_state'][key]=[lit(value)['bits']]*5
    code['vector']=[statement('y',binary('pow',load('x'),load('lo'))),statement('y',lit(0))]
    code['effects']['reads']=['x','lo'];refresh_code(model,code)
    oracle(model,tmp_path/'oracle',success=f64_finite)
    with pytest.raises(FloatingPointError):execute(model,tmp_path/backend,backend)
    assert not (tmp_path/backend/'transport/summary.json').exists()
