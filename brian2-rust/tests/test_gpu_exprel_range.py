"""Finite exprel results must not fail because an internal exp overflows."""
from decimal import Decimal,localcontext
import json
import brian2 as b
import numpy as np
import pytest
from brian2_rust.export import lower_network
from test_gpu_expression_contract import load,unary,statement,portable,oracle
from test_gpu_refractory import refresh_code
from test_gpu_spike_generator import BACKENDS,execute
from test_gpu_monitors import setup,DT
from test_metal_delays import device
from test_metal_plasticity import equivalent


# Neighbouring f32 inputs straddle the last finite result, independently checked
# using 80-digit Decimal exponentiation, not the shared GPU/CPU helper.
BOUND=np.float32(93.25821)
VALUES=np.asarray([-80,-1,-.126,-.125,-.0011,-.0001,0,.0001,.0011,.125,.126,1,80,88,
    np.nextafter(np.float32(88.722839),np.float32(-np.inf)),np.float32(88.722839),
    90,93,93.25,np.nextafter(BOUND,np.float32(-np.inf)),BOUND],np.float32)
VALUES=np.unique(np.r_[VALUES,np.linspace(-.13,.13,521,dtype=np.float32),
    np.nextafter(np.float32(.125),np.float32(0)),
    np.nextafter(np.float32(-.125),np.float32(0))])


def reference(values):
    with localcontext() as ctx:
        ctx.prec=80
        return np.array([float((x.exp()-1)/x) if x else 1.
            for x in map(lambda v:Decimal(float(v)),values)],np.float64)


def model_at(path,domain,values):
    setup(path)
    pop=b.NeuronGroup(len(values),'v:1' if domain=='synapse' else 'x:1\ny:1\nz:1',threshold='True' if domain=='synapse' else 'False',reset='',dt=DT,name='population')
    if domain!='synapse':pop.x=values
    objects=[pop]
    if domain=='synapse':
        syn=b.Synapses(pop,pop,'x:1\ny:1\nz:1',on_pre='y=x; z=x',clock=pop.clock,name='projection')
        syn.connect(i=np.arange(len(values)),j=np.arange(len(values)));syn.x=values;objects.append(syn)
    else:
        pop.run_regularly('y=x; z=x')
        objects.append(b.StateMonitor(pop,['y','z'],record=True))
        if domain=='dag':
            syn=b.Synapses(pop,pop,'w:1',on_pre='y_post+=w',clock=pop.clock,name='projection')
            syn.connect(i=np.array([],dtype=np.int32),j=np.array([],dtype=np.int32));objects.append(syn)
    model=lower_network(b.Network(*objects),2*DT)
    key='synapses' if domain=='synapse' else 'populations'
    code=next(c for c in model['definition'][key][0]['code_objects'] if c['kind']==('synapses' if domain=='synapse' else 'run_regularly'))
    portable(model,'stable_relative_exp',unary('exprel',load('arg')))
    code['vector']=[statement('y',unary('exprel',load('x'))),
        statement('z',dict(op='call',function='stable_relative_exp',arguments=[load('x')]))]
    refresh_code(model,code)
    return model,key


@pytest.mark.parametrize('backend',BACKENDS)
@pytest.mark.parametrize('domain',['population','dag','synapse'])
def test_finite_range_and_nested_function(device,tmp_path,backend,domain):
    model,key=model_at(tmp_path/'ref',domain,VALUES)
    expected=reference(VALUES)
    assert np.isfinite(expected.astype(np.float32)).all()
    independent=oracle(model,tmp_path/'oracle')
    np.testing.assert_allclose(independent[key][0]['states']['y'],expected,rtol=2e-14,atol=2e-14)
    (tmp_path/'exprel-model.json').write_text(json.dumps(model)+'\n')
    actual=execute(model,tmp_path/backend,backend,'sparse')
    equivalent(actual,independent)
    arrays={'input':VALUES,'decimal_reference':expected}
    for name in ('y','z'):
        value=actual[key][0]['states'][name]
        assert np.isfinite(value).all()
        np.testing.assert_allclose(value,expected,rtol=2e-5,atol=1e-7)
        arrays['actual/'+name]=value
        arrays['reference/'+name]=independent[key][0]['states'][name]
    np.testing.assert_array_equal(arrays['actual/y'],arrays['actual/z'])
    np.savez_compressed(tmp_path/'exprel-results.npz',**arrays)


@pytest.mark.parametrize('backend',BACKENDS)
@pytest.mark.parametrize('domain',['population','dag','synapse'])
def test_result_overflow_still_fails(device,tmp_path,backend,domain):
    values=np.array([np.nextafter(BOUND,np.float32(np.inf))],np.float32)
    with np.errstate(over='ignore'):assert not np.isfinite(reference(values).astype(np.float32)).any()
    model,key=model_at(tmp_path/'ref',domain,values)
    independent=oracle(model,tmp_path/'oracle')
    assert np.isfinite(independent[key][0]['states']['y']).all()
    (tmp_path/'exprel-overflow-model.json').write_text(json.dumps(model)+'\n')
    with pytest.raises(FloatingPointError):execute(model,tmp_path/backend,backend,'sparse')
    assert not (tmp_path/backend/'transport/summary.json').exists()
    (tmp_path/'exprel-overflow.json').write_text(json.dumps(dict(input=float(values[0]),
        decimal_reference=float(reference(values)[0]),expected='float32-result-overflow',raised=True))+'\n')


@pytest.mark.parametrize('backend',BACKENDS)
@pytest.mark.parametrize('neurons',[32,4096])
def test_hh_full_arrays_match_updated_f32_control(device,tmp_path,backend,neurons):
    from gpu_hh import brian_run,configuration,oracle as hh_oracle,checks
    config=configuration(neurons,512)
    expected=hh_oracle(neurons,512)
    results={}
    for label,engine in [('reference','cpu-f32'),('actual',backend)]:
        device.reinit()
        directory=tmp_path/label;directory.mkdir()
        results[label],_=brian_run(engine,neurons,512,directory,False)
    for key,value in results['actual'].items():
        np.testing.assert_array_equal(value,results['reference'][key],err_msg=key)
    gates,details=checks(results['actual'],expected,config)
    # The existing HH f64 gate previously failed. Keep that diagnostic intact;
    # this regression requires full-array equality to the updated f32 control.
    np.savez_compressed(tmp_path/'exprel-hh-results.npz',
        **{label+'/'+k:v for label,values in {**results,'independent':expected}.items() for k,v in values.items()})
    (tmp_path/'exprel-hh-checks.json').write_text(json.dumps(dict(configuration=config,
        f32_exact=True,f64_gates=gates,f64_details=details,f64_passed=all(gates.values())))+'\n')
