"""Typed constant exponents must retain casts before integer-power selection."""
import hashlib,json,subprocess
from pathlib import Path
import brian2 as b
import numpy as np
import pytest
from brian2_rust.export import lower_network
from brian2_rust.native import _constant_number,write_project
from brian2_rust.results import load_results
from test_gpu_expression_contract import lit,load,binary,unary,statement,oracle
from test_gpu_refractory import refresh_code
from test_gpu_spike_generator import BACKENDS,execute
from test_gpu_monitors import setup,DT
from test_metal_delays import device


def cast(dtype,value):return dict(op='cast',dtype=dtype,arg=value)
def integer(dtype,value):return dict(op='integer',dtype=dtype,value=str(value))


EXPONENTS={
    'saturate':(cast('f64',cast('u32',lit(-1))),0),
    'saturate64':(cast('f64',cast('u64',lit(-3))),0),
    'narrowzero':(cast('f64',cast('u32',integer('u64',2**32))),0),
    'narrow':(cast('f64',cast('i32',integer('u64',2**32+1))),1),
    'narrowneg':(cast('f64',cast('i32',integer('u64',2**32-1))),-1),
    'truncate':(cast('f64',cast('i32',lit(1.75))),1),
    'rounded':(unary('f32_to_f64',unary('f64_to_f32',lit(2.99999999))),3),
    'negated':(unary('neg',cast('f64',cast('u32',lit(-1)))),0),
    'literal':(lit(3),3),
    'negative':(unary('neg',lit(3)),-3),
    'promoted':(cast('f64',integer('i64',2)),2),
    'promotedliteral':(cast('f64',lit(2)),2),
}
VALUES=np.array([-2,-.5,.5,1,2],dtype=np.float64)


def model_at(path,domain):
    setup(path)
    names={key:'out'+str(i) for i,key in enumerate(EXPONENTS)}
    equations='x:1\nsink:1'+('' if domain=='synapse' else '\n'+'\n'.join(n+':1' for n in names.values()))
    pop=b.NeuronGroup(len(VALUES),equations,threshold='True' if domain=='synapse' else 'False',reset='',dt=DT,name='population')
    pop.x=VALUES;objects=[pop]
    if domain=='synapse':
        syn=b.Synapses(pop,pop,'\n'.join(n+':1' for n in names.values()),
                      on_pre='; '.join(n+'=x_pre' for n in names.values()),clock=pop.clock,name='projection')
        syn.connect(i=np.arange(len(VALUES)),j=np.arange(len(VALUES)));objects.append(syn)
    else:
        pop.run_regularly('; '.join(n+'=x' for n in names.values()))
        objects.append(b.StateMonitor(pop,list(names.values()),record=True))
        if domain=='dag':
            syn=b.Synapses(pop,pop,'w:1',on_pre='sink_post+=w',clock=pop.clock,name='projection')
            syn.connect(i=np.array([],dtype=np.int32),j=np.array([],dtype=np.int32));objects.append(syn)
    model=lower_network(b.Network(*objects),2*DT)
    owner=model['definition']['synapses' if domain=='synapse' else 'populations'][0]
    code=next(c for c in owner['code_objects'] if c['kind']==('synapses' if domain=='synapse' else 'run_regularly'))
    code['vector']=[statement(names[k],binary('pow',load('x_pre' if domain=='synapse' else 'x'),e)) for k,(e,_) in EXPONENTS.items()]
    refresh_code(model,code)
    return model,names


@pytest.mark.parametrize('backend',[*BACKENDS,'aot'])
@pytest.mark.parametrize('domain',['population','dag','synapse'])
def test_typed_power_exponents_match_independent_reference(device,tmp_path,backend,domain):
    model,names=model_at(tmp_path/'ref',domain)
    expected=oracle(model,tmp_path/'oracle')
    if backend=='aot':
        source,instance,_=write_project(model,tmp_path/'aot');exe=source.parent/'b2-native'
        subprocess.run(['rustc','--edition=2021','-O',str(source),'-o',str(exe)],check=True,capture_output=True)
        subprocess.run([str(exe),str(instance),str(tmp_path/'aot-result')],check=True,capture_output=True)
        actual=load_results(model,tmp_path/'aot-result')
    else:actual=execute(model,tmp_path/backend,backend,'sparse')
    key='synapses' if domain=='synapse' else 'populations'
    arrays={'input':VALUES}
    for name,(_,exponent) in EXPONENTS.items():
        a=actual[key][0]['states'][names[name]];e=expected[key][0]['states'][names[name]]
        np.testing.assert_array_equal(e,VALUES**exponent,err_msg=name)
        # Typed/dynamic exponents reach native pow; only proven constants
        # use the exact multiplication specialization on these dyadic inputs.
        if backend=='aot' or _constant_number(EXPONENTS[name][0]) is not None:
            np.testing.assert_array_equal(a,e,err_msg=name)
        else:
            from test_gpu_power_domain import assert_power_values
            assert_power_values(a,e)
        arrays['actual/'+name]=a;arrays['reference/'+name]=e
    (tmp_path/'power-model.json').write_text(json.dumps(model,sort_keys=True)+'\n')
    np.savez_compressed(tmp_path/'power-results.npz',**arrays)
    (tmp_path/'power-checks.json').write_text(json.dumps(dict(backend=backend,domain=domain,
        exponents={k:v for k,(_,v) in EXPONENTS.items()},arrays={k:dict(dtype=str(v.dtype),shape=list(v.shape),sha256=hashlib.sha256(v.tobytes()).hexdigest()) for k,v in arrays.items()}),indent=2)+'\n')


def test_integer_power_specialization_preserves_only_proven_constant_values():
    for key,(node,expected) in EXPONENTS.items():
        assert _constant_number(node)==(expected if key in {'literal','negative','promoted','promotedliteral'} else None)


@pytest.mark.parametrize('backend',BACKENDS)
def test_saturated_zero_exponent_does_not_invent_division_fault(device,tmp_path,backend):
    from test_gpu_expression_contract import base
    model,code=base(tmp_path/'ref')
    model['instance']['populations'][0]['initial_state']['x']=[lit(0)['bits']]*5
    code['vector']=[statement('y',binary('pow',load('x'),EXPONENTS['saturate'][0]))]
    refresh_code(model,code)
    expected=oracle(model,tmp_path/'oracle')
    actual=execute(model,tmp_path/backend,backend)
    np.testing.assert_array_equal(expected['populations'][0]['states']['y'],np.ones(5))
    np.testing.assert_array_equal(actual['populations'][0]['states']['y'],np.ones(5))
