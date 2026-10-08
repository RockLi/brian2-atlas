"""Logical Tick precision, checked offsets and synaptic temporal expressions."""
import json
import hashlib
import subprocess

import brian2 as b
import numpy as np
import pytest

from brian2_rust.export import lower_network
from brian2_rust.protocol import attach_protocol
from brian2_rust.results import load_results
from brian2_rust.spec import bits
from brian2_rust.metal_synapses import canonical_projection
from test_gpu_spike_generator import BACKENDS, execute
from test_gpu_monitors import setup, event_equivalent, DT
from test_gpu_refractory import refresh_code, make_model
from test_metal_delays import device, ROOT

LIMIT=2**53


def step(time=0):
    return dict(op='timestep',time=dict(op='mul',left=dict(op='literal',bits=bits(float(time))),
                                      right=dict(op='load',name='dt')),dt=dict(op='load',name='dt'))


def offset(tick,value):
    return dict(op='tick_offset',tick=tick,offset=value)


def floating(tick):
    return dict(op='tick_to_f64',arg=tick)


def oracle(model,path,success=True):
    path.mkdir();source=path/'model.json';source.write_text(json.dumps(model))
    run=subprocess.run([str(ROOT/'target/release/b2-runner'),str(source),str(path/'results')],capture_output=True,text=True)
    assert (run.returncode==0)==success,run.stderr
    if success:return load_results(model,path/'results')
    assert any(word in run.stderr.lower() for word in ('timestep','finite','tick')),run.stderr


def population_model(tmp_path,expression):
    setup(tmp_path/'ref')
    pop=b.NeuronGroup(3,'x:1',dt=DT)
    pop.run_regularly('x+=1',when='groups')
    model=lower_network(b.Network(pop,b.StateMonitor(pop,'x',record=True)),3*DT)
    code=next(c for c in model['definition']['populations'][0]['code_objects'] if c['kind']=='run_regularly')
    code['scalar']=[]
    code['vector']=[dict(target='x',dtype='f64',dimensions=[0.]*7,condition=None,value=expression)]
    refresh_code(model,code)
    return model,code


@pytest.mark.parametrize('backend',BACKENDS)
def test_tick_temporaries_keep_unit_offsets_above_float32_precision(device,tmp_path,backend):
    model,code=population_model(tmp_path,floating(step()))
    # Explicit private Tick variables: these used to be emitted as float,
    # silently dropping +1 at 2^25 even if expression calls return int64.
    tick=dict(op='load',name='_tick');later=dict(op='load',name='_later')
    code['vector']=[dict(target='_tick',dtype='tick',dimensions=[0.]*7,condition=None,value=step(2**25)),
                    dict(target='_later',dtype='tick',dimensions=[0.]*7,condition=None,value=offset(tick,1)),
                    dict(target='x',dtype='f64',dimensions=[0.]*7,condition=None,
                         value=dict(op='bool_to_f64',arg=dict(op='gt',left=later,right=tick)))]
    refresh_code(model,code)
    expected=oracle(model,tmp_path/'oracle')
    np.testing.assert_array_equal(expected['populations'][0]['states']['x'],1)
    event_equivalent(execute(model,tmp_path/backend,backend),expected)


@pytest.mark.parametrize('backend',BACKENDS)
@pytest.mark.parametrize('sign',[1,-1])
@pytest.mark.parametrize('extra',[0,1,2])
def test_offset_exact_range_and_reference_boundary_ties(device,tmp_path,backend,sign,extra):
    # +/- (2^53+1) rounds to +/-2^53 in the frozen f64 executor.
    # +/- (2^53+2) is outside its accepted execution range.
    expr=floating(offset(offset(step(),sign*LIMIT),sign*extra))
    model,_=population_model(tmp_path,expr)
    expected=oracle(model,tmp_path/'oracle',success=extra!=2)
    if extra==2:
        with pytest.raises(FloatingPointError,match='tick_offset'):
            execute(model,tmp_path/backend,backend)
        assert not (tmp_path/backend/'transport'/'summary.json').exists()
    else:
        np.testing.assert_array_equal(expected['populations'][0]['states']['x'],sign*LIMIT)
        event_equivalent(execute(model,tmp_path/backend,backend),expected)


@pytest.mark.parametrize('backend',BACKENDS)
def test_timestep_rejects_values_outside_reference_exact_range(device,tmp_path,backend):
    model,_=population_model(tmp_path,floating(step(2**54)))
    oracle(model,tmp_path/'oracle',success=False)
    with pytest.raises(FloatingPointError,match='timestep'):
        execute(model,tmp_path/backend,backend)
    assert not (tmp_path/backend/'transport'/'summary.json').exists()


@pytest.mark.parametrize('backend',BACKENDS)
@pytest.mark.parametrize('masked',[False,True])
def test_tick_fault_obeys_statement_mask(device,tmp_path,backend,masked):
    model=make_model(device,tmp_path,'masked')
    code=next(c for c in model['definition']['populations'][0]['code_objects'] if c['kind']=='state_update')
    next(s for s in code['vector'] if s['target']=='v')['value']=floating(offset(offset(step(),LIMIT),2))
    model['instance']['populations'][0]['refractory']['initial_not_refractory']=[not masked]*3
    refresh_code(model,code)
    expected=oracle(model,tmp_path/'oracle',success=masked)
    if masked:event_equivalent(execute(model,tmp_path/backend,backend),expected)
    else:
        with pytest.raises(FloatingPointError,match='tick_offset'):execute(model,tmp_path/backend,backend)


@pytest.mark.parametrize('backend',BACKENDS)
def test_logical_operand_eager_fault_is_not_short_circuited(device,tmp_path,backend):
    bad=offset(offset(step(),LIMIT),2)
    expr=dict(op='bool_to_f64',arg=dict(op='or',left=dict(op='boolean',value=True),
             right=dict(op='eq',left=bad,right=step())))
    model,_=population_model(tmp_path,expr)
    oracle(model,tmp_path/'oracle',success=False)
    with pytest.raises(FloatingPointError,match='tick_offset'):execute(model,tmp_path/backend,backend)


def synaptic_network(expression='timestep(t,dt)+1',empty=False,plastic=False):
    source=b.NeuronGroup(2,'v:1',threshold='False' if empty else 'True',reset='',dt=DT,name='source')
    target=b.NeuronGroup(3,'v:1',threshold='v>100',reset='v=0',dt=2*DT,name='target')
    syn=b.Synapses(source,target,'w:1',on_pre=f'v_post += {expression}'+('; w+=1' if plastic else ''),
                   on_post='w+=timestep(t,dt)+1' if plastic else None,clock=source.clock)
    syn.connect(i=[1,0,0],j=[1,0,1]);syn.pre.delay=[0,1,2]*DT
    if plastic:syn.post.delay=np.asarray([0,1,2])*2*DT
    monitor=b.EventMonitor(target,'spike',variables='v',when='end')
    return b.Network(source,target,syn,monitor),target,syn,monitor


@pytest.mark.parametrize('backend',BACKENDS)
@pytest.mark.parametrize('route',['scan','sparse'])
def test_synaptic_timestep_delays_and_own_clock(device,tmp_path,backend,route):
    setup(tmp_path/'ref');net,*_=synaptic_network()
    model=lower_network(net,8*DT)
    assert canonical_projection(model,0)
    event_equivalent(execute(model,tmp_path/backend,backend,route),oracle(model,tmp_path/'oracle'))


@pytest.mark.parametrize('backend',BACKENDS)
@pytest.mark.parametrize('empty',[False,True])
def test_synaptic_scalar_timestep_fault_even_without_events(device,tmp_path,backend,empty):
    setup(tmp_path/'ref');net,*_=synaptic_network(empty=empty)
    model=lower_network(net,2*DT)
    code=next(c for c in model['definition']['synapses'][0]['code_objects'] if c['kind']=='synapses')
    code['scalar'].append(dict(target='_fault',dtype='tick',dimensions=[0.]*7,condition=None,value=step(-1)))
    refresh_code(model,code)
    oracle(model,tmp_path/'oracle',success=False)
    with pytest.raises(FloatingPointError,match='timestep'):execute(model,tmp_path/backend,backend)
    assert not (tmp_path/backend/'transport'/'summary.json').exists()


@pytest.mark.parametrize('backend',BACKENDS)
def test_synaptic_temporal_call_selects_checked_domain(device,tmp_path,backend):
    setup(tmp_path/'ref')
    source=b.NeuronGroup(1,'v:1',threshold='True',reset='',dt=DT)
    target=b.NeuronGroup(1,'v:1',dt=DT)
    # A temporal operation nested inside a portable Function must select
    # the checked synapse domain just like a direct expression.
    syn=b.Synapses(source,target,on_pre='v_post+=1',clock=source.clock);syn.connect()
    model=lower_network(b.Network(source,target,syn),4*DT)
    body=floating(offset(dict(op='timestep',time=dict(op='load',name='value'),dt=dict(op='load',name='dt')),1))
    model['definition']['functions'].append(dict(name='time_value',semantic_version='1.0.0',abi='b2ir-function-v1',
        effects=dict(stateful=False,deterministic=True,thread_safe=True,rng=False),
        implementations={'b2ir-expression-v1':hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':')).encode()).hexdigest()},
        backend_implementations={},arguments=[
        dict(name=name,dtype='f64',dimensions=[0.,0.,1.,0.,0.,0.,0.]) for name in ('value','dt')],
        return_dtype='f64',return_dimensions=[0.]*7,body=body))
    code=next(c for c in model['definition']['synapses'][0]['code_objects'] if c['kind']=='synapses')
    code['vector'][-1]['value']=dict(op='add',left=dict(op='load',name='v_post'),right=dict(op='call',function='time_value',
       arguments=[dict(op='load',name='t'),dict(op='load',name='dt')]))
    code['effects']['reads'].append('t')
    refresh_code(model,code)
    assert canonical_projection(model,0)
    event_equivalent(execute(model,tmp_path/backend,backend),oracle(model,tmp_path/'oracle'))


@pytest.mark.parametrize('backend',BACKENDS[1:])
@pytest.mark.parametrize('queued',[False,True])
def test_tick_synapses_pending_restore_and_queued(device,tmp_path,backend,queued):
    outputs=[]
    for engine in ('reference',backend):
        device.reinit();setup(tmp_path/engine,engine,build_on_run=not queued,
             **(dict(numeric_mode='float32',event_delivery='sparse') if engine!='reference' else {}))
        net,target,syn,monitor=synaptic_network(plastic=True)
        net.run(4*DT)
        if not queued:net.store('pending')
        net.run(12*DT)
        if queued:device.build()
        else:
            before=np.asarray(target.v[:]).copy();net.restore('pending');net.run(12*DT)
            np.testing.assert_array_equal(target.v[:],before)
        outputs.append([np.asarray(x).copy() for x in (target.v[:],syn.w[:],monitor.v[:],monitor.i[:],monitor.t[:])])
    for actual,expected in zip(*outputs,strict=True):np.testing.assert_array_equal(actual,expected)


@pytest.mark.parametrize('backend',BACKENDS)
def test_quiet_reset_vector_does_not_evaluate_tick_fault(device,tmp_path,backend):
    model=make_model(device,tmp_path,'constant')
    codes=model['definition']['populations'][0]['code_objects']
    threshold=next(c for c in codes if c['kind']=='threshold')
    next(s for s in threshold['vector'] if s['target']=='_cond')['value']=dict(op='boolean',value=False)
    refresh_code(model,threshold)
    reset=next(c for c in codes if c['kind']=='reset')
    next(s for s in reset['vector'] if s['target']=='v')['value']=floating(offset(offset(step(),LIMIT),2))
    refresh_code(model,reset)
    event_equivalent(execute(model,tmp_path/backend,backend),oracle(model,tmp_path/'oracle'))


def test_frontend_inplace_tick_conversion_remains_aot_compatible(device,tmp_path):
    outputs=[]
    for engine in ('reference','aot'):
        device.reinit();setup(tmp_path/engine,engine)
        net,target,syn,monitor=synaptic_network(plastic=True)
        net.run(16*DT)
        outputs.append([np.asarray(x).copy() for x in (target.v[:],syn.w[:],monitor.v[:],monitor.i[:],monitor.t[:])])
    for actual,expected in zip(*outputs,strict=True):np.testing.assert_array_equal(actual,expected)
