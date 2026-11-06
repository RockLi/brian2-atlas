"""Integer widths, wrapping operations and bools across native GPU buffers."""
import json
import subprocess

import brian2 as b
import numpy as np
import pytest

from brian2_rust.results import load_results
from test_gpu_spike_generator import BACKENDS, execute
from test_gpu_monitors import setup, event_equivalent, DT
from test_metal_delays import device, ROOT
from test_population import portable_integer_step, portable_boolean_not


def typed_population(events):
    pop=b.NeuronGroup(4,'''a : integer
        wide : integer
        u : integer
        huge : integer
        r : integer
        q : integer
        enabled : boolean
        increment : integer (constant, shared)
        divisor : integer (constant)
        v : 1''',threshold='enabled',reset='v+=0.25',dt=DT,
        dtype=dict(a=np.int32,wide=np.int64,u=np.uint32,huge=np.uint64,r=np.int64,q=np.int32,increment=np.int64),
        namespace=dict(portable_integer_step=portable_integer_step,portable_boolean_not=portable_boolean_not))
    pop.a=[2**31-1,-2**31,-7,0x7f800001]
    pop.wide=[2**63-1,-2**63,2**53+1,0x7f8000017f800001]
    pop.u=[2**32-1,2**31+1,0x7f800001,0]
    pop.huge=[2**64-1,2**63+3,0x7f8000017f800001,2**53+1]
    pop.divisor=[-1,3,-3,7];pop.increment=2**53+1
    pop.run_regularly('''r = wide % -3
        q = a // divisor
        wide = portable_integer_step(wide)
        a = -a + 3
        u = u * 3 + 1
        huge = huge * 3 + 1
        enabled = portable_boolean_not(enabled)
        wide += increment''',when='groups')
    variables=['a','wide','u','huge','r','q','enabled','increment','divisor','v']
    objects=[pop,b.StateMonitor(pop,variables,record=[3,1,3,0])]
    if events:objects.append(b.EventMonitor(pop,'spike',variables=variables,when='end'))
    return b.Network(*objects),pop


@pytest.mark.parametrize('backend',BACKENDS)
@pytest.mark.parametrize('events',[False,True])
def test_typed_states_parameters_functions_and_monitors(device,tmp_path,backend,events):
    setup(tmp_path/'ref')
    net,pop=typed_population(events);net.run(4*DT)
    model=json.loads((device.last_run_directory/'model.json').read_text())
    expected=load_results(model,device.last_run_directory/'rust')
    actual=execute(model,tmp_path/backend,backend)
    event_equivalent(actual,expected)
    for name in ('a','wide','u','huge','enabled'):
        assert actual['populations'][0]['states'][name].dtype==expected['populations'][0]['states'][name].dtype
    if backend!='cpu-f32':event_equivalent(load_results(model,tmp_path/backend/'transport'),actual)


def synaptic_network():
    source=b.NeuronGroup(3,'c:integer\nv:1',threshold='True',reset='',dt=DT,dtype=dict(c=np.int64),name='source')
    target=b.NeuronGroup(4,'c:integer\nbits:integer\nflag:boolean\nv:1\ntotal:1',threshold='True',reset='',dt=2*DT,
                         dtype=dict(c=np.int64,bits=np.uint64),name='target')
    source.c=[2**53+1,2**53+2,2**53+3];target.c=2**63-3;target.bits=2**64-1
    syn=b.Synapses(source[1:],target[1:],'''w:integer
        k:integer (constant)
        wide:integer
        enabled:boolean
        dx/dt=128*Hz:1 (clock-driven)
        total_post=x:1 (summed)''',on_pre='''w += 1
        wide += c_pre
        c_post += wide
        bits_post += 1
        flag_post = enabled
        enabled = not enabled
        v_post += x''',on_post='w = w // k',clock=source.clock,method='euler',
        dtype=dict(w=np.int32,wide=np.int64),name='typed_synapse')
    syn.connect(i=[1,0,0,1],j=[1,0,0,2]);syn.w=[-7,-2**31,2**31-1,0x7f800001]
    syn.wide=[2**53+1,-2**63,2**63-1,2**53+3];syn.k=[-3,-1,3,-3]
    syn.pre.delay=[0,1,2,3]*DT;syn.post.delay=np.array([0,1,2,1])*2*DT
    monitors=[b.StateMonitor(target,['c','bits','flag','v','total'],record=True),
              b.EventMonitor(target,'spike',variables=['c','bits','flag','v','total'],when='end')]
    return b.Network(source,target,syn,*monitors),target,syn,monitors


@pytest.mark.parametrize('backend',BACKENDS)
@pytest.mark.parametrize('route',['scan','sparse'])
def test_typed_delayed_pre_post_plasticity_and_subgroups(device,tmp_path,backend,route):
    setup(tmp_path/'ref');net,target,syn,monitors=synaptic_network();net.run(8*DT)
    model=json.loads((device.last_run_directory/'model.json').read_text())
    expected=load_results(model,device.last_run_directory/'rust')
    actual=execute(model,tmp_path/backend,backend,route)
    event_equivalent(actual,expected)
    if backend!='cpu-f32':event_equivalent(load_results(model,tmp_path/backend/'transport'),actual)


@pytest.mark.parametrize('backend',BACKENDS[1:])
@pytest.mark.parametrize('queued',[False,True])
def test_typed_device_pending_restore_and_queued(device,tmp_path,backend,queued):
    outputs=[]
    for engine in ('reference',backend):
        device.reinit()
        setup(tmp_path/engine,engine,build_on_run=not queued,
              **(dict(numeric_mode='float32',event_delivery='sparse') if engine!='reference' else {}))
        net,target,syn,monitors=synaptic_network();net.run(4*DT)
        if not queued:net.store('pending')
        net.run(4*DT)
        if queued:device.build()
        else:
            before=np.asarray(syn.wide[:]).copy();net.restore('pending');net.run(4*DT)
            np.testing.assert_array_equal(syn.wide[:],before)
        outputs.append([np.asarray(syn.wide[:]).copy(),np.asarray(syn.w[:]).copy(),
                        np.asarray(target.c[:]).copy(),np.asarray(target.bits[:]).copy(),
                        np.asarray(monitors[1].c[:]).copy()])
    for a,e in zip(*outputs,strict=True):np.testing.assert_array_equal(a,e)


@pytest.mark.parametrize('backend',BACKENDS)
def test_float_integer_cast_saturation_and_integer_narrowing(device,tmp_path,backend):
    setup(tmp_path/'ref')
    pop=b.NeuronGroup(6,'v:1 (constant)\na:integer\nbb:integer\nc:integer\nd:integer\nsmall:integer',dt=DT,
                      dtype=dict(a=np.int32,bb=np.int64,c=np.uint32,d=np.uint64,small=np.int32))
    pop.v=np.array([-2**80,-2**63,0,1.75,2**63,2**80],np.float64)
    pop.run_regularly('a=int(v); bb=int(v); c=int(v); d=int(v); small=bb',when='groups')
    b.Network(pop).run(DT)
    model=json.loads((device.last_run_directory/'model.json').read_text())
    event_equivalent(execute(model,tmp_path/backend,backend),load_results(model,device.last_run_directory/'rust'))


@pytest.mark.parametrize('backend',BACKENDS)
@pytest.mark.parametrize('masked',[False,True])
def test_integer_division_faults_respect_statement_masks(device,tmp_path,backend,masked):
    from test_gpu_refractory import make_model,refresh_code
    model=make_model(device,tmp_path,'masked')
    pop=model['definition']['populations'][0]
    code=next(c for c in pop['code_objects'] if c['kind']=='state_update')
    statement=next(s for s in code['vector'] if s['target']=='v')
    statement['value']=dict(op='cast',dtype='f64',arg=dict(op='mod',
        left=dict(op='integer',dtype='i32',value='7'),right=dict(op='integer',dtype='i32',value='0')))
    model['instance']['populations'][0]['refractory']['initial_not_refractory']=[not masked]*pop['count']
    refresh_code(model,code)
    path=tmp_path/'fault.json';path.write_text(json.dumps(model))
    check=subprocess.run([str(ROOT/'target/release/b2-runner'),str(path),str(tmp_path/'expected')],capture_output=True,text=True)
    if masked:
        assert check.returncode==0,check.stderr
        event_equivalent(execute(model,tmp_path/backend,backend),load_results(model,tmp_path/'expected'))
    else:
        assert check.returncode!=0 and 'modulo by zero' in check.stderr
        with pytest.raises(FloatingPointError):execute(model,tmp_path/backend,backend)


@pytest.mark.parametrize('backend',BACKENDS)
@pytest.mark.parametrize('dtype',[np.int32,np.int64,np.uint32,np.uint64])
def test_integer_division_widths_and_signed_minimum(device,tmp_path,backend,dtype):
    from test_gpu_refractory import refresh_code
    setup(tmp_path/'ref');signed=np.dtype(dtype).kind=='i';info=np.iinfo(dtype)
    names=['a','divisor','rem','quot','absolute','negative']
    pop=b.NeuronGroup(4,'\n'.join(name+':integer' for name in names),dt=DT,dtype={name:dtype for name in names})
    pop.a=np.array([info.min,info.max,-7 if signed else 7,7],dtype=dtype)
    pop.divisor=np.array([-1 if signed else 1,3,-3 if signed else 3,3],dtype=dtype)
    pop.run_regularly('rem=a%divisor; quot=a//divisor; absolute=a; negative=a',when='groups')
    b.Network(pop).run(DT)
    model=json.loads((device.last_run_directory/'model.json').read_text())
    if signed:
        code=next(c for c in model['definition']['populations'][0]['code_objects'] if c['kind']=='run_regularly')
        for statement in code['vector']:
            if statement['target'] in {'absolute','negative'}:
                statement['value']=dict(op='abs' if statement['target']=='absolute' else 'neg',arg=dict(op='load',name='a'))
        refresh_code(model,code)
    path=tmp_path/'model.json';path.write_text(json.dumps(model))
    subprocess.run([str(ROOT/'target/release/b2-runner'),str(path),str(tmp_path/'expected')],check=True,capture_output=True)
    event_equivalent(execute(model,tmp_path/backend,backend),load_results(model,tmp_path/'expected'))


@pytest.mark.parametrize('backend',BACKENDS)
def test_checked_integer_temporary_in_floating_synapse(device,tmp_path,backend):
    setup(tmp_path/'ref')
    source=b.NeuronGroup(2,'v:1',threshold='True',reset='',dt=DT);source.v=[-7,7]
    target=b.NeuronGroup(2,'v:1',clock=source.clock)
    syn=b.Synapses(source,target,on_pre='v_post += int(v_pre)//3',clock=source.clock)
    syn.connect(i=[0,1,1],j=[0,0,1]);b.Network(source,target,syn).run(2*DT)
    model=json.loads((device.last_run_directory/'model.json').read_text())
    event_equivalent(execute(model,tmp_path/backend,backend,'sparse'),load_results(model,device.last_run_directory/'rust'))


def test_typed_monitor_allocation_counts_both_integer_words(device,tmp_path):
    from brian2_rust.cuda import build_cuda_plan
    from brian2_rust.metal import population_arrays
    from brian2_rust.metal_monitors import monitor_arrays
    setup(tmp_path/'ref')
    pop=b.NeuronGroup(2,'wide:integer',dt=DT,dtype=dict(wide=np.int64))
    monitor=b.StateMonitor(pop,'wide',record=True);b.Network(pop,monitor).run(8*DT)
    model=json.loads((device.last_run_directory/'model.json').read_text());definition=model['definition']['populations'][0]
    plan=build_cuda_plan(model,numeric_mode='float32')
    with pytest.raises(MemoryError,match='recording buffer'):
        population_arrays(model,0,plan.kernels[0],80)
    with pytest.raises(MemoryError,match='EventMonitor snapshot'):
        monitor_arrays(definition,dict(variables=['wide']),80)
