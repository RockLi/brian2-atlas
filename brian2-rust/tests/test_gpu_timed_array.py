"""Native buffered TimedArray lookup, bounds and Device continuation."""
import copy
import json
import subprocess

import brian2 as b
import numpy as np
import pytest

from brian2_rust.export import lower_network
from brian2_rust.cuda import build_cuda_plan
from brian2_rust.metal import population_arrays
from brian2_rust.metal_timed_array import timed_nodes
from brian2_rust.plan import PlanValidationError
from brian2_rust.protocol import attach_protocol
from brian2_rust.results import load_results
from brian2_rust.spec import bits
from test_gpu_spike_generator import BACKENDS, execute
from test_metal_delays import device, ROOT
from test_metal_plasticity import equivalent
from test_gpu_refractory import refresh_code

DT = b.second/1024


def setup(device, tmp_path):
    b.set_device('rust_standalone', engine='reference', directory=tmp_path/'ref',
                 runner=ROOT/'target/release/b2-runner')


def oracle(model, path, *, success=True):
    path.mkdir()
    source=path/'model.json';source.write_text(json.dumps(model))
    run=subprocess.run([str(ROOT/'target/release/b2-runner'),str(source),str(path/'results')],capture_output=True,text=True)
    if not success:
        assert run.returncode!=0 and 'TimedArray' in run.stderr
        return
    assert run.returncode==0,run.stderr
    return load_results(model,path/'results')


@pytest.mark.parametrize('backend', BACKENDS)
def test_one_two_dimensions_multiclock_and_dedup(device, tmp_path, backend):
    setup(device,tmp_path)
    wave=b.TimedArray([0,1,4,2],dt=2*DT)
    field=b.TimedArray([[0,10,20],[1,11,21],[2,12,22],[3,13,23]],dt=2*DT)
    pops=[];monitors=[]
    for scale in (1,2):
        pop=b.NeuronGroup(3,'x:1\ny:1\nz:1\ngain:1 (constant)\nshift:second (constant)',
                          dt=scale*DT,namespace={'wave':wave,'field':field})
        pop.gain=[1,2,4];pop.shift=[-4,0,100]*DT
        pop.run_regularly('x=wave(t); y=field(t,i); z=gain*(wave(t)+field(t+shift,i))')
        pops.append(pop);monitors.append(b.StateMonitor(pop,['x','y','z'],record=True))
    model=lower_network(b.Network(*pops,*monitors),10*DT)
    actual=execute(model,tmp_path/backend,backend)
    equivalent(actual,oracle(model,tmp_path/'oracle'),exact=True)
    plan=build_cuda_plan(model,numeric_mode='float32')
    for p,kernel in enumerate(plan.kernels):
        arrays,_=population_arrays(model,p,kernel,512*1024**2)
        # Six parameter values + four wave + twelve field, despite repeated calls.
        assert arrays[1].size==22
    np.testing.assert_array_equal(actual['populations'][0]['states']['y'],[3,13,23])


@pytest.mark.parametrize('backend', BACKENDS)
def test_float_index_truncation_and_runtime_bounds(device,tmp_path,backend):
    setup(device,tmp_path)
    field=b.TimedArray([[2,4,8],[3,5,9]],dt=DT)
    pop=b.NeuronGroup(1,'x:1',dt=DT,namespace={'field':field});pop.run_regularly('x=field(t,i)')
    model=lower_network(b.Network(pop),2*DT)
    for index in (0.75,1.5,2.9,-0.1,3.0):
        case=copy.deepcopy(model)
        next(timed_nodes(case['definition']))['index']={'op':'literal','bits':bits(index)}
        for code in case['definition']['populations'][0]['code_objects']:refresh_code(case,code)
        path=tmp_path/(backend+str(index))
        if 0<=index<3:
            actual=execute(case,path,backend)
            np.testing.assert_array_equal(actual['populations'][0]['states']['x'],[3,5,9][int(index)])
            equivalent(actual,oracle(case,tmp_path/('oracle'+str(index))),exact=True)
        else:
            with pytest.raises(FloatingPointError,match='TimedArray'):execute(case,path,backend)
            assert not (path/'transport').exists()


@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('route', ['scan','sparse'])
def test_delayed_synaptic_time_and_subgroup_columns(device,tmp_path,backend,route):
    setup(device,tmp_path)
    field=b.TimedArray(np.arange(24).reshape(8,3)/8,dt=DT)
    source=b.SpikeGeneratorGroup(4,[0,2,1],[0,1,2]*DT,period=4*DT,dt=DT)
    target=b.NeuronGroup(5,'v:1',clock=source.clock)
    syn=b.Synapses(source,target[1:4],'w:1',on_pre='w=field(t,j); v_post+=w',
                   clock=source.clock,namespace={'field':field})
    syn.connect(i=[2,0,2,1,0],j=[2,0,0,1,0]);syn.delay=[0,1,3,1,2]*DT
    monitor=b.StateMonitor(target,'v',record=True)
    model=lower_network(b.Network(source,target,syn,monitor),10*DT)
    equivalent(execute(model,tmp_path/backend,backend,route),oracle(model,tmp_path/'oracle'),exact=True)
    invalid=copy.deepcopy(model)
    next(timed_nodes(invalid['definition']['synapses']))['index']={'op':'literal','bits':bits(3)}
    for code in invalid['definition']['synapses'][0]['code_objects']:refresh_code(invalid,code)
    with pytest.raises(FloatingPointError,match='TimedArray'):
        execute(invalid,tmp_path/'invalid',backend,route)
    assert not (tmp_path/'invalid'/'transport').exists()


@pytest.mark.parametrize('backend', BACKENDS)
def test_poisson_rates_from_timed_input(device,tmp_path,backend):
    setup(device,tmp_path)
    stimulus=b.TimedArray(np.array([[0,256,512,1024],[1024,512,256,0]])*b.Hz,dt=4*DT)
    pop=b.PoissonGroup(4,'stimulus(t,i)',dt=DT,namespace={'stimulus':stimulus})
    monitor=b.SpikeMonitor(pop)
    model=lower_network(b.Network(pop,monitor),12*DT,rng_seed=1729)
    equivalent(execute(model,tmp_path/backend,backend),oracle(model,tmp_path/'oracle'),exact=True)


@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('masked', [True,False])
def test_checked_lookup_respects_masks_and_eager_boolean_operands(device,tmp_path,backend,masked):
    setup(device,tmp_path)
    field=b.TimedArray([[1]],dt=DT)
    pop=b.NeuronGroup(1,'dv/dt=field(t,i)/second:1 (unless refractory)',dt=DT,
                      threshold='False',reset='v=0',refractory='True',method='euler',namespace={'field':field})
    model=lower_network(b.Network(pop),DT)
    code=next(c for c in model['definition']['populations'][0]['code_objects'] if c['kind']=='state_update')
    bad=copy.deepcopy(next(timed_nodes(code)));bad['index']={'op':'literal','bits':bits(1)}
    if masked:
        model['instance']['populations'][0]['refractory']['initial_not_refractory']=[False]
        stmt=next(s for s in code['vector'] if s['target']=='v')
        assert stmt['condition']=='not_refractory'
        stmt['value']=bad
    else:
        stmt=next(s for s in code['vector'] if s['target']=='not_refractory')
        stmt['value']={'op':'or','left':{'op':'boolean','value':True},'right':{
            'op':'gt','left':bad,'right':{'op':'literal','bits':bits(0)}}}
    refresh_code(model,code)
    if masked:
        equivalent(execute(model,tmp_path/backend,backend),oracle(model,tmp_path/'oracle'),exact=True)
    else:
        oracle(model,tmp_path/'oracle',success=False)
        with pytest.raises(FloatingPointError,match='TimedArray'):execute(model,tmp_path/backend,backend)


@pytest.mark.parametrize('backend', BACKENDS)
def test_time_clamping_before_integer_conversion(device,tmp_path,backend):
    setup(device,tmp_path)
    wave=b.TimedArray([10,20,30],dt=2*DT)
    stamps=np.array([-1e38,-1,0,float(DT),2*float(DT),4*float(DT),1e38])
    pop=b.NeuronGroup(len(stamps),'x:1\nstamp:second (constant)',dt=DT,namespace={'wave':wave})
    pop.stamp=stamps*b.second;pop.run_regularly('x=wave(stamp)')
    model=lower_network(b.Network(pop),DT)
    actual=execute(model,tmp_path/backend,backend)
    # Division by epsilon overflows float32 at both extremes. Clamp before cast.
    np.testing.assert_array_equal(actual['populations'][0]['states']['x'],[10,10,10,10,20,30,30])
    equivalent(actual,oracle(model,tmp_path/'oracle'),exact=True)


def test_tables_are_buffers_and_planning_rejects_unrepresentable_values(device,tmp_path):
    setup(device,tmp_path)
    wave=b.TimedArray(np.arange(10000,dtype=float),dt=DT)
    pop=b.NeuronGroup(1,'x:1',dt=DT,namespace={'wave':wave});pop.run_regularly('x=wave(t)')
    model=lower_network(b.Network(pop),DT)
    plan=build_cuda_plan(model,numeric_mode='float32')
    small=copy.deepcopy(model)
    for code in small['definition']['populations'][0]['code_objects']:
        for table in timed_nodes(code):
            table['rows']=10;table['values']=table['values'][:10]
        refresh_code(small,code)
    small_plan=build_cuda_plan(small,numeric_mode='float32')
    # Table data must stay in buffers: changing 10 rows to 10,000 may only
    # enlarge shape literals, irrespective of shared helper source length.
    assert abs(len(plan.kernels[0].source)-len(small_plan.kernels[0].source))<32
    with pytest.raises(MemoryError):population_arrays(model,0,plan.kernels[0],1024)
    for field,value in [('epsilon',bits(1e-100)),('values',[bits(1e100)]*10000)]:
        invalid=copy.deepcopy(model);next(timed_nodes(invalid['definition']))[field]=value
        attach_protocol(invalid)
        with pytest.raises(PlanValidationError,match='TimedArray'):build_cuda_plan(invalid,numeric_mode='float32')


@pytest.mark.parametrize('backend', BACKENDS[1:])
@pytest.mark.parametrize('queued', [False,True])
def test_device_segment_restore_queued_inputs(device,tmp_path,backend,queued):
    results=[]
    for engine in ('reference',backend):
        device.reinit()
        opts={'numeric_mode':'float32'} if engine!='reference' else {}
        b.set_device('rust_standalone',engine=engine,directory=tmp_path/engine,
                     runner=ROOT/'target/release/b2-runner',build_on_run=not queued,**opts)
        wave=b.TimedArray([0,1,3,2,5],dt=2*DT)
        pop=b.NeuronGroup(3,'v:1',dt=DT,namespace={'wave':wave},name='population')
        pop.run_regularly('v+=wave(t)')
        mon=b.StateMonitor(pop,'v',record=True);net=b.Network(pop,mon)
        net.run(3*DT)
        if queued:
            net.run(9*DT);device.build()
        else:
            net.store('input');net.run(9*DT)
            expected=np.asarray(mon.v).copy()
            net.restore('input');net.run(9*DT)
            np.testing.assert_array_equal(np.asarray(mon.v),expected)
        results.append((np.asarray(pop.v[:]).copy(),np.asarray(mon.v).copy()))
    for actual,expected in zip(*results,strict=True):np.testing.assert_array_equal(actual,expected)
