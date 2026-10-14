"""Frozen StateMonitor aggregation and scheduled spike-variable snapshots."""
import json

import brian2 as b
import numpy as np
import pytest

from brian2_rust.cuda import build_cuda_plan
from brian2_rust.export import lower_network
from brian2_rust.metal_monitors import monitor_arrays
from brian2_rust.plan import PlanValidationError
from brian2_rust.results import load_results
from test_gpu_spike_generator import BACKENDS, execute
from test_metal_delays import device, ROOT
from test_metal_plasticity import equivalent

DT=b.second/1024


def setup(tmp_path,engine='reference',**options):
    b.set_device('rust_standalone',engine=engine,directory=tmp_path,
                 runner=ROOT/'target/release/b2-runner',**options)


def event_equivalent(actual,expected):
    equivalent(actual,expected,exact=True)
    for a,e in zip(actual['populations'],expected['populations'],strict=True):
        assert set(a.get('event_monitors',{}))==set(e.get('event_monitors',{}))
        for name,monitor in a.get('event_monitors',{}).items():
            reference=e['event_monitors'][name]
            for field in ('ticks','indices','times'):
                np.testing.assert_array_equal(monitor[field],reference[field])
            for variable,values in monitor['values'].items():
                np.testing.assert_array_equal(values,reference['values'][variable])


@pytest.mark.parametrize('backend',BACKENDS)
@pytest.mark.parametrize('coupled',[False,True])
def test_multiple_state_monitors_share_first_snapshot_and_union(device,tmp_path,backend,coupled):
    setup(tmp_path/'ref')
    pop=b.NeuronGroup(4,'v:1\ngain:1 (constant)',threshold='v>1',reset='v=0',dt=DT)
    pop.v=[0,.25,.5,.75];pop.gain=[.25,.5,.75,1]
    pop.run_regularly('v+=gain',name='m_update')
    first=b.StateMonitor(pop,'v',record=[3,1,3],name='a_first')
    second=b.StateMonitor(pop,['v','gain'],record=[2,0],name='z_last')
    empty=b.StateMonitor(pop,'gain',record=[],name='zz_empty')
    objects=[pop,first,second,empty]
    if coupled:
        target=b.NeuronGroup(1,'v:1',dt=2*DT)
        syn=b.Synapses(pop,target,on_pre='v_post+=0.125',delay=DT,clock=pop.clock)
        syn.connect();objects.extend([target,syn])
    b.Network(*objects).run(6*DT)
    model=json.loads((device.last_run_directory/'model.json').read_text())
    expected=load_results(model,device.last_run_directory/'rust')
    equivalent(execute(model,tmp_path/backend,backend,'sparse'),expected,exact=True)
    # The frozen transport's first scheduled monitor owns the union snapshot;
    # later monitor nodes cannot overwrite it after the intervening start update.
    np.testing.assert_array_equal(first.v[:,0],[.75,.25,.75])
    np.testing.assert_array_equal(second.v[:,0],[.5,0])
    if coupled:
        plan=build_cuda_plan(model,numeric_mode='float32',event_delivery='sparse')
        assert sum('/state_monitor/' in node for node in plan.elided_nodes)==2


def monitored_network(coupled,dtype=np.float64):
    pop=b.NeuronGroup(4,'v:1\ngain:1 (constant)\nshared_gain:1 (shared, constant)',
                      threshold='v>1',reset='v=0',dt=DT,name='observed',dtype=dtype)
    pop.v=[0,.5,1,1.5];pop.gain=[.25,.5,.75,1];pop.shared_gain=.5
    pop.run_regularly('v+=gain',name='advance')
    objects=[pop,b.StateMonitor(pop,'v',record=[3,1,3],name='a_state'),
             b.StateMonitor(pop,['v','gain'],record=[2,0],name='z_state')];probes=[]
    for name,when,order in [('start','start',1),('before','thresholds',-1),
                            ('threshold','thresholds',1),('synapses','after_synapses',1),
                            ('reset','after_resets',1),('end','end',1)]:
        probe=b.EventMonitor(pop,'spike',variables=['v','gain','shared_gain'],when=when,order=order,name='probe_'+name)
        objects.append(probe);probes.append(probe)
    count=b.EventMonitor(pop,'spike',variables=[],name='count_only')
    objects.append(count);probes.append(count)
    if coupled:
        other=b.NeuronGroup(2,'v:1',threshold='v>0.5',reset='v=0',dt=2*DT)
        syn=b.Synapses(pop,other,'w:1',on_pre='v_post+=w; w+=0.125',clock=pop.clock)
        syn.connect(i=[0,1,2,3],j=[0,1,0,1]);syn.w=.25;syn.delay=[0,1,2,1]*DT
        objects.extend([other,syn,b.EventMonitor(other,'spike',variables='v',when='after_resets',name='other_reset')])
    return b.Network(*objects),pop,probes


@pytest.mark.parametrize('backend',BACKENDS)
@pytest.mark.parametrize('coupled',[False,True])
@pytest.mark.parametrize('window',[None,3])
@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_event_monitor_order_variables_and_transport(device,tmp_path,backend,coupled,window,dtype):
    setup(tmp_path/'ref',recording_window_steps=window)
    net,pop,probes=monitored_network(coupled,dtype);net.run(8*DT)
    model=json.loads((device.last_run_directory/'model.json').read_text())
    expected=load_results(model,device.last_run_directory/'rust')
    actual=execute(model,tmp_path/backend,backend,'sparse')
    event_equivalent(actual,expected)
    if backend!='cpu-f32':
        loaded=load_results(model,tmp_path/backend/'transport')
        event_equivalent(loaded,actual)
        values=next(p for p in loaded['populations'] if 'probe_threshold' in p['event_monitors'])['event_monitors']['probe_threshold']['values']
        assert values['v'].dtype==np.dtype(dtype)
    # Snapshots are taken at the monitor slot, including previous-tick fired
    # flags before threshold and reset values after reset; they are not copies
    # of the standard SpikeMonitor's threshold values.
    assert len(probes[0].i)>0
    assert np.all(probes[2].v[:]>1)
    np.testing.assert_array_equal(probes[4].v[:],0)
    np.testing.assert_array_equal(probes[2].shared_gain[:],.5)


@pytest.mark.parametrize('backend',BACKENDS[1:])
@pytest.mark.parametrize('queued',[False,True])
def test_event_monitor_device_windows_continuation_restore(device,tmp_path,backend,queued):
    outputs=[]
    for engine in ('reference',backend):
        device.reinit()
        opts=dict(numeric_mode='float32',event_delivery='sparse') if engine!='reference' else {}
        setup(tmp_path/engine,engine,build_on_run=not queued,recording_window_steps=3,**opts)
        net,pop,probes=monitored_network(True)
        net.run(4*DT)
        if not queued:net.store('sampled')
        net.run(4*DT)
        if queued:device.build()
        else:
            before=[np.asarray(m.v[:]).copy() for m in probes[:-1]]
            net.restore('sampled');net.run(4*DT)
            for m,values in zip(probes[:-1],before,strict=True):np.testing.assert_array_equal(m.v[:],values)
        outputs.append([[np.asarray(m.i[:]).copy(),np.asarray(m.t[:]/b.second).copy()]+
                        [np.asarray(getattr(m,name)[:]).copy() for name in ('v','gain','shared_gain') if name in m.record_variables]
                        for m in probes])
        outputs[-1].append([np.asarray(net['a_state'].v[:]).copy(),np.asarray(net['z_state'].v[:]).copy(),
                            np.asarray(net['z_state'].gain[:]).copy()])
    for a,e in zip(*outputs,strict=True):
        for actual,expected in zip(a,e,strict=True):np.testing.assert_array_equal(actual,expected)


def test_monitor_memory_preflight_and_custom_event_plan(device,tmp_path):
    setup(tmp_path/'ref')
    net,pop,_=monitored_network(False)
    model=lower_network(net,8*DT)
    p=model['definition']['populations'][0];monitor=p['event_monitors'][0]
    with pytest.raises(MemoryError,match='EventMonitor'):monitor_arrays(p,monitor,1)
    custom=b.NeuronGroup(1,'v:1',events={'crossing':'v>1'},dt=DT)
    event=b.EventMonitor(custom,'crossing',variables='v')
    other=lower_network(b.Network(custom,event),DT)
    plan=build_cuda_plan(other,numeric_mode='float32')
    assert any(stage.role=='event-monitor' for stage in plan.dispatches)


@pytest.mark.parametrize('backend',BACKENDS)
def test_empty_event_monitor_and_default_spike_monitor_coexist(device,tmp_path,backend):
    setup(tmp_path/'ref')
    quiet=b.NeuronGroup(2,'v:1',threshold='False',reset='',dt=DT)
    active=b.NeuronGroup(2,'v:1',threshold='True',reset='v+=0.25',dt=2*DT)
    monitors=[b.EventMonitor(quiet,'spike',variables='v',name='quiet_events'),
              b.EventMonitor(active,'spike',variables='v',when='after_resets',name='active_events'),
              b.SpikeMonitor(active)]
    b.Network(quiet,active,*monitors).run(6*DT)
    model=json.loads((device.last_run_directory/'model.json').read_text())
    expected=load_results(model,device.last_run_directory/'rust')
    actual=execute(model,tmp_path/backend,backend)
    event_equivalent(actual,expected)
    if backend!='cpu-f32':event_equivalent(load_results(model,tmp_path/backend/'transport'),actual)
