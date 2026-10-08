"""Independent named event flags, routes, histories and continuation on GPU."""
import json

import brian2 as b
import numpy as np
import pytest

from brian2_rust.results import load_results
from test_gpu_spike_generator import BACKENDS, execute
from test_gpu_monitors import event_equivalent
from test_metal_delays import device, ROOT

DT=b.second/1024


def setup(path,engine='reference',**options):
    b.set_device('rust_standalone',engine=engine,directory=path,runner=ROOT/'target/release/b2-runner',**options)


def network(mutable):
    source=b.NeuronGroup(4,'dv/dt=512*Hz:1 (unless refractory)\ndx/dt=256*Hz:1\ny:1',
        threshold='v>=1',reset='v=0',refractory=2*DT,
        events={'alpha':'x>=0.5','zeta':'x>=0.25'},dt=DT,method='euler',name='source')
    source.run_on_event('alpha','x=0')
    source.run_on_event('zeta','y+=0.125')
    source.x=[0,.25,0,.25]
    target=b.NeuronGroup(5,'v:1',threshold='v>=2',reset='v=0',events={'reply':'v>=0.5'},dt=2*DT,name='target')
    pathways={'spike_path':'v_post+=w','alpha_path':'v_post+=2*w','zeta_path':'v_post+=4*w'}
    if mutable:pathways={name:code+'; w+=0.0625' for name,code in pathways.items()}
    options=dict(on_post='w-=0.03125') if mutable else {}
    events=dict(spike_path='spike',alpha_path='alpha',zeta_path='zeta')
    if mutable:events['post']='reply'
    syn=b.Synapses(source[1:4],target[1:5],'w:1'+('' if mutable else ' (constant)'),
                   on_pre=pathways,on_event=events,clock=source.clock,name='projection',**options)
    syn.connect(i=[2,0,1,0,2],j=[1,1,2,1,0]);syn.w=.125
    syn.spike_path.delay=[0,1,2,0,3]*DT
    syn.alpha_path.delay=[2,0,3,1,2]*DT
    syn.zeta_path.delay=0*DT
    if mutable:syn.post.delay=np.array([0,1,2,1,3])*2*DT
    probes=[b.EventMonitor(source,event,variables=['v','x','y'],name='source_'+event) for event in ('alpha','spike','zeta')]
    probes.append(b.EventMonitor(source,'alpha',variables=['x','y'],when='end',name='alpha_after_reset'))
    probes.append(b.EventMonitor(target,'reply',variables='v',name='target_reply'))
    states=[b.StateMonitor(source,['v','x','y'],record=True),b.StateMonitor(target,'v',record=True)]
    spike=b.SpikeMonitor(source)
    return b.Network(source,target,syn,*probes,*states,spike),source,target,syn,probes,states,spike


def compare(actual,expected):
    event_equivalent(actual,expected)
    for a,e in zip(actual['populations'],expected['populations'],strict=True):
        assert set(a['event_streams'])==set(e['event_streams'])
        for event in a['event_streams']:
            for field in ('ticks','indices'):
                np.testing.assert_array_equal(a['event_streams'][event][field],e['event_streams'][event][field])


@pytest.mark.parametrize('backend',BACKENDS)
@pytest.mark.parametrize('route',['scan','sparse'])
@pytest.mark.parametrize('mutable',[False,True])
def test_named_pre_post_delay_routes_and_refractory_independence(device,tmp_path,backend,route,mutable):
    setup(tmp_path/'ref')
    net,source,target,syn,probes,states,spike=network(mutable);net.run(12*DT)
    model=json.loads((device.last_run_directory/'model.json').read_text())
    expected=load_results(model,device.last_run_directory/'rust')
    actual=execute(model,tmp_path/backend,backend,route)
    compare(actual,expected)
    if backend!='cpu-f32':compare(load_results(model,tmp_path/backend/'transport'),actual)
    assert len(probes[2].i)>len(spike.i)
    np.testing.assert_array_equal(probes[3].x[:],0)


@pytest.mark.parametrize('backend',BACKENDS)
@pytest.mark.parametrize('record',[False,True])
def test_custom_only_population_without_spike_or_synapse(device,tmp_path,backend,record):
    setup(tmp_path/'ref')
    pop=b.NeuronGroup(2,'dv/dt=512*Hz:1\nz:1',events={'crossing':'v>=1','quiet':'False'},dt=DT,method='euler')
    pop.run_on_event('crossing','v=0; z+=0.25')
    monitor=b.EventMonitor(pop,'crossing',variables=['v','z'],when='end') if record else None
    net=b.Network(pop,*([monitor] if record else []));net.run(6*DT)
    model=json.loads((device.last_run_directory/'model.json').read_text())
    expected=load_results(model,device.last_run_directory/'rust')
    actual=execute(model,tmp_path/backend,backend)
    compare(actual,expected)
    assert len(actual['populations'][0]['event_streams']['crossing']['ticks'])==6
    assert actual['populations'][0]['last_spikes'].size==0
    if backend!='cpu-f32':compare(load_results(model,tmp_path/backend/'transport'),actual)


@pytest.mark.parametrize('backend',BACKENDS[1:])
@pytest.mark.parametrize('queued',[False,True])
def test_named_event_pending_segment_restore_and_queued(device,tmp_path,backend,queued):
    outputs=[]
    for engine in ('reference',backend):
        device.reinit()
        options=dict(numeric_mode='float32',event_delivery='sparse') if engine!='reference' else {}
        setup(tmp_path/engine,engine,build_on_run=not queued,**options)
        net,source,target,syn,probes,states,spike=network(True)
        net.run(4*DT)
        if not queued:net.store('pending')
        net.run(8*DT)
        if queued:device.build()
        else:
            before=np.asarray(syn.w[:]).copy();net.restore('pending');net.run(8*DT)
            np.testing.assert_array_equal(syn.w[:],before)
        outputs.append([np.asarray(syn.w[:]).copy(),np.asarray(source.v[:]).copy(),np.asarray(target.v[:]).copy()]+
                       [np.asarray(m.t[:]/b.second).copy() for m in probes]+
                       [np.asarray(m.i[:]).copy() for m in probes]+
                       [np.asarray(getattr(m,name)[:]).copy() for m in probes for name in sorted(set(m.record_variables)-{'i','t'})])
    for a,e in zip(*outputs,strict=True):np.testing.assert_array_equal(a,e)


@pytest.mark.parametrize('backend',BACKENDS)
@pytest.mark.parametrize('route',['scan','sparse'])
def test_late_custom_threshold_does_not_restrict_early_spike_delay(device,tmp_path,backend,route):
    setup(tmp_path/'ref')
    source=b.NeuronGroup(2,'v:1',threshold='True',reset='',events={'late':'True'},dt=DT)
    source.set_event_schedule('late',when='end')
    target=b.NeuronGroup(2,'v:1',clock=source.clock)
    syn=b.Synapses(source,target,on_pre={'early':'v_post+=1','later':'v_post+=2'},
                   on_event={'early':'spike','later':'late'},clock=source.clock)
    syn.connect(i=[0,1],j=[0,1]);syn.early.delay=DT;syn.later.delay=0*DT
    monitor=b.EventMonitor(source,'late',variables='v',when='end',order=1)
    b.Network(source,target,syn,monitor).run(4*DT)
    model=json.loads((device.last_run_directory/'model.json').read_text())
    expected=load_results(model,device.last_run_directory/'rust')
    actual=execute(model,tmp_path/backend,backend,route)
    compare(actual,expected)
    np.testing.assert_array_equal(target.v[:],9)
