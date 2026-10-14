"""Delayed consumers before endpoint thresholds, including run-boundary sampling."""
import json

import brian2 as b
import numpy as np
import pytest

from brian2_rust.results import load_results
from brian2_rust.schedule import pathway_sample_lags
from brian2_rust.capabilities import CapabilityError
from test_gpu_spike_generator import BACKENDS,execute
from test_gpu_monitors import event_equivalent,DT
from test_metal_delays import device,ROOT


def network(event='spike',mutable=False):
    pre=b.NeuronGroup(5,'v:1',threshold='v==1',reset='',events={'pulse':'v==2'},dt=DT,name='pre')
    pre.run_regularly('v=int(t/dt)%3')
    post=b.NeuronGroup(5,'v:1',threshold='v>=1',reset='v-=1',dt=2*DT,name='post')
    post.run_regularly('v+=0.25')
    syn=b.Synapses(pre[1:4],post[1:],'w:1'+('' if mutable else ' (constant)'),
                   on_pre=('w+=0.125; ' if mutable else '')+'v_post+=w',
                   on_post='w-=0.0625' if mutable else None,
                   on_event={'pre':event,'post':'spike'} if mutable else event,
                   clock=pre.clock,name='projection')
    syn.connect(i=[2,0,1,0],j=[0,1,2,1]);syn.w=[.125,.25,.5,.125]
    syn.pre.delay=[0,1,3,2]*DT
    if mutable:
        syn.post.delay=np.asarray([0,2,1,3])*2*DT
    monitor=b.EventMonitor(pre,event,variables='v',when='end')
    net=b.Network(pre,post,syn,monitor)
    net.schedule=['start','groups','synapses','thresholds','resets','end']
    return net,post,syn,monitor


def setup(device,path,engine='reference',queued=False,window=None):
    device.reinit()
    opts=dict(numeric_mode='float32',event_delivery='sparse') if engine in {'metal','cuda'} else {}
    b.set_device('rust_standalone',engine=engine,directory=path,runner=ROOT/'target/release/b2-runner',
                 build_on_run=not queued,recording_window_steps=window,**opts)


@pytest.mark.parametrize('backend',BACKENDS)
@pytest.mark.parametrize('route',['scan','sparse'])
@pytest.mark.parametrize('mutable',[False,True])
@pytest.mark.parametrize('event',['spike','pulse'])
def test_early_delayed_pathways_match_reference(device,tmp_path,backend,route,mutable,event):
    setup(device,tmp_path/'ref');net,*_=network(event,mutable);net.run(12*DT)
    model=json.loads((device.last_run_directory/'model.json').read_text())
    expected=load_results(model,device.last_run_directory/'rust')
    assert set(pathway_sample_lags(model['definition']).values())=={1}
    actual=execute(model,tmp_path/backend,backend,route)
    event_equivalent(actual,expected)
    if not mutable:
        # Independent arrival-time oracle: the producer's tick is sampled one
        # tick later, then delayed. Target regular update/threshold use 2*DT.
        state=np.zeros(5)
        for tick in range(12):
            if tick%2==0:state+=.25
            for target,delay,weight in zip([1,2,3,2],[0,1,3,2],[.125,.25,.5,.125]):
                emission=tick-delay-1
                if emission>=0 and emission%3==(1 if event=='spike' else 2):state[target]+=weight
            if tick%2==0:state[state>=1]-=1
        post_index=next(i for i,pop in enumerate(model['definition']['populations']) if pop['name']=='post')
        np.testing.assert_array_equal(actual['populations'][post_index]['states']['v'],state)


@pytest.mark.parametrize('backend',['reference',*BACKENDS[1:]])
@pytest.mark.parametrize('queued',[False,True])
@pytest.mark.parametrize('event',['spike','pulse'])
def test_early_pending_continuation_restore_and_queued(device,tmp_path,backend,queued,event):
    values=[]
    for split in (False,True):
        setup(device,tmp_path/str(split),backend,queued)
        net,post,syn,monitor=network(event,True)
        if split:
            net.run(6*DT)
            if not queued:net.store('pending')
            net.run(6*DT)
            if not queued:
                expected=np.asarray(syn.w[:]).copy();net.restore('pending');net.run(6*DT)
                np.testing.assert_array_equal(syn.w[:],expected)
        else:net.run(12*DT)
        if queued:device.build()
        values.append([np.asarray(post.v[:]).copy(),np.asarray(syn.w[:]).copy(),
                       np.asarray(monitor.i[:]).copy(),np.asarray(monitor.t[:]/b.second).copy()])
    for a,e in zip(*values,strict=True):np.testing.assert_array_equal(a,e)


@pytest.mark.parametrize('backend',['reference',*BACKENDS[1:]])
def test_early_pathway_window_includes_sample_lag(device,tmp_path,backend):
    results=[]
    for window in (None,4):
        setup(device,tmp_path/str(window),backend,window=window)
        net,post,syn,_=network('pulse',True)
        net.run(12*DT);net.run(12*DT)
        results.append([np.asarray(post.v[:]).copy(),np.asarray(syn.w[:]).copy()])
    for a,e in zip(*results,strict=True):np.testing.assert_array_equal(a,e)
    setup(device,tmp_path/'short',backend,window=3)
    net,*_=network('pulse',True)
    with pytest.raises((CapabilityError,ValueError),match='maximum delay plus pathway sample lag'):
        net.run(12*DT)


@pytest.mark.parametrize('backend',['reference',*BACKENDS[1:]])
@pytest.mark.parametrize('delay',[0,3])
def test_uniform_pending_source_items_including_zero_delay(device,tmp_path,backend,delay):
    states=[]
    for split in (False,True):
        setup(device,tmp_path/str(split),backend)
        net,post,syn,_=network('pulse')
        syn.pre.delay=delay*DT
        if split:net.run(6*DT);net.run(6*DT)
        else:net.run(12*DT)
        states.append(np.asarray(post.v[:]).copy())
    np.testing.assert_array_equal(*states)


@pytest.mark.parametrize('backend',['reference','aot',*BACKENDS[1:]])
def test_generator_first_boundary_event_deferred_once(device,tmp_path,backend):
    for split in (False,True):
        setup(device,tmp_path/str(split),backend)
        source=b.SpikeGeneratorGroup(2,[0,1],[0,2]*DT,period=4*DT,dt=DT)
        target=b.NeuronGroup(2,'v:1',clock=source.clock)
        syn=b.Synapses(source,target,on_pre='v_post+=0.25',clock=source.clock)
        syn.connect(i=[0,1],j=[0,1]);syn.delay=[0,3]*DT
        net=b.Network(source,target,syn)
        net.schedule=['start','groups','synapses','thresholds','resets','end']
        if split:net.run(DT);net.run(7*DT)
        else:net.run(8*DT)
        np.testing.assert_array_equal(target.v[:],[.5,.25])
