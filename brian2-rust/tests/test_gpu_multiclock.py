"""Coupled clocks: native schedule ordering, delay rings and continuation."""
from dataclasses import replace
import json
import subprocess

import brian2 as b
import numpy as np
import pytest

from brian2_rust.cuda import build_cuda_plan
from brian2_rust.gpu_schedule import dispatch_ticks
from brian2_rust.plan import ClockActivation, PlanValidationError, verify_execution_plan
from brian2_rust.results import load_results
from brian2_rust.spec import bits
from brian2_rust.metal import number
from brian2_rust.protocol import attach_protocol
from test_gpu_spike_generator import BACKENDS, execute
from test_metal_delays import device, ROOT
from test_metal_plasticity import equivalent

DT = b.second/1024


def test_dispatch_clock_merge_uses_global_stage_order_and_f64_ties():
    clocks = (ClockActivation(0,bits(.1),0,4), ClockActivation(1,bits(.15),0,3))
    assert list(dispatch_ticks(clocks,(1,0,1))) == [
        (0,0),(1,0),(2,0), (1,1), (0,1),(2,1), (1,2), (0,2),(1,3),(2,2)]
    clocks = (ClockActivation(0,bits(1/1024),2**25+1,3),
              ClockActivation(1,bits(2/1024),2**24+1,1),
              ClockActivation(2,bits(1/4096),0,0))
    assert list(dispatch_ticks(clocks,(0,1))) == [(0,2**25+1),(0,2**25+2),(1,2**24+1),(0,2**25+3)]


def network(mutable):
    pre=b.NeuronGroup(3,'v:1\nu:1',threshold='v>=1',reset='v-=1',dt=DT,name='pre')
    pre.v=[.5,1,0]; pre.run_regularly('v+=0.5')
    post=b.NeuronGroup(4,'v:1\nu:1',threshold='v>2',reset='v-=2',dt=1.5*DT,name='post')
    post.run_regularly('v+=0.25')
    if mutable:
        syn=b.Synapses(pre,post,'w:1\ndx/dt=1/(16*second):1 (clock-driven)\nu_post=w:1 (summed)\nu_pre=x:1 (summed)',
            on_pre='w+=0.125; v_post+=w',on_post='w-=0.0625',
            clock=pre.clock,method='euler',name='projection')
    else:
        syn=b.Synapses(pre,post,'w:1 (constant)',on_pre='v_post+=w',clock=pre.clock,name='projection')
    syn.connect(i=[2,0,0,1,2],j=[1,1,1,2,0]);syn.w=[.25,.5,.125,.25,.5]
    syn.pre.delay=[0,1,3,2,1]*DT
    if mutable:syn.post.delay=np.array([0,1,2,1,3])*1.5*DT
    coarse=b.NeuronGroup(2,'v:1\nu:1',dt=2*DT,name='coarse')
    relay=b.Synapses(post,coarse,on_pre='v_post+=0.125',clock=post.clock,name='relay')
    relay.connect(i=[0,1,2,3],j=[0,1,0,1]);relay.delay=1.5*DT
    monitors=[b.StateMonitor(p,['v','u'],record=True) for p in (pre,post,coarse)]
    spikes=[b.SpikeMonitor(p) for p in (pre,post)]
    return b.Network(pre,post,coarse,syn,relay,*monitors,*spikes),pre,post,syn,monitors,spikes


@pytest.mark.parametrize('backend',BACKENDS)
@pytest.mark.parametrize('route',['scan','sparse'])
@pytest.mark.parametrize('mutable',[False,True])
def test_coupled_clocks_delays_state_post_and_summed(device,tmp_path,backend,route,mutable):
    b.set_device('rust_standalone',engine='reference',directory=tmp_path/'ref',runner=ROOT/'target/release/b2-runner')
    net,*_=network(mutable);net.run(12*DT)
    model=json.loads((device.last_run_directory/'model.json').read_text())
    expected=load_results(model,device.last_run_directory/'rust')
    plan=build_cuda_plan(model,numeric_mode='float32',event_delivery=route)
    assert len(plan.logical.clocks)==3
    assert len({s.clock for s in plan.dispatches}) >= 2
    node_clocks={node.id:node.clock for node in plan.logical.nodes}
    for kernel,dispatch in zip(plan.kernels,plan.dispatches,strict=True):
        assert all(node_clocks[n]==dispatch.clock for n in kernel.nodes)
    bad=replace(plan,dispatches=(replace(plan.dispatches[0],clock=(plan.dispatches[0].clock+1)%3),*plan.dispatches[1:]))
    with pytest.raises(PlanValidationError):verify_execution_plan(bad,model)
    actual=execute(model,tmp_path/backend,backend,route)
    equivalent(actual,expected,exact=True)
    for a,e in zip(actual['populations'],expected['populations'],strict=True):
        for event in a['event_streams']:
            for field in ('ticks','indices'):
                np.testing.assert_array_equal(a['event_streams'][event][field],e['event_streams'][event][field])


@pytest.mark.parametrize('backend',BACKENDS[1:])
@pytest.mark.parametrize('queued',[False,True])
def test_multiclock_pending_segment_restore_and_queued(device,tmp_path,backend,queued):
    results=[]
    for engine in ('reference',backend):
        device.reinit()
        options=dict(numeric_mode='float32',event_delivery='sparse') if engine!='reference' else {}
        b.set_device('rust_standalone',engine=engine,directory=tmp_path/engine,
                     runner=ROOT/'target/release/b2-runner',build_on_run=not queued,**options)
        net,pre,post,syn,monitors,spikes=network(True)
        net.run(6*DT)
        if not queued:net.store('pending')
        net.run(6*DT)
        if queued:device.build()
        else:
            before=np.asarray(syn.w[:]).copy()
            net.restore('pending');net.run(6*DT)
            np.testing.assert_array_equal(syn.w[:],before)
        results.append([np.asarray(p.v[:]).copy() for p in (pre,post)]+
                       [np.asarray(syn.w[:]).copy(),np.asarray(syn.x[:]).copy()]+
                       [np.asarray(m.v[:]).copy() for m in monitors]+
                       [np.asarray(s.t[:]/b.second).copy() for s in spikes]+
                       [np.asarray(s.i[:]).copy() for s in spikes])
    for actual,expected in zip(results[1],results[0],strict=True):
        np.testing.assert_array_equal(actual,expected)


@pytest.mark.parametrize('backend',BACKENDS)
def test_generator_coupled_large_absolute_ticks_and_pending(device,tmp_path,backend):
    b.set_device('rust_standalone',engine='reference',directory=tmp_path/'ref',runner=ROOT/'target/release/b2-runner')
    source=b.SpikeGeneratorGroup(3,[0,1,2],[0,1,3]*DT,period=4*DT,dt=DT)
    target=b.NeuronGroup(2,'v:1',dt=1.5*DT)
    target.run_regularly('v+=0.125')
    syn=b.Synapses(source,target,on_pre='v_post+=0.25',clock=source.clock)
    syn.connect(i=[0,1,2,0],j=[0,1,0,1]);syn.delay=[1,2,3,4]*DT
    monitors=[b.StateMonitor(target,'v',record=True),b.SpikeMonitor(source)]
    net=b.Network(source,target,syn,*monitors);net.run(6*DT);net.run(6*DT)
    model=json.loads((device.last_run_directory/'model.json').read_text())
    shift=3*2**25
    model['run']['start']=bits((6+shift)/1024)
    for definition,activation in zip(model['definition']['clocks'],model['run']['clocks'],strict=True):
        activation['start_tick']+=int(shift/(1024*number(definition['dt'])))
    for pop in model['instance']['populations']:
        if pop.get('spike_generator'):
            pop['spike_generator']['spike_ticks']=[t+shift for t in pop['spike_generator']['spike_ticks']]
    for path in model['instance']['synapses'][0]['pathways']:
        assert path['pending']
        for event in path['pending']:event['delivery_tick']+=shift
    attach_protocol(model)
    file=tmp_path/'shifted.json';file.write_text(json.dumps(model))
    subprocess.run([str(ROOT/'target/release/b2-runner'),str(file),str(tmp_path/'oracle')],check=True,capture_output=True)
    expected=load_results(model,tmp_path/'oracle')
    equivalent(execute(model,tmp_path/backend,backend,'sparse'),expected,exact=True)
