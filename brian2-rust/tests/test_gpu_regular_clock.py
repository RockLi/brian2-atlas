"""One population can run code objects on independent validated clocks."""
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import brian2 as b
import pytest
from brian2_rust.export import lower_network
from brian2_rust.metal import build_metal_plan
from brian2_rust.cuda import build_cuda_plan
from brian2_rust.metal_dag import run_dag
from test_metal_delays import device,ROOT
from test_gpu_spike_generator import BACKENDS,execute
from test_gpu_refractory import reference,compare
from test_cuda_graphs import result_exact
from test_gpu_workgroup import save
DT=b.second/1024


def setup(path,engine='reference',**kwargs):
    b.set_device('rust_standalone',engine=engine,directory=path,runner=ROOT/'target/release/b2-runner',
        **(dict(numeric_mode='float32',event_delivery='sparse') if engine in {'metal','cuda'} else {}),**kwargs)


def network(ratio=2,coupled=False,stochastic=False):
    pop=b.NeuronGroup(257,'dv/dt=512/second:1 (unless refractory)\nx:1\nage:1\nseen:second\nstep:second\ntau:second',
        threshold='v>=1',reset='v=0',refractory='tau',method='euler',dt=DT,name='population')
    pop.v=(np.arange(257)%4)/4;pop.tau=(1+np.arange(257)%3)*DT;pop.lastspike=-DT
    table=b.TimedArray(np.arange(8)/16,dt=2*DT)
    regular=pop.run_regularly('x+=v; age=timestep(t-lastspike,dt); seen=t; step=dt'+('; x+=table(t)+rand()/16' if stochastic else ''),
        dt=ratio*DT,when='after_synapses' if coupled else 'before_thresholds',name='regular')
    if stochastic:pop.namespace['table']=table
    mon=b.StateMonitor(pop,['v','x','age','seen','step'],record=True);spikes=b.SpikeMonitor(pop)
    objects=[pop,mon,spikes];syn=None
    if coupled:
        syn=b.Synapses(pop,pop,'w:1',on_pre='v_post+=w; w+=1.0/256',clock=pop.clock,name='projection')
        edges=np.arange(771);syn.connect(i=edges%257,j=(edges*7)%257);syn.w=1/64;syn.delay=(edges%3)*DT
        objects.append(syn)
    return b.Network(*objects),pop,mon,spikes,syn


def test_independent_regular_clock_has_explicit_nodes_and_population_storage(device,tmp_path):
    setup(tmp_path/'ref');net,*_=network();model=lower_network(net,12*DT)
    for build in (build_metal_plan,build_cuda_plan):
        plan=build(model,numeric_mode='float32')
        assert plan.dispatches and len(plan.logical.clocks)==2
        nodes={n.id:n for n in plan.logical.nodes}
        assert set(n for k in plan.kernels for n in k.nodes)|set(plan.elided_nodes)==set(nodes)
        for kernel,dispatch in zip(plan.kernels,plan.dispatches):
            assert all(nodes[n].clock==dispatch.clock for n in kernel.nodes)
            assert kernel.start_tick==plan.logical.clocks[dispatch.clock].start_tick
            assert kernel.steps==plan.logical.clocks[dispatch.clock].steps
        assert any(d.clock!=model['definition']['populations'][0]['clock'] for d in plan.dispatches)
    (tmp_path/'model.json').write_text(json.dumps(model,indent=2)+'\n')


@pytest.mark.parametrize('backend',BACKENDS)
@pytest.mark.parametrize('ratio',[.5,1.5,2])
@pytest.mark.parametrize('coupled',[False,True])
def test_regular_own_dt_time_refractory_and_full_results(device,tmp_path,backend,ratio,coupled):
    setup(tmp_path/'ref');net,*_=network(ratio,coupled);model=lower_network(net,12*DT)
    expected=reference(model,tmp_path)
    actual=execute(model,tmp_path/backend,backend,'sparse');compare(actual,expected)
    save(tmp_path/'regular-clock-results.npz',actual,expected)
    (tmp_path/'regular-clock-case.json').write_text(json.dumps(dict(ratio=ratio,coupled=coupled,backend=backend))+'\n')


@pytest.mark.parametrize('backend',BACKENDS[1:])
def test_foreign_clock_rng_and_timed_array_match_cpu_profile(device,tmp_path,backend):
    setup(tmp_path/'ref');b.seed(42);net,*_=network(.5,True,True);model=lower_network(net,12*DT)
    folder=tmp_path/'control';folder.mkdir()
    owner=SimpleNamespace(model=model,plan=build_metal_plan(model,numeric_mode='float32',event_delivery='sparse'),directory=folder,compile_seconds=0,device_name='CPU f32')
    expected=run_dag(owner,max_buffer_bytes=512*1024**2,compute='cpu-f32',workers=3)
    actual=execute(model,tmp_path/backend,backend,'sparse');result_exact(actual,expected)
    save(tmp_path/'regular-clock-results.npz',actual,expected)


@pytest.mark.parametrize('backend',BACKENDS[1:])
@pytest.mark.parametrize('queued',[False,True])
def test_foreign_clock_device_continuation_restore_and_queued(device,tmp_path,backend,queued):
    results=[]
    for engine in ('reference',backend):
        device.reinit();setup(tmp_path/engine,engine,build_on_run=not queued)
        net,pop,mon,spikes,syn=network(1.5,True)
        net.run(6*DT)
        if not queued:net.store('middle')
        net.run(6*DT)
        if queued:device.build()
        else:
            state=np.asarray(pop.x[:]).copy();net.restore('middle');net.run(6*DT)
            np.testing.assert_array_equal(pop.x[:],state)
        results.append([np.asarray(v).copy() for v in (pop.v[:],pop.x[:],pop.age[:],pop.seen[:],pop.step[:],
            pop.lastspike[:],pop.not_refractory[:],mon.v,mon.x,mon.age,mon.seen,mon.step,spikes.i[:],spikes.t[:],syn.w[:])])
    for a,e in zip(results[1],results[0]):np.testing.assert_array_equal(a,e)
    np.savez_compressed(tmp_path/'regular-clock-device.npz',**{f'{label}/{i}':a for label,arrays in zip(('reference','actual'),results) for i,a in enumerate(arrays)})
