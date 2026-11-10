"""Versioned compact output with independent streaming canonical-byte proof."""
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

import brian2 as brian
import brian2_rust
from brian2_rust.export import lower_network
import numpy as np
import pytest

from brian2_rust.distributed import write_mpi_project, compile_mpi_project, run_mpi_project
from brian2_rust.mpi_spike_output import compact_spike_output_source
from brian2_rust.results import load_results, _strict_event_order
from brian2_rust.protocol import attach_protocol
from test_mpi import device as device, real_mpi, RUNNER, network_model
from test_mpi_prebuild import model_and_owners

spec = importlib.util.spec_from_file_location('compact_compare', Path(__file__).parents[1]/'tools/mam_compare_compact_output.py')
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


def test_ordering_u32_boundaries_and_reversed_records():
    ticks = np.array([0, 2**32-1, 2**32-1], dtype='<u4')
    indices = np.array([2**32-1, 0, 2**32-1], dtype='<u4')
    assert _strict_event_order(ticks, indices)
    assert not _strict_event_order(ticks[::-1], indices[::-1])
    assert not _strict_event_order(np.array([3, 3], dtype='<u4'), np.array([4, 4], dtype='<u4'))
    ticks = np.arange(131075, dtype='<u4')
    indices = np.zeros(len(ticks), dtype='<u4')
    assert _strict_event_order(ticks, indices)
    ticks[131073] = 0
    assert not _strict_event_order(ticks, indices)


@pytest.mark.usefixtures('device')
def test_output_gates_and_source_drift(tmp_path):
    model = network_model()
    for value in [1, 'yes']:
        with pytest.raises(TypeError, match='must be a boolean'):
            write_mpi_project(model, tmp_path/'bad-type', compact_spike_output=value)
    with pytest.raises(ValueError, match='requires compact_spike_history'):
        write_mpi_project(model, tmp_path/'bad-history', compact_spike_output=True)
    assert not (tmp_path/'bad-history').exists()
    write_mpi_project(model, tmp_path/'history', compact_spike_history=True)
    source = (tmp_path/'history/main.rs').read_text()
    compact, info = compact_spike_output_source(model, source)
    assert info['result_version'] == 4 and 'B2EVT002' in compact
    for field,value in [('count',2**32+1),('events',['custom']),('event_monitors',[{}])]:
        bad=copy.deepcopy(model);bad['definition']['populations'][0][field]=value
        with pytest.raises(ValueError):compact_spike_output_source(bad,source)
    bad=copy.deepcopy(model);bad['run']['clocks'][0]['start_tick']=2**32
    with pytest.raises(ValueError):compact_spike_output_source(bad,source)
    with pytest.raises(ValueError):compact_spike_output_source(model,source.replace('p0_spikes.len()*16','p0_spikes.len()*32'))


def equal_values(a, b):
    if isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:equal_values(a[key], b[key])
    elif isinstance(a, (tuple,list)):
        assert len(a) == len(b)
        for x,y in zip(a,b):equal_values(x,y)
    else:
        assert a == b


@real_mpi
@pytest.mark.usefixtures('device')
@pytest.mark.parametrize('mode',['ordinary','projections','populations'])
@pytest.mark.parametrize('fixture',['recurrent','prebuilt','fixed_total','fixed_monitored','no_events','windowed','silent','stream_only'])
def test_compact_output_mpi_and_independent_bytes(tmp_path, mode, fixture):
    if fixture=='fixed_monitored':
        clock=brian.Clock(dt=brian.ms)
        a=brian.NeuronGroup(5,'x:1',threshold='timestep(t,dt)%2==0',reset='x=0',clock=clock)
        z=brian.NeuronGroup(7,'x:1',threshold='timestep(t,dt)%3==0',reset='x=0',clock=clock)
        objects=[a,z]+[brian.SpikeMonitor(g) for g in (a,z)]+[brian.StateMonitor(g,'x',record=True) for g in (a,z)]
        for q,(src,tgt) in enumerate([(a,z),(z,a),(a,a)]):
            syn=brian.Synapses(src,tgt,'w:1 (constant)',on_pre='x_post+=w',clock=clock)
            brian2_rust.connect_fixed_total(syn,101+q,seed=91+q,
                initializers={'w':brian2_rust.Uniform(-1,1)},
                delay_initializer=brian2_rust.Uniform(0*clock.dt,4*clock.dt))
            objects.append(syn)
        model=lower_network(brian.Network(*objects),12*clock.dt);owners=(1,None)
    elif fixture=='no_events':
        clock=brian.Clock(dt=brian.ms);a=brian.NeuronGroup(3,'dx/dt=100*Hz:1',clock=clock)
        model=lower_network(brian.Network(a,brian.StateMonitor(a,'x',record=True)),4*clock.dt);owners=None
    elif fixture in ('prebuilt','fixed_total'):
        model, owners = model_and_owners(hybrid=fixture=='prebuilt')
    else:
        model = network_model(monitor=fixture!='stream_only');owners=None
        if fixture == 'windowed':
            for pop in model['definition']['populations']:pop['monitor']['window_steps']=5
            attach_protocol(model)
        if fixture == 'silent':
            for pop in model['instance']['populations']:
                pop['parameters']['drive']=['0000000000000000']*len(pop['parameters']['drive'])
            attach_protocol(model)
    kwargs=dict(ranks=4,population_owners=owners,compact_projections=mode=='projections',
                compact_populations=mode=='populations',prebuild_shared_topology=fixture in ('prebuilt','fixed_total','fixed_monitored'),
                compact_queue_indices=fixture in ('prebuilt','fixed_total','fixed_monitored'),compact_spike_history=True)
    reports=[]
    for name,compact in [('wide',False),('compact',True)]:
        project=tmp_path/name
        write_mpi_project(model,project,compact_spike_output=compact,**kwargs)
        compile_mpi_project(project,opt_level=1,panic_strategy='abort')
        reports.append(run_mpi_project(project,tmp_path/(name+'-out'),timeout=30))
    write_mpi_project(model,tmp_path/'default',**kwargs)
    before=json.loads((tmp_path/'wide/manifest.json').read_text());after=json.loads((tmp_path/'compact/manifest.json').read_text())
    assert json.loads((tmp_path/'default/manifest.json').read_text())==before
    assert before['plan_sha256']==after['plan_sha256']
    if fixture in ('fixed_total','fixed_monitored') and mode!='ordinary':
        assert after['projection_compaction']['compacted']
        if mode=='populations':assert after['population_compaction']['compacted']
    assert 'spike_output_compaction' not in before and after['spike_output_compaction']['bytes_per_record']==8
    for name,h in before['files'].items():
        if name!='main.rs':assert after['files'][name]==h
    protocol=tmp_path/'model.json';protocol.write_text(json.dumps(model))
    subprocess.run([str(RUNNER),str(protocol),str(tmp_path/'reference')],check=True,capture_output=True,timeout=30)
    for name in ['results.bin','events.bin']:
        if name=='events.bin' and fixture=='no_events':
            assert not (tmp_path/'wide-out'/name).exists() and not (tmp_path/'compact-out'/name).exists()
        else:assert (tmp_path/'wide-out'/name).read_bytes()==(tmp_path/'reference'/name).read_bytes()
    result=comparison.compare_pair(model,tmp_path/'reference',tmp_path/'compact-out')
    assert result['complete'] and result['canonical_v3_bytes_exact'] and not result['reconstructed_files_created']
    assert reports[0]['rank_work']==reports[1]['rank_work']
    assert reports[0]['spike_history_capacity_bytes']==reports[1]['spike_history_capacity_bytes']
    a=load_results(model,tmp_path/'wide-out',include_times=False);b=load_results(model,tmp_path/'compact-out',include_times=False)
    equal_values(a['populations'],b['populations'])
    for pop in b['populations']:assert pop['spike_ticks'].dtype==np.dtype('<u4')
    if mode=='ordinary' and fixture=='recurrent':
        c=load_results(model,tmp_path/'compact-out')
        d=load_results(model,tmp_path/'wide-out')
        equal_values(c['populations'],d['populations'])
        bad=tmp_path/'mixed';shutil.copytree(tmp_path/'compact-out',bad)
        shutil.copyfile(tmp_path/'wide-out/events.bin',bad/'events.bin')
        with pytest.raises(RuntimeError,match='marker'):load_results(model,bad,include_times=False)
        with pytest.raises(ValueError,match='event version'):comparison.compare_pair(model,tmp_path/'reference',bad)
        shutil.copyfile(tmp_path/'compact-out/events.bin',bad/'events.bin')
        with (bad/'results.bin').open('r+b') as f:f.seek(-9,2);f.write(b'\xff')
        with pytest.raises(ValueError,match='non-spike bytes'):comparison.compare_pair(model,tmp_path/'reference',bad)
