"""Exact public recordings, checked bounds and retained allocation savings."""
import copy
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from brian2_rust.distributed import write_mpi_project, compile_mpi_project, run_mpi_project
from brian2_rust.mpi_spike_history import compact_spike_history_source, RECORD_HELPER
from brian2_rust.protocol import attach_protocol
from test_mpi import device as device, real_mpi, RUNNER, network_model
from test_mpi_prebuild import model_and_owners


def test_record_bounds_and_serialization(tmp_path):
    rustc = shutil.which('rustc')
    if not rustc:
        pytest.skip('rustc required')
    # Boundary values serialize identically without allocating billions of cells.
    code = RECORD_HELPER + r'''
fn main() {
    let mut wide=Vec::<(usize,usize)>::new();
    let mut duplicate=Vec::<(usize,usize)>::new();
    let mut narrow=Vec::<(u32,u32)>::new();
    for i in 0..4097usize {
        let item=(i*17,i%71);
        wide.push(item);duplicate.push(item);narrow.push(mpi_spike_record(item.0,item.1));
        assert_eq!((wide.capacity()+duplicate.capacity())*16,narrow.capacity()*8*4);
    }
    for &(t,n) in &[(0,0),(u32::MAX as usize,0),(0,u32::MAX as usize)] {
        let (a,b)=mpi_spike_record(t,n);
        assert_eq!((t as i64).to_le_bytes(),(a as i64).to_le_bytes());
        assert_eq!((n as i64).to_le_bytes(),(b as i64).to_le_bytes());
    }
    for (a,b) in wide.iter().zip(narrow.iter()) {
        assert_eq!(a.0 as u64,b.0 as u64);assert_eq!(a.1 as u64,b.1 as u64);
    }
    if usize::BITS>32 {
        assert!(std::panic::catch_unwind(||mpi_spike_record(u32::MAX as usize+1,0)).is_err());
        assert!(std::panic::catch_unwind(||mpi_spike_record(0,u32::MAX as usize+1)).is_err());
    }
}
'''
    p=tmp_path/'test.rs';p.write_text(code)
    subprocess.run([rustc,'--edition=2021','-C','opt-level=1',str(p),'-o',str(tmp_path/'test')],check=True,capture_output=True,timeout=30)
    subprocess.run([str(tmp_path/'test')],check=True,capture_output=True,timeout=30)


@pytest.mark.usefixtures('device')
def test_history_gates_and_emitter_drift(tmp_path):
    model=network_model()
    write_mpi_project(model,tmp_path/'wide',ranks=2)
    source=(tmp_path/'wide/main.rs').read_text()
    candidate,info=compact_spike_history_source(model,source)
    assert info['retained_copies']==1 and 'event_history_0' not in candidate
    for key,value,pattern in [('events',['spike','custom'],'ordinary spike'),
                              ('event_monitors',[{}],'EventMonitor'),
                              ('count',2**32+1,'neuron exceeds')]:
        bad=copy.deepcopy(model);bad['definition']['populations'][0][key]=value
        with pytest.raises(ValueError,match=pattern):compact_spike_history_source(bad,source)
    bad=copy.deepcopy(model);bad['run']['clocks'][0]['start_tick']=2**32
    with pytest.raises(ValueError,match='tick exceeds'):compact_spike_history_source(bad,source)
    with pytest.raises(ValueError,match='source differs'):
        compact_spike_history_source(model,source.replace('p0_spikes.push','changed.push'))
    with pytest.raises(TypeError,match='must be a boolean'):
        write_mpi_project(model,tmp_path/'invalid',compact_spike_history=1)
    assert not (tmp_path/'invalid').exists()


@real_mpi
@pytest.mark.usefixtures('device')
@pytest.mark.parametrize('mode',['ordinary','projections','populations'])
@pytest.mark.parametrize('fixture',['recurrent','prebuilt','windowed','silent'])
def test_history_exact_mpi_and_independent_outputs(tmp_path,mode,fixture,opt_level=1):
    if fixture=='prebuilt':
        model,owners=model_and_owners(hybrid=True)
    else:
        model=network_model();owners=None
        if fixture=='windowed':
            for pop in model['definition']['populations']:
                pop['monitor']['window_steps'] = 5
            attach_protocol(model)
        if fixture=='silent':
            for pop in model['instance']['populations']:
                pop['parameters']['drive']=['0000000000000000']*len(pop['parameters']['drive'])
            attach_protocol(model)
    reports=[]
    for label,enabled in [('wide',False),('shared32',True)]:
        project=tmp_path/label
        write_mpi_project(model,project,ranks=4,population_owners=owners,
            compact_projections=mode=='projections',compact_populations=mode=='populations',
            prebuild_shared_topology=fixture=='prebuilt',compact_queue_indices=fixture=='prebuilt',
            compact_spike_history=enabled)
        compile_mpi_project(project,opt_level=opt_level,panic_strategy='abort')
        reports.append(run_mpi_project(project,tmp_path/(label+'-out'),timeout=30))
    before=json.loads((tmp_path/'wide/manifest.json').read_text())
    after=json.loads((tmp_path/'shared32/manifest.json').read_text())
    assert before['plan_sha256']==after['plan_sha256']
    for n,h in before['files'].items():
        if n!='main.rs':assert after['files'][n]==h
    assert 'spike_history_compaction' not in before and after['spike_history_compaction']['retained_copies']==1
    path=tmp_path/'model.json';path.write_text(json.dumps(model))
    subprocess.run([str(RUNNER),str(path),str(tmp_path/'reference')],check=True,capture_output=True,timeout=30)
    for name in ['results.bin','events.bin']:
        assert (tmp_path/'wide-out'/name).read_bytes()==(tmp_path/'shared32-out'/name).read_bytes()==(tmp_path/'reference'/name).read_bytes()
    assert reports[0]['rank_work']==reports[1]['rank_work']
    report=reports[1];assert report['spike_history_copies']==1 and report['spike_history_index_bits']==32
    assert (report['spike_history_records']==0) == (fixture=='silent')
    assert report['spike_history_capacity_bytes']>=8*report['spike_history_records']


@real_mpi
@pytest.mark.usefixtures('device')
@pytest.mark.parametrize('fixture',['recurrent','prebuilt'])
def test_o3_history_exact_outputs(tmp_path,fixture):
    test_history_exact_mpi_and_independent_outputs(tmp_path,'populations',fixture,opt_level=3)
