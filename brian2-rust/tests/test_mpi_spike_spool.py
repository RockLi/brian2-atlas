"""Disk-spooled MPI history must preserve complete output bytes and work."""
import json
from pathlib import Path
import subprocess
import sys
import pytest
from brian2_rust.distributed import write_mpi_project,compile_mpi_project,run_mpi_project
from brian2_rust.mpi_spike_spool import SPOOL_HELPER,spool_spike_history_source
from test_mpi import device as device,real_mpi,network_model
from test_mpi_prebuild import model_and_owners

@pytest.mark.usefixtures('device')
def test_spool_generation_gates(tmp_path):
    model=network_model()
    for v in [True,0,-1,2.5,512*2**30+1]:
        with pytest.raises(ValueError,match='integer'):
            write_mpi_project(model,tmp_path/'bad',spike_spool_bytes=v)
    with pytest.raises(ValueError,match='requires compact_spike_output'):
        write_mpi_project(model,tmp_path/'bad',spike_spool_bytes=100)
    for total,population in [(None,1),(100,True),(100,0),(100,101),(100,1.5)]:
        with pytest.raises(ValueError,match='population_bytes'):
            write_mpi_project(model,tmp_path/'bad',compact_spike_history=True,
                compact_spike_output=True,spike_spool_bytes=total,spike_spool_population_bytes=population)
    assert not (tmp_path/'bad').exists()
    write_mpi_project(model,tmp_path/'plain',compact_spike_history=True,compact_spike_output=True)
    original=(tmp_path/'plain/main.rs').read_text()
    transformed,info=spool_spike_history_source(model,original,100)
    assert info['maximum_bytes']==100 and info['maximum_spike_file_overlap_bytes']==300
    _,limited=spool_spike_history_source(model,original,100,16)
    assert limited['maximum_spike_file_overlap_bytes']==216
    assert 'MpiSpikeSpool::new' in transformed
    with pytest.raises(ValueError,match='differs'):
        spool_spike_history_source(model,original.replace('p0_spikes.push','p0_spikes.changed'),100)
    write_mpi_project(model,tmp_path/'spooled',compact_spike_history=True,compact_spike_output=True,spike_spool_bytes=100)
    assert (tmp_path/'spooled/main.rs').read_text()==transformed

@real_mpi
@pytest.mark.usefixtures('device')
@pytest.mark.parametrize('mode',['ordinary','projections','populations'])
@pytest.mark.parametrize('fixture',['recurrent','stream_only','fixed_total','silent','windowed','no_events'])
def test_spooled_mpi_exact_bytes(tmp_path,mode,fixture):
    if fixture=='fixed_total': model,owners=model_and_owners(hybrid=False)
    elif fixture=='no_events':
        import brian2 as b
        from brian2_rust.export import lower_network
        c=b.Clock(dt=b.ms);a=b.NeuronGroup(3,'dx/dt=100*Hz:1',clock=c)
        model=lower_network(b.Network(a,b.StateMonitor(a,'x',record=True)),4*c.dt);owners=None
    else:
        model=network_model(monitor=fixture!='stream_only');owners=None
        from brian2_rust.protocol import attach_protocol
        if fixture=='windowed':
            for pop in model['definition']['populations']:pop['monitor']['window_steps']=5
            attach_protocol(model)
        if fixture=='silent':
            for pop in model['instance']['populations']:
                pop['parameters']['drive']=['0000000000000000']*len(pop['parameters']['drive'])
            attach_protocol(model)
    common=dict(ranks=4,population_owners=owners,compact_spike_history=True,compact_spike_output=True,
        compact_projections=mode=='projections',compact_populations=mode=='populations',
        prebuild_shared_topology=fixture=='fixed_total',compact_queue_indices=fixture=='fixed_total')
    reports=[]
    for name,budget in [('memory',None),('spool',2**20)]:
        p=tmp_path/name;write_mpi_project(model,p,spike_spool_bytes=budget,spike_spool_population_bytes=2**18 if budget else None,**common)
        compile_mpi_project(p,opt_level=1,panic_strategy='abort')
        reports.append(run_mpi_project(p,tmp_path/(name+'-out'),timeout=30))
    for name in ['results.bin','events.bin']:
        if fixture=='no_events' and name=='events.bin':
            assert not (tmp_path/'spool-out'/name).exists();continue
        assert (tmp_path/'memory-out'/name).read_bytes()==(tmp_path/'spool-out'/name).read_bytes()
    assert reports[0]['rank_work']==reports[1]['rank_work']
    assert reports[0]['spike_history_records']==reports[1]['spike_history_records']
    assert reports[1]['spike_spool_maximum_population_bytes']==2**18
    assert reports[1]['spike_spool_bytes']==reports[1]['spike_history_records']*8
    assert reports[1]['spike_history_capacity_bytes']<=len(model['definition']['populations'])*65536
    assert not (tmp_path/'spool-out/spike-spool').exists()
    left=json.loads((tmp_path/'memory/manifest.json').read_text());right=json.loads((tmp_path/'spool/manifest.json').read_text())
    assert left['plan_sha256']==right['plan_sha256']
    for n,h in left['files'].items():
        if n!='main.rs':assert right['files'][n]==h

@real_mpi
@pytest.mark.usefixtures('device')
@pytest.mark.parametrize('population_limit',[None,8])
def test_spool_budget_abort_has_no_success(tmp_path,population_limit):
    p=tmp_path/'project';write_mpi_project(network_model(),p,ranks=2,
        compact_spike_history=True,compact_spike_output=True,spike_spool_bytes=8 if population_limit is None else 2**20,
        spike_spool_population_bytes=population_limit)
    compile_mpi_project(p,opt_level=1,panic_strategy='abort')
    with pytest.raises(RuntimeError,match='spool.*byte budget exceeded'):
        run_mpi_project(p,tmp_path/'out',timeout=30)
    assert not (tmp_path/'out/summary.json').exists()
    assert not (tmp_path/'out/results.bin').exists()

@pytest.mark.skipif(sys.platform!='linux',reason='Linux POSIX cache release')
def test_spool_core_boundaries_and_io_failures(tmp_path):
    harness=r'''
use std::io::Write;
fn main() -> std::io::Result<()> {
 let dir=std::path::PathBuf::from(std::env::args_os().nth(1).unwrap());
 assert!(MpiSpoolBudget::with_population_limit(100,101).is_err());
 assert!(MpiSpoolBudget::with_population_limit(100,0).is_err());
 let limit=MpiSpoolBudget::with_population_limit(24,8)?;
 let mut limited=MpiSpikeSpool::new(dir.join("limited"),limit.clone());
 limited.push((7,9))?;
 assert!(limited.push((8,10)).unwrap_err().to_string().contains("population byte budget"));
 assert_eq!(limit.used.get(),8);assert_eq!(limited.len(),1);
 let mut failed=MpiBoundedOutput::new(std::fs::File::create(dir.join("too-small"))?,0);
 assert!(limited.copy_final(&mut failed,&dir).is_err());assert!(dir.join("limited").exists());
 let mut failed=MpiBoundedOutput::new(std::fs::File::open(dir.join("too-small"))?,8);
 assert!(limited.copy_final(&mut failed,&dir).is_err());assert!(dir.join("limited").exists());
 let mut output=MpiBoundedOutput::new(std::fs::File::create(dir.join("durable"))?,16);
 assert!(limited.copy_final(&mut output,&dir.join("missing-dir")).is_err());
 assert!(dir.join("limited").exists());
 // A new output starts from a clean prefix after the deliberate directory failure.
 let mut output=MpiBoundedOutput::new(std::fs::File::create(dir.join("durable"))?,16);
 limited.copy_final(&mut output,&dir)?;
 assert!(!dir.join("limited").exists());assert!(limited.copy_to(&mut Vec::new()).is_err());
 assert_eq!(std::fs::read(dir.join("durable"))?,[7u32.to_le_bytes(),9u32.to_le_bytes()].concat());
 let mut later=MpiSpikeSpool::new(dir.join("later"),limit.clone());later.push((1,2))?;
 // Flush the buffered record, then truncate so copy sees the damaged source.
 later.copy_to(&mut Vec::new())?;
 std::fs::OpenOptions::new().write(true).open(dir.join("later"))?.set_len(0)?;
 assert!(later.copy_final(&mut output,&dir).is_err());assert!(dir.join("later").exists());
 assert_eq!(std::fs::read(dir.join("durable"))?,[7u32.to_le_bytes(),9u32.to_le_bytes()].concat());
 later.remove()?;
 let budget=MpiSpoolBudget::new(16)?;
 let mut a=MpiSpikeSpool::new(dir.join("a"),budget.clone());
 let mut b=MpiSpikeSpool::new(dir.join("b"),budget.clone());
 a.push((u32::MAX,0))?;b.push((0,u32::MAX))?;
 assert!(a.push((0,0)).unwrap_err().to_string().contains("budget"));
 assert_eq!(a.len(),1);assert_eq!(budget.used.get(),16);
 let mut bytes=Vec::new();a.copy_to(&mut bytes)?;a.copy_to(&mut bytes)?;
 assert_eq!(bytes,[255,255,255,255,0,0,0,0].repeat(2));
 a.remove()?;b.remove()?;assert!(!dir.join("a").exists());assert!(a.push((0,0)).is_err());
 let budget=MpiSpoolBudget::new(64)?;
 let mut bad=MpiSpikeSpool::new(dir.join("missing/child"),budget.clone());
 assert!(bad.push((1,1)).is_err());assert!(bad.copy_to(&mut Vec::new()).is_err());
 assert_eq!(bad.len(),0);assert_eq!(budget.used.get(),8);
 let mut empty=MpiSpikeSpool::new(dir.join("empty"),budget.clone());
 empty.copy_to(&mut Vec::new())?;empty.remove()?;assert!(!dir.join("empty").exists());
 let mut collision=MpiSpikeSpool::new(dir.join("existing"),budget.clone());
 std::fs::write(dir.join("existing"),b"original")?;
 assert!(collision.push((1,1)).is_err());assert_eq!(std::fs::read(dir.join("existing"))?,b"original");
 let mut many=MpiSpikeSpool::new(dir.join("many"),MpiSpoolBudget::new(8*300001)?);
 many.extend((0..300001).map(|n|(n,n^0xabcdef)))?;
 let mut bytes=Vec::new();many.copy_to(&mut bytes)?;
 assert_eq!(bytes.len(),8*300001);assert_eq!(many.capacity(),8192);
 for (n,pair) in bytes.chunks_exact(8).enumerate() {
   assert_eq!(&pair[..4],&(n as u32).to_le_bytes());
   assert_eq!(&pair[4..],&((n as u32)^0xabcdef).to_le_bytes());
 }
 std::fs::OpenOptions::new().write(true).open(dir.join("many"))?.set_len(8)?;
 assert!(many.copy_to(&mut Vec::new()).unwrap_err().to_string().contains("size mismatch"));
 many.remove()?;
 let mut out=MpiBoundedOutput::new(std::fs::File::create(dir.join("output"))?,70*1024*1024);
 for _ in 0..70 { out.write_all(&vec![37u8;1024*1024])?; }
 assert!(out.write_all(&[1]).unwrap_err().to_string().contains("budget"));
 out.flush()?;drop(out);
 let mut file=std::fs::File::open(dir.join("output"))?;
 assert_eq!(file.metadata()?.len(),70*1024*1024);
 use std::io::Read;
 let mut block=[0u8;65536];
 loop { let n=file.read(&mut block)?;if n==0 {break;}assert!(block[..n].iter().all(|&b|b==37)); }
 Ok(())
}
'''
    p=tmp_path/'core.rs';p.write_text(SPOOL_HELPER+harness)
    subprocess.run(['rustc','--edition=2021','-O',str(p),'-o',str(tmp_path/'core')],check=True,capture_output=True)
    subprocess.run([str(tmp_path/'core'),str(tmp_path)],check=True,capture_output=True,timeout=30)
