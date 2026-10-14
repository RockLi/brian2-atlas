"""Narrow FIFO entries preserve bits, domains, immutable inputs and reporting."""
import copy
import json
from pathlib import Path
import shutil
import subprocess

import brian2 as b
import pytest

from brian2_rust.distributed import write_mpi_project, compile_mpi_project, run_mpi_project
from brian2_rust.mpi_queue_compact import compact_additive_kernel, compact_queue_source
from test_mpi import device as device, real_mpi, RUNNER
from brian2_rust.export import lower_network
from brian2_rust.protocol import attach_protocol
from test_mpi_prebuild import model_and_owners

ROOT = Path(__file__).resolve().parents[1]


@real_mpi
@pytest.mark.usefixtures('device')
@pytest.mark.parametrize('uniform', [False, True])
def test_narrow_restored_pending_matches_reference(tmp_path, uniform):
    clock = b.Clock(dt=b.ms)
    source = b.NeuronGroup(5, 'x:1', threshold='True', reset='x=0', clock=clock)
    target = b.NeuronGroup(7, 'x:1', clock=clock)
    synapses = b.Synapses(source, target, 'w:1 (constant)', on_pre='x_post+=w', clock=clock)
    synapses.connect(i=[4,0,2,0], j=[1,6,3,6])
    synapses.w = [0.5,-0.25,0.125,0.75]
    synapses.delay = (3 if uniform else [1,3,2,4])*clock.dt
    model = lower_network(b.Network(source,target,synapses), 17*clock.dt)
    for syn in model['instance']['synapses']:
        path = syn['pathways'][0]
        # AtlasIR admits only arrivals in this activation; Device keeps later
        # events in its continuation state for future segments.
        path['pending'] = [dict(delivery_tick=2, item=0), dict(delivery_tick=6, item=1)]
    attach_protocol(model)
    write_mpi_project(model, tmp_path/'mpi', ranks=2, compact_queue_indices=True)
    compile_mpi_project(tmp_path/'mpi', opt_level=0)
    run_mpi_project(tmp_path/'mpi', tmp_path/'observed')
    (tmp_path/'model.json').write_text(json.dumps(model))
    subprocess.run([str(RUNNER),str(tmp_path/'model.json'),str(tmp_path/'reference')],
                   check=True,capture_output=True,timeout=30)
    for name in ('results.bin','events.bin'):
        assert (tmp_path/'observed'/name).read_bytes()==(tmp_path/'reference'/name).read_bytes()


@pytest.mark.parametrize('checked', [False, True])
def test_narrow_kernel_matches_wide_fifo_and_state_bits(tmp_path, checked):
    rustc = shutil.which('rustc')
    if not rustc:
        pytest.skip('rustc required')
    kernel = (ROOT/'python/brian2_rust/mpi_runtime/additive.rs').read_text()
    (tmp_path/'baseline.rs').write_text(kernel)
    (tmp_path/'candidate.rs').write_text(compact_additive_kernel(kernel))
    harness = (ROOT/'mpi-evidence/event-delivery-kernel/harness.rs').read_text()
    # Conversion is an oracle adapter, not part of production or timing tests.
    wrapper = '''mod candidate {
        include!("candidate.rs");
        pub fn apply(c: &mut super::Case, tick: usize, end: usize, uniform: Option<usize>, scalar: bool) -> usize {
            let delays = match uniform { Some(d)=>MpiDelays::Uniform(d), None=>MpiDelays::Edges(&c.delays) };
            let weights = if scalar { MpiWeights::Scalar(-0.125) } else { MpiWeights::Edges(&c.weights) };
            let mut queue: Vec<Vec<u32>> = c.queue.iter().map(|q|q.iter().map(|&v|u32::try_from(v).unwrap()).collect()).collect();
            let n = mpi_additive_projection(&c.fired,c.source_start,c.offsets.len()-1,tick,end,
                &c.offsets,&c.targets,delays,weights,&mut queue,&mut c.state,c.target_start,c.owned_start);
            c.queue = queue.iter().map(|q|q.iter().map(|&v|v as usize).collect()).collect(); n
        }
        pub fn bounds() {
            mpi_queue_index_bounds(0,0);
            mpi_queue_index_bounds(u32::MAX as usize,u32::MAX as usize);
            if usize::BITS > 32 {
                let size = u32::MAX as usize + 1;
                mpi_queue_index_bounds(size,size);
                assert!(std::panic::catch_unwind(||mpi_queue_index_bounds(size+1,0)).is_err());
                assert!(std::panic::catch_unwind(||mpi_queue_index_bounds(0,size+1)).is_err());
            }
        }
    }'''
    harness = harness.replace('kernel!(candidate, "candidate.rs");', wrapper)
    harness = harness.replace('    validate();', '    validate(); candidate::bounds();')
    (tmp_path/'harness.rs').write_text(harness)
    command = [rustc,'--edition=2021','-C','opt-level=1','-C',
        'overflow-checks='+('yes' if checked else 'no'), str(tmp_path/'harness.rs'),'-o',str(tmp_path/'check')]
    if checked:
        command += ['--cfg','checked_overflow']
    subprocess.run(command,check=True,capture_output=True,timeout=30)
    report = subprocess.run([str(tmp_path/'check')],check=True,capture_output=True,text=True,timeout=30)
    assert json.loads(report.stdout) == {'validation_calls':29484,'overflow_checks':checked}


@pytest.mark.usefixtures('device')
def test_narrow_rejects_unsupported_domains_pathways_and_source_drift(tmp_path):
    model,owners = model_and_owners()
    write_mpi_project(model,tmp_path/'wide',ranks=4,population_owners=owners)
    source = (tmp_path/'wide/main.rs').read_text()
    bad = copy.deepcopy(model);bad['definition']['synapses'][0]['source_count'] = 2**32+1
    with pytest.raises(ValueError,match='source index exceeds u32'):
        compact_queue_source(bad,source)
    bad = copy.deepcopy(model)
    bad['definition']['synapses'][0]['code_objects'][0]['scalar'] = [{'unsupported':True}]
    with pytest.raises(ValueError,match='canonical additive'):
        compact_queue_source(bad,source)
    with pytest.raises(ValueError,match='generated source differs'):
        compact_queue_source(model,source.replace('queue_capacity_items.checked_mul','changed.checked_mul'))
    with pytest.raises(TypeError,match='must be a boolean'):
        write_mpi_project(model,tmp_path/'invalid',compact_queue_indices=1)
    assert not (tmp_path/'invalid').exists()


@real_mpi
@pytest.mark.usefixtures('device')
@pytest.mark.parametrize('mode',['ordinary','projections','populations'])
@pytest.mark.parametrize('hybrid',[False,True])
def test_narrow_mpi_prebuild_exact_inputs_results_and_capacity(tmp_path,mode,hybrid):
    model,owners = model_and_owners(hybrid=hybrid)
    reports = []
    for name,narrow in [('wide',False),('narrow',True)]:
        project = tmp_path/name
        write_mpi_project(model,project,ranks=4,population_owners=owners,
            compact_projections=mode=='projections',compact_populations=mode=='populations',
            prebuild_shared_topology=True,compact_queue_indices=narrow)
        compile_mpi_project(project,opt_level=1,panic_strategy='abort')
        reports.append(run_mpi_project(project,tmp_path/(name+'-out'),timeout=30))
    old = json.loads((tmp_path/'wide/manifest.json').read_text())
    new = json.loads((tmp_path/'narrow/manifest.json').read_text())
    assert old['plan_sha256'] == new['plan_sha256']
    assert 'queue_compaction' not in old and new['queue_compaction']['index_bits']==32
    for name,digest in old['files'].items():
        if name != 'main.rs':
            assert new['files'][name]==digest
    (tmp_path/'model.json').write_text(json.dumps(model))
    subprocess.run([str(RUNNER),str(tmp_path/'model.json'),str(tmp_path/'reference')],
                   check=True,capture_output=True,timeout=30)
    for name in ['results.bin','events.bin']:
        assert (tmp_path/'wide-out'/name).read_bytes() == (tmp_path/'narrow-out'/name).read_bytes() == (tmp_path/'reference'/name).read_bytes()
    assert sum(reports[0]['rank_queue_capacity_bytes']) > 0
    assert reports[0]['rank_queue_capacity_bytes'] == [2*v for v in reports[1]['rank_queue_capacity_bytes']]
    assert reports[0]['rank_work'] == reports[1]['rank_work']
