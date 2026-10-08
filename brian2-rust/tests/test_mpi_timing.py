"""Stage clocks distinguish injected rank skew from simulation and collection."""
import hashlib
import json
import subprocess

import brian2 as b
import numpy as np
import pytest

import brian2_rust
from brian2_rust.distributed import write_mpi_project, compile_mpi_project, run_mpi_project
from brian2_rust.export import lower_network
from test_mpi import device as device, real_mpi, RUNNER


@real_mpi
@pytest.mark.usefixtures('device')
@pytest.mark.parametrize('compact', [False, True])
def test_rank_stage_clocks_separate_initialization_and_simulation_skew(tmp_path, compact):
    clock = b.Clock(dt=b.ms)
    a = b.NeuronGroup(5, 'x:1', threshold='timestep(t,dt)%2==0', reset='x=0', clock=clock)
    z = b.NeuronGroup(7, 'x:1', clock=clock)
    syn = b.Synapses(a, z, 'w:1 (constant)', on_pre='x_post+=w', clock=clock)
    brian2_rust.connect_fixed_total(
        syn, 101, seed=17, initializers={'w': brian2_rust.Uniform(-1, 1)},
        delay_initializer=brian2_rust.Uniform(0*clock.dt, 3*clock.dt))
    model = lower_network(b.Network(a, z, syn, b.StateMonitor(z, 'x', record=True)), 6*clock.dt)
    project = tmp_path/'mpi'
    write_mpi_project(model, project, ranks=2, population_owners=(0, 1), compact_populations=compact)
    # Inject delays only into this test executable. A rendezvous makes the slow
    # rank known without depending on topology size, machine speed or I/O cache.
    source = (project/'main.rs').read_text()
    for anchor, milliseconds in [('    let mpi_local_initialization_seconds = ', 1000),
                                 ('    let mpi_local_simulation_seconds = ', 500)]:
        assert source.count(anchor) == 1
        source = source.replace(anchor,
            '    mpi_check(unsafe { b2mpi_barrier() })?;\n'
            f'    if mpi.rank == 1 {{ std::thread::sleep(std::time::Duration::from_millis({milliseconds})); }}\n'
            + anchor)
    (project/'main.rs').write_text(source)
    manifest = json.loads((project/'manifest.json').read_text())
    manifest['files']['main.rs'] = hashlib.sha256(source.encode()).hexdigest()
    (project/'manifest.json').write_text(json.dumps(manifest))
    compile_mpi_project(project, opt_level=1, panic_strategy='abort')
    run_mpi_project(project, tmp_path/'result', timeout=20)
    report = json.loads((tmp_path/'result/mpi-runtime.json').read_text())
    assert report['stage_timing_schema'] == 'b2-mpi-stage-timing-v1'
    assert report['rank_stage_columns'] == [
        'initialization_local', 'initialization_wait', 'simulation_local',
        'simulation_wait', 'result_collection_and_reporting']
    stages = np.array(report['rank_stage_seconds']).reshape(2, 5)
    assert np.isfinite(stages).all() and (stages >= 0).all()
    assert stages[1, 0] >= 0.9 and stages[0, 1] >= 0.7
    assert stages[1, 2] >= 0.45 and stages[0, 3] >= 0.3
    summary = json.loads((tmp_path/'result/summary.json').read_text())
    assert summary['timings']['initialization_seconds'] >= sum(stages[0, :2])
    assert summary['timings']['simulation_and_recording_seconds'] >= sum(stages[0, 2:])
    (tmp_path/'model.json').write_text(json.dumps(model))
    subprocess.run([str(RUNNER), str(tmp_path/'model.json'), str(tmp_path/'reference')],
                   check=True, capture_output=True)
    for name in ['results.bin', 'events.bin']:
        assert (tmp_path/'result'/name).read_bytes() == (tmp_path/'reference'/name).read_bytes()
