"""Batched report framing across a partial final batch and ownership layouts."""
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
@pytest.mark.parametrize('target_owner', [1, None])
def test_projection_report_partial_batch_preserves_counts_and_result(tmp_path, target_owner):
    clock = b.Clock(dt=b.ms)
    a = b.NeuronGroup(3, 'x:1', threshold='timestep(t,dt)%2==0', reset='x=0', clock=clock)
    z = b.NeuronGroup(5, 'x:1', clock=clock)
    objects = [a, z, b.StateMonitor(z, 'x', record=True)]
    for q in range(129):
        syn = b.Synapses(a, z, 'w:1 (constant)', on_pre='x_post+=w', clock=clock)
        brian2_rust.connect_fixed_total(
            syn, q % 3 + 1, seed=17+q,
            initializers={'w': brian2_rust.Uniform(-1, 1)},
            delay_initializer=brian2_rust.Uniform(0*clock.dt, 3*clock.dt))
        objects.append(syn)
    model = lower_network(b.Network(*objects), 6*clock.dt)
    project = tmp_path/'mpi'
    write_mpi_project(model, project, ranks=4, population_owners=(3, target_owner), compact_populations=True)
    stats = json.loads((project/'manifest.json').read_text())['projection_compaction']
    assert stats['projection_report_batch_size'] == 128
    assert stats['projection_report_collectives'] == 2
    compile_mpi_project(project, opt_level=1, panic_strategy='abort')
    run_mpi_project(project, tmp_path/'result', timeout=30)
    report = json.loads((tmp_path/'result/mpi-runtime.json').read_text())
    assert report['projection_report_collectives'] == 2
    assert report['projection_report_packet_values'] == 128*5
    assert len(report['procedural_topology']) == 129
    for q, (observed, instance, definition) in enumerate(zip(
            report['procedural_topology'], model['instance']['synapses'],
            model['definition']['synapses'], strict=True)):
        edges = instance['topology']['edge_count']
        assert observed['projection'] == q and observed['global_edges'] == edges
        ranks = np.array(observed['rank_stats']).reshape(4, 2)
        assert (ranks.sum(axis=0) == edges).all()
        assert len(observed['build_seconds']) == 4
        assert np.isfinite(observed['build_seconds']).all()
        if target_owner is not None:
            assert observed['construction'] == 'target-owner-local'
            assert ranks[target_owner].tolist() == [edges, edges]
            assert observed['rank_csr_offset_bytes'][target_owner] == 8*(definition['source_count']+1)
            assert sum(observed['rank_csr_offset_bytes']) == 8*(definition['source_count']+1)
        else:
            assert observed['construction'] == 'distributed-draw-ranges'
    (tmp_path/'model.json').write_text(json.dumps(model))
    subprocess.run([str(RUNNER), str(tmp_path/'model.json'), str(tmp_path/'reference')],
                   check=True, capture_output=True)
    for name in ['results.bin', 'events.bin']:
        assert (tmp_path/'result'/name).read_bytes() == (tmp_path/'reference'/name).read_bytes()
