"""Delay ring optimization preserves exact FIFO and f64 state behavior."""
from pathlib import Path
import json
import shutil
import subprocess

import brian2 as b
import pytest

import brian2_rust
from brian2_rust.distributed import write_mpi_project, compile_mpi_project, run_mpi_project
from brian2_rust.export import lower_network
from test_mpi import device as device, real_mpi, RUNNER

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('checked', [False, True])
def test_additive_ring_matches_frozen_kernel_including_overflow(tmp_path, checked):
    rustc = shutil.which('rustc')
    if not rustc:
        pytest.skip('rustc required')
    evidence = ROOT/'mpi-evidence/event-delivery-kernel'
    shutil.copy2(evidence/'baseline.rs', tmp_path/'baseline.rs')
    shutil.copy2(ROOT/'python/brian2_rust/mpi_runtime/additive.rs', tmp_path/'candidate.rs')
    shutil.copy2(evidence/'harness.rs', tmp_path/'harness.rs')
    command = [rustc, '--edition=2021', '-C', 'opt-level=1', '-C',
               'overflow-checks='+('yes' if checked else 'no'),
               str(tmp_path/'harness.rs'), '-o', str(tmp_path/'check')]
    if checked:
        command += ['--cfg', 'checked_overflow']
    subprocess.run(command, check=True, capture_output=True, timeout=30)
    result = subprocess.run([str(tmp_path/'check')], check=True, capture_output=True,
                            text=True, timeout=30)
    report = json.loads(result.stdout)
    assert report == {'validation_calls': 29484, 'overflow_checks': checked}


@real_mpi
@pytest.mark.usefixtures('device')
@pytest.mark.parametrize('maximum_delay', [0, 3, 19])
@pytest.mark.parametrize('compact', [False, True])
@pytest.mark.parametrize('narrow_queue', [False, True])
def test_additive_ring_full_mpi_matches_independent_reference(tmp_path, maximum_delay, compact, narrow_queue):
    clock = b.Clock(dt=b.ms)
    source = b.NeuronGroup(5, 'x:1', threshold='True', reset='x=0', clock=clock)
    target = b.NeuronGroup(7, 'x:1', clock=clock)
    syn = b.Synapses(source[1:4], target[2:6], 'w:1 (constant)', on_pre='x_post+=w', clock=clock)
    brian2_rust.connect_fixed_total(syn, 103, seed=42,
        initializers={'w': brian2_rust.Uniform(-1, 1)},
        delay_initializer=brian2_rust.Uniform(0*clock.dt, maximum_delay*clock.dt))
    model = lower_network(b.Network(source, target, syn, b.StateMonitor(target, 'x', record=True)), 7*clock.dt)
    project = tmp_path/'mpi'
    write_mpi_project(model, project, ranks=4, population_owners=(3, 1), compact_populations=compact,
                      compact_queue_indices=narrow_queue)
    compile_mpi_project(project, opt_level=1, panic_strategy='abort')
    run_mpi_project(project, tmp_path/'result', timeout=30)
    (tmp_path/'model.json').write_text(json.dumps(model))
    subprocess.run([str(RUNNER), str(tmp_path/'model.json'), str(tmp_path/'reference')],
                   check=True, capture_output=True, timeout=30)
    for name in ['results.bin', 'events.bin']:
        assert (tmp_path/'result'/name).read_bytes() == (tmp_path/'reference'/name).read_bytes()


@real_mpi
@pytest.mark.usefixtures('device')
def test_repeated_corrupt_input_keeps_abort_diagnostic(tmp_path):
    clock = b.Clock(dt=b.ms)
    group = b.NeuronGroup(3, 'x:1', clock=clock)
    model = lower_network(b.Network(group), 2*clock.dt)
    project = tmp_path/'mpi'
    write_mpi_project(model, project, ranks=2)
    compile_mpi_project(project, opt_level=1, panic_strategy='abort')
    data = bytearray((project/'instance.bin').read_bytes())
    data[-1] ^= 1
    (tmp_path/'corrupt.bin').write_bytes(data)
    for trial in range(20):
        result = subprocess.run(['mpiexec', '-n', '2', str(project/'b2-mpi'),
            str(tmp_path/'corrupt.bin'), str(tmp_path/('error-'+str(trial)))],
            capture_output=True, text=True, timeout=10)
        assert result.returncode != 0
        assert 'MPI instance differs from compiled plan' in result.stderr
        assert not (tmp_path/('error-'+str(trial))).exists()
