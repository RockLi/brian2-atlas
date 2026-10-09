"""MPI training: actual rank execution, exact CPU parity and boundary replay."""
import copy
import json
import os
import pickle
from pathlib import Path
import subprocess
import sys

import brian2 as b
import numpy as np
import pytest

from test_mpi import device, RUNNER, reference, assert_same_result, real_mpi
from brian2_rust.distributed import write_mpi_project, compile_mpi_project, run_mpi_project
from brian2_rust.export import lower_network
from brian2_rust.results import load_results


@pytest.fixture(autouse=True)
def quick_compiler(monkeypatch):
    # Correctness does not depend on optimization. Existing MPI tests retain
    # optimized paths; avoid recompiling the same branch model at -O3 40 times.
    import brian2_rust.distributed as distributed
    original=distributed.compile_mpi_project
    monkeypatch.setattr(distributed,'compile_mpi_project',
                        lambda path,**kwargs: original(path,opt_level=0,**kwargs))


def training_network(*, triplet=False, uniform=False, structure=False):
    clock = b.Clock(dt=b.second / 1024, name='training_clock')
    group = b.NeuronGroup(3, 'dv/dt=drive*1024*Hz:1 (unless refractory)\n'
                         'drive:1\nx:1', threshold='v>1', reset='v=rand()*0.125',
                         refractory=clock.dt, method='euler', clock=clock,
                         name='training_neurons')
    group.drive = [0.5, 0.25, 0.5]
    model = 'w:1\ndapre/dt=-apre/(8*ms):1 (event-driven)\n'
    model += 'dapost/dt=-apost/(11*ms):1 (event-driven)\nflag:boolean\ncount:integer'
    pre = 'apre+=0.125; w=clip(w+apost,0,1); x_post+=w; count+=1; flag=not flag'
    post = 'apost-=0.0625; w=clip(w+apre,0,1); x_post=x_post*0.5+w'
    if triplet:
        model += '\ndslow/dt=-slow/(17*ms):1 (event-driven)'
        post = 'w=clip(w+apre*slow,0,1); slow+=0.125; apost-=0.0625; x_post+=w'
    if structure:
        model += '\nlive:integer\nborn:integer\nactive_after:second'
    syn = b.Synapses(group, group, model, on_pre=pre, on_post=post,
                     clock=clock, name='training_synapses')
    # Original creation order differs from source-major; duplicate edges and
    # bidirectional loops exercise both CSR orders and rank-global identities.
    syn.connect(i=[2, 0, 1, 0, 2, 1], j=[0, 2, 0, 2, 1, 2])
    syn.w = [0.125, 0.25, 0.375, 0.5, 0.625, 0.75]
    syn.delay = (3 if uniform else np.array([3, 4, 1, 2, 5, 3])) * clock.dt
    syn.post.delay = 2 * clock.dt
    if structure:
        syn.live = 1
    spikes = b.SpikeMonitor(group, name='training_spikes')
    monitor = b.StateMonitor(group, ['v', 'x'], record=True, name='training_monitor')
    return b.Network(group, syn, spikes, monitor), clock, group, syn, spikes, monitor


def snapshot(device, net, group, syn, spikes, monitor):
    return {
        'time': float(net.t / b.second),
        'neurons': [np.asarray(group.state(name)[:]).tobytes() for name in ('v', 'x', 'lastspike', 'not_refractory')],
        'synapses': {name: np.asarray(syn.state(name)[:]).tobytes() for name in
                     ('w', 'apre', 'apost', 'lastupdate', 'flag', 'count') + (('slow',) if 'slow' in syn.variables else ())},
        'spikes': [np.asarray(spikes.i[:]).tobytes(), np.asarray(spikes.t[:]).tobytes()],
        'monitor': [np.asarray(monitor.v[:]).tobytes(), np.asarray(monitor.x[:]).tobytes()],
        'pending': copy.deepcopy(device._pending_events),
        'rng': device._rng_seed,
    }


@real_mpi
@pytest.mark.parametrize('ranks', [1, 2, 4])
@pytest.mark.parametrize('triplet', [False, True])
def test_stdp_reference_exact(device, tmp_path, ranks, triplet):
    net, clock, *_ = training_network(triplet=triplet)
    model = lower_network(net, 19 * clock.dt, rng_seed=123)
    expected = reference(model, tmp_path / 'reference')
    write_mpi_project(model, tmp_path / 'mpi', ranks=ranks, runner=RUNNER)
    compile_mpi_project(tmp_path / 'mpi')
    run_mpi_project(tmp_path / 'mpi', tmp_path / 'result')
    assert_same_result(expected, load_results(model, tmp_path / 'result'))


@real_mpi
@pytest.mark.parametrize('ranks', [1, 2, 4])
@pytest.mark.parametrize('uniform', [False, True])
def test_segment_checkpoint_branch_exact(device, tmp_path, ranks, uniform, rank_backends=None):
    def setup(tag):
        device.reinit()
        b.set_device('rust_standalone', engine='mpi', ranks=ranks,
                     directory=tmp_path / tag, runner=RUNNER,
                     **({'rank_backends': rank_backends, 'numeric_mode': 'mixed-f32'} if rank_backends else {}))
        b.seed(123)
        return training_network(triplet=True, uniform=uniform)
    net, clock, group, syn, spikes, monitor = setup('whole')
    net.run(19 * clock.dt)
    whole = snapshot(device, net, group, syn, spikes, monitor)
    net, clock, group, syn, spikes, monitor = setup('segments')
    # Zero duration is a supported initialization path.
    net.run(0 * clock.dt)
    net.run(6 * clock.dt)
    assert any(device._pending_events.values())
    net.store('training', filename=tmp_path / 'checkpoint')
    if ranks == 2 and not uniform:
        # A fresh Python process and a fresh MPI communicator resume the same
        # checkpoint. No object, queue or RNG survives via process memory.
        script = '''
import pickle, sys
from test_mpi_training import *
from brian2.devices.device import all_devices
import brian2_rust.distributed as distributed
compile_original=distributed.compile_mpi_project
distributed.compile_mpi_project=lambda path:compile_original(path,opt_level=0)
d=all_devices['rust_standalone']; d.reinit()
backends=json.loads(sys.argv[4])
b.set_device('rust_standalone', engine='mpi', ranks=2, runner=RUNNER,
 directory=Path(sys.argv[2]), **({'rank_backends':backends, 'numeric_mode':'mixed-f32'} if backends else {}))
net,clock,g,s,sp,m=training_network(triplet=True, uniform=False)
net.restore('training',filename=sys.argv[1],restore_random_state=True)
net.run(13*clock.dt)
with open(sys.argv[3], 'wb') as f: pickle.dump(snapshot(d,net,g,s,sp,m),f)
'''
        env = os.environ.copy()
        env['PYTHONPATH'] = str(Path(__file__).parent) + os.pathsep + env.get('PYTHONPATH', '')
        subprocess.run([sys.executable, '-c', script, str(tmp_path / 'checkpoint'),
                        str(tmp_path / 'fresh'), str(tmp_path / 'snapshot'), json.dumps(rank_backends)],
                       env=env, check=True, capture_output=True, text=True, timeout=120)
        assert pickle.loads((tmp_path / 'snapshot').read_bytes()) == whole
    net.run(13 * clock.dt)
    assert snapshot(device, net, group, syn, spikes, monitor) == whole
    device.build_options['ranks'] = ranks+1
    if rank_backends:
        device.build_options['rank_backends'] = [*rank_backends, 'cpu']
    with pytest.raises(Exception,match='configuration mismatch'):
        net.run(clock.dt)
    with pytest.raises(Exception,match='configuration mismatch'):
        net.store('changed', filename=tmp_path/'invalid-checkpoint')
    assert not (tmp_path/'invalid-checkpoint').exists()
    device.build_options['ranks'] = ranks
    if rank_backends:
        device.build_options['rank_backends'] = rank_backends[::-1]
        with pytest.raises(Exception,match='configuration mismatch'):
            net.run(clock.dt)
        device.build_options['rank_backends'] = rank_backends
    device.build_options['engine'] = 'reference'
    with pytest.raises(Exception,match='configuration mismatch: engine'):
        net.run(clock.dt)
    device.build_options['engine'] = 'mpi'
    net.restore('training', filename=tmp_path / 'checkpoint', restore_random_state=True)
    # Validation modifies rates and weights, then restore rewinds the branch.
    group.drive = 0
    syn.w = 0
    net.run(3 * clock.dt)
    net.restore('training', filename=tmp_path / 'checkpoint', restore_random_state=True)
    net.run(13 * clock.dt)
    assert snapshot(device, net, group, syn, spikes, monitor) == whole


@pytest.mark.skipif(os.environ.get('B2_TEST_GPU') != '1' or os.environ.get('B2_TEST_MPI') != '1', reason='requires actual MPI GPU')
def test_mixed_gpu_training_replay(device, tmp_path):
    test_segment_checkpoint_branch_exact(device, tmp_path, 2, False,
                                         ['cpu', os.environ.get('B2_TEST_MPI_GPU', 'metal')])


@real_mpi
@pytest.mark.parametrize('uniform', [False, True])
def test_segments_shorter_than_pending_delay(device, tmp_path, uniform):
    def setup(tag):
        device.reinit()
        b.set_device('rust_standalone', engine='mpi', ranks=2,
                     directory=tmp_path / tag, runner=RUNNER)
        b.seed(123)
        return training_network(triplet=True, uniform=uniform)
    net, clock, group, syn, spikes, monitor = setup('whole')
    net.run(19*clock.dt)
    expected = snapshot(device, net, group, syn, spikes, monitor)
    net, clock, group, syn, spikes, monitor = setup('short')
    for ticks in (6, 1, 1, 11):
        net.run(ticks*clock.dt)
    assert snapshot(device, net, group, syn, spikes, monitor) == expected


@real_mpi
def test_host_write_and_structural_old_event(device, tmp_path):
    b.set_device('rust_standalone', engine='mpi', ranks=4,
                 directory=tmp_path / 'structure', runner=RUNNER)
    net, clock, group, syn, spikes, monitor = training_network(uniform=True, structure=True)
    net.run(3 * clock.dt)
    assert any(device._pending_events.values())
    # Regrow every candidate at the boundary. All old pre and post arrivals
    # must be ignored before even the automatic trace update/lastupdate.
    syn.live = 1
    syn.born = 1
    syn.active_after = net.t + syn.delay[:]
    syn.w = 0.875
    syn.apre = syn.apost = 0
    group.drive = 0
    group.v = 0
    before_update = np.asarray(syn.lastupdate[:]).copy()
    net.run(2 * clock.dt)
    np.testing.assert_array_equal(syn.w[:], 0.875)
    np.testing.assert_array_equal(syn.apre[:], 0)
    np.testing.assert_array_equal(syn.apost[:], 0)
    np.testing.assert_array_equal(syn.lastupdate[:], before_update)


def test_checkpoint_policy_and_corruption(device, tmp_path):
    b.set_device('rust_standalone', engine='mpi', ranks=2, runner=RUNNER)
    net, clock, group, syn, *_ = training_network()
    net.store('training', filename=tmp_path / 'checkpoint')
    group.v = 0.75
    device.build_options['ranks'] = 4
    with pytest.raises(Exception, match='configuration mismatch: ranks|configuration mismatch: rank_backends, ranks'):
        net.restore('training', filename=tmp_path / 'checkpoint')
    np.testing.assert_array_equal(group.v[:], 0.75)
    device.build_options['ranks'] = 2
    content = (tmp_path / 'checkpoint').read_bytes()
    (tmp_path / 'checkpoint').write_bytes(content[:len(content)//2])
    with pytest.raises(Exception):
        net.restore('training', filename=tmp_path / 'checkpoint')
    np.testing.assert_array_equal(group.v[:], 0.75)


def test_structural_generation_fields_require_pre_pathway(device, tmp_path):
    clock=b.Clock(dt=b.ms)
    group=b.NeuronGroup(2,'x:1',threshold='True',reset='x=0',clock=clock)
    syn=b.Synapses(group,group,'live:integer\nborn:integer\nactive_after:second',
                   on_post='live=1',clock=clock)
    syn.connect(i=[0],j=[1])
    model=lower_network(b.Network(group,syn),2*clock.dt)
    with pytest.raises(Exception,match='generation fields require a pre pathway'):
        write_mpi_project(model,tmp_path/'unsupported',ranks=2,runner=RUNNER)
    assert not (tmp_path/'unsupported').exists()


@real_mpi
def test_custom_dynamics_after_groups(device,tmp_path):
    net,clock,group,*_=training_network()
    group.run_regularly('x=x+int(v>0)',when='after_groups')
    model=lower_network(net,9*clock.dt,rng_seed=123)
    expected=reference(model,tmp_path/'reference')
    write_mpi_project(model,tmp_path/'mpi',ranks=4,runner=RUNNER)
    compile_mpi_project(tmp_path/'mpi',opt_level=0)
    run_mpi_project(tmp_path/'mpi',tmp_path/'result')
    assert_same_result(expected,load_results(model,tmp_path/'result'))
