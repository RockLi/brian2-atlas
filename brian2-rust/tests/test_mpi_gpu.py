"""Rank heterogeneity: real MPI+GPU parity, explicit arithmetic and abort gates."""
from dataclasses import replace
import hashlib
import json
import os
import platform
import subprocess

import brian2 as b
import numpy as np
import pytest

from test_mpi import (device, network_model, reference, assert_same_result, RUNNER,
                      build_distributed_plan, verify_distributed_plan, write_mpi_project,
                      compile_mpi_project, run_mpi_project, load_results)
from brian2_rust.plan import PlanValidationError, build_execution_plan
from brian2_rust.mpi_gpu import backend_policy, PROFILE
from brian2_rust.export import lower_network

GPU = os.environ.get("B2_TEST_MPI_GPU", "metal" if platform.system() == "Darwin" else "cuda")
real_gpu = pytest.mark.skipif(os.environ.get("B2_TEST_MPI") != "1" or os.environ.get("B2_TEST_GPU") != "1",
    reason="set B2_TEST_MPI=1 B2_TEST_GPU=1 for real local MPI and Metal/CUDA")


@pytest.mark.parametrize("backends,mode", [
    (["cpu", "metal"], "reference-f64"), (["cpu", "cuda"], "float32"),
    (["cpu", "cpu"], "mixed-f32"), (["cpu"], "mixed-f32"),
    ("cpu,metal", "mixed-f32"), (["cpu", "auto"], "mixed-f32"),
    (["cpu", "cuda:-1"], "mixed-f32"), (["cpu", "cuda:01"], "mixed-f32"),
    (["cpu", "metal:1"], "mixed-f32"), (["cpu", None], "mixed-f32"),
])
def test_reject_invalid_policy(backends, mode):
    with pytest.raises(PlanValidationError):
        backend_policy(2, backends, mode)


def test_cpu_contract_unchanged_and_gpu_identity(device):
    model = network_model()
    cpu = build_distributed_plan(model, runner=RUNNER)
    assert cpu == build_distributed_plan(model, runner=RUNNER, rank_backends=["cpu", "cpu"])
    assert "rank_backends" not in cpu.to_dict()
    mixed = build_distributed_plan(model, runner=RUNNER, rank_backends=["cpu", "cuda:1"], numeric_mode="mixed-f32")
    assert mixed.numeric_profile == PROFILE
    assert mixed.shards == cpu.shards and mixed.routes == cpu.routes
    assert mixed.sha256 != cpu.sha256
    assert build_execution_plan(model, backend="mpi", runner=RUNNER,
        rank_backends=["cpu", "cuda:1"], numeric_mode="mixed-f32") == mixed
    with pytest.raises(PlanValidationError):
        verify_distributed_plan(replace(mixed, numeric_profile="reference-f64"), model)
    with pytest.raises(PlanValidationError):
        build_execution_plan(model, rank_backends=["cpu", "cuda"])


def test_artifacts_bind_device_selection_and_sources(device, tmp_path):
    model = network_model()
    path = tmp_path / "mpi"
    plan = write_mpi_project(model, path, runner=RUNNER,
                            rank_backends=["cpu", "cuda:2"], numeric_mode="mixed-f32")
    from brian2_rust.distributed import _verify_artifact
    manifest = _verify_artifact(path)
    assert "mpi_gpu_cuda.cu" in manifest["files"]
    assert 'gpu_scope' in (path / 'main.rs').read_text()
    assert 'cudaSetDevice' in (path / 'mpi_gpu_cuda.cu').read_text()
    with pytest.raises(PlanValidationError, match="backends differ"):
        write_mpi_project(model, tmp_path / "wrong", runner=RUNNER, plan=plan, rank_backends=["cuda", "cpu"])
    assert not (tmp_path / "wrong").exists()
    with (path / "mpi_gpu_cuda.cu").open("a") as output:
        output.write("\n// altered\n")
    with pytest.raises(ValueError, match="artifact changed"):
        _verify_artifact(path)


def test_device_requires_explicit_mixed_precision(device):
    with pytest.raises(PlanValidationError, match="mixed-f32"):
        device.activate(engine="mpi", ranks=2, rank_backends=["cpu", "metal"])
    device.activate(engine="mpi", ranks=2, rank_backends=["cpu", "metal"], numeric_mode="mixed-f32")
    with pytest.raises(NotImplementedError):
        device.activate(engine="aot", rank_backends=["cpu", "metal"])


def test_unsupported_gpu_operation_rejected_before_output(device, tmp_path):
    clock = b.Clock(dt=b.second / 1024)
    table = b.TimedArray([1., 2., 3.], dt=clock.dt)
    group = b.NeuronGroup(2, 'dv/dt = table(t)*Hz : 1', method='euler', clock=clock)
    model = lower_network(b.Network(group), 2*clock.dt, namespace={"table": table})
    with pytest.raises(PlanValidationError, match="TimedArray"):
        write_mpi_project(model, tmp_path / "bad", runner=RUNNER,
                          rank_backends=["cpu", GPU], numeric_mode="mixed-f32")
    assert not (tmp_path / "bad").exists()


def execute(model, path, backends, owners=None):
    plan = write_mpi_project(model, path, ranks=len(backends), runner=RUNNER,
                            rank_backends=backends, numeric_mode="mixed-f32", population_owners=owners)
    compile_mpi_project(path, opt_level=0)
    report = run_mpi_project(path, path / "result", timeout=60)
    assert report["numeric_profile"] == PROFILE
    assert report["rank_backends"] == backends
    for rank, backend in enumerate(backends):
        expected = sum(model["definition"]["populations"][p]["steps"]
                       for p, pop in enumerate(model["definition"]["populations"])
                       for code in pop["code_objects"] if code["kind"] == "state_update"
                       and plan.shards[rank].population_ranges[p][1] > plan.shards[rank].population_ranges[p][0])
        assert report["rank_gpu_dispatches"][rank] == (0 if backend == "cpu" else expected)
    return load_results(model, path / "result")


@real_gpu
@pytest.mark.parametrize("backends,owners", [
    (["cpu", GPU], None), ([GPU, "cpu"], None),
    (["cpu", GPU, "cpu"], None), ([GPU], None),
    (["cpu", GPU], [0, 1]), (["cpu", GPU, "cpu", GPU], None),
])
def test_real_mixed_recurrent_delay_parity(device, tmp_path, backends, owners):
    # Powers of two allow exact comparison despite the explicit f32 update path.
    # Includes both directions, duplicate edges, nonuniform delays, refractory,
    # sparse monitors, negative zero, and a zero-sized population shard (4 ranks).
    model = network_model()
    expected = reference(model, tmp_path / "reference")
    actual = execute(model, tmp_path / "mpi", backends, owners)
    assert_same_result(expected, actual)


@real_gpu
def test_real_mixed_numeric_contract(device, tmp_path):
    clock = b.Clock(dt=b.second / 1024)
    group = b.NeuronGroup(4, 'dv/dt = -v*Hz : 1', method='euler', clock=clock)
    group.v = [0.3, 0.7, 0.3, 0.7]
    model = lower_network(b.Network(group), 13*clock.dt)
    expected = reference(model, tmp_path / "reference")
    actual = execute(model, tmp_path / "mpi", ["cpu", GPU])
    initial = np.array([0.3, 0.7], dtype=np.float32)
    for _ in range(13):
        initial = initial - np.float32(1/1024)*initial
    values = actual["populations"][0]["states"]["v"]
    np.testing.assert_array_equal(values[:2], expected["populations"][0]["states"]["v"][:2])
    np.testing.assert_array_equal(values[2:], initial.astype(np.float64))
    assert not np.array_equal(values[2:], expected["populations"][0]["states"]["v"][2:])


@real_gpu
def test_gpu_failure_aborts_peers_without_cpu_fallback(device, tmp_path):
    model = network_model()
    path = tmp_path / "mpi"
    write_mpi_project(model, path, runner=RUNNER, rank_backends=["cpu", GPU], numeric_mode="mixed-f32")
    compile_mpi_project(path, opt_level=0)
    name = "libmpi-metal.dylib" if GPU == "metal" else "libmpi-cuda.so"
    (path / name).unlink()
    # Bypass the launcher's earlier integrity rejection to test rank failure
    # while the CPU peer is inside MPI collectives.
    process = subprocess.run(["mpiexec", "-n", "2", str(path / "b2-mpi"),
        str(path / "instance.bin"), str(path / "result")], capture_output=True, text=True, timeout=30)
    assert process.returncode != 0
    assert not (path / "result" / "mpi-runtime.json").exists()


@real_gpu
def test_real_rank_local_counter_rng(device, tmp_path):
    clock = b.Clock(dt=b.second / 1024)
    group = b.NeuronGroup(5, 'dv/dt = rand()*Hz : 1', method='euler', clock=clock)
    model = lower_network(b.Network(group), 7*clock.dt, rng_seed=789)
    expected = reference(model, tmp_path / 'reference')
    single = execute(model, tmp_path / 'gpu', [GPU])
    mixed = execute(model, tmp_path / 'mixed', ['cpu', GPU])
    key = lambda result: result['populations'][0]['states']['v']
    np.testing.assert_array_equal(key(mixed)[:2], key(expected)[:2])
    np.testing.assert_array_equal(key(mixed)[2:], key(single)[2:])


@real_gpu
def test_real_brian_device_entry(device, tmp_path):
    b.set_device('rust_standalone', runner=RUNNER, directory=tmp_path / 'device',
                 engine='mpi', ranks=2, rank_backends=['cpu', GPU], numeric_mode='mixed-f32')
    clock = b.Clock(dt=b.second / 1024)
    group = b.NeuronGroup(4, 'dv/dt = 1024*Hz : 1', method='euler', clock=clock)
    monitor = b.StateMonitor(group, 'v', record=True)
    b.Network(group, monitor).run(3*clock.dt)
    np.testing.assert_array_equal(group.v[:], np.full(4, 3.))
    np.testing.assert_array_equal(monitor.v[:], np.tile([0., 1., 2.], (4, 1)))
    assert device.last_execution_plan.rank_backends == ('cpu', GPU)


def test_gpu_build_requires_complete_library_hashes(device, tmp_path):
    model = network_model()
    path = tmp_path / 'mpi'
    write_mpi_project(model, path, runner=RUNNER,
                      rank_backends=['cpu', 'cuda'], numeric_mode='mixed-f32')
    binary = b'unexecuted test artifact'
    (path / 'b2-mpi').write_bytes(binary)
    build = {'executable_sha256': hashlib.sha256(binary).hexdigest()}
    (path / 'build.json').write_text(json.dumps(build))
    with pytest.raises(ValueError, match='library inventory'):
        run_mpi_project(path, path / 'result')
    (path / 'libmpi-cuda.so').write_bytes(binary)
    build['gpu_libraries'] = {'libmpi-cuda.so': '0'*64}
    (path / 'build.json').write_text(json.dumps(build))
    with pytest.raises(ValueError, match='GPU library changed'):
        run_mpi_project(path, path / 'result')
    assert not (path / 'result').exists()
