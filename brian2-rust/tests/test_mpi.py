"""MPI: independent reference parity, partition invariance and fail-closed gates."""
import copy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys

import brian2 as b
from brian2.devices.device import all_devices
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import brian2_rust
from brian2_rust.distributed import (build_distributed_plan, verify_distributed_plan,
    write_mpi_project, compile_mpi_project, run_mpi_project, _explicit_routes)
from brian2_rust.export import lower_network
from brian2_rust.plan import PlanValidationError, build_execution_plan, explain_plan
from brian2_rust.protocol import attach_protocol, canonical_bytes
from brian2_rust.results import load_results

RUNNER = Path(os.environ.get("B2_RUNNER", ROOT / "target/release/b2-runner"))
real_mpi = pytest.mark.skipif(os.environ.get("B2_TEST_MPI") != "1",
    reason="set B2_TEST_MPI=1 with mpicc/mpiexec and local socket permission")


@pytest.fixture
def device():
    previous = b.get_device()
    native = all_devices["rust_standalone"]
    native.reinit()
    b.set_device("rust_standalone", runner=RUNNER)
    b.seed(123)
    yield native
    native.reinit()
    b.set_device(previous)


def network_model(*, stochastic=False, empty=False, monitor=True):
    clock = b.Clock(dt=b.second / 1024, name="mpi_clock")
    equation = "dv/dt = drive*1024*Hz : 1 (unless refractory)\ndrive:1 (constant)\nx:1"
    e = b.NeuronGroup(7, equation, threshold="v > 1", reset="v=0", refractory=2*clock.dt,
                      method="euler", clock=clock, name="mpi_e")
    i = b.NeuronGroup(3, equation, threshold="v > 1", reset="v=0", refractory=clock.dt,
                      method="euler", clock=clock, name="mpi_i")
    e.drive = [0.5, 0.25, 0.125, 0.5, 0.25, 0.125, 0.5]
    i.drive = [0.25, 0.5, 0.125]
    # Keep exact negative-zero bits in an untouched state.
    e.x = -0.0
    i.x = -0.0
    on_pre = "v_post += w * (0.5 + rand())" if stochastic else "v_post += w"
    ei = b.Synapses(e, i, "w:1 (constant)", on_pre=on_pre, clock=clock, name="mpi_ei")
    ie = b.Synapses(i, e, "w:1 (constant)", on_pre="v_post += w", clock=clock, name="mpi_ie")
    if empty:
        ei.connect(i=np.array([], dtype=int), j=np.array([], dtype=int))
        ie.connect(i=np.array([], dtype=int), j=np.array([], dtype=int))
    else:
        # Unsorted source order and duplicate edges with heterogeneous delays.
        ei.connect(i=[6, 0, 3, 0, 2, 4, 1, 5], j=[0, 2, 1, 2, 0, 1, 2, 0])
        ei.w = [0.125, 0.25, -0.125, 0.5, 0.125, 0.25, 0.125, 0.25]
        ei.delay = np.array([1, 3, 2, 1, 4, 2, 3, 1]) * clock.dt
        ie.connect(i=[2, 0, 1, 1], j=[0, 6, 2, 5])
        ie.w = -0.125
        ie.delay = 2*clock.dt
    objects = [e, i, ei, ie]
    if monitor:
        objects += [b.StateMonitor(e, ["v", "x"], record=[6, 0, 3]),
                    b.StateMonitor(i, "v", record=True), b.SpikeMonitor(e), b.SpikeMonitor(i)]
    return lower_network(b.Network(*objects), 17*clock.dt, rng_seed=123)


def reference(model, directory):
    path = directory.with_suffix(".json")
    path.write_text(json.dumps(model))
    subprocess.run([str(RUNNER), str(path), str(directory)], check=True, capture_output=True)
    return load_results(model, directory)


def assert_same_result(expected, actual):
    """Check exact stored bits, not allclose or value-only signed-zero equality."""
    def compare(a, b):
        if isinstance(a, dict):
            assert set(a) == set(b)
            for key in a:
                compare(a[key], b[key])
        elif isinstance(a, (list, tuple)):
            assert len(a) == len(b)
            for left, right in zip(a, b, strict=True):
                compare(left, right)
        elif isinstance(a, np.ndarray):
            assert a.shape == b.shape and a.dtype == b.dtype
            assert a.tobytes() == b.tobytes()
        else:
            assert a == b
    compare(expected["populations"], actual["populations"])
    compare(expected["synapses"], actual["synapses"])
    assert expected["metadata"]["synaptic_events"] == actual["metadata"]["synaptic_events"]


@pytest.mark.parametrize("owners", [(None, None), (1, 2)])
@pytest.mark.parametrize("delays", [[3], [7, 6, 5, 4, 3, 2, 1, 8, 9]])
def test_vectorized_explicit_routes_match_scalar_contract(owners, delays):
    sources = [6, 0, 3, 0, 2, 4, 1, 5, 6]
    targets = [0, 4, 1, 2, 0, 1, 3, 0, 4]
    synapse = {"source_population": 0, "target_population": 1,
               "source_start": 0, "target_start": 0}
    values = {"source": sources, "target": targets,
              "pathways": [{"delay_ticks": delays}]}
    pops = [{"count": 7}, {"count": 5}]
    ranks = 4
    expected = {}
    expected_incoming = [0] * ranks
    expected_delay = None
    for edge, (source, target) in enumerate(zip(sources, targets, strict=True)):
        source_rank = owners[0] if owners[0] is not None else ((source + 1) * ranks - 1) // 7
        target_rank = owners[1] if owners[1] is not None else ((target + 1) * ranks - 1) // 5
        expected[source_rank, target_rank] = expected.get((source_rank, target_rank), 0) + 1
        expected_incoming[target_rank] += 1
        if source_rank != target_rank:
            delay = delays[0 if len(delays) == 1 else edge]
            expected_delay = delay if expected_delay is None else min(expected_delay, delay)
    routes, incoming, minimum_delay = _explicit_routes(
        synapse, values, pops, ranks, owners, chunk_edges=3)
    assert routes == [(*route, count) for route, count in sorted(expected.items())]
    assert incoming == expected_incoming
    assert minimum_delay == expected_delay


def test_plan_ownership_and_integrity(device):
    model = network_model()
    before = canonical_bytes(model)
    plan = build_execution_plan(model, backend="mpi", ranks=4)
    assert plan.sha256 == build_distributed_plan(model, ranks=4).sha256
    assert canonical_bytes(model) == before
    for p, pop in enumerate(model["definition"]["populations"]):
        indices = [i for shard in plan.shards for i in range(*shard.population_ranges[p])]
        assert indices == list(range(pop["count"]))
    for q, syn in enumerate(model["instance"]["synapses"]):
        assert sum(s.incoming_edges[q] for s in plan.shards) == len(syn["source"])
    assert any(a != b for _, a, b, _ in plan.routes)
    assert plan.min_cross_rank_delay_ticks >= 1
    assert "rank-local" in explain_plan(plan)
    with pytest.raises(PlanValidationError):
        verify_distributed_plan(replace(plan, ranks=2), model)
    with pytest.raises(PlanValidationError):
        verify_distributed_plan(replace(plan, exchange_nodes=()), model)
    bad = copy.deepcopy(model)
    bad["instance"]["rng_seed"] += 1
    attach_protocol(bad)
    with pytest.raises(PlanValidationError):
        verify_distributed_plan(plan, bad)


@pytest.mark.parametrize("ranks", [0, -1, 257, True, 1.5])
def test_invalid_rank_count(device, ranks):
    with pytest.raises(PlanValidationError, match="ranks"):
        build_distributed_plan(network_model(), ranks=ranks)


def test_plan_accepts_zero_delay_and_rejects_unvalidated_effects(device, tmp_path):
    model = network_model()
    model["instance"]["synapses"][0]["pathways"][0]["delay_ticks"][0] = 0
    model["instance"]["synapses"][0]["pathways"][0]["delay"][0] = "0000000000000000"
    attach_protocol(model)
    plan = write_mpi_project(model, tmp_path / "zero-delay")
    assert plan.min_cross_rank_delay_ticks == 0
    model = network_model()
    model["definition"]["schedule"]["nodes"][0]["effects"]["reads"] = ["forged"]
    attach_protocol(model)
    with pytest.raises(PlanValidationError):
        build_distributed_plan(model)


def test_large_timed_array_is_a_hashed_mpi_artifact(device, tmp_path):
    clock = b.Clock(dt=b.ms)
    table = b.TimedArray(
        np.arange(10_000, dtype=np.float64).reshape(5000, 2), dt=clock.dt)
    group = b.NeuronGroup(
        2, "dx/dt=table(t, i)*Hz : 1", method="euler", clock=clock,
        namespace={"table": table})
    model = lower_network(b.Network(group), 2*clock.dt)
    path = tmp_path / "timed-array"
    write_mpi_project(model, path, ranks=2)

    from brian2_rust.distributed import _verify_artifact
    manifest = _verify_artifact(path)
    [filename] = manifest["timed_array_files"]
    payload = path / filename
    assert payload.stat().st_size == table.values.nbytes
    assert manifest["files"][filename] == hashlib.sha256(
        payload.read_bytes()).hexdigest()
    assert "include_bytes!" in (path / "main.rs").read_text()

    with payload.open("ab") as output:
        output.write(b"altered")
    with pytest.raises(ValueError, match="artifact changed"):
        _verify_artifact(path)


def test_presynaptic_state_rejected_and_local_plasticity_accepted(device):
    clock = b.Clock(dt=b.ms)
    g = b.NeuronGroup(2, "v:1", threshold="v>1", reset="v=0", clock=clock)
    s = b.Synapses(g, g, on_pre="v_post += v_pre", clock=clock)
    s.connect(i=[0], j=[1]); s.delay = b.ms
    model = lower_network(b.Network(g, s), 3*b.ms)
    with pytest.raises(PlanValidationError, match="presynaptic"):
        build_distributed_plan(model)
    s2 = b.Synapses(g, g, "w:1", on_pre="w+=1; v_post+=w", clock=clock)
    s2.connect(i=[0], j=[1]); s2.delay = b.ms
    plan = build_distributed_plan(lower_network(b.Network(g, s2), 3*b.ms))
    assert plan.schema == "b2-distributed-plan-v1"


def stateful_summed_model():
    clock = b.Clock(dt=b.ms, name="stateful_clock")
    source = b.SpikeGeneratorGroup(
        5, [4, 0, 2, 1, 4], np.array([0, 0, 2, 4, 5])*clock.dt,
        clock=clock, name="stateful_source")
    target = b.NeuronGroup(4, "total:1\nlabel:integer (constant)",
                           clock=clock, name="stateful_target")
    target.label = [3, -2, 7, 0]
    synapse = b.Synapses(
        source, target,
        "dx/dt=-x/(10*ms):1 (clock-driven)\n"
        "ds/dt=-s/(20*ms)+x*(1-s)/(5*ms):1 (clock-driven)\n"
        "total_post=s:1 (summed)\n"
        "w:1 (constant)",
        on_pre="x += w", method="rk4", clock=clock, name="stateful_synapse")
    # Unsorted sources, duplicate endpoints and heterogeneous delays exercise
    # target-owned sharding while retaining original edge identities.
    synapse.connect(i=[4, 0, 2, 0, 3, 1, 4], j=[0, 3, 1, 3, 2, 0, 2])
    synapse.w = [0.5, 0.25, 0.75, -0.125, 0.375, 0.625, 0.125]
    synapse.delay = np.array([1, 2, 1, 3, 2, 1, 2])*clock.dt
    monitor = b.StateMonitor(target, ["total", "label"], record=True)
    return lower_network(b.Network(source, target, synapse, monitor),
                         9*clock.dt, rng_seed=321)


def test_stateful_postsummed_plan_and_source(device, tmp_path):
    model = stateful_summed_model()
    plan = write_mpi_project(model, tmp_path / "mpi", ranks=4)
    assert plan.schema == "b2-distributed-plan-v1"
    assert sum(shard.incoming_edges[0] for shard in plan.shards) == 7
    source = (tmp_path / "mpi/main.rs").read_text()
    assert "for edge in 0..s0_local_edge_count" in source
    assert "collect_indexed_bits(&s0_original_edges" in source
    assert "for target_state in 0.max(p1_start)..4.min(p1_stop)" in source
    assert "rank_peak_rss_bytes" in source
    summed = source.split("// canonical node synapse/0/code/0", 1)[1].split(
        "// canonical node", 1)[0]
    state_update = source.split("// canonical node synapse/0/code/1", 1)[1].split(
        "// canonical node", 1)[0]
    assert "target_state" in summed and "partition_point" not in summed
    assert "partition_point" not in state_update


@real_mpi
@pytest.mark.parametrize("ranks", [1, 2, 4])
def test_stateful_postsummed_matches_reference(device, tmp_path, ranks):
    model = stateful_summed_model()
    expected = reference(model, tmp_path / "reference")
    write_mpi_project(model, tmp_path / "mpi", ranks=ranks)
    compile_mpi_project(tmp_path / "mpi")
    report = run_mpi_project(tmp_path / "mpi", tmp_path / "result")
    assert len(report["rank_peak_rss_bytes"]) == ranks
    assert all(value > 0 for value in report["rank_peak_rss_bytes"])
    assert_same_result(expected, load_results(model, tmp_path / "result"))


def binary_model(tmp_path, *, mutate_transmission=False):
    from brian2_rust.binary_topology import HEADER, MAGIC
    path = tmp_path / "graph.b2csr"
    # Subgroup endpoints, an empty source row, duplicate and unsorted targets.
    with path.open("wb") as stream:
        stream.write(HEADER.pack(MAGIC, 5, 5, 7, 1))
        stream.write(np.array([0, 2, 2, 4, 6, 7], dtype="<u8").tobytes())
        stream.write(np.array([4, 1, 0, 0, 3, 1, 2], dtype="<u4").tobytes())
        stream.write(np.array([0.5, -0.25, 0.125, 0.25, -0.25, 0.5, 0.25], dtype="<f8").tobytes())
    clock = b.Clock(dt=b.second / 1024)
    g = b.NeuronGroup(8, "dv/dt=256*Hz:1 (unless refractory)\ntransmission:1",
                     threshold="v>1", reset="v=0", refractory=2*clock.dt,
                     method="euler", clock=clock)
    g.transmission = [1, 0, 1, 0, 1, 0, 1, 1]
    s = b.Synapses(g[2:7], g[1:6], "w:1 (constant)",
                   on_pre={"immediate": "v_post += w*transmission_pre*(0.5+rand())",
                           "later": "v_post += w*transmission_pre"}, clock=clock)
    brian2_rust.connect_binary_csr(s, path, parameters={"w": 0})
    s.immediate.delay = 0*b.second
    s.later.delay = 2*clock.dt
    generator = b.SpikeGeneratorGroup(2, [0, 1, 0, 1], np.array([0, 0, 4, 7])*clock.dt,
                                      clock=clock)
    input_syn = b.Synapses(generator, g,
        on_pre="transmission_post += 1" if mutate_transmission else "v_post += 0.5",
        delay=0*b.second, clock=clock)
    input_syn.connect(i=[0, 1], j=[0, 7])
    monitor = b.StateMonitor(g, ["v", "transmission"], record=True)
    spikes = b.SpikeMonitor(g)
    return lower_network(b.Network(g, s, generator, input_syn, monitor, spikes),
                         17*clock.dt, rng_seed=123), path


def test_readonly_pre_state_proof_covers_other_projections(device, tmp_path):
    model, _ = binary_model(tmp_path, mutate_transmission=True)
    with pytest.raises(PlanValidationError, match="presynaptic state proven unwritten"):
        build_distributed_plan(model)


@real_mpi
@pytest.mark.parametrize("ranks", [1, 2, 4])
def test_binary_csr_readonly_pre_state_and_zero_delay_match_reference(device, tmp_path, ranks):
    model, graph = binary_model(tmp_path)
    expected = reference(model, tmp_path / "reference")
    project, result = tmp_path / "mpi", tmp_path / "result"
    plan = write_mpi_project(model, project, ranks=ranks)
    assert plan.readonly_pre_states
    assert sum(sum(shard.incoming_edges) for shard in plan.shards) == 9
    if ranks > 1:
        assert plan.min_cross_rank_delay_ticks == 0
    compile_mpi_project(project)
    # Runtime is self-contained after export; no external CSR is consulted.
    graph.rename(graph.with_suffix(".saved"))
    run_mpi_project(project, result)
    assert_same_result(expected, load_results(model, result))
    for name in ("results.bin", "events.bin"):
        assert (result/name).read_bytes() == (tmp_path/"reference"/name).read_bytes()

    # Same-size edits must be checked inside the streaming MPI reader itself.
    corrupt = tmp_path / "corrupt.bin"
    payload = bytearray((project/"instance.bin").read_bytes())
    payload[48] ^= 1
    corrupt.write_bytes(payload)
    process = subprocess.run(["mpiexec", "-n", str(ranks), str(project/"b2-mpi"),
                              str(corrupt), str(tmp_path/"corrupt-output")],
                             capture_output=True, text=True, timeout=30)
    assert process.returncode != 0
    assert "MPI instance differs from compiled plan" in process.stderr


@real_mpi
@pytest.mark.parametrize("ranks", [1, 2, 4])
@pytest.mark.parametrize("stochastic", [False, True])
def test_recurrent_delays_match_reference(device, tmp_path, ranks, stochastic):
    model = network_model(stochastic=stochastic)
    expected = reference(model, tmp_path / "reference")
    plan = write_mpi_project(model, tmp_path / "mpi", ranks=ranks)
    compile_mpi_project(tmp_path / "mpi")
    observed = run_mpi_project(tmp_path / "mpi", tmp_path / "result")
    assert_same_result(expected, load_results(model, tmp_path / "result"))
    work = np.array(observed["rank_work"]).reshape(ranks, 3)
    assert work[:, 0].sum() == 10
    assert work[:, 2].sum() == expected["metadata"]["synaptic_events"]
    assert len(plan.exchange_nodes) == 2
    assert observed["exchange_calls_per_rank"] == 17
    assert observed["spike_exchange_strategy"] == "consecutive-producers-same-clock"
    assert work[:, 1].sum() == sum(len(p["event_streams"]["spike"]["ticks"]) for p in expected["populations"])


@real_mpi
@pytest.mark.parametrize("empty,monitor", [(True, True), (False, False)])
def test_empty_projection_and_no_monitor(device, tmp_path, empty, monitor):
    model = network_model(empty=empty, monitor=monitor)
    expected = reference(model, tmp_path / "reference")
    write_mpi_project(model, tmp_path / "mpi", ranks=4)
    compile_mpi_project(tmp_path / "mpi")
    run_mpi_project(tmp_path / "mpi", tmp_path / "result")
    assert_same_result(expected, load_results(model, tmp_path / "result"))


@real_mpi
@pytest.mark.parametrize("generator", [True, False])
def test_generator_subgroup_order_and_quiet_rank(device, tmp_path, generator):
    clock = b.Clock(dt=b.second/1024)
    if generator:
        source = b.SpikeGeneratorGroup(5, [4, 1, 3, 0, 2], [0, 0, 0, 0, 0]*clock.dt, clock=clock)
        endpoint, indices = source, [4, 1, 3, 1]
    else:
        source = b.NeuronGroup(5, "v:1", threshold="t < dt", reset="v=0", clock=clock)
        endpoint, indices = source[1:5], [3, 0, 2, 0]
    target = b.NeuronGroup(3, "v:1", clock=clock)
    syn = b.Synapses(endpoint, target[1:3], "w:1 (constant)", on_pre="v_post+=w", clock=clock)
    syn.connect(i=indices, j=[0, 0, 0, 1])
    syn.w = [-1e16, 1e16, 1, -0.25]
    syn.delay = clock.dt
    state = b.StateMonitor(target, "v", record=True)
    model = lower_network(b.Network(source, target, syn, state), 3*clock.dt)
    expected = reference(model, tmp_path / "reference")
    write_mpi_project(model, tmp_path / "mpi", ranks=4)
    compile_mpi_project(tmp_path / "mpi")
    run_mpi_project(tmp_path / "mpi", tmp_path / "result")
    assert_same_result(expected, load_results(model, tmp_path / "result"))


@real_mpi
def test_mpi_wrong_world_and_one_rank_input_failure_abort(device, tmp_path):
    model = network_model()
    write_mpi_project(model, tmp_path / "mpi", ranks=2)
    executable = compile_mpi_project(tmp_path / "mpi")
    instance = tmp_path / "mpi/instance.bin"
    bad = tmp_path / "bad.bin"
    data = bytearray(instance.read_bytes()); data[-1] ^= 1; bad.write_bytes(data)
    for command, message in [
        (["mpiexec", "-n", "1", str(executable), str(instance), str(tmp_path / "wrong-world")], "world size"),
        (["mpiexec", "-n", "1", str(executable), str(instance), str(tmp_path / "bad-result"),
          ":", "-n", "1", str(executable), str(bad), str(tmp_path / "bad-result")], "instance differs")]:
        failed = subprocess.run(command, capture_output=True, text=True, timeout=20)
        assert failed.returncode != 0
        assert message in failed.stderr
    assert not (tmp_path / "bad-result").exists()
    instance.write_bytes(bad.read_bytes())
    with pytest.raises(ValueError, match="artifact changed"):
        run_mpi_project(tmp_path / "mpi", tmp_path / "result")


@real_mpi
def test_device_mpi_results_and_continuation(device, tmp_path):
    b.set_device("rust_standalone", engine="mpi", ranks=4, directory=tmp_path / "device", runner=RUNNER)
    clock = b.Clock(dt=b.second/1024)
    pop = b.NeuronGroup(5, "dv/dt=512*Hz:1", threshold="v>1", reset="v=rand()", clock=clock, method="euler")
    syn = b.Synapses(pop, pop, "w:1 (constant)", on_pre="v_post += w", clock=clock)
    syn.connect(i=[4, 0, 1, 3], j=[0, 4, 3, 1]); syn.w=0.25; syn.delay=2*clock.dt
    monitor = b.StateMonitor(pop, "v", record=True)
    spikes = b.SpikeMonitor(pop)
    net = b.Network(pop, syn, monitor, spikes)
    net.run(9*clock.dt)
    model = json.loads((tmp_path / "device/model.json").read_text())
    expected = reference(model, tmp_path / "reference")
    actual = load_results(model, tmp_path / "device/rust")
    assert_same_result(expected, actual)
    np.testing.assert_array_equal(pop.v[:], expected["populations"][0]["states"]["v"])
    np.testing.assert_array_equal(monitor.v[:].T, expected["populations"][0]["trace"]["v"])
    report = device.explain_plan(format="dict")
    assert report["runtime_binding"]["backend"] == "mpi"
    assert report["runtime_binding"]["observed"]["ranks"] == 4
    net.run(clock.dt)
    assert net.t == 10 * clock.dt
    assert (tmp_path / "device/run-0002").exists()


def test_reject_cross_population_clock_and_functions(device):
    clock1, clock2 = b.Clock(dt=b.ms), b.Clock(dt=2*b.ms)
    a = b.NeuronGroup(2, "v:1", clock=clock1)
    c = b.NeuronGroup(2, "v:1", clock=clock2)
    with pytest.raises(PlanValidationError, match="shared clock"):
        build_distributed_plan(lower_network(b.Network(a, c), 4*b.ms))


def test_cpu_mpi_options_do_not_leak_to_other_backends(device):
    with pytest.raises(PlanValidationError, match="ranks requires"):
        build_execution_plan(network_model(), ranks=2)
    with pytest.raises(PlanValidationError, match="reference-f64"):
        build_execution_plan(network_model(), backend="mpi", numeric_mode="float32")


@real_mpi
def test_completely_empty_ranks_and_multiple_pathways(device, tmp_path):
    clock = b.Clock(dt=b.second/1024)
    pop = b.NeuronGroup(2, "v:1\nx:1", threshold="t==0*second", reset="v=0", clock=clock)
    pop.x = -0.0
    syn = b.Synapses(pop, pop, on_pre={"first": "x_post+=1", "second": "x_post*=2"}, clock=clock)
    syn.connect(i=[1, 0, 0], j=[0, 1, 1])
    syn.first.delay = clock.dt
    syn.second.delay = 2*clock.dt
    monitor = b.StateMonitor(pop, "x", record=[1, 0])
    model = lower_network(b.Network(pop, syn, monitor), 4*clock.dt)
    expected = reference(model, tmp_path / "reference")
    write_mpi_project(model, tmp_path / "mpi", ranks=4)
    compile_mpi_project(tmp_path / "mpi")
    observed = run_mpi_project(tmp_path / "mpi", tmp_path / "result")
    assert_same_result(expected, load_results(model, tmp_path / "result"))
    work = np.array(observed["rank_work"]).reshape(4, 3)
    assert np.count_nonzero(work[:, 0] == 0) == 2
    assert not np.any(work[work[:, 0] == 0, :])


@real_mpi
def test_mixed_plans_abort_before_simulation_and_launcher_timeout(device, tmp_path):
    model = network_model()
    write_mpi_project(model, tmp_path / "first", ranks=2)
    a = compile_mpi_project(tmp_path / "first")
    changed = copy.deepcopy(model)
    changed["instance"]["rng_seed"] += 1
    attach_protocol(changed)
    write_mpi_project(changed, tmp_path / "second", ranks=2)
    c = compile_mpi_project(tmp_path / "second")
    command = ["mpiexec", "-n", "1", str(a), str(tmp_path / "first/instance.bin"), str(tmp_path / "out"),
               ":", "-n", "1", str(c), str(tmp_path / "second/instance.bin"), str(tmp_path / "out")]
    result = subprocess.run(command, capture_output=True, text=True, timeout=20)
    assert result.returncode != 0 and "different compiled plans" in result.stderr
    assert not (tmp_path / "out").exists()
    launcher = tmp_path / "slow-launcher"
    launcher.write_text("#!/bin/sh\nprintf 'started\\n'\nexec sleep 60\n")
    launcher.chmod(0o755)
    with pytest.raises(subprocess.TimeoutExpired):
        # Include startup time on a loaded host while remaining well below
        # the 60-second workload; exercise capture of output before timeout.
        run_mpi_project(tmp_path / "first", tmp_path / "out", mpiexec=launcher, timeout=2)
    assert not (tmp_path / "out").exists()
    assert "started" in (tmp_path / "first/launch.log").read_text()

@real_mpi
def test_each_rank_reads_only_its_shard_and_rejects_remote_corruption(device, tmp_path):
    model, _ = binary_model(tmp_path)
    expected = reference(model, tmp_path / 'reference')
    project = tmp_path / 'project'
    write_mpi_project(model, project, ranks=2)
    compile_mpi_project(project)
    roots = []
    for rank in range(2):
        root = tmp_path / f'node-{rank}'
        root.mkdir()
        shutil.copyfile(project / 'instance.bin', root / 'instance.bin')
        shutil.copyfile(project / f'instance.rank-{rank}.bin', root / f'instance.rank-{rank}.bin')
        roots.append(root)
    def command(output):
        return ['mpiexec', '-n', '1', str(project/'b2-mpi'), str(roots[0]/'instance.bin'), str(output),
                ':', '-n', '1', str(project/'b2-mpi'), str(roots[1]/'instance.bin'), str(output)]
    result = tmp_path / 'result'
    subprocess.run(command(result), check=True, capture_output=True, timeout=30)
    assert_same_result(expected, load_results(model, result))
    # Corrupt the other rank's finite initial state with the same file length.
    shard = roots[1] / 'instance.rank-1.bin'
    payload = bytearray(shard.read_bytes()); payload[48] ^= 1; shard.write_bytes(payload)
    process = subprocess.run(command(tmp_path/'corrupt-output'), capture_output=True, text=True, timeout=30)
    assert process.returncode != 0
    assert 'MPI instance differs from compiled plan' in process.stderr


@real_mpi
def test_rank_local_neuron_rng_parameters_and_empty_owners(device, tmp_path):
    clock = b.Clock(dt=b.second/1024)
    g = b.NeuronGroup(3, 'dv/dt=drive*1024*Hz:1\ndrive:1 (constant)',
                     threshold='v>1', reset='v=rand()', method='euler', clock=clock)
    g.drive = [0.125, 0.25, 0.5]
    mon = b.StateMonitor(g, 'v', record=[2, 0, 1])
    spikes = b.SpikeMonitor(g)
    model = lower_network(b.Network(g, mon, spikes), 13*clock.dt, rng_seed=321)
    expected = reference(model, tmp_path/'reference')
    write_mpi_project(model, tmp_path/'project', ranks=4)
    compile_mpi_project(tmp_path/'project')
    run_mpi_project(tmp_path/'project', tmp_path/'result')
    assert_same_result(expected, load_results(model, tmp_path/'result'))


def test_streaming_sha256_matches_independent_hashlib(tmp_path):
    import hashlib
    implementation = ROOT / 'python/brian2_rust/mpi_runtime/sha256.rs'
    main = tmp_path/'digest.rs'
    main.write_text(implementation.read_text() + r'''
fn main() {
    for n in [0usize,3,55,56,63,64,65,128,65536,1000000] {
        let bytes: Vec<u8>=(0..n).map(|i|(i%251) as u8).collect();
        for chunk in [1usize,7,64,65,65536] {
            let mut sha=Sha256::new(); for part in bytes.chunks(chunk) { sha.update(part); }
            let hex: String=sha.finish().iter().map(|b|format!("{b:02x}")).collect();
            println!("{n} {chunk} {hex}");
        }
    }
}
''')
    subprocess.run(['rustc','--edition=2021','-O',str(main),'-o',str(tmp_path/'digest')],check=True,capture_output=True)
    result=subprocess.run([str(tmp_path/'digest')],check=True,capture_output=True,text=True)
    for line in result.stdout.splitlines():
        n, chunk, digest=line.split()
        assert digest==hashlib.sha256(bytes(i%251 for i in range(int(n)))).hexdigest()


@real_mpi
def test_node_local_rank_counts_same_processor_names(tmp_path):
    main = tmp_path/'placement.c'
    main.write_text('''#include <stdio.h>
int b2mpi_init(int *, int *);
int b2mpi_local_rank(void);
int b2mpi_finalize(void);
int main(void) {
    int rank, size;
    if (b2mpi_init(&rank, &size)) return 1;
    printf("%d %d\\n", rank, b2mpi_local_rank());
    return b2mpi_finalize();
}
''')
    binary = tmp_path/'placement'
    subprocess.run(['mpicc','-std=c11',str(main),str(ROOT/'python/brian2_rust/mpi_runtime/bridge.c'),'-o',str(binary)],check=True,capture_output=True)
    # Hydra can keep forwarding an inherited open stdin after this tiny
    # application has exited. This noninteractive launcher needs no input.
    result = subprocess.run(['mpiexec','-n','3',str(binary)],stdin=subprocess.DEVNULL,
                            check=True,capture_output=True,text=True,timeout=30)
    assert sorted(tuple(map(int,line.split())) for line in result.stdout.splitlines()) == [(0,0),(1,1),(2,2)]


def procedural_model(distribution='normal', edges=65539):
    clock = b.Clock(dt=b.second/1024)
    source = b.NeuronGroup(7, 'v:1\ngain:1', threshold='timestep(t, dt) % 2 == 0', reset='v=0', clock=clock)
    target = b.NeuronGroup(8, 'x:1', clock=clock)
    source.gain = np.arange(7)*0.125
    syn = b.Synapses(source[1:6], target[2:5], 'w:1 (constant)',
                     on_pre='x_post += w * gain_pre * (0.75 + 0.5*rand())', clock=clock)
    if distribution == 'normal':
        weight = brian2_rust.ClippedNormal(0.125,0.03125,minimum=0.0,maximum=0.25)
        delay = brian2_rust.ClippedNormal(2*clock.dt,0.5*clock.dt,minimum=0*clock.dt,maximum=4*clock.dt)
    else:
        weight = brian2_rust.Uniform(-0.125,0.25)
        delay = brian2_rust.Uniform(0*clock.dt,4*clock.dt)
    brian2_rust.connect_fixed_total(syn,edges,seed=0xfedcba9876543210,
                                    initializers={'w':weight},delay_initializer=delay)
    monitor = b.StateMonitor(target,'x',record=True)
    return lower_network(b.Network(source,target,syn,monitor),7*clock.dt,rng_seed=999)


@real_mpi
@pytest.mark.parametrize('ranks',[1,2,4])
@pytest.mark.parametrize('distribution',['normal','uniform'])
def test_distributed_fixed_total_recipe_matches_reference(device,tmp_path,ranks,distribution):
    model = procedural_model(distribution)
    expected = reference(model,tmp_path/'reference')
    project = tmp_path/'mpi'
    plan = write_mpi_project(model,project,ranks=ranks)
    assert plan.procedural_projections == (0,)
    assert all(s.incoming_edges == (None,) for s in plan.shards)
    assert plan.min_cross_rank_delay_ticks is None
    assert max(p.stat().st_size for p in project.glob('instance.rank-*.bin')) < 2048
    compile_mpi_project(project)
    observed = run_mpi_project(project,tmp_path/'out')
    assert_same_result(expected,load_results(model,tmp_path/'out'))
    stats = np.array(observed['procedural_topology'][0]['rank_stats']).reshape(ranks,2)
    assert stats[:,0].sum() == stats[:,1].sum() == 65539
    assert stats[:,1].max()-stats[:,1].min() <= 1
    if ranks == 4:
        assert list(stats[:,0] == 0) == [True,False,False,True]


@real_mpi
def test_procedural_budget_aborts_before_simulation(device,tmp_path,monkeypatch):
    model = procedural_model(edges=101)
    write_mpi_project(model,tmp_path/'mpi',ranks=4)
    compile_mpi_project(tmp_path/'mpi')
    # The global total fits 4*40, but owner rank 1 receives about two thirds.
    monkeypatch.setenv('B2_MPI_MAX_LOCAL_EDGES','40')
    with pytest.raises(RuntimeError,match='edge budget exceeded'):
        run_mpi_project(tmp_path/'mpi',tmp_path/'out',timeout=20)
    assert not (tmp_path/'out').exists()


@real_mpi
def test_procedural_scalar_uniform_delay_and_cumulative_budget(device,tmp_path,monkeypatch):
    clock=b.Clock(dt=b.second/1024)
    pop=b.NeuronGroup(2,'v:1',threshold='t==0*second',reset='v=0',clock=clock)
    objects=[pop,b.StateMonitor(pop,'v',record=True)]
    for seed in [17,19]:
        syn=b.Synapses(pop,pop,'w:1 (constant, shared)',on_pre='v_post+=w*(0.5+rand())',clock=clock)
        brian2_rust.connect_fixed_total(syn,1,seed=seed)
        syn.w=0.125;syn.delay=clock.dt;objects.append(syn)
    model=lower_network(b.Network(*objects),3*clock.dt)
    expected=reference(model,tmp_path/'reference')
    write_mpi_project(model,tmp_path/'mpi',ranks=4)
    compile_mpi_project(tmp_path/'mpi')
    run_mpi_project(tmp_path/'mpi',tmp_path/'out')
    assert_same_result(expected,load_results(model,tmp_path/'out'))
    # On one rank, each projection separately fits, but their cumulative total does not.
    write_mpi_project(model,tmp_path/'one',ranks=1)
    compile_mpi_project(tmp_path/'one')
    monkeypatch.setenv('B2_MPI_MAX_LOCAL_EDGES','1')
    with pytest.raises(RuntimeError,match='edge budget exceeded'):
        run_mpi_project(tmp_path/'one',tmp_path/'budget-out')
    assert not (tmp_path/'budget-out').exists()


@pytest.mark.parametrize('owners',[(0,), (0,4), (False,1), (-1,0), ('0',1)])
def test_population_ownership_rejects_invalid_mapping(device,owners):
    with pytest.raises(PlanValidationError,match='population_owners'):
        build_distributed_plan(network_model(),ranks=4,population_owners=owners)


@real_mpi
@pytest.mark.parametrize('kind',['explicit','binary','procedural'])
@pytest.mark.parametrize('owners',[(3,1),(None,2),(2,None)])
def test_population_mapping_reference_parity(device,tmp_path,kind,owners):
    model = (network_model(stochastic=True) if kind=='explicit' else
             binary_model(tmp_path)[0] if kind=='binary' else procedural_model(edges=32771))
    assert len(model['definition']['populations']) == len(owners)
    expected=reference(model,tmp_path/'reference')
    plan=build_distributed_plan(model,ranks=4,population_owners=owners)
    verify_distributed_plan(plan,model)
    with pytest.raises(PlanValidationError,match='differs'):
        verify_distributed_plan(replace(plan,population_owners=(0,0)),model)
    project=tmp_path/'mpi'
    write_mpi_project(model,project,ranks=4,plan=plan)
    compile_mpi_project(project)
    report=run_mpi_project(project,tmp_path/'out')
    assert_same_result(expected,load_results(model,tmp_path/'out'))
    for name in ['results.bin','events.bin']:
        if (tmp_path/'reference'/name).exists():
            assert (tmp_path/'reference'/name).read_bytes()==(tmp_path/'out'/name).read_bytes()
    work=np.array(report['rank_work']).reshape(4,3)
    assert list(work[:,0])==[sum(hi-lo for lo,hi in shard.population_ranges) for shard in plan.shards]
    for q, syn in enumerate(model['definition']['synapses']):
        owner=owners[syn['target_population']]
        if owner is not None and kind=='procedural':
            stats=np.array(report['procedural_topology'][q]['rank_stats']).reshape(4,2)
            assert stats[owner,0]==32771
            assert np.count_nonzero(stats[:,0])==1
            np.testing.assert_array_equal(stats[:,0],stats[:,1])
            assert report["procedural_topology"][q]["construction"]=="target-owner-local"
            csr=report['procedural_topology'][q]['rank_csr_offset_bytes']
            assert all(size == 0 for rank,size in enumerate(csr) if rank != owner)
            assert csr[owner] == (syn['source_count']+1)*8


@real_mpi
@pytest.mark.parametrize('owners',[(None,), (3,)])
@pytest.mark.parametrize('panic_strategy',[None,'abort'])
def test_mapped_direct_poisson_input_matches_reference(device,tmp_path,owners,panic_strategy):
    clock=b.Clock(dt=0.1*b.ms)
    pop=b.NeuronGroup(19,'dv/dt=(-v+current)/(10*ms):1 (unless refractory)\ndcurrent/dt=-current/(0.5*ms):1',
                      threshold='v>1',reset='v=rand()*0.1',refractory=0.2*b.ms,method='euler',clock=clock)
    pop.run_regularly('current += 5*poisson(2.5)',when='synapses')
    model=lower_network(b.Network(pop,b.SpikeMonitor(pop),b.StateMonitor(pop,['v','current'],record=True)),20*clock.dt,rng_seed=71)
    expected=reference(model,tmp_path/'reference')
    write_mpi_project(model,tmp_path/'mpi',ranks=4,population_owners=owners)
    compile_mpi_project(tmp_path/'mpi',opt_level=1,panic_strategy=panic_strategy)
    build=json.loads((tmp_path/'mpi/build.json').read_text())
    assert build['opt_level']==1 and build.get('panic')==panic_strategy
    run_mpi_project(tmp_path/'mpi',tmp_path/'out')
    assert_same_result(expected,load_results(model,tmp_path/'out'))


@real_mpi
@pytest.mark.parametrize('ranks',[1,2,4])
def test_brian_poisson_input_matches_reference(device,tmp_path,ranks):
    """PoissonInput is a canonical population runner, including under sharding."""
    clock=b.Clock(dt=0.1*b.ms)
    pop=b.NeuronGroup(19,'x:1',clock=clock)
    stimulus=b.PoissonInput(pop,'x',N=3,rate=2.5*b.kHz,weight=0.125)
    state=b.StateMonitor(pop,'x',record=True)
    model=lower_network(b.Network(pop,stimulus,state),20*clock.dt,rng_seed=71)
    code=next(c for c in model['definition']['populations'][0]['code_objects']
              if c['kind']=='poisson_input')
    assert code['when']=='synapses'
    expected=reference(model,tmp_path/'reference')
    write_mpi_project(model,tmp_path/'mpi',ranks=ranks)
    compile_mpi_project(tmp_path/'mpi',opt_level=1)
    run_mpi_project(tmp_path/'mpi',tmp_path/'out')
    assert_same_result(expected,load_results(model,tmp_path/'out'))
    for name in ['results.bin','events.bin']:
        if (tmp_path/'reference'/name).exists():
            assert (tmp_path/'reference'/name).read_bytes()==(tmp_path/'out'/name).read_bytes()


def test_invalid_panic_strategy_is_rejected_before_compilation(tmp_path):
    with pytest.raises(ValueError,match='panic_strategy'):
        compile_mpi_project(tmp_path/'missing',panic_strategy='silent')


@real_mpi
@pytest.mark.parametrize('uniform',[True,False])
@pytest.mark.parametrize('topology',['explicit','procedural'])
def test_shared_additive_kernel_preserves_queue_order(device,tmp_path,uniform,topology):
    clock=b.Clock(dt=b.second/1024)
    source=b.NeuronGroup(5,'v:1',threshold='timestep(t,dt)%2==0',reset='v=0',clock=clock)
    target=b.NeuronGroup(7,'x:1',clock=clock)
    syn=b.Synapses(source[1:4],target[2:6],'w:1 (constant)',on_pre='x_post+=w',clock=clock)
    if topology=='procedural':
        delay=None if uniform else brian2_rust.Uniform(0*clock.dt,4*clock.dt)
        brian2_rust.connect_fixed_total(syn,257,seed=98,initializers={'w':brian2_rust.Uniform(-1,1)},delay_initializer=delay)
        if uniform:syn.delay=2*clock.dt
    else:
        syn.connect(i=[2,0,1,0,2,0,1],j=[3,0,1,0,3,0,1])
        syn.w=[1e12,0.125,-1e12,1e12,0.25,-1e12,0.5]
        syn.delay=2*clock.dt if uniform else np.array([0,3,1,2,0,1,2])*clock.dt
    scalar=b.Synapses(source,target,'gain:1 (constant, shared)',on_pre='x_post+=gain',clock=clock)
    scalar.connect(i=[4,0],j=[6,0]);scalar.gain=0.0625;scalar.delay=clock.dt
    model=lower_network(b.Network(source,target,syn,scalar,b.StateMonitor(target,'x',record=True)),9*clock.dt)
    expected=reference(model,tmp_path/'reference')
    write_mpi_project(model,tmp_path/'mpi',ranks=4,population_owners=(3,1))
    emitted=(tmp_path/'mpi/main.rs').read_text()
    assert emitted.count('+= mpi_additive_projection(')==2
    compile_mpi_project(tmp_path/'mpi')
    run_mpi_project(tmp_path/'mpi',tmp_path/'out')
    assert_same_result(expected,load_results(model,tmp_path/'out'))


@real_mpi
def test_owner_local_interleaves_collectives_and_cumulative_budget(device,tmp_path,monkeypatch):
    clock=b.Clock(dt=b.second/1024)
    a=b.NeuronGroup(5,'x:1',threshold='timestep(t,dt)%2==0',reset='x=0',clock=clock,name='owned_a')
    c=b.NeuronGroup(7,'x:1',clock=clock,name='split_b')
    objects=[a,c,b.StateMonitor(a,'x',record=True),b.StateMonitor(c,'x',record=True)]
    for q,(target,count) in enumerate([(a,31),(c,13),(a,37)]):
        syn=b.Synapses(a,target,'w:1 (constant)',on_pre='x_post+=w*(0.5+rand())',clock=clock,name=f'interleaved_{q}')
        brian2_rust.connect_fixed_total(syn,count,seed=17+q,initializers={'w':brian2_rust.Uniform(-1,1)})
        syn.delay=clock.dt;objects.append(syn)
    model=lower_network(b.Network(*objects),5*clock.dt,rng_seed=51)
    expected=reference(model,tmp_path/'reference')
    write_mpi_project(model,tmp_path/'mpi',ranks=4,population_owners=(1,None))
    compile_mpi_project(tmp_path/'mpi')
    observed=run_mpi_project(tmp_path/'mpi',tmp_path/'out')
    assert_same_result(expected,load_results(model,tmp_path/'out'))
    assert [x['construction'] for x in observed['procedural_topology']]==['target-owner-local','distributed-draw-ranges','target-owner-local']
    monkeypatch.setenv('B2_MPI_MAX_LOCAL_EDGES','67')
    with pytest.raises(RuntimeError,match='edge budget exceeded'):
        run_mpi_project(tmp_path/'mpi',tmp_path/'budget',timeout=20)


@real_mpi
@pytest.mark.parametrize('boundary', ['none', 'monitor', 'pathway'])
def test_spike_batches_flush_before_consumers(device, tmp_path, boundary):
    clock = b.Clock(dt=b.ms, name='batch_clock')
    a = b.NeuronGroup(5, 'v:1', threshold='v>0.5', reset='v=0', clock=clock, name='batch_a')
    z = b.NeuronGroup(3, 'v:1', threshold='v>0.5', reset='v=0', clock=clock, name='batch_z')
    a.v = 1
    a.thresholder['spike'].order = 0
    z.thresholder['spike'].order = 10 if boundary == 'monitor' else 0
    s = b.Synapses(a, z, on_pre='v_post += 1', clock=clock, name='batch_synapse')
    s.connect(i=[0, 1, 2, 3, 4], j=[0, 1, 2, 0, 1])
    s.delay = 0*b.ms
    objects = [a, z, s]
    if boundary != 'pathway':
        objects.append(b.SpikeMonitor(a))
    model = lower_network(b.Network(*objects), b.ms)
    expected = reference(model, tmp_path/'reference')
    write_mpi_project(model, tmp_path/'mpi', ranks=4, population_owners=(3, 1))
    compile_mpi_project(tmp_path/'mpi')
    actual = run_mpi_project(tmp_path/'mpi', tmp_path/'result')
    assert_same_result(expected, load_results(model, tmp_path/'result'))
    assert actual['exchange_calls_per_rank'] == (2 if boundary == 'monitor' else 1)
    capacity = actual['rank_queue_capacity_bytes']
    assert len(capacity) == 4 and capacity[1] >= 5*8
    assert capacity[0] == capacity[2] == capacity[3] == 0
    # The target lives on a different rank and must receive these zero-delay
    # events this tick, including when no monitor triggers the exchange first.
    target_index = next(i for i, pop in enumerate(model['definition']['populations']) if pop['name'] == 'batch_z')
    target = expected['populations'][target_index]
    assert np.array_equal(target['states']['v'], [2, 2, 1])


@real_mpi
def test_sparse_rotating_owner_collection_completes_before_buffer_release(tmp_path):
    binary=tmp_path/'collect-probe'
    subprocess.run(['mpicc','-std=c11','-O2','-Wall','-Wextra','-Werror',
                    str(ROOT/'tools/mpi_collect_probe.c'),
                    str(ROOT/'python/brian2_rust/mpi_runtime/bridge.c'),'-o',str(binary)],
                   check=True,capture_output=True,text=True,timeout=60)
    with subprocess.Popen(['mpiexec','-n','4',str(binary)],
                          stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, text=True, start_new_session=True) as run:
        try:
            stdout, stderr=run.communicate(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(run.pid, signal.SIGKILL)
            run.communicate()
            raise
        assert run.returncode==0, stdout+stderr
    report=next(json.loads(line) for line in stdout.splitlines()
                if line.startswith('{"schema":"mpi-collect-probe-v1"'))
    assert report['ranks']==4 and report['bad']==0
    assert 'pending' not in stderr.lower()
