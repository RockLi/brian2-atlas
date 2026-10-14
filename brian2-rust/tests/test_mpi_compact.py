"""Opt-in MPI projection compaction: exact numeric order and artifact identity."""

import json
import subprocess
import pytest
from test_mpi import device as device, RUNNER, real_mpi
import brian2 as b
import brian2_rust
from brian2_rust.export import lower_network
from brian2_rust.distributed import (
    write_mpi_project,
    compile_mpi_project,
    run_mpi_project,
)
from brian2_rust.mpi_compact import compact_source


@real_mpi
@pytest.mark.usefixtures("device")
@pytest.mark.parametrize("populations", [False, True])
@pytest.mark.parametrize(
    "barrier", ["none", "nonadditive", "monitor", "clock", "delay", "weight"]
)
def test_compact_projection_order_and_layout(tmp_path, barrier, populations):
    clock = b.Clock(dt=b.ms, name="boundary_clock")
    other = b.Clock(dt=2 * b.ms, name="boundary_other_clock")
    a = b.NeuronGroup(
        5,
        "x:1",
        threshold="timestep(t,dt)%2==0",
        reset="x=0",
        clock=clock,
        name="boundary_a",
    )
    z = b.NeuronGroup(
        7,
        "x:1",
        threshold="timestep(t,dt)%3==0",
        reset="x=0",
        clock=clock,
        name="boundary_z",
    )
    extra = (
        b.NeuronGroup(
            5,
            "x:1",
            threshold="timestep(t,dt)%2==0",
            reset="x=0",
            clock=other,
            name="boundary_extra",
        )
        if barrier == "clock"
        else None
    )
    objects = [
        a,
        z,
        b.StateMonitor(a, "x", record=True),
        b.StateMonitor(z, "x", record=True),
    ]
    if extra is not None:
        objects.append(extra)
    for q in range(5):
        src, tgt = [(a, z[1:6]), (z, a), (a, a[1:4])][q % 3]
        middle = q == 2
        if middle and extra is not None:
            src = extra
        scalar = middle and barrier == "weight"
        syn = b.Synapses(
            src,
            tgt,
            "w:1 (constant, shared)" if scalar else "w:1 (constant)",
            on_pre="x_post*=w" if middle and barrier == "nonadditive" else "x_post+=w",
            clock=other if middle and barrier == "clock" else clock,
            name=f"boundary_s{q}",
        )
        kwargs = {} if scalar else {"initializers": {"w": brian2_rust.Uniform(-1, 1)}}
        if not (middle and barrier == "delay"):
            kwargs["delay_initializer"] = brian2_rust.Uniform(
                0 * clock.dt, 4 * clock.dt
            )
        brian2_rust.connect_fixed_total(syn, 101 + q, seed=91 + q, **kwargs)
        if scalar:
            syn.w = 0.25
        objects.append(syn)
    if barrier == "monitor":
        objects.append(
            b.StateMonitor(
                a,
                "x",
                record=True,
                when="synapses",
                order=-1,
                name="boundary_s1z_monitor",
            )
        )
    if barrier == "monitor":
        from brian2_rust.capabilities import CapabilityError

        with pytest.raises(CapabilityError, match="monitor.state.schedule"):
            lower_network(b.Network(*objects), 12 * clock.dt)
        return
    model = lower_network(b.Network(*objects), 12 * clock.dt)
    if barrier == "clock":
        from brian2_rust.plan import PlanValidationError

        with pytest.raises(PlanValidationError, match="only one shared clock"):
            write_mpi_project(model, tmp_path / "mpi", ranks=4)
        return
    (tmp_path / "model.json").write_text(json.dumps(model))
    write_mpi_project(model, tmp_path / "ordinary", ranks=4, population_owners=(3, 1))
    write_mpi_project(
        model,
        tmp_path / "mpi",
        ranks=4,
        population_owners=(3, 1),
        compact_projections=not populations,
        compact_populations=populations,
    )
    ordinary = json.loads((tmp_path / "ordinary/manifest.json").read_text())
    manifest = json.loads((tmp_path / "mpi/manifest.json").read_text())
    stats = manifest["projection_compaction"]
    if populations:
        aggregate = manifest["population_compaction"]
        assert aggregate["compacted"] == stats["compacted"]
        if aggregate["compacted"]:
            assert aggregate["population_structs"] == 2
            assert aggregate["collection_moves"] == 2
    assert ordinary["plan_sha256"] == manifest["plan_sha256"]
    for name, digest in ordinary["files"].items():
        if name != "main.rs":
            assert manifest["files"][name] == digest
    if barrier == "none":
        assert (stats["additive_groups"], stats["additive_nodes"]) == (1, 5)
    elif barrier == "nonadditive":
        assert (stats["additive_groups"], stats["additive_nodes"]) == (2, 4)
    else:
        assert not stats["compacted"]
        assert manifest["files"]["main.rs"] == ordinary["files"]["main.rs"]
    # A generator change must stop compaction, never emit a partial rewrite.
    if stats["compacted"]:
        original = (tmp_path / "ordinary/main.rs").read_text()
        with pytest.raises(ValueError, match="generated source differs"):
            compact_source(
                model,
                original.replace(
                    "MPI topology recipe mismatch", "changed reader check"
                ),
            )
    compile_mpi_project(tmp_path / "mpi", opt_level=1, panic_strategy="abort")
    subprocess.run(
        [str(RUNNER), str(tmp_path / "model.json"), str(tmp_path / "reference")],
        check=True,
        capture_output=True,
    )
    run_mpi_project(tmp_path / "mpi", tmp_path / "result", timeout=30)
    for name in ["results.bin", "events.bin"]:
        assert (tmp_path / "reference" / name).read_bytes() == (
            tmp_path / "result" / name
        ).read_bytes()
