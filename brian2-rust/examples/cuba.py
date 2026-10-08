"""Two-population CUBA conformance: 80 Exc + 20 Inh, four projections."""

import argparse
import json
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

import brian2 as b
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import brian2_rust  # noqa: E402, F401


def make_network(refractory_ms=None):
    voltage_flag = " (unless refractory)" if refractory_ms is not None else ""
    options = {} if refractory_ms is None else {"refractory": refractory_ms*b.ms}
    model = f"""
        dv/dt = (drive - v + I_syn) / tau : 1{voltage_flag}
        dI_syn/dt = -I_syn / tau_syn : 1
        drive : 1 (constant)
        """
    exc = b.NeuronGroup(
        80, model,
        threshold="v > 1", reset="v = 0", method="euler", dt=0.1*b.ms,
        namespace={"tau": 10*b.ms, "tau_syn": 5*b.ms}, name="exc", **options,
    )
    inh = b.NeuronGroup(
        20, model,
        threshold="v > 1", reset="v = 0", method="euler", dt=0.1*b.ms,
        namespace={"tau": 10*b.ms, "tau_syn": 5*b.ms}, name="inh", **options,
    )
    initial_v = np.linspace(0, 0.9, 100)
    drive = np.linspace(1.1, 1.7, 100)
    exc.v, inh.v = initial_v[:80], initial_v[80:]
    exc.drive, inh.drive = drive[:80], drive[80:]

    # Explicit deterministic topology; edge order deliberately interleaves sources.
    offsets = np.array([1, 3, 7, 11, 17, 23, 31, 43])
    source = np.tile(np.arange(100), len(offsets))
    target = (source + np.repeat(offsets, 100)) % 100
    weight = np.where(source < 80, 0.04, -0.16) * (0.8 + 0.1*(target % 5))
    projections = []
    for source_kind, source_group, source_start in [("e", exc, 0), ("i", inh, 80)]:
        for target_kind, target_group, target_start in [("e", exc, 0), ("i", inh, 80)]:
            selected = ((source >= source_start) &
                        (source < source_start + len(source_group)) &
                        (target >= target_start) &
                        (target < target_start + len(target_group)))
            projection = b.Synapses(
                source_group, target_group, "w : 1 (constant)",
                on_pre="I_syn_post += w", delay=0.3*b.ms,
                clock=source_group.clock,
                name=f"connections_{source_kind}{target_kind}")
            projection.connect(i=source[selected] - source_start,
                               j=target[selected] - target_start)
            projection.w = weight[selected]
            projections.append(projection)
    exc_state = b.StateMonitor(exc, ["v", "I_syn"], record=True, name="exc_state")
    inh_state = b.StateMonitor(inh, ["v", "I_syn"], record=True, name="inh_state")
    exc_spikes = b.SpikeMonitor(exc, name="exc_spikes")
    inh_spikes = b.SpikeMonitor(inh, name="inh_spikes")
    network = b.Network(exc, inh, *projections, exc_state, inh_state,
                        exc_spikes, inh_spikes)
    topology = {"source": source, "target": target, "weight": weight}
    return (network, (exc, inh), projections, (exc_state, inh_state),
            (exc_spikes, inh_spikes), topology)


def run_backend(backend, output, refractory_ms=None):
    output.mkdir(parents=True, exist_ok=False)
    if backend in {"aot", "reference"}:
        b.set_device("rust_standalone", directory=output / "project", engine=backend)
    elif backend == "cpp":
        if sys.platform == "win32":
            b.prefs.codegen.cpp.extra_compile_args = ["/O2", "/fp:strict", "/std:c++17"]
        else:
            b.prefs.codegen.cpp.extra_compile_args = ["-O2", "-std=c++17", "-fno-fast-math", "-ffp-contract=off"]
            b.prefs.devices.cpp_standalone.extra_make_args_unix = ["-j2"]
        b.prefs.devices.cpp_standalone.openmp_threads = 0
        b.set_device("cpp_standalone", directory=str(output / "project"))
    else:
        b.set_device("runtime")
        b.prefs.codegen.target = "numpy"
    network, groups, projections, states, spikes, topology = make_network(refractory_ms)
    network.run(100*b.ms)
    exc, inh = groups
    exc_state, inh_state = states
    exc_spikes, inh_spikes = spikes
    spike_ticks = np.concatenate([
        np.rint(exc_spikes.t / exc.clock.dt).astype(np.int64),
        np.rint(inh_spikes.t / inh.clock.dt).astype(np.int64),
    ])
    spike_i = np.concatenate([np.asarray(exc_spikes.i[:]),
                              np.asarray(inh_spikes.i[:]) + len(exc)])
    order = np.lexsort((spike_i, spike_ticks))
    spike_ticks, spike_i = spike_ticks[order], spike_i[order]
    spike_t = spike_ticks * float(exc.clock.dt / b.second)
    delay_ticks = int(float(projections[0].delay[:] / exc.clock.dt) + 0.5)
    outdegree = np.bincount(topology["source"], minlength=100)
    delivered = int(np.sum(outdegree[spike_i[spike_ticks + delay_ticks < len(exc_state.t)]]))
    refractory_results = {} if refractory_ms is None else {
        "lastspike": np.concatenate([exc.lastspike[:]/b.second,
                                      inh.lastspike[:]/b.second]),
        "not_refractory": np.concatenate([exc.not_refractory[:],
                                           inh.not_refractory[:]]),
    }
    np.savez(
        output / "results.npz",
        v=np.concatenate([exc_state.v, inh_state.v]),
        I_syn=np.concatenate([exc_state.I_syn, inh_state.I_syn]),
        t=exc_state.t / b.second, spike_i=spike_i, spike_t=spike_t,
        spike_tick=spike_ticks,
        count=np.concatenate([exc_spikes.count[:], inh_spikes.count[:]]),
        final_v=np.concatenate([exc.v[:], inh.v[:]]),
        final_I_syn=np.concatenate([exc.I_syn[:], inh.I_syn[:]]),
        source=topology["source"], target=topology["target"], weight=topology["weight"],
        network_t=float(network.t / b.second),
        clock_t=np.array([float(exc.clock.t / b.second), float(inh.clock.t / b.second)]),
        **refractory_results,
    )
    if backend in {"aot", "reference"}:
        summary = json.loads((output / "project/rust/summary.json").read_text())
        if summary["synaptic_events"] != delivered:
            raise AssertionError("Rust synaptic delivery count differs from spike/topology expectation")
    metadata = {
        "backend": backend, "brian2": b.__version__, "python": platform.python_version(),
        "platform": platform.platform(), "neurons": len(exc) + len(inh),
        "populations": {"exc": len(exc), "inh": len(inh)},
        "synapses": sum(len(projection) for projection in projections),
        "projections": {projection.name: len(projection) for projection in projections},
        "steps": len(exc_state.t),
        "spikes": int(exc_spikes.num_spikes + inh_spikes.num_spikes),
        "synaptic_events": delivered,
        "delay_ticks": delay_ticks,
        "refractory_ms": refractory_ms,
        "cpp_flags": b.prefs.codegen.cpp.extra_compile_args if backend == "cpp" else None,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"{backend}: {metadata['spikes']} spikes, {delivered} synaptic deliveries", flush=True)


def compare(output, refractory_ms=None):
    report = {"neurons": 100, "synapses": 800, "states": ["v", "I_syn"], "steps": 1000,
              "delay_ticks": 3, "state_tolerance": {"rtol": 1e-12, "atol": 1e-14},
              "comparisons": {}}
    report["refractory_ms"] = refractory_ms
    with np.load(output / "aot/results.npz") as rust:
        for backend in ["reference", "numpy", "cpp"]:
            with np.load(output / backend / "results.npz") as reference:
                for name in ["v", "I_syn", "final_v", "final_I_syn"]:
                    np.testing.assert_allclose(rust[name], reference[name], rtol=1e-12, atol=1e-14)
                for name in ["spike_i", "spike_tick", "count", "source", "target", "weight"]:
                    np.testing.assert_array_equal(rust[name], reference[name])
                for name in ["t", "spike_t", "network_t", "clock_t"]:
                    np.testing.assert_allclose(rust[name], reference[name], rtol=0, atol=1e-15)
                if refractory_ms is not None:
                    np.testing.assert_allclose(rust["lastspike"], reference["lastspike"], rtol=0, atol=1e-15)
                    np.testing.assert_array_equal(rust["not_refractory"], reference["not_refractory"])
                metadata = json.loads((output / backend / "metadata.json").read_text())
                report["comparisons"][backend] = {
                    "max_abs_v_error": float(np.max(np.abs(rust["v"] - reference["v"]))),
                    "max_abs_I_syn_error": float(np.max(np.abs(rust["I_syn"] - reference["I_syn"]))),
                    "spikes": len(rust["spike_i"]), "synaptic_events": metadata["synaptic_events"],
                    "spike_order_equal": True, "passed": True,
                }
    (output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["all", "aot", "reference", "numpy", "cpp"],
                        default="all")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--refractory-ms", type=float,
                        help="fixed refractory duration in ms; freezes v and leaves I_syn active")
    args = parser.parse_args()
    if args.refractory_ms is not None and (not np.isfinite(args.refractory_ms) or args.refractory_ms < 0):
        parser.error("--refractory-ms must be finite and non-negative")
    if args.backend != "all":
        if args.output is None:
            parser.error("--output is required with an individual backend")
        run_backend(args.backend, args.output.resolve(), args.refractory_ms)
        return
    if args.output is None:
        (ROOT / "output").mkdir(exist_ok=True)
        output = Path(tempfile.mkdtemp(prefix="cuba-", dir=ROOT / "output"))
    else:
        output = args.output.resolve()
        output.mkdir(parents=True, exist_ok=False)
    for backend in ["aot", "reference", "numpy", "cpp"]:
        options = [] if args.refractory_ms is None else ["--refractory-ms", str(args.refractory_ms)]
        with (output / f"{backend}.log").open("w") as log:
            result = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--backend", backend,
                 "--output", str(output / backend), *options], stdout=log, stderr=subprocess.STDOUT,
            )
        if result.returncode:
            raise RuntimeError(f"{backend} failed; inspect {output / (backend + '.log')}")
        print(f"{backend} completed", flush=True)
    compare(output, args.refractory_ms)
    print(f"Artifacts: {output}")


if __name__ == "__main__":
    main()
