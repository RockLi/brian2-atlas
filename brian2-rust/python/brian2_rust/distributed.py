"""Experimental MPI planning and AOT launch, outside the frozen B2IR ABI.

Partitions mutable state, incoming CSR and input files while retaining global
identities. Read-only presynaptic replicas and final rank-zero output remain.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess

from .plan import LogicalPlan, PlanValidationError, _derive_execution_plan, validate_model
from .protocol import canonical_bytes, migrate_model


@dataclass(frozen=True)
class RankShard:
    rank: int
    # Per-population half-open ranges; population offset + local index is global ID.
    population_ranges: tuple[tuple[int, int], ...]
    incoming_edges: tuple[int | None, ...]


@dataclass(frozen=True)
class DistributedPlan:
    schema: str
    definition_sha256: str
    instance_sha256: str
    run_sha256: str
    logical: LogicalPlan
    ranks: int
    shards: tuple[RankShard, ...]
    # (projection, source rank, destination rank, edge count), for explain only.
    routes: tuple[tuple[int, int, int, int], ...]
    min_cross_rank_delay_ticks: int | None
    # Logical producer obligations, not physical collective call counts.
    # Consecutive producers can share an exchange before the next consumer.
    exchange_nodes: tuple[str, ...]
    # State read by on_pre but proven unwritten throughout this activation.
    readonly_pre_states: tuple[tuple[int, str], ...] = ()
    # Procedural incoming counts are measured at runtime; routes remain unresolved.
    procedural_projections: tuple[int, ...] = ()
    # None splits a population; an integer assigns the entire population.
    population_owners: tuple[int | None, ...] = ()
    transport: str = "mpi-allgatherv-spikes"
    synchronization: str = "each-canonical-threshold-every-tick"
    ownership: str = "contiguous-per-population-target-owned"
    storage: str = "rank-local-neuron-and-mutable-synapse-state-incoming-csr-readonly-pre-replicas"
    numeric_profile: str = "reference-f64"
    rng_identity: str = "unchanged-global-population-index-and-edge-id"
    local_emitter: str = "canonical-slot-shared-additive-v4"

    # Empty preserves the original CPU plan identity and artifact format.
    rank_backends: tuple[str, ...] = ()

    def to_dict(self):
        result = asdict(self)
        if not self.rank_backends:
            result.pop("rank_backends")
        return result

    def to_json(self, *, indent=2):
        return json.dumps(self.to_dict(), sort_keys=True, indent=indent) + "\n"

    @property
    def sha256(self):
        return hashlib.sha256(canonical_bytes(self.to_dict())).hexdigest()

    @property
    def policy_sha256(self):
        # The executable binds the entire instance; replacement requires rebuild.
        return self.sha256


def _owner(index, count, ranks):
    return ((index + 1) * ranks - 1) // count


def _binary_routes(synapse, topology, pops, ranks, owners):
    """Count routes in bounded CSR slices, preserving original edge identities."""
    import numpy as np
    from .binary_topology import inspect_csr, csr_arrays, file_hash
    info = inspect_csr(topology["path"])
    if file_hash(info["path"]) != topology["sha256"]:
        raise PlanValidationError("MPI binary CSR changed after export")
    offsets, targets, _ = csr_arrays(info)
    source_count = pops[synapse["source_population"]]["count"]
    target_count = pops[synapse["target_population"]]["count"]
    routes = []
    for source_rank in range(ranks):
        lo, hi = _population_range(source_count, source_rank, ranks, owners[synapse["source_population"]])
        start = max(0, lo - synapse["source_start"])
        end = min(synapse["source_count"], hi - synapse["source_start"])
        if end <= start:
            continue
        counts = np.zeros(ranks, dtype=np.int64)
        stop = int(offsets[end])
        for first in range(int(offsets[start]), stop, 65536):
            target = targets[first:min(first + 65536, stop)].astype(np.int64)
            owner = ((target + synapse["target_start"] + 1) * ranks - 1) // target_count
            assigned = owners[synapse["target_population"]]
            if assigned is None:
                counts += np.bincount(owner, minlength=ranks)
            else:
                counts[assigned] += len(target)
        routes.extend((source_rank, target_rank, int(count))
                      for target_rank, count in enumerate(counts) if count)
    return routes


def _population_range(count, rank, ranks, owner):
    if owner is None:
        return count * rank // ranks, count * (rank + 1) // ranks
    return (0, count) if rank == owner else (0, 0)


def _explicit_routes(synapse, values, pops, ranks, owners, *, chunk_edges=1_048_576):
    """Count explicit edge routes in bounded vectorized chunks.

    Route counts and the minimum cross-rank delay are integer-only planning
    results.  Chunking avoids a Python iteration and delay-list allocation per
    edge while preserving the exact distributed-plan contract.
    """
    import numpy as np

    source_values, target_values = values["source"], values["target"]
    edge_count = len(source_values)
    if len(target_values) != edge_count:
        raise PlanValidationError("explicit source and target arrays differ in length")
    source_count = pops[synapse["source_population"]]["count"]
    target_count = pops[synapse["target_population"]]["count"]
    source_owner = owners[synapse["source_population"]]
    target_owner = owners[synapse["target_population"]]
    counts = np.zeros(ranks * ranks, dtype=np.int64)
    minimum_delay = None
    pathways = values["pathways"]
    delay_arrays = [path["delay_ticks"] for path in pathways]

    for first in range(0, edge_count, chunk_edges):
        last = min(first + chunk_edges, edge_count)
        source = np.asarray(source_values[first:last], dtype=np.int64)
        target = np.asarray(target_values[first:last], dtype=np.int64)
        if source_owner is None:
            source_rank = ((source + synapse["source_start"] + 1) * ranks - 1) // source_count
        else:
            source_rank = np.full(last - first, source_owner, dtype=np.int64)
        if target_owner is None:
            target_rank = ((target + synapse["target_start"] + 1) * ranks - 1) // target_count
        else:
            target_rank = np.full(last - first, target_owner, dtype=np.int64)
        counts += np.bincount(source_rank * ranks + target_rank,
                              minlength=ranks * ranks)
        cross = source_rank != target_rank
        if np.any(cross):
            for delays in delay_arrays:
                if len(delays) == 1:
                    candidate = delays[0]
                else:
                    candidate = int(np.min(np.asarray(delays[first:last], dtype=np.int64)[cross]))
                minimum_delay = candidate if minimum_delay is None else min(minimum_delay, candidate)

    routes = []
    incoming = [0] * ranks
    for pair, count in enumerate(counts):
        if count:
            source_rank, target_rank = divmod(pair, ranks)
            value = int(count)
            routes.append((source_rank, target_rank, value))
            incoming[target_rank] += value
    return routes, incoming, minimum_delay


def _derive_distributed_plan(model, ranks, population_owners=None, rank_backends=None, numeric_mode="reference-f64"):
    if type(ranks) is not int or not 1 <= ranks <= 256:
        raise PlanValidationError("MPI ranks must be an integer in 1..256")
    from .mpi_gpu import backend_policy
    backends = backend_policy(ranks, rank_backends, numeric_mode)
    model = migrate_model(model)
    d, inst = model["definition"], model["instance"]

    def require(condition, reason):
        if not condition:
            raise PlanValidationError(f"MPI v1: {reason}")

    require(len(d["clocks"]) == 1, "only one shared clock is supported")
    require(not d["functions"], "portable/native Functions are not yet supported")
    pops = d["populations"]
    owners = tuple([None] * len(pops) if population_owners is None else population_owners)
    require(len(owners) == len(pops) and all(o is None or type(o) is int and 0 <= o < ranks for o in owners),
            "population_owners must contain one None or valid rank per population")
    written = [set(name for code in p["code_objects"] for name in code["effects"]["writes"])
               for p in pops]
    for p, pop in enumerate(pops):
        if pop["refractory"] is not None:
            written[p].update(("lastspike", "not_refractory"))
    for synapse in d["synapses"]:
        for code in synapse["code_objects"]:
            for side, endpoint in (("pre", "source"), ("post", "target")):
                aliases = synapse[f"{side}_state_aliases"]
                written[synapse[f"{endpoint}_population"]].update(
                    aliases[name] for name in code["effects"]["writes"] if name in aliases)
    readonly_pre_states = set()
    for p in pops:
        require(not p.get("linked_variables"), "linked variables need cross-rank state exchange")
        require(not p.get("event_monitors"), "EventMonitor is not yet supported")
        require(set(p["events"]) <= {"spike"}, "only spike events are supported")
        require(all(s["dtype"] in {"bool", "f32", "f64", "i32", "i64", "u32", "u64"}
                    for s in p["states"] + p["parameters"]),
                "population storage dtype is unsupported")
        require(p["refractory"] is None or p["refractory"]["mode"] == "fixed",
                "only fixed refractory is supported")
        for code in p["code_objects"]:
            expected = {"state_update": "groups", "threshold": "thresholds", "reset": "resets",
                        "run_regularly": "synapses", "poisson_input": "synapses"}
            require(code["kind"] in expected and (code["when"] == expected[code["kind"]] or
                    code['kind'] == 'run_regularly' and code['when'] == 'after_groups'),
                    "population code must use a supported canonical slot")
    incoming = [[0] * len(d["synapses"]) for _ in range(ranks)]
    route_counts = {}
    min_cross_delay = None
    procedural = []
    for q, (syn, values) in enumerate(zip(d["synapses"], inst["synapses"], strict=True)):
        topology = values["topology"]
        require(topology["kind"] in {"explicit", "binary_csr", "fixed_total"},
                "topology must be explicit, binary CSR or fixed_total")
        require(all(s["dtype"] in {"bool", "f32", "f64", "i32", "i64", "u32", "u64"}
                    for s in syn["states"] + syn["parameters"]), "synapse storage dtype is unsupported")
        require(not {'live', 'born', 'active_after'} <= {s['name'] for s in syn['states']} or
                any(path['kind'] == 'pre' for path in values['pathways']),
                'bounded structural generation fields require a pre pathway')
        allowed = {"synapses": "synapses", "synapses_post": "synapses", "synapse_state_update": "groups",
                   "summed_variable": "groups"}
        require(all(c["kind"] in allowed and c["when"] == allowed[c["kind"]]
                    for c in syn["code_objects"]),
                "only standard pre/post, clock-driven state update and postsynaptic summed code are supported")
        require(not syn["states"] or topology["kind"] != "fixed_total",
                "mutable fixed_total synapses require distributed initializer support")
        require(topology['kind'] != 'fixed_total' or all(c['start_tick'] == 0 for c in model['run']['clocks']),
                'procedural fixed-total supports only a fresh activation')
        require(topology['kind'] != 'fixed_total' or all(s['dtype'] == 'f64' for s in syn['parameters']),
                'procedural synapse parameters must be f64')
        require(topology['kind'] != 'binary_csr' or all(s['dtype'] == 'f64' for s in syn['parameters'] if s['index_domain'] != 'scalar'),
                'binary CSR columns require f64 parameters')
        for code in syn["code_objects"]:
            for alias in set(code["effects"]["reads"]) & set(syn["pre_state_aliases"]):
                state = syn["pre_state_aliases"][alias]
                source = syn["source_population"]
                require(state not in written[source],
                        "on_pre may only read presynaptic state proven unwritten throughout the activation")
                readonly_pre_states.add((source, state))
            writes = set(code["effects"]["writes"])
            synapse_states = {state["name"] for state in syn["states"]}
            if code["kind"] in {"synapses", "synapses_post"}:
                require(writes <= synapse_states | set(syn["post_state_aliases"]),
                        "on_pre may only write local synapse or target-neuron state")
            elif code["kind"] == "synapse_state_update":
                require(writes <= synapse_states,
                        "clock-driven synapse updates may only write local synapse state")
            else:
                require(code.get("summed_target") == "post",
                        "only postsynaptic summed variables are target-owned without a cross-rank reduction")
                require(not writes, "summed-variable code must write only its declared reduction target")
        for path in values["pathways"]:
            require(path["kind"] in {"pre", "post"} and path["event"] == "spike", "only pre/post spike pathways are supported")
            require(not path["pending"] or topology["kind"] != "fixed_total", "procedural pending events are unsupported")
            require(all(delay >= 0 for delay in path["delay_ticks"]), "synaptic delays must be nonnegative integer ticks")
        if topology["kind"] == "fixed_total":
            procedural.append(q)
            for rank in range(ranks):
                incoming[rank][q] = None
            continue
        if topology["kind"] == "binary_csr":
            require(all(len(path["delay_ticks"]) == 1 for path in values["pathways"]),
                    "binary CSR requires uniform pathway delays")
            for a, b, count in _binary_routes(syn, topology, pops, ranks, owners):
                incoming[b][q] += count
                route_counts[q, a, b] = count
                if a != b:
                    for path in values["pathways"]:
                        delay = path["delay_ticks"][0]
                        min_cross_delay = delay if min_cross_delay is None else min(min_cross_delay, delay)
            continue
        routes, target_counts, minimum_delay = _explicit_routes(
            syn, values, pops, ranks, owners)
        for rank, count in enumerate(target_counts):
            incoming[rank][q] += count
        for a, b, count in routes:
            route_counts[q, a, b] = count
        if minimum_delay is not None:
            min_cross_delay = minimum_delay if min_cross_delay is None else min(min_cross_delay, minimum_delay)
    if backends:
        from .mpi_gpu import kernels
        kernels(model)  # Reject unsupported GPU operations before writing artifacts.
    execution = _derive_execution_plan(model)
    exchange_nodes = []
    for node in execution.logical.nodes:
        if node.owner_kind == "population" and (node.operation == "event_source" or (
                node.operation == "code_object" and
                pops[node.owner_index]["code_objects"][node.item_index]["kind"] == "threshold")):
            exchange_nodes.append(node.id)
    shards = tuple(RankShard(rank,
        tuple(_population_range(p["count"], rank, ranks, owner) for p, owner in zip(pops, owners)),
        tuple(incoming[rank])) for rank in range(ranks))
    return DistributedPlan("b2-distributed-plan-v1", execution.definition_sha256,
        execution.instance_sha256, execution.run_sha256, execution.logical, ranks, shards,
        tuple((*key, value) for key, value in sorted(route_counts.items())),
        None if procedural else min_cross_delay, tuple(exchange_nodes),
        tuple(sorted(readonly_pre_states)), tuple(procedural), owners,
        rank_backends=backends,
        numeric_profile="b2-mpi-cpu-f64-gpu-update-f32-v0" if backends else "reference-f64",
        ownership="explicit-population-owner-or-contiguous-target-owned" if any(o is not None for o in owners) else "contiguous-per-population-target-owned")


def build_distributed_plan(model, *, ranks=2, runner=None, population_owners=None,
                           rank_backends=None, numeric_mode="reference-f64"):
    """Independently validate B2IR, then derive the complete MPI v1 contract."""
    return _derive_distributed_plan(validate_model(model, runner=runner), ranks, population_owners, rank_backends, numeric_mode)


def verify_distributed_plan(plan, model):
    if not isinstance(plan, DistributedPlan) or canonical_bytes(plan.to_dict()) != canonical_bytes(
            _derive_distributed_plan(model, plan.ranks, plan.population_owners, plan.rank_backends or None,
                                     "mixed-f32" if plan.rank_backends else "reference-f64").to_dict()):
        raise PlanValidationError("distributed plan differs from model, ownership or MPI policy")


def explain_distributed_plan(plan):
    lines = [f"DistributedPlan {plan.schema} / {plan.ranks} ranks / runtime unbound",
             f"  numerical contract: {plan.numeric_profile}",
             *([f"  rank backends: {plan.rank_backends}; GPU scope: population state updates"] if plan.rank_backends else []),
             f"  ownership: {plan.ownership}", f"  transport: {plan.transport}",
             f"  synchronization: {plan.synchronization}",
             f"  storage: {plan.storage}",
             f"  minimum cross-rank delay: {plan.min_cross_rank_delay_ticks} ticks (no batching yet)",
             "  final collection: owner-selected bit patterns; no floating-point reduction"]
    for shard in plan.shards:
        lines.append(f"  rank {shard.rank}: neuron ranges={shard.population_ranges}, incoming edges={shard.incoming_edges}")
    if plan.procedural_projections:
        lines.append(f"  procedural projections={plan.procedural_projections}: incoming counts and routes unknown until runtime; no lookahead minimum claimed")
    return "\n".join(lines) + "\n"


def _checked(command, **kwargs):
    result = subprocess.run([str(arg) for arg in command], capture_output=True, text=True, **kwargs)
    if result.returncode:
        raise RuntimeError(f"MPI backend failed ({command[0]}):\n{result.stderr or result.stdout}")
    return result


def write_mpi_project(model, directory, *, ranks=2, runner=None, plan=None,
                      population_owners=None, compact_projections=False,
                      compact_populations=False, prebuild_shared_topology=False,
                      compact_queue_indices=False, compact_spike_history=False,
                      compact_spike_output=False, spike_spool_bytes=None,
                      spike_spool_population_bytes=None, rank_backends=None,
                      numeric_mode=None):
    """Write a reviewable, immutable-instance MPI AOT project; do not compile.

    Opt-in projection compaction reduces repeated fixed-total loading, reporting
    and adjacent additive code. Unsupported layouts keep the ordinary emitter.
    Population compaction additionally aggregates population fields and implies
    projection compaction. The plan and rank-local input bytes are unchanged.
    Shared topology prebuild is an independent physical optimization. Its
    retained cache defaults to 128 MiB/rank, overridable via
    B2_MPI_MAX_PREBUILT_TOPOLOGY_BYTES; service guards still bound temporaries.
    Compact queue indices use u32 for canonical additive pathways only, with
    checked source/edge domains. FIFO and scalar additions remain unchanged.
    Compact spike history shares one checked u32 recorder between the two
    unchanged binary outputs; named events/EventMonitor are not supported.
    Compact spike output additionally opts into result v4 / event v2 with
    eight-byte spike pairs. It requires checked compact spike history.
    spike_spool_bytes optionally bounds a shared Linux64 disk history, using
    fixed per-population buffers. It requires compact output and explicit disk
    admission for temporary spools plus both final files; the byte budget only
    bounds spool payload, not final files or unrelated state monitors.
    An optional population limit bounds temporary overlap during final output:
    at most twice the total spool limit plus one population limit in spike
    payload, plus all non-spike bytes, filesystem overhead, inputs and reserve.
    Completed event prefixes are synced before their population spool is removed;
    a later failure leaves durable prefixes and remaining spools, not a success
    summary. This is not a restart/checkpoint protocol.
    """
    from .native import generate_source
    from .binary_topology import file_hash
    if type(prebuild_shared_topology) is not bool:
        raise TypeError('prebuild_shared_topology must be a boolean')
    if type(compact_queue_indices) is not bool:
        raise TypeError('compact_queue_indices must be a boolean')
    if type(compact_spike_history) is not bool:
        raise TypeError('compact_spike_history must be a boolean')
    if type(compact_spike_output) is not bool:
        raise TypeError('compact_spike_output must be a boolean')
    if compact_spike_output and not compact_spike_history:
        raise ValueError('compact_spike_output requires compact_spike_history')
    if spike_spool_bytes is not None:
        if type(spike_spool_bytes) is not int or not 1 <= spike_spool_bytes <= 512*2**30:
            raise ValueError('spike_spool_bytes must be an integer in 1..512 GiB')
        if not compact_spike_output:
            raise ValueError('spike_spool_bytes requires compact_spike_output')
    if spike_spool_population_bytes is not None:
        if (spike_spool_bytes is None or type(spike_spool_population_bytes) is not int
                or not 1 <= spike_spool_population_bytes <= spike_spool_bytes):
            raise ValueError('spike_spool_population_bytes must be an integer in 1..spike_spool_bytes')
    current = validate_model(model, runner=runner)
    if plan is None:
        plan = _derive_distributed_plan(current, ranks, population_owners, rank_backends, numeric_mode or "reference-f64")
    else:
        from .mpi_gpu import backend_policy
        if rank_backends is not None or numeric_mode is not None:
            requested = backend_policy(ranks, rank_backends if rank_backends is not None else plan.rank_backends or None,
                                       numeric_mode or ("mixed-f32" if plan.rank_backends else "reference-f64"))
            if requested != plan.rank_backends:
                raise PlanValidationError("requested rank backends differ from plan")
        if population_owners is not None and tuple(population_owners) != plan.population_owners:
            raise PlanValidationError("requested population ownership differs from plan")
        verify_distributed_plan(plan, current)
        if ranks != plan.ranks:
            raise PlanValidationError("requested MPI rank count differs from plan")
    if plan.rank_backends and (compact_projections or compact_populations or prebuild_shared_topology
            or compact_queue_indices or compact_spike_history or compact_spike_output or spike_spool_bytes is not None):
        raise PlanValidationError("MPI GPU offload does not yet support source compaction/prebuild options")
    source = generate_source(current, distributed_plan=plan)
    history_compaction = None
    if compact_spike_history:
        from .mpi_spike_history import compact_spike_history_source
        source, history_compaction = compact_spike_history_source(current, source)
    compaction = None
    if compact_projections or compact_populations:
        from .mpi_compact import compact_source
        source, compaction = compact_source(current, source)
    output_compaction = None
    if compact_spike_output:
        from .mpi_spike_output import compact_spike_output_source
        source, output_compaction = compact_spike_output_source(current, source)
    spool = None
    if spike_spool_bytes is not None:
        from .mpi_spike_spool import spool_spike_history_source
        source, spool = spool_spike_history_source(current, source, spike_spool_bytes, spike_spool_population_bytes)
    population_compaction = None
    if compact_populations:
        population_compaction = {"compacted": False}
        if compaction["compacted"]:
            from .mpi_population_compact import aggregate_population_source
            source, population_compaction = aggregate_population_source(current, source)
    prebuild = None
    if prebuild_shared_topology:
        from .mpi_prebuild import prebuild_shared_source
        source, prebuild = prebuild_shared_source(current, plan, source)
    queue_compaction = None
    if compact_queue_indices:
        from .mpi_queue_compact import compact_queue_source
        source, queue_compaction = compact_queue_source(current, source)
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "main.rs").write_text(source)
    from .native import write_timed_array_blobs
    timed_blobs = write_timed_array_blobs(current, directory)
    timed_files = tuple(filename for _symbol, filename, _payload in timed_blobs)
    (directory / "execution-plan.json").write_text(plan.to_json())
    shim = Path(__file__).with_name("mpi_runtime") / "bridge.c"
    shutil.copyfile(shim, directory / "mpi_bridge.c")
    from .mpi_partition import write_shards
    shard_hashes = write_shards(current, directory, plan)
    manifest = {"schema": "b2-mpi-artifact-v0", "plan_sha256": plan.sha256,
                "ranks": plan.ranks, "files": {},
                "timed_array_files": timed_files}
    manifest["storage"] = "rank-local-v1"
    if compaction is not None:
        manifest["projection_compaction"] = compaction
    if population_compaction is not None:
        manifest["population_compaction"] = population_compaction
    if prebuild is not None:
        manifest['topology_prebuild'] = prebuild
    if queue_compaction is not None:
        manifest['queue_compaction'] = queue_compaction
    if history_compaction is not None:
        manifest['spike_history_compaction'] = history_compaction
    if spool is not None:
        manifest["spike_spool"] = spool
    if output_compaction is not None:
        manifest['spike_output_compaction'] = output_compaction
    gpu_files = ()
    if plan.rank_backends:
        from .mpi_gpu import write_sources
        gpu_files = write_sources(current, directory, plan.rank_backends)
        manifest["rank_backends"] = plan.rank_backends
    for name in (*gpu_files, *timed_files, "main.rs", "mpi_bridge.c", "instance.bin", "execution-plan.json", "instance-identities.rs", *shard_hashes):
        manifest["files"][name] = file_hash(directory / name)
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return plan


def _verify_artifact(directory):
    from .binary_topology import file_hash
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("schema") != "b2-mpi-artifact-v0":
        raise ValueError("invalid MPI artifact manifest")
    expected = {"main.rs", "mpi_bridge.c", "instance.bin", "execution-plan.json"}
    expected.update(manifest.get("timed_array_files", []))
    if manifest.get("storage") == "rank-local-v1":
        expected.update(f"instance.rank-{r}.bin" for r in range(manifest["ranks"]))
        expected.add("instance-identities.rs")
    plan_data = json.loads((directory / "execution-plan.json").read_text())
    if manifest.get("rank_backends") != plan_data.get("rank_backends"):
        raise ValueError("MPI artifact rank backends differ from execution plan")
    if manifest.get("rank_backends"):
        from .mpi_gpu import source_inventory, backend_policy
        backend_policy(manifest["ranks"], manifest["rank_backends"], "mixed-f32")
        expected.update(source_inventory(manifest["rank_backends"]))
    if set(manifest["files"]) != expected:
        raise ValueError("invalid MPI artifact file inventory")
    for name, digest in manifest["files"].items():
        if file_hash(directory / name) != digest:
            raise ValueError(f"MPI artifact changed: {name}; rebuild required")
    return manifest


def compile_mpi_project(directory, *, mpicc="mpicc", rustc="rustc", opt_level=3, panic_strategy=None):
    """Compile the C ABI shim with the MPI implementation's own wrapper."""
    if type(opt_level) is not int or opt_level not in (0,1,2,3):
        raise ValueError("MPI compiler opt_level must be an integer in 0..3")
    if panic_strategy not in (None, "unwind", "abort"):
        raise ValueError("MPI compiler panic_strategy must be unwind or abort")
    directory = Path(directory).resolve()
    manifest = _verify_artifact(directory)
    # Select the repository pin even when the caller's MPI project is outside
    # this checkout, and reject an explicitly selected older compiler.
    from .device import ROOT, RustStandaloneDevice
    rustc_verbose, rustc_host = RustStandaloneDevice._pinned_rustc(rustc)
    compiler = shutil.which(str(mpicc))
    if compiler is None:
        raise FileNotFoundError(f"MPI compiler not found: {mpicc}; install MPICH or Open MPI")
    _checked([compiler, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-c",
              directory / "mpi_bridge.c", "-o", directory / "mpi_bridge.o"])
    gpu_build = None
    if manifest.get("rank_backends"):
        from .mpi_gpu import compile_libraries
        gpu_build = compile_libraries(directory, manifest["rank_backends"])
    executable = directory / "b2-mpi"
    _checked([rustc, "--edition=2021", "-C", f"opt-level={opt_level}",
              *(["-C", f"panic={panic_strategy}"] if panic_strategy is not None else []), "-C", f"linker={compiler}",
              "-C", f"link-arg={directory / 'mpi_bridge.o'}",
              *(["-l", "dl"] if gpu_build is not None and os.uname().sysname != "Darwin" else []), directory / "main.rs", "-o", executable], cwd=ROOT)
    build = {"schema": "b2-mpi-build-v0", "executable_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
             "opt_level": opt_level, "rustc": rustc_verbose.splitlines()[0],
             "rustc_host": rustc_host,
             "mpicc": compiler, "compiler_version": _checked([compiler, "--version"]).stdout.strip()}
    if gpu_build is not None:
        build["gpu_libraries"] = gpu_build
    if panic_strategy is not None:
        build["panic"] = panic_strategy
    (directory / "build.json").write_text(json.dumps(build, indent=2) + "\n")
    return executable


def run_mpi_project(directory, output, *, mpiexec="mpiexec", launcher_args=(), timeout=120):
    """Launch actual MPI processes. Caller can supply hostfile options as a list.

    Executable/instance must be visible at the same paths on every host. v0
    output is written by rank zero. This function never invokes a shell.
    """
    directory, output = Path(directory).resolve(), Path(output).resolve()
    manifest = _verify_artifact(directory)
    executable = directory / "b2-mpi"
    build = json.loads((directory / "build.json").read_text())
    if hashlib.sha256(executable.read_bytes()).hexdigest() != build["executable_sha256"]:
        raise ValueError("MPI executable changed; rebuild required")
    from .mpi_gpu import library_inventory
    if set(build.get("gpu_libraries", {})) != set(library_inventory(manifest.get("rank_backends", ()))):
        raise ValueError("invalid MPI GPU build library inventory")
    for name, digest in build.get("gpu_libraries", {}).items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"MPI GPU library changed: {name}; rebuild required")
    if output.exists():
        raise FileExistsError(f"MPI output already exists: {output}")
    if isinstance(launcher_args, (str, bytes)):
        raise TypeError("launcher_args must be a sequence of individual arguments")
    command = [str(mpiexec), *map(str, launcher_args), "-n", str(manifest["ranks"]),
               str(executable), str(directory / "instance.bin"), str(output)]
    # A timeout must stop the launcher and its local ranks, not orphan a job.
    with subprocess.Popen(command, stdin=subprocess.DEVNULL,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          text=True, start_new_session=True) as process:
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            stdout, stderr = process.communicate()
            (directory / "launch.log").write_text(stdout + stderr)
            raise
    (directory / "launch.log").write_text(stdout + stderr)
    if process.returncode:
        raise RuntimeError(f"MPI run failed ({process.returncode}):\n{stderr or stdout}")
    observed = json.loads((output / "mpi-runtime.json").read_text())
    if observed["plan_sha256"] != manifest["plan_sha256"] or observed["ranks"] != manifest["ranks"]:
        raise ValueError("MPI runtime does not match compiled plan")
    return observed
