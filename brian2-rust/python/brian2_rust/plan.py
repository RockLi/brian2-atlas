"""Versioned, immutable execution plans derived from validated B2IR.

The plan is backend-private: it never changes the frozen B2IR document. JSON
is an inspection format, not trusted executable input. The public builder
uses the independent Rust validator before deriving a plan.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

from .protocol import canonical_bytes, migrate_model, verify_protocol
from .schedule import execution_effects
from .planner import (compact_cpu_choices, general_cpu_choices, _uses_v6,
                      fixed_phase_eligible, slot_cpu_choices)

PLAN_SCHEMA = "b2-execution-plan-v0"
PLANNER_VERSION = "cpu-plan-3"


class PlanValidationError(ValueError):
    """A plan does not describe the validated model or available backend."""


@dataclass(frozen=True)
class LogicalNode:
    id: str
    ordinal: int
    operation: str
    owner_kind: str
    owner_index: int
    item_index: int
    clock: int
    reads: tuple[str, ...]
    writes: tuple[str, ...]
    dependencies: tuple[str, ...]


@dataclass(frozen=True)
class ClockActivation:
    clock: int
    dt: str
    start_tick: int
    steps: int


@dataclass(frozen=True)
class LogicalPlan:
    nodes: tuple[LogicalNode, ...]
    clocks: tuple[ClockActivation, ...]


@dataclass(frozen=True)
class BufferSpec:
    resource: str
    dtype: str
    elements: int | None
    payload_bytes: int | None
    lifetime: str
    certainty: str
    role: str


@dataclass(frozen=True)
class Decision:
    subject: str
    selected: str
    reason: str
    nodes: tuple[str, ...] = ()


@dataclass(frozen=True)
class CpuPhysicalPlan:
    emitter: str
    # Canonical JSON makes nested choices immutable without retaining model
    # dicts or topology arrays. Consumers receive a detached mutable copy.
    choices_json: str
    decisions: tuple[Decision, ...]
    buffers: tuple[BufferSpec, ...]
    runtime_inputs: tuple[str, ...] = (
        "B2_NUM_THREADS", "B2_THREAD_AFFINITY", "allowed_cpus", "topology")

    def choices(self):
        return json.loads(self.choices_json)


@dataclass(frozen=True)
class ExecutionPlan:
    schema: str
    planner_version: str
    definition_sha256: str
    instance_sha256: str
    run_sha256: str
    logical: LogicalPlan
    cpu: CpuPhysicalPlan

    def to_dict(self):
        result = asdict(self)
        result["cpu"]["choices"] = json.loads(result["cpu"].pop("choices_json"))
        return result

    def to_json(self, *, indent=2):
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True,
                          indent=indent, allow_nan=False) + "\n"

    @property
    def sha256(self):
        return hashlib.sha256(canonical_bytes(self.to_dict())).hexdigest()


    @property
    def policy_sha256(self):
        """Specialization identity for instance replacement (values excluded)."""
        identity = self.to_dict()
        identity.pop("instance_sha256")
        return hashlib.sha256(canonical_bytes(identity)).hexdigest()


@dataclass(frozen=True)
class RuntimeBinding:
    schema: str
    plan_sha256: str
    backend: str
    observations_json: str

    def to_dict(self):
        return {"schema": self.schema, "plan_sha256": self.plan_sha256,
                "backend": self.backend, "observed": json.loads(self.observations_json)}

    def to_json(self):
        return json.dumps(self.to_dict(), sort_keys=True, indent=2, allow_nan=False) + "\n"


def bind_execution_plan(plan, metadata, *, requested_threads=None, requested_affinity=None):
    """Attach reported runtime facts after successful execution, never predict them.

    The caller must supply metadata from this plan's artifact. CPU artifacts bind
    source and plan identities in their manifest; Metal reports the plan hash.
    Uninstrumented queue capacity/owner maps remain explicitly unavailable.
    """
    from .metal import MetalPlan
    from .cuda import CudaPlan
    from .wasm import WasmPlan
    if isinstance(plan, WasmPlan):
        if metadata.get("plan_sha256") != plan.sha256 or metadata.get("numeric_profile") != plan.numeric_profile or metadata.get("backend") != "wasm":
            raise PlanValidationError("WASM result belongs to a different plan or numeric profile")
        return RuntimeBinding("b2-runtime-binding-v0", plan.sha256, "wasm",
                              canonical_bytes(metadata).decode())
    from .distributed import DistributedPlan
    if isinstance(plan, DistributedPlan):
        observed = metadata.get("mpi", {})
        if (observed.get("plan_sha256") != plan.sha256
                or type(observed.get("ranks")) is not int or observed["ranks"] != plan.ranks
                or observed.get("numeric_profile") != plan.numeric_profile
                or observed.get("transport") != plan.transport):
            raise PlanValidationError("MPI result belongs to a different plan or rank count")
        return RuntimeBinding("b2-runtime-binding-v0", plan.sha256, "mpi",
                              canonical_bytes(observed).decode())
    if isinstance(plan, (MetalPlan,CudaPlan)):
        if (metadata.get("plan_sha256") != plan.sha256 or metadata.get("numeric_profile") != plan.numeric_profile
                or metadata.get("rng_profile") != plan.rng_profile
                or metadata.get("initializations", []) != [asdict(item) for item in plan.initializations]):
            raise PlanValidationError("GPU result belongs to a different plan or numeric profile")
        backend = "cuda" if isinstance(plan,CudaPlan) else "metal"
        keys = ("device", "numeric_profile", "rng_profile", "initializations", "initialization_seconds", f"{backend}_timings", "run_seconds", "timings", "cuda_runtime")
    else:
        threads = metadata.get("threads")
        if type(threads) is not int or not 1 <= threads <= 256:
            raise PlanValidationError("CPU result lacks a valid effective worker count")
        backend = "cpu"
        keys = ("threads", "thread_affinity", "thread_cpus", "parallel_state_update",
                "parallel_on_pre", "parallel_event_enqueue", "parallel_poisson_input",
                "parallel_plasticity", "parallel_summed_variable", "phase_profile", "timings")
    observed = {key: metadata[key] for key in keys if key in metadata}
    if backend == "cpu":
        observed.update(requested_threads=requested_threads, requested_affinity=requested_affinity,
                        owner_map="not_instrumented", queue_capacities="not_instrumented")
    return RuntimeBinding("b2-runtime-binding-v0", plan.sha256, backend,
                          canonical_bytes(observed).decode())


def _logical_plan(model):
    definition = model["definition"]
    nodes, last_writer, readers = [], {}, {}
    for ordinal, node in enumerate(definition["schedule"]["nodes"]):
        effects = execution_effects(definition, node)
        reads, writes = set(effects["reads"]), set(effects["writes"])
        dependencies = {last_writer[r] for r in reads | writes if r in last_writer}
        for resource in writes:
            dependencies.update(readers.get(resource, ()))
            last_writer[resource] = node["id"]
            readers[resource] = set()
        for resource in reads:
            readers.setdefault(resource, set()).add(node["id"])
        nodes.append(LogicalNode(
            node["id"], ordinal, node["operation"], node["owner_kind"],
            node["owner_index"], node["item_index"], node["clock"],
            tuple(sorted(reads)), tuple(sorted(writes)), tuple(sorted(dependencies))))
    clocks = tuple(ClockActivation(i, clock["dt"], run["start_tick"], run["steps"])
                   for i, (clock, run) in enumerate(zip(
                       definition["clocks"], model["run"]["clocks"], strict=True)))
    return LogicalPlan(tuple(nodes), clocks)


def _buffers(model, choices=None):
    widths = {"bool": 1, "f32": 4, "f64": 8, "i32": 4, "u32": 4,
              "i64": 8, "u64": 8}
    buffers = []
    for kind, plural in (("population", "populations"), ("synapse", "synapses")):
        for index, (definition, instance) in enumerate(zip(
                model["definition"][plural], model["instance"][plural], strict=True)):
            if kind == "population":
                count = definition["count"]
            else:
                from .planner import _synapse_edge_count
                count = _synapse_edge_count(instance)
            for category, table in (("state", "states"), ("parameter", "parameters")):
                for symbol in definition[table]:
                    elements = 1 if symbol["index_domain"] == "scalar" else count
                    buffers.append(BufferSpec(
                        f"{kind}/{index}/{category}/{symbol['name']}", symbol["dtype"],
                        elements, elements * widths[symbol["dtype"]], "instance",
                        "exact_declared_payload", "state" if category == "state" else "parameter"))
            if kind == "population":
                buffers.append(BufferSpec(f"population/{index}/recording", "mixed",
                                          None, None, "run_and_continuation",
                                          "unknown_dynamic", "monitor_and_events"))
                if definition["refractory"] is not None:
                    for name, dtype in (("lastspike", "f64"), ("not_refractory", "bool")):
                        buffers.append(BufferSpec(f"population/{index}/refractory/{name}",
                                                  dtype, count, count * widths[dtype],
                                                  "instance", "exact_declared_payload", "refractory"))
            else:
                buffers.append(BufferSpec(f"synapse/{index}/queues", "mixed", None,
                                          None, "run_and_continuation",
                                          "unknown_dynamic", "pending_events"))
                buffers.append(BufferSpec(f"synapse/{index}/topology", "mixed", count,
                                          None, "instance", "runtime_layout", "topology"))
    for node, caches in (choices or {}).get("summed_endpoint_caches", {}).items():
        for position, cache in enumerate(caches):
            buffers.append(BufferSpec(
                f"synapse/{node.replace('/', '/code/')}/endpoint-exp/{position}",
                "f64", cache["count"], 8 * cache["count"], "node_activation",
                "exact_declared_payload", "endpoint_expression_cache"))
    return tuple(buffers)


def _derive_execution_plan(model):
    """Internal entry after independent semantic validation; does not copy arrays."""
    verify_protocol(model)
    canonical = not fixed_phase_eligible(model)
    logical = _logical_plan(model)
    compact = bool(_uses_v6(model))
    choices = (slot_cpu_choices(model) if canonical else
               compact_cpu_choices(model) if compact else general_cpu_choices(model))
    decisions = []
    if canonical:
        decisions.append(Decision("schedule", "canonical-serial",
                                  "fixed-phase equivalence or slot capability check failed; exact node order retained"))
        for node in choices.get("summed_endpoint_caches", {}):
            decisions.append(Decision(
                f"synapse/{node.replace('/', '/code/')}", "endpoint-exp-cache",
                "pure f64 expression reads stable endpoint state and scalar inputs; refreshed in the original node with edge reduction order retained"))
    if compact:
        decisions.append(Decision("events", "target-owned" if choices["parallel_event_capable"]
                                  else "serial", choices["event_reason"]))
    else:
        for p, fused in choices["population_threshold_fusion"].items():
            decisions.append(Decision(
                f"population/{p}/update-threshold", "fused" if fused else "separate",
                "same-activation effect-safe bundle and work threshold met" if fused else
                "fusion requires supported event/refractory, safe schedule and sufficient work"))
        for route, parallel in enumerate(choices["route_parallel"]):
            decisions.append(Decision(
                f"route/{route}", "target-owned" if parallel else "serial",
                "pathway effects, pending-delay and work checks passed" if parallel else
                "pathway effects, pending-delay or work threshold prevents target ownership"))
        for q, index in choices["final_only_summed"]:
            decisions.append(Decision(f"synapse/{q}/code/{index}", "final-active-tick",
                                      "no completed-resource reader during the run; original slot retained"))
    hashes = model["protocol"]["layers"]
    return ExecutionPlan(
        PLAN_SCHEMA, PLANNER_VERSION, hashes["definition"], hashes["instance"], hashes["run"],
        logical, CpuPhysicalPlan("slot-v1" if canonical else "compact-v6" if compact else "general-v1",
                                canonical_bytes(choices).decode(), tuple(decisions), _buffers(model, choices)))


def validate_model(model, *, runner=None):
    """Verify wire identity and independent Rust semantics, without simulation."""
    current = migrate_model(model)
    from ._runtime import executable_path
    executable = executable_path("b2-runner", runner)
    if not executable.is_file():
        raise FileNotFoundError(f"build the Rust B2IR validator before planning: {executable}")
    with tempfile.TemporaryDirectory(prefix="b2-plan-validate-") as directory:
        path = Path(directory) / "model.json"
        path.write_bytes(canonical_bytes(current))
        checked = subprocess.run([str(executable), "--validate", str(path)],
                                 capture_output=True, text=True)
    if checked.returncode:
        raise PlanValidationError(checked.stderr or checked.stdout)
    return current


def build_execution_plan(model, *, runner=None, backend="cpu", numeric_mode=None, event_delivery=None,
                         synapse_sparse=False, synapse_prefix=False, synapse_fusion=False,
                         ranks=None, rank_backends=None):
    """Public validated backend plan builder. Does not create result artifacts."""
    options=dict(synapse_sparse=synapse_sparse,synapse_prefix=synapse_prefix,synapse_fusion=synapse_fusion)
    if backend not in {"metal","cuda"} and any(value is not False for value in options.values()):
        raise PlanValidationError("synapse policies require a GPU backend")
    if backend == "mpi":
        from .distributed import build_distributed_plan
        if event_delivery is not None:
            raise PlanValidationError("MPI uses its own spike transport")
        return build_distributed_plan(model, ranks=2 if ranks is None else ranks, runner=runner,
                                      rank_backends=rank_backends, numeric_mode=numeric_mode or "reference-f64")
    if rank_backends is not None:
        raise PlanValidationError("rank_backends requires backend='mpi'")
    if ranks is not None:
        raise PlanValidationError("ranks requires backend='mpi'")
    if backend == "wasm":
        from .wasm import build_wasm_plan
        if event_delivery is not None:
            raise PlanValidationError("event_delivery requires a GPU backend")
        return build_wasm_plan(model, runner=runner, numeric_mode=numeric_mode)
    if backend == "metal":
        from .metal import build_metal_plan
        return build_metal_plan(model, runner=runner, numeric_mode=numeric_mode,
                                event_delivery="scan" if event_delivery is None else event_delivery,**options)
    if backend == "cuda":
        from .cuda import build_cuda_plan
        return build_cuda_plan(model,runner=runner,numeric_mode=numeric_mode,
                               event_delivery="scan" if event_delivery is None else event_delivery,**options)
    if event_delivery is not None:
        raise PlanValidationError("event_delivery requires a GPU backend")
    if backend != "cpu" or numeric_mode not in (None, "reference-f64"):
        raise PlanValidationError("CPU planning requires backend='cpu' and reference-f64")
    return _derive_execution_plan(validate_model(model, runner=runner))


def verify_execution_plan(plan, model, *, synapse_sparse=False, synapse_prefix=False, synapse_fusion=False):
    """Re-derive all choices as well as identity; hashes alone are not a proof.

    The caller must use the independent B2IR validator at an external input
    boundary. This verifier is also used inside the validated AOT pipeline.
    """
    from .metal import MetalPlan, _derive_metal_plan
    from .distributed import DistributedPlan, verify_distributed_plan
    from .cuda import CudaPlan, _derive_cuda_plan
    options=dict(synapse_sparse=synapse_sparse,synapse_prefix=synapse_prefix,synapse_fusion=synapse_fusion)
    if not isinstance(plan,(MetalPlan,CudaPlan)) and any(value is not False for value in options.values()):
        raise PlanValidationError("synapse policies require a GPU plan")
    if isinstance(plan, DistributedPlan):
        return verify_distributed_plan(plan, model)
    from .wasm import WasmPlan, _derive_wasm_plan
    if isinstance(plan, WasmPlan):
        if canonical_bytes(plan.to_dict()) != canonical_bytes(_derive_wasm_plan(model).to_dict()):
            raise PlanValidationError("WASM execution plan does not match model or compiler policy")
        return
    if isinstance(plan,(MetalPlan,CudaPlan)):
        derive=_derive_cuda_plan if isinstance(plan,CudaPlan) else _derive_metal_plan
        expected=derive(model,numeric_mode="float32",event_delivery=plan.event_delivery if plan.dispatches else "scan",**options)
        if canonical_bytes(plan.to_dict()) != canonical_bytes(expected.to_dict()):
            backend="CUDA" if isinstance(plan,CudaPlan) else "Metal"
            raise PlanValidationError(f"{backend} execution plan does not match model or compiler policy")
        return
    if (not isinstance(plan, ExecutionPlan) or
            canonical_bytes(plan.to_dict()) !=
            canonical_bytes(_derive_execution_plan(model).to_dict())):
        raise PlanValidationError("execution plan does not match model, effects or CPU policy")


def explain_plan(plan, *, format="text", binding=None):
    if binding is not None and binding.plan_sha256 != plan.sha256:
        raise PlanValidationError("runtime binding belongs to a different plan")
    if binding is not None and format in {"dict", "json"}:
        result = {"plan": plan.to_dict(), "runtime_binding": binding.to_dict()}
        return result if format == "dict" else json.dumps(result, sort_keys=True, indent=2) + "\n"
    if binding is not None and format == "text":
        return (explain_plan(plan, format="text").replace("runtime unbound", "runtime observed")
                .replace("  runtime workers/affinity/ownership: unresolved until execution\n", "")
                + "  runtime observations: " + json.dumps(binding.to_dict()["observed"], sort_keys=True) + "\n")
    if format == "json":
        return plan.to_json()
    if format == "dict":
        return plan.to_dict()
    if format != "text":
        raise ValueError("format must be text, json or dict")
    from .distributed import DistributedPlan, explain_distributed_plan
    if isinstance(plan, DistributedPlan):
        return explain_distributed_plan(plan)
    from .metal import MetalPlan
    from .cuda import CudaPlan
    if isinstance(plan, (MetalPlan,CudaPlan)):
        backend = "CUDA" if isinstance(plan,CudaPlan) else "Metal"
        return (f"ExecutionPlan {plan.schema} [{backend}; {plan.numeric_profile}]\n"
                + f"  strategy: {plan.strategy}\n"
                + f"  clocks: {len(plan.logical.clocks)}; f64 host scheduling\n"
                + ("  mutable target events: ordered bitmaps\n" if any(d.role=='target-owned-bitset-synapse-pathway' for d in plan.dispatches) else "")
                + (f"  RNG: {plan.rng_profile}\n" if plan.rng_profile is not None else "")
                + "".join(f"  initialization: projection {item.projection}, {item.edge_count} edges, {item.strategy}, sha256={item.sha256}\n" for item in plan.initializations)
                + "".join(f"  {k.entry}: {k.neurons} lanes, {k.steps} ticks, "
                          f"{len(k.nodes)} ordered nodes\n" for k in plan.kernels)
                + "  numeric mode: explicitly approximate float32\n")
    from .wasm import WasmPlan
    if isinstance(plan, WasmPlan):
        return (f"ExecutionPlan {plan.schema} [WASM; {plan.numeric_profile}]\n"
                f"  plan: {plan.sha256}\n"
                f"  strategy: {plan.strategy}\n"
                f"  logical: {len(plan.logical.nodes)} nodes, {len(plan.logical.clocks)} clocks\n")
    lines = [f"ExecutionPlan {plan.schema} [{plan.cpu.emitter}; runtime unbound]",
             f"  plan: {plan.sha256}",
             f"  logical: {len(plan.logical.nodes)} nodes, {len(plan.logical.clocks)} clocks"]
    for decision in plan.cpu.decisions:
        lines.append(f"  {decision.subject}: {decision.selected} ({decision.reason})")
    payload = sum(b.payload_bytes or 0 for b in plan.cpu.buffers)
    lines += [f"  declared payload: {payload} bytes (excludes runtime layout/queue/recording overhead)",
              "  runtime workers/affinity/ownership: unresolved until execution"]
    return "\n".join(lines) + "\n"
