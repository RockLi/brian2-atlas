"""CPU legality and cost policy, independent of source emission.

Input models must have passed the B2IR semantic validator. This module never
emits Rust, mutates a model, compiles code or starts a simulation.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from .schedule import execution_effects

SOURCE_BATCH_MIN_EDGES = 65_536
PARALLEL_EVENT_MIN_EDGES = 2_048
PARALLEL_MIN_TOTAL_WORK = 500_000
PARALLEL_TARGET_TASK_WORK = 100_000
PARALLEL_MIN_TASK_ITEMS = 64
POST_SYNAPSE_INDIRECT_ACCESS_WORK = 64


@dataclass(frozen=True)
class TargetParallelPlan:
    """Shared compile-time decision for a target-owned event route."""

    edge_count: int
    eligible: bool
    reason: str


def _number(bits):
    return struct.unpack(">d", bytes.fromhex(bits))[0]


def _synapse_edge_count(instance):
    topology = instance.get("topology", {"kind": "explicit"})
    if topology["kind"] == "explicit":
        return len(instance["source"])
    return topology["edge_count"]


def _explicit_topology(instance):
    return instance.get("topology", {"kind": "explicit"})["kind"] == "explicit"


def code_work(code):
    """Conservative generated-work proxy for adaptive threading.

    Transcendentals, division, powers and distribution sampling cost much more
    than an arithmetic/load AST node.  Keeping that distinction here lets all
    generated neuron phases share one scheduler without model-specific rules.
    """
    weights = {
        "div": 3, "mod": 3, "floordiv": 3, "pow": 4,
        "exp": 8, "expm1": 8, "exprel": 8, "log": 8,
        "log10": 8, "log1p": 8, "sqrt": 6,
        "sin": 8, "cos": 8, "tan": 10,
        "sinh": 8, "cosh": 8, "tanh": 8,
        "arcsin": 10, "arccos": 10, "arctan": 10,
        "rand": 4, "randn": 32, "binomial": 128, "poisson": 64,
    }

    def count(value):
        if isinstance(value, dict):
            own = weights.get(value.get("op"), 1)
            return own + sum(count(child) for child in value.values())
        if isinstance(value, list):
            return sum(count(child) for child in value)
        return 0
    return max(1, count(code["vector"]))


def post_synapse_work(synapse, code):
    """Include indirect state traffic in a postsynaptic edge batch's cost.

    Target CSR traverses indices into source-ordered state arrays. Arithmetic
    alone underestimates these gathers, especially for sparse firing with many
    plastic traces per edge. Scalar parameters incur no indirect array access.
    The same estimate must govern pool creation and runtime dispatch.
    """
    states = {symbol["name"] for symbol in synapse["states"]}
    indirect = states | {symbol["name"] for symbol in synapse["parameters"]
                         if symbol["index_domain"] != "scalar"}
    accesses = (len(set(code["effects"]["reads"]) & indirect) +
                len(set(code["effects"]["writes"]) & states))
    return max(code_work(code), POST_SYNAPSE_INDIRECT_ACCESS_WORK * accesses)


def _v7_regular_parallel_capable(model, q, code):
    synapse = model["definition"]["synapses"][q]
    instance = model["instance"]["synapses"][q]
    # Each edge owns its state. Population aliases are read-only for the
    # entire dispatch; the next scheduled code object waits for all lanes.
    return (_explicit_topology(instance) and
            bool(code["effects"]["writes"]) and
            set(code["effects"]["writes"]) <=
            {symbol["name"] for symbol in synapse["states"]} and
            parallel_phase_capable(_synapse_edge_count(instance), code_work(code)))


def parallel_task_limit(items, work, threads=64):
    """Maximum useful tasks for a uniform generated phase.

    The work ceiling exposes more lanes for compact expensive models such as
    HH, while the item floor prevents tiny shards for cheap memory-bound loops.
    """
    if items <= 0 or work <= 0 or threads <= 0:
        return 1
    by_items = (items + PARALLEL_MIN_TASK_ITEMS - 1) // PARALLEL_MIN_TASK_ITEMS
    total_work = items * work
    by_work = (total_work + PARALLEL_TARGET_TASK_WORK - 1) // PARALLEL_TARGET_TASK_WORK
    return max(1, min(threads, by_items, by_work))


def parallel_phase_capable(items, work):
    return (items * work >= PARALLEL_MIN_TOTAL_WORK and
            parallel_task_limit(items, work) > 1)


def target_parallel_pathway(synapse, code, same_population=False):
    """Whether target ownership makes an on_pre pathway race-free.

    A worker owns every write to a target-neuron state in its target shard and
    every synapse edge occurs in exactly one shard.  Pre-neuron state is only
    read.  For a recurrent projection, reject the one hazardous case where a
    pathway reads through a pre alias and writes the same underlying state
    through a post alias.
    """
    writes = set(code["effects"]["writes"])
    reads = set(code["effects"]["reads"])
    synaptic_states = {symbol["name"] for symbol in synapse["states"]}
    post_aliases = synapse["post_state_aliases"]
    pre_aliases = synapse["pre_state_aliases"]
    if not writes or not writes <= (set(post_aliases) | synaptic_states):
        return False
    # A name that resolves as a pre-side write is conservatively rejected,
    # even if a malformed/ambiguous schema also exposes it as a post alias.
    if writes & set(pre_aliases):
        return False
    if same_population:
        pre_reads = {pre_aliases[name] for name in reads if name in pre_aliases}
        post_writes = {post_aliases[name] for name in writes if name in post_aliases}
        if pre_reads & post_writes:
            return False
    return True


def target_parallel_plan(pathways):
    """Plan target-owned execution for one or more ordered pathways.

    ``pathways`` contains ``(synapse, code, same_population, edge_count)``
    tuples. Both the compact single-population generator and the general
    multi-population generator use this decision so new partition policies do
    not acquire separate eligibility semantics.
    """
    edge_count = sum(pathway[3] for pathway in pathways)
    if edge_count < PARALLEL_EVENT_MIN_EDGES:
        return TargetParallelPlan(edge_count, False, "event batch too small")

    pre_reads, recurrent_post_writes = set(), set()
    for synapse, code, same_population, _edge_count in pathways:
        if not target_parallel_pathway(synapse, code, same_population):
            return TargetParallelPlan(edge_count, False, "unsafe pathway effects")
        if same_population:
            for name in code["effects"]["reads"]:
                if name in synapse["pre_state_aliases"]:
                    pre_reads.add(synapse["pre_state_aliases"][name])
            for name in code["effects"]["writes"]:
                if name in synapse["post_state_aliases"]:
                    recurrent_post_writes.add(
                        synapse["post_state_aliases"][name])
    if pre_reads & recurrent_post_writes:
        return TargetParallelPlan(
            edge_count, False, "cross-pathway recurrent dependency")
    return TargetParallelPlan(edge_count, True, "target-owned")


def _has_runtime_random(model):
    def visit(value):
        if isinstance(value, dict):
            return value.get("op") in {"rand", "randn", "binomial", "poisson"} or any(
                visit(item) for item in value.values())
        if isinstance(value, list):
            return any(visit(item) for item in value)
        return False
    return visit(model["definition"]["populations"]) or visit(
        model["definition"]["synapses"])


def _plastic_pathway(synapse, code):
    """Whether a pathway only touches synaptic state/parameters and time."""
    def has_random(value):
        if isinstance(value, dict):
            return value.get("op") in {"rand", "randn", "binomial", "poisson"} or any(
                has_random(child) for child in value.values())
        if isinstance(value, list):
            return any(has_random(child) for child in value)
        return False

    states = {symbol["name"] for symbol in synapse["states"]}
    parameters = {symbol["name"] for symbol in synapse["parameters"]}
    allowed = states | parameters | {"dt", "t", "N", "N_pre", "N_post"}
    return (bool(code["effects"]["writes"]) and
            set(code["effects"]["writes"]) <= states and
            set(code["effects"]["reads"]) <= allowed and
            not has_random(code))


def _target_parallel_route(model, members, plastic_calls):
    """Check target-owner safety, including dependencies across projections."""
    definitions = model["definition"]["synapses"]
    source_population = definitions[members[0]]["source_population"]
    pathways = []
    for q in members:
        synapse = definitions[q]
        pathway = next(code for code in synapse["code_objects"]
                       if code["kind"] == "synapses")
        if (q, "synapses") in plastic_calls:
            return False
        pathways.append((
            synapse, pathway,
            synapse["target_population"] == source_population,
            _synapse_edge_count(model["instance"]["synapses"][q]),
        ))
    return target_parallel_plan(pathways).eligible


def _v7_fused_summed_groups(model):
    """Find independent post-summed projections worth one shared dispatch."""
    definitions = model["definition"]
    instances = model["instance"]["synapses"]
    candidates = {}
    for q, synapse in enumerate(definitions["synapses"]):
        readable = ({item["name"] for item in synapse["states"]} |
                    {item["name"] for item in synapse["parameters"]})
        for code in synapse["code_objects"]:
            if (code["kind"] != "summed_variable" or code["order"] >= 0 or
                    code.get("summed_target") != "post" or code.get("scalar")):
                continue
            target_population = definitions["populations"][
                synapse["target_population"]]
            target_state = next(
                (state for state in target_population["states"]
                 if state["name"] == code.get("summed_state")), None)
            if target_state is None:
                raise ValueError(
                    "post-summed code object references a missing target state")
            if (code["clock"] == target_population["clock"] and
                    target_state["dtype"] == "f64" and
                    set(code["effects"]["reads"]) <= readable and
                    not code["scalar"]):
                candidates.setdefault(synapse["target_population"], []).append(
                    (q, code))
    groups = {}
    for target, entries in candidates.items():
        destinations = [
            (entry[1]["summed_state"],
             definitions["synapses"][entry[0]]["target_start"],
             definitions["synapses"][entry[0]]["target_count"])
            for entry in entries]
        edge_count = sum(_synapse_edge_count(instances[q]) for q, _ in entries)
        total_work = sum(
            _synapse_edge_count(instances[q]) * code_work(code)
            for q, code in entries)
        average_work = max(1, (total_work + edge_count - 1) // edge_count)
        schedule_nodes = [
            _schedule_code_node(
                model, "synapse", q,
                next(index for index, item in enumerate(
                     definitions["synapses"][q]["code_objects"])
                     if item is code))
            for q, code in entries
        ]
        if (len(entries) >= 2 and len(set(destinations)) == len(destinations) and
                _schedule_can_fuse_bundles(
                    model, [[node] for node in schedule_nodes]) and
                parallel_phase_capable(edge_count, average_work)):
            groups[target] = entries
    return groups


def _schedule_code_node(model, owner_kind, owner_index, item_index):
    matches = [
        node for node in model["definition"]["schedule"]["nodes"]
        if (node["operation"] == "code_object" and
            node["owner_kind"] == owner_kind and
            node["owner_index"] == owner_index and
            node["item_index"] == item_index)
    ]
    if len(matches) != 1:
        raise ValueError("B2IR schedule does not uniquely reference a code object")
    return matches[0]


def _schedule_effects_conflict(left, right, model):
    """Whether changing the total order of two nodes can change semantics."""
    left_effects = execution_effects(model["definition"], left)
    right_effects = execution_effects(model["definition"], right)
    left_reads = set(left_effects["reads"])
    left_writes = set(left_effects["writes"])
    right_reads = set(right_effects["reads"])
    right_writes = set(right_effects["writes"])
    return bool((left_writes & (right_reads | right_writes)) or
                (right_writes & left_reads))


def fixed_schedule_is_semantically_equivalent(model):
    """Prove that the optimized fixed-phase AOT order matches the IR schedule.

    The current generator emits broad cache-friendly phase loops.  It may use
    that order only when every node that crosses the canonical Brian order is
    independent according to the validated Effect Algebra.  This turns
    formerly implicit phase assumptions into a fail-closed legality proof.
    """
    definition = model["definition"]
    instances = model["instance"]["synapses"]

    def key(node):
        owner = node["owner_index"]
        item = node["item_index"]
        if node["operation"] == "state_monitor":
            return 2, owner, item
        if node["operation"] == "spike_monitor":
            return 8, owner, item
        if node["operation"] == "event_monitor":
            return 9, owner, item
        if node["operation"] == "event_source":
            return 7, owner, item
        if node["operation"] != "code_object":
            raise ValueError("AOT schedule contains an unknown operation")
        if node["owner_kind"] == "population":
            code = definition["populations"][owner]["code_objects"][item]
            ranks = {
                "subexpression_update": 0,
                "state_update": 4,
                "threshold": 7,
                "poisson_input": 12,
                "reset": 14,
                "run_regularly": 15,
            }
            rank = ranks.get(code["kind"])
            if rank is None:
                raise ValueError("AOT schedule contains an unsupported population operation")
            regular_key = ((code["order"], code["name"])
                           if code["kind"] == "run_regularly" else item)
            return rank, owner, regular_key
        code = definition["synapses"][owner]["code_objects"][item]
        if code["kind"] == "synapse_subexpression_update":
            rank = 1
        elif code["kind"] == "summed_variable":
            rank = 3 if code["order"] < 0 else 6
        elif code["kind"] == "synapse_state_update":
            rank = 5
        elif code["kind"] == "synapse_run_regularly":
            rank = 5.5 if code["when"] == "groups" else 16
        elif code["kind"] == "synapses_post":
            rank = 13
        elif code["kind"] == "synapses":
            pathway = next(
                position for position, value in enumerate(instances[owner]["pathways"])
                if value["name"] == code["pathway_name"])
            rank = 10 if pathway == 0 else 11
        else:
            raise ValueError("AOT schedule contains an unsupported synapse operation")
        return rank, owner, item

    canonical = definition["schedule"]["nodes"]
    fixed = sorted(canonical, key=key)
    canonical_position = {node["id"]: position
                          for position, node in enumerate(canonical)}
    for fixed_position, right in enumerate(fixed):
        for left in fixed[:fixed_position]:
            if (canonical_position[left["id"]] >
                    canonical_position[right["id"]] and
                    _schedule_effects_conflict(left, right, model)):
                return False
    return True


def _schedule_can_contract(model, first, second):
    """Prove that ``second`` may move next to ``first`` for kernel fusion."""
    nodes = model["definition"]["schedule"]["nodes"]
    positions = {node["id"]: index for index, node in enumerate(nodes)}
    first_position = positions[first["id"]]
    second_position = positions[second["id"]]
    if (first_position >= second_position or
            first["clock"] != second["clock"]):
        return False
    return all(not _schedule_effects_conflict(node, second, model)
               for node in nodes[first_position + 1:second_position])


def _schedule_can_fuse_bundles(model, bundles):
    """Prove that ordered node bundles may contract into one AOT dispatch.

    Dependencies inside a bundle are retained (for example a population state
    update followed by its threshold). Nodes in different bundles must commute,
    and every selected node must commute with non-selected schedule nodes that
    it crosses while the bundles contract at the first selected position.
    """
    if len(bundles) < 2 or any(not bundle for bundle in bundles):
        return False
    schedule = model["definition"]["schedule"]["nodes"]
    positions = {node["id"]: index for index, node in enumerate(schedule)}
    selected = [node for bundle in bundles for node in bundle]
    if len({node["id"] for node in selected}) != len(selected):
        return False
    clock_definitions = model["definition"].get("clocks")
    run_clocks = model.get("run", {}).get("clocks")
    if clock_definitions is None or run_clocks is None:
        clocks = {node["clock"] for node in selected}
    else:
        clocks = {
            (clock_definitions[node["clock"]]["dt"],
             run_clocks[node["clock"]]["start_tick"],
             run_clocks[node["clock"]]["steps"])
            for node in selected
        }
    if len(clocks) != 1:
        return False
    if any(any(positions[left["id"]] >= positions[right["id"]]
               for left, right in zip(bundle, bundle[1:]))
           for bundle in bundles):
        return False
    if any(_schedule_effects_conflict(left, right, model)
           for index, left_bundle in enumerate(bundles)
           for right_bundle in bundles[index + 1:]
           for left in left_bundle for right in right_bundle):
        return False
    selected_ids = {node["id"] for node in selected}
    anchor = min(positions[node["id"]] for node in selected)
    for node in selected:
        position = positions[node["id"]]
        if any(other["id"] not in selected_ids and
               _schedule_effects_conflict(other, node, model)
               for other in schedule[anchor:position]):
            return False
    return True


def _v7_summed_final_only(model, q, code):
    """Whether a summed state is observable only after the final tick."""
    definition = model["definition"]
    synapse = definition["synapses"][q]
    population_index = (
        synapse["source_population"]
        if code["summed_target"] == "pre"
        else synapse["target_population"])
    state = code["summed_state"]
    resource = f"population/{population_index}/state/{state}"
    # Resolve all readers through the completed whole-model resource graph.
    # Local symbol names miss linked variables in other populations, whose
    # aliases need not share the summed state's name.
    return not any(
        resource in execution_effects(definition, node)["reads"]
        for node in definition["schedule"]["nodes"])


def fixed_phase_eligible(model):
    """Retain proven fixed-phase lowering only within its supported slots."""
    populations = model["definition"]["populations"]
    synapses = model["definition"]["synapses"]
    return (not any(code.get("adaptive") is not None
                    for population in populations
                    for code in population["code_objects"]) and
            not any(synapse.get("state_monitors") or
                    synapse.get("linked_variables")
                    for synapse in synapses) and
            fixed_schedule_is_semantically_equivalent(model) and
            all(m["clock"] == p["clock"]
                for p in populations for m in p.get("state_monitors", [])) and
            all(m["when"] == "after_thresholds" and m["order"] == 1 and
                m["clock"] == p["clock"]
                for p in populations for m in p.get("event_monitors", [])) and
            all((c["kind"] != "threshold" or c["when"] in {"thresholds", "after_thresholds"}) and
                (c["kind"] != "reset" or c["when"] in {"resets", "after_resets"}) and
                (c["kind"] != "run_regularly" or c["when"] == "end")
                for p in populations for c in p["code_objects"]))


def slot_cpu_choices(model):
    """Canonical serial execution: no reorder, fusion, or reduction hoisting."""
    choices = general_cpu_choices(model)
    from .summed_cache import summed_endpoint_caches
    choices["summed_endpoint_caches"] = {
        f"{q}/{index}": caches
        for q, synapse in enumerate(model["definition"]["synapses"])
        for index, code in enumerate(synapse.get("code_objects", ()))
        if (caches := summed_endpoint_caches(model, q, code))
    }
    choices["needs_event_dump"] = any(p.get("events") for p in model["definition"]["populations"])
    old_routes = choices["synapse_routes"]
    old_keys = choices["route_keys"]
    routed_synapses = [
        q for q in range(len(model["definition"]["synapses"]))
        if q in old_routes
    ]
    choices["route_keys"] = [old_keys[old_routes[q]] for q in routed_synapses]
    choices["route_members"] = [[q] for q in routed_synapses]
    choices["synapse_routes"] = {
        q: route for route, q in enumerate(routed_synapses)
    }
    choices["route_parallel"] = [False] * len(choices["route_keys"])
    for key in ("fused_heterogeneous_groups", "fused_population_groups", "plastic_calls", "final_only_summed", "summed_owner_synapses"):
        choices[key] = []
    for key in ("fused_route_members", "fused_route_lookup", "population_parallel_work",
                "population_poisson_work", "fused_population_lookup", "fused_summed_groups"):
        choices[key] = {}
    choices["population_threshold_fusion"] = {p: False for p in range(len(model["definition"]["populations"]))}
    for key in ("parallel_state_capable", "parallel_poisson_capable", "parallel_summed_capable",
                "parallel_plastic_capable", "parallel_capable"):
        choices[key] = False
    return choices


def _uses_v6(model):
    """Use the same lossless legacy projection for code and instance layout."""
    only_synapse = (model["definition"]["synapses"] or [None])[0]
    full_population_synapse = (
        only_synapse is None or
        (only_synapse["source_start"] == only_synapse["target_start"] == 0 and
         only_synapse["source_count"] == only_synapse["target_count"] ==
         model["definition"]["populations"][0]["count"]))
    return (fixed_phase_eligible(model) and
            len(model["definition"]["populations"]) == 1 and
            len(model["definition"]["synapses"]) <= 1 and
            not any("cpu" in function.get("backend_implementations", {})
                    for function in model["definition"].get("functions", [])) and
            model["definition"]["populations"][0]["states"] and
            any(code["kind"] == "state_update" for code in
                model["definition"]["populations"][0]["code_objects"]) and
            not model["definition"]["populations"][0].get("linked_variables") and
            not model["definition"]["populations"][0].get("event_monitors") and
            set(model["definition"]["populations"][0]["events"]) <= {"spike"} and
            all(symbol["dtype"] == "f64" for symbol in
                model["definition"]["populations"][0]["states"] +
                model["definition"]["populations"][0]["parameters"]) and
            all(symbol["dtype"] == "f64" for synapse in
                model["definition"]["synapses"] for symbol in
                synapse["states"] + synapse["parameters"]) and
            (model["definition"]["populations"][0]["refractory"] is None or
             model["definition"]["populations"][0]["refractory"]["mode"] == "fixed") and
            not any(code["kind"] in {"subexpression_update", "run_regularly", "synapse_run_regularly"}
                    for code in
                    model["definition"]["populations"][0]["code_objects"]) and
            set(model["definition"]["populations"][0]["monitor"]["variables"]) <= {
                state["name"] for state in
                model["definition"]["populations"][0]["states"]} and
            all(sum(code["kind"] == "synapses" for code in synapse["code_objects"]) == 1
                    and not any(code["kind"] in {"synapses_post", "summed_variable",
                                                  "synapse_subexpression_update", "synapse_run_regularly"}
                                for code in synapse["code_objects"])
                for synapse in model["definition"]["synapses"]) and
            not _has_runtime_random(model) and
            _number(model["run"]["start"]) == 0 and
            all(not pathway["pending"] for syn in model["instance"]["synapses"]
                for pathway in syn["pathways"]) and
            (not model["instance"]["synapses"] or
             (_explicit_topology(model["instance"]["synapses"][0]) and
              len(model["instance"]["synapses"][0]["source"]) > 0)) and
            full_population_synapse)


def general_cpu_choices(model):
    """Choose the existing general emitter's physical policies exactly once."""
    d, inst = model["definition"], model["instance"]
    populations = d["populations"]
    syn_defs, syn_insts = d["synapses"], inst["synapses"]
    uniform_delays = []
    route_members = []
    route_keys = []
    for q, (syn_def, syn_inst) in enumerate(zip(syn_defs, syn_insts, strict=True)):
        pre_pathways = [pathway for pathway in syn_inst["pathways"]
                        if pathway["kind"] == "pre"]
        if not pre_pathways:
            # Pure summed/state or post-only Synapses have no presynaptic event
            # route. Post pathways use their target-owned CSR below.
            uniform_delays.append(None)
            continue
        delays = pre_pathways[0]["delay_ticks"]
        uniform = (delays[0] if delays and
                   all(value == delays[0] for value in delays) else None)
        uniform_delays.append(uniform)
        event = pre_pathways[0]["event"]
        key = (syn_def["source_population"], syn_def["source_start"],
               syn_def["source_count"], event, uniform)
        if (not route_keys or route_keys[-1] != key or
                route_members[-1][-1] != q - 1):
            route_keys.append(key)
            route_members.append([])
        route_members[-1].append(q)
    plastic_calls = {}
    parallel_plastic_capable = False
    for q, syn_def in enumerate(syn_defs):
        for kind in ("synapses", "synapses_post"):
            codes = [item for item in syn_def["code_objects"]
                     if item["kind"] == kind]
            code = codes[0] if len(codes) == 1 else None
            if code is not None and _plastic_pathway(syn_def, code):
                plastic_calls[q, kind] = True
                edge_count = _synapse_edge_count(syn_insts[q])
                if kind == "synapses_post":
                    items, work = edge_count, post_synapse_work(syn_def, code)
                else:
                    items = syn_def["source_count"]
                    average_degree = max(
                        1, (edge_count + items - 1) // items)
                    work = average_degree * code_work(code)
                parallel_plastic_capable |= parallel_phase_capable(items, work)
    route_parallel = []
    for route, (key, members) in enumerate(
            zip(route_keys, route_members, strict=True)):
        heterogeneous = key[4] is None
        eligible = (_target_parallel_route(model, members, plastic_calls) and
                    (not heterogeneous or not any(
                        syn_insts[q]["pathways"][0]["pending"]
                        for q in members)))
        route_parallel.append(eligible)
    fused_heterogeneous_groups = []
    pending_group = []
    pending_clock = None
    for route, (key, members) in enumerate(
            zip(route_keys, route_members, strict=True)):
        source = key[0]
        clock = (populations[source]["dt"], populations[source]["steps"])
        cross_shard_safe = all(not (
            set(next(code for code in syn_defs[q]["code_objects"]
                     if code["kind"] == "synapses")["effects"]["reads"]) &
            set(syn_defs[q]["pre_state_aliases"])) for q in members)
        consecutive = (not pending_group or
                       route_members[pending_group[-1]][-1] + 1 == members[0])
        eligible = (key[4] is None and route_parallel[route] and
                    cross_shard_safe and consecutive and
                    (pending_clock is None or pending_clock == clock))
        if not eligible:
            if len(pending_group) > 1:
                fused_heterogeneous_groups.append(pending_group)
            pending_group = []
            pending_clock = None
        if key[4] is None and route_parallel[route] and cross_shard_safe:
            if not pending_group:
                pending_clock = clock
            if pending_clock == clock:
                pending_group.append(route)
    if len(pending_group) > 1:
        fused_heterogeneous_groups.append(pending_group)
    fused_route_members = {}
    for group, routes in enumerate(fused_heterogeneous_groups):
        members = [q for route in routes for q in route_members[route]]
        fused_route_members[group] = members
    fused_route_lookup = {
        route: group for group, routes in enumerate(fused_heterogeneous_groups)
        for route in routes}
    synapse_routes = {q: route for route, members in enumerate(route_members)
                      for q in members}
    population_parallel_work = {}
    population_poisson_work = {}
    population_threshold_fusion = {}
    for p, pop in enumerate(populations):
        state_entry = next(((index, code) for index, code in
                            enumerate(pop["code_objects"])
                            if code["kind"] == "state_update"), None)
        threshold_entry = next(((index, code) for index, code in
                                enumerate(pop["code_objects"])
                                if code["kind"] == "threshold"), None)
        state_code = None if state_entry is None else state_entry[1]
        threshold_code = None if threshold_entry is None else threshold_entry[1]
        if state_code is not None:
            combined_work = code_work(state_code)
            if threshold_code is not None:
                combined_work += code_work(threshold_code)
            population_parallel_work[p] = combined_work
            schedule_safe = False
            if threshold_entry is not None:
                state_node = _schedule_code_node(
                    model, "population", p, state_entry[0])
                threshold_node = _schedule_code_node(
                    model, "population", p, threshold_entry[0])
                schedule_safe = _schedule_can_contract(
                    model, state_node, threshold_node)
            population_threshold_fusion[p] = (
                threshold_code is not None and pop.get("events") == ["spike"] and
                sum(code["kind"] == "threshold"
                    for code in pop["code_objects"]) == 1 and schedule_safe and
                (pop["refractory"] is None or
                 pop["refractory"]["mode"] == "fixed") and
                parallel_phase_capable(pop["count"], combined_work))
        poisson_work = [code_work(code) for code in pop["code_objects"]
                        if code["kind"] == "poisson_input"]
        if poisson_work:
            population_poisson_work[p] = poisson_work
    parallel_state_capable = any(
        parallel_phase_capable(populations[p]["count"], work)
        for p, work in population_parallel_work.items())
    parallel_poisson_capable = any(
        parallel_phase_capable(populations[p]["count"], work)
        for p, works in population_poisson_work.items() for work in works)
    fused_summed_groups = _v7_fused_summed_groups(model)
    parallel_summed_capable = bool(fused_summed_groups) or any(
        code["summed_target"] == "post" and
        parallel_phase_capable(_synapse_edge_count(syn_insts[q]), code_work(code))
        for q, syn_def in enumerate(syn_defs)
        for code in syn_def["code_objects"]
        if code["kind"] == "summed_variable")
    population_clock_groups = {}
    for p, work in population_parallel_work.items():
        if parallel_phase_capable(populations[p]["count"], work):
            key = (populations[p]["dt"], populations[p]["steps"])
            population_clock_groups.setdefault(key, []).append(p)
    fused_population_groups = []
    for group in population_clock_groups.values():
        bundles = []
        for p in group:
            state_index = next(
                index for index, code in enumerate(populations[p]["code_objects"])
                if code["kind"] == "state_update")
            bundle = [_schedule_code_node(
                model, "population", p, state_index)]
            if population_threshold_fusion[p]:
                threshold_index = next(
                    index for index, code in enumerate(
                        populations[p]["code_objects"])
                    if code["kind"] == "threshold")
                bundle.append(_schedule_code_node(
                    model, "population", p, threshold_index))
            bundles.append(bundle)
        if _schedule_can_fuse_bundles(model, bundles):
            fused_population_groups.append(group)
    fused_population_lookup = {
        p: group for group in fused_population_groups for p in group}
    parallel_regular_capable = any(
        _v7_regular_parallel_capable(model, q, code)
        for q, synapse in enumerate(syn_defs)
        for code in synapse["code_objects"]
        if code["kind"] == "synapse_run_regularly")
    parallel_synapse_state_capable = any(
        _v7_regular_parallel_capable(model, q, code)
        for q, synapse in enumerate(syn_defs)
        for code in synapse["code_objects"]
        if code["kind"] == "synapse_state_update")
    parallel_capable = (parallel_regular_capable or
                        parallel_synapse_state_capable or
                        parallel_state_capable or parallel_poisson_capable or
                        parallel_plastic_capable or parallel_summed_capable or
                        any(route_parallel) or any(
                            not _explicit_topology(synapse)
                            for synapse in syn_insts))
    summed_owner_synapses = [q for q, synapse in enumerate(syn_defs)
        if any(code["kind"] == "summed_variable" and code["summed_target"] == "post" and
               parallel_phase_capable(_synapse_edge_count(syn_insts[q]), code_work(code))
               for code in synapse["code_objects"])]
    return {
        "summed_owner_synapses": summed_owner_synapses,
        "needs_event_dump": any(any(e != "spike" for e in p.get("events", [])) or
                                p.get("event_monitors") for p in populations),
        "uniform_delays": uniform_delays,
        "route_members": route_members,
        "route_keys": route_keys,
        "route_parallel": route_parallel,
        "fused_heterogeneous_groups": fused_heterogeneous_groups,
        "fused_route_members": fused_route_members,
        "fused_route_lookup": fused_route_lookup,
        "synapse_routes": synapse_routes,
        "population_parallel_work": population_parallel_work,
        "population_poisson_work": population_poisson_work,
        "population_threshold_fusion": population_threshold_fusion,
        "parallel_state_capable": parallel_state_capable,
        "parallel_poisson_capable": parallel_poisson_capable,
        "parallel_summed_capable": parallel_summed_capable,
        "parallel_plastic_capable": parallel_plastic_capable,
        "parallel_synapse_state_capable": parallel_synapse_state_capable,
        "fused_population_groups": fused_population_groups,
        "fused_population_lookup": fused_population_lookup,
        "parallel_capable": parallel_capable,
        "plastic_calls": list(plastic_calls),
        "fused_summed_groups": {
            target: [(q, syn_defs[q]["code_objects"].index(code))
                     for q, code in entries]
            for target, entries in fused_summed_groups.items()
        },
        "final_only_summed": [
            [q, index] for q, synapse in enumerate(syn_defs)
            for index, code in enumerate(synapse["code_objects"])
            if code["kind"] == "summed_variable" and
            _v7_summed_final_only(model, q, code)
        ],
    }


def compact_cpu_choices(model):
    """Policies for the lossless v6 projection; the input remains B2IR v1."""
    d, inst = model["definition"], model["instance"]
    pop = d["populations"][0]
    synapse = d["synapses"][0] if d["synapses"] else None
    syn_inst = inst["synapses"][0] if synapse is not None else None
    edge_count = _synapse_edge_count(syn_inst) if syn_inst is not None else 0
    delays = syn_inst["pathways"][0]["delay_ticks"] if syn_inst is not None else []
    uniform = delays[0] if delays and all(v == delays[0] for v in delays) else None
    code = next(c for c in pop["code_objects"] if c["kind"] == "state_update")
    work = code_work(code)
    state = parallel_phase_capable(pop["count"], work)
    threshold = any(c["kind"] == "threshold" for c in pop["code_objects"])
    event = (target_parallel_plan([
        (synapse, next(c for c in synapse["code_objects"] if c["kind"] == "synapses"),
         True, edge_count)]) if synapse is not None and threshold and uniform is not None
        else TargetParallelPlan(edge_count, False, "no uniform threshold route"))
    return {
        "source_batch_min_edges": SOURCE_BATCH_MIN_EDGES,
        "state_work": work, "parallel_state_capable": state,
        "parallel_event_capable": event.eligible,
        "parallel_capable": state or event.eligible,
        "event_reason": event.reason,
        "pack_uniform_edges": synapse is not None and uniform is not None and not synapse["states"],
        "uniform_delay": uniform, "delay_groups": sorted(set(delays), reverse=True),
    }
