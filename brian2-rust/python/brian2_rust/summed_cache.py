"""Pure endpoint-expression reuse inside one canonical summed activation.

The reduction retains its original edge order. Only f64 exponentials whose
inputs stay unchanged throughout this node may move to an endpoint loop.
No model values or names specific to a published workload enter this policy.
"""
from __future__ import annotations

import json


def expression_key(node):
    return json.dumps(node, sort_keys=True, separators=(",", ":"))


def summed_endpoint_caches(model, q, code):
    """Describe profitable, side-effect-free exp caches; otherwise return []."""
    if (code["kind"] != "summed_variable" or code["effects"]["writes"] or
            any(statement.get("condition") for statement in code["vector"])):
        return []
    synapse = model["definition"]["synapses"][q]
    endpoint = code["summed_target"]
    side = "source" if endpoint == "pre" else "target"
    count = synapse[f"{side}_count"]
    # A cache adds a vector allocation and an indirect read. Avoid sparse or
    # tiny projections where repeated arithmetic is the cheaper policy.
    from .planner import _synapse_edge_count
    edges = _synapse_edge_count(model["instance"]["synapses"][q])
    if count <= 0 or edges < max(64, 4 * count):
        return []
    population = model["definition"]["populations"][synapse[f"{side}_population"]]
    f64_states = {s["name"] for s in population["states"] if s["dtype"] == "f64"}
    aliases = {name for name, state in synapse[f"{endpoint}_state_aliases"].items()
               if state in f64_states and state != code["summed_state"]}
    stable = {s["name"] for s in synapse["parameters"]
              if s["index_domain"] == "scalar" and s["dtype"] == "f64"}
    stable.update(s["target"] for s in code["scalar"] if s["dtype"] == "f64")
    # A vector temporary can shadow a parameter or scalar temporary. Reject
    # that dependency rather than reusing a value from the wrong assignment.
    rebound = {s["target"] for s in code["vector"]}
    aliases -= rebound
    stable -= rebound

    def pure(node):
        op = node.get("op")
        if op == "literal":
            return True, set()
        if op == "load":
            name = node["name"]
            return name in aliases | stable, {name}
        if op in {"neg", "exp"}:
            return pure(node["arg"])
        if op in {"add", "sub", "mul", "div"}:
            a, left = pure(node["left"])
            b, right = pure(node["right"])
            return a and b, left | right
        # RNG, Function calls, linked arrays, casts and indexing remain in the
        # edge loop, even when a less conservative proof might be possible.
        return False, set()

    found = {}

    def visit(node):
        if not isinstance(node, dict):
            return
        if node.get("op") == "exp":
            legal, reads = pure(node)
            if legal and reads & aliases:
                found.setdefault(expression_key(node), dict(
                    expression=node, endpoint=endpoint,
                    aliases=sorted(reads & aliases), count=count,
                    begin=synapse[f"{side}_start"]))
                return
        for child in node.values():
            if isinstance(child, dict):
                visit(child)

    for statement in code["vector"]:
        visit(statement["value"])
    return list(found.values())
