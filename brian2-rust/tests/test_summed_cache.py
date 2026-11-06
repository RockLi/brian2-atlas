"""Endpoint reuse must reject edge-dependent and reduction-dependent inputs."""
import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from brian2_rust.summed_cache import summed_endpoint_caches
from brian2_rust.native import _v7_block
from brian2_rust.plan import _buffers


def load(name):
    return {"op": "load", "name": name}


@pytest.fixture
def case():
    def symbol(name, dtype="f64", domain="neuron"):
        return dict(name=name, dtype=dtype, index_domain=domain)
    population = dict(count=8, states=[symbol("v"), symbol("total")],
                      parameters=[], refractory=None)
    expression = dict(op="exp", arg=dict(op="mul", left=load("scale"),
                                        right=load("v_post")))
    code = dict(kind="summed_variable", summed_target="post",
                summed_state="total", effects=dict(reads=["v_post", "scale", "w"], writes=[]),
                scalar=[dict(target="scale", dtype="f64", condition=None,
                             value=dict(op="neg", arg=load("c")))],
                vector=[dict(target="_synaptic_var", dtype="f64", condition=None,
                             value=dict(op="mul", left=load("w"), right=expression))])
    synapse = dict(source_population=0, target_population=1,
                   source_start=0, target_start=2, source_count=8, target_count=4,
                   pre_state_aliases={"v_pre": "v", "total_pre": "total"},
                   post_state_aliases={"v_post": "v", "total_post": "total"},
                   states=[symbol("w", domain="synapse")],
                   parameters=[symbol("c", domain="scalar")], code_objects=[code])
    model = dict(definition=dict(populations=[copy.deepcopy(population), population],
                                 synapses=[synapse], functions=[]),
                 instance=dict(populations=[{}, {}], synapses=[dict(source=[0] * 80, target=[0] * 80,
                                             pathways=[])]))
    return model, code, expression


def test_cache_has_endpoint_range_and_deduplicates_expression(case):
    model, code, expression = case
    code["vector"][0]["value"] = dict(op="add", left=expression, right=copy.deepcopy(expression))
    caches = summed_endpoint_caches(model, 0, code)
    assert len(caches) == 1
    assert caches[0]["begin"] == 2 and caches[0]["count"] == 4
    assert caches[0]["aliases"] == ["v_post"]


@pytest.mark.parametrize("name", ["w", "v_pre", "total_post", "j", "unknown"])
def test_edge_other_endpoint_or_reduction_reads_stay_inside_loop(case, name):
    model, code, expression = case
    expression["arg"] = load(name)
    assert summed_endpoint_caches(model, 0, code) == []


@pytest.mark.parametrize("arg", [dict(op="rand", stream=1),
                                 dict(op="call", function="external", arguments=[]),
                                 dict(op="cast", dtype="f64", arg=load("v_post"))])
def test_random_calls_and_casts_are_not_hoisted(case, arg):
    model, code, expression = case
    expression["arg"] = arg
    assert summed_endpoint_caches(model, 0, code) == []


@pytest.mark.parametrize("change", ["parameter", "state_dtype", "condition", "rebind", "write"])
def test_unstable_or_unsupported_inputs_reject_cache(case, change):
    model, code, expression = case
    if change == "parameter":
        model["definition"]["synapses"][0]["parameters"][0]["index_domain"] = "synapse"
        expression["arg"]["left"] = load("c")
    elif change == "state_dtype":
        model["definition"]["populations"][1]["states"][0]["dtype"] = "f32"
    elif change == "condition":
        code["vector"][0]["condition"] = "guard"
    elif change == "rebind":
        code["vector"].insert(0, dict(target="scale", dtype="f64", condition=None, value=load("w")))
    else:
        code["effects"]["writes"] = ["v_post"]
    assert summed_endpoint_caches(model, 0, code) == []


@pytest.mark.parametrize("edges", [0, 1, 15, 63])
def test_empty_and_small_projections_keep_original_loop(case, edges):
    model, code, _ = case
    model["instance"]["synapses"][0]["source"] = [0] * edges
    assert summed_endpoint_caches(model, 0, code) == []


def test_presynaptic_sum_uses_presynaptic_identity(case):
    model, code, expression = case
    code["summed_target"] = "pre"
    expression["arg"]["right"] = load("v_pre")
    caches = summed_endpoint_caches(model, 0, code)
    assert caches[0]["endpoint"] == "pre" and caches[0]["count"] == 8
    scalar, vector, _ = _v7_block(model, code, 0, "synapse", "edge", 0, summed_caches=caches)
    assert "sum_cache_0_0[source]" in "\n".join(vector)
    assert "p0_state_0[sum_cache_state]" in "\n".join(scalar)


def test_cache_refreshes_in_node_and_does_not_shadow_scalar_cse(case):
    model, code, _ = case
    caches = summed_endpoint_caches(model, 0, code)
    scalar, vector, _ = _v7_block(model, code, 0, "synapse", "edge", 0, summed_caches=caches)
    header, body = "\n".join(scalar), "\n".join(vector)
    assert header.index("let r1 =") < header.index("let sum_cache_0_0:")
    assert "(2..6).map(|sum_cache_state|" in header
    assert "p1_state_0[sum_cache_state]" in header
    assert "let sum_cache_0_0_r1 = r1 *" in header
    assert header.count(".exp()") == 1 and ".exp()" not in body
    assert "sum_cache_0_0[target]" in body
    _, uncached, _ = _v7_block(model, code, 0, "synapse", "edge", 0)
    assert ".exp()" in "\n".join(uncached)
    buffers = _buffers(model, {"summed_endpoint_caches": {"0/0": caches}})
    scratch = next(b for b in buffers if b.role == "endpoint_expression_cache")
    assert scratch.elements == 4 and scratch.payload_bytes == 32
    assert scratch.lifetime == "node_activation"
