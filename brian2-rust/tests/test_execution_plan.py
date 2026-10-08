"""Executable plans must agree with validated semantics and actual codegen."""
import copy
from dataclasses import FrozenInstanceError, replace
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from brian2_rust.plan import (build_execution_plan, explain_plan, PlanValidationError,
                             verify_execution_plan)
from brian2_rust import planner
from brian2_rust.native import generate_source, write_project
from brian2_rust.protocol import attach_protocol, canonical_bytes


@pytest.fixture
def model():
    return json.loads((ROOT / "tests/golden/b2ir-v1/minimal-v1.json").read_text())


def test_plan_preserves_frozen_bytes_and_completes_implicit_effects(model):
    before = canonical_bytes(model)
    plan = build_execution_plan(model)
    assert canonical_bytes(model) == before
    assert build_execution_plan(model).sha256 == plan.sha256
    threshold = next(n for n in plan.logical.nodes if "/event/spike" in " ".join(n.writes))
    assert "population/0/refractory/lastspike" in threshold.writes
    assert "population/0/refractory/not_refractory" in threshold.writes
    updater = next(n for n in plan.logical.nodes
                   if "population/0/refractory/lastspike" in n.reads)
    assert updater.id in threshold.dependencies
    with pytest.raises(FrozenInstanceError):
        plan.schema = "forged"
    detached = plan.cpu.choices()
    detached["parallel_capable"] = "forged"
    assert plan.cpu.choices().get("parallel_capable") != "forged"


def test_plan_and_model_drift_fail_before_emission(model, tmp_path):
    plan = build_execution_plan(model)
    bad = replace(plan, cpu=replace(plan.cpu, choices_json='{"forged":true}'))
    with pytest.raises(PlanValidationError):
        write_project(model, tmp_path / "bad", plan=bad)
    assert not (tmp_path / "bad").exists()
    changed = copy.deepcopy(model)
    changed["instance"]["rng_seed"] += 1
    attach_protocol(changed)
    with pytest.raises(PlanValidationError):
        generate_source(changed, plan=plan)
    missing = replace(plan, logical=replace(plan.logical, nodes=plan.logical.nodes[1:]))
    with pytest.raises(PlanValidationError):
        verify_execution_plan(missing, model)


def test_public_builder_checks_semantics_not_only_hashes(model):
    bad = copy.deepcopy(model)
    bad["definition"]["schedule"]["nodes"][0]["effects"]["reads"] = ["forged"]
    attach_protocol(bad)
    with pytest.raises(PlanValidationError):
        build_execution_plan(bad)


def test_sidecar_describes_actual_source_and_runtime_is_unbound(model, tmp_path):
    plan = build_execution_plan(model)
    source, _, _ = write_project(model, tmp_path / "native", plan=plan)
    assert source.read_text() == generate_source(model)
    assert json.loads((source.parent / "execution-plan.json").read_text()) == json.loads(plan.to_json())
    assert "runtime unbound" in explain_plan(plan)
    assert "declared payload" in explain_plan(plan)
    assert json.loads(explain_plan(plan, format="json")) == json.loads(plan.to_json())
    with pytest.raises(ValueError):
        explain_plan(plan, format="xml")


def test_runtime_observations_are_bound_to_one_plan(model):
    from brian2_rust.plan import bind_execution_plan
    plan = build_execution_plan(model)
    metadata = {"threads": 1, "thread_affinity": False, "thread_cpus": [],
                "parallel_state_update": False, "parallel_on_pre": False}
    binding = bind_execution_plan(plan, metadata, requested_threads=8, requested_affinity="auto")
    observed = explain_plan(plan, format="dict", binding=binding)["runtime_binding"]["observed"]
    assert observed["requested_threads"] == 8
    assert observed["threads"] == 1
    assert observed["owner_map"] == "not_instrumented"
    assert "runtime observed" in explain_plan(plan, binding=binding)
    with pytest.raises(PlanValidationError, match="different plan"):
        explain_plan(plan, binding=replace(binding, plan_sha256="0"*64))
    with pytest.raises(PlanValidationError, match="worker"):
        bind_execution_plan(plan, {"threads": True})


def test_artifact_policy_drift_rejected_even_if_source_unchanged(model, tmp_path):
    from brian2_rust.artifact import write_compatible_instance, ArtifactCompatibilityError
    _, _, manifest = write_project(model, tmp_path/"native")
    manifest["plan_policy_sha256"] = "0"*64
    (tmp_path/"native/manifest.json").write_text(json.dumps(manifest))
    destination = tmp_path/"replacement.bin"
    with pytest.raises(ArtifactCompatibilityError, match="plan policy"):
        write_compatible_instance(model, tmp_path/"native", destination)
    assert not destination.exists()


def test_slot_plan_skips_synapses_without_event_routes(monkeypatch):
    """Pure state/summed synapses do not have a pre/post event route."""
    monkeypatch.setattr(
        planner,
        "general_cpu_choices",
        lambda _model: {
            "synapse_routes": {0: 0, 2: 0},
            "route_keys": [("population/0/event/spike", 0)],
        },
    )
    slot_model = {
        "definition": {
            "populations": [{}],
            "synapses": [{}, {}, {}],
        }
    }

    choices = planner.slot_cpu_choices(slot_model)

    assert choices["route_keys"] == [
        ("population/0/event/spike", 0),
        ("population/0/event/spike", 0),
    ]
    assert choices["route_members"] == [[0], [2]]
    assert choices["synapse_routes"] == {0: 0, 2: 1}
    assert 1 not in choices["synapse_routes"]
