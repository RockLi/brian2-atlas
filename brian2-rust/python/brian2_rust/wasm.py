"""Browser physical plans and self-contained AtlasIR/plan bundles.

Only portable reference semantics are selected. Browser-side Rust independently
revalidates both the wire model and the entire logical/physical plan.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

from .plan import LogicalPlan, PlanValidationError, _logical_plan, validate_model
from .protocol import canonical_bytes


@dataclass(frozen=True)
class WasmPlan:
    schema: str
    planner_version: str
    definition_sha256: str
    instance_sha256: str
    run_sha256: str
    logical: LogicalPlan
    numeric_profile: str = "reference-f64"
    strategy: str = "serial-tiled-v1"

    def to_dict(self):
        return asdict(self)

    def to_json(self, *, indent=2):
        return json.dumps(self.to_dict(), sort_keys=True, indent=indent,
                          ensure_ascii=False, allow_nan=False) + "\n"

    @property
    def sha256(self):
        return hashlib.sha256(canonical_bytes(self.to_dict())).hexdigest()


def _derive_wasm_plan(model):
    for synapse in model["instance"]["synapses"]:
        if synapse.get("topology", {}).get("kind") == "binary_csr":
            raise PlanValidationError("WASM requires inline or procedural topology; binary CSR file paths are unsupported")
    if any(function["body"] is None for function in model["definition"]["functions"]):
        raise PlanValidationError("WASM requires portable Function bodies; native-only Functions are unsupported")
    hashes = model["protocol"]["layers"]
    return WasmPlan("b2-wasm-plan-v0", "wasm-plan-1", hashes["definition"],
                    hashes["instance"], hashes["run"], _logical_plan(model))


def build_wasm_plan(model, *, runner=None, numeric_mode=None):
    if numeric_mode not in (None, "reference-f64"):
        raise PlanValidationError("WASM planning requires reference-f64")
    return _derive_wasm_plan(validate_model(model, runner=runner))


def export_wasm_bundle(model, path, *, runner=None, plan=None):
    """Write a portable bundle for BrowserExecutor, returning its verified plan."""
    current = validate_model(model, runner=runner)
    expected = _derive_wasm_plan(current)
    if plan is not None and (not isinstance(plan, WasmPlan) or
            canonical_bytes(plan.to_dict()) != canonical_bytes(expected.to_dict())):
        raise PlanValidationError("WASM execution plan does not match model or compiler policy")
    Path(path).write_bytes(canonical_bytes({"schema": "b2-wasm-bundle-v0",
        "model_json": canonical_bytes(current).decode(),
        "plan_json": canonical_bytes(expected.to_dict()).decode(), "plan_sha256": expected.sha256}))
    return expected
