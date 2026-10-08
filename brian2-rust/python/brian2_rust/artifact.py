"""Validated instance replacement for a compiled brian2-rust AOT artifact."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import copy
from pathlib import Path

from .native import generate_source, write_instance
from .protocol import CURRENT_SCHEMA, attach_protocol, migrate_model
from .results import load_results
from .plan import build_execution_plan, bind_execution_plan, PlanValidationError


class ArtifactCompatibilityError(ValueError):
    """Raised when model code does not match an existing native artifact."""


def _manifest(native_directory: Path) -> dict:
    path = native_directory / "manifest.json"
    try:
        manifest = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ArtifactCompatibilityError(
            f"cannot read AOT manifest: {path}") from error
    if (manifest.get("schema") != "b2-native-probe-v0" or
            manifest.get("engine") != "rustc-aot" or
            not isinstance(manifest.get("source_sha256"), str)):
        raise ArtifactCompatibilityError("invalid AOT artifact manifest")
    return manifest


def compatible_source_sha256(model: dict) -> str:
    """Return the hash of the exact Rust source generated for ``model``."""
    return hashlib.sha256(generate_source(model).encode()).hexdigest()


def write_compatible_instance(
    model: dict,
    native_directory: str | Path,
    destination: str | Path,
) -> dict:
    """Write instance data after proving Definition, Run and code compatibility.

    This permits hot replacement of values already represented in B2IR's
    instance layer. Definition/Run changes (including external C Function
    implementations) and changes to specialized Rust code are rejected.
    """
    native_directory = Path(native_directory)
    destination = Path(destination)
    manifest = _manifest(native_directory)
    candidate = copy.deepcopy(model)
    if candidate.get("schema") == CURRENT_SCHEMA:
        # This API is the explicit mutation boundary for Instance replacement.
        # Re-seal the requested values, then compare immutable layer identities
        # with the artifact before writing anything. Rust source alone does not
        # include native C translation units or all semantic metadata.
        attach_protocol(candidate)
    else:
        candidate = migrate_model(candidate)
    for layer in ("definition", "run"):
        expected = manifest.get(f"{layer}_sha256")
        if expected is None:
            raise ArtifactCompatibilityError(
                "artifact lacks semantic layer hashes; rebuild the AOT artifact")
        if candidate["protocol"]["layers"][layer] != expected:
            raise ArtifactCompatibilityError(
                f"model {layer} is incompatible with the compiled AOT artifact")
    try:
        plan = build_execution_plan(candidate)
    except (PlanValidationError, ValueError) as error:
        raise ArtifactCompatibilityError(f"replacement instance is invalid: {error}") from error
    if manifest.get("plan_policy_sha256") != plan.policy_sha256:
        raise ArtifactCompatibilityError("execution plan policy is incompatible; rebuild the AOT artifact")
    actual_source = compatible_source_sha256(candidate)
    if actual_source != manifest["source_sha256"]:
        raise ArtifactCompatibilityError(
            "model definition is incompatible with the compiled AOT artifact")
    destination.parent.mkdir(parents=True, exist_ok=True)
    instance_sha256 = write_instance(candidate, destination)
    return {
        "source_sha256": actual_source,
        "execution_plan_sha256": plan.sha256,
        "plan_policy_sha256": plan.policy_sha256,
        "instance_sha256": instance_sha256,
        "path": str(destination),
    }


def run_compatible_instance(
    model: dict,
    native_directory: str | Path,
    instance_path: str | Path,
    output_directory: str | Path,
    *,
    threads: int = 1,
    thread_affinity: str = "auto",
) -> dict:
    """Validate, write and execute one instance with an existing AOT binary."""
    if type(threads) is not int or not 1 <= threads <= 256:
        raise ValueError("threads must be an integer in 1..256")
    native_directory = Path(native_directory)
    output_directory = Path(output_directory)
    if output_directory.exists():
        raise FileExistsError(f"output directory already exists: {output_directory}")
    binary = native_directory / ("b2-native.exe" if os.name == "nt" else "b2-native")
    if not binary.is_file():
        raise FileNotFoundError(f"missing AOT executable: {binary}")
    instance = write_compatible_instance(
        model, native_directory, instance_path)
    environment = os.environ.copy()
    environment["B2_NUM_THREADS"] = str(threads)
    environment["B2_THREAD_AFFINITY"] = thread_affinity
    result = subprocess.run(
        [str(binary), str(instance_path), str(output_directory)],
        env=environment, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(
            f"AOT instance execution failed:\n{result.stderr or result.stdout}")
    loaded = load_results(model, output_directory)
    candidate = copy.deepcopy(model)
    attach_protocol(candidate)
    plan = build_execution_plan(candidate)
    binding = bind_execution_plan(plan, loaded["metadata"], requested_threads=threads,
                                  requested_affinity=thread_affinity)
    (output_directory / "execution-plan.json").write_text(plan.to_json())
    (output_directory / "runtime-binding.json").write_text(binding.to_json())
    return {"instance": instance, "results": loaded, "runtime_binding": binding.to_dict()}
