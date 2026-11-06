"""AOT artifact compatibility and instance-only replacement."""

import copy
import hashlib
import json
import sys
from pathlib import Path

import brian2 as b
import pytest
from brian2.devices.device import all_devices

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import brian2_rust  # noqa: E402
from brian2_rust.export import lower_network  # noqa: E402
from brian2_rust.native import generate_source, write_project  # noqa: E402
from brian2_rust.protocol import attach_protocol  # noqa: E402


def model_with_gain(gain):
    group = b.NeuronGroup(
        4,
        "dv/dt = -gain*v/(2*ms) : 1\ngain : 1 (constant, shared)",
        method="euler",
        dt=0.1 * b.ms,
        name="artifact_group",
    )
    group.v = 1
    group.gain = gain
    monitor = b.StateMonitor(group, "v", record=[0], name="artifact_monitor")
    return b.Network(group, monitor)


@b.implementation("b2ir-c-abi-v1", "double artifact_identity(double x) { return x; }",
                  name="artifact_identity")
@b.check_units(x=1, result=1)
def native_identity(x):
    return x


def test_native_function_drift_is_rejected_before_instance_write(tmp_path):
    previous = b.get_device()
    device = all_devices["rust_standalone"]
    try:
        device.reinit()
        b.set_device("rust_standalone", runner=ROOT / "target/release/b2-runner")
        group = b.NeuronGroup(
            1, "dv/dt=native_identity(v)/ms:1", method="euler", dt=.1*b.ms,
            namespace={"native_identity": native_identity})
        baseline = lower_network(b.Network(group), .2*b.ms)
        native = tmp_path / "native"
        write_project(baseline, native)
        changed = copy.deepcopy(baseline)
        implementation = changed["definition"]["functions"][0]["backend_implementations"]["cpu"]
        implementation["source"] = implementation["source"].replace("return x;", "return 0.0;")
        implementation["source_sha256"] = hashlib.sha256(implementation["source"].encode()).hexdigest()
        attach_protocol(changed)
        assert generate_source(changed) == generate_source(baseline)
        destination = tmp_path / "replacement.bin"
        with pytest.raises(brian2_rust.ArtifactCompatibilityError, match="definition.*incompatible"):
            brian2_rust.write_compatible_instance(changed, native, destination)
        assert not destination.exists()
        # Artifacts made before semantic hashes were recorded cannot prove
        # compatibility, even if their Rust source hash happens to match.
        manifest_path = native / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest.pop("definition_sha256")
        manifest_path.write_text(json.dumps(manifest))
        with pytest.raises(brian2_rust.ArtifactCompatibilityError, match="rebuild"):
            brian2_rust.write_compatible_instance(baseline, native, destination)
        assert not destination.exists()
    finally:
        b.set_device(previous)
        device.reinit()


def test_instance_change_is_accepted_and_definition_drift_is_rejected(tmp_path):
    previous = b.get_device()
    device = all_devices["rust_standalone"]
    try:
        device.reinit()
        b.start_scope()
        b.set_device("rust_standalone", runner=ROOT / "target/release/b2-runner")
        baseline = lower_network(model_with_gain(1.0), 1 * b.ms)
        native = tmp_path / "native"
        _, original_instance, manifest = write_project(baseline, native)

        changed = copy.deepcopy(baseline)
        changed["instance"]["populations"][0]["parameters"]["gain"] = [
            "4000000000000000"]  # 2.0 as IEEE-754 bits
        replacement = tmp_path / "gain-2.bin"
        metadata = brian2_rust.write_compatible_instance(
            changed, native, replacement)
        assert replacement.read_bytes() != original_instance.read_bytes()
        assert metadata["source_sha256"] == manifest["source_sha256"]
        assert metadata["instance_sha256"] != manifest["instance_sha256"]

        drifted = copy.deepcopy(changed)
        drifted["definition"]["populations"][0]["steps"] += 1
        with pytest.raises(
                brian2_rust.ArtifactCompatibilityError, match="incompatible"):
            brian2_rust.write_compatible_instance(
                drifted, native, tmp_path / "invalid.bin")

        saved = json.loads((native / "manifest.json").read_text())
        assert saved == manifest
    finally:
        b.set_device(previous)
        device.reinit()
        b.start_scope()
