"""Frozen B2IR v1 byte, migration and cross-language conformance vectors."""

from __future__ import annotations

import hashlib
import copy
import json
import os
import subprocess
import sys
import tempfile
import random
import struct
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "tests" / "golden" / "b2ir-v1"
sys.path.insert(0, str(ROOT / "python"))

from brian2_rust.protocol import (  # noqa: E402
    attach_protocol,
    canonical_bytes,
    layer_hashes,
    migrate_model,
    verify_protocol,
)
from brian2_rust.schedule import execution_effects  # noqa: E402
from brian2_rust.native import _schedule_can_contract, _schedule_can_fuse_bundles  # noqa: E402


def load(name: str) -> dict:
    return json.loads((GOLDEN / name).read_text())


def test_frozen_v1_canonical_bytes_and_python_hashes():
    model = load("minimal-v1.json")
    expected = load("hashes.json")
    assert model["schema"] == expected["schema"] == "b2ir-v1"
    verify_protocol(model)
    assert layer_hashes(model) == expected["layers"]
    assert hashlib.sha256(canonical_bytes(model)).hexdigest() == expected[
        "canonical_document_sha256"]


def test_probe_migration_corpus_converges_to_identical_v1():
    expected = load("minimal-v1.json")
    for version in (34, 35, 36, 37):
        assert migrate_model(load(f"minimal-v{version}.json")) == expected


def test_rust_agrees_on_hashes_and_executes_every_migration_vector():
    runner = ROOT / "target" / "release" / "b2-runner"
    assert runner.is_file(), "build the release Rust runner before tests"
    expected = load("hashes.json")["layers"]
    with tempfile.TemporaryDirectory() as temporary:
        temporary = Path(temporary)
        for name in ("minimal-v1.json", "minimal-v34.json", "minimal-v35.json",
                     "minimal-v36.json", "minimal-v37.json"):
            source = GOLDEN / name
            checked = subprocess.run(
                [str(runner), "--canonical-hashes", str(source)],
                capture_output=True, text=True, check=True)
            assert json.loads(checked.stdout) == expected
            output = temporary / name.removesuffix(".json")
            executed = subprocess.run(
                [str(runner), str(source), str(output)],
                env={**os.environ, "PATH": ""}, capture_output=True, text=True)
            assert executed.returncode == 0, executed.stderr
            assert (output / "results.bin").is_file()
            assert (output / "summary.json").is_file()


@pytest.mark.parametrize("version,corruption", [
    (version, corruption)
    for version in (34, 35, 36, 37, 1)
    for corruption in ("missing", "extra", "version", "hash")
    if (version, corruption) != (34, "missing")
])
def test_python_and_rust_reject_invalid_envelopes(tmp_path, version, corruption):
    model = load(f"minimal-v{version}.json")
    if version == 34:
        # V34 is the only schema that must not have an envelope at all.
        model["protocol"] = copy.deepcopy(load("minimal-v1.json")["protocol"])
    if corruption == "missing":
        model.pop("protocol")
    elif corruption == "extra":
        model["protocol"]["unexpected"] = True
    elif corruption == "version":
        model["protocol"]["version"]["major"] = True
    else:
        model["protocol"]["layers"]["definition"] = "0" * 64
    with pytest.raises(ValueError):
        migrate_model(model)
    source = tmp_path / "invalid.json"
    source.write_text(json.dumps(model))
    output = tmp_path / "results"
    checked = subprocess.run(
        [str(ROOT / "target/release/b2-runner"), str(source), str(output)],
        capture_output=True, text=True)
    assert checked.returncode != 0
    assert not output.exists()


def test_cross_language_dimension_numbers_round_trip(tmp_path):
    model = load("minimal-v1.json")
    values = [1e-7, -1e-5, 0.0001, -0.0, 5e-324, 1e-300,
              1.2345678901234567e-6, 999999.9999999999]
    rng = random.Random(8107)
    while len(values) < 7 * 126:
        value = struct.unpack("!d", rng.randbytes(8))[0]
        if abs(value) <= 1e6:
            values.append(value)
    for index in range(126):
        name = f"dimension_probe_{index}"
        model["definition"]["populations"][0]["parameters"].append({
            "name": name, "dtype": "f64", "index_domain": "scalar",
            "dimensions": values[index * 7:(index + 1) * 7],
        })
        model["instance"]["populations"][0]["parameters"][name] = ["0000000000000000"]
    attach_protocol(model)
    source = tmp_path / "dimensions.json"
    source.write_text(json.dumps(model))
    checked = subprocess.run(
        [str(ROOT / "target/release/b2-runner"), "--canonical-hashes", str(source)],
        capture_output=True, text=True)
    assert checked.returncode == 0, checked.stderr
    assert json.loads(checked.stdout) == layer_hashes(model)


def test_execution_effects_complete_frozen_refractory_graph_without_rehashing():
    model = load("minimal-v1.json")
    original = copy.deepcopy(model)
    population = model["definition"]["populations"][0]
    for node in model["definition"]["schedule"]["nodes"]:
        effects = execution_effects(model["definition"], node)
        if node["operation"] != "code_object":
            continue
        kind = population["code_objects"][node["item_index"]]["kind"]
        if kind == "threshold":
            assert "population/0/refractory/not_refractory" in effects["reads"]
            assert {"population/0/refractory/not_refractory",
                    "population/0/refractory/lastspike"} <= set(effects["writes"])
        elif kind == "state_update":
            assert "population/0/refractory/lastspike" in effects["reads"]
            assert "population/0/refractory/not_refractory" in effects["writes"]
        else:
            assert effects == node["effects"]
        assert execution_effects(model["definition"], {**node, "effects": effects}) == effects
    assert model == original
    verify_protocol(model)


def test_fusion_cannot_cross_an_implicit_lastspike_write():
    model = load("minimal-v1.json")
    threshold = next(node for node in model["definition"]["schedule"]["nodes"]
                     if node["id"] == "population/0/code/1")
    anchor = {"id": "anchor", "clock": 0, "effects": {"reads": [], "writes": []}}
    observer = {"id": "observer", "clock": 0, "effects": {
        "reads": ["population/0/refractory/lastspike"], "writes": []}}
    model["definition"]["schedule"]["nodes"] = [anchor, threshold, observer]
    assert not _schedule_can_contract(model, anchor, observer)
    assert not _schedule_can_fuse_bundles(model, [[anchor, observer], [threshold]])


def test_streamed_canonical_hashes_match_standard_encoder_at_array_boundaries():
    from io import BytesIO
    from brian2_rust.protocol import _canonical_chunks, write_canonical

    values = ["escaped\\quote\"\n雪", 0, -0.0, 5e-324, True, None, [], {}]
    rng = random.Random(20260907)
    for length in (0, 1, 65535, 65536, 65537, 131073):
        value = {"z": [values[rng.randrange(len(values))] for _ in range(length)],
                 "a": {"empty": [], "numeric": [-1e100, 1e-7, 1.25]}}
        expected = canonical_bytes(value)
        assert b"".join(_canonical_chunks(value)) == expected
        stream = BytesIO()
        write_canonical(value, stream)
        assert stream.getvalue() == expected
    for invalid in (float("nan"), float("inf"), -float("inf")):
        with pytest.raises(ValueError):
            b"".join(_canonical_chunks({"values": [invalid]}))


def test_project_snapshot_preserves_public_model_and_rejects_tampering(tmp_path):
    from brian2_rust.native import generate_source, write_instance, write_project

    model = load("minimal-v1.json")
    original = copy.deepcopy(model)
    source = generate_source(model)
    write_instance(model, tmp_path / "expected.bin")
    generated, instance, _ = write_project(model, tmp_path / "project")
    assert model == original
    assert generated.read_text() == source
    assert instance.read_bytes() == (tmp_path / "expected.bin").read_bytes()
    model["instance"]["rng_seed"] += 1
    for operation in (lambda: generate_source(model),
                      lambda: write_instance(model, tmp_path / "bad.bin"),
                      lambda: write_project(model, tmp_path / "bad-project")):
        with pytest.raises(ValueError, match="hash mismatch"):
            operation()
    assert not (tmp_path / "bad.bin").exists()
    assert not (tmp_path / "bad-project").exists()
