"""Canonical layer hashing and explicit probe-schema migrations for AtlasIR."""

from __future__ import annotations

import copy
import hashlib
import json

from .encoded_array import EncodedArray, IndexArray

IR_NAME = "AtlasIR"
CURRENT_SCHEMA = "atlasir-v1"
LEGACY_V1_SCHEMA = "b2ir-v1"
SUPPORTED_V1_SCHEMAS = (CURRENT_SCHEMA, LEGACY_V1_SCHEMA)
PREVIOUS_SCHEMAS = (
    "b2ir-gate0-probe-v34",
    "b2ir-gate0-probe-v35",
    "b2ir-gate0-probe-v36",
    "b2ir-gate0-probe-v37",
)
CANONICAL_ENCODING = "atlasir-canonical-json-v1"


def _protocol_header(schema):
    if schema == CURRENT_SCHEMA:
        name, encoding = "atlasir", CANONICAL_ENCODING
    elif schema == LEGACY_V1_SCHEMA or schema in PREVIOUS_SCHEMAS:
        name, encoding = "b2ir", "b2ir-canonical-json-v1"
    else:
        raise ValueError(f"unsupported AtlasIR schema {schema!r}")
    return {"name": name, "version": {"major": 1, "minor": 0},
            "canonical_encoding": encoding, "hash_algorithm": "sha256"}

_JSON_SCALARS = frozenset({str, int, float, bool, type(None)})


def _json_default(value):
    if isinstance(value, (EncodedArray, IndexArray)):
        return list(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def canonical_bytes(value) -> bytes:
    """Encode an AtlasIR value with the versioned canonical JSON profile."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=_json_default,
    ).encode("utf-8")


def _canonical_chunks(value):
    """Canonical JSON with bounded buffers for large flat instance arrays."""
    if isinstance(value, (EncodedArray, IndexArray)):
        yield from value.canonical_chunks()
    elif isinstance(value, dict):
        yield b"{"
        for index, key in enumerate(sorted(value)):
            if index:
                yield b","
            yield canonical_bytes(key)
            yield b":"
            yield from _canonical_chunks(value[key])
        yield b"}"
    elif isinstance(value, list):
        yield b"["
        for start in range(0, len(value), 65536):
            if start:
                yield b","
            chunk = value[start:start + 65536]
            if not set(map(type, chunk)) <= _JSON_SCALARS:
                for index, item in enumerate(chunk):
                    if index:
                        yield b","
                    yield from _canonical_chunks(item)
            else:
                # Retain the standard C encoder's exact escaping and numeric
                # formatting without materializing the complete instance JSON.
                yield canonical_bytes(chunk)[1:-1]
        yield b"]"
    else:
        yield canonical_bytes(value)


def write_canonical(value, stream):
    """Write canonical UTF-8 JSON to a binary stream with bounded array buffers."""
    for chunk in _canonical_chunks(value):
        stream.write(chunk)


def layer_hashes(model: dict) -> dict[str, str]:
    hashes = {}
    for layer in ("definition", "instance", "run"):
        digest = hashlib.sha256()
        for chunk in _canonical_chunks(model[layer]):
            digest.update(chunk)
        hashes[layer] = digest.hexdigest()
    return hashes


def attach_protocol(model: dict) -> dict:
    """Attach an envelope for a current or explicitly legacy v1 model."""
    if model.get("schema") not in SUPPORTED_V1_SCHEMAS:
        raise ValueError(f"cannot envelope unsupported schema {model.get('schema')!r}")
    model["protocol"] = {
        **_protocol_header(model.get("schema")),
        "layers": layer_hashes(model),
    }
    return model


def verify_protocol(model: dict) -> None:
    protocol = model.get("protocol")
    expected_header = _protocol_header(model.get("schema"))
    if (not isinstance(protocol, dict) or
            set(protocol) != {*expected_header, "layers"} or
            canonical_bytes({name: protocol.get(name) for name in expected_header})
            != canonical_bytes(expected_header)):
        raise ValueError("invalid AtlasIR protocol envelope")
    if protocol.get("layers") != layer_hashes(model):
        raise ValueError("AtlasIR canonical layer hash mismatch")


def migrate_model(model: dict) -> dict:
    """Return a verified current model or migrate a supported predecessor."""
    migrated = copy.deepcopy(model)
    schema = migrated.get("schema")
    if schema == LEGACY_V1_SCHEMA:
        verify_protocol(migrated)
        migrated["schema"] = CURRENT_SCHEMA
        return attach_protocol(migrated)
    if schema in PREVIOUS_SCHEMAS:
        if schema == PREVIOUS_SCHEMAS[0]:
            if "protocol" in migrated:
                raise ValueError("legacy AtlasIR has a protocol envelope")
        else:
            verify_protocol(migrated)
            migrated.pop("protocol")
        for population in migrated["definition"]["populations"]:
            population.setdefault("linked_variables", [])
        for function in migrated["definition"]["functions"]:
            function["abi"] = "b2ir-function-v1"
            function.setdefault("backend_implementations", {})
        migrated["schema"] = CURRENT_SCHEMA
        return attach_protocol(migrated)
    if schema != CURRENT_SCHEMA:
        raise ValueError(f"unsupported AtlasIR schema {schema!r}")
    verify_protocol(migrated)
    return migrated
