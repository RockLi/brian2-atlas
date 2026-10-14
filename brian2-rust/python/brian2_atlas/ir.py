"""AtlasIR v1 serialization, integrity checks and predecessor migration.

New models use atlasir-v1. Legacy envelopes are verified before explicit
migration; canonical layer identities and simulation semantics are preserved.
"""
from brian2_rust.protocol import (
    IR_NAME, CURRENT_SCHEMA, CANONICAL_ENCODING, PREVIOUS_SCHEMAS, LEGACY_V1_SCHEMA,
    attach_protocol, canonical_bytes, layer_hashes, migrate_model,
    verify_protocol, write_canonical,
)

__all__ = [
    "IR_NAME", "CURRENT_SCHEMA", "CANONICAL_ENCODING", "PREVIOUS_SCHEMAS", "LEGACY_V1_SCHEMA",
    "attach_protocol", "canonical_bytes", "layer_hashes", "migrate_model",
    "verify_protocol", "write_canonical",
]
