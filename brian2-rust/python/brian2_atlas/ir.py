"""AtlasIR v1 serialization, integrity checks and predecessor migration.

The stable v1 wire identifiers retain their b2ir spellings. Renaming the
representation does not alter stored models or canonical layer identities.
"""
from brian2_rust.protocol import (
    IR_NAME, CURRENT_SCHEMA, CANONICAL_ENCODING, PREVIOUS_SCHEMAS,
    attach_protocol, canonical_bytes, layer_hashes, migrate_model,
    verify_protocol, write_canonical,
)

__all__ = [
    "IR_NAME", "CURRENT_SCHEMA", "CANONICAL_ENCODING", "PREVIOUS_SCHEMAS",
    "attach_protocol", "canonical_bytes", "layer_hashes", "migrate_model",
    "verify_protocol", "write_canonical",
]
