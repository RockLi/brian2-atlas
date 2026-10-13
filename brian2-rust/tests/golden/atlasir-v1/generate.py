"""Generate current AtlasIR envelope vectors from the immutable legacy corpus."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python"))
from brian2_rust.protocol import canonical_bytes, layer_hashes, migrate_model

HERE = Path(__file__).resolve().parent
model = migrate_model(json.loads((HERE.parent / "b2ir-v1/minimal-v1.json").read_text()))
for name, value in {
    "minimal-v1.json": model,
    "hashes.json": {"schema": model["schema"], "layers": layer_hashes(model),
                    "canonical_document_sha256": hashlib.sha256(canonical_bytes(model)).hexdigest()},
}.items():
    (HERE / name).write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
