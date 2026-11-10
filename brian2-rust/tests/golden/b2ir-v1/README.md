# Legacy AtlasIR vectors

These files preserve the original wire bytes and hashes for compatibility tests.
The archived `generate.py` belongs to the original exporter revision; do not run
it against the current exporter. Generate current vectors with
`../atlasir-v1/generate.py` instead.

The current corpus changes only envelope identifiers. Tests verify the legacy
corpus before migration and compare layer hashes and native result bytes across
both names and the supported probe predecessors.
