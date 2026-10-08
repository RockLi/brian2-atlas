# Atlas native backend

This directory retains the `brian2_rust` Python module and the locked Rust
reference engine. The CPU integration includes B2IR validation, the Brian Device,
model-specialized AOT compilation, monitors, continuation and topology readers.
Shared plan helpers are included where CPU imports require them. GPU/WASM, MPI,
training, user-facing packaging and their complete runtime assets are separate
migration stages; this development commit is not a formal release.

From the repository root, with Rust/rustup and a C/C++ build toolchain installed:

```sh
python -m pip install -e '.[test]' scipy pyarrow
python dev/atlas/run_backend.py --suite core
python dev/atlas/run_backend.py --suite cpu --timeout 1200
```

The command selects Rust 1.98.1, builds the locked native runner and tests this
checkout. `--runner` accepts an explicitly selected prebuilt runner. `--output`
controls test artifacts; `CARGO_TARGET_DIR`, `CARGO_HOME`, `TMPDIR` and the usual
Python cache variables can place build and temporary data on external storage.
For the 20,000-term compiler stress case on a loaded host, set
`B2_TEST_RUSTC_TIMEOUT=900`; the model size and numerical checks are unchanged.

Source hashes, adaptations, named failure resolutions and actual validation
scope are recorded in [the CPU port manifest](../migration/p2-cpu-port.json).
Installed wheel/sdist qualification is still a later migration stage.
