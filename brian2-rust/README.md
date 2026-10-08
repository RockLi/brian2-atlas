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

## GPU and browser backends

The CUDA and Apple Metal simulation runtimes and browser WASM/WebGPU code are now included. Build browser assets with `python brian2-rust/tools/build_wasm.py`; the Rust WASM target and wasm-bindgen 0.2.100 are required. Serve the resulting `brian2-rust/output/wasm` directory over localhost to use the browser examples.

Actual migration checks covered an NVIDIA L4, Apple Metal, and Chrome with a non-fallback Apple WebGPU adapter. The tested scope and source hashes are in `migration/p3-gpu-wasm-port.json`. GPU regression execution is opt-in and requires the matching device/toolchain; a skipped hardware case does not count as passed. Native training, distributed runtime integration and packaged installation follow in separate stages.
