# Atlas native backend

The `brian2_atlas` module provides native CPU/AOT simulation, GPU/browser backends,
distributed MPI simulation and native training. Both native executables and their
runtime resources are included in Atlas platform wheels. This is a development
version; see the [root installation guide](../README.md) and the stage manifests
for the verified scope.

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
Source, wheel and sdist installation gates are recorded in the packaging manifest.

## GPU and browser backends

The CUDA and Apple Metal simulation runtimes and browser WASM/WebGPU code are now included. Build browser assets with `python brian2-rust/tools/build_wasm.py`; the Rust WASM target and wasm-bindgen 0.2.100 are required. Serve the resulting `brian2-rust/output/wasm` directory over localhost to use the browser examples.

Actual migration checks covered an NVIDIA L4, Apple Metal, and Chrome with a non-fallback Apple WebGPU adapter. The tested scope and source hashes are in `migration/p3-gpu-wasm-port.json`. GPU regression execution is opt-in and requires the matching device/toolchain; a skipped hardware case does not count as passed. Training and MPI are included; installation builds both native executables.

## MPI simulation

Distributed planning, generated MPI projects, partitioning, collective transport and MPI regression inputs are included. Install an MPI implementation providing `mpicc` and `mpiexec` alongside the pinned Rust compiler. Tests using actual MPI require `B2_TEST_MPI=1`. The migration validated same-host 1/2/4-rank execution, exact reference results, continuation and coordinated failure exits; see `migration/p4-mpi-port.json`. Cross-host execution has not been qualified by this migration.

## Native training

`NativeLIFTrainer`, Brian network conversion, dynamic clocks, state/checkpoint continuation and native CPU/Metal/CUDA/MPI training are included. Build the training executable with `cargo +1.98.1 build --release --locked --bin b2-train --manifest-path brian2-rust/Cargo.toml`. Source-mode tests can select it with `B2_TRAIN_RUNNER`; installed users obtain the bundled executable automatically.

The [initial training qualification](../migration/p5-native-training-port.json) records CPU/Metal numerical and checkpoint behavior, callback-training increments and the initial CUDA/MPI scope. The [CUDA follow-up](../migration/cuda-cutoff-followup.json) subsequently passed all 708 previously skipped cases from 15 modules plus one C ABI regression on a Modal NVIDIA L4: 709 passed, zero failed or skipped. It includes two same-host MPI ranks sharing one GPU; cross-host and multi-GPU training remain outside this qualification. The [public API qualification](../migration/atlas-public-api.json) covers Atlas aliases and installed-package CPU/AOT/Metal flows.
