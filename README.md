# Brian2 Atlas

Brian2 Atlas combines the Brian2 modelling interface with native simulation,
GPU/browser backends, distributed MPI execution and native training. This
repository contains the product implementation and its regression tests.
[The preprint repository](https://github.com/RockLi/brian2-atlas-preprint) contains
the manuscript, historical experiments and reproduction materials.

Atlas retains the upstream Brian2 Git history, author attribution and
[CeCILL 2.1 licence](LICENSE). The migration baseline is Brian2 commit
`27b5431168cf9959f0c27894cd4d92cdc6ad5c31`.

## Install from source

This is development version **0.1.0.dev0**; a public package release has not been
published. The distribution name is `brian2-atlas`; existing Python imports stay
`brian2` and `brian2_rust`. The Brian compatibility version is
`2.10.1.post241`, independent of the Atlas distribution version.

Use Python 3.12 or newer, Rust/rustup and a working C/C++ compiler:

```sh
rustup toolchain install 1.98.1 --profile minimal
python -m venv .venv
. .venv/bin/activate
python -m pip install .
```

A source install builds both `b2-runner` and `b2-train` with the locked Rust
sources. Platform wheels contain these executables and all runtime source
resources; installing a matching wheel does not rebuild them. Specialized AOT
simulation still needs the pinned Rust compiler. CUDA needs the NVIDIA toolkit
and CuPy; Metal needs the corresponding Apple compiler/runtime; MPI needs
`mpicc` and `mpiexec`.

The optional `brian2-rust/tools/install_user.sh` uses `uv` to create a coherent
user installation and expose `brian2-atlas-python`, `brian2-atlas`, `b2-runner`
and `b2-train`. Set `BRIAN2_ATLAS_HOME`, `BRIAN2_ATLAS_BIN` and
`BRIAN2_ATLAS_PYTHON` to choose its locations and interpreter. It records source
and binary provenance in `install.json` and refuses to overwrite unrelated
commands. It installs a snapshot; rerun after source changes.

Builds and caches can live on external storage using `CARGO_TARGET_DIR`,
`CARGO_HOME`, `UV_CACHE_DIR`, `PIP_CACHE_DIR`, `TMPDIR` and `XDG_CACHE_HOME`.
Installed runtime output goes to temporary directories; set
`BRIAN2_ATLAS_OUTPUT_DIR` to choose a persistent output parent. An explicit
`B2_RUNNER` or `B2_TRAIN_RUNNER` overrides the bundled executable.

## First simulation

```python
import brian2 as b
import brian2_rust

b.set_device("rust_standalone", engine="reference")
group = b.NeuronGroup(
    1, "dv/dt = (1.5-v)/(10*ms) : 1",
    threshold="v>1", reset="v=0", method="euler", dt=0.1*b.ms,
)
spikes = b.SpikeMonitor(group)
b.Network(group, spikes).run(100*b.ms)
print(spikes.num_spikes)  # 9
```

Use `engine="aot"` for model-specialized CPU execution. Backend capabilities
are checked explicitly; unsupported model constructs raise a capability error.
See [the backend guide](brian2-rust/README.md) for GPU, browser, MPI and training.

## Validation and support scope

The migration has run on macOS arm64/Python 3.14 with actual CPU/AOT, Apple
Metal and same-host MPI, and on a Modal NVIDIA L4 with CUDA simulation/training.
Chrome WASM Workers and a non-fallback Apple WebGPU adapter were exercised.
Source-directory, wheel and isolated-sdist installs passed representative
simulation, training and fresh-process checkpoint checks on macOS arm64.

These results do not qualify every model, device, CUDA test variant, Windows,
cross-host MPI or multiple GPUs. The hosted CI configuration covers Linux and
macOS distribution builds; its results must be checked separately. Exact source
identities, numerical gates, skipped cases and known validation limits are in
[the migration records](migration/README.md).

```sh
python -m pip install -e '.[test]' scipy pyarrow
python dev/atlas/run_frontend.py
python dev/atlas/run_backend.py --suite cpu --timeout 1200
```

Training source tests need a built `b2-train`; hardware suites are opt-in.
`dev/atlas/check_installed.py --report /path/to/report.json` verifies an installed
package, and `--metal` adds actual Metal training. Run it with an installed
interpreter and without `PYTHONPATH`, `B2_RUNNER` or `B2_TRAIN_RUNNER` overrides.

## Upstream and research provenance

The `dev` branch is the Atlas development branch. `upstream` points to
[brian-team/brian2](https://github.com/brian-team/brian2); product ports retain
source hashes and adaptation notes. Historical paper results remain tied to the
source versions that produced them. New paper reproduction will pin an Atlas
commit in the preprint repository. Public package publication, a formal release
and paper submission are separate steps.
