# Brian2 Atlas 0.2.0

Release date: 2026-11-08.

Atlas 0.2.0 packages the current simulation, native training and experimental
neuromorphic conversion implementation for installation with pip. It is alpha
research software: a stable package version does not imply complete backend or
model coverage. Unsupported constructs raise capability errors.

## Distribution and installation

- Distribution name: `brian2-atlas`; install with
  `python -m pip install brian2-atlas==0.2.0` in a new virtual environment.
- Prebuilt wheels: CPython 3.12, 3.13 and 3.14 on Linux x86_64 (glibc 2.28+),
  macOS 15+ Apple Silicon (arm64), and macOS 15+ Intel (x86_64).
  A source archive is also provided. Windows wheels are not published.
- Wheels include Brian2 Cython extensions, `b2-runner`, `b2-train`, runtime
  resources and build provenance. Basic reference CPU simulation and training
  use the bundled executables. AOT execution and source installs require
  Rust 1.98.1 and native build tools.
- Atlas supplies the `brian2` import package and must not share an environment
  with the upstream `Brian2` distribution. The frontend compatibility version
  remains separate from the Atlas distribution version.

## Scope and compatibility

Use `brian2` for modelling and `brian2_atlas` for the public Atlas API.
`brian2_rust` remains a compatibility alias. Backend/toolchain and model limits
are described in the [installation guide](INSTALLATION.md) and
[backend guide](../brian2-rust/README.md).

This release includes native training, checkpoint continuation and experimental
neuromorphic software conversion developed after `v0.1.0`. Target-specific
semantic and resource constraints still apply. Software conversion tests do not
establish physical chip execution, power or latency. GPU, MPI and vendor SDKs
require separate platform-specific setup and are outside the Linux/macOS CPU wheel
publication gate.

The `v0.1.0` source tag and bioRxiv v1 evidence archive remain frozen historical
results; these release notes do not extend those results to 0.2.0.

## Release validation

The release process first publishes `testpypi-v0.2.0` to TestPyPI, then publishes
`v0.2.0` from the merged `dev` commit. Both workflows build from the sdist, repair
and test each Linux/macOS wheel, and download the uploaded files to verify their
SHA-256 hashes. Fresh installed environments exercise CPU reference/AOT
simulation, native training and gradients, checkpoint continuation, and bounded
integer conversion/C host-kernel checks.

Actual outcomes and per-interpreter reports are retained in the
[GitHub Actions runs](https://github.com/RockLi/brian2-atlas/actions).
See [the quickstart](QUICKSTART.md) for simulation and native training examples.
