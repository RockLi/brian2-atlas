# Installing Brian2 Atlas

The pip distribution is named `brian2-atlas`. It provides `brian2` for modelling,
`brian2_atlas` for the public backend API, the compatibility import `brian2_rust`,
and the commands `b2-runner` and `b2-train`.

## Choose an isolated environment

Use Python 3.12 or newer. Atlas contains a modified Brian2 frontend, so install it
in a new environment without the upstream `Brian2` distribution. Pip treats the
two distribution names as different projects even though they write the same
`brian2/` files. Installing or uninstalling either in a shared environment can
overwrite or remove the other's files. Recreate such an environment if necessary.

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
```

## Install a published release

Install version `0.2.0` from PyPI:

```sh
python -m pip install brian2-atlas==0.2.0
```

To require a prebuilt Atlas wheel and avoid a local Atlas source build:

```sh
python -m pip install --only-binary=brian2-atlas brian2-atlas==0.2.0
```

If no matching wheel exists, the second command fails. The ordinary command can
fall back to a source archive, which requires the build tools below. A macOS
arm64 wheel cannot be used on Intel macOS or Linux; CPython minor versions also
need matching wheels because Atlas includes Cython extensions.

The `0.2.0` release provides CPython 3.12, 3.13 and 3.14 wheels for Linux x86_64
(glibc 2.28+) and macOS 15+ on Apple Silicon (arm64) and Intel (x86_64), plus an
sdist. Both macOS architectures build and run the installation gate on native
runners. Earlier macOS versions require a separately validated source build.
Windows wheels are not published.
Windows, free-threaded CPython and every Python/architecture combination are not
qualified merely by the `Python >=3.12` metadata.

## Install from source

For the `0.2.0` release:

```sh
git clone --branch v0.2.0 https://github.com/RockLi/brian2-atlas.git
cd brian2-atlas
rustup toolchain install 1.98.1 --profile minimal
python -m pip install .
```

Install rustup first if it is unavailable. Source builds need a working C/C++
compiler and linker (Xcode Command Line Tools on macOS; a development compiler
toolchain and Python development headers on Linux). Pip installs the Python
build dependencies in an isolated environment. Cargo builds both native engines
from `Cargo.lock`; the first build also needs access to the Rust dependency
registry. `uv.lock` does not constrain dependencies resolved by `pip install`.

For development features, check out an explicit development commit and use the
same install command. Record that commit with your results. To reproduce the
historical research release, use `v0.1.0`; the conversion and training additions
described in the `0.2.0` guides are not part of that older tag.

To install a wheel supplied by a maintainer, pass its actual path to pip:

```sh
python -m pip install /path/to/brian2_atlas-VERSION-PYTHON-ABI-PLATFORM.whl
```

## Backend requirements

| Operation | Additional requirements after a matching wheel install |
| --- | --- |
| Reference CPU simulation and basic native CPU training | Bundled native executables are sufficient for the quickstart |
| Specialized AOT simulation | Rust/rustup with toolchain 1.98.1 and a native linker |
| CUDA | Compatible NVIDIA GPU, driver, CUDA toolkit and matching CuPy installation; simulation uses `numeric_mode="float32"` |
| Metal | Supported Apple hardware and compiler/runtime; simulation uses `numeric_mode="float32"` |
| MPI | MPI implementation providing `mpicc` and `mpiexec`, plus requirements of the selected CPU/GPU path |
| Browser export | The separate `export_wasm_bundle` workflow and browser/toolchain requirements in the backend guide |
| Neuromorphic software targets | Target-specific SDKs and, for the custom SpiNNaker2 host kernel, a C compiler |

Installing the package does not install GPU drivers, MPI or vendor SDKs. See the
[backend guide](../brian2-rust/README.md) and
[target compatibility guide](../brian2-rust/examples/neuromorphic/COMPATIBILITY.md)
before selecting these paths.

## Optional dependencies

Published extras can be selected with `python -m pip install 'brian2-atlas[cloud]'`.
From a checkout use `python -m pip install '.[cloud]'` instead.

| Extra | Purpose |
| --- | --- |
| `cloud` | Modal SDK |
| `examples` | Plotting and example dependencies including joblib, noise, OpenCV, progressbar2 and SBI |
| `test` | pytest and test plugins; individual suites can require further dependencies |
| `docs` | Sphinx, IPython and sphinx-tabs for the inherited Sphinx documentation |

The `examples` extra can be large and may have narrower platform support than
the core package. It is not required for the quickstart.

## Verify and configure

```sh
python -c 'from importlib.metadata import version; import brian2, brian2_atlas; print("Atlas", version("brian2-atlas")); print("Brian compatibility", brian2.__version__)'
b2-runner --help
b2-train --help
```

Run the [quickstart](QUICKSTART.md) to check actual simulation and training.
The native commands' help output is diagnostic; it is not a simulation test.

Runtime output normally goes to temporary directories. Set
`BRIAN2_ATLAS_OUTPUT_DIR` for a persistent output parent and `XDG_CACHE_HOME` for
runtime caches. Source builds can use `CARGO_TARGET_DIR`, `CARGO_HOME`,
`PIP_CACHE_DIR` and `TMPDIR` to relocate build storage. `B2_RUNNER` and
`B2_TRAIN_RUNNER` explicitly override the bundled binaries; unset these variables
when checking a newly installed wheel.

If pip reports no matching distribution, check publication status, Python
version, index configuration and wheel platform. If it starts compiling, no
compatible wheel was selected. If imports resolve to a checkout or another
environment, run outside the checkout, unset `PYTHONPATH`, and use the intended
environment's `python -m pip`.
