# Building and publishing Brian2 Atlas

Atlas is packaged for pip as `brian2-atlas`. Publish a source distribution
(`sdist`) and qualified platform wheels to PyPI so users can install it with
`python -m pip install brian2-atlas`. The wheel includes the Brian2 Cython
extensions, both native Rust executables, build provenance and runtime sources.
It is not a universal `py3-none-any` package.

## Release scope and publication procedure

The checkout has PEP 517 metadata, a locked native build, an installed-package
gate and Linux/macOS distribution CI. CI checks the package description with
`twine check --strict`. The separate `release-testpypi.yml` workflow builds
CPython 3.12–3.14 wheels from the sdist for Linux x86_64 (manylinux 2.28) and
macOS 15+ (arm64 and x86_64), and tests each repaired wheel on a native runner. Versioned `testpypi-v*` tags upload to
TestPyPI and trigger a fresh download/install gate. Branch pushes and manual
runs only build. The separate `release-pypi.yml` workflow builds on relevant PRs
and manual runs, and publishes only matching stable `v*` tags already merged
into `dev`. See [TestPyPI setup](TESTPYPI.md) and [production setup](PYPI.md).

Before each public upload:

1. Choose the exact release commit and version. `v0.1.0` is a frozen source
   release. The first packaging rehearsal used `0.2.0.dev1`; the current release
   is `0.2.0`. For subsequent candidates,
   choose an unused version, update `pyproject.toml`,
   refresh `uv.lock` with `uv lock`, and update release notes/citation metadata
   where applicable. Never present current development as the frozen tag.
2. Verify control of the `brian2-atlas` project name on PyPI and TestPyPI. Absence
   from search results does not guarantee that a name can be registered.
3. Choose and test the supported Python/OS/architecture matrix. The TestPyPI matrix
   covers Linux x86_64 and macOS arm64/x86_64 / CPython 3.12–3.14, publishing
   nine wheels and the sdist.
4. Require all platform build/install jobs to pass for the exact candidate.
   Ordinary `linux_x86_64` wheels from a host build are not PyPI release artifacts.
5. Configure project ownership and upload credentials or Trusted Publishing,
   rehearse on TestPyPI, then publish the exact validated artifacts.

The bundled frontend shares paths with upstream `Brian2`. Preserve the isolated
environment guidance in user documentation. Keep the combined license metadata
and notices described in [LICENSE_SCOPE.md](../LICENSE_SCOPE.md).

## Build a source archive and wheel

Use a clean checkout of the intended release commit. The following commands are
for a POSIX shell with Python 3.12+, C/C++ compilers and rustup installed. Choose
a fresh output directory for each candidate; do not mix old and new artifacts.

```sh
python3 -m venv build/release-tools
. build/release-tools/bin/activate
python -m pip install --upgrade pip build twine
rustup toolchain install 1.98.1 --profile minimal
export ATLAS_ROOT="$PWD"
export ATLAS_RELEASE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/atlas-release.XXXXXX")"
export CARGO_TARGET_DIR="$ATLAS_ROOT/build/release-cargo"
python -m build --sdist --outdir "$ATLAS_RELEASE_DIR/sdist"
python -m pip wheel --no-deps --wheel-dir "$ATLAS_RELEASE_DIR/wheels" \
  "$ATLAS_RELEASE_DIR"/sdist/*.tar.gz
python -m twine check --strict "$ATLAS_RELEASE_DIR"/sdist/*.tar.gz \
  "$ATLAS_RELEASE_DIR"/wheels/*.whl
```

Building the wheel from the archive verifies that the sdist contains the build
inputs. Isolated builds install their own Python build dependencies. Inspect
both archives before uploading: expected contents include both import packages,
the Brian2 frontend, the native sources in the sdist, `_bin/b2-runner`,
`_bin/b2-train`, `_bin/build.json` in the wheel, runtime kernels/templates and
license notices. Exclude local experiment outputs, credentials and caches.

`README.pypi.md` supplies the PyPI description. It uses absolute links and is
included in the sdist with the user and maintainer guides. Update its version
scope and documentation links for the selected release; links to `dev` describe
development, not an immutable release. `twine check` validates metadata and
description rendering, not native compatibility or model correctness.

## Platform wheel qualification

Use [cibuildwheel](https://cibuildwheel.pypa.io/en/stable/) to build the selected
CPython versions in suitable environments. Start with Linux x86_64 and macOS
arm64 if these are the release targets; add other architectures only with tests.

- **Linux:** build inside the chosen manylinux image with Rust 1.98.1 installed
  inside that container. Run auditwheel inspection/repair and test the resulting
  manylinux wheel. Do not rename a host `linux_*` wheel to claim manylinux
  compatibility. Check the bundled Rust executables' dynamic dependencies and
  glibc requirements as well as the Python extensions.
- **macOS:** build natively for each selected architecture. Align the Rust and
  C/C++ deployment targets with the wheel's advertised minimum macOS version;
  test on that baseline. Inspect dependencies with delocate/`otool -L`. A build
  on a recent macOS host alone does not qualify an older deployment target.
- **Windows and free-threaded CPython:** remain outside the qualified release
  matrix until native builds, runtime paths and installed-flow checks pass.

Auditwheel/delocate can change packaged binary bytes. Atlas verifies its bundled
executables against `_bin/build.json`; if a repair changes either executable,
regenerate the recorded binary SHA-256 values in the repaired wheel and update
wheel `RECORD` using wheel-aware tooling. Re-run the installed gate on the final
wheel. Do not disable the integrity check to make repaired artifacts pass.

The release matrix uses a digest-pinned manylinux 2.28 image, native macOS 15 runners and
`dev/atlas/repair_wheel.py` to repair each wheel, refresh executable hashes when
needed, and regenerate `RECORD` before installed tests. macOS repair also checks
both bundled executables against the wheel architecture. Linux arm64 and
musllinux are not part of this release matrix.

## Validate an installed candidate

Continue in the same shell, with `ATLAS_ROOT` and `ATLAS_RELEASE_DIR` set. These
commands expect exactly one wheel matching this test machine. For a full release
matrix, run the same gate separately against each matching final wheel.

```sh
export ATLAS_VERSION="$(python -c 'import tomllib; print(tomllib.load(open("pyproject.toml", "rb"))["project"]["version"])')"
python -m venv "$ATLAS_RELEASE_DIR/installed"
"$ATLAS_RELEASE_DIR/installed/bin/python" -m pip install \
  "$ATLAS_RELEASE_DIR"/wheels/*.whl
export TMPDIR="$ATLAS_RELEASE_DIR/runtime"
export XDG_CACHE_HOME="$ATLAS_RELEASE_DIR/cache"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
unset PYTHONPATH B2_RUNNER B2_TRAIN_RUNNER RUSTUP_TOOLCHAIN BRIAN2_ATLAS_OUTPUT_DIR
cd "$ATLAS_RELEASE_DIR"
"$ATLAS_RELEASE_DIR/installed/bin/python" "$ATLAS_ROOT/dev/atlas/check_installed.py" \
  --expected-version "$ATLAS_VERSION" --report "$ATLAS_RELEASE_DIR/installed.json"
"$ATLAS_RELEASE_DIR/installed/bin/python" -m pip check
```

This gate checks import isolation, executable hashes, command entry points,
packaged resources, reference/AOT simulation against NumPy, CPU training,
fresh-process checkpoint continuation, and current development's integer
conversion/C kernel path. Rust and a C compiler are needed for this gate even
though the basic CPU quickstart runs from a wheel without them. Use the gate
from the matching source commit; frozen tags do not contain all development APIs.
`--metal` adds an actual Metal training comparison on a suitable machine.

Also run both scripts in [QUICKSTART.md](QUICKSTART.md) from outside the checkout.
Retain CI results for the frontend and CPU suites at the release commit. Record
the environment, commit, versions, final filenames and SHA-256 checksums with
the validation reports. Hardware-specific claims need their corresponding
hardware tests; package installation alone does not qualify them.

## TestPyPI rehearsal and production upload

The commands below perform external uploads. Run them only for the chosen
release, after validation and account configuration. PyPI and TestPyPI have
separate accounts, project ownership and credentials. Do not store tokens in the
repository or paste them into commands recorded in shell history.

For a manual rehearsal, use the release-tools interpreter and upload only the
validated sdist and final portable wheels:

```sh
python -m twine upload --repository testpypi \
  "$ATLAS_RELEASE_DIR"/sdist/*.tar.gz "$ATLAS_RELEASE_DIR"/wheels/*.whl
```

To check the uploaded candidate without resolving dependencies from TestPyPI,
download that exact project/version from TestPyPI, then install the local wheel
with the normal PyPI index for dependencies. Run on a machine with a matching
uploaded wheel:

```sh
python -m pip download --index-url https://test.pypi.org/simple/ \
  --no-deps --only-binary=:all: --dest "$ATLAS_RELEASE_DIR/from-testpypi" \
  "brian2-atlas==$ATLAS_VERSION"
python -m venv "$ATLAS_RELEASE_DIR/test-installed"
"$ATLAS_RELEASE_DIR/test-installed/bin/python" -m pip install \
  --index-url https://pypi.org/simple/ "$ATLAS_RELEASE_DIR"/from-testpypi/*.whl
```

Compare the downloaded wheel's checksum with the candidate and repeat the
installed gate using `test-installed/bin/python`. When the release is ready,
upload the same artifacts to production:

```sh
python -m twine upload --repository pypi \
  "$ATLAS_RELEASE_DIR"/sdist/*.tar.gz "$ATLAS_RELEASE_DIR"/wheels/*.whl
```

Download and test the production wheel in another clean environment. Publish
release notes with the version, exact source commit, wheel coverage, backend
scope and known limitations. PyPI does not allow replacement of an uploaded
filename: fix a bad release with a new version, and yank the faulty release if
appropriate instead of rebuilding under the same version.

For CI publishing, PyPI recommends
[Trusted Publishing](https://docs.pypi.org/trusted-publishers/using-a-publisher/).
Register the repository owner, repository, exact workflow filename and GitHub
environment on each index. Give only the publish job `id-token: write`, protect
the release environment, and have it consume already-tested artifacts.
Production uses `release-pypi.yml` with environment `pypi`; TestPyPI uses
`release-testpypi.yml` with environment `testpypi`. Configure the two publishers
separately as described in [PYPI.md](PYPI.md) and [TESTPYPI.md](TESTPYPI.md).

## Packaging references

- [PyPA project metadata and build backend requirements](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/)
- [PyPA packaging flow](https://packaging.python.org/en/latest/flow/)
- [cibuildwheel build and repair options](https://cibuildwheel.pypa.io/en/stable/options/)
- [PyPI publishing guide](https://docs.pypi.org/publishing/publishing-to-pypi/)
