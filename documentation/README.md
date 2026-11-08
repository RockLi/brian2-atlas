# Brian2 Atlas documentation

Start with [installation](INSTALLATION.md), then run the
[simulation and training quickstart](QUICKSTART.md). Maintainers can follow the
[PyPI release procedure](RELEASING.md) to build and validate distribution files.
The first Linux release rehearsal uses `0.2.0.dev1`; see the
[TestPyPI setup and verification guide](TESTPYPI.md).

These documents describe this checkout. The frozen `v0.1.0` source release and
the current development branch have different capabilities; consult the same
tag or commit as the code you use. A source tag does not imply PyPI availability.

| Topic | Guide |
| --- | --- |
| CPU, CUDA, Metal, MPI and browser execution | [Backend guide](../brian2-rust/README.md) |
| Native training and supported lowering | [Training guide](../brian2-rust/TRAINING.md) |
| Runnable simulations, training and checkpoints | [Source examples](../brian2-rust/examples/README.md) |
| Model representation and compatibility | [AtlasIR](../brian2-rust/AtlasIR.md) |
| Architecture and capability boundaries | [Architecture](../brian2-rust/architecture/README.md) |
| Experimental neuromorphic conversion | [Target compatibility](../brian2-rust/examples/neuromorphic/COMPATIBILITY.md) |
| Licensing and attribution | [License scope](../LICENSE_SCOPE.md) |
| Historical installation and backend validation | [Migration records](../migration/README.md) |

For an issue report, include the Atlas distribution version, Python version,
OS/architecture, chosen engine, a minimal model and the complete error.
Retrieve the Atlas version with
`python -c 'from importlib.metadata import version; print(version("brian2-atlas"))'`.
`brian2.__version__` reports the frontend compatibility version instead.
