# Brian2 Atlas

Brian2 Atlas provides native simulation and training backends for the Brian2
modelling interface. Define spiking neural networks in Python, then run supported
models with the bundled native CPU engine. Additional backends support
model-specialized CPU execution, CUDA, Apple Metal, MPI and browser export.

This is alpha research software. Backend support depends on the model and the
installed toolchains; unsupported constructs raise capability errors.

## Installation

For the `0.2.0.dev1` TestPyPI rehearsal, follow the
[TestPyPI download and installation steps](https://github.com/RockLi/brian2-atlas/blob/testpypi-v0.2.0.dev1/documentation/TESTPYPI.md).

Requires Python 3.12 or newer. Use a new virtual environment: this distribution
includes `brian2` and must not be installed alongside the separate upstream
`Brian2` distribution, which owns the same import paths.

Once a release is published on PyPI:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install brian2-atlas
```

A matching platform wheel includes `b2-runner`, `b2-train` and the Python
extensions. A source installation requires Rust/Cargo 1.98.1 and C/C++ compilers.
Specialized AOT execution needs Rust even after installing a wheel. CUDA, Metal
and MPI require additional system software. Windows is not currently qualified.
See the [installation guide](https://github.com/RockLi/brian2-atlas/blob/testpypi-v0.2.0.dev1/documentation/INSTALLATION.md)
for source installation, platform coverage and optional dependencies.

## First simulation

```python
import brian2 as b
import brian2_atlas

b.set_device("atlas", engine="reference")
neurons = b.NeuronGroup(
    1, "dv/dt = (1.5-v)/(10*ms) : 1",
    threshold="v>1", reset="v=0", method="euler", dt=0.1*b.ms,
)
spikes = b.SpikeMonitor(neurons)
b.Network(neurons, spikes).run(100*b.ms)
print(spikes.num_spikes)  # 9
```

Use `brian2` for modelling and `brian2_atlas` for Atlas APIs. The older
`brian2_rust` import remains a compatibility alias. The Atlas distribution
version and Brian2 compatibility version are separate values.

## Documentation

- [Quickstart: simulation and native training](https://github.com/RockLi/brian2-atlas/blob/testpypi-v0.2.0.dev1/documentation/QUICKSTART.md)
- [Backend guide](https://github.com/RockLi/brian2-atlas/blob/testpypi-v0.2.0.dev1/brian2-rust/README.md)
- [Training guide](https://github.com/RockLi/brian2-atlas/blob/testpypi-v0.2.0.dev1/brian2-rust/TRAINING.md)
- [AtlasIR specification](https://github.com/RockLi/brian2-atlas/blob/testpypi-v0.2.0.dev1/brian2-rust/AtlasIR.md)
- [Issues](https://github.com/RockLi/brian2-atlas/issues)

These links identify the `0.2.0.dev1` TestPyPI candidate. Development includes
experimental neuromorphic software conversion; see the repository's versioned
guides for the admitted models and evidence. Software conversion results do not
establish physical chip execution, power or latency.

## License and citation

The combined distribution uses **CeCILL-2.1 AND Apache-2.0**. Upstream Brian2 and
derived code retain their original terms; original Atlas engine work uses
Apache-2.0. See the
[license scope](https://github.com/RockLi/brian2-atlas/blob/testpypi-v0.2.0.dev1/LICENSE_SCOPE.md).
Use [CITATION.cff](https://github.com/RockLi/brian2-atlas/blob/testpypi-v0.2.0.dev1/CITATION.cff)
for Atlas and retain the upstream Brian2 citation when using its frontend.
The [preprint repository](https://github.com/RockLi/brian2-atlas-preprint)
contains the manuscript and historical experiment records.
