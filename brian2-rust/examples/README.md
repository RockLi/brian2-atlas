# Atlas source examples

Run from the repository root with Python dependencies installed. Examples that address the source runner directly need its conventional Cargo output:

```sh
python -m pip install -e .
cargo +1.98.1 build --release --locked --manifest-path brian2-rust/Cargo.toml --bin b2-runner --bin b2-train
export PYTHONPATH="$PWD/brian2-rust/python:$PWD"
python brian2-rust/examples/minimal.py
```

The smallest example runs a single LIF neuron and prints its state and spikes. `lif.py`, `population.py`, `refractory.py`, `synapses.py`, `two_populations.py` and `cuba.py` cover simulation and backend comparisons. Use `--help` where the script provides a CLI; some small examples run directly.

`brunel_device.py` builds the network, while `brunel_sweep.py` demonstrates reuse of one compiled AOT artifact across instance parameters. The sweep is a reusable user example and a regression dependency; historical paper sweeps remain in the preprint repository.

Native training and fresh-process resume:

```sh
python brian2-rust/examples/native_supervised_training.py --steps 2 --checkpoint training.json
python brian2-rust/examples/native_supervised_training.py --steps 1 --checkpoint training.json --resume
python brian2-rust/examples/native_graph_training.py --steps 2 --checkpoint graph.json
```

The graph example includes convolution, recurrence and readout parameters. Its `--equations`, `--backend` and `--mpi-ranks` options require the corresponding backend. The short runs above exercise execution and checkpoint continuity; they are not an accuracy benchmark.

`mpi_training.py` demonstrates two-rank STDP with delayed events and a boundary checkpoint. Launch the Python driver once; it starts MPI workers itself:

```sh
python brian2-rust/examples/mpi_training.py --directory output/mpi-first --checkpoint mpi.json
python brian2-rust/examples/mpi_training.py --directory output/mpi-resume --checkpoint mpi.json --resume
```

This requires a working local MPI toolchain/runtime. `mpi_cuba.py`, `mpi_procedural.py` and `mpi_heterogeneous.py` provide further distributed examples. FlyWire import/device examples require their explicitly supplied circuit data; the repository's bounded FlyWire WASM/native regression uses its retained circuit subset.

Run the bounded source example gate with `python dev/atlas/run_backend.py --suite examples`. Install joblib, progressbar2 and OpenCV to execute their cases. SBI is optional and its case skips if absent. Browser/FlyWire WASM tests additionally need the built WASM package and Node; they are validated separately. No result here claims full-scale paper reproduction.
