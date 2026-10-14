# Native Atlas training

Use `brian2_atlas` as the public backend namespace. Native CPU, Metal and CUDA
training and same-host MPI are available through `NativeLIFTrainer`; the Brian
frontend can lower supported networks through `lower_brian_dynamic_training`.
The native engine performs forward execution, surrogate-gradient BPTT and
optimizer updates. It does not require Brian2CUDA or Brian2GeNN.

## Xylo integer training

`NativeXyloTrainer` runs a dedicated CPU plan with Xylo Audio 2 integer forward
dynamics, int8 quantization-aware weights, surrogate BPTT and native Adam/SGD.
It exports inference through AtlasIR to XyloSim without dynamics adaptation.
The v1 plan is feedforward; `xylo_recurrent_training_plan` adds a v2 recurrent
bank, an absolute readout window and explicit evaluation-state continuation.
Training remains fresh full-sequence BPTT; chunks are inference-only and the
total sequence is limited to 128 ticks. This is separate from `NativeLIFTrainer`'s
general plans and does not inherit GPU/MPI or arbitrary Brian equation support.
See the [recurrent task, state API and memory controls](examples/neuromorphic/native_xylo_recurrent/README.md).
See the [contract, API and held-out evidence](examples/neuromorphic/native_xylo_integer/README.md).

The recurrent integer model is not portable through the existing exact
SpiNNaker2 LIF bridge. A [stock current-LIF substitution](examples/neuromorphic/native_xylo_recurrent/SPINNAKER2.md)
retains all frozen weights but reaches 50% on the task where native/XyloSim
reach 100%; spike and state traces differ under both tested clock conventions.
It remains a diagnostic. A separate [custom SpiNNaker2 integer host kernel](examples/neuromorphic/native_xylo_recurrent/SPINNAKER2_INTEGER.md)
now preserves the recurrent model and all weights, reaching 100% with exact
spike/current/membrane agreement. Use `export_native_xylo(..., target="spinnaker2")`
for this explicit profile. It is a C software extension, not stock LIF execution
or deployed SpiNNaker2 firmware.

## Start from a runnable example

From an installed source checkout:

```sh
python brian2-rust/examples/native_supervised_training.py --steps 2 --checkpoint training.json
python brian2-rust/examples/native_supervised_training.py --steps 1 --checkpoint training.json --resume
python brian2-rust/examples/native_graph_training.py --equations --backend cpu --steps 2 --checkpoint graph.json
```

Use `--backend metal` on a supported Apple GPU, or `--backend cuda` with a CUDA
GPU and toolkit. MPI additionally needs a working `mpicc` and `mpiexec`; the
Python driver launches its workers. See the [example guide](examples/README.md).
Installed platform wheels resolve their bundled `b2-train` automatically. For
source-mode checks, build both native binaries and select them explicitly:

```sh
cargo +1.98.1 build --release --locked --manifest-path brian2-rust/Cargo.toml --bin b2-runner --bin b2-train
export B2_TRAIN_RUNNER="$PWD/brian2-rust/target/release/b2-train"
export B2_RUNNER="$PWD/brian2-rust/target/release/b2-runner"
export PYTHONPATH="$PWD/brian2-rust/python:$PWD"
python -m pytest -q brian2-rust/tests/test_native_training.py
```

## Supported lowering and state behavior

The training contract covers supported Euler, stochastic Heun and Milstein
state updates; multiple state/parameter banks; time and refractory guards;
ordered pre/post synaptic pathways; delays, summed variables and dynamic
synaptic state; complete or truncated BPTT; and explicit surrogate/detach
policies. This increment extends addressed random draws inside supported scalar
and vector callbacks, scheduled updates and whole-neuron phases, including
selected Poisson draws and captured array effects.

Lowering validates the particular network and callback operations. General
Python execution, arbitrary external side effects and all Brian semantics are
not implied by these examples. Unsupported lowering must be handled explicitly.

`step` commits optimizer and physical state. `evaluate` and `gradients` return
results without committing that state, including before the first training
step. Checkpoints retain continuation identity, clock and addressed random
stream state. Selected Poisson candidate domains are validated, and an invalid
candidate must fail without partially committing the training state.

## Verification

Hardware tests are opt-in: `B2_TEST_GPU=1` enables Metal,
`B2_TEST_CUDA_TRAIN=1` enables CUDA training, and `B2_TEST_MPI=1` enables MPI.
Skipped hardware cases are not passing executions. The increment's reproducible
CPU, Metal, MPI and single-L4 CUDA gates and exact selected test identities live
in [preprint](https://github.com/RockLi/brian2-atlas-preprint/tree/dev/validation/training-port-f73).
GPU gates are bounded representative checks across test functions and rank
counts; they are not the full parameter Cartesian product. Same-host MPI ranks
may share one GPU. Cross-host mixed-device and multiple-physical-GPU training
are outside this increment's acceptance scope.

Historical development documentation and receipts are preserved in the preprint
repository. They are separate from results measured on this Atlas source tree.
The `v0.1.0` tag and paper release remain unchanged.

## Export a trained LIF network

The [native MNIST conversion example](examples/neuromorphic/native_training/README.md)
connects `NativeLIFTrainer` to SpiNNaker2 software inference. Both dense layers
are trained with the native CPU forward/backward/optimizer, then frozen with
`freeze_native_lif` and exported through AtlasIR with `export_native_lif`.
Quantization is explicit and is measured separately from conversion.

This first bridge accepts dense v1 plans, subtract reset, beta=0.5, binary
inputs, power-of-two target scales and at most 16 inference ticks. It exports
weights for fresh zero-state inference; it does not export the optimizer or
resume a live training state on the target. Other native training plans and
on-chip training remain outside this path.


## Integer count inputs and fixed class-code readout

The recurrent integer plan accepts optional `input_max` (1–15; omitted means
binary input) and `class_codes` (2–32 distinct, nonzero binary rows with equal
Hamming weight). The codebook width must equal the physical output population,
which remains limited to eight neurons. Cross entropy uses decoded class logits;
its gradients propagate through the fixed codebook into all three neural weight
banks. Count-input gradients use the actual integer input amplitude.

Frozen artifacts and the exact Xylo/custom SpiNNaker2 bridges preserve both
fields. The host decoder sums integer output counts before logit scaling; it
adds no trained weights or physical neurons. Continuation state retains physical
output counts, so chunk boundaries do not change decoded logits. Existing plans
without these fields retain binary inputs and direct output-count readout.

The [block-sequential MNIST example](examples/neuromorphic/sequential_mnist/README.md)
exercises ten classes using native training and both exact software targets.
It explicitly separates real-data classification accuracy from conversion parity.
The [third training round](examples/neuromorphic/sequential_mnist/ROUND3.md)
reaches 97.47% on all 10,000 official test images, with exact agreement across
141,480,000 neural values per target. The selected 16→126→5 model has
10,584 trainable weights and 36 ticks. Recurrent v2 admits 128 hidden cells;
the example's grouped recurrence preserves fanout ≤63 by projecting inactive
weights and optimizer moments to zero after each native update. V1 retains 64.
The 98% target remains unmet. Earlier test scores were known; selection within this round used
validation only, followed by one final full-test audit.

The [second training round](examples/neuromorphic/sequential_mnist/REFINEMENT.md)
reaches 97.28% on all 10,000 official test images, with zero differences across
73,440,000 spike/current/membrane values per target. The same 16→63→5 model uses
deskewed rows over 36 ticks, thresholds 64/256, and 5,292 learned weights.
Validation selects checkpoints before the full-test audit; the earlier 94.77%
test result was already known. The 98% target was not reached.
