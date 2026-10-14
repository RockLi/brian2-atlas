# Native recurrent Xylo training and stateful inference

Atlas trains a recurrent integer network, exports it through AtlasIR, and
executes the same model in stock XyloSim. Both native and SDK inference retain
state across chunks. A delayed order-discrimination task checks that the learned
model uses history rather than just event counts.

This extends the [integer feedforward trainer](../native_xylo_integer/README.md)
with a separate v2 plan. The v1 MNIST model and the earlier approximate float-LIF
adaptation keep their existing contracts and historical evidence.

## Task and held-out result

The network has **3 inputs, 24 recurrent hidden neurons and 2 outputs**, running
for 40 one-millisecond ticks. Inputs are A, B and a query channel. Each sequence
contains three A events, three B events and four query events. Class 0 is AB;
class 1 is BA. Readout counts include only ticks 32–39.

For each timing template, the dataset contains both labels with A/B swapped.
Both classes have identical channel counts and exactly the same suffix from
tick 22 onward. There are ten silent ticks before the query begins at tick 32.
A model needs information from the earlier symbol inputs to answer the query.
It can solve this simple task by retaining the last symbol; this is not a test
of complex temporal grammar or long-duration memory.

| Execution / control | Correct / 1,872 | Accuracy |
|---|---:|---:|
| Native Atlas, selected recurrent integer model | 1,872 | **100%** |
| Stock XyloSim, same frozen model and weights | 1,872 | **100%** |
| Native model with recurrent weights set to zero, no retraining | 936 | 50% |
| Native model with all neural state reset immediately before query | 936 | 50% |
| Any deterministic classifier using only per-channel event totals | At most 936 | At most 50% |

All 936 held-out paired inputs change prediction when A and B are swapped.
The recurrence and state-reset controls establish dependence on memory in this
selected model. They do not prove that no other architecture can solve the task.
This is a synthetic mechanism/engineering test, not a real-world accuracy or
energy benchmark. No MNIST comparison or hardware-performance claim is made.

## SpiNNaker2 transfer boundary

The [same-weight stock SDK diagnostic](SPINNAKER2.md) runs this topology and
all three frozen weight banks with `lif_curr_exp_32bit`. Accuracy drops to 50%,
and spike/current/membrane traces diverge under both raw and shifted clocks.
The existing exact SpiNNaker2 converter rejects this integer source. The
diagnostic does not add an equivalent SpiNNaker2 backend for this model.

## Training and split discipline

The generator enumerates all pairs of three positions chosen from eight slots
for each symbol window: 56 × 56 = 3,136 timing templates. A seeded permutation
splits templates before either class is instantiated:

- 1,800 training templates, 3,600 sequences.
- 400 validation templates, 800 sequences.
- 936 test templates, 1,872 sequences.

The split sets are disjoint and both classes remain balanced in every split.
Timing is new at test time within the same generator; the test does not measure
transfer to longer delays, noise distributions or new symbol vocabularies.

Native CPU `b2-train` performs 1,200 full-sequence BPTT/Adam updates of 32 samples,
seed 20261014. All three weight banks learn: input, readout and hidden recurrence.
The recurrent matrix starts near a positive diagonal to provide a memory-friendly
initialization. Validation starts at 67.875%. The earliest highest validation
checkpoint is step 200, at 100%; 69/72 input, 46/48 readout and 529/576 recurrent
latent weights differ from initialization. Later equal scores do not replace it.
Test labels, SDK results and ablations are used only after checkpoint selection.

Forward dynamics use the same integer rule as v1: int8 nearest-even weights,
int16 saturation, shifts of one, minimum decay, subtract reset and hidden/output
spike caps 31/1. External input is accumulated before recurrent input, each in
source order. Recurrence and readout consume the previous tick's hidden counts.
Backward uses fast-sigmoid count surrogates, detached reset and rounding STE,
including gradients through recurrent connections across all training ticks.
No TensorFlow/PyTorch or on-chip training is involved.

## Exact execution and chunking

On every test sequence, selected-checkpoint evaluation, frozen native evaluation
and XyloSim agree without timing shifts or numerical tolerance:

| Compared quantity | Values | Differences |
|---|---:|---:|
| Per-tick hidden/output spike counts | 1,946,880 | 0 |
| Synaptic currents | 1,946,880 | 0 |
| Post-reset membrane states | 1,946,880 | 0 |
| Readout logits | 3,744 | 0 |

Both native and SDK execution were compared with full runs using chunk sizes
7 and 13 on all 1,872 sequences. Persistent SDK sessions also used irregular
boundaries `[1, 9, 14, 22, 32, 33, 40]` on the full test set, checking cumulative
readout counts at each boundary. All comparisons are exact. Unit tests include
one-tick chunks, silence, repeated inputs, negative states and mixed saturation.

A separate concrete AtlasIR replay of 32 test sequences matches **99,840**
current/membrane/spike-count values plus all event ticks and neuron indices.
The SDK worker reads only the exported configuration and raster, without labels
or native reference arrays. The official Samna validator accepts the model.

## State and API contract

- Training plan: `atlas-xylo-training-plan-v2`, with `recurrent: true` and an
  absolute `readout_start` tick.
- Weight bank order: **input, readout, recurrent**, in source-major layout.
- Frozen artifact: `atlas.native-xylo-frozen.v2`; conversion still uses the
  existing `xylo-audio2-temporal-v1` profile, not a new target.
- Native evaluation continuation includes currents, post-reset membranes,
  previous spike counts, elapsed ticks and accumulated readout counts. Its
  identity binds the plan and quantized weights; mismatches, invalid dimensions,
  out-of-range states and an overlong horizon are rejected.
- A sequence starts from zero state. Evaluation chunks carry state explicitly;
  they do not mutate optimizer state. The SDK session retains its own simulator
  state and `reset()` starts a new independent sequence.
- Logits are accumulated output counts times `logit_scale`, divided by the
  number of elapsed ticks in the readout window. Before that window opens,
  counts/logits are zero and a classification is not yet available.

```python
from brian2_atlas import NativeXyloTrainer, xylo_recurrent_training_plan
from brian2_atlas.neuromorphic import (
    freeze_native_xylo, export_native_xylo, run_simulator, XyloInferenceSession,
)

plan = xylo_recurrent_training_plan([3, 24, 2], readout_start=32)
# weights contains flattened input, readout and recurrent banks.
trainer = NativeXyloTrainer(plan, weights=weights, runner=train_runner)
trainer.step(training_rasters, labels)  # full-sequence BPTT

first = trainer.evaluate(test_rasters[:, :14], test_labels)
last = trainer.evaluate(test_rasters[:, 14:], test_labels,
                        initial=first["continuation"])
# Or concatenate traces and return the cumulative readout in one call:
result = trainer.evaluate_chunks(test_rasters, test_labels, chunk_steps=7)

frozen = freeze_native_xylo(trainer)
bundle = export_native_xylo(frozen, "xylo-export", steps=40, runner=atlas_runner)
result = run_simulator(bundle, test_rasters, chunk_steps=7)

session = XyloInferenceSession(bundle)  # optional Xylo SDK environment
session.push(test_rasters[0, :14])
last = session.push(test_rasters[0, 14:])
session.reset()
```

Stateful **inference** is supported; state-carry training and TBPTT are rejected.
The current v2 horizon is at most 128 ticks total across chunks, with at most
16 inputs, 64 hidden neurons and 8 outputs, binary input, shifts fixed at one,
zero bias and CPU training only. This does not admit unbounded always-on streams,
GPU/MPI training, arbitrary Brian training graphs or SDK state restoration from
an Atlas checkpoint. Native continuation can be serialized and passed to another
native process with the same plan and quantized weights.

The SDK boundary test exposed a real resource limit: recurrent fanout is at most
63 nonzero targets per hidden neuron. Both frozen export and direct temporal
AtlasIR conversion now reject larger fanout before SDK execution. A 64-neuron
recurrent layer is allowed only when each row satisfies that bound. The mixed
saturation case with 63 recurrent targets passes actual SDK comparison.

## Evidence and reproduction

- [Held-out audit, controls, source hashes and replay receipts](evidence/audit.json).
- [Native checkpoint](fixtures/checkpoint.json), [frozen weights](fixtures/frozen.json)
  and [training/selection record](fixtures/training.json).
- [Per-sequence logits, counts, labels, template IDs and controls](evidence/per_sample.npz).
- [SDK/session execution receipt](evidence/execution.json) and
  [validated configuration](evidence/hardware-config.json).
- [Regression receipt](evidence/regression.json): 206 affected Python tests pass,
  with 36 opt-in GPU/MPI skips; all 51 Rust library tests pass. The existing
  50-case temporal suite still matches 505,455 state values, including 7-tick
  chunks. CI includes the new recurrent CPU tests; SDK checks remain optional.

From the repository root, set `PYTHONPATH=brian2-rust/python:.`, build the native
runners with the pinned toolchain and keep the optional SDK in a separate environment.

```sh
python brian2-rust/examples/neuromorphic/native_xylo_recurrent/train.py \
  --runner /path/to/b2-train --output /tmp/xylo-recurrent-trained

python brian2-rust/examples/neuromorphic/native_xylo_recurrent/verify.py \
  --trained /tmp/xylo-recurrent-trained --train-runner /path/to/b2-train \
  --runner /path/to/b2-runner --xylo-python /path/to/xylo-env/bin/python \
  --output /tmp/xylo-recurrent-verified
```

Use fresh output directories. Checkpoint restoration requires the recorded
native binary identity; with another build, retrain or export the frozen weights
directly. Full traces and concrete AtlasIR files are regenerated by verification;
compact evidence and hashes are checked in. All chip execution remains simulated;
physical deployment, power and hardware latency are unverified.
