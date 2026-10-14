# Third sequential MNIST training round

**97.47% on all 10,000 official test images**, compared with 97.28% in the
previous round. The 98% target remains unmet. Validation selected `wide-precision` at 98.1000%
on 6,000 images before this round opened the test arrays.

| Execution | Correct / total | Accuracy | Spike/current/membrane differences |
|---|---:|---:|---:|
| Atlas native integer forward | 9,747 / 10,000 | 97.47% | Reference |
| Stock XyloSim | 9,747 / 10,000 | 97.47% | 0 / 141,480,000 |
| SpiNNaker2 custom C host kernel | 9,747 / 10,000 | 97.47% | 0 / 141,480,000 |

Each target matches 47,160,000 spike counts, 47,160,000 synaptic currents,
47,160,000 membrane values and 100,000 class logits, with zero tolerance and no
clock adjustment. All ten classes and every official test image are included.
The selected latent checkpoint, frozen integer model and independently compiled
native binary agree. Ten concrete AtlasIR replays match
141,480 neural values and their spike events.
Whole-sequence and 1/7/13-tick chunks agree on the first 64 images.

## Model and inference contract

The selected graph is **16 inputs → 126 recurrent hidden neurons → 5
physical outputs**, with 10,584 trainable weights, 18,522 stored matrix
entries and two recurrent groups. Inference uses **36 ticks** and
readout ticks `[16, 36)`. Hidden/output thresholds are
64/512; current and membrane shifts remain one.
The same frozen file, three integer weight banks and fixed two-of-five host
codebook are used by all executions. There is no per-target retraining,
recalibration, clock shift or alternate decoder.

The encoder is `rows-full-counts`, with per-image host deskew and
8 class-independent query ticks. Input counts are 0–15.
One model processes one sequence per image. The task remains block-sequential
MNIST, not the canonical 784-step benchmark or a native event-camera dataset.
The cost includes host preprocessing and the neural sequence. These software
checks do not measure chip latency, energy, physical placement or deployment.
SpiNNaker2 executes the Atlas-owned C99 host kernel through SDK graph containers;
it does not execute stock Brian2Backend neurons or PE firmware.

## What changed

- Added training-only smooth elastic displacement and an interleaved row encoder
  that retains all 784 pixels in 56 data ticks. The encoder converts unsigned
  pixels before scaling, so values such as 255 cannot overflow during count
  quantization. These were candidates, not automatic changes to selected inference.
- Added checkpoint weight averaging followed by fresh native training. It yields
  one frozen set of weights; it does not ensemble predictions.
- Expanded recurrent v2 training/export, Xylo temporal conversion and the custom
  SpiNNaker2 integer host profile from 64 to 128 hidden neurons. V1 feedforward training/export retains 64. No integer
  neuron transition, rounding, saturation, gradient or optimizer formula changed.
- Preserved the Xylo limit of 63 nonzero recurrent targets per neuron. The wider
  candidate has two independent 63-cell groups feeding one jointly trained
  readout. After each native Adam update, cross-group weights and both optimizer
  moments are projected to zero. Active input, recurrent and readout weights
  all train in native Atlas; no external training framework is used.
- Compared two wider initializations: split readout weights with threshold 256,
  or retain readout weights with threshold 512. Both require new native training;
  neither is asserted equivalent to its narrower parent at initialization.

The 128-hidden bound is a tested compiler/runtime limit, not the silicon maximum.
A 128-hidden boundary regression compares actual SDK traces with native results,
including chunk boundaries. The C kernel also passes address/undefined-behavior
sanitizers at its largest admitted dimensions. A separate four-training-image
126-hidden preflight checked both SDKs before the larger MNIST runs.

## Validation selection

Training used original indices `[0, 54000)` and validation `[54000, 60000)`.
The previous 94.77% and 97.28% official-test results were already known. This is
an iterative development benchmark, not a globally untouched test set. Within
this round, candidate selection used validation only; no official test arrays
were opened until the selected model and encoder were fixed. The audit first
reproduced the selected validation score, then evaluated the full test set.
No model tuning followed that final test audit.

| Candidate | Best validation | Updates in this stage | Hidden | Ticks |
|---|---:|---:|---:|---:|
| Previous-round frozen baseline | 97.9000% | 8,000 | 63 | 36 |
| long-affine | 97.9667% | 16,000 | 63 | 36 |
| elastic | 97.7667% | 16,000 | 63 | 36 |
| margin-elastic | 97.6333% | 16,000 | 63 | 36 |
| long-query | 97.7833% | 12,000 | 63 | 44 |
| interleaved | 96.8833% | 12,000 | 63 | 64 |
| wide-126 | 98.0000% | 6,000 | 126 | 36 |
| wide-precision | 98.1000% | 6,000 | 126 | 36 |
| averaged-fine | 98.0333% | 2,000 | 63 | 36 |
| averaged-refine | 98.0500% | 6,000 | 63 | 36 |

These runs combine changes to encoding, initialization, schedule and augmentation;
the table does not isolate a causal effect for each change. Every selected
checkpoint is retained with its exact training-script snapshot, arguments,
hashes and validation history. Checkpoints may precede the final training step.
The winning local checkpoint is update 6,000.

A separate training-only diagnostic found 97.1833% accuracy for the parent on
6,000 randomly selected training images, motivating the capacity experiments.
A validation-only probe of the parent's fixed 1/3/5/9-view policies produced
97.9000%, 97.8500%, 97.8667% and 97.8500%. None improved single-view validation,
so multi-view inference was rejected and never received an official-test audit.
Raw validation predictions and the probe source are retained.

## Memory controls and interpretation

| Full-test control, selected weights without retraining | Accuracy |
|---|---:|
| Selected recurrent model | 97.47% |
| Zero all recurrent weights | 10.24% |
| Clear neural state before tick 16 | 18.11% |

The reset preserves accumulated output counts. These controls test the selected
model's dependence on recurrent memory, not whether a separately trained
feedforward model could solve MNIST. There are 50 tied highest
class scores, resolved consistently by lowest-index argmax.
Compared with the previous round, this test run changes accuracy by
+0.19 percentage points and the error count
from 272 to 253. Exact conversion and classification accuracy are
separate findings. The 98% target remains unmet.
The gain is small relative to doubling the trainable weight count. This run
does not establish a favorable accuracy/compute trade-off for widening; the
smaller previous fixture remains available. Paired prediction changes are
recorded in [the comparison receipt](round3/evidence/previous-round-comparison.json):
87 previous errors were corrected, while 68 previously correct images became errors.

## Evidence and reproduction

- [Frozen model](round3/fixtures/frozen.json), [training report](round3/fixtures/training.json),
  [validation selection](round3/fixtures/selection.json).
- [Full audit](round3/evidence/audit.json), [per-image logits and labels](round3/evidence/per_sample.npz),
  [candidate protocol](round3/evidence/protocol.json), [file hashes](round3/manifest.json).
- [Candidate checkpoints and source snapshots](round3/candidates/),
  [regression receipts](round3/evidence/regression.json), [native build hashes](round3/evidence/builds.json).

The original `fixtures/`, `evidence/` and `refinement/` are unchanged. Large raw
trace shards are retained outside Git; the audit records each shard's hash and
comparison totals. The retained per-image data can independently reproduce
accuracy and confusion matrices, but not substitute for rerunning state traces.

From the repository root, build the native binaries and install the separate
SDK environments described in the example README, then run:

```sh
export PYTHONPATH="$PWD/brian2-rust/python:$PWD"
python brian2-rust/examples/neuromorphic/sequential_mnist/verify.py \
  --dataset /path/to/mnist.npz \
  --trained brian2-rust/examples/neuromorphic/sequential_mnist/round3/fixtures \
  --train-runner /path/to/b2-train --runner /path/to/b2-runner \
  --xylo-python /path/to/xylo/bin/python \
  --spinnaker2-python /path/to/spinnaker2/bin/python \
  --frozen-only --output /tmp/sequential-round3-audit
```

The recorded audit did not use `--frozen-only`: it also compared the selected
latent checkpoint and an independently compiled Rust 1.86.0 trainer. A rebuilt
executable can reproduce frozen inference, but cannot restore a checkpoint bound
to a different executable hash. Native wide binaries were built with the pinned
Rust 1.98.1 toolchain; this round's independent 1.86.0 build is not a wheel package.
For training reproduction, use each retained script and CLI arguments, remapping
only dataset, executable, parent checkpoint and output paths. New output paths
must not already exist. Parent checkpoints from the previous round remain in
`refinement/`.

**96 affected Python tests and 51 Rust library tests passed with no skips.**
After the final widening-initialization edit, all 17 sequential MNIST tests were
rerun successfully; those 17 are included in the 96, not additional distinct tests.
