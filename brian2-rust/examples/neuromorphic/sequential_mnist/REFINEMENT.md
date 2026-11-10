# Sequential MNIST refinement: training and encoding

This is a second development round after the first official-test result of
94.77% was known. Candidate and checkpoint selection uses the same 54,000/6,000
partition of the official training data. The test split is reserved for the
selected model's final accuracy and cross-backend audit in this round; it is
not a previously unseen benchmark across the whole project.

**97.28% on all 10,000 official test images**, up from 94.77% in the first
round. The 98% target was not reached. The selected validation score was 97.9000% on 6,000
images, reproduced before opening this round's test arrays.

| Execution | Correct / total | Accuracy | Spike/current/membrane differences |
|---|---:|---:|---:|
| Atlas native integer forward | 9,728 / 10,000 | 97.28% | Reference |
| Stock XyloSim | 9,728 / 10,000 | 97.28% | 0 / 73,440,000 |
| SpiNNaker2 custom C host kernel | 9,728 / 10,000 | 97.28% | 0 / 73,440,000 |

Each target matches 24,480,000 spike counts, 24,480,000 synaptic currents,
24,480,000 membrane values, and 100,000 class logits exactly. The latent
checkpoint, frozen model and independently packaged native executable agree.
Ten concrete AtlasIR replays also match 73,440 neural values and their spike
events. Whole-sequence versus 1/7/13-tick chunking agrees on the first 64 images.

## What changes

The integer neuron transition, 16-input/63-hidden network and both target
runtimes retain their existing arithmetic. Candidates use the existing fixed
five-output pair codebook or eight-output balanced codebook; neither exceeds
the eight physical output neurons. A separate
candidate scales warm-start weights and neuron thresholds together before
retraining, to use more of the int8 range; this is a changed parameterization,
not a claim of unchanged predictions during training. All input,
recurrent and readout weights learn in Atlas. The host preprocessing options
are explicit parts of the input contract:

- `--deskew` removes horizontal slant from per-image intensity moments, using
  zero-filled horizontal bilinear interpolation. It fits no dataset parameters.
- `--augmentation-shift` samples independent zero-filled translations for
  training batches only. Validation and inference never use random augmentation.
- `--augmentation-affine` adds training-only rotation and independent axis scaling,
  with zero-filled bilinear interpolation. The final trial used ±8 degrees and
  scales in 0.92–1.08. It does not add inference views.
- `rows-columns-counts` scans 14 pooled rows followed by the 14 pooled columns.
- `rows-full-counts` preserves all 28 vertical positions and pools horizontally.
- `rows-position-counts` adds an explicit 1–14 row-position count to the last
  input channel, followed by a constant query count of 15.
- `gradient-counts` supplies sixteen local descriptors: eight unsigned gradient
  orientation counts, intensity and relative edge strength, plus cell coordinates.
  This is fixed host feature extraction, not a learned CNN or raw pixel sequence.
- `--repeats` repeats the scan within a single stateful neural sequence.
- `--query-ticks` declares the final query duration. Total duration remains
  bounded to 128 ticks.

All encoders start from ordinary images. Deskew and scan construction run on the
host; the experiment does not claim native event-camera input or on-chip image
preprocessing. It remains block-sequential MNIST, not the 784-step benchmark.

`--resume` retains optimizer moments and only permits learning-rate/sampling-seed
changes to the native plan. `--warm-start` reuses latent weights but resets Adam,
allowing another encoding, threshold or readout window with the same neural
sizes and codebook. `--warm-weight-scale` optionally multiplies all three latent
weight banks and rejects an out-of-range initialization rather than clipping.
Each report records which operation was used, source hashes,
training-only augmentation and the selected validation checkpoint.

## Acceptance

Select by validation accuracy, reproduce the selected score on all 6,000
validation images with the current encoder, then run all 10,000 test images through native
Atlas, stock XyloSim and the custom SpiNNaker2 C host kernel. Require identical
frozen weight banks, every spike count/current/membrane value and all class
logits, with zero clock shift or tolerance. Report the measured accuracy even
if it remains below 98%. Preserve the previous result and all candidate reports.

## Selected model and cost

Validation selected `full-rows-affine`, local update 2,000.
The neural graph remains **16 inputs → 63 recurrent hidden neurons → 5 outputs**,
with **5,292 learned weights** and the same fixed host two-of-five class codebook.
All input, recurrent and readout weights were trained in native Atlas.
The hidden/output thresholds are 64/256; current and membrane shifts remain 1.
No neural transition or target runtime was changed in this refinement.

Host preprocessing deskews each image from its own intensity moments, then
averages horizontal pairs to produce 28 rows of 14 count-valued inputs (0–15).
One row arrives per tick. A constant input and eight final query ticks are
identical for all classes. Readout uses ticks `[16, 36)`.
There is one inference sequence per image, no ensemble or test-time augmentation.
This is **36 ticks instead of 22**, with unchanged neuron/weight counts;
the longer sequence and host deskew have a cost that this experiment does not
measure in hardware. It is not a power or latency result.

| Full-test control (same selected weights, no retraining) | Accuracy |
|---|---:|
| Selected recurrent model | 97.28% |
| Zero all recurrent weights | 10.23% |
| Clear neural state before tick 16 | 17.90% |

The state-reset control preserves accumulated output counts. There are
68 tied highest class scores; all use the same lowest-index argmax rule.
The error count changed from 523 to 272; this is a
2.51 percentage-point accuracy improvement and
48.0% fewer errors on this test set. It establishes
better classification with exact conversion, not a hardware result or an
accuracy ceiling for either chip.

## Candidate results

These are validation scores, not additional test-set evaluations. Each row
combines its encoding, initialization, training schedule and augmentation;
this is not a controlled attribution of gain to one change.

| Candidate | Best validation | Stage updates | Ticks | Outputs |
|---|---:|---:|---:|---:|
| First-round baseline | 95.2333% | 8,000 | 22 | 5 |
| rows-aug | 95.0333% | 6,000 | 22 | 5 |
| deskew-aug | 95.6167% | 6,000 | 22 | 5 |
| position-refine | 96.3500% | 6,000 | 22 | 5 |
| precision | 96.9333% | 8,000 | 22 | 5 |
| gradient | 90.1833% | 10,000 | 24 | 5 |
| precision-eight | 96.7333% | 8,000 | 22 | 8 |
| repeat-precision | 95.2833% | 7,000 | 50 | 5 |
| precision-clean | 97.4167% | 16,000 | 22 | 5 |
| full-rows-precision | 97.4333% | 8,000 | 36 | 5 |
| full-rows-clean | 97.8667% | 10,000 | 36 | 5 |
| full-rows-affine | 97.9000% | 8,000 | 36 | 5 |

The unfinished `cross-scan` and `position` runs stopped after a turn interruption;
they are recorded separately and were not eligible for final selection.
`position-refine` resumed the best available position checkpoint. Two early
runs were restarted after bounding preprocessing memory; exact encoding
regressions cover whole-array versus batched preprocessing. Aborted runs with
no validation checkpoint are retained as source snapshots and interruption
receipts, not successful training results.

A validation-only multi-view probe on `precision` reached 96.9333%, 97.1833%,
97.3333% and 97.2500% for 1/3/5/9 views respectively. The five-view gain required
five neural sequences per image and was not adopted. It was not evaluated on
the official test set or included in the cross-backend accuracy claim.
The raw validation predictions and integer-count rescoring script are retained.

## Evidence and reproduction

- [Selected frozen model](refinement/fixtures/frozen.json),
  [training report](refinement/fixtures/training.json) and
  [validation-only selection](refinement/fixtures/selection.json).
- [Full audit](refinement/evidence/audit.json),
  [per-image logits and labels](refinement/evidence/per_sample.npz),
  [regression checks](refinement/evidence/regression.json) and
  [file hashes](refinement/manifest.json).
- [Candidate reports, checkpoints and exact training snapshots](refinement/candidates/).
  The selected lineage is baseline `fixtures` → `deskew-aug` → `precision` →
  `full-rows-precision` → `full-rows-clean` → `full-rows-affine`. Each report records its parent hashes and all CLI
  arguments. Retained checkpoints are the best validation checkpoints, not
  necessarily the last optimization step.

The original `fixtures/` and `evidence/` remain unchanged. The same 54,000/6,000
training partition was reused. The prior 94.77% official-test score was known;
this is a second development round, not a globally untouched benchmark.
No further training or model selection followed this round's final test audit.

To reproduce inference, build the native binaries, install the separate SDK
environments described in the main example README, then run from the repo root:

```sh
export PYTHONPATH="$PWD/brian2-rust/python:$PWD"
python brian2-rust/examples/neuromorphic/sequential_mnist/verify.py \
  --dataset /path/to/mnist.npz \
  --trained brian2-rust/examples/neuromorphic/sequential_mnist/refinement/fixtures \
  --train-runner /path/to/b2-train --runner /path/to/b2-runner \
  --xylo-python /path/to/xylo/bin/python \
  --spinnaker2-python /path/to/spinnaker2/bin/python \
  --frozen-only --output /tmp/sequential-refinement-audit
```

`--frozen-only` permits a new native build without restoring a checkpoint bound
to another executable. It still rechecks validation and compares every test
spike, current, membrane and logit. The recorded audit used the original trainer
and additionally compared the latent checkpoint and a separately packaged build;
these comparisons were not skipped. Retraining the selected lineage from saved optimizer checkpoints
requires the recorded training executable; use the exact command arguments and
training-script snapshots for each stage, remapping only dataset, runner,
parent-checkpoint and output paths. The first-round README supplies the initial
training recipe. New output directories must not already exist.


The final training stage can be replayed from its saved parent with the original
training executable:

```sh
python brian2-rust/examples/neuromorphic/sequential_mnist/train.py \
  --dataset /path/to/mnist.npz --runner /path/to/recorded-b2-train \
  --resume brian2-rust/examples/neuromorphic/sequential_mnist/refinement/candidates/full-rows-clean \
  --output /tmp/sequential-affine-trained --encoding rows-full-counts --deskew \
  --hidden 63 --codebook pairs5 --threshold 64 --output-threshold 256 \
  --readout-ticks 20 --query-ticks 8 --augmentation-shift 1 --augmentation-affine 8 \
  --lr .0003 --updates 8000 --validate-every 2000 --seed 20261030
```

Ten affected regression tests passed with no skips, including both real SDK
environments, analytic resampling checks and preprocessing memory equivalence.
There are no Rust or target-runtime changes in this round. SpiNNaker2 still uses
Atlas's custom C99 host kernel with SDK graph containers, not stock Brian2Backend,
PE firmware or physical hardware. XyloSim uses the validated Audio 2 configuration.
