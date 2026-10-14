# Native integer training on block-sequential MNIST

This experiment trains all input, recurrent and readout weights with Atlas's
native Rust integer forward pass, surrogate BPTT, quantization STE and Adam.
It then freezes one checkpoint and exports the same neural graph and integer
weight banks through AtlasIR to stock XyloSim and the custom SpiNNaker2 C host
kernel. No TensorFlow or PyTorch training is used.

The task contains all ten MNIST classes. This is a block-sequential image task,
not the standard 784-step, one-pixel-per-tick sMNIST benchmark, and not an event
camera dataset. Host preprocessing starts with ordinary MNIST images.

## Measured result

**94.77% on all 10,000 official test images. The 98% target was not reached.**

| Execution | Correct / total | Accuracy | Neural trace differences vs native |
|---|---:|---:|---:|
| Atlas native integer forward | 9,477 / 10,000 | 94.77% | Reference |
| Stock XyloSim | 9,477 / 10,000 | 94.77% | 0 / 44,880,000 |
| SpiNNaker2 custom C host kernel | 9,477 / 10,000 | 94.77% | 0 / 44,880,000 |

Each target comparison includes 14,960,000 values of each of spike counts,
synaptic currents and membrane potentials, plus 100,000 identical class logits.
There are no tolerance or clock adjustments. Ten concrete AtlasIR replays check
another 44,880 state values and their nonzero spike events. The selected latent
checkpoint, frozen weights and separately packaged native binary also agree on
all test images. The training executable used Rust 1.86.0; the packaged build
used the repository's pinned Cargo 1.98.1. Both executable hashes are recorded.

The selected model is **16 inputs → 63 recurrent hidden neurons → 5 physical
outputs**, with 5,292 trainable weights, 22 ticks, and readout ticks `[8, 22)`.
The fixed two-of-five codebook supplies ten class scores. Hidden/output
thresholds are 16/64, synaptic and membrane shifts are 1, weight scale is 32,
and logit scale is 12. All three weight banks are trained.

Training used 5,000 updates at learning rate 0.001, then 8,000 more at 0.0003
with preserved Adam moments and a new sampling seed. The best validation
checkpoint was at total optimizer step 11,000: **95.23%** on 6,000 validation
images. Later validation tied and then decreased; earliest-tie selection kept
that checkpoint. No further training or selection followed the test result.

| Full-test control | Accuracy |
|---|---:|
| Selected model | 94.77% |
| Set all recurrent weights to zero | 10.10% |
| Clear currents, membranes and pending spikes before tick 8 | 18.93% |

These are ablations of the selected model, without retraining. They show its
dependence on recurrent state; they do not compare against a separately trained
feedforward classifier. Counts accumulated before reset are preserved. In the
selected readout window, no counts have accumulated before tick 8.

There are 195 tied highest-score predictions, resolved to the lowest class index
without label-dependent handling. The confusion matrix and every sample's logits
are retained. This experiment establishes exact conversion of a learned model
on real data; it does not establish a 98% classifier, a chip accuracy ceiling,
or hardware performance.

Evidence: [full audit](evidence/audit.json), [per-image results](evidence/per_sample.npz),
[validation selection](fixtures/selection.json), [training history](fixtures/training.json),
[frozen model](fixtures/frozen.json), and [regression checks](evidence/regression.json).
The five completed count-input candidates had best validation accuracies of
91.58% (initial five-output model), 89.95% (delayed-only readout), 93.45%
(eight-output codebook), 92.15% (earlier-checkpoint refinement), and 95.23%
(selected refinement). [Exploratory binary runs](evidence/exploratory.json) are
also retained, including unsuccessful configurations. These are combined recipe
comparisons, not an isolated measurement of encoding alone.

## Data and selection

The Keras-distributed `mnist.npz` file must have SHA-256
`731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1`.
The first 54,000 official training images train the model; the remaining 6,000
select checkpoints and encoding/training settings. Training does not index the
`x_test` or `y_test` archive members. The official 10,000-image test split is
opened only after model selection. Every test image is evaluated, with no class
filter, confidence filter or discarded tie. `argmax` resolves ties to the lowest
class index; reports also count ties.

The short count encoder averages each 2x2 image block to produce a 14x14 image,
rounds intensities to 0–15 using nearest-even rounding, and presents one row per
tick on 14 input channels. Channel 14 supplies a constant count of one; channel
15 supplies a count of one during eight final query ticks. Both auxiliary inputs
are identical for all images and labels. One scan therefore takes 22 ticks.

The exploratory binary encoders use either 49 thresholded 4x4 patches or 14
pooled rows encoded over four binary ticks each. The latter can add the same
constant/query channels. The experiment records the chosen encoder, readout
window and all validation checkpoints explicitly.

## Ten classes within eight physical outputs

Xylo Audio 2 has [eight physical output neurons](https://rockpool.ai/devices/xylo-overview.html). The implementation retains that
limit. An optional fixed binary `class_codes` matrix maps physical output counts
to class scores:

```
class_count[c] = sum(output_count[j] * class_codes[c][j])
logit[c] = class_count[c] * logit_scale / readout_ticks
```

The five-output codebook assigns the ten pairs of active outputs to digits 0–9.
The eight-output alternative uses ten balanced Hadamard codes with pairwise
Hamming distance at least four. Codes are fixed before training; they have no
learned parameters. Native cross entropy differentiates through the fixed map
to train the neural weights. Conversion preserves the exact same codebook.
Integer count sums are formed before floating-point scaling.

**This class decoder executes on the host.** AtlasIR contains the physical neural
network; the frozen artifact and native conversion bridge carry the explicit
readout contract. The experiment does not claim ten physical Xylo outputs or an
on-chip implementation of the decoder.

## Verification

`verify.py` compares the selected latent checkpoint against its frozen integer
weights, then checks every test sample against two isolated, label-free SDK
workers. Each worker receives only the converted bundle and input counts.

- Full-test accuracy, confusion matrix, tie count and per-image logits.
- Exact hidden/output spike counts, synaptic currents and membrane potentials.
- Identical input, recurrent and readout banks in both exported plans.
- Zero time shift and zero numerical tolerance.
- Whole versus 1/7/13-tick execution on the first 64 test images.
- Concrete AtlasIR replay of the first test image from each class.
- Full-test controls with recurrent weights zeroed or neural state cleared at
  the readout boundary. The state-reset control preserves accumulated counts.

Trace files are sharded to bound memory use. Committed evidence retains hashes,
per-image results, training history and the selected frozen model. Intermediate
trace shards can be regenerated by the commands below.

XyloSim uses an official validated Audio 2 configuration. SpiNNaker2 uses SDK
graph containers and Atlas's independent C99 host implementation of the same
integer dynamics. It is not stock `Brian2Backend`, PE firmware, hardware routing
or physical chip execution. Neither result measures chip power or latency.

## Reproduce

Build the current `b2-train` and `b2-runner`, and set `PYTHONPATH` to the repository
root and `brian2-rust/python`. Optional SDK environments remain separate:
Xylo requires Rockpool, XyloSim and Samna; SpiNNaker2 requires the SDK and a C99
compiler. Training checkpoints bind to the exact native trainer executable.
The checked-in optimizer checkpoint requires the recorded training binary.
For portable inference of `fixtures/`, pass `--frozen-only` to skip only the
latent-checkpoint comparison; native/Xylo/C trace checks still run on all images.
An optional `--portable-train-runner` additionally compares another native build.
The recorded full audit did not skip the checkpoint comparison.

```
python brian2-rust/examples/neuromorphic/sequential_mnist/train.py \
  --dataset /path/to/mnist.npz --runner /path/to/b2-train \
  --output /tmp/sequential-trained --encoding rows-counts \
  --hidden 63 --threshold 16 --output-threshold 64 --rec-init .2 \
  --lr .001 --readout-ticks 14 --updates 5000 --validate-every 1000

python brian2-rust/examples/neuromorphic/sequential_mnist/train.py \
  --dataset /path/to/mnist.npz --runner /path/to/b2-train \
  --resume /tmp/sequential-trained --output /tmp/sequential-refined \
  --encoding rows-counts --codebook pairs5 --hidden 63 --threshold 16 \
  --output-threshold 64 --rec-init .2 --lr .0003 --readout-ticks 14 \
  --updates 8000 --validate-every 1000 --seed 20261016

python brian2-rust/examples/neuromorphic/sequential_mnist/verify.py \
  --dataset /path/to/mnist.npz --trained /tmp/sequential-refined \
  --train-runner /path/to/b2-train --runner /path/to/b2-runner \
  --xylo-python /path/to/xylo/bin/python \
  --spinnaker2-python /path/to/spinnaker2/bin/python \
  --output /tmp/sequential-audit
```

The training report records exact command arguments. `--resume` restores the
selected optimizer checkpoint, preserves Adam moments and permits only a changed
learning rate or sampling seed. It does not transfer optimizer state to a chip.

`select_model.py --runs /path/to/run1 /path/to/run2 --output /tmp/selected`
compares completed validation reports without opening MNIST. The report records
all candidates and the selection rule. The checked-in training snapshots preserve
the exact scripts used in both stages. Local verification passed 124 affected
Python tests and 51 Rust library tests, plus packaged-SDK execution checks.
