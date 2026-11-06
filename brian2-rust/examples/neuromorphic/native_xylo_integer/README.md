# Native integer training for Xylo Audio 2

Atlas now trains with the admitted Xylo integer transition in its native CPU
`b2-train` engine. The forward pass uses quantized weights, integer current and
membrane state, saturation, minimum decay and Xylo spike-count limits throughout
training. Freezing exports the weights already used by that forward pass.
**No neuron-dynamics adaptation or additional quantization is needed at export.**

This is a dedicated, bounded training plan, not automatic differentiation of
arbitrary AtlasIR. It joins the existing `xylo-audio2-temporal-v1` conversion
profile through an explicit Brian2 model and AtlasIR.

The [v2 recurrent extension](../native_xylo_recurrent/README.md) adds native
recurrent training and chunked inference for a delayed order task. This page
and its MNIST evidence describe the original feedforward v1 contract.

## Held-out accuracy

All results below use **all 4,157 official MNIST test images with labels 0–3**.
They are four-class results, not full ten-class MNIST or the older 98.74% MLP.

| Model | Input / hidden / output | Ticks | Native integer | XyloSim |
|---|---|---:|---:|---:|
| Previous float-LIF model with adapted Xylo dynamics | 16 / 32 / 4, binary input | 16 | Not applicable | 3,143 / 4,157 = 75.61% |
| New integer training, same input encoding and dimensions | 16 / 32 / 4, binary input | 16 | 3,227 / 4,157 = 77.63% | 77.63% |
| New integer training, pulse-density input and wider hidden layer | 16 / 64 / 4 | 32 | 3,832 / 4,157 = **92.18%** | **92.18%** |

The last model improves on the previous Xylo result by **16.57 percentage
points**. The smaller control improves by 2.02 points. These are different
trained checkpoints, not a same-weight performance comparison. The control
retains the previous input encoding and dimensions, but training dynamics,
initialization and optimization differ; it is not a causal ablation of integer
rounding alone. The larger gain also involves input encoding, width and duration.
One seed was evaluated; no statistical significance claim is made.

## Exact conversion checks

For the 16/64/4 model, native integer training evaluation and stock XyloSim agree
on all 4,157 samples, without tick shifts or output correction:

| Compared quantity | Values | Differences |
|---|---:|---:|
| Per-tick hidden and output spike counts | 9,045,632 | 0 |
| Synaptic currents | 9,045,632 | 0 |
| Post-reset membrane states | 9,045,632 | 0 |
| Classification logits | 16,628 | 0 |
| Predicted classes | 4,157 | 0 |

Evaluating the frozen weights also matches the selected training checkpoint
exactly for every recorded quantity. A separate concrete AtlasIR replay of
32 test samples matches all **208,896** current/membrane/spike-count values and
all event ticks/indices. The smaller control has the same exact checks over
2,394,432 values per state type and a 55,296-value AtlasIR replay.

The SDK runs only the exported configuration and input raster; it receives no
labels or reference values. Samna accepts both hardware-sized configurations.
The stock simulator is reset between images, with no replicated graph and no
patches to vendor dynamics. This does not establish physical placement,
hardware execution, power or chip latency.

The existing SpiNNaker2 LIF profile does **not** admit this new integer-current
model. Exact native→Xylo inference is established; a shared native temporal
transition across two chip SDKs remains open. The previous
[float-LIF adaptation](../native_xylo/README.md) stays available with explicit
approximation opt-in and its original evidence.

## Forward and backward contract

Plan schema: `atlas-xylo-training-plan-v1`. Frozen schema:
`atlas.native-xylo-frozen.v1`. Both banks are trainable; no external ML framework
is used for forward execution, gradients or optimizer updates.

For each tick and neuron, with `clip16` saturating to [-32768,32767]:

```text
D(z) = (z >> 1) + int(z == 1)
I_sum = previous I
for incoming edges in source order:
    I_sum = clip16(I_sum + integer_weight * input_count)
I = I_sum - D(I_sum)
u = clip16(previous V + clip16(-D(previous V) + I))
q = clamp(floor(u / threshold), 0, spike_cap)
V = u - q * threshold
```

Input events enter the hidden layer in the current tick. Readout neurons use
hidden counts from the previous tick. Hidden/output spike caps are 31/1;
threshold equality produces a spike. All sequences start with zero state.
Latent f64 weights are constrained to the signed int8 range divided by a
power-of-two scale. Forward weights use nearest-even rounding of latent weights
times that scale, exactly as frozen export does.

The backward pass is explicitly approximate: it uses a sum of fast-sigmoid
surrogates at spike-count boundaries, detached reset, a quantization STE and
nominal half-decay derivatives through integer shifts/minimum decay. Saturated
operations block gradients outside the int16 interval. Native Adam or SGD
updates latent weights and optimizer moments in f64. Gradients are not claimed
to be derivatives of the discontinuous integer forward function. An independent
local-surrogate finite-difference oracle checks the implemented reverse pass.

Current bounds: CPU only, one dense hidden layer, 1–16 inputs, 1–64 hidden
neurons, 1–8 outputs, binary input rasters, 1–128 ticks, shifts fixed at one,
positive integer thresholds, no bias, no recurrence, no state carry, no on-chip
learning. The 64-hidden bound belongs to this compiler pilot. General Brian
training plans, GPU/MPI training and trainable neuron parameters are not admitted
by this new plan. Tape allocation is bounded before forward/backward buffers.

## Training and selection

Training uses 22,310 images from original training indices below 54,000;
validation uses 2,444 images from the remaining 6,000 indices. Both contain only
labels 0–3. Both runs use seed 20261014, 1,800 updates of 64 examples, native
Adam, weight scale 32 and thresholds 32. Validation every 300 updates selects
the highest accuracy with earliest ties. Step 1,800 is selected in both runs:
92.31% validation for the larger model and 78.81% for the control. Test arrays
are read only after selection in `verify.py`.

The larger model averages 7×7 pixel blocks into 4×4 features, rounds each average
to a level from 0 to 16, then emits a deterministic pulse-density raster for
32 ticks. The control thresholds the same averages at 50 and repeats binary
inputs for 16 ticks, matching the previous experiment's preprocessing. This is
an image rate-encoding task, not evidence of temporal event-stream reasoning.

## API

```python
import numpy as np
from brian2_atlas import NativeXyloTrainer, xylo_training_plan
from brian2_atlas.neuromorphic import freeze_native_xylo, export_native_xylo, run_simulator

plan = xylo_training_plan([16, 64, 4])
rng = np.random.default_rng(20261014)
weights = [rng.normal(.02, .18, 16*64), rng.normal(.02, .12, 64*4)]
trainer = NativeXyloTrainer(plan, weights=weights, runner=train_runner)
trainer.step(binary_rasters, labels)
frozen = freeze_native_xylo(trainer)
bundle = export_native_xylo(frozen, "xylo-export", steps=32, runner=atlas_runner)
result = run_simulator(bundle, binary_rasters)  # requires optional Xylo SDKs
assert result["dynamics_equivalent"]
```

`step` commits optimizer state; `evaluate` and `gradients` leave it unchanged.
Fresh-process checkpoint restore verifies plan and native binary identity.
Live neuron-state continuation is rejected. Every exported projection must
contain a nonzero weight; all-zero banks are outside the current converter.

## Regression validation

195 affected Python tests pass, including 18 new integer-training tests, actual
XyloSim and SpiNNaker2 SDK checks, and existing native training/bridge tests.
36 opt-in GPU/MPI cases are skipped; the new plan is CPU-only. All 51 Rust
library tests pass. Boundary checks include saturation, nearest-even rounding,
negative/minimum decay, threshold equality, silence, repeated inputs and spike
caps. CI now builds `b2-train` and runs the new CPU tests; optional vendor SDK
checks remain local. See the [regression receipt](evidence/regression.json).

## Evidence and reproduction

- [Full held-out audit](evidence/audit.json), including confusion matrix, SDK
  versions, source hashes and AtlasIR replay receipts.
- [Selected checkpoint](fixtures/checkpoint.json), [frozen weights](fixtures/frozen.json)
  and [training/selection receipt](fixtures/training.json).
- [Per-sample predictions and counts](evidence/per_sample.npz).
- [Official configuration](evidence/hardware-config.json).
- [Smaller control audit](control/evidence/audit.json) and
  [control selection record](control/fixtures/training.json).

From the repository root, use `PYTHONPATH=brian2-rust/python:.`. Build both native
binaries with the pinned toolchain. Keep the optional Rockpool/XyloSim/Samna
installation in its own environment.

```sh
python brian2-rust/examples/neuromorphic/native_xylo_integer/train.py \
  --dataset /path/to/mnist.npz --runner /path/to/b2-train --output /tmp/xylo-integer-trained

python brian2-rust/examples/neuromorphic/native_xylo_integer/verify.py \
  --dataset /path/to/mnist.npz --trained /tmp/xylo-integer-trained \
  --runner /path/to/b2-runner --train-runner /path/to/b2-train \
  --xylo-python /path/to/xylo-env/bin/python --output /tmp/xylo-integer-verified
```

For the control, add `--encoding binary --hidden 32 --ticks 16` to training.
Use a fresh output directory for each run. The checked-in checkpoint requires
the recorded native binary identity to restore; retrain with a different build,
or use the frozen artifact directly with `export_native_xylo` for inference.
The dataset hash is recorded in each audit. Full traces are regenerated rather
than checked in; per-sample arrays contain indices, labels, both predictions and
both count matrices. There is no accuracy selection on SDK or test results.
