# Native Atlas training: same weights on SpiNNaker2 and Xylo

This experiment connects a native-trained model to a second software target.
One **16→32→4** dense LIF network is trained in Atlas, frozen once, and exported
to SpiNNaker2 and XyloSim with the same topology, integer weight banks and input
rasters. Both layers learn; no target-specific retraining or threshold fitting
is performed.

**The two targets do not execute the same transition function.** The existing
SpiNNaker2 bridge preserves the bounded native LIF dynamics. Xylo requires an
explicit, approximate port to its integer current/membrane model. Similar task
accuracy does not establish spike or state equivalence.

## Held-out results

The test set contains **all 4,157 official MNIST test images with labels 0–3**.
This is a four-class task with 16 binary input features, not the previous
10-class 98.74% MLP or the 196-input native SNN.

| Executor | Correct / 4,157 | Accuracy |
|---|---:|---:|
| Native Atlas, learned float weights | 3,175 | 76.38% |
| Native Atlas, frozen quantized weights | 3,142 | 75.58% |
| SpiNNaker2 SDK, same frozen weights | 3,142 | 75.58% |
| XyloSim, same frozen weights with adapted dynamics | 3,143 | 75.61% |

The one-image net accuracy difference is not an improvement claim: **234
predictions differ** between the SDKs. Quantization costs 0.79 percentage points
on this checkpoint. The strong 4×4 binary input compression limits what this
small task can demonstrate; it is a conversion comparison, not a competitive
MNIST benchmark.

## Pulse and state comparison

No time shifting, output correction or tolerance relaxation is used.

| Comparison | Values | Result |
|---|---:|---|
| Quantized native Atlas vs SpiNNaker2 spike raster | 2,394,432 | Exact |
| Reconstructed native final membrane vs SpiNNaker2 | 149,652 | Exact |
| SpiNNaker2 vs Xylo per-tick spike counts | 2,394,432 | 217,243 differ (9.07%) |
| SpiNNaker2 vs Xylo spike presence | 2,394,432 | 204,160 differ (8.53%) |
| SpiNNaker2 vs Xylo output counts | 16,628 | 8,121 differ |
| SpiNNaker2 vs Xylo membrane, target-scaled units | 2,394,432 | 2,154,749 differ; mean absolute difference 2.3580, maximum 60.9849 |

Xylo emits up to three hidden spikes per tick in this run; native LIF emits at
most one. Xylo's extra synaptic-current state has no counterpart in the source
LIF model. Membrane differences compare the declared scaled coordinates and
same tick indices; they are not an equivalence proof between different models.

A separate concrete AtlasIR replay uses the first 32 images of the same test
subset on each path:

- The native-LIF AtlasIR model matches 18,432 SpiNNaker2 voltage values and all
  spike events exactly.
- The explicitly adapted Xylo AtlasIR model matches **55,296 current, membrane
  and spike-count values**, plus all event ticks/indices, exactly in XyloSim.
- The Xylo configuration is accepted by the official Samna configuration
  validator. Each SDK sample starts with reset state on the same 16/32/4 graph;
  sample batching does not expand Xylo's hardware dimensions.

Thus the adapted AtlasIR → Xylo lowering is verified on the replay subset.
Native LIF → Xylo remains a measured approximation, not an exact conversion.

## Why an explicit adaptation is necessary

The existing source model uses floating-point beta=0.5 LIF dynamics, a strict
`v > threshold` test and one subtract-reset spike per tick. Its SpiNNaker2
coordinate transform uses `target_weight = scale * beta * learned_weight`.

The Xylo port reuses those frozen integer weights and the scaled thresholds
without further rounding. It uses synaptic and membrane shifts of one, zero
bias, zero recurrence and the existing `xylo-audio2-temporal-v1` profile:

- Synaptic current retains history and decays after incoming events.
- State decay uses integer shifts with minimum decay, and signed 16-bit
  saturation.
- Threshold equality emits a spike; hidden neurons allow up to 31 spikes per
  tick, while output neurons allow one.
- Scheduling differs from the source LIF; no post-hoc alignment is applied.

Using shift one matches the source membrane decay coefficient. It does not
remove the extra current filter or integer rounding. The converter records
`dynamics_equivalent: false`; the strict default rejects Xylo without opt-in.
This experiment does not demonstrate a general exact native-LIF mapping to
Xylo, and does not prove such a mapping impossible.

## API

```python
from brian2_atlas.neuromorphic import export_native_lif, run_simulator

# frozen is the same integrity-checked artifact used for SpiNNaker2.
bundle = export_native_lif(
    frozen, "xylo-export", steps=16, runner=atlas_runner,
    target="xylosim", allow_dynamics_approximation=True,
)
result = run_simulator(bundle, binary_rasters)  # batch × 16 × 16
```

The adapted LIF bridge preserves the source trainer's finite, positive
`logit_scale`, including values above 100. This scale is applied to observed
output counts on the host. The separate native Xylo integer training plans
retain their `(0, 100]` bound. Tests cover export admission and stock XyloSim
state, count and logit comparisons at scales 4 and 128.

The new path accepts exactly one hidden layer, at most 16 inputs, 64 hidden
neurons and 8 outputs, 1–16 ticks, binary inputs, integer scaled thresholds in
[1,32767], and the existing frozen weight range [-15,15]. Input and readout
projections must contain nonzero weights. It exports fresh inference only;
optimizer state, state continuation and on-chip learning are not supported.
SDK dependencies remain optional and isolated from conversion.

## Training and selection

`train.py` reads only the official training arrays. Of the original first
54,000 training indices, 22,310 images have labels 0–3; the remaining 6,000
indices contain 2,444 validation images in those classes. Preprocessing averages
7×7 blocks into 4×4 features, thresholds at 50, and repeats binary inputs for
16 ticks.

Native CPU `b2-train` performs surrogate BPTT and Adam for 900 updates, seed
20261014. The checkpoint with highest quantized **native validation** accuracy
is selected, with earliest ties. The selected checkpoint is step 600; all 512
input and 128 readout weights differ from initialization. No SDK test accuracy
or test labels participate in selection. Both targets receive the same
nearest-even quantized checkpoint with scales [16,16], without clipping.

## Regression validation

127 affected Python tests pass with both installed SDK environments enabled.
The existing 50-case Xylo temporal suite was also rerun: all 505,455 recorded
state values match, and 7-step chunking remains identical to a full run.
See the [regression receipt](evidence/regression.json).

## Evidence and reproduction

- [Comparison audit, confusion matrices and source hashes](evidence/audit.json)
- [Training and checkpoint-selection record](evidence/training.json)
- [Per-sample predictions and differences](evidence/per_sample.npz)
- [Frozen weights](fixtures/frozen.json) and [native checkpoint](fixtures/checkpoint.json)
- [Validated Xylo hardware configuration](evidence/hardware-config.json)

The per-sample `predictions` columns are original test index, true label, native
float prediction, native quantized prediction, SpiNNaker2 prediction and Xylo
prediction. Full traces and concrete AtlasIR replay files are regenerated by
`verify.py`; their hashes are recorded in the audit.

From the repository root, set `PYTHONPATH=brian2-rust/python:.`. Supply a trusted
native Atlas runner and native training runner. Use separate SDK environments:
this run used py-spinnaker2 0.8.0 with upstream Brian2 2.10.1, and Rockpool /
XyloSim / Samna versions recorded in the audit.

```sh
python brian2-rust/examples/neuromorphic/native_xylo/train.py \
  --dataset /path/to/mnist.npz --runner /path/to/b2-train --output /tmp/xylo-trained

python brian2-rust/examples/neuromorphic/native_xylo/verify.py \
  --dataset /path/to/mnist.npz --trained /tmp/xylo-trained \
  --runner /path/to/b2-runner --train-runner /path/to/b2-train \
  --spinnaker-python /path/to/spinnaker-env/bin/python \
  --xylo-python /path/to/xylo-env/bin/python --output /tmp/xylo-comparison
```

To replay the checked-in checkpoint, use `--trained` pointing to `fixtures/`
(the training receipt is also included there). Dataset SHA-256 is
`731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1`.

All execution is software-only. Configuration validity does not establish
physical placement, chip timing, power or latency. SpiNNaker2 here uses its
upstream Brian2 backend, not an independent silicon emulator.
