# Fourth sequential MNIST training and readout round

**97.44% on all 10,000 official test images** (9,744/10,000), compared
with 97.47% in the previous round. The 98% test target remains unmet.
Validation selected `compact-cosine` at 98.1333% on 6,000 images before this round
opened the official test arrays.

| Execution | Correct / total | Accuracy | Spike/current/membrane differences |
|---|---:|---:|---:|
| Atlas native integer forward | 9,744 / 10,000 | 97.44% | Reference |
| Stock XyloSim | 9,744 / 10,000 | 97.44% | 0 / 73,440,000 |
| SpiNNaker2 custom C host kernel | 9,744 / 10,000 | 97.44% | 0 / 73,440,000 |

The same frozen neural model, integer weights and fixed class codebook run in
all three executions. Each target matches **24,480,000 spike counts,
24,480,000 currents and 24,480,000 membrane values**, plus all 100,000 class
scores. There is no tolerance, clock shift, target-specific retraining or
recalibration. The selected latent checkpoint and an independently compiled
native binary agree with frozen inference. Ten concrete AtlasIR replays match
73,440 neural values and spike events;
1/7/13-tick chunks agree on the first 64 images.

The branch now retains this compact model as a fixed conversion regression
baseline. Further accuracy tuning is deferred. This acceptance decision does
not change the recorded result or claim that the 98% target was achieved.
See the [branch acceptance record](../ACCEPTANCE.md).

## Selected model

The selected graph is **16→63→5**, with **5,292 active trainable weights**,
5,292 stored matrix entries and 36 ticks.
Hidden/output thresholds are 64/256;
readout uses ticks [16, 36) and `spikes`.
The selected change is a cosine learning-rate schedule on the compact spike-count
model, warm-started from round three's `averaged-refine` checkpoint (98.05%
validation). It retains the original two-of-five host codebook and readout window.
Active weights are halved from the previous selected 126-hidden model's 10,584,
while test accuracy is 0.03 percentage points lower. This is a smaller model at
similar accuracy, not a new accuracy best; speed and energy were not measured.

Host preprocessing remains per-image deskew, horizontal averaging of pixel
pairs and deterministic 0–15 count encoding over 28 image rows, followed by
8 class-independent query ticks. The task is block-sequential MNIST, not the
canonical 784-step benchmark or an event-camera dataset. No TensorFlow,
PyTorch, learned host classifier, prediction ensemble or test-time augmentation
is used. Input, recurrent and neural readout weight banks train in native Atlas.

Compared with the preceding wider test result, this model corrects
70 previous errors and introduces 73 new errors;
173 predictions change. These paired statistics were computed after
model selection and were not used to choose another candidate.

## Training and readout experiments

- Cosine decay changes the learning rate while preserving Adam moments. The
  compact control and auxiliary-rate candidate use the same sampling seed and
  augmentation, so their training objectives can be compared directly.
- The optional rate objective adds mean squared error between each output's
  average spike rate and fixed 0.1/0.9 class-code targets. Its derivative is
  implemented in the native reverse pass; it adds no inference weights.
- Eight-output initialization retains input and recurrence weights and fits a
  readout from codebook entries alone. It does not use sample labels or claim
  initial spiking equivalence. All three banks then retrain natively.
- The optional current objective uses signed output Isyn sums, a fixed codebook
  and output-threshold normalization. It directly seeds current derivatives
  during BPTT. For fixed weights, the readout choice leaves neural transitions
  and emitted spikes unchanged. The decoder
  runs on the host and needs state recording; this is not an on-chip classifier.
  Its distinct `atlas.native-xylo-current-bridge.v1` export tag makes older
  runtimes reject the decoder rather than silently use spike counts.
- Two compact refinements test clean-sample fine-tuning and a later spike-readout
  window. Network width and input encoding remain fixed.

Validation-only probes also tried raw current, rectified current, pre-reset
voltage and a final membrane correction on existing weights. Direct decoder
substitution did not provide a substantial improvement; the current objective
was therefore retrained rather than reported as a successful drop-in change.
Only the selected model received this round's full-test audit. Other candidates'
validation results are not test accuracies or full-dataset SDK results.

## Validation selection

| Candidate | Readout | Sizes | Best validation | Selected / completed stage updates |
|---|---|---|---:|---:|
| Previous round baseline | spikes | 16→126→5 | 98.1000% | 6,000 / 6,000 |
| compact-cosine | spikes | 16→63→5 | 98.1333% | 2,000 / 8,000 |
| compact-rate | spikes | 16→63→5 | 98.0333% | 2,000 / 8,000 |
| code8-ce | spikes | 16→63→8 | 98.1000% | 11,000 / 12,000 |
| code8-rate | spikes | 16→63→8 | 98.0833% | 10,000 / 12,000 |
| wide-rate | spikes | 16→126→5 | 98.1000% | 2,000 / 8,000 |
| current5 | synaptic_current | 16→63→5 | 97.6833% | 11,000 / 12,000 |
| current8 | synaptic_current | 16→63→8 | 98.0667% | 11,000 / 12,000 |
| compact-clean | spikes | 16→63→5 | 98.1167% | 3,000 / 6,000 |
| compact-late | spikes | 16→63→5 | 97.8667% | 6,000 / 8,000 |

Training uses original indices [0, 54000); validation uses [54000, 60000).
The earlier 94.77%, 97.28% and 97.47% test results were already known. This is an
iterative development benchmark, not a globally untouched test set. Candidate
and checkpoint selection within this round used validation only, with earliest
checkpoint ties and listed candidate ties. The final audit first reproduced
98.1333% validation accuracy, then opened all 10,000 official test images.
No model or decoder tuning followed that audit.

The wide run reached its 98.10% parent baseline at update 2,000, so the validation-based early-stop rule did not fire and its full budget completed.
The original protocol and the validation-only amendment are both retained;
any incomplete run is explicitly identified rather than counted as completed.

## Verification and limits

- **117 Python tests and 51 Rust tests pass**, with zero local skips. They include
  independent local-surrogate directional derivatives, unchanged forward states,
  legacy plans, malformed contracts, checkpoint restoration, signed current
  history, codebook variants and both SDKs. The 38 example tests were rerun
  after versioning the current-readout bridge; actual preceding-version
  runtimes also reject the new tag as intended.
- Memory controls retain the selected weights: zero recurrence gives
  10.23%; clearing neural state before tick
  16 while preserving readout history gives
  17.50%.
- Stock XyloSim runs a Samna-validated configuration. SpiNNaker2 runs the Atlas C99
  host integer kernel through SDK graph containers, not stock Brian2Backend,
  PE firmware or a physical chip.
- Host preprocessing, fixed decoding and optional state recording remain part
  of the execution cost. No hardware placement, latency, power or energy result
  is established. Integer neuron limits and recurrent fanout are unchanged.

## Evidence and reproduction

[Selected fixtures](round4/fixtures/selection.json),
[full-test audit](round4/evidence/audit.json),
[paired comparison](round4/evidence/previous-round-comparison.json),
[regression receipt](round4/evidence/regression.json) and
[manifest](round4/manifest.json) retain model, source and output hashes.
Candidate training reports contain the exact commands, seeds, losses and
validation histories. Large per-tick traces remain in the local experiment
archive; the repository retains shard hashes and every test image's class scores.

Build the current native runners and use the two optional SDK environments as
in the [example setup](README.md). For an independent build, replay
the committed frozen weights without attempting to restore a checkpoint tied
to the original executable hash:

```sh
python verify.py --dataset /path/to/mnist.npz \
  --trained round4/fixtures --runner /path/to/b2-runner \
  --train-runner /path/to/b2-train --frozen-only \
  --xylo-python /path/to/xylo-env/bin/python \
  --spinnaker2-python /path/to/spinnaker2-env/bin/python \
  --output /path/to/new-round4-audit
```

The recorded audit additionally restores the original latent checkpoint and
checks a separate Rust 1.86 build against the pinned Rust 1.98.1 build. A new
build can reproduce frozen inference even when its executable hash prevents
restoring that exact optimizer checkpoint. Previous rounds remain unchanged.
