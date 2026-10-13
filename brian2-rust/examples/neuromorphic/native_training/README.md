# Native Atlas training → AtlasIR → SpiNNaker2

This example trains both layers of a small MNIST spiking network with Atlas's
native CPU engine, freezes and quantizes the learned weights, and runs inference
through the existing AtlasIR converter in the stock SpiNNaker2 software SDK.
TensorFlow and PyTorch do not participate in training or inference.

## Result

The complete official 10,000-image MNIST test split was evaluated:

| Execution | Correct | Accuracy |
|---|---:|---:|
| Atlas native LIF, learned floating-point weights | 9,453 | 94.53% |
| Atlas native LIF, explicitly quantized weights | 9,380 | 93.80% |
| SpiNNaker2 SDK, converted quantized model | 9,380 | 93.80% |

Quantization costs **0.73 percentage points**. Conversion adds **zero prediction
or classification-count differences** relative to the quantized model.

The full native-versus-SDK audit compares 11,840,000 binary spike-raster entries
(10,000 images × 16 ticks × 74 hidden/output neurons), 100,000 output counts and
740,000 reconstructed final membrane states. All match exactly. A separate
native `b2-runner` replay of concrete AtlasIR instances covers the first **64
test images**, comparing 75,776 voltage samples with the SDK. This bounded IR
replay is distinct from the full-test native-training-runner comparison.

See [the audit and confusion matrices](evidence/audit.json),
[SDK execution receipt](evidence/sdk-execution.json),
[training record](fixtures/training.json) and
[local regression record](evidence/validation.json).
The local regression selection passed 132 tests; five opt-in Metal tests were
skipped. The final bridge-only run passed all 12 tests, including actual SDK
execution with independent sample resets and different batch partitions.

## Model and training

The network is **196→64→10 LIF**, with subtract reset, beta=0.5 and threshold=1.
Each 28×28 image is mean-pooled over 2×2 blocks, thresholded at pixel value 80,
and repeated as a binary input spike raster for 16 ticks. Classification uses
output spike counts, with the usual first-index argmax tie rule.

Training uses the first 54,000 official training images. The remaining 6,000
select the checkpoint by quantized validation accuracy; test arrays are not
read by the training script. There are 1,800 Adam steps of 128 randomly drawn
samples, with both layers trainable. The native engine performs forward
execution, surrogate-gradient BPTT and optimizer updates. NumPy handles dataset
preparation, initialization, batching, quantization and metrics only.

The selected checkpoint is step 1,800. Validation accuracy improves from 13.08%
at initialization to 95.38% before quantization and 94.55% after quantization.
11,328 input-layer weights and all 640 readout weights changed from their initial
values. This is not readout-only training.

This is a new temporal SNN with a smaller binary input representation. Its
accuracy is not directly comparable to the earlier 98.74% TensorFlow integer
MLP, which has different architecture and preprocessing. MNIST has already
been used during this project's development; this is not a newly blinded
benchmark. Pilot settings were explored on validation images, not the test split.

## The bridge

```python
from brian2_atlas.neuromorphic import freeze_native_lif, export_native_lif

# trainer is an already-trained NativeLIFTrainer with an admitted dense v1 plan.
# These scales are for the published example; oversized weights are rejected.
frozen = freeze_native_lif(trainer, scales=[16, 16], quantize=True)
bundle = export_native_lif(
    frozen, "converted-native-lif", steps=16,
    target="spinnaker2", runner="/absolute/path/to/b2-runner",
)
```

`freeze_native_lif` is non-mutating. It records the training plan/state hashes,
runner identity and optimizer step, then explicitly rounds target weights to
nearest-even integers. It never clips. The new frozen weights are evaluated
in a fresh `NativeLIFTrainer` before SDK comparison. `quantize=False` rejects
weights that are not already representable.

`export_native_lif` constructs a Brian2 inference model, exports real AtlasIR,
and calls the existing `spinnaker2-lif-v1` converter and native validator. It
adds no IR schema or target profile. The saved source is a **zero-input
template**; subsequent sample identity is recorded separately by input hashes.
In an isolated SDK environment, `run_simulator(bundle, rasters, batch_size=100)`
executes independent replicas. The worker receives no labels or Atlas results.
Neural updates execute in the unmodified SDK backend.

The first bridge accepts dense v1 training plans with beta=0.5, subtract reset,
unmasked weights, binary input, power-of-two scales, at most 256 inputs and 256
total LIF neurons, and **1–16 inference ticks**. Scaled thresholds must retain
their float32 value; target weights must fit [-15,15]. Other training schemas,
extra dynamic behavior, nonzero initial state and live continuation are outside
this contract. A checkpoint supplies learned weights, not a target-side optimizer
or live training-state restore.

### Schedule and state mapping

Native training thresholds all populations before adding current-tick synaptic
input. The SDK consumes previous-tick spikes before thresholding. Copying the
same weight numbers into the SDK would change the network.

For one layer, let `J(t)` be its weighted current-tick spike input. Native state
and spikes follow:

```text
s(t) = [beta * v(t-1) > theta]
v(t) = beta * v(t-1) - theta * s(t) + J(t)
```

The exported SDK state is `z(t) = scale * (v(t) - J(t))`. It uses weights
`scale * beta * W`, threshold `scale * theta`, the same beta, and zero synaptic
delay under the SDK's documented schedule. Substituting this state relation
preserves spike ticks and count logits; no output time shift is applied.
Final native state is reconstructed using the same declared relation and checked
against the native runner.

The beta=0.5 and power-of-two scale restrictions avoid the threshold ambiguity
observed with non-dyadic quantization. A beta=0.875 prototype differed at exact
threshold ties by floating-point rounding; that prototype is not admitted by
this bridge. The new network also exposed scalar-delay broadcasting in the
existing converter: it previously truncated connection construction through
`zip`. Broadcasting now retains every connection, with regression tests for
scalar weights, scalar delays and invalid lengths.

## Reproduce

Use an Atlas source environment and native binaries built from this checkout.
Use the separate [SpiNNaker2 SDK environment](../spinnaker2/README.md) for SDK
execution. The published run used py-spinnaker2 0.8.0 and upstream Brian2 2.10.1.
The SDK uses Brian2 as its backend; it is not an independent silicon emulator.

```sh
export PYTHONPATH="$PWD:$PWD/brian2-rust/python"
export DEMO="$PWD/brian2-rust/examples/neuromorphic/native_training"
export DATASET=/absolute/path/to/mnist.npz
export B2_RUNNER=/absolute/path/to/b2-runner
export B2_TRAIN_RUNNER=/absolute/path/to/b2-train
export SDK_PYTHON=/absolute/path/to/spinnaker2-env/bin/python
export RUN=/absolute/path/to/new-output-directory

python "$DEMO/train.py" --dataset "$DATASET" --runner "$B2_TRAIN_RUNNER" \
  --output "$RUN/trained"
python "$DEMO/verify.py" --dataset "$DATASET" --trained "$RUN/trained" \
  --runner "$B2_RUNNER" --train-runner "$B2_TRAIN_RUNNER" \
  --sdk-python "$SDK_PYTHON" --output "$RUN/verified"
```

The official archive is available from
[TensorFlow's MNIST dataset mirror](https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz).
Both scripts check its SHA-256 against the published archive identity.
Output directories must be new. `--limit` on verification is for smoke runs and
must not be reported as full-test accuracy.

The selected checkpoint and frozen model are included in [fixtures](fixtures).
To replay without retraining, pass `--trained "$DEMO/fixtures"`. The native
checkpoint restore checks its training binary hash; this replay needs the same
binary identity. On another platform/build, rerun training first. Frozen-model
conversion itself does not need the original training binary.

```sh
export B2_SPINNAKER2_PYTHON="$SDK_PYTHON"
python -m pytest -q brian2-rust/tests/test_native_lif_conversion.py \
  brian2-rust/tests/test_spinnaker2_b2ir.py
```

Training and SDK wall times describe host software execution. Physical-chip
placement, hardware arithmetic, power, hardware latency and on-chip learning
have not been validated by this example.
