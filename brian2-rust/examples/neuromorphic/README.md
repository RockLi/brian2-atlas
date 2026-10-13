# Frozen TensorFlow models → AtlasIR → software simulators

This is an experimental **conversion and inference** surface in the Atlas
product repository. The static examples below use external TensorFlow training.
The separate [native training example](native_training/README.md) trains a temporal
LIF classifier in Atlas and exports it to SpiNNaker2. The
[character language-model example](language_model/README.md) uses the same
bridge for next-character prediction and per-token inference comparisons.
SDKs execute on CPU;
there is no chip execution, energy measurement, on-chip training, or automatic
conversion of arbitrary Keras/SavedModel graphs.

The example imports an explicit frozen integer checkpoint into a real Brian2
model, exports AtlasIR, validates it with the current native Atlas runner, and
compiles three backend-private plans. New exports use the `atlasir-v1` wire schema.

## Compatibility and expanded coverage

The [compatibility checklist](COMPATIBILITY.md) covers all six targets, their
numeric and temporal contracts, rejected combinations, SDK limitations and
hardware-validation gaps. The [expanded suite](coverage/README.md) varies
network weights and dimensions, tests clipping/rounding and sample resets,
and includes a 30-cell executable compatibility check. Static comparisons and
temporal pulse comparisons are reported separately.

## Verified result (2026-10-13)

The complete 10,000-image MNIST test split was freshly executed through the
frozen TF SavedModel, the new-repository Atlas runner and all three SDKs:

| Target | Correct | Accuracy | State differences from Atlas |
|---|---:|---:|---:|
| TensorFlow 2.16.2 frozen integer model | 9,003 | 90.03% | 0 class-count differences |
| Atlas Rust reference | 9,003 | 90.03% | reference |
| Akida 2.19.3 simulator | 9,003 | 90.03% | 0 / 200,000 |
| Lava 0.10.0 CPU fixed-point | 9,003 | 90.03% | 0 / 200,000 |
| Specksim (Sinabs 3.1.3 / Samna 0.48.6) | 9,003 | 90.03% | 0 / 200,000 |

Specksim also matched all 100,000 final membrane residues. This accuracy is the
small model's own accuracy, with **zero conversion loss**. The older 98.74% TF
MLP was separately reconverted and rerun in Akida/Lava, comparing with its
historical Atlas reference. It now also runs through the public SpiNNaker2
converter at **98.74%**, matching **2,660,000 states** from a fresh native Atlas
replay. See [the full-model result](spinnaker2_dense/README.md). It remains
outside the Specksim profile.
See [the compact audit](evidence/common-audit.json) for identities and confusion
matrices. Large source/run artifacts are kept under ignored `output/neuromorphic`.

## API and support

```python
from brian2_atlas.neuromorphic import export_ir

bundle = export_ir(
    "source.json", "converted-specksim", target="specksim",
    runner="/absolute/path/to/b2-runner", profile="binary-count-v1",
)
```

`convert_ir` returns a bundle without writing files. `export_ir` writes the
canonical source and bundle, refusing to overwrite existing destinations.
`run_simulator(bundle, inputs)` optionally executes in a compatible SDK Python
environment. SDKs are not imported during conversion. The standalone worker
loads the same runtime source in isolated Python 3.10/3.11 environments, because
those vendor versions do not share Atlas's Python/NumPy requirements.

| Profile | Akida | Lava | Specksim |
|---|---|---|---|
| `dense-int8-v1` | Quantized signed/biased static MLP | Same through stock fixed-point processes | Explicitly unsupported |
| `binary-count-v1` | Shared count classifier | Shared count classifier | Stock convolution + IAF count classifier |

The shared profile admits binary inputs, nonnegative integer weights no greater
than the power-of-two threshold (2…64), no effective bias, an identity readout,
and an algebraic proof that clipping never activates over **any** binary input.
The source instance and every inference input are checked against that binary
domain. Fractional/out-of-range inputs, signed weights, active clipping, biased
or nonidentity readouts are rejected instead of approximated.

For a zero-state subtract-reset IAF receiving serialized nonnegative increments
`w <= threshold`, each input event produces at most one spike. Consequently:

```
final_count = floor(sum(input * weight) / threshold)
final_voltage = sum(input * weight) % threshold
```

This is final-value equivalence to a **static** source model. Specksim's internal
event times are an encoding, not the source model's spike times. It is not a
new demonstration of temporal SNN portability. Akida's implicit rounding is
cancelled with a compiler-derived half-step bias; no trained bias is discarded.

The original 98.74% 784→256→10 TF classifier remains a distinct model. Its
Akida/Lava conversion can be checked with `legacy_preflight.py`;
`spinnaker2_dense/prepare.py` exercises the new SpiNNaker2 dense path; the shared Specksim profile rejects it. The new common
TF model is a small 784→10 quantized count classifier, not a renamed version of
that old model. Quantization can change the float model's predictions: the
frozen **integer TF SavedModel** is the equivalence reference.

## Reproduce

From the repository root, set these paths explicitly:

```sh
export PYTHONPATH="$PWD/brian2-rust/python"
export B2_RUNNER="/absolute/path/to/current/b2-runner"
export DATASET="/absolute/path/to/mnist.npz"
export SDK_ENV="/absolute/path/on/data-volume/atlas-sdk-env"
export DEMO="$PWD/brian2-rust/examples/neuromorphic"
export RUN="$PWD/output/neuromorphic"
mkdir -p "$RUN" "$SDK_ENV"
```

Build the runner from this checkout (`cargo build --locked --release --bin
b2-runner` in `brian2-rust`; use `CARGO_TARGET_DIR` on a data volume if needed).
The examples explicitly opt into larger population/statement budgets. Default
limits remain 32 mutable states, 128 parameters and 128 statements. The native
scratch budget is also opt-in (`B2_MAX_SCRATCH_VALUES`), for the larger old MLP.

Build a small runtime and install large SDK dependencies on a data volume:

```sh
docker build -f "$DEMO/Dockerfile.runtime" -t atlas-neuro-runtime:20261013 "$DEMO"
docker run --rm -v "$SDK_ENV:/env" -v "$DEMO:/code:ro" \
  --entrypoint sh atlas-neuro-runtime:20261013 -c \
  'python -m venv /env && mkdir -p /env/tmp && TMPDIR=/env/tmp /env/bin/pip install --no-cache-dir -r /code/requirements-sdk.txt && TMPDIR=/env/tmp /env/bin/pip install --no-cache-dir --force-reinstall --no-deps samna==0.48.6 --index-url https://gitlab.com/api/v4/projects/27423070/packages/pypi/simple'
```

Akida uses a separate image with `akida==2.19.3`, `numpy==2.2.6` and Python 3.11
(`Dockerfile.akida`). Vendor SDK licence terms still apply.

Train using only the first 54,000 official training images and select integer
weights using the remaining 6,000. Binary preprocessing is `pixel >= 128`.
Training never reads test arrays; the project's prior MNIST experiments mean
this is not a newly blinded benchmark.

```sh
docker run --rm --network none --cpus 2 --memory 3g \
  -v "$SDK_ENV:/env:ro" -v "$DEMO:/code:ro" \
  -v "$DATASET:/mnist.npz:ro" -v "$RUN:/work" \
  atlas-neuro-runtime:20261013 /code/train_tf.py \
  --dataset /mnist.npz --output /work/trained

python "$DEMO/prepare.py" --dataset "$DATASET" --trained "$RUN/trained" \
  --runner "$B2_RUNNER" --output "$RUN/common"
```

`prepare.py` runs all 10,000 test images afresh through Atlas, in bounded batches,
and derives all three bundles from the same canonical IR. For a smoke run use
`--limit 100 --batch-size 100`; do not report that as full-test accuracy.

Run the actual frozen TF SavedModel:

```sh
docker run --rm --network none --cpus 2 --memory 3g \
  -v "$SDK_ENV:/env:ro" -v "$DEMO:/code:ro" -v "$RUN:/work" \
  atlas-neuro-runtime:20261013 /code/tf_infer.py \
  --model /work/trained/saved_model --inputs /work/common/inputs.npz --output /work/common/tf
```

Execute each target. For `lava` and `specksim`, use:

```sh
export TARGET=specksim # repeat with lava

docker run --rm --network none --cpus 2 --memory 3g \
  -v "$SDK_ENV:/env:ro" -v "$DEMO:/code:ro" \
  -v "$PWD/brian2-rust/python/brian2_rust:/api:ro" -v "$RUN:/work" \
  atlas-neuro-runtime:20261013 /code/sdk_worker.py \
  --bundle "/work/common/$TARGET/bundle.json" --inputs /work/common/inputs.npz \
  --runtime /api/neuromorphic_runtime.py --output "/work/common/$TARGET-run"
```

For `akida`, use image `atlas-akida-minimal:2.19.3`, add `--entrypoint python`,
and omit the SDK_ENV mount; the remaining mounts/arguments are identical.
The worker never reads labels or reference predictions. Host processing is
input encoding, output transport/counting, and observation; no neural layer is
computed on the host to fill a missing simulator operation.

```sh
python "$DEMO/audit.py" --run "$RUN/common" --trained "$RUN/trained"
```

Audit requires exact TF/Atlas/SDK state and prediction agreement, verifies
artifact hashes, and also compares Specksim final membrane residues. It reports
classification accuracy, confusion matrices, versions and source identities.
SDK wall time includes compilation, polling and CPU execution; it is not chip
latency. All example output directories refuse overwrites.

The verified frozen checkpoint and TF SavedModel are included in
`fixtures/tf-count/`. To replay without retraining, copy that directory to
`$RUN/trained` before the preparation step. Its hashes match the evidence.

## Boundary checks

`prepare_probes.py --run "$RUN/common" --runner "$B2_RUNNER"` creates 14
all-zero, all-one, single-input, alternating and seeded-random cases with fresh
Atlas output. Run `sdk_worker.py` against `common/probes/inputs.npz`, keeping the
same common target bundles and using `--batch-size 2` for Specksim to exercise
state resets. Run `tf_infer.py` on the same inputs. Compare each full state
matrix to `probes/atlas.npz`, and Specksim voltage to the integer remainder.
All 14 cases passed on the three SDKs and the frozen TF model; see
[evidence/boundary-audit.json](evidence/boundary-audit.json). These are boundary
cases, not additional classification samples.

For external drives, use a filesystem suitable for many small files. The
verified environment uses an APFS sparse bundle on a T7 data disk; SDK packages
are outside the repository and are not part of the wheel or source package.

## Temporal SpiNNaker2 pilot

The separate [SpiNNaker2 LIF example](spinnaker2/README.md) converts a native
time-dependent AtlasIR model to the official SDK Brian2 software backend.
It checks membrane traces and spike timesteps, rather than reusing the static
TF classifier profile. Its scope and evidence are documented separately.


## Additional count targets

The same frozen 90.03% model and source IR now also run in the official
SpiNNaker2 Brian2 backend and XyloSim v2 C++ core. Both matched the freshly
rerun Atlas results over all 10,000 MNIST test images and the boundary inputs.
See [the conversion details and reproduction steps](count_targets/README.md).
These use `binary-count-v1`; neither admits the larger signed/biased MLP.
SpiNNaker2 expands the count into comparator neurons. XyloSim uses recorded
hidden spikes and exceeds Xylo Audio 2's hardware input-channel limit, which
is explicitly recorded in its plan. Both are software-only conversion targets.


## Temporal Xylo Audio 2 profile

The [Xylo temporal example](xylo_temporal/README.md) converts small integer
current/membrane networks, including recurrent spike multiplicity, to official
hardware-sized configurations. The expanded 50-case suite passed the vendor
validator and matched Atlas at every recorded tick, including 7-step chunks. This is separate from the
MNIST count encoding above: it uses the standard output neurons and admits at
most 16 input channels. Physical chip execution remains untested.

## DYNAP-SE2 analog-model pilot

The [DynapSim example](dynapse/README.md) adds `target="dynapse"` with the
separate `dynapse-dpi-v1` profile. Sixteen bounded circuit-model cases matched
stock Rockpool/JAX within declared float tolerances, with exact spike timing.
This path includes adaptation and refractory dynamics, but does not accept the
static TF classifier or arbitrary LIF networks and does not validate a physical
hardware configuration.
