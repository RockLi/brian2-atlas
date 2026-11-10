# Original 98.74% MNIST MLP on SpiNNaker2

Verified 2026-10-13 through the public Atlas conversion API and the stock
`spinnaker2==0.8.0` Brian2 CPU backend. The unchanged, externally trained
784→256→10 integer model correctly classifies **9,874 / 10,000 images (98.74%)**.
All **2,660,000** hidden and output states match a fresh native Atlas execution
exactly, including when compared as float64 bit patterns. Predictions also
match the earlier Atlas/Akida/Lava results. No training was repeated.

The [audit](evidence/audit.json) verifies the original frozen weight-file hash,
every weight and effective bias, hidden shifts, official MNIST dataset hash,
and `pixel // 2` preprocessing. The SDK worker receives only the compiled
bundle and inputs; labels and references enter the later audit. This is the
existing quantized TensorFlow-trained model, not the separate 90.03% classifier
or an automatic importer for arbitrary TensorFlow graphs.

## What changed

The first feasibility experiment needed 127 comparators for every hidden
activation. Seven successive binary decisions now compute the same integer.
For `S = xW + bias + half_step`, decision k, from 6 down to 0, is:

```
b[k] = S - step * sum(b[j] * 2**j for j > k) > step * 2**k - 0.5
h = sum(b[k] * 2**k for k in range(7))
```

This equals `clip(floor(S / step), 0, 127)`: negative S emits no bits; S above
the upper limit emits all seven. The half-unit threshold implements an exact
integer comparison under the SDK's strict `>` rule. It is not fitted to MNIST.

Each layer uses stock `lif_32bit` neurons with `alpha_decay=0`, clearing state
before that tick's synaptic arrivals. Seven input bit planes retain values
0–127. A constant input spike supplies the hidden bias and rounding offset.
Delays align each sample with its earlier decisions, and align all seven bits
at the affine readout. One image enters each tick; output is read at the fixed
**eight-tick latency**. The SDK executes both weighted layers, activation and
reconstruction; the host only encodes inputs and reads output voltages.

| Resource | Initial threshold bank | Binary decisions |
|---|---:|---:|
| Hidden comparison neurons | 32,512 | 1,792 |
| First-layer connections | 130,590,544 | 7,197,904 |
| Total connections in the implemented graph | — | 7,224,378 |

First-layer connections are reduced by **94.49% (18.14×)**. The complete graph
uses 5,489 input neurons (including one constant) and 266 readout neurons.
The recorded SDK call took about **51.3 seconds**, including graph construction
and output collection. This is a local CPU runtime, not a chip performance
measurement. The initial large threshold graph was not executed as a timing
baseline, so no measured speedup is claimed.

## Admission and evidence

Use `target="spinnaker2", profile="dense-int8-v1"`. The native Atlas validator
and existing dense parser still validate the source. The new backend plan
preserves weights, bias, clipping and state order without changing `atlasir-v1`.

The encoding admits the dense profile's bounded signed integer arithmetic
only when every expanded synapse fits the SDK's 20-bit magnitude. This limits
hidden shifts to 1–13 and can additionally reject a large hidden bias. Delays
are 0–6 ticks. Readout absolute bounds remain below 2**24, making public
float32 voltage observations exact for the integer results. The profile is
static; its internal event timings do not preserve source SNN spike timing.

[SDK regression coverage](evidence/coverage.json) includes 22 admitted small
signed/biased fixtures and all 16,384 inputs of the original 2→3→2 probe:
83,904 states agree with Atlas. Hidden shifts 1–13, threshold boundaries,
clipping, signed readouts, silence and repeated inputs are covered. Three
fixtures with shifts 14–16 are explicitly rejected. Nonuniform hidden shifts
are admitted by the construction but were not covered by this fixture suite.
The [public compatibility check](evidence/compatibility.json) now admits 11
of 30 profile/target combinations and explicitly rejects 19.

The backend is the SDK's upstream Brian2 implementation. It is not an
independent silicon emulator, a placed hardware graph, or evidence of low
power. The expansion remains large; hardware placement and efficiency would
need separate work. Training and on-chip learning are unchanged.

## Reproduce

Use the Atlas Python environment for conversion and an isolated upstream
Python environment with `spinnaker2==0.8.0`, `brian2==2.10.1`, `numpy==2.3.4`
and `scipy==1.18.1` for SDK execution. Pin the SDK revision listed in the
[manifest](evidence/manifest.json). The native runner is supplied explicitly.
Set `DEMO` to this directory and `RUN` to a new output directory. `SOURCE` is
the original validated AtlasIR, `INPUTS` its full-test input NPZ, `TRAINED` the
original `gray-rounded256` checkpoint directory, `DATASET` official MNIST NPZ,
and `HISTORICAL` the earlier `test-results.npz`.

```sh
python "$DEMO/prepare.py" --source "$SOURCE" --inputs "$INPUTS" \
  --runner "$B2_RUNNER" --output "$RUN"

"$SDK_PYTHON" "$DEMO/../sdk_worker.py" \
  --bundle "$RUN/converted/bundle.json" --inputs "$RUN/inputs.npz" \
  --runtime "$PWD/brian2-rust/python/brian2_rust/neuromorphic_runtime.py" \
  --output "$RUN/sdk"

python "$DEMO/audit.py" --run "$RUN" --trained "$TRAINED" \
  --dataset "$DATASET" --historical "$HISTORICAL"
```

`prepare.py` explicitly opts into the original model's larger state, parameter,
statement and scratch budgets. Defaults are unchanged. It reuses the source
population size (500 here), preserves the definition hash across 20 input
batches, and retains each input-instance hash. Large artifacts stay in the
ignored output directory; compact evidence is checked into the repository.

For the small SDK regression suite, first prepare the existing
`../coverage` and `../dense_feasibility` fixtures, then:

```sh
python "$DEMO/prepare_coverage.py" --static-run "$STATIC_RUN" \
  --exhaustive-run "$EXHAUSTIVE_RUN" --runner "$B2_RUNNER" --output "$COVERAGE"
"$SDK_PYTHON" "$DEMO/run_coverage.py" --run "$COVERAGE" \
  --runtime "$PWD/brian2-rust/python/brian2_rust/neuromorphic_runtime.py"
```
