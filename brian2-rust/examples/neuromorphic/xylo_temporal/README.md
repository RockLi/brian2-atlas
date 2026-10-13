# AtlasIR → hardware-sized temporal Xylo Audio 2 networks

The example defines a small integer spiking network in Brian2, exports AtlasIR,
runs the native Atlas Rust executor and lowers the same IR to an official
SynSense Xylo Audio 2 configuration. Rockpool/Samna validate that configuration;
XyloSim v2 then executes **from the validated configuration**.

## Expanded coverage — 2026-10-13

The expanded suite contains **50 networks**: the original 13 cases, 32 seeded
random networks and five size, duration or chunk-boundary cases. All 50 pass
Samna's official configuration validator.

- **505,455 current, membrane and spike-count values** match native Atlas exactly.
- **1,311,454 hidden spikes** and **31,352 output spikes**, counting multiplicity,
  agree; the native spike event pairs also match.
- Every network also runs in chunks of **7 steps** on one persistent SDK model.
  All 505,455 values match the full SDK run. A dedicated case fires immediately
  before a chunk boundary, delivers to the output after the boundary and retains
  recurrent activity through later silent-input chunks.
- Dimensions reach **16 inputs / 64 hidden / 8 outputs**, the current profile's
  limit. Durations cover **1 through 10,000 steps** at selected lengths.
- Random inputs exercise counts 0–15, all synaptic shifts 0–15 and membrane
  shifts 1–15, weight endpoints -128/127, and signed bias endpoints.
- **8 cases** independently match Brian2 NumPy states and spike events, including
  random networks, the largest dimensions, one neuron and one tick.
- **130 Python tests** pass across the Xylo and existing conversion regressions.

The random seed is `20261013`. The audit lists the parameters actually covered;
these are sampled semantic cases, not an exhaustive proof or a code-coverage
percentage. This change does not expand the admitted neuron semantics, the IR
schema or the native executor. Chunking tests SDK state preservation within one
run; it does not add Atlas checkpoint/resume support.

See the expanded [audit](evidence/expanded/audit.json),
[Brian2 comparison](evidence/expanded/brian2-audit.json) and
[source identities](evidence/expanded/manifest.json). The original evidence below
is retained unchanged.

## Original verification — 2026-10-13

- **13 networks** passed the official configuration validator.
- **38,016** recorded current, membrane and spike-count values agreed exactly
  at the same ticks, with no output-time shift or numerical tolerance.
- **48,314 hidden spikes** and **2,865 output spikes**, counting multiplicity,
  agreed. The native Atlas spike event `(tick, neuron)` pairs also agree with
  the nonzero SDK spike-count raster.
- Cases range from 2 inputs / 3 hidden / 2 output neurons to
  **16 inputs / 16 hidden / 8 output neurons**, for 64–1,024 time steps.
- Saturation, signed minimum decay and the hidden 31-spike / output 1-spike
  per-tick limits were exercised, not merely admitted.
- The Brian2 NumPy executor independently matched Atlas for the recurrent and
  mixed-saturation cases, checking the source event-delivery order as well.
- **124 Python tests** passed, including the existing conversion regressions.

See [the audit](evidence/audit.json), [source and SDK identities](evidence/manifest.json)
and the [saved hardware configurations](evidence/configurations/).

The input/output dimensions fit the target, and the official configuration
validator accepts the result. No chip was connected, so this does not measure
hardware timing, power or deployment success. It is a conversion/dynamics test,
with no training or classification accuracy result.

## Source semantics and admitted subset

Use `target="xylosim", profile="xylo-audio2-temporal-v1"` with
`brian2_atlas.neuromorphic.export_ir`. The static MNIST profile remains separate.

The three real Brian2 populations are an input raster source, a hidden population
and an output population. The raster assigns each input's integer spike count
`q` at the start of the tick. Source events fire when `q > 0`. Hidden and output
neurons have synaptic current `I`, membrane voltage `v` and count `q`.

Brian2 emits one event per firing neuron per step. Multiplicity is explicit:
each synapse delivers `w*q_pre`, and `q` records up to 31 hidden spikes or one
output spike. The comparison checks **both multiplicity and event presence**.
This uses existing states, events and AST operations in `b2ir-v1`; it introduces
no new wire-format opcode or change to the Rust executor.

The tick order is:

1. Set input spike counts and generate their events.
2. Deliver input, previous-tick hidden recurrence, then previous-tick readout
   events. Clip each connection's current addition to signed 16 bits.
3. Apply integer synaptic decay, membrane decay, saturating integration and bias.
4. Compute counts, generate events, subtract `count*threshold`, then record state.

For an integer value `x` and divisor `D = 2**dash`, Xylo's signed decay is:

```
decay(x,D) = floor(x/D) + int(0 < x < D)
```

This includes the minimum positive decay of one. Negative values use arithmetic
right-shift behavior. All stored values are exact integers represented in
Atlas's `reference-f64` storage. The converter matches the update, threshold,
reset and synapse ASTs literally, including all saturation stages and ordering.
It does not infer neuron semantics from names.

The admitted pilot has:

- One 1 ms clock, zero initial state, 1–10,000 steps.
- 1–16 input channels, counts 0–15 per channel per tick.
- 1–64 hidden neurons and 1–8 output neurons. The hidden limit is intentionally
  below the chip's 1,000-neuron limit.
- Signed 8-bit integer weights, no weight shift or quantization.
- One current channel; synaptic shifts 0–15 and membrane shifts **1–15**.
- Positive signed 16-bit thresholds and signed 16-bit biases.
- Full end-of-step `I`, `v`, `q` recording and spike event recording.
- Static input and readout projections, with optional recurrence; sorted unique
  connections and fixed delivery order to preserve noncommutative saturation.
- No aliases, second current channel, explicit delays, refractory behavior,
  plasticity, continuation, initial pending events or adaptive integration.

Membrane shift zero is excluded: the stock C++ implementation negates a decay
value into an `int16_t` temporary, whose overflow at the most negative state
would require a separate wraparound contract. This profile does not approximate
that case. Arbitrary floating LIF equations are likewise rejected.

## Cases

The suite includes recurrence, silence, an impulse at tick zero, an event at the
last tick, threshold equality, positive/negative minimum decay, multi-spike
limits, positive saturation, negative saturation, mixed-sign saturation whose
result depends on delivery order, bias-only activity, a seeded 1,024-step run,
and the full input/output dimensions.

The last-tick case produces hidden activity without prematurely delivering it
to the output. The mixed-saturation case delivers negative and positive
recurrent currents in order after saturated input. Tests also reject reordered
connections, wrong schedules, fractional parameters, out-of-range weights,
excess input channels, wrong decay/reset ASTs and stale source hashes.

## Reproduce

Use Python 3.12 in a separate SDK environment. Install the actual Samna wheel
from SynSense's package index, followed by the simulation/configuration packages:

```sh
python3.12 -m venv /absolute/path/to/xylo-env
/absolute/path/to/xylo-env/bin/pip install --no-cache-dir samna==0.48.6 \
  --index-url https://gitlab.com/api/v4/projects/27423070/packages/pypi/simple
/absolute/path/to/xylo-env/bin/pip install --no-cache-dir \
  rockpool==3.1.0 xylosim==0.1.3 numpy==2.5.4 scipy==1.18.1 \
  bitstruct==8.23.0 packaging==26.3
/absolute/path/to/xylo-env/bin/pip check
```

XyloSim builds a small C++ extension. The tested system is macOS arm64 with a
C++ compiler installed. The SDK sources and compiled extension were not patched.
No hardware discovery or connection call is made.

From the Atlas checkout, with its Python dependencies and native runner built:

```sh
export PYTHONPATH="$PWD:$PWD/brian2-rust/python"
export B2_RUNNER=/absolute/path/to/b2-runner
python brian2-rust/examples/neuromorphic/xylo_temporal/compare.py \
  --runner "$B2_RUNNER" \
  --sdk-python /absolute/path/to/xylo-env/bin/python \
  --output output/neuromorphic/xylo-temporal

python brian2-rust/examples/neuromorphic/xylo_temporal/verify_brian2.py \
  --run output/neuromorphic/xylo-temporal

python -m pytest -q \
  brian2-rust/tests/test_xylo_b2ir.py \
  brian2-rust/tests/test_spinnaker2_b2ir.py \
  brian2-rust/tests/test_neuromorphic.py \
  brian2-rust/tests/test_akida_b2ir.py \
  brian2-rust/tests/test_lava_dense_b2ir.py
```

To reproduce the expanded suite and its independent Brian2 check:

```sh
python brian2-rust/examples/neuromorphic/xylo_temporal/compare.py \
  --suite coverage --chunk-steps 7 \
  --runner "$B2_RUNNER" \
  --sdk-python /absolute/path/to/xylo-env/bin/python \
  --output output/neuromorphic/xylo-coverage
python brian2-rust/examples/neuromorphic/xylo_temporal/verify_brian2.py \
  --coverage --run output/neuromorphic/xylo-coverage
```

Choose a new output directory. Each case keeps its source IR, converted bundle,
native Atlas results, SDK output and serialized official configuration. The
separate SDK worker gets bundles only; comparison with Atlas happens afterwards.
The audit records identities, versions and per-case counts. With `--chunk-steps`,
it also records the chunked output hash and comparison count; the raw
`sdk-chunked.npz` is saved alongside each full-run result.

References: [official Xylo limits and simulator workflow](https://rockpool.ai/devices/quick-xylo/deploy_to_xylo.html),
[official configuration API](https://rockpool.ai/reference/_autosummary/devices.xylo.syns61201.config_from_specification.html),
[XyloSim 0.1.3 C++ source distribution](https://pypi.org/project/xylosim/0.1.3/).
