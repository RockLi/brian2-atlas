# Neuromorphic conversion compatibility

Verified 2026-10-13. This document describes the public
`brian2_atlas.neuromorphic` conversion API in this repository, not everything a
vendor chip or SDK can do. All execution evidence below is from software.

## Public profile matrix

**Verified** means an admitted subset has executed in the named SDK and has
been compared with native Atlas. **Rejected** means the API refuses the pairing.
It does not mean the vendor itself lacks that capability.

| Profile / source semantics | Akida | Lava | Specksim | SpiNNaker2 | XyloSim | DynapSim |
|---|---|---|---|---|---|---|
| `binary-count-v1`: frozen binary-input count classifier | Verified | Verified | Verified | Verified | Verified | Rejected |
| `dense-int8-v1`: signed weights, bias, quantized hidden activation, affine readout | Verified | Verified, uniform hidden shift | Rejected | Rejected | Rejected | Rejected |
| `spinnaker2-lif-v1`: explicit Euler LIF, recurrence, delayed events | Rejected | Rejected | Rejected | Verified | Rejected | Rejected |
| `xylo-audio2-temporal-v1`: bounded integer current/membrane dynamics | Rejected | Rejected | Rejected | Rejected | Verified | Rejected |
| `dynapse-dpi-v1`: bounded analog DPI circuit model | Rejected | Rejected | Rejected | Rejected | Rejected | Verified |

The executable [compatibility check](coverage/check_compatibility.py) tests all
30 cells: **10 admitted, 20 explicitly rejected**. It tests conversion routing;
separate SDK comparisons establish execution behavior.
[Machine-readable result](dynapse/evidence/compatibility.json).

## Functional checklist

| Function | Current status and limits |
|---|---|
| Validate the actual AtlasIR before lowering | Verified for all six targets; canonical source and native runner hashes retained |
| Same frozen model and weights across targets | Verified for the shared binary count profile; no retraining during conversion |
| Import arbitrary TensorFlow/Keras graphs | Not implemented; examples import an explicit frozen integer checkpoint through Brian2/AtlasIR |
| Preserve static hidden and output values | Exact for the static profiles in the tested domain |
| Preserve source spike timing | Verified for all three temporal profiles in their tested domains; static event encoding is not source spike semantics |
| Signed weights and bias | Akida/Lava dense profile; SpiNNaker2 and Xylo temporal profiles, with their respective bounds |
| Stateful recurrence | Temporal SpiNNaker2 and Xylo; not admitted by the static profiles |
| Explicit synaptic delays | SpiNNaker2 LIF: integer delays 0–7 ticks; Xylo temporal rejects explicit delays and uses its fixed tick schedule |
| Independent samples without state leakage | Static probes include repeated zero/one inputs; Specksim additionally matches at batch sizes 2 and 5 |
| Stateful chunked input | Xylo temporal: 7-step chunks match a full run; not verified for the other temporal SDK path |
| Atlas checkpoint/resume through conversion | Not implemented; temporal profiles require a fresh zero-state run |
| Analog-model adaptation and refractory dynamics | DynapSim pilot only; bounded circuit equations, no mismatch or physical configuration validation |
| Plasticity, STDP, on-chip learning, surrogate training through this API | Not implemented; external training is separate from conversion |
| Native temporal convolutional SNN lowering | Not implemented; Specksim uses a convolution to encode a static classifier |
| Official hardware configuration validation | Xylo temporal: all 50 tested configurations accepted by Samna |
| Physical chip execution / placement | Not verified by these suites |
| Chip power / latency / energy advantage | Not measured; CPU SDK wall time is not chip latency |

These limits concern the conversion API. They do not describe Atlas's separate
CPU/GPU simulation and training surfaces.

## Numeric and resource limits

- **Shared binary count:** input values 0/1; integer weights from 0 to a uniform
  threshold, which is a power of two from 2 to 64; zero effective bias; identity
  readout; interval proof that every hidden count is at most 127. Comparison is
  of final counts. Specksim also checks the final membrane remainder.
- **Akida/Lava dense:** inputs are integers 0–127; weights -127–127; hidden
  `clip(floor((affine + half_step)/step), 0, 127)` with shifts 1–16; biases must
  have an exact int8-times-power-of-two representation. Accumulator bounds are
  checked before conversion. Lava requires one uniform hidden shift and checks
  its expanded graph's int32 intermediates. This suite exercised every uniform
  shift; Akida's admitted nonuniform shift capability was not expanded here.
- **SpiNNaker2 temporal:** at most 256 input and 256 LIF neurons, 1–10,000 steps,
  1 ms clocks, zero initial voltage, signed integer weights -15–15, delays 0–7,
  supported subtractive or fixed-value reset. Only the documented Euler update
  and schedule are admitted. The SDK uses upstream Brian2; it is not an
  independent chip emulator. Voltage uses an absolute tolerance of 1e-12;
  spike ticks and neuron indices must match exactly.
- **Xylo temporal:** at most 16 inputs, 64 hidden neurons and 8 outputs in the
  current pilot, 1–10,000 steps; input counts 0–15; signed weights -128–127;
  signed 16-bit state/bias; synaptic shifts 0–15, membrane shifts 1–15;
  hidden/output spike caps 31/1. Exact integer trace comparison. The 64-hidden
  limit is this compiler profile's limit, not the silicon maximum.
- **Count encodings are not hardware fit claims:** SpiNNaker2 expands counts
  into comparator neurons. The static 784-input MNIST Xylo encoding exceeds
  Xylo Audio 2's 16 hardware input channels. Only its separate temporal profile
  uses validated hardware-sized configurations.

- **DynapSim:** 1–8 independent DPI neurons, diagonal nonnegative weights,
  binary raster, 1–512 steps at 1 ms; fixed synthetic circuit/layout parameters
  and enumerated bias/threshold/adaptation/refractory settings. No recurrence,
  inhibitory gates, device mismatch or static TF model conversion. The stock
  SDK is an approximate analog model, not a bit-accurate emulator; float traces
  use stated tolerances and pulse ticks must match. See the
  [exact contract and errors](dynapse/README.md).

The converter source and detailed examples define further AST, monitoring,
ordering and overflow requirements. Sampled tests do not prove every possible
combination of accepted parameters.

## Execution evidence

| Backend / suite | Models | Compared state values | Result |
|---|---:|---:|---|
| Akida: 20 count + 25 dense | 45 | 4,511 | Exact |
| Lava: 20 count + 25 dense | 45 | 4,511 | Exact |
| Specksim: count | 20 | 2,242 + 1,121 membrane residues | Exact; batch sizes 2 and 5 agree |
| SpiNNaker2: count | 20 | 2,242 | Exact; one all-zero model uses the proven zero fast path, 19 execute the SDK |
| XyloSim: count | 20 | 2,242 | Exact |
| SpiNNaker2: temporal LIF | 48 | 101,940 voltage samples | Max float64 error 1.14e-13; 35,093 spike events exactly match |
| XyloSim: temporal, preceding suite | 50 | 505,455 | Exact, including 7-step chunks |
| DynapSim: circuit dynamics | 16 | 36,985 | Float32/f64 tolerance checks; 510 spike events exactly match |

Static cases include exhaustive small binary domains, zero/full-scale/one-hot
inputs, repeated inputs, seeded random weights and samples, all six count
thresholds, count 127, zero kernels, all 16 dense shifts, signed odd coefficients,
bias, clipping and rounding boundaries. They are synthetic correctness cases,
not additional classification examples or a code-coverage percentage.

The expanded SpiNNaker2 suite includes both reset types, all delays 0–7,
recurrence, 256/256 dimensions, one-step and 10,000-step runs. SDK public float32
voltage values equal the Atlas trace cast to float32. Small float64 differences
remain within the stated tolerance; integer and pulse checks use no tolerance.

**SDK timestamp issue found and handled:** in the 10,000-step case,
`py-spinnaker2` public `get_spikes()` truncates floating timestamps, reporting
31 events one tick early. Atlas now recovers integer ticks by rounding the SDK's
aligned raw monitor times, rejecting off-clock timestamps. The audit preserves
the public SDK differences; no dynamics, delays or SDK source were changed.

**159 Python regression tests passed**, including timestamp regression tests,
semantic admission failures and existing backend checks.

- [DynapSim execution audit](dynapse/evidence/audit.json)
- [Static execution audit](coverage/evidence/static-audit.json)
- [Expanded SpiNNaker2 audit](coverage/evidence/spinnaker2-audit.json)
- [Xylo temporal evidence](xylo_temporal/evidence/expanded/audit.json)
- [Source identities and environment](coverage/evidence/manifest.json)
- [Reproduction commands](coverage/README.md)

The frozen TF MNIST model's earlier full-test result remains **90.03% on all
five count targets**, with exact conversion equivalence. The older **98.74%**
signed/biased MLP was checked only on Akida/Lava against historical Atlas output;
it is a different model. Neither accuracy experiment was rerun in this coverage
extension. See the [original TF results](README.md) and
[additional count-target results](count_targets/README.md).

## Acceptance checklist for a new capability

- [x] Validate admitted and rejected profile/target combinations.
- [x] Derive each SDK plan from the same validated source used by native Atlas.
- [x] Run real SDKs; keep source, result, implementation and SDK identities.
- [x] Test numerical boundaries, silence, repeated input and seeded random cases.
- [x] Compare full observable states; compare exact pulse timing where promised.
- [x] Record discovered limitations instead of shifting outputs or loosening tolerance.
- [ ] Add task-level accuracy evidence for temporal networks.
- [ ] Verify physical placement, execution and measured energy/latency.

The unchecked items are future work, not prerequisites already satisfied by
these software conversion tests.
