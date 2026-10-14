# Neuromorphic conversion compatibility

Updated 2026-10-14; individual evidence records retain their run dates. This document describes the public
`brian2_atlas.neuromorphic` conversion API in this repository, not everything a
vendor chip or SDK can do. All execution evidence below is from software.

## Public profile matrix

**Verified** means an admitted subset has executed in the named SDK and has
been compared with native Atlas. **Custom C** names our SDK host extension,
not a vendor neuron model or firmware implementation. **Rejected** means the API refuses the pairing.
It does not mean the vendor itself lacks that capability.

| Profile / source semantics | Akida | Lava | Specksim | SpiNNaker2 | XyloSim | DynapSim |
|---|---|---|---|---|---|---|
| `binary-count-v1`: frozen binary-input count classifier | Verified | Verified | Verified | Verified | Verified | Rejected |
| `dense-int8-v1`: signed weights, bias, quantized hidden activation, affine readout | Verified | Verified, uniform hidden shift | Rejected | Verified, encoded weight bounds | Rejected | Rejected |
| `spinnaker2-lif-v1`: explicit Euler LIF, recurrence, delayed events | Rejected | Rejected | Rejected | Verified | Rejected | Rejected |
| `xylo-audio2-temporal-v1`: bounded integer current/membrane dynamics | Rejected | Rejected | Rejected | Rejected | Verified | Rejected |
| `dynapse-dpi-v1`: bounded analog DPI circuit model | Rejected | Rejected | Rejected | Rejected | Rejected | Verified |
| `spinnaker2-xylo-integer-v1`: bounded Xylo integer dynamics, custom host implementation | Rejected | Rejected | Rejected | Custom C, exact | Rejected | Rejected |

The executable [compatibility check](coverage/check_compatibility.py) tests all
36 cells: **12 admitted, 24 explicitly rejected**. It tests conversion routing;
separate SDK comparisons establish execution behavior.
[Machine-readable result](native_xylo_recurrent/evidence/spinnaker2_integer/compatibility.json).
The preceding [30-cell result](spinnaker2_dense/evidence/compatibility.json) is retained as historical evidence.

### Rejected does not mean impossible

The original 98.74% signed/biased MNIST MLP now runs through the public
converter on **Akida, Lava and SpiNNaker2**. The new SpiNNaker2 run preserves
the same frozen weights and matches a fresh native Atlas replay on all
10,000 test images and 2,660,000 states. Its seven-bit decision encoding
reduces first-layer connections from 130.59 million to 7.20 million.
See the [full-model audit and reproduction](spinnaker2_dense/README.md).

Specksim's direct path rejects bias and can change counts with signed event
order. Xylo Audio 2's direct mapping exceeds input-channel and accumulator
limits. DynapSim has no established exact integer-MLP mapping. These are
specific current obstacles, not universal impossibility claims.
See [the experiments, original-model bounds and next steps](dense_feasibility/README.md).


### Native Atlas training source

The [native LIF example](native_training/README.md) adds a source bridge into
`spinnaker2-lif-v1`; it does not add a seventh target or another profile-matrix
row. Both layers are trained in `b2-train`. Freezing explicitly quantizes the
weights, and conversion uses a declared state transformation to reconcile
native training and SDK scheduling. The full MNIST test audit reports float,
quantized-native and SDK accuracy separately.

The [character language-model example](language_model/README.md) exercises
the same bridge with a fixed four-character input context and a 32-character
output distribution. It measures held-out perplexity and per-token parity.
Each prediction starts from zero LIF state; autoregressive context updates
happen on the host. It does not add EGRU, Transformer or persistent inter-token
neural-state support.

The bridge accepts dense v1 training plans, beta=0.5, subtract reset, binary
inputs, power-of-two scales, up to 256 inputs and 256 total LIF neurons, and
1–16 inference ticks. Scaled weights must fit [-15,15] without clipping.
Simulator batching replicates independent admitted networks; it does not claim
that a batched graph fits physical hardware. Other training schemas, masks,
live state continuation and on-chip learning are rejected or not implemented.

### Native weights on Xylo: approximate adaptation

The [same-weight comparison](native_xylo/README.md) exports one native-trained
16→32→4 model to both SpiNNaker2 and XyloSim. Xylo requires explicit
`allow_dynamics_approximation=True`: topology and integer weight banks are
preserved, but integer current/membrane dynamics replace the native float LIF
transition. This is not a newly admitted cell in the exact profile matrix.

On all 4,157 test digits 0–3, native quantized and SpiNNaker2 accuracy is 75.58%,
while Xylo reaches 75.61%. There are 234 prediction differences and 217,243
per-tick spike-count differences. Accuracy similarity does not establish
cross-target equivalence. The adapted AtlasIR matches all 55,296 recorded Xylo
state values on a 32-image replay; the configuration passes Samna validation.
The bridge rejects models beyond 16 inputs, 64 hidden neurons and 8 outputs.

### Native Xylo integer training: exact source bridge

[Native integer training](native_xylo_integer/README.md) adds a CPU training plan
with Xylo current/membrane dynamics in every forward pass, quantization-aware
weights and surrogate BPTT. `freeze_native_xylo` / `export_native_xylo` join the
existing temporal profile; it does not itself add another profile.
No dynamics approximation is required. This model does not map to the existing
SpiNNaker2 LIF profile.

On all 4,157 test digits 0–3, a 16→64→4, 32-tick network reaches **92.18%** in
native Atlas and XyloSim, with zero differences across 9,045,632 values of each
of spikes, currents and membranes. A 32-image AtlasIR replay checks 208,896
values exactly. A binary-input 16→32→4, 16-tick control reaches **77.63%**.
Both checkpoints are selected on validation only. Encoding, model size and
training differ from the previous 75.61% adaptation; this is not a same-model
accuracy improvement or a ten-class MNIST result.

### Recurrent integer training and chunked Xylo inference

The [v2 recurrent source bridge](native_xylo_recurrent/README.md) trains input,
readout and recurrent weight banks, with a delayed readout window. Native and
stock SDK execution both reach 100% on 1,872 held-out synthetic AB/BA sequences;
zeroing recurrence or resetting state at query gives 50%. All 5,840,640
spike/current/membrane values match. Both executors preserve full-run results
with 7- and 13-tick chunks; SDK sessions also match irregular boundaries on every
test sequence. Concrete AtlasIR replay checks another 99,840 values exactly.

This is evidence of learned memory in a bounded order task, not a real-world
benchmark or proof of long-duration generalization. Training remains fresh
full-sequence BPTT; only inference carries state, for at most 128 total ticks.

A [same-weight SpiNNaker2 diagnostic](native_xylo_recurrent/SPINNAKER2.md) of this
model executes stock `lif_curr_exp_32bit`, but reaches only 936/1,872 (50%).
Raw-clock comparison finds 777,187 spike-count, 1,755,891 current and 1,727,071
membrane differences; applying the SDK clock shift does not restore parity or
accuracy. The exact converter rejects the source. This is a documented transfer
failure and does not add an admitted cell to the profile matrix above.

The subsequent [custom SpiNNaker2 C host kernel](native_xylo_recurrent/SPINNAKER2_INTEGER.md)
is an explicit new profile. It preserves the same recurrent integer model and
weights: all 1,872 sequences are correct, with zero differences across 5,840,640
neural-state values and all logits. Full runs and 1/7/13-tick chunks agree.
A fresh 32-sequence AtlasIR replay matches 99,840 state values and events.
The SDK graph feeds an independent compiled C kernel; this is neither stock
`Brian2Backend` execution nor a chip firmware implementation.

### Full ten-class block-sequential MNIST

The [native sequential example](sequential_mnist/README.md) uses the same
16→63→5 recurrent integer model and weights in native Atlas, stock XyloSim
and the custom SpiNNaker2 C host profile. All three reach **97.44%** on the
complete 10,000-image official test set (previous wider result: 97.47%). Each
target matches 73,440,000 spike/current/membrane values and 100,000 logits exactly.
The 98% test target remains unmet. The selected latent checkpoint, frozen model, independent native build
and ten concrete AtlasIR replays agree. [The fourth-round report](sequential_mnist/ROUND4.md)
records validation selection, training/readout candidates and controls. Optional
current readout uses host state recording; the selected readout is `spikes`.

Recurrent v2 training/export admits 16/128/8 dimensions, count inputs up to 15,
and a fixed host class codebook. Recurrent fanout remains ≤63. The retained
round-three wider example uses independent groups and projects cross-group weights and Adam moments to
zero. V1 feedforward retains 16/64/8. No integer neuron transition changed.
The selected input is `rows-full-counts` with host deskew and 36 ticks;
this is not the canonical 784-step sequential benchmark. Earlier test scores
were known before this development round. No physical mapping, power or chip
execution is established by these software checks.

## Functional checklist

| Function | Current status and limits |
|---|---|
| Validate the actual AtlasIR before lowering | Verified for all six targets; canonical source and native runner hashes retained |
| Same frozen model and weights across targets | Verified for the shared binary count profile; no retraining during conversion |
| Import arbitrary TensorFlow/Keras graphs | Not implemented; examples import an explicit frozen integer checkpoint through Brian2/AtlasIR |
| Preserve static hidden and output values | Exact for the static profiles in the tested domain |
| Preserve source spike timing | Verified for all three temporal profiles in their tested domains; static event encoding is not source spike semantics |
| Signed weights and bias | Akida/Lava/SpiNNaker2 dense profile; SpiNNaker2 and Xylo temporal profiles, with their respective bounds |
| Stateful recurrence | Temporal SpiNNaker2 and Xylo under their own dynamics; the custom SpiNNaker2 integer host profile also preserves Xylo recurrence; stock current-LIF substitution does not |
| Explicit synaptic delays | SpiNNaker2 LIF: integer delays 0–7 ticks; Xylo temporal rejects explicit delays and uses its fixed tick schedule |
| Independent samples without state leakage | Static probes include repeated zero/one inputs; Specksim additionally matches at batch sizes 2 and 5 |
| Stateful chunked input | Xylo temporal and native recurrent v2: full runs match fixed and irregular chunks; custom SpiNNaker2 integer host execution matches 1/7/13-tick chunks; other temporal SDK paths are unverified |
| Fixed class-code readout | Native recurrent integer training and exact Xylo/custom SpiNNaker2 bridges; 2–32 classes decoded on the host from at most 8 physical outputs |
| Signed output-current readout | Optional native recurrent v2 objective and exact Xylo/custom SpiNNaker2 host decoding; sums recorded output Isyn, preserves signed chunk history and checks output-threshold normalization; requires state recording, not on-chip classification |
| Output-rate auxiliary objective | Optional native recurrent v2 training loss with fixed class codes; exported spike-count inference is unchanged; independent surrogate derivatives verified |
| Native trained weights → inference | Exact bounded LIF bridge to SpiNNaker2 and approximate LIF adaptation to Xylo (1–16 ticks); separate exact integer-trained Xylo and custom SpiNNaker2 C host bridges (1–128 ticks). All start fresh sequences; recurrent Xylo v2 carries inference state between chunks |
| Atlas live checkpoint/resume through conversion | Not implemented; native bridge exports weights for fresh inference, not optimizer/continuation state |
| Analog-model adaptation and refractory dynamics | DynapSim pilot only; bounded circuit equations, no mismatch or physical configuration validation |
| Plasticity, STDP, on-chip learning, surrogate training through this API | Not implemented; native or external training occurs before freezing and conversion |
| Native temporal convolutional SNN lowering | Not implemented; Specksim uses a convolution to encode a static classifier |
| Official hardware configuration validation | Xylo temporal: original 50-configuration suite accepted by Samna; wider boundary and selected MNIST configurations also validated |
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
- **SpiNNaker2 dense:** seven binary decision stages using stock `lif_32bit`;
  integer inputs 0–127, signed weights and exact biases. Every encoded weight
  must fit a 20-bit magnitude, limiting hidden shifts to 1–13 and rejecting
  oversized encoded bias. Fixed eight-tick output latency, delays 0–6, no
  state carryover between images. This is a static arithmetic encoding.
- **SpiNNaker2 temporal:** at most 256 input and 256 LIF neurons, 1–10,000 steps,
  1 ms clocks, zero initial voltage, signed integer weights -15–15, delays 0–7,
  supported subtractive or fixed-value reset. Only the documented Euler update
  and schedule are admitted. The SDK uses upstream Brian2; it is not an
  independent chip emulator. Voltage uses an absolute tolerance of 1e-12;
  spike ticks and neuron indices must match exactly.
- **Xylo temporal:** at most 16 inputs, 128 hidden neurons and 8 outputs in the
  current pilot, 1–10,000 steps; input counts 0–15; signed weights -128–127;
  signed 16-bit state/bias; synaptic shifts 0–15, membrane shifts 1–15;
  hidden/output spike caps 31/1; recurrent fanout at most 63 per hidden neuron. Exact integer trace comparison. The 128-hidden
  limit is this compiler profile's limit, not the silicon maximum.
- **Native integer Xylo training/export:** the temporal profile restricted to
  inputs up to an explicit count limit of 15, one hidden layer, zero bias and
  shifts one. V1 is feedforward with 16/64/8 bounds; v2 admits 16/128/8,
  hidden recurrence (fanout ≤63), a fixed host class codebook, a readout window
  and explicit evaluation-state carry. The entire sequence, including chunks,
  is 1–128 ticks. Forward uses int8 nearest-even weights and int16 state;
  backward uses surrogate gradients and rounding STE. CPU only; training is
  fresh full-sequence BPTT, without TBPTT or general Brian-plan lowering.
- **Custom SpiNNaker2 integer host profile:** 16/128/8 dimensions, 128 ticks,
  shifts one, bias zero, int8 weights and int16 state; uses SDK graph containers
  and Atlas C arithmetic. Firmware mapping hooks explicitly reject deployment.
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
| SpiNNaker2: dense fixtures + exhaustive two-input probe | 23 | 83,904 | Exact; shifts 14–16 explicitly rejected |
| SpiNNaker2: original 98.74% MNIST MLP | 1 | 2,660,000 | Exact against fresh Atlas; 9,874/10,000 correct |
| XyloSim: count | 20 | 2,242 | Exact |
| SpiNNaker2: temporal LIF | 48 | 101,940 voltage samples | Max float64 error 1.14e-13; 35,093 spike events exactly match |
| XyloSim: temporal, preceding suite | 50 | 505,455 | Exact, including 7-step chunks |
| XyloSim: native integer-trained four-class MNIST | 1 | 27,136,896 spike/current/membrane values | Exact; 3,832/4,157 correct; 208,896-value concrete AtlasIR replay also exact |
| XyloSim + custom SpiNNaker2: block-sequential MNIST | 1 per target | 141,480,000 neural values per target | Exact; 9,747/10,000 correct on all ten classes; 98% target unmet |
| XyloSim: recurrent integer order task | 1 | 5,840,640 spike/current/membrane values | Exact, including fixed/irregular chunks; 1,872/1,872 correct; both memory controls 50% |
| SpiNNaker2 custom C host kernel: same recurrent order task | 1 | 5,840,640 spike/current/membrane values | Exact, including 1/7/13-tick chunks; 1,872/1,872 correct; firmware unimplemented |
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

**166 Python regression tests passed**, including timestamp regression tests,
semantic admission failures and existing backend checks.

- [DynapSim execution audit](dynapse/evidence/audit.json)
- [Static execution audit](coverage/evidence/static-audit.json)
- [Expanded SpiNNaker2 audit](coverage/evidence/spinnaker2-audit.json)
- [Xylo temporal evidence](xylo_temporal/evidence/expanded/audit.json)
- [Source identities and environment](coverage/evidence/manifest.json)
- [Reproduction commands](coverage/README.md)

The frozen TF MNIST model's earlier full-test result remains **90.03% on all
five count targets**, with exact conversion equivalence. The older **98.74%**
signed/biased MLP is a different model. It was checked on Akida/Lava against
historical Atlas output and has now been rerun on SpiNNaker2 against a fresh
Atlas reference, with 98.74% accuracy and zero state differences.
See the [original TF results](README.md) and
[additional count-target results](count_targets/README.md).

## Acceptance checklist for a new capability

- [x] Validate admitted and rejected profile/target combinations.
- [x] Derive each SDK plan from the same validated source used by native Atlas.
- [x] Run real SDKs; keep source, result, implementation and SDK identities.
- [x] Test numerical boundaries, silence, repeated input and seeded random cases.
- [x] Compare full observable states; compare exact pulse timing where promised.
- [x] Record discovered limitations instead of shifting outputs or loosening tolerance.
- [x] Add task-level accuracy evidence for native LIF networks (MNIST and fixed-context character prediction).
- [x] Measure same frozen weights on a second target, distinguishing approximate adaptation from exact lowering.
- [x] Train with Xylo integer dynamics and preserve native spike/state traces after export.
- [ ] Preserve one native temporal transition function across two target SDKs.
- [x] Add a bounded synthetic event-order task with learned recurrence and memory controls.
- [ ] Add a real event-stream task and test temporal generalization beyond the training generator.
- [ ] Verify physical placement, execution and measured energy/latency.

The unchecked items are future work, not prerequisites already satisfied by
these software conversion tests.
