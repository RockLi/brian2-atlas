# Neuromorphic conversion branch acceptance

This branch delivers bounded model conversion and reproducible software
inference. Classification accuracy is a model-quality check, not the criterion
for conversion correctness. Further MNIST accuracy tuning is deferred; the PR
remains open for review and is not merged by this closeout.

## Fixed inference baseline

The [fourth-round model](sequential_mnist/ROUND4.md) is retained without further
training, decoder changes or test-set selection:

| Property | Recorded result |
|---|---|
| Task | Ten-class block-sequential MNIST, all 10,000 official test images |
| Model | 16 inputs, 63 recurrent hidden neurons, 5 physical outputs; 5,292 active weights |
| Readout | Fixed host two-of-five spike-count decoder; 36 ticks, readout [16,36) |
| Accuracy | 97.44% in native Atlas, stock XyloSim and the custom SpiNNaker2 C host kernel |
| Neural equivalence | Zero differences in 73,440,000 spike/current/membrane values per target |
| Class scores | All 100,000 logits exactly equal |
| Selection | Validation-selected; previous development test scores were known |

The previous wider model remains the accuracy best at 97.47%. The 98% target
was not reached and is not being redefined as achieved. The compact model has
half as many active weights and nearly the same accuracy. The saved
[model and test evidence](sequential_mnist/round4/manifest.json) are unchanged.
The full dataset audit is retained; closeout reruns targeted native/SDK
regressions because valid integer arithmetic and model weights do not change.

## Interface and compatibility review

- The public conversion surface remains `brian2_atlas.neuromorphic`:
  `convert_ir`, `export_ir`, `run_simulator`, the LIF/Xylo freeze/export bridges
  and `XyloInferenceSession`. Conversion imports no vendor SDKs.
- [Six targets and six profiles](COMPATIBILITY.md) retain 12 admitted and
  24 rejected pairings. A fresh 36-cell routing check confirms this matrix.
  This does not imply that every model runs on all six targets.
- Native-training bundles require an explicit inference input batch. Their
  exported zero-input raster is a template; omitting inputs now raises
  `ValueError`. Direct temporal AtlasIR bundles still execute their embedded
  input schedule.
- Native Xylo and custom SpiNNaker2 readout dimensions, finite scale, time
  window, count range, class codes and current normalization are validated
  before SDK model allocation. Invalid metadata cannot advance an Xylo session.
- Spike-count decoding remains the default. Current decoding retains its
  distinct `atlas.native-xylo-current-bridge.v1` tag; old runtimes reject it.
  The optional rate objective affects training only. Legacy frozen v1/v2
  spike models keep their existing behavior.
- Plans are trusted compiler artifacts. These checks define API error behavior;
  they do not make arbitrary hand-written or untrusted simulator plans safe.

## Regression gates

The closeout adds 67 SDK-free interface cases and verifies 340 distinct Python
tests with zero failures or skips, including the affected conversion, native
training and actual XyloSim/SpiNNaker2 suites. See the [local receipt](acceptance/regression.json)
and [36-cell routing result](acceptance/compatibility.json).
The existing DynapSim rejection test now accepts the specific diagnostic for
an unsupported custom SpiNNaker2 profile; that pairing remains rejected.

GitHub Actions now runs conversion/profile tests as well as native integer
training tests on Python 3.12 and 3.14, and explicitly runs the Rust library
tests. Vendor SDK cases remain optional in CI; local XyloSim and SpiNNaker2
runs cover those integrations separately. A passing SDK-free job is not a
claim that all six vendor runtimes executed in CI.

The Linux/Python 3.12 and macOS/Python 3.14 installed-wheel gates now exercise
native current-readout training, freeze/export to both integer targets, and
compile and compare the packaged C kernel against native states outside the
checkout, without vendor SDKs. They also retain the existing simulation,
checkpoint-resume and package-immutability checks.

Use the exact PR head's [GitHub checks](https://github.com/RockLi/brian2-atlas/pull/2/checks)
as the hosted CI status. Historical green runs are not evidence for a later
commit. The PR description records the final reviewed head and run.

## Deferred work

Physical chip placement, firmware execution, energy and latency remain
unverified. SpiNNaker2 integer inference is the Atlas C99 host implementation
using SDK graph containers, not stock `Brian2Backend` or PE firmware.
General graph import, unrestricted temporal models, on-chip training and
higher-accuracy MNIST experiments are outside this branch's acceptance scope.
