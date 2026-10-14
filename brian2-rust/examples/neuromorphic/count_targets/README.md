# The same frozen MNIST classifier on SpiNNaker2 and XyloSim

This adds two software targets to the existing `binary-count-v1` conversion
profile. The source is the same external TensorFlow integer checkpoint and
canonical AtlasIR used by Akida, Lava and Specksim. No retraining, weight
clipping or label-dependent adjustment is used.

## Verified result — 2026-10-13

| Execution | Correct / samples | Accuracy | State differences from fresh Atlas |
|---|---:|---:|---:|
| Frozen TF reference | 9,003 / 10,000 | 90.03% | 0 class-count differences |
| Native Atlas Rust runner, rerun this experiment | 9,003 / 10,000 | 90.03% | reference |
| SpiNNaker2 0.8.0, official Brian2 backend | 9,003 / 10,000 | 90.03% | 0 / 200,000 |
| XyloSim 0.1.3, official v2 C++ core | 9,003 / 10,000 | 90.03% | 0 / 200,000 |

Both SDKs also passed the 14 common boundary inputs (zero, one, individual
pixels, alternating and seeded random inputs), together with a repeated
20-image smoke subset. Xylo's internal currents, post-reset voltages and all
38 hidden spike counts matched the derived integer equations for every sample.
The TF SavedModel result is the previous actual TF execution: its model,
input and output hashes were rechecked. Atlas and both new SDKs were freshly run.

The conversion regression suite passed 91 tests. See [the audit](evidence/audit.json)
and [source/environment identities](evidence/manifest.json). These targets use the existing AtlasIR state and event semantics. New exports
use `atlasir-v1`; the recorded audit retains its original wire identifiers.

**This is the 90.03% 784→10 count classifier.** The 98.74% signed, biased MLP
remains outside these targets' admitted profile. The result demonstrates zero
conversion loss for the smaller model, not a new accuracy improvement.

## What the two compilers emit

Both start with the existing proof for `floor(S/T)`, where `S` is the
nonnegative integer weighted sum of binary pixels. Clipping is provably inactive
for every binary input. The source has no spike-time contract. The SDK worker
gets only inputs and the compiled bundle, never labels or reference outputs.
Host work is input encoding, spike counting/reordering and recording; all
weighted sums and threshold/reset operations execute inside the official SDK.

### SpiNNaker2

Use `lif_no_delay` (signed 8-bit weights), which holds all original weights
without clipping. For each class create a comparator neuron at each reachable
threshold `k*T - 0.5`, with reset to zero and `alpha_decay=0`. In the SDK Euler
implementation this clears membrane state before each image. For integer `S`:

```
sum(k=1..max_count) [S > k*T - 0.5] = floor(S/T)
```

One image is encoded per tick. The compiler specifies the one-tick input
pipeline delay. The runtime reads the SDK's raw monitor and rounds its time
values to ticks; the SDK's public spike dictionary truncates `int(t/dt)`, which
can assign floating timestamps to the preceding tick on long runs. No neuron
or synapse implementation is patched. Weight quantization stays enabled.

This model expands to **784 input neurons and 1,061 comparator neurons**.
The stock backend uses upstream Brian2, not a chip arithmetic emulator. The
expansion is a correctness pilot, with no hardware resource placement or
throughput claim. It is separate from the [temporal LIF profile](../spinnaker2/README.md).

### SynSense Xylo Audio 2 software core

Use the official `xylosim.v2.XyloLayer` C++ simulator directly. Rockpool, Samna
and hardware access are not required for this runtime. Each hidden neuron can
emit at most 31 spikes per tick, so a class count is split into groups of 31.
The original weights are retained; input weight shift 1 and synaptic decay
shift 1 give `2*S - ((2*S)>>1) = S` from a fresh zero state. Compiler-derived
bias `-31*k*T` selects a count group:

```
sum(k=0..ceil(max_count/31)-1) clip(floor((S-31*k*T)/T),0,31) = floor(S/T)
```

The runtime resets the complete SDK state before each image. All intermediate
values stay within signed 16-bit range over the admitted input domain; no
saturation is relied upon. There are **38 hidden neurons** and one disconnected
output neuron. Classification reads the recorded hidden spikes.

**This MNIST configuration does not fit the chip's input interface:** it uses
784 input channels, while Xylo Audio 2 has 16. The simulator permits the larger
input. The bundle explicitly records that violation. Recorded hidden counts
are also a different readout from the chip's eight output-neuron interface.
No hardware deployment or mapping is claimed. A deployable next experiment
needs a smaller input representation and a hardware-compatible readout.

## Reproduce

First reproduce the [common example](../README.md), including boundary inputs.
It supplies `common/source.json`, input/label files, TF reference results and
metadata. The frozen checkpoint is included under `../fixtures/tf-count`.
Use the native runner built from this repository.

Install SpiNNaker2 in the separate Python 3.12 environment described in the
[temporal example](../spinnaker2/README.md#reproduce), pinned to commit
`8a1c4c526aec0f2cab73b8c37c9491fac0118b42`. Install the Xylo runtime separately:

```sh
python3.12 -m venv /absolute/path/to/xylo-env
/absolute/path/to/xylo-env/bin/pip install xylosim==0.1.3 numpy==2.5.4
/absolute/path/to/xylo-env/bin/pip check
```

XyloSim builds a small native extension; the verified machine is macOS arm64
with its C++ compiler installed. The downloaded source archive hash is recorded.

From the repository root, choose fresh output paths:

```sh
export PYTHONPATH="$PWD:$PWD/brian2-rust/python"
export DEMO="$PWD/brian2-rust/examples/neuromorphic"
export COMMON="$PWD/output/neuromorphic/common"
export RUN="$PWD/output/neuromorphic/count-new-targets"
export B2_RUNNER=/absolute/path/to/b2-runner

python "$DEMO/count_targets/prepare.py" --common "$COMMON" --output "$RUN" --runner "$B2_RUNNER"

# Repeat with TARGET=xylosim and its separate Python environment.
export TARGET=spinnaker2
export SDK_PYTHON=/absolute/path/to/spinnaker2-env/bin/python
PYTHONPATH='' "$SDK_PYTHON" "$DEMO/sdk_worker.py" \
  --bundle "$RUN/$TARGET/bundle.json" --inputs "$RUN/smoke.npz" \
  --runtime "$PWD/brian2-rust/python/brian2_rust/neuromorphic_runtime.py" \
  --output "$RUN/$TARGET-smoke"
PYTHONPATH='' "$SDK_PYTHON" "$DEMO/sdk_worker.py" \
  --bundle "$RUN/$TARGET/bundle.json" --inputs "$COMMON/inputs.npz" \
  --runtime "$PWD/brian2-rust/python/brian2_rust/neuromorphic_runtime.py" \
  --output "$RUN/$TARGET-full"

# After both targets finish, rerun Atlas and verify every result and identity.
python "$DEMO/count_targets/verify.py" --common "$COMMON" --run "$RUN" \
  --trained "$DEMO/fixtures/tf-count" --runner "$B2_RUNNER"
```

The verification output directory refuses overwrites. It preserves a fresh
Atlas result and an audit containing each batch's canonical source hash,
SDK versions and file identities. SDK timing is CPU software wall time only.

Primary references: [SpiNNaker2 models and weight formats](https://spinnaker2.gitlab.io/py-spinnaker2/tutorials/deep_dive/02_connectivity.html),
[Xylo software simulation and hardware dimensions](https://rockpool.ai/devices/quick-xylo/deploy_to_xylo.html),
[XyloSim release](https://pypi.org/project/xylosim/0.1.3/).
