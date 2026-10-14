# Recurrent integer model: custom SpiNNaker2 software kernel

The same frozen 3→24→2 recurrent model now executes through an **Atlas custom
C host kernel registered with the SpiNNaker2 SDK**. Native Atlas, stock XyloSim
and the custom kernel all classify 1,872/1,872 held-out sequences correctly.
Spikes, currents, membranes and logits match exactly, without clock shifts,
tolerance, retraining or weight changes.

This is a new software extension. It does **not** change the result of the
[stock current-LIF diagnostic](SPINNAKER2.md), which remains at 50%. It is not
execution in stock `Brian2Backend`, a silicon simulator, or deployed SpiNNaker2
firmware. The current extension refuses firmware mapping/build hooks rather
than treating a host result as evidence of chip support.

## Execution path

```text
Frozen native Xylo integer model (unchanged v2 weights)
    → Brian2 network → validated AtlasIR
    → spinnaker2-xylo-integer-v1 plan
    → SpiNNaker2 SDK Population / Projection objects
    → Atlas IntegerHostBackend
    → separately compiled C99 integer kernel
    → per-tick spike counts, currents, membranes and readout logits
```

The new profile uses the existing strict Xylo AtlasIR admission, then narrows it
to shifts of one, zero bias and at most 128 ticks. It preserves input-before-
recurrence ordering, previous-tick hidden counts for recurrence and readout,
source-order accumulation with per-edge saturation, and hidden/output caps 31/1.
The original `spinnaker2-lif-v1` profile continues to reject these dynamics.

The SDK's documented custom-model registry registers
`atlas_xylo_integer_host_v1`. Its populations expose `I`, `v`, `q` and spike
records. The backend reads thresholds and signed matrices back from SDK objects;
all three matrices are checked against the converted plan on every batch.
Spike records repeat the tick once per emitted count. This is a software
observation format, not an implemented chip event-packet encoding.

All neural state updates execute in the shipped C source. The worker never
calls the Atlas Rust runner, XyloSim, or Brian2 differential-equation execution.
Labels and reference traces stay outside the worker. Python handles graph
construction, transport, recording and readout aggregation. The C kernel uses
explicit int32 intermediates bounded by the admitted contract, saturates to
int16, and spells out negative floor division instead of relying on the
implementation-defined behavior of signed right shift. It has no floating
neuron arithmetic or external numerical-library dependency.

## Held-out comparison

The dataset, checkpoint, split and readout window are the existing
[delayed AB/BA task](README.md). The frozen file SHA-256 remains
`404a39f2024033f2633bcd0fd8bc1c6efd847ef1ab9afcbadf2f81e07b8e9ffd`.
The input, readout and recurrent weights are unchanged. Fresh evaluations use
all 1,872 held-out sequences from 936 templates disjoint from training and
validation. The model is not selected again using target results.

| Execution | Correct / 1,872 | Accuracy |
|---|---:|---:|
| Native Atlas Rust integer forward | 1,872 | 100% |
| Stock XyloSim | 1,872 | 100% |
| Custom SpiNNaker2 SDK C host kernel | 1,872 | 100% |

| Quantity | Values | Native vs XyloSim differences | Native vs custom C differences |
|---|---:|---:|---:|
| Hidden/output spike counts | 1,946,880 | 0 | 0 |
| Synaptic currents | 1,946,880 | 0 | 0 |
| Post-reset membranes | 1,946,880 | 0 | 0 |
| Readout logits | 3,744 | 0 | 0 |

Each comparison covers **5,840,640 neural-state values**, plus logits.
The C path's full runs at batch size 64 also match 1-, 7- and 13-tick chunks
at batch size 37 over the entire test set. Chunks preserve the C state;
batches and sequences start fresh. A fresh 32-sequence concrete AtlasIR replay
matches another **99,840 state values** and all nonzero spike event locations
against the custom kernel. Multiplicity is checked separately through `q` and
SDK repeated-tick spike records.

[Audit](evidence/spinnaker2_integer/audit.json) includes source/native executable
hashes, C source and compiled-library hashes, compiler identity, SDK versions,
SDK weight readback identity, input identity, result hashes, chunk checks and
AtlasIR replay receipts. [Per-sample results](evidence/spinnaker2_integer/per_sample.npz)
retain labels, template IDs and all three sets of logits. Full traces are
created by reproduction and are not committed.

This remains a small synthetic memory test, not a real-world benchmark or
proof of long-duration generalization. It establishes equivalence of this
bounded model across the three software implementations.

## API and limits

```python
bundle = export_native_xylo(
    frozen, "integer-export", steps=40, runner=atlas_runner,
    target="spinnaker2",
)
# Execute in the isolated SpiNNaker2 SDK environment:
result = run_simulator(bundle, samples, batch_size=64, chunk_steps=7)
assert result["execution_kind"] == "custom-c-host-kernel"
assert result["dynamics_equivalent"]
assert not result["stock_sdk_neuron_model"]
assert not result["firmware_adapter_implemented"]
```

The default target of `export_native_xylo` remains `xylosim`. The new target
supports frozen v1 feedforward and v2 recurrent integer sources; the full
accuracy result above covers the recurrent fixture. Direct `convert_ir` and
`export_ir` use the explicit `spinnaker2-xylo-integer-v1` profile. No new IR
schema is introduced.

- At most 16 inputs, 128 hidden and 8 output neurons, 1–128 ticks, dt=1 ms.
- Signed int8 weights, integer thresholds 1–32767, zero bias, shifts one.
- Embedded raster counts 0–15; native training declares its input count limit
  explicitly (binary by default). Frozen v1 retains a 64-hidden bound.
- Input/readout and optional hidden recurrence, no explicit delays or plasticity.
- Conservative Xylo-source recurrent fanout limit 63 is retained. This is not
  a SpiNNaker2 placement limit or proof of fit on one processing element.
- Independent batches of 1–128 sequences. Calls start from zero; `chunk_steps`
  carries state only inside one call. There is no public persistent SDK session
  or arbitrary state-restoration API for this extension yet.
- A C99 `cc` compiler is required for execution. Conversion needs no SDK/compiler.
  Only the shipped C source is compiled, never code or paths supplied in a bundle.

The custom profile is a sixth profile on the existing six targets: the routing
matrix now checks 36 combinations, with 12 admitted and 24 rejected. The new
cell is specifically marked as a custom host kernel, distinct from stock SDK
support.

## Kernel and deployment boundary

`spinnaker2_integer_kernel/kernel.c` and `kernel.h` define a bounded C buffer
ABI. State is `[batch][I,V,Q][cell]`; traces are `[batch][tick][I,V,Q][cell]`.
Each call validates dimensions, weights, thresholds, input counts and initial
state before changing buffers. The Python wrapper validates buffer shapes and
copies continuation state. The CMake project builds a host shared library.

The custom application explicitly rejects `pe_config`, compiler flags, vertex
creation and splitting. Its CMake project rejects use as a SpiNNaker2 firmware
application. A deployable version still needs the PE ABI, model-region layout,
clock/event transport carrying count multiplicity, deterministic arrival and
saturation ordering, resource mapping, ARM toolchain build and hardware tests.
No packet payload encoding, real-time deadline, placement, energy or latency
claim follows from the current host run. The C kernel is the arithmetic core
for that future adapter; it is not that adapter.

Regression tests compare the C implementation against native Rust on learned,
random, signed-odd and mixed-saturation cases; check threshold equality,
minimum decay and readout timing; verify rejection without buffer mutation;
and run an extreme 128-tick case under address/undefined-behavior sanitizers.
SDK tests check replica/reset isolation, count-record multiplicity, chunking,
counted-input AtlasIR replay and firmware-hook rejection. These tests add
independent arithmetic evidence beyond the trained checkpoint.

## Reproduce

Use the main Atlas environment, current native binaries, an isolated Xylo
SDK environment and the existing isolated SpiNNaker2 SDK environment with `cc`
available. The source and SDK Python environments must remain separate.

```sh
export PYTHONPATH="$PWD/brian2-rust/python:$PWD"
python brian2-rust/examples/neuromorphic/native_xylo_recurrent/verify_integer_spinnaker2.py \
  --train-runner /path/to/b2-train \
  --runner /path/to/b2-runner \
  --xylo-python /path/to/xylo-env/bin/python \
  --spinnaker2-python /path/to/spinnaker2-env/bin/python \
  --output /tmp/recurrent-spinnaker2-integer
```

The output directory must not exist. Evaluation loads the original frozen
integer weights directly; it does not retrain or restore an optimizer state.
Outputs contain both conversion bundles, fresh SDK traces, concrete AtlasIR
replays, per-sample logits and the audit. To build the standalone host kernel:

```sh
cmake -S brian2-rust/python/brian2_rust/spinnaker2_integer_kernel -B /tmp/atlas-s2-host
cmake --build /tmp/atlas-s2-host
```
