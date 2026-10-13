# AtlasIR → SpiNNaker2 software backend

This pilot converts a small, time-dependent LIF network from AtlasIR into
`py-spinnaker2` populations and projections. It runs the source with the native
Atlas Rust runner and the converted network with the SDK's unmodified
`Brian2Backend`, using upstream Brian2 in a separate Python environment.

The existing AtlasIR schema already carries the required state updates,
thresholds, resets, connectivity, delays, spike inputs and schedule. No IR
schema extension or Rust executor change was needed.

## Expanded coverage — 2026-10-13

With `--coverage`, the suite now runs **48 networks**, including 32 seeded
random cases, one neuron, one tick, 256 input / 256 LIF neurons and 10,000 steps.
It compares **101,940 voltage samples** and **35,093 spike events**. Spike ticks
and indices match exactly; maximum float64 voltage error is **1.14e-13**, within
`atol=1e-12, rtol=0`. SDK public float32 traces match Atlas cast to float32.

The longest case exposed the SDK's floating timestamp truncation: its public
`get_spikes()` reports 31 events one tick early. The adapter now rounds aligned
raw monitor timestamps and preserves the public timestamps in the result for
diagnostics. No event delay or SDK code was changed. Four regression tests
cover near-integer and invalid timestamps; all 134 shared Python tests pass.

See the [expanded audit](../coverage/evidence/spinnaker2-audit.json),
[reproduction guide](../coverage/README.md) and
[compatibility checklist](../COMPATIBILITY.md). This remains an upstream Brian2
software backend comparison, not a chip-emulator or hardware test.

## Original result

Verified on macOS arm64, 2026-10-13:

- 12 actual SDK comparisons: 1,596 voltage samples and 394 output spikes.
- Maximum absolute error in the SDK's internal float64 voltage trace: **0**.
- SDK public float32 voltage records match the Atlas trace cast to float32.
- Both public SDK spike records and raw monitor records match Atlas exactly
  in neuron index and timestep. No time shift is applied to make them agree.
- Final membrane states match. The acceptance tolerance for float64 voltage
  is `atol=1e-12, rtol=0`; exact equality was observed in these cases.
- 83 Python tests passed, including 21 new admission/API tests and the existing
  Akida, Lava and Specksim conversion regressions.

The cases cover recurrent excitation/inhibition, subtractive reset, reset to a
nonzero value, zero and seven-tick delays, events pending at the end of a run,
threshold equality, threshold crossing, ±15 weights, silent input and seeded
input spikes. Each case starts with zero membrane state.

See [the audit](evidence/audit.json), [source identities](evidence/source-manifest.json)
and [SDK environment](evidence/sdk-freeze.txt). These are software results;
there is no hardware execution, resource mapping, energy measurement or MNIST
accuracy result in this pilot. The SDK itself uses Brian2, so this is not a
comparison against an independent chip emulator.

## Admitted profile

Use `target="spinnaker2", profile="spinnaker2-lif-v1"` with the existing
`brian2_atlas.neuromorphic.export_ir` API. This profile is separate from the
static TF classifier profiles used by the other three targets.

The deliberately narrow model is:

- One static `SpikeGeneratorGroup` and one LIF population, at most 256 neurons
  each; an input projection and an optional recurrent projection.
- Aligned 1 ms clocks, a fresh run from time zero, at most 10,000 steps.
- Dimensionless voltage with the explicit Euler equation
  `dv/dt = -v/tau + i_offset/dt`, threshold `v > threshold`, and either
  `v = v_reset` or `v -= threshold`.
- Constant per-neuron `tau`, `i_offset`, `threshold`, `v_reset`; parameters must
  survive the admitted alpha-decay conversion without loss. Nonzero initial
  voltage is rejected because the current SDK's `lif` backend ignores `v_init`.
- Static nonzero integer weights in `[-15, 15]`, integer delays in `[0, 7]`
  ticks. SDK quantization remains enabled but cannot change accepted values.
- No multapses, plasticity, refractory period, extra code, current-based LIF,
  custom events, subgroups, continuation or pending input queues.
- Full spike recording and end-of-step voltage recording.

The compiler checks the actual update/threshold/reset ASTs; it does not infer
semantics from a population name. This first version requires the documented
variable names and disables Brian2 loop-invariant codegen optimizations when
exporting, so its accepted AST is explicit. Unsupported forms fail conversion.

The SDK uses `start → groups → synapses → thresholds → resets → end`, which
places synapse execution before thresholding. The source must explicitly use
that same order; a default Brian2 schedule is rejected. Under this schedule,
a zero-delay spike at tick 0 is consumed at tick 1. AtlasIR retains this
ordering rather than compensating by changing delays or shifting output spikes.

## Reproduce

The tested SDK is `py-spinnaker2` 0.8.0, source commit
`8a1c4c526aec0f2cab73b8c37c9491fac0118b42`, with Brian2 2.10.1,
NumPy 2.3.4 and SciPy 1.18.1. Use Python 3.12: although the SDK advertises
Python >=3.10, its pinned PACMAN2 dependency requires >=3.12.

Install the official source archive in a separate environment:

```sh
# Choose new directories for these paths.
export SDK_SRC=/absolute/path/to/py-spinnaker2
export SDK_DIST=/absolute/path/to/py-spinnaker2-source
export SDK_ENV=/absolute/path/to/spinnaker2-env

git clone https://gitlab.com/spinnaker2/py-spinnaker2.git "$SDK_SRC"
git -C "$SDK_SRC" checkout --detach 8a1c4c526aec0f2cab73b8c37c9491fac0118b42
mkdir -p "$SDK_DIST"
git -C "$SDK_SRC" archive HEAD | tar -x -C "$SDK_DIST"
python3.12 -m venv "$SDK_ENV"
"$SDK_ENV/bin/python" -m pip install "$SDK_DIST"
"$SDK_ENV/bin/python" -m pip check
```

Installing an archive avoids the checkout-only hardware-submodule setup hook,
which failed under the macOS system Bash in this run. The archive contains the
unaltered Python simulator source. Chip-side libraries and toolchain setup
are not part of this software-only test. The audit records the installed
`brian2_sim.py` hash and it matches the pinned source checkout.

From the Atlas repository, with its Python dependencies and native runner built:

```sh
export PYTHONPATH="$PWD:$PWD/brian2-rust/python"
export B2_RUNNER=/absolute/path/to/b2-runner
python brian2-rust/examples/neuromorphic/spinnaker2/compare.py \
  --runner "$B2_RUNNER" \
  --sdk-python "$SDK_ENV/bin/python" \
  --output output/neuromorphic/spinnaker2-lif
```

Choose a new output directory on each run. The script exports each source,
validates it with Atlas, executes Atlas, then starts a separate SDK worker with
`PYTHONPATH` cleared. The worker reads only converted bundles, not reference
traces or labels. Comparison happens after both executions finish. SDK logs,
all source IR files, plans, outputs and a compact `audit.json` are retained.

For SDK-free admission tests and existing conversion regressions:

```sh
python -m pytest -q \
  brian2-rust/tests/test_spinnaker2_b2ir.py \
  brian2-rust/tests/test_neuromorphic.py \
  brian2-rust/tests/test_akida_b2ir.py \
  brian2-rust/tests/test_lava_dense_b2ir.py
```

References: [official software backend documentation](https://spinnaker2.gitlab.io/py-spinnaker2/user_guide/brian2.html),
[pinned SDK source](https://gitlab.com/spinnaker2/py-spinnaker2/-/tree/8a1c4c526aec0f2cab73b8c37c9491fac0118b42).
