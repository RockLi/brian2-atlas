# AtlasIR → DYNAP-SE2 DynapSim pilot

This adds a sixth **software target**, `dynapse`, using Rockpool's stock
`LinearJax` and `DynapSim` modules on CPU. It admits one explicit circuit-model
subset through `profile="dynapse-dpi-v1"`. No SDK source, Atlas executor or IR
wire schema was changed.

DynapSim is an approximate model of DYNAP-SE2 analog circuit dynamics, not a
bit-accurate hardware emulator. These tests establish conversion agreement with
that numerical model. They do not establish hardware fit, parameter
quantization, physical spike timing, accuracy, power or latency.

## Verified result — 2026-10-13

- **16 cases**, from 1 to 8 neurons and 1 to 512 steps at 1 ms.
- **36,985 trace values**: synaptic current, adaptation current, membrane current,
  membrane voltage and spike multiplicity. Also checked each final refractory
  timer; the stock SDK does not record its full timer trace.
- **510 spike events**, matching exact tick and neuron, with matching spike
  counts. The observed cases emitted at most one spike per neuron per tick.
- Stock SDK float32 compared to Atlas reference-f64 using **rtol=2e-4**;
  absolute tolerance is **1e-17 A** for currents and **2e-6 V** for voltage.
  Acceptance is `abs(error) <= atol + rtol*abs(reference)`. Final refractory
  timers use the same rtol and atol=1e-9 s. Tolerances were set before the runs;
  no time alignment or output shift is applied.
- Maximum observed errors: adaptation **1.83e-16 A**, membrane current
  **2.75e-12 A**, synaptic current **6.59e-13 A**, voltage **2.73e-6 V**.
- **3 independent Brian2 NumPy cases** match native Atlas, including adaptation
  and eight neurons, at rtol=1e-12, atol=1e-24, with exact event agreement.
- **159 Python regression tests passed**, including 25 new admission/API checks.
- The full public compatibility matrix now checks **30 cells**:
  **10 admitted, 20 rejected**.

See the [SDK audit](evidence/audit.json), [Brian2 check](evidence/brian2-audit.json),
[compatibility check](evidence/compatibility.json) and
[implementation identities](evidence/manifest.json). This is a numerical
correctness experiment, not an MNIST or training experiment.

## Conversion contract

The example starts with a real Brian2 `NeuronGroup` containing the explicit
DPI circuit update and an embedded binary input raster. Native Atlas executes
that source IR. The converter validates the same IR, verifies its complete
normalized definition against a reviewed fingerprint, extracts the actual
input weights and circuit parameters, and emits the SDK plan.

The SDK executes a diagonal `LinearJax` layer followed by `DynapSim`. Weight
application and all neural dynamics run inside stock SDK modules; the host
only assembles the graph and reads recordings. Parameters and raw input are
passed through without fitting them to Atlas output.

Admitted behavior:

- One population of 1–8 independent DPI neurons, with one binary input channel
  per neuron and diagonal weights in `{0, 0.5, 1, 2, 4}`.
- Synaptic current decay, nonlinear membrane feedback, spike-triggered
  adaptation, reset and refractory timer dynamics.
- A fresh run of 1–512 steps; initial currents equal `Io`, other state is zero.
- SI magnitudes stored in dimensionless f64 variables, matching Rockpool's
  numerical API. The runtime uses the stock float32 SDK computation.
- Fixed synthetic circuit/layout parameters and small enumerated choices for
  DC current, threshold, adaptation weight and refractory current. The exact
  values are defined in `dynapse_model.py` and checked in `dynapse_b2ir.py`.
  These are exactly representable test parameters, not calibrated silicon
  settings or a quantized hardware configuration.
- All six native states recorded at the end of every step; the SDK comparison
  covers its five available traces and final refractory state.

The fingerprint includes equations, state and parameter declarations,
threshold, schedule, recording and names. Only validated neuron count, run
length, recording indices and raster contents vary. This deliberately accepts
the supplied source form, not mathematically equivalent arbitrary expressions.
Changing the source equation requires a reviewed new contract and SDK tests.

Not admitted: recurrence, inhibitory/multichannel synapses, extra populations,
custom schedules, device mismatch, nondefault initialization, continuation,
plasticity, training, arbitrary LIF networks or frozen static TF classifiers.
No generic LIF-to-DPI approximation or MNIST conversion is claimed.

## Reproduce

Use a fresh isolated Python 3.12 environment:

```sh
python3.12 -m venv /absolute/path/to/dynapse-env
/absolute/path/to/dynapse-env/bin/pip install --no-cache-dir \
  rockpool==3.1.0 jax==0.4.35 jaxlib==0.4.35 numpy==2.2.6 scipy==1.18.1
/absolute/path/to/dynapse-env/bin/pip check

export PYTHONPATH="$PWD:$PWD/brian2-rust/python"
export B2_RUNNER=/absolute/path/to/b2-runner
python brian2-rust/examples/neuromorphic/dynapse/compare.py \
  --runner "$B2_RUNNER" --sdk-python /absolute/path/to/dynapse-env/bin/python \
  --output output/neuromorphic/dynapse
python brian2-rust/examples/neuromorphic/dynapse/verify_brian2.py \
  --run output/neuromorphic/dynapse
```

The tested host is macOS arm64; SDK execution uses CPU and its default float32
mode. Samna is not needed for this simulator-only path. Optional Torch/Samna
import warnings do not indicate hardware execution. SDK workers receive only
converted bundles, never reference traces. Results refuse overwrites and keep
source IR, plans, native results, SDK records and hashes.

For public conversion use:

```python
from brian2_atlas.neuromorphic import export_ir
bundle = export_ir("source.json", "converted", target="dynapse",
                   profile="dynapse-dpi-v1", runner="/absolute/path/to/b2-runner")
# In the separate SDK environment: run_simulator(bundle), with no samples argument.
```

Primary references: [Rockpool developer notes and deployment distinction](https://rockpool.ai/devices/DynapSE/developer.html),
[DynapSim API](https://rockpool.ai/reference/_autosummary/devices.dynapse.DynapSim.html),
[simulation model scope](https://rockpool.ai/reference/_autosummary/rockpool.devices.dynapse.simulation.html).
