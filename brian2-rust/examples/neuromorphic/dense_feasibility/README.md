# Can the 98.74% MNIST MLP reach the other targets?

Assessed 2026-10-13. A rejected public profile is an implementation boundary,
not a proof that a chip cannot compute the model. The public compatibility
matrix remains unchanged by these feasibility experiments.

## Findings

| Target | Current evidence | What remains |
|---|---|---|
| Akida / Lava | Full original model previously verified | Already supported by the bounded dense profile |
| SpiNNaker2 | Exact small signed/biased MLP mechanism demonstrated with stock SDK | A proper AtlasIR lowerer, resource reduction, full original-model execution and MNIST accuracy audit |
| Specksim | Direct bias conversion rejected; signed event order changes output counts | A new encoding or approximate ANN-to-SNN conversion; exact preservation has not been established |
| XyloSim / Audio 2 | Current direct mapping exceeds channel and accumulator limits | Partitioning, temporal encoding and exact arithmetic handling; hardware mapping separately |
| DynapSim | Circuit-model profile only; no exact integer MLP lowering | A separately designed representation or approximate model conversion and a fresh accuracy study |

None of the latter four targets has been shown universally incapable of the
computation. Nor does the available evidence justify promising unchanged
98.74% accuracy on them.

## Original model inspected

The saved validated model is 784→256→10. First-layer weights range -14…7,
second-layer coefficients -61…36, and the hidden right shift is 7. Both layers
have bias; hidden values use rounding and clipping to [0,127].

Across the legal independent input domain 0…127, the first affine expression
including bias and rounding offset can attain extrema **-122,364 and 85,170**
(across neurons). Each of the 256 hidden neurons has at least one attainable
extreme outside signed 16-bit range. The output absolute interval bound is
501,787; this is a conservative bound, not an observed MNIST maximum.

These facts matter for Xylo Audio 2's 16-bit state and 16 input channels. Its
8 output neurons also cannot directly represent a 10-neuron output layer in a
one-to-one mapping. Bias and signed weights themselves are supported by Xylo;
calling it a general lack of those operations would be incorrect. Software
encodings can exceed hardware dimensions, but that is not deployment evidence.

## SpiNNaker2 experiment

The installed SDK's `lif_32bit` application supports 20 weight bits, distinct
from the narrower `lif_no_delay` path used by our count converter. Its stock
Brian2 backend can execute the required signed synaptic arithmetic.

The experiment compiles a **2→3→2** quantized MLP into:

1. Seven binary input planes per input, retaining exact integer input values.
2. 127 threshold comparators per hidden unit. For integer affine value `S`,
   counting `S + bias + 64 > 128*k - 0.5`, for k=1…127, implements the source
   rounded/clipped hidden activation exactly.
3. Nonfiring readout neurons whose recorded voltage is the affine output.
   Bias and weighted sums are executed by SDK neurons and synapses. The host
   does not supply neural-layer results or apply a missing activation.

All **16,384** two-input combinations in 0…127 were executed in native Atlas
(in two batches) and the stock SDK, with weight quantization enabled.
**81,920 observed hidden/output values matched exactly.** The cases exercise
negative coefficients, positive and negative bias, rounding boundaries,
active upper/lower clipping and signed affine outputs. SDK public float32
records represent these integer results exactly.

This is a mechanism experiment driven by a validated dense plan, not the new
public `target="spinnaker2", profile="dense-int8-v1"` converter. It does not
run the original 784→256→10 model or establish its 98.74% accuracy on this target.
The SDK remains an upstream Brian2 software backend, not a chip emulator.

Resource cost is substantial: the naive same construction for the original
model would require 5,488 input bit-plane neurons, 32,512 hidden comparators,
and **130,590,544 first-layer connections**, before the readout. These are
construction estimates, not hardware mapping results. A practical next step
must reduce this expansion before a full-model run or efficiency claim.

## Specksim counterexample

The installed Sinabs conversion function raises `Biases are not supported!`
for a biased convolution. This is also documented by the SDK maintainers.

A second actual SDK test uses weights `[2, -1]`, IAF threshold 2 and minimum
membrane -100. Both input features fire once. Positive-first order produces
one output spike and final membrane -1; negative-first order produces zero
spikes and final membrane 1. The static target count is `floor((2-1)/2)=0`.

An early positive spike cannot be withdrawn when inhibition arrives later.
This disproves a direct order-independent count mapping for that signed case;
it does **not** disprove every possible encoding. Constant input channels may
encode bias, and additional control phases may address timing, but neither
constitutes a complete validated conversion of the original MLP yet.

## Reproduce

Use the pinned environments from the existing simulator examples.

```sh
export PYTHONPATH="$PWD:$PWD/brian2-rust/python"
export B2_RUNNER=/absolute/path/to/b2-runner
export RUN="$PWD/output/neuromorphic/dense-feasibility"
export CODE="$PWD/brian2-rust/examples/neuromorphic/dense_feasibility"
python "$CODE/prepare.py" --runner "$B2_RUNNER" --output "$RUN"
PYTHONPATH='' PYTHONNOUSERSITE=1 /absolute/path/to/spinnaker2-env/bin/python \
  "$CODE/spinnaker_probe.py" --run "$RUN"
# Run speck_probe.py inside the existing Specksim SDK container, mounting CODE
# read-only and RUN writable. It writes a small JSON result:
# /code/speck_probe.py --output /work/speck-probe.json
python "$CODE/audit.py" --run "$RUN" \
  --legacy-bundle /absolute/path/to/original-mlp/akida/bundle.json \
  --speck-probe "$RUN/speck-probe.json"
```

All scripts refuse output overwrites. See [saved findings](evidence/audit.json)
and [script identities](evidence/manifest.json). This change makes no public
capability or model-accuracy claim beyond the tests above.

Primary sources: [SpiNNaker2 neuron models](https://spinnaker2.gitlab.io/py-spinnaker2/api_reference/neuron-models.html),
[SDK Brian2 backend](https://spinnaker2.gitlab.io/py-spinnaker2/user_guide/brian2.html),
[Specksim supported architecture and bias restriction](https://sinabs.readthedocs.io/v3.1.1/speck/specksim.html),
[Xylo architecture](https://rockpool.ai/devices/xylo-overview.html),
[DynapSim simulation/deployment distinction](https://rockpool.ai/devices/DynapSE/developer.html).
