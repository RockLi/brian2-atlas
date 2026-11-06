# Native Atlas character language model → AtlasIR → SpiNNaker2

This example trains a small next-character model in native Atlas, freezes and
explicitly quantizes its weights, exports a real AtlasIR inference graph, and
executes that graph through the stock `py-spinnaker2` Brian2 backend. It reports
held-out character perplexity and compares every evaluated token's neural
outputs with the quantized native model.

The model has a **fixed four-character context**. Each prediction starts with
zero LIF state and runs for 16 ticks. The host advances the context window by
one character. This is a bounded feedforward language model implemented with
LIF dynamics, not an EGRU, a Transformer, or a language model with recurrent
neural state carried between tokens. It uses the existing native LIF conversion
contract; no AtlasIR schema extension or new target profile is needed.

## Verified result

All **9,996 held-out next-character targets** in the declared test split were
evaluated. The validation-selected checkpoint is step 1,500 of the 3,000-step
training run. Both weight matrices changed: all 20,608 parameters were updated.

| Model | Character perplexity ↓ | Bits per character ↓ | Top-1 accuracy |
|---|---:|---:|---:|
| Uniform baseline | 32.0000 | 5.0000 | 16.63% |
| Training-only unigram | 19.9992 | 4.3219 | 16.63% |
| Training-only bigram | 11.4313 | 3.5149 | 25.24% |
| Native Atlas, float weights | 8.0880 | 3.0158 | 40.82% |
| Native Atlas, quantized weights | 9.9241 | 3.3109 | 36.96% |
| Converted SpiNNaker2 SDK | 9.9241 | 3.3109 | 36.96% |

Quantization adds **1.8361 perplexity**. Conversion adds **zero**:
all 319,872 logits and log probabilities, 25,589,760 hidden/output spike
entries, and 1,599,360 reconstructed final membrane values match exactly.
The separate native AtlasIR replay matches all 163,840 voltage values for
64 token contexts. Independent 32-character greedy generation also matches at
every token, logit and spike.

The quantized model beats the unigram and bigram baselines on this split. Its
four-character context remains limited; the greedy sample can repeat and is
not evidence of useful open-ended text generation. The measured sample is
recorded verbatim in the audit. Later float checkpoints improve validation
loss but can lose more after quantization; selection uses quantized validation
perplexity, not the final optimizer step or the test set.

Evidence: [full audit](evidence/audit.json), [per-token audit](evidence/token-audit.npz),
[training and selection history](fixtures/training.json),
[SDK execution](evidence/sdk-execution.json), and [local checks](evidence/validation.json).
The checkpoint and frozen model are included under `fixtures/`.
Training took 1,054.99 seconds; SDK test inference and result recording took
265.04 seconds on the development host. These are CPU software timings, not
chip latency or energy measurements.

The larger batched model also exposed repeated sorting/copying of the SDK
voltage dictionary in Atlas's result collector. The adapter now fetches that
dictionary once per batch. A real-SDK regression asserts the access count;
22 language/bridge tests and 28 SpiNNaker2 profile tests pass. The original
12 temporal SDK cases were rerun with exact voltage/spike agreement
([audit](evidence/temporal-regression.json)). This collection change does not
alter neural computation or the SDK.

## Model and training

- Fixed vocabulary of 32 characters: space, newline, `a`–`z`, `.`, `,`, `'`, `?`.
- Four position-specific one-hot vectors plus one constant spike input: 129 inputs.
- 128 hidden LIF neurons and 32 output LIF neurons; 20,608 learned weights.
- Euler decay beta=0.5, threshold=1, subtract reset; binary inputs repeated for
  16 ticks. Both weight matrices are trained, not just a readout.
- Logits are output spike counts × 8 / 16. Softmax defines the next-character
  distribution. Cross entropy, surrogate BPTT and Adam updates execute in
  `b2-train` on CPU, with no TensorFlow or PyTorch training dependency.
- 3,000 optimizer steps, batch size 128, learning rate 0.005, seed 20261013.
- Every 500 steps, freeze using the largest power-of-two scale per layer that
  keeps target weights inside [-15,15]. Rounding is nearest-even; no clipping.
  Select the checkpoint with the lowest quantized validation perplexity.

## Dataset and scoring

Source: [Tiny Shakespeare from char-rnn](https://github.com/karpathy/char-rnn/tree/master/data/tinyshakespeare).
The downloaded `input.txt` must have SHA-256
`86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`.
The corpus is downloaded separately and is not bundled with this example.

Normalization lowercases text, replaces every character outside the fixed
vocabulary with a space, and retains the first 200,000 normalized characters.
It does not collapse whitespace or learn preprocessing from held-out data.
The contiguous split is defined before training:

| Split | Character interval, half open | Next-character targets |
|---|---|---:|
| Training | [0, 180000) | 179,996 |
| Validation | [180000, 190000) | 9,996 |
| Test | [190000, 200000) | 9,996 |

Contexts and targets stay inside their own split. A target at position `p`
uses only positions `[p-4, p)`. Test evaluation uses teacher forcing: preceding
observed characters form the context. The test split is not evaluated during
training or checkpoint selection. This is a small, normalized corpus subset,
not a directly comparable standard Tiny Shakespeare benchmark score.

Perplexity is `exp(mean(-log p(next_character | context)))`, using stable
float64 log-softmax. Bits per character is the same mean loss divided by
`log(2)`. Uniform, unigram and bigram baselines use the same vocabulary and
target positions. Unigram and bigram counts are fit only on the training
interval, with fixed add-0.1 smoothing.

## Validation boundaries

The audit separates quantization from conversion:

1. Float native model versus the explicitly quantized native model: measure
   any perplexity change caused by rounding weights.
2. Quantized native model versus SDK on every test context: compare logits,
   log probabilities, argmax predictions, all hidden/output spike ticks, and
   final membrane state under the declared state transform.
3. Concrete native AtlasIR replay on the first 64 test contexts: compare its
   entire voltage trajectory and spikes with the SDK. The exported conversion
   template has zero input; these concrete replays contain actual test inputs.
4. Independent greedy generation of 32 characters from the fixed prompt
   `the ` in native Atlas and the SDK. Each loop feeds back its own predictions;
   compare every generated token, logit and spike. Ties choose the first
   vocabulary entry. This checks feedback consistency, not text quality.

The SDK worker receives only model and input files, never labels, native
outputs or reference states. Encoding, replica batching, count-to-logit
scaling, softmax/argmax and context-window updates are host operations; all
learned neural computation executes in the stock SDK. The native final state
is reconstructed for comparison using `v = z / scale + J`, as in the
[native LIF bridge](../native_training/README.md).

The target uses the upstream SDK's Brian2 backend, not an independent silicon
emulator. No physical placement, on-chip execution, memory fit, chip latency
or energy advantage is established. Batched replicas are a software testing
convenience. Arbitrary LLM conversion, EGRU gates, KV caches, long-context
attention and persistent inter-token neuron state are outside this example.

## Reproduce

Use the Atlas source environment and the separate
[SpiNNaker2 SDK environment](../spinnaker2/README.md). The example adds no
mandatory dependency to Atlas. Build native CPU `b2-train` and `b2-runner`, then:

```sh
export PYTHONPATH="$PWD/brian2-rust/python:$PWD"
LM=brian2-rust/examples/neuromorphic/language_model
OUT=output/neuromorphic/language-model
mkdir -p "$OUT"
curl -L --fail \
  https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt \
  -o "$OUT/input.txt"

python "$LM/train.py" \
  --dataset "$OUT/input.txt" \
  --runner /path/to/b2-train \
  --output "$OUT/trained"

python "$LM/verify.py" \
  --dataset "$OUT/input.txt" \
  --trained "$OUT/trained" \
  --runner /path/to/b2-runner \
  --train-runner /path/to/b2-train \
  --sdk-python /path/to/spinnaker2-environment/bin/python \
  --output "$OUT/verified"
```

Output directories must be new. The verifier defaults to all 9,996 test targets;
`--limit` is only for a smaller smoke check, marked as incomplete in the audit.
It fails if quantized perplexity does not beat the training-only unigram
baseline, or if any asserted conversion comparison differs.

The verifier checks checkpoint and frozen-file hashes before evaluation.
Native checkpoint restore additionally checks the training binary identity;
retrain when using a different build. Frozen export itself does not need the
original training executable. The scripts print progress and write reports,
full native/SDK arrays, concrete IR replay receipts, and a compact per-token
audit with target positions, negative log likelihoods, predictions and spike
difference counts.

```sh
B2_TRAIN_RUNNER=/path/to/b2-train \
B2_RUNNER=/path/to/b2-runner \
B2_SPINNAKER2_PYTHON=/path/to/spinnaker2-environment/bin/python \
python -m pytest -q \
  brian2-rust/tests/test_character_language_model.py \
  brian2-rust/tests/test_native_lif_conversion.py \
  brian2-rust/tests/test_spinnaker2_b2ir.py
```
