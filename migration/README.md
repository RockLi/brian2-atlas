# Atlas migration provenance

Atlas starts from official Brian2 commit `27b5431168cf9959f0c27894cd4d92cdc6ad5c31`.
The legacy development checkout was captured at `81eb571d78292a0e24a8e79980ea0d5c26e8a48c`, including working-tree and untracked changes. File hashes, not the HEAD alone, identify the imported contents. Its original upstream merge base was `cf0b25cce13b620aac740873e88ba91061a70275`.

Stage records describe file origins, transformations and validation. Tests for later backend stages are not implied by a passing frontend stage. The frontend, CPU, GPU/WASM, MPI and native training stages are integrated. Distribution packaging has independent installation gates. The migration remains in progress until final source reconciliation, concurrent training increments and migration reporting are complete.

The fixed upstream baseline passed the selected NumPy frontend regression: 277 passed, 7 skipped, 7 deselected. See `upstream-baseline.json` for the exact suite scope and report identity. Native backend validation remains separate.

The indirect-index port passed 194 tests (1 skipped, 3 deselected) across the code generation, synapse and monitor gate. Two additional maintained regressions execute nested references through real NumPy and compiled Cython code objects. See `p1b-indirect-indices.json`; `frontend-source-commits.json` records legacy history separately from captured working-tree contents.

Five independent upstream-example compatibility changes are recorded in `example-compatibility.json`: NumPy 2 profiling/array membership, Matplotlib artist input, importable Gaussian connection generation, and mathematical docstrings. All five files parse; all three connection generators were executed at 32 neurons. Full scientific runs and performance sweeps were not repeated. Backend-dependent example ports and their scoped validation are recorded under Product examples and tools below.

The CPU/B2IR/AOT port is recorded in `p2-cpu-port.json`. Its 27 test files were covered by the core, extended and supplemental gates, with named focused reruns resolving the original failures. Three macOS cache-advice skips remain explicit. The portable `dev/atlas/run_backend.py` command and Linux CPU CI configuration are included; hosted CI results are recorded separately from local acceptance.

### GPU/WASM port

`p3-gpu-wasm-port.json` records this port, its source hashes, the WASM address-width and browser-authoring fixes, and the actual CUDA/Metal/Node/Chrome verification scope. Source provenance and hardware scope remain explicit; native training has its own record below.

### MPI and native training

[`p4-mpi-port.json`](p4-mpi-port.json) covers actual same-host MPI execution,
reference parity, continuation and collective failure exits.
[`p5-native-training-port.json`](p5-native-training-port.json) records the full
training port, captured concurrent increments, numerical/checkpoint verification,
CUDA shader repair and the exact boundary between executed and skipped cases.
Intermediate failed or timed-out runs are retained and matched to focused
resolution evidence; they are not relabelled as clean full-suite passes.

### Product examples and tools

[`p6-user-example-port.json`](p6-user-example-port.json) records 43 source inputs,
37 isolated core passes (one SBI dependency skip), seven actual CLI flows,
46 official-example regression passes and 30 canonical entrypoint passes
(one SBI dependency skip). It includes six backend-compatible upstream example
ports, native training/MPI checkpoint examples, reusable analysis utilities and
the Brunel artifact sweep dependency. Captured cache/build ignore rules are
recorded separately in [`source-ignore-port.json`](source-ignore-port.json).

### Distribution preparation

[`p7-packaging-port.json`](p7-packaging-port.json) records the package/runtime
layout, isolated-build repair, native-binary provenance and installed checks.
The distribution is `brian2-atlas` version `0.1.0.dev0`; Python imports remain
`brian2` and `brian2_rust`. Source, wheel and sdist acceptance currently covers
macOS arm64/Python 3.14. Hosted Linux/macOS CI is configured separately; no hosted
pass or Windows distribution support is implied by these local results.

### Remaining migration audit

The complete initial 59,309-path mapping is preserved in the preprint repository,
including explicit generated-file exclusions. Later evidence is captured in
separate checksummed supplements. Final concurrent source integration and
distribution checks must complete before the full migration is declared complete.
The preprint repository retains historical source identities; its new
PD14 end-to-end reproduction pins Atlas commit `6677a5bafd3b703ab56b6ed176e9aad70f4638cd`
and has passed bounded build-to-simulation validation. Publishing a
package, creating a formal release and submitting the paper are outside this
migration's execution scope.

### Later training increments

[`concurrent-training-increment-2.json`](concurrent-training-increment-2.json)
records batch capture lifetimes, private temporaries, masked operand copies and
emission-addressed random draws. The full isolated gate passed 1,059 cases; 524
actual-CUDA variants were skipped locally. Integration in the product checkout
passed 43 CPU/Metal/local-MPI and compiler checks with no failures or skips.
Earlier CUDA acceptance does not imply execution of these newer skipped variants.
