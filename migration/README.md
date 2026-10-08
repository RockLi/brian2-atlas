# Atlas migration provenance

Atlas starts from official Brian2 commit `27b5431168cf9959f0c27894cd4d92cdc6ad5c31`.
The legacy development checkout was captured at `81eb571d78292a0e24a8e79980ea0d5c26e8a48c`, including working-tree and untracked changes. File hashes, not the HEAD alone, identify the imported contents. Its original upstream merge base was `cf0b25cce13b620aac740873e88ba91061a70275`.

Stage records describe file origins, transformations and validation. Tests for later backend stages are not implied by a passing frontend stage. The migration is in progress; packaging, backend integration and release preparation remain pending.

The fixed upstream baseline passed the selected NumPy frontend regression: 277 passed, 7 skipped, 7 deselected. See `upstream-baseline.json` for the exact suite scope and report identity. Native backend validation remains separate.

The indirect-index port passed 194 tests (1 skipped, 3 deselected) across the code generation, synapse and monitor gate. Two additional maintained regressions execute nested references through real NumPy and compiled Cython code objects. See `p1b-indirect-indices.json`; `frontend-source-commits.json` records legacy history separately from captured working-tree contents.

Five independent upstream-example compatibility changes are recorded in `example-compatibility.json`: NumPy 2 profiling/array membership, Matplotlib artist input, importable Gaussian connection generation, and mathematical docstrings. All five files parse; all three connection generators were executed at 32 neurons. Full scientific runs and performance sweeps were not repeated. Backend-dependent example ports remain in progress.
