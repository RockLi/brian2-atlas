# Brian2 Atlas migration and release preparation

## Accepted scope and source identity

The migration is complete for the **user-approved third frozen product increment**, captured at **2026-10-08 16:10:08 UTC** (2026-10-09 00:10:08 Singapore time). The user explicitly assigned subsequent original-checkout development to a separate future port. This is release preparation for development version `0.1.0.dev0`; no package publication, formal release or paper submission was performed.

- Product code checkpoint: `243b9e47b83a48d1f38381b04ad8315df168a4b4`, on Atlas `dev`.
- Fixed official Brian2 baseline: `27b5431168cf9959f0c27894cd4d92cdc6ad5c31`; complete upstream ancestry is retained.
- Legacy capture HEAD: `81eb571d78292a0e24a8e79980ea0d5c26e8a48c`. Captured uncommitted contents are identified by file hashes and immutable supplements, not by this HEAD alone.
- Product: [RockLi/brian2-atlas](https://github.com/RockLi/brian2-atlas), default `dev`; `origin` is the product repository and `upstream` is `brian-team/brian2`.
- Research: [RockLi/brian2-atlas-preprint](https://github.com/RockLi/brian2-atlas-preprint), default `main`. Documentation/index commits may follow the product code checkpoint without changing its accepted runtime source.

[The explicit cutoff record](migration-cutoff.json) includes the four changed implementation paths and one new test observed after this cutoff. They are identified for future porting and are not counted as accepted product changes here. The original checkout remains available for ongoing development; migration checkouts, builds, environments, caches and large archives were placed on T7.

## Complete input disposition

The preprint [final source map](https://github.com/RockLi/brian2-atlas-preprint/blob/main/migration/migration-cutoff-disposition.json) covers **59,821 paths**, including the original 59,309-path inventory, three product captures and the later research supplement. It has **zero unresolved dispositions**. The compressed per-path map retains original identities, later captured versions, current target commits/hashes, archive members and explicit generated-file exclusions. It verifies 2,645 current baseline destination references and 31 product paths with later captures. Seven upstream workflow destination references were corrected to their actual preserved location, `.github/upstream-workflows/`.

The initial product-candidate set contained 1,806 files: 964 primary Atlas destinations and 842 primary preprint destinations. All product phases are committed, with stage-specific adaptations recorded. New official upstream fixes were preserved rather than replacing the upstream tree with the older development checkout. The implementation retains `brian2-rust` and `brian2_rust`; package name, metadata, source/wheel builds, executable wrappers, optional dependencies, CI and user examples are in the product repository. Upstream author and license files are retained.

The later research supplement preserves 499 files: 496 development evidence files, one updated historical development log, a PDF QA record already present in preprint, and one developer-documentation file already identical in Atlas. All eight preservation archives were rehashed successfully at final acceptance: **8,068,338,516 bytes**. They remain local T7 archives; the preprint indices state locations, sizes and SHA-256 values. Generated exclusions are recorded as exclusions, not as archived evidence.

## Validation evidence

Counts below refer to the exact stage sources and scopes in the linked records. Later integration and artifact checks establish the accepted cutoff; a stage result is not a claim that every test was rerun on every backend at the final commit.

| Area | Verified scope and evidence |
| --- | --- |
| Official/frontend | Fixed baseline: 277 passed, 7 skipped, 7 deselected. Indirect-index gate: 194 passed, 1 skipped, 3 deselected; two real NumPy/Cython runtime regressions. [Baseline](upstream-baseline.json), [frontend port](p1b-indirect-indices.json). |
| CPU/B2IR/AOT | Rust core, selected core/extended/supplemental suites, actual runtime and compiler stress; original failed cases retain matched successful reruns. Three platform cache-advice skips remain explicit. [CPU record](p2-cpu-port.json). |
| GPU/browser | Actual NVIDIA L4 simulation, Apple Metal, full-library Node WASM and real Chrome Worker/WebGPU checks. Node: 40 passes; browser Worker: 9 cases; equation and scale scopes are recorded separately. [GPU/WASM record](p3-gpu-wasm-port.json). |
| MPI | Actual same-host multi-rank parity, continuation and coordinated failures: 17 isolated cases plus 7 canonical checks. [MPI record](p4-mpi-port.json). |
| Native training | CPU/Metal/checkpoints, captured callbacks, selected 18 CUDA training cases and 12 CUDA clock/checkpoint cases including same-host MPI. Timed-out partial runs remain explicitly distinguished from clean reruns. [Training record](p5-native-training-port.json). |
| Increment 2 | 1,059 isolated passes, 524 explicit actual-CUDA skips; 43 canonical integration passes. [Record](concurrent-training-increment-2.json). |
| Final increment 3 | 380 isolated passes, 184 explicit actual-CUDA skips; 52 final canonical integration passes, no skips/failures. [Record](concurrent-training-increment-3.json). |
| Product examples | 37 isolated core passes/one optional SBI skip, 46 official-example passes, seven real CLI flows, 30 canonical passes/one SBI skip. Includes native/MPI fresh-process checkpoint continuation. [Examples record](p6-user-example-port.json). |
| Distribution | Direct-source, wheel and isolated-sdist routes passed earlier packaging gates. Final audited sdist-to-wheel and fresh source-user-install passed CPU/reference/AOT, Metal training and fresh-process checkpoint checks, without source import or binary environment overrides. Installed package files remained unchanged. Twelve additional installed-wheel CPU/Metal tests checked new whole-caller forward/restore and independent gradients. [Final acceptance](final-distribution-acceptance.json). |
| Preprint | Clean retained-data build reproduced 10 figures, 7 tables and 16 references; 44-page PDF layout/links were checked. [Preprint build records](https://github.com/RockLi/brian2-atlas-preprint/tree/main/migration). |
| New reproduction | Pinned PD14 Cargo consumer at Atlas `6677a5bafd3b703ab56b6ed176e9aad70f4638cd`; topology-shape test and repeated fresh-process bounded simulation passed: 772 neurons, 29,885 synapses, 91 delivered events, spike counts `[1,0,1,0,1,0,0,0]`. [Reproduction command](https://github.com/RockLi/brian2-atlas-preprint/tree/main/experiments/reproduction/pd14). |

## Final distribution candidates

Both artifacts are retained under `/atlas-storage/0001/build/dist-candidate-v5/` on the T7-backed workspace:

| Artifact | SHA-256 |
| --- | --- |
| `brian2_atlas-0.1.0.dev0.tar.gz` | `28a5d7f179da3e7375a38ede496bdaac713746ab1f7dd30c6f870b8bf0b4f48e` |
| `brian2_atlas-0.1.0.dev0-cp314-cp314-macosx_26_0_arm64.whl` | `eae2c8554665aceb004c10270c7c8f74b772c4dca9055b3612fc2f55ef4afeeb` |

The sdist contains 1,137 files. All 170 required native/Python/runtime/build/license inputs match the committed cutoff, and all 117 packaged backend source/resource files were compared directly with the wheel. The frontend, build configuration and README did not change between the candidate base and accepted code commit. The final source installer recorded that exact commit with an empty source diff and worked through relative paths containing spaces. The installed-training test's first collection attempt lacked its external SciPy oracle dependency; that unsuccessful attempt is retained, and the dependency-complete rerun passed all 12 cases.

## Historical reproducibility and remaining publication work

The preprint preserves historical results and actual source identities separately from current product validation. A verified Git bundle retains nine recorded experiment revisions. All 626 files in the reviewed source manifest were restored exactly, including three recovered uncommitted Git objects. A portable archive tool restored and verified 22 selected historical evidence files.

Six intermediate contents from two old FlyWire MNIST development runs remain unavailable after snapshot/Git/worktree and targeted T7 backup searches. All four affected latest source files are preserved exactly. Per the user's direction, latest implementations are maintained and this pre-existing historical limitation is documented rather than blocking migration. [Historical restoration guide](https://github.com/RockLi/brian2-atlas-preprint/blob/main/archives/HISTORICAL_SOURCES.md). Other legacy campaign entry points retain their original paths/environment assumptions; the new PD14 workflow is the separately validated portable execution route. Neither the archive restoration nor that bounded PD14 run claims new full-scale scientific results.

The remaining release decisions/work are explicit:

1. Port development after the approved cutoff separately, with its own frozen inputs and tests.
2. Inspect hosted Linux/macOS CI results and qualify additional intended platforms/hardware. Hosted passes, Windows, cross-host MPI and multiple GPUs are not established here; newer skipped CUDA variants are not passes.
3. Publish selected preservation archives/data at durable public URLs if public reproduction is required; current archive indices describe local availability only.
4. Choose the public release version/tag, publish distribution artifacts and prepare a formal release separately. Paper submission is also separate.

The original goal's repository split, approved-scope product migration, installation/reproduction preparation and evidence handoff are complete. These deferred publication and post-cutoff development tasks are not silently included in the validation claims above.
