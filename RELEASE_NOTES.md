# Brian2 Atlas 0.1.0

First research software release, paired with the
[bioRxiv v1 evidence archive](https://github.com/RockLi/brian2-atlas-preprint/tree/biorxiv-v1).

- Brian2 modelling frontend with the public `brian2_atlas` / `atlas` Device API.
- AtlasIR and independent CPU/reference, AOT, CUDA, Metal, MPI and browser execution.
- Native training and checkpoint workflows. Atlas does not require Brian2CUDA,
  Brian2GeNN or GeNN as runtime dependencies.

The accepted implementation uses the third frozen migration increment. Version
and release metadata changed after acceptance; scientific algorithms and inputs
did not. Recorded experiments retain their measured revisions.

See [validation and support scope](README.md#validation-and-support-scope) and
[retained acceptance records](migration/README.md) for tested platforms and limits.
This research release does not imply validation of every device or model, PyPI
publication, a bioRxiv submission, or an assigned DOI.

Original Atlas code uses Apache-2.0. Brian2 and derived material retain CeCILL-2.1
and upstream component notices; see [license scope](LICENSE_SCOPE.md).

`v0.1.0` is an immutable annotated release tag. `main` holds the frozen release
snapshot; ongoing work belongs on `dev`.
