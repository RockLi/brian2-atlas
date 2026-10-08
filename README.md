# Brian2 Atlas

Brian2 Atlas extends [Brian2](https://github.com/brian-team/brian2) with heterogeneous and distributed neural simulation and native training. This repository is being assembled through independently validated ports from the research checkout.

## Current migration checkpoint

The committed implementation currently contains the upstream Brian2 frontend and the first verified Atlas function-semantics port. Native CPU, GPU, WASM, MPI and training backends are being ported in dependency order. See [migration provenance](migration/README.md) for the fixed upstream commit, source hashes and stage validation. Release packaging and the complete support matrix remain in progress.

[Paper, experimental records and reproduction materials](https://github.com/RockLi/brian2-atlas-preprint) are maintained in a separate repository. Historical results keep their actual experimental code identities.

## Development

Use Python 3.12 or newer and an isolated environment:

```sh
python -m pip install -e '.[test]' scipy matplotlib
python dev/atlas/run_frontend.py
```

The development command validates selected upstream frontend suites using the NumPy target; pass `--target cython` or explicit pytest paths to select a different scope. CI checks Python 3.12 and 3.14. Current package metadata is inherited from Brian2 until the distribution port is complete.

Upstream authorship, contributors and the CeCILL 2.1 license are retained in `AUTHORS`, `CONTRIBUTORS` and `LICENSE`. The [original upstream README](README.upstream.md) preserves the original project description and links.
