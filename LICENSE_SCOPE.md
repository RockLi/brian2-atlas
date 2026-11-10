# License scope

The original Atlas engine work by Xinjun Li is available under
[Apache-2.0](LICENSE). This checkout also contains Brian2 and is **not an
Apache-only distribution**. The following boundaries apply to the current source
tree; a new license notice does not erase notices in earlier versions.

| Material | Applicable terms |
| --- | --- |
| Original Atlas engine code in `brian2-rust/`, its original Python interfaces, tests and tools; original `dev/atlas/` tooling and `_atlas_build.py` | Apache-2.0 for the author's original contributions, subject to the exceptions below |
| Upstream Brian2 files, Brian2-derived modifications, and any copied upstream material | CeCILL-2.1 and the component notices preserved in `LICENSES/Brian2-LICENSE`; the Apache grant does not replace these terms |
| Third-party dependencies, copied examples, model sources, data and images | Their existing notices and source-specific terms; no relicensing is granted here |

Brian2's license addresses contributions, internal modules and distribution of
modified software in Articles 1 and 5.3.2. This repository does not claim that
importing an Apache-licensed component removes those obligations for a combined
Brian2 distribution. Package metadata therefore lists both licenses. The native
Rust crate identifies the original Atlas engine license as Apache-2.0; dependency
licenses remain separate. Existing grants for earlier revisions remain intact.

`LICENSES/Brian2-LICENSE`, `AUTHORS`, `CONTRIBUTORS` and `UPSTREAM_CITATION.cff`
retain the upstream notices and citation metadata. Cite Atlas using
`CITATION.cff`, and cite Brian2 when its modelling frontend is used.

This repository contains the implementation and regression tests. The companion
preprint repository contains manuscript, figures, experimental code and evidence,
with its own content-specific license scope. Development branch availability is
not a package release or qualification of every supported backend/platform.
