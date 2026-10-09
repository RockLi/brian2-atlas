# Publication review — 2026-10-09

Status: **license/citation and requested editorial updates completed; absolute
path removal from published history remains unresolved**. This is not a claim
that the repositories are ready for unrestricted public redistribution.

## Credential and history review

Gitleaks 8.30.1 used its built-in rules without repository allowlists or inline
allow comments. All local refs were included, with merge diffs in the engine
scan and archive traversal enabled. Origin was fetched and the audited branch
heads matched. The independent object pass examined all reachable commit/tag
messages and blobs, recursively including tar/zip archives and gzip streams.
No archive members were skipped by that pass.

| Repository | Reachable commits | Unique blobs | Actual credential findings | Current blobs with user-home or volume paths |
| --- | ---: | ---: | ---: | ---: |
| brian2-atlas | 7,280 | 16,598 | 0 | 11 |
| brian2-atlas-preprint | 25 | 2,128 | 0 | 110 |

The evidence repository's 11 generic-key candidates are file-hash entries.
Each value was verified as SHA-256 of actual source bytes in the scanned
objects/archives. They were not suppressed before scanning. No private-key
headers or credential-file candidates were found. Pattern scans cannot prove
absence of secrets; unreachable hosting objects, reflogs and hosting caches
are outside this review. No credentials were sent to a service for testing.

The path counts refer to Git blobs, including archive containers. Historical
object versions containing home/volume paths number 15 in the engine and 123
in the evidence repository. Additional absolute paths include generic Linux
test/CI examples; those require classification, not blanket replacement.
The linked JSON lists affected repository-relative filenames without disclosing
the matched host paths or identifier strings.

A normal cleanup commit would leave earlier path strings retrievable. Complete
removal requires a coordinated history rewrite or a separate sanitized
publication history, followed by commit-citation updates and integrity checks.
Original experimental evidence must be preserved privately with its original
hashes before any public-export redaction; a redacted artifact needs a new hash
and must not be represented as byte-identical original evidence. No history
rewrite or force-push has been performed.

## License and citation scope

Original Atlas engine/code is offered under Apache-2.0. Original compendium
code is Apache-2.0, and original paper/data/figures are CC BY 4.0. Imported Brian2
and third-party materials retain their terms. See the root LICENSE_SCOPE.md;
the combined Brian2 checkout cannot be described as Apache-only.

Both CITATION.cff files pass the official 1.2.0 schema. The original Brian2
license was preserved byte-for-byte, and upstream citation metadata is retained
in the engine repository. Python distribution metadata was generated and checked
for the combined license expression and bundled license/notice files. A full
new release wheel and hardware qualification are separate release work.

## Manuscript changes

The duplicate dated Figure 2 regression sentence was removed in both languages;
the supplementary method retains the comparison and command without test-log
noise. The peak-memory clause now explicitly identifies the 660 GiB host limit
for the 656.436 GiB maximum other-host peak, verified against the per-host data.
Scientific inputs, measured values and runtime algorithms are unchanged.

The [machine-readable report](publication-audit-20261009.json) records the
audited revisions and precise scope. New review/documentation commits follow
those audited base revisions; their changed-file scan is recorded separately.
