# Publication review — 2026-10-09

Status: **the rewritten public-history candidate passed the scoped checks below**.
The user authorized history cleanup, citation updates and force-push. Original
histories and evidence are preserved in verified private backups, outside this
public tree. Publication uses explicit remote leases to avoid overwriting new work.

## History and credentials

The complete reachable histories were checked with Gitleaks 8.30.1, including
merge diffs and archives, without project allowlists or inline allow comments.
An independent all-object pass examined commit/tag messages, every blob and
recursive tar/zip/gzip contents. No archive members were skipped.

| Candidate | Reachable commits at scan | Unique blobs | Home/volume path records after cleanup | Confirmed credentials |
| --- | ---: | ---: | ---: | ---: |
| brian2-atlas | 7,281 | 16,608 | 0 | 0 |
| brian2-atlas-preprint | 26 | 2,147 | 0 | 0 |

The engine's remaining generic Linux/Windows path examples are byte-identical
upstream Brian2 objects. The upstream master commit and all upstream tag refs
are unchanged. All matched path patterns in the evidence history were removed.
The evidence scanner's 11 generic-key candidates remain verified source SHA-256
entries; each value was matched against actual original source bytes. No private
keys or credential-file candidates were found. Pattern scans are not proof of
absence, and existing clones, unreachable hosting objects and hosting caches
are outside the local cleanup's deletion guarantees.

[Public commit mapping](public-commit-map.json) resolves old source identities.
[Public export mapping](public-path-redaction.json) binds original evidence
hashes to redacted copies. Original hash fields in historical records have not
been silently relabeled. Synthetic paths are descriptive placeholders; supply
actual locations when running archived scripts.

## Compatibility and paper checks

All 428 tracked runtime/frontend files compared byte-identically before and
after path cleanup. The affected MPI launch tests passed 75 cases after restoring
the same-source local Cython build prerequisites. The initial missing-extension
attempt is retained privately and is not counted as a successful test run.

The manuscript build passed its numerical and structural assertions using strict
public-export hash/size verification. All 34 retained V3 inputs were verified,
and modified copies were rejected. Active citations and the pinned PD14 Cargo
revision were updated to the rewritten history. The source program itself is
unchanged. PDF exports replace local file annotations with pinned repository
links so that reading copies do not reveal build-machine paths.

The dated Figure 2 maintenance sentence was removed in both languages. The
memory clause identifies the 660 GiB host limit for the 656.436 GiB other-host
peak. Measurement values and figure geometry are unchanged.

## License and citation scope

Original Atlas engine and compendium code are Apache-2.0; original paper, data
and figures are CC BY 4.0. Brian2 and other third-party materials retain their
terms. The original Brian2 license is preserved byte-for-byte. The combined
engine checkout is not Apache-only; see LICENSE_SCOPE.md. Both CITATION.cff files
passed the official 1.2.0 schema, and generated Python distribution metadata
includes the combined license expression and all license/notice files.

The [JSON report](publication-audit-20261009.json) distinguishes original audit
heads from the rewritten candidates. Subsequent source/QA commits are checked
as a separate final delta. This review does not publish a package, assign a
paper DOI, or qualify every platform/backend.
