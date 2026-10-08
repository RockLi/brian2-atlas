# Atlas migration provenance

Atlas starts from official Brian2 commit `27b5431168cf9959f0c27894cd4d92cdc6ad5c31`.
The legacy development checkout was captured at `81eb571d78292a0e24a8e79980ea0d5c26e8a48c`, including working-tree and untracked changes. File hashes, not the HEAD alone, identify the imported contents. Its original upstream merge base was `cf0b25cce13b620aac740873e88ba91061a70275`.

Stage records describe file origins, transformations and validation. Tests for later backend stages are not implied by a passing frontend stage. The migration is in progress; packaging, backend integration and release preparation remain pending.
