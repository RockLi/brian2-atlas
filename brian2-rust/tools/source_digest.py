#!/usr/bin/env python3
"""Compute a reproducible digest for the sources used by B2IR validation.

Build products, caches, benchmark output, native extension binaries, generated
version metadata, and the freeze report itself are intentionally excluded. The
digest can therefore be compared across machines even when their Python and
Rust build artifacts differ.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


INCLUDED_SUFFIXES = {
    ".c",
    ".h",
    ".json",
    ".md",
    ".py",
    ".pyx",
    ".rs",
    ".toml",
}
INCLUDED_NAMES = {"Cargo.lock", "LICENSE"}
EXCLUDED_PARTS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "output",
    "target",
}
EXCLUDED_NAMES = {"B2IR_FREEZE_REPORT.md", "_version.py"}


def selected_files(repository_root: Path) -> list[Path]:
    roots = (repository_root / "brian2", repository_root / "brian2-rust")
    files: list[Path] = []
    for root in roots:
        for path in root.rglob("*"):
            relative = path.relative_to(repository_root)
            if not path.is_file():
                continue
            if any(part in EXCLUDED_PARTS for part in relative.parts):
                continue
            if path.name in EXCLUDED_NAMES:
                continue
            if path.suffix not in INCLUDED_SUFFIXES and path.name not in INCLUDED_NAMES:
                continue
            files.append(path)
    return sorted(files, key=lambda path: path.relative_to(repository_root).as_posix())


def source_digest(repository_root: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    files = selected_files(repository_root)
    for path in files:
        relative = path.relative_to(repository_root).as_posix().encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest(), len(files)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        action="store_true",
        help="print the relative path and SHA-256 of every selected file",
    )
    parser.add_argument(
        "repository_root",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    args = parser.parse_args()
    root = args.repository_root.resolve()
    if args.manifest:
        for path in selected_files(root):
            relative = path.relative_to(root).as_posix()
            print(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {relative}")
        return
    digest, count = source_digest(root)
    print(f"sha256={digest} files={count}")


if __name__ == "__main__":
    main()
