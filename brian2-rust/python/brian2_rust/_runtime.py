"""Locate native executables and writable paths in source and installed layouts."""
from __future__ import annotations

import os
from pathlib import Path
import tempfile

PACKAGE_ROOT = Path(__file__).resolve().parent


def source_root():
    candidate = PACKAGE_ROOT.parents[1]
    if ((candidate / "Cargo.toml").is_file()
            and candidate / "python" / "brian2_rust" == PACKAGE_ROOT):
        return candidate
    return None


def runtime_root():
    """Working directory carrying the pinned toolchain, never an output path."""
    return source_root() or PACKAGE_ROOT


def cargo_target():
    source = source_root()
    if source is None:
        raise RuntimeError("Installed Atlas wheels contain native executables; no source build directory exists")
    configured = os.environ.get("CARGO_TARGET_DIR")
    if configured:
        target = Path(configured).expanduser()
        return target if target.is_absolute() else source / target
    return source / "target"


def executable_path(name, explicit=None):
    if name not in {"b2-runner", "b2-train"}:
        raise ValueError(f"Unknown Atlas executable: {name}")
    configured = explicit if explicit is not None else os.environ.get(
        "B2_RUNNER" if name == "b2-runner" else "B2_TRAIN_RUNNER")
    if configured is not None:
        return Path(configured).expanduser().resolve()
    filename = name + (".exe" if os.name == "nt" else "")
    bundled = PACKAGE_ROOT / "_bin" / filename
    if bundled.is_file() or source_root() is None:
        return bundled
    return cargo_target() / "release" / filename


def cache_root():
    base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")).expanduser()
    return base / "brian2-atlas"


def create_run_directory():
    base = os.environ.get("BRIAN2_ATLAS_OUTPUT_DIR")
    if base is not None:
        base = Path(base).expanduser().resolve()
        base.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="brian2-atlas-", dir=base))
