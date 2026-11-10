"""Bounded reader for monitor chunks produced by ``rust_standalone``."""

import hashlib
import json
from pathlib import Path

import numpy as np


class MonitorStream:
    """Validated, lazy view of an on-disk monitor stream.

    Each iteration loads one bounded ``.npz`` chunk and returns independent
    NumPy arrays. StateMonitor values use ``(time, recorded_index)`` layout.
    """

    def __init__(self, path):
        root = Path(path).expanduser().resolve()
        if root.is_file():
            root = root.parent
        try:
            manifest = json.loads((root / "manifest.json").read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"invalid monitor stream: {error}") from error
        if (manifest.get("schema") != "b2-monitor-stream-v1" or
                not isinstance(manifest.get("monitors"), dict) or
                not isinstance(manifest.get("chunks"), list)):
            raise RuntimeError("invalid monitor stream manifest")
        self.path = root
        self.manifest = manifest

    @property
    def monitors(self):
        return tuple(self.manifest["monitors"])

    @property
    def complete(self):
        return self.manifest.get("complete") is True

    def iter_chunks(self, monitor):
        """Yield one dictionary of copied arrays for each completed chunk."""
        if monitor not in self.manifest["monitors"]:
            raise KeyError(f"unknown streamed monitor {monitor!r}")
        for expected, chunk in enumerate(self.manifest["chunks"], start=1):
            if (chunk.get("index") != expected or
                    Path(chunk.get("directory", "")).name !=
                    chunk.get("directory") or
                    set(chunk.get("monitors", {})) !=
                    set(self.manifest["monitors"])):
                raise RuntimeError("invalid monitor stream chunk entry")
            stored_file = chunk["monitors"][monitor]
            if (set(stored_file) != {"file", "bytes", "sha256"} or
                    Path(stored_file["file"]).name != stored_file["file"]):
                raise RuntimeError("invalid monitor stream file entry")
            source = self.path / chunk["directory"] / stored_file["file"]
            try:
                if source.stat().st_size != stored_file["bytes"]:
                    raise ValueError("byte length mismatch")
                with source.open("rb") as raw:
                    digest = hashlib.file_digest(raw, "sha256").hexdigest()
                if digest != stored_file["sha256"]:
                    raise ValueError("sha256 mismatch")
                with np.load(source, allow_pickle=False) as stored:
                    arrays = {name: stored[name].copy()
                              for name in stored.files}
            except (OSError, ValueError) as error:
                raise RuntimeError(
                    f"invalid monitor stream chunk {expected}: {error}") from error
            yield {
                "index": expected,
                "start_seconds": chunk["start_seconds"],
                "end_seconds": chunk["end_seconds"],
                "arrays": arrays,
            }


def open_monitor_stream(path):
    """Open a monitor stream without loading any chunk arrays."""
    return MonitorStream(path)


def iter_monitor_chunks(path, monitor):
    """Convenience iterator over one named Monitor's bounded chunks."""
    return open_monitor_stream(path).iter_chunks(monitor)


__all__ = ["MonitorStream", "open_monitor_stream", "iter_monitor_chunks"]
