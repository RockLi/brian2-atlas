"""Repair a Linux wheel, refresh native hashes, and regenerate wheel RECORD."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def refresh_native_hashes(package: Path) -> None:
    manifest = package / "brian2_rust" / "_bin" / "build.json"
    provenance = json.loads(manifest.read_text())
    if provenance["schema"] != "atlas-native-build-v1":
        raise ValueError("Unknown native build provenance schema")
    previous = provenance["binaries"]
    if set(previous) != {"b2-runner", "b2-train"}:
        raise ValueError("Expected both Linux native executables")
    current = {
        name: hashlib.sha256((manifest.parent / name).read_bytes()).hexdigest()
        for name in previous
    }
    # Preserve the original build hashes when auditwheel changes binary bytes.
    if current != previous:
        provenance["pre_repair_binaries"] = previous
        provenance["binaries"] = current
        manifest.write_text(json.dumps(provenance, indent=2) + "\n")


def main() -> None:
    wheel, destination = map(Path, sys.argv[1:])
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="atlas-wheel-repair-") as temporary:
        root = Path(temporary)
        subprocess.run([
            "auditwheel", "repair", "--plat", "manylinux_2_28_x86_64",
            "--wheel-dir", str(root / "repaired"), str(wheel),
        ], check=True)
        repaired, = (root / "repaired").glob("*.whl")
        subprocess.run([
            sys.executable, "-m", "wheel", "unpack", str(repaired),
            "--dest", str(root / "unpacked"),
        ], check=True)
        package, = (root / "unpacked").iterdir()
        refresh_native_hashes(package)
        subprocess.run([
            sys.executable, "-m", "wheel", "pack", str(package),
            "--dest-dir", str(destination),
        ], check=True)


if __name__ == "__main__":
    main()
