"""Repair a Linux/macOS wheel, refresh native hashes, and regenerate RECORD."""
from __future__ import annotations

import argparse
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
        raise ValueError("Expected both native executables")
    current = {
        name: hashlib.sha256((manifest.parent / name).read_bytes()).hexdigest()
        for name in previous
    }
    # Preserve original build hashes when platform repair changes binary bytes.
    if current != previous:
        provenance["pre_repair_binaries"] = previous
        provenance["binaries"] = current
        manifest.write_text(json.dumps(provenance, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--platform", choices=("linux", "macos"), default="linux")
    parser.add_argument("--arch", choices=("arm64", "x86_64"))
    args = parser.parse_args()
    if args.platform == "macos" and not args.arch:
        parser.error("macOS repair requires --arch")
    wheel = args.wheel
    destination = args.destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="atlas-wheel-repair-") as temporary:
        root = Path(temporary)
        if args.platform == "macos":
            command = ["delocate-wheel", "--require-archs", args.arch,
                       "--wheel-dir", str(root / "repaired"), str(wheel)]
        else:
            command = ["auditwheel", "repair", "--plat", "manylinux_2_28_x86_64",
                       "--wheel-dir", str(root / "repaired"), str(wheel)]
        subprocess.run(command, check=True)
        repaired, = (root / "repaired").glob("*.whl")
        subprocess.run([
            sys.executable, "-m", "wheel", "unpack", str(repaired),
            "--dest", str(root / "unpacked"),
        ], check=True)
        package, = (root / "unpacked").iterdir()
        if args.platform == "macos":
            # The bundled Rust commands must match the native runner/wheel too.
            for name in ("b2-runner", "b2-train"):
                subprocess.run(["lipo", str(package / "brian2_rust" / "_bin" / name),
                                "-verify_arch", args.arch], check=True)
        refresh_native_hashes(package)
        subprocess.run([
            sys.executable, "-m", "wheel", "pack", str(package),
            "--dest-dir", str(destination),
        ], check=True)


if __name__ == "__main__":
    main()
