"""Build both native engines into platform wheels from the locked Rust sources."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from setuptools.command.build_py import build_py


class AtlasBuildPy(build_py):
    def run(self):
        super().run()
        root = Path(__file__).resolve().parent
        backend = root / "brian2-rust"
        env = os.environ.copy()
        env["RUSTUP_TOOLCHAIN"] = "1.98.1"
        target = Path(env.get("CARGO_TARGET_DIR", root / "build" / "atlas-cargo")).resolve()
        version = subprocess.check_output(["cargo", "--version"], cwd=backend, env=env, text=True).strip()
        if not version.startswith("cargo 1.98.1 "):
            raise RuntimeError(f"Atlas requires Cargo 1.98.1; got {version}")
        subprocess.run(["cargo", "build", "--release", "--locked", "--manifest-path",
                        str(backend / "Cargo.toml"), "--target-dir", str(target),
                        "--bin", "b2-runner", "--bin", "b2-train"],
                       cwd=backend, env=env, check=True)
        destination = Path(self.build_lib) / "brian2_rust" / "_bin"
        destination.mkdir(parents=True, exist_ok=True)
        binaries = {}
        for name in ("b2-runner", "b2-train"):
            filename = name + (".exe" if os.name == "nt" else "")
            shutil.copy2(target / "release" / filename, destination / filename)
            binaries[filename] = hashlib.sha256((destination / filename).read_bytes()).hexdigest()
        inputs = sorted((backend / "src").rglob("*.rs"))
        inputs += [backend / name for name in ("Cargo.toml", "Cargo.lock", "rust-toolchain.toml")]
        provenance = {"schema": "atlas-native-build-v1", "cargo": version,
                      "binaries": binaries,
                      "sources": {str(p.relative_to(backend)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in inputs}}
        (destination / "build.json").write_text(json.dumps(provenance, indent=2) + "\n")
