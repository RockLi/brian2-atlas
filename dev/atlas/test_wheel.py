"""Run the installed flow outside the source tree and retain a JSON report."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import tomllib


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    version = tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"]
    report = args.report.resolve() if args.report else None
    with tempfile.TemporaryDirectory(prefix="atlas-wheel-test-") as directory:
        work = Path(directory)
        env = os.environ.copy()
        for name in ("PYTHONPATH", "B2_RUNNER", "B2_TRAIN_RUNNER",
                     "BRIAN2_ATLAS_OUTPUT_DIR", "RUSTUP_TOOLCHAIN"):
            env.pop(name, None)
        env.update(TMPDIR=str(work), XDG_CACHE_HOME=str(work / "cache"),
                   PYTHONDONTWRITEBYTECODE="1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
        report = report or work / "installed.json"
        subprocess.run([
            sys.executable, str(root / "dev/atlas/check_installed.py"),
            "--expected-version", version, "--report", str(report),
        ], cwd=work, env=env, check=True)
        subprocess.run([sys.executable, "-m", "pip", "check"], cwd=work, env=env, check=True)
        result = json.loads(report.read_text())
        print(json.dumps({"status": result["status"], "version": result["distribution"],
                          "python": result["python"], "native_build": result["native_build"]}, indent=2))


if __name__ == "__main__":
    main()
