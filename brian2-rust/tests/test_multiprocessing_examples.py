"""Run Brian's standalone multiprocessing examples with the Rust Device."""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = ROOT.parent
RUNNER = ROOT / "target/release/b2-runner"
EXAMPLES = REPOSITORY / "examples/multiprocessing"


@pytest.mark.parametrize(
    "script, optional_module",
    [
        ("02_using_standalone.py", None),
        ("03_standalone_joblib.py", "joblib"),
    ],
)
def test_official_standalone_multiprocessing_example(
    tmp_path, script, optional_module
):
    if optional_module:
        pytest.importorskip(optional_module)

    environment = os.environ.copy()
    python_path = [str(ROOT / "python"), str(REPOSITORY)]
    if environment.get("PYTHONPATH"):
        python_path.append(environment["PYTHONPATH"])
    environment.update({
        "PYTHONPATH": os.pathsep.join(python_path),
        "BRIAN2_STANDALONE_DEVICE": "rust_standalone",
        "BRIAN2_STANDALONE_MODULE": "brian2_rust",
        "BRIAN2_EXAMPLE_PROCESSES": "2",
        "BRIAN2_EXAMPLE_SIMULATIONS": "2",
        "B2_RUNNER": str(RUNNER),
        "MPLBACKEND": "Agg",
        "MPLCONFIGDIR": environment.get(
            "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "b2-matplotlib")),
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
    })
    run_directory = tmp_path / script.removesuffix(".py")
    run_directory.mkdir()

    completed = subprocess.run(
        [sys.executable, str(EXAMPLES / script)],
        cwd=run_directory,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "Done in" in completed.stdout
    assert completed.stdout.count("FINISHED") == 2
