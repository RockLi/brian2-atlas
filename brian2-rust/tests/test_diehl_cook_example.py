"""Bounded end-to-end checks for the Diehl-Cook MNIST example."""

import gzip
import importlib.util
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = ROOT.parent
RUNNER = ROOT / "target/release/b2-runner"
EXAMPLE = REPOSITORY / "examples/frompapers/Diehl_Cook_2015.py"


def _write_idx(path, values):
    values = np.asarray(values, dtype=np.uint8)
    header = struct.pack(">HBB", 0, 0x08, values.ndim)
    header += struct.pack(">" + "I" * values.ndim, *values.shape)
    with gzip.open(path, "wb") as handle:
        handle.write(header + values.tobytes())


def _mnist_fixture(path):
    train_images = np.zeros((2, 28, 28), dtype=np.uint8)
    train_images[0, 8:20, 12:16] = 255
    train_images[1, 12:16, 8:20] = 255
    test_images = train_images[::-1].copy()
    _write_idx(path / "train-images-idx3-ubyte.gz", train_images)
    _write_idx(path / "train-labels-idx1-ubyte.gz", [5, 2])
    _write_idx(path / "t10k-images-idx3-ubyte.gz", test_images)
    _write_idx(path / "t10k-labels-idx1-ubyte.gz", [2, 5])
    return train_images


def _environment():
    environment = os.environ.copy()
    python_path = [str(ROOT / "python"), str(REPOSITORY)]
    if environment.get("PYTHONPATH"):
        python_path.append(environment["PYTHONPATH"])
    environment.update({
        "PYTHONPATH": os.pathsep.join(python_path),
        "MPLBACKEND": "Agg",
        "MPLCONFIGDIR": environment.get("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "b2-diehl-cook-matplotlib")),
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
    })
    return environment


def test_mnist_reader_accepts_gzip_without_shifting_labels(tmp_path):
    pytest.importorskip("progressbar")
    expected_images = _mnist_fixture(tmp_path)
    spec = importlib.util.spec_from_file_location("diehl_cook_example", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.MNIST_PATH = tmp_path

    images, labels = module.read_mnist(True)

    np.testing.assert_array_equal(labels, [5, 2])
    np.testing.assert_allclose(images, expected_images.reshape(2, -1) / 8.0)


def test_rust_runs_reduced_train_observe_test_pipeline(tmp_path):
    pytest.importorskip("progressbar")
    if not RUNNER.is_file():
        pytest.skip("release b2-runner is not built")
    mnist = tmp_path / "mnist"
    data = tmp_path / "data"
    mnist.mkdir()
    _mnist_fixture(mnist)
    common = [
        "--mnist-path", str(mnist),
        "--data-path", str(data),
        "--n-train", "1",
        "--n-observe", "1",
        "--n-test", "1",
        "--neurons", "2",
        "--save-points", "1",
        "--presentation-ms", "0.1",
        "--rest-ms", "0.1",
        "--min-spikes", "0",
        "--device", "rust_standalone",
        "--engine", "reference",
        "--runner", str(RUNNER),
    ]
    for mode in ("train", "observe", "test"):
        completed = subprocess.run(
            [
                sys.executable,
                str(EXAMPLE),
                mode,
                *common,
                "--build-directory", str(tmp_path / f"build-{mode}"),
            ],
            cwd=tmp_path,
            env=_environment(),
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr

    assert np.load(data / "weights.npy").shape == (784 * 2,)
    assert np.load(data / "theta.npy").shape == (2,)
    assert np.load(data / "assign.npy").shape == (2,)
    confusion = np.load(data / "confusion.npy")
    assert confusion.shape == (10, 10)
    assert np.isfinite(confusion).all()
