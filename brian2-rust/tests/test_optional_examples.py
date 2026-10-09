"""Bounded integration checks for examples with optional dependencies."""

import importlib.util
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = ROOT.parent
RUNNER = ROOT / "target/release/b2-runner"


def example_environment(tmp_path):
    environment = os.environ.copy()
    python_path = [str(ROOT / "python"), str(REPOSITORY)]
    if environment.get("PYTHONPATH"):
        python_path.append(environment["PYTHONPATH"])
    environment.update({
        "PYTHONPATH": os.pathsep.join(python_path),
        "B2_RUNNER": str(RUNNER),
        "MPLBACKEND": "Agg",
        "MPLCONFIGDIR": environment.get(
            "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "b2-matplotlib")),
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
    })
    return environment


@pytest.mark.parametrize("device_name,module_name", [("atlas", "brian2_atlas"), ("rust_standalone", "brian2_rust")])
def test_opencv_example_uses_python_frames_with_rust(tmp_path, device_name, module_name):
    cv2 = pytest.importorskip("cv2")
    video_path = tmp_path / "tiny.avi"
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 24.0, (8, 6))
    if not writer.isOpened():
        pytest.skip("OpenCV installation cannot encode an MJPG test video")
    for value in (0, 96, 192):
        writer.write(np.full((6, 8, 3), value, dtype=np.uint8))
    writer.release()
    assert video_path.is_file()
    # Reproduce the official sample's metadata mismatch: OpenCV 4.13 reports
    # 1400 indexed frames for that file but can decode only 1179.  AVI stores
    # the count in both the main and video-stream headers.
    video_data = bytearray(video_path.read_bytes())
    avih = video_data.index(b"avih")
    strh = video_data.index(b"strh")
    video_data[avih + 24:avih + 28] = struct.pack("<I", 10)
    video_data[strh + 40:strh + 44] = struct.pack("<I", 10)
    video_path.write_bytes(video_data)

    environment = example_environment(tmp_path)
    environment.update({
        "BRIAN2_STANDALONE_DEVICE": device_name,
        "BRIAN2_STANDALONE_MODULE": module_name,
        "BRIAN2_STANDALONE_DIRECTORY": str(tmp_path / "device"),
        "BRIAN2_OPENCV_VIDEO": str(video_path),
    })
    run_directory = tmp_path / "run"
    run_directory.mkdir()

    completed = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY / "examples/advanced/opencv_movie.py"),
        ],
        cwd=run_directory,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "OpenCV decodes 3 of 10 reported frames" in completed.stdout
    assert list((tmp_path / "device").glob("**/model.json"))


@pytest.mark.parametrize("device_name,module_name", [("atlas", "brian2_atlas"), ("rust_standalone", "brian2_rust")])
def test_opencv_example_accepts_bounded_camera_stream_with_rust(tmp_path, device_name, module_name):
    pytest.importorskip("cv2")
    environment = example_environment(tmp_path)
    environment.update({
        "BRIAN2_STANDALONE_DEVICE": device_name,
        "BRIAN2_STANDALONE_MODULE": module_name,
        "BRIAN2_STANDALONE_DIRECTORY": str(tmp_path / "device"),
        "BRIAN2_OPENCV_CAMERA": "2",
        "BRIAN2_OPENCV_FRAMES": "3",
        "BRIAN2_OPENCV_SIZE": "4x3",
        "BRIAN2_OPENCV_GUI": "0",
    })
    example = REPOSITORY / "examples/advanced/opencv_movie.py"
    program = f"""
import runpy

import cv2
import numpy as np


class FakeCamera:
    def __init__(self, source):
        assert source == 2
        self.opened = True
        self.value = 0

    def isOpened(self):
        return self.opened

    def get(self, property_id):
        return {{
            cv2.CAP_PROP_FRAME_WIDTH: 8,
            cv2.CAP_PROP_FRAME_HEIGHT: 6,
            cv2.CAP_PROP_FPS: 24,
            cv2.CAP_PROP_FRAME_COUNT: 0,
        }}.get(property_id, 0)

    def read(self):
        self.value += 1
        frame = np.full((6, 8, 3), self.value * 32, dtype=np.uint8)
        return True, frame

    def release(self):
        self.opened = False


cv2.VideoCapture = FakeCamera
runpy.run_path({str(example)!r}, run_name="__main__")
print("CAMERA_STREAM_OK")
"""
    completed = subprocess.run(
        [sys.executable, "-c", program],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "camera 2; 8x6 -> 4x3" in completed.stdout
    assert "CAMERA_STREAM_OK" in completed.stdout
    assert list((tmp_path / "device").glob("**/model.json"))


def test_sbi_example_simulation_core_runs_with_rust(tmp_path):
    if importlib.util.find_spec("sbi") is None:
        pytest.skip("sbi is not installed")

    example = REPOSITORY / "examples/advanced/modelfitting_sbi.py"
    device_directory = tmp_path / "device"
    program = f"""
import importlib.util
from pathlib import Path

import brian2 as b
import brian2_rust
from brian2.devices.device import all_devices

device = all_devices["rust_standalone"]
device.reinit()
b.set_device(
    "rust_standalone",
    runner=Path({str(RUNNER)!r}),
    engine="reference",
    directory=Path({str(device_directory)!r}),
)
spec = importlib.util.spec_from_file_location("modelfitting_sbi", Path({str(example)!r}))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
result = module.simulate(
    [[32.0, 1.0]])
statistics = module.calculate_summary_statistics(result)
assert result["v"].shape == (1, 7000)
assert statistics.shape == (1, 4)
assert __import__("numpy").isfinite(statistics).all()
print("SBI_SMOKE_OK")
"""
    completed = subprocess.run(
        [sys.executable, "-c", program],
        cwd=tmp_path,
        env=example_environment(tmp_path),
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "SBI_SMOKE_OK" in completed.stdout
