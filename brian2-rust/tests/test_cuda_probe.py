"""Local protocol/lowering checks; real CUDA execution is a separate Modal gate."""
import io
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"examples"))
from cuda_probe import make_bundle, cuda_source, compare_arrays
from brian2_rust.results import load_results


def test_cuda_bundle_control_matches_independent_rust(tmp_path):
    model_path = ROOT/"tests/golden/b2ir-v1/minimal-v1.json"
    runner = ROOT/"target/release/b2-runner"
    manifest, payload = make_bundle(model_path,tmp_path/"bundle",runner=runner)
    reference = tmp_path/"reference"
    subprocess.run([str(runner),str(model_path),str(reference)],check=True,capture_output=True)
    model = json.loads(model_path.read_text())
    expected = load_results(model,reference)["populations"][0]
    with np.load(io.BytesIO(payload),allow_pickle=False) as arrays:
        state = arrays["expected_0"]
        n = model["definition"]["populations"][0]["count"]
        for i,symbol in enumerate(model["definition"]["populations"][0]["states"]):
            np.testing.assert_allclose(state[i*n:(i+1)*n],expected["states"][symbol["name"]],rtol=2e-6)
        np.testing.assert_array_equal(arrays["expected_5"],expected["counts"])
    assert manifest["stages"][0]["tick"] is False
    source = manifest["stages"][0]["source"]
    assert 'extern "C" __global__ void population_0' in source
    assert "[[" not in source and "blockIdx.x" in source


def test_cuda_rejects_unknown_abi():
    with pytest.raises(ValueError,match="ABI"):
        cuda_source(SimpleNamespace(source="kernel void bad(threadgroup uint *x [[buffer(0)]]) {}"))


def test_cuda_canonical_lane_and_tick_abi():
    source = cuda_source(SimpleNamespace(source="""kernel void stage(device float *s [[buffer(0)]],
        constant long &tick [[buffer(1)]], uint lane [[thread_position_in_grid]]) {
        if (lane) return; s[0] += float(tick);
    }"""))
    assert "long tick" in source and "uint lane = blockIdx.x" in source
    assert "if (lane) return" in source and "[[" not in source


def test_cuda_comparison_never_tolerates_event_errors():
    assert not compare_arrays(np.array([1],np.int64),np.array([2],np.int64),rtol=10,atol=10)["passed"]
    assert not compare_arrays(np.array([np.nan],np.float32),np.array([np.nan],np.float32),rtol=1,atol=1)["passed"]
    assert compare_arrays(np.array([1.00001],np.float32),np.array([1],np.float32),rtol=1e-4,atol=0)["passed"]
