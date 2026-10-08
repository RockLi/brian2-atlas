"""Native CUDA plan, reference conformance and Brian Device lifecycle."""
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess

import brian2 as b
import numpy as np
import pytest

from brian2_rust import build_execution_plan,bind_execution_plan,explain_plan,verify_execution_plan
from brian2_rust.cuda import CudaExecutor,CUDA_PROFILE,write_cuda_results
from brian2_rust.plan import PlanValidationError
from brian2_rust.results import load_results
from test_metal_delays import device
from test_metal_dag import coupled
from test_metal_plasticity import plasticity_case,segmented_case,equivalent

ROOT = Path(__file__).resolve().parents[1]
real_cuda = pytest.mark.skipif(os.environ.get("B2_TEST_CUDA")!="1",reason="requires NVIDIA CUDA")


def test_cuda_plan_identity_and_explicit_precision(tmp_path):
    model=json.loads((ROOT/"tests/golden/b2ir-v1/minimal-v1.json").read_text())
    with pytest.raises(PlanValidationError,match="explicit"):
        build_execution_plan(model,backend="cuda")
    plan=build_execution_plan(model,backend="cuda",numeric_mode="float32")
    assert plan.numeric_profile==CUDA_PROFILE
    assert "CUDA" in explain_plan(plan)
    verify_execution_plan(plan,model)
    for bad in (replace(plan,compiler_options=("--use_fast_math",)),
                replace(plan,kernels=(replace(plan.kernels[0],source="bad"),))):
        with pytest.raises(PlanValidationError,match="CUDA"):
            verify_execution_plan(bad,model)
        with pytest.raises(PlanValidationError,match="CUDA plan"):
            CudaExecutor(model,tmp_path,numeric_mode="float32",plan=bad)
    with pytest.raises(PlanValidationError,match="profile"):
        bind_execution_plan(plan,{"plan_sha256":plan.sha256,"numeric_profile":"b2-metal-f32-v0"})


def test_cuda_device_requires_explicit_float32(device,tmp_path):
    with pytest.raises(NotImplementedError,match="float32"):
        b.set_device("rust_standalone",engine="cuda",directory=tmp_path)


@real_cuda
def test_cuda_golden_transport_replay_and_close(tmp_path):
    path=ROOT/"tests/golden/b2ir-v1/minimal-v1.json"
    model=json.loads(path.read_text())
    subprocess.run([str(ROOT/"target/release/b2-runner"),str(path),str(tmp_path/"reference")],check=True)
    expected=load_results(model,tmp_path/"reference")
    with CudaExecutor(model,tmp_path/"cuda",numeric_mode="float32") as executor:
        first=executor.run(); equivalent(first,expected)
        equivalent(first,executor.run(compute="cpu-f32",workers=2),exact=True)
        write_cuda_results(model,first,tmp_path/"transport")
        loaded=load_results(model,tmp_path/"transport")
        equivalent(loaded,first,exact=True)
        assert loaded["metadata"]["engine"]=="cuda"
        assert bind_execution_plan(executor.plan,loaded["metadata"]).backend=="cuda"
        for state in first["populations"][0]["states"].values():state.fill(123)
        equivalent(executor.run(),expected)
        with pytest.raises(MemoryError):executor.run(max_buffer_bytes=1)
        with pytest.raises(ValueError):executor.run(max_buffer_bytes=0)
        with pytest.raises(PlanValidationError):
            write_cuda_results(model,executor.run(compute="cpu-f32"),tmp_path/"wrong-profile")
    with pytest.raises(RuntimeError,match="closed"):executor.run()


@real_cuda
@pytest.mark.parametrize("route",["scan","sparse"])
def test_cuda_coupled_reference(coupled,tmp_path,route):
    model,expected=coupled
    with CudaExecutor(model,tmp_path/route,numeric_mode="float32",event_delivery=route) as executor:
        actual=executor.run()
        equivalent(actual,expected)
        mirror=executor.run(compute="cpu-f32",workers=2)
        equivalent(actual,mirror,exact=True)
        for a,e in zip(actual["populations"],expected["populations"],strict=True):
            for name in a["event_streams"]:
                for key in ("ticks","indices"):
                    np.testing.assert_array_equal(a["event_streams"][name][key],e["event_streams"][name][key])
        equivalent(executor.run(),actual,exact=True)
        with pytest.raises(MemoryError):executor.run(max_buffer_bytes=1)


@real_cuda
@pytest.mark.parametrize("integrate",[False,True])
def test_cuda_plasticity_integration(device,tmp_path,integrate):
    plasticity_case(device,tmp_path,integrate,CudaExecutor,write_cuda_results)


@real_cuda
def test_cuda_device_pending_store_restore(device,tmp_path):
    segmented_case(device,tmp_path,"cuda")
