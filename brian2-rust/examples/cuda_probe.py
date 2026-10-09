"""Experimental CUDA lowering of the existing validated Metal kernel subset.

This is a conformance/benchmark probe, not a registered Brian Device backend.
Remote execution needs only NumPy and CuPy; model validation and CPU controls
run locally before upload. No CPU fallback is used for CUDA execution.
"""
import ctypes
import hashlib
import io
import json
from pathlib import Path
import re
import time

import numpy as np

PROFILE = "b2-cuda-f32-probe-v0"
OPTIONS = ("--std=c++17", "--fmad=false", "--ftz=false",
           "--prec-div=true", "--prec-sqrt=true")
CUDA_HEADER = r'''
#include <cuda_runtime.h>
#include <math.h>
using uint = unsigned int;
using ulong = unsigned long;
using uchar = unsigned char;
using atomic_uint = uint;
static_assert(sizeof(long) == 8, "AtlasIR tick storage needs 64-bit long");
static_assert(sizeof(ulong) == 8, "AtlasIR counters need 64-bit ulong");
template<class T> __device__ inline T as_type(uint bits);
template<> __device__ inline float as_type<float>(uint bits) { return __uint_as_float(bits); }
__device__ inline float clamp(float x, float lo, float hi) { return fminf(fmaxf(x,lo),hi); }
__device__ inline float sign(float x) { return (x>0.0f)-(x<0.0f); }
constexpr int memory_order_relaxed = 0;
__device__ inline uint atomic_fetch_add_explicit(uint *p,uint x,int) { return atomicAdd(p,x); }
__device__ inline uint atomic_load_explicit(uint *p,int) { return atomicAdd(p,0u); }
__device__ inline void atomic_store_explicit(uint *p,uint x,int) { atomicExch(p,x); }
'''


def cuda_source(kernel):
    """Lower only our generated scalar/buffer ABI, never arbitrary Metal code."""
    source = kernel.source
    if source.count("kernel void ") != 1:
        raise ValueError("Expected one generated kernel entry")
    source = source.replace("#include <metal_stdlib>", "").replace("using namespace metal;", "")
    source = source.replace("#pragma clang fp contract(off)", "")
    source = re.sub(r"\bdevice\s+", "", source)
    source = re.sub(r"\bconstant long &tick\b", "long tick", source)
    source = re.sub(r"\[\[buffer\(\d+\)\]\]", "", source)
    source, count = re.subn(r",\s*uint ([A-Za-z_]\w*)\s*\[\[thread_position_in_grid\]\]\s*\)\s*\{",
                            lambda m: f") {{\nuint {m[1]} = blockIdx.x * blockDim.x + threadIdx.x;", source)
    if count != 1 or "[[" in source or re.search(r"\b(?:constant|threadgroup)\b", source):
        raise ValueError("Unsupported generated Metal ABI")
    source = re.sub(r"\binline\b", "__device__ inline", source)
    source = source.replace("kernel void ", 'extern "C" __global__ void ')
    return CUDA_HEADER + source


def make_bundle(model_path, directory, *, route="scan", runner=None, max_bytes=256*1024**2):
    """Rust-validate AtlasIR, prepare storage, and execute the same-f32 CPU control."""
    from brian2_rust.metal import (MetalExecutor, _derive_metal_plan, population_arrays)
    from brian2_rust.metal_dag import _prepare_dag_storage, _cpu_dag
    from brian2_rust.plan import validate_model

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    raw = Path(model_path).read_bytes()
    model = validate_model(json.loads(raw), runner=runner)
    plan = _derive_metal_plan(model, numeric_mode="float32", event_delivery=route)
    executor = object.__new__(MetalExecutor)  # CPU controls do not need an Apple GPU.
    executor.model, executor.plan, executor.directory = model, plan, directory
    stages, outputs = [], []
    if plan.dispatches:
        arrays, _, _ = _prepare_dag_storage(executor, max_bytes)
        for k, dispatch in zip(plan.kernels, plan.dispatches, strict=True):
            stages.append(dict(entry=k.entry, source=cuda_source(k), lanes=dispatch.lanes,
                               bindings=list(dispatch.bindings), tick=True, role=dispatch.role))
        # Scratch event ranks have intentionally nondeterministic reservation order.
        # They are not model outputs; all neuron state/recording, synapse values,
        # delivered counters and delay history/cursors are compared below.
        writable = {binding for d in plan.dispatches for binding, dtype in
                    zip(d.bindings, d.types, strict=True) if not dtype.startswith("const ")}
        outputs = [i for i in sorted(writable) if not plan.buffers[i].endswith("/active_ranks")]
        clock = plan.logical.clocks[0]
        start, steps = clock.start_tick, clock.steps
        cpu = _cpu_dag(executor)
        def control(storage):
            ptrs = (ctypes.c_void_p*len(storage))(*(a.ctypes.data for a in storage))
            cpu.cpu_dag(ptrs, 1)
    else:
        arrays = []
        for k in plan.kernels:
            data, _ = population_arrays(model, k.population, k, max_bytes)
            base = len(arrays)
            arrays.extend(data)
            stages.append(dict(entry=k.entry, source=cuda_source(k), lanes=k.neurons,
                               bindings=list(range(base, base+len(data))), tick=False, role="population"))
            outputs.extend(base+i for i in (0, 2, 4, 5, 6, 7, 8, 9))
        start, steps = 0, 1  # Each population kernel contains its own tick loop.
        cpu = executor._cpu_mirror()
        def control(storage):
            for stage in stages:
                ptrs = (ctypes.c_void_p*len(stage["bindings"]))(*(storage[i].ctypes.data for i in stage["bindings"]))
                getattr(cpu, "cpu_"+stage["entry"])(ptrs, 1)
    if not stages or any(s["lanes"] < 1 for s in stages):
        raise ValueError("CUDA probe requires nonempty dispatches")
    if 2*sum(a.nbytes for a in arrays) > max_bytes:
        raise MemoryError("Initial and reference storage exceed upload budget")
    expected = [a.copy() for a in arrays]
    begin = time.perf_counter()
    control(expected)
    cpu_seconds = time.perf_counter()-begin
    payload = io.BytesIO()
    np.savez_compressed(payload, **{f"input_{i}": a for i,a in enumerate(arrays)},
                        **{f"expected_{i}": expected[i] for i in outputs})
    data = payload.getvalue()
    manifest = dict(schema=PROFILE, model_sha256=hashlib.sha256(raw).hexdigest(),
                    plan_sha256=plan.sha256, model_layers=model["protocol"]["layers"],
                    array_count=len(arrays), outputs=outputs, start_tick=start, steps=steps,
                    route=route, stages=stages, compiler_options=list(OPTIONS),
                    control="local generated CPU f32, 1 worker; not independent Rust f64",
                    control_seconds=cpu_seconds, payload_sha256=hashlib.sha256(data).hexdigest(),
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (directory/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    (directory/"arrays.npz").write_bytes(data)
    return manifest, data


def compare_arrays(actual, expected, *, rtol, atol):
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        return dict(passed=False, reason="shape or dtype mismatch")
    exact = actual.tobytes() == expected.tobytes()
    if expected.dtype.kind == "f":
        finite = bool(np.isfinite(actual).all() and np.isfinite(expected).all())
        error = float(np.max(np.abs(actual.astype(np.float64)-expected), initial=0))
        return dict(passed=finite and bool(np.allclose(actual, expected, rtol=rtol, atol=atol)),
                    bitwise_exact=exact, finite=finite, max_abs_error=error if finite else None)
    return dict(passed=exact, bitwise_exact=exact)  # Never tolerate spike/tick/count differences.


def benchmark_bundle(manifest, payload, repeats=3, rtol=0.0, atol=0.0):
    """Run on one real CUDA device; ordered stream launches enforce stage barriers."""
    import cupy as cp
    import platform
    import subprocess

    if manifest["schema"] != PROFILE or hashlib.sha256(payload).hexdigest() != manifest["payload_sha256"]:
        raise ValueError("CUDA probe bundle identity mismatch")
    if not 1 <= repeats <= 20 or not np.isfinite([rtol,atol]).all() or min(rtol,atol) < 0:
        raise ValueError("Invalid repeats or tolerances")
    with np.load(io.BytesIO(payload), allow_pickle=False) as data:
        initial = [data[f"input_{i}"] for i in range(manifest["array_count"])]
        expected = {i:data[f"expected_{i}"] for i in manifest["outputs"]}
    properties = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    # CuPy 13's source compiler unconditionally adds -ftz=true. Compile cubins
    # explicitly so our recorded arithmetic options remain authoritative.
    import tempfile
    started = time.perf_counter()
    modules, kernels = [], []
    arch = f"sm_{properties['major']}{properties['minor']}"
    with tempfile.TemporaryDirectory(prefix="b2-cuda-") as temporary:
        for stage in manifest["stages"]:
            source = Path(temporary)/(stage["entry"]+".cu")
            binary = source.with_suffix(".cubin")
            source.write_text(stage["source"])
            compiled = subprocess.run(["nvcc","--cubin",f"--gpu-architecture={arch}",
                                       *manifest["compiler_options"],str(source),"-o",str(binary)],
                                      capture_output=True,text=True,timeout=120)
            if compiled.returncode:
                raise RuntimeError(f"CUDA compilation failed for {stage['entry']}:\n{compiled.stderr}")
            module = cp.RawModule(path=str(binary))
            kernels.append(module.get_function(stage["entry"]))
            modules.append(module)
    compile_seconds = time.perf_counter()-started
    name = properties["name"]
    report = dict(schema=PROFILE, status="running", actual_gpu=name.decode() if isinstance(name,bytes) else name,
                  compute_capability=[properties["major"], properties["minor"]],
                  device_memory_bytes=properties["totalGlobalMem"], host=platform.platform(),
                  cuda_driver=cp.cuda.runtime.driverGetVersion(), cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),
                  cupy=cp.__version__, numpy=np.__version__, compile_seconds=compile_seconds,
                  nvcc=subprocess.check_output(["nvcc","--version"],text=True),
                  gpu_architecture=arch,
                  model_sha256=manifest["model_sha256"], plan_sha256=manifest["plan_sha256"],
                  payload_sha256=manifest["payload_sha256"], source_sha256=manifest["source_sha256"],
                  compiler_options=manifest["compiler_options"], rtol=rtol, atol=atol,
                  nvidia_smi=subprocess.check_output(["nvidia-smi"],text=True),
                  timing_scope="single CUDA stream; GPU interval includes host launch gaps; no CUDA Graph yet",
                  control=manifest["control"], trials=[])
    stream = cp.cuda.Stream(non_blocking=True)
    for trial in range(-1, repeats):  # One warmup, checked but excluded from measured trials.
        begin = time.perf_counter()
        with stream:
            gpu_arrays = [cp.asarray(a) for a in initial]
            stream.synchronize()
            input_seconds = time.perf_counter()-begin
            start, end = cp.cuda.Event(), cp.cuda.Event()
            launch_started = time.perf_counter()
            start.record(stream)
            for tick in range(manifest["start_tick"], manifest["start_tick"]+manifest["steps"]):
                for kernel, stage in zip(kernels,manifest["stages"],strict=True):
                    args = tuple(gpu_arrays[i] for i in stage["bindings"])
                    if stage["tick"]:
                        args += (np.int64(tick),)
                    kernel(((stage["lanes"]+127)//128,), (128,), args, stream=stream)
            end.record(stream)
            end.synchronize()
            command_seconds = time.perf_counter()-launch_started
            read_started = time.perf_counter()
            actual = {i:cp.asnumpy(gpu_arrays[i],stream=stream) for i in expected}
            readback_seconds = time.perf_counter()-read_started
        run_seconds = time.perf_counter()-begin
        checks = {str(i):compare_arrays(actual[i],expected[i],rtol=rtol,atol=atol) for i in expected}
        result = dict(repeat=trial, passed=all(c["passed"] for c in checks.values()), checks=checks,
                      input_seconds=input_seconds, command_seconds=command_seconds,
                      gpu_interval_seconds=cp.cuda.get_elapsed_time(start,end)/1000,
                      readback_seconds=readback_seconds, run_seconds=run_seconds)
        if trial == -1:
            report["warmup"] = result
        else:
            report["trials"].append(result)
        if not result["passed"]:
            report["status"] = "correctness_failed"
            return report
    report["status"] = "passed"
    return report
