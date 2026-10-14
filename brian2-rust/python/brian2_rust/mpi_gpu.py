"""Opt-in MPI rank GPU offload; canonical host event/queue ordering is retained.

Only population state-update nodes are offloaded. Host storage and other nodes
remain f64, while these kernels use the existing approximate GPU f32 algebra.
No GPU library is loaded by CPU ranks; failures abort the MPI communicator.
"""
from pathlib import Path
import hashlib
import json
import platform
import re
from types import SimpleNamespace

from .plan import PlanValidationError

PROFILE = "b2-mpi-cpu-f64-gpu-update-f32-v0"


def backend_policy(ranks, backends, numeric_mode):
    if backends is None:
        backends = ("cpu",) * ranks
    if not isinstance(backends, (tuple, list)) or len(backends) != ranks:
        raise PlanValidationError("rank_backends must contain one backend per MPI rank")
    if any(not isinstance(b, str) or not re.fullmatch(r"cpu|metal|cuda(?::(?:0|[1-9][0-9]{0,3}))?", b) for b in backends):
        raise PlanValidationError("MPI rank backend must be cpu, metal, cuda or cuda:<device ordinal>")
    gpu = any(b != "cpu" for b in backends)
    if numeric_mode != ("mixed-f32" if gpu else "reference-f64"):
        raise PlanValidationError("MPI GPU ranks require numeric_mode='mixed-f32'; CPU-only MPI requires reference-f64")
    return tuple(backends) if gpu else ()


def _updates(model):
    return [(p, i, code) for p, pop in enumerate(model["definition"]["populations"])
            for i, code in enumerate(pop["code_objects"]) if code["kind"] == "state_update"]


def _fields(pop):
    fields = [(s["name"], f"state_{i}[i]", s["dtype"]) for i, s in enumerate(pop["states"])]
    fields += [(s["name"], f"parameter_{i}" + ("" if s["index_domain"] == "scalar" else "[i]"), s["dtype"])
               for i, s in enumerate(pop["parameters"])]
    if pop["refractory"] is not None:
        fields += [("lastspike", "lastspike[i]", "f64"), ("not_refractory", "not_refractory[i]", "bool")]
    return fields


def kernels(model):
    from .metal import _Expressions, _PRELUDE, _literal, number
    result = []
    for ordinal, (p, item, code) in enumerate(_updates(model)):
        pop = model["definition"]["populations"][p]
        if pop["count"] > 2**32-1:
            raise PlanValidationError("MPI GPU population count exceeds uint32 dispatch domain")
        fields = _fields(pop)
        symbols = {name: f"values[{k}ul*count+lane]" for k, (name, _, _) in enumerate(fields)}
        if pop["refractory"] is not None:
            symbols["not_refractory"] = f"({symbols['not_refractory']} != 0.0f)"
        symbols.update(i="i", N=f"{pop['count']}ul", t="time", dt=_literal(number(pop["dt"])))
        dtypes = {name: dtype for name, _, dtype in fields}
        dtypes.update(i="index", N="index", t="f64", dt="f64")
        # RNG scalar domains need a separate once-per-rank evaluation contract.
        from .metal_random import has_random
        if has_random(code["scalar"]):
            raise PlanValidationError("MPI GPU state updates do not support scalar RNG")
        block = _Expressions(symbols, {}, math_error="fault", dtypes=dtypes,
                             rng={"seed": model["instance"]["rng_seed"], "index": "ulong(i)"})
        block.statements(code["scalar"])
        block.snapshot = dict(symbols)
        block.statements(code["vector"])
        body = list(block.lines)
        for name in code["effects"]["writes"]:
            if name == "not_refractory":
                raise PlanValidationError("MPI GPU state update cannot modify refractory flags")
            if name not in {s["name"] for s in pop["states"]}:
                raise PlanValidationError(f"MPI GPU unsupported state-update write: {name}")
            body.append(f"{symbols[name]} = {block.symbols[name]};")
        entry = f"mpi_update_{ordinal}"
        source = _PRELUDE + f'''\nkernel void {entry}(device float *values [[buffer(0)]],
    device const ulong *meta [[buffer(1)]], device uint *faults [[buffer(2)]],
    uint lane [[thread_position_in_grid]]) {{
    const ulong count = meta[0];
    if (ulong(lane) >= count) return;
    const ulong i = meta[1] + ulong(lane);
    const long tick = long(meta[2]);
    const float time = as_type<float>(uint(meta[3]));
    bool fault = false;
''' + "\n".join(body) + "\nfaults[lane] = uint(fault);\n}\n"
        result.append(SimpleNamespace(entry=entry, source=source, population=p, item=item))
    if not result:
        raise PlanValidationError("MPI GPU offload requires a population state-update node")
    return result


def emit_update(model, p, item):
    pop = model["definition"]["populations"][p]
    ordinal = next(k for k, (q, i, _) in enumerate(_updates(model)) if (q, i) == (p, item))
    fields = _fields(pop)
    code = pop["code_objects"][item]
    lines = [f"let count = p{p}_stop-p{p}_start;", "if count != 0 {",
             f"mpi_gpu.prepare(count, {len(fields)})?;"]
    if pop["refractory"] is not None:
        lines += [f"for i in p{p}_start..p{p}_stop {{",
                  f"p{p}_not_refractory[i] = u8::from(p{p}_tick >= p{p}_refractory_until[i]);", "}"]
    for _, access, _ in fields:
        lines += [f"for i in p{p}_start..p{p}_stop {{ mpi_gpu.push(p{p}_{access} as f64)?; }}"]
    lines += [f"mpi_gpu.run({ordinal}, [count as u64, p{p}_start as u64, p{p}_tick as u64, (time as f32).to_bits() as u64, rng_seed])?;"]
    for name in code["effects"]["writes"]:
        field = next(k for k, (n, _, _) in enumerate(fields) if n == name)
        state = next(k for k, s in enumerate(pop["states"]) if s["name"] == name)
        lines += [f"for i in p{p}_start..p{p}_stop {{ p{p}_state_{state}[i] = mpi_gpu.values[{field}*count+i-p{p}_start] as f64; }}"]
    return lines + ["}"]


def library_inventory(backends):
    return tuple((["libmpi-metal.dylib"] if "metal" in backends else []) +
                 (["libmpi-cuda.so"] if any(b.startswith("cuda") for b in backends) else []))


def source_inventory(backends):
    files = []
    if "metal" in backends:
        files += ["mpi_gpu_metal.m", "mpi_metal_bridge.m", "clocks.h"]
    if any(b.startswith("cuda") for b in backends):
        files += ["mpi_gpu_cuda.cu"]
    return tuple(files)


def write_sources(model, directory, backends):
    from .cuda_codegen import cuda_source, CUDA_HEADER
    planned = kernels(model)
    if "metal" in backends:
        runtime = Path(__file__).with_name("metal_runtime")
        for original, output in (("bridge.m", "mpi_metal_bridge.m"), ("clocks.h", "clocks.h")):
            (directory / output).write_bytes((runtime / original).read_bytes())
        sources = ",\n".join(json.dumps(k.source) for k in planned)
        entries = ", ".join(json.dumps(k.entry) for k in planned)
        wrapper = '''#include "mpi_metal_bridge.m"
static const char *sources[] = {SOURCES};
static const char *entries[] = {ENTRIES};
void *b2gpu_create(uint32_t kernel, int device, char *error, size_t capacity) {
    if (kernel >= sizeof(entries)/sizeof(*entries) || device != 0) return NULL;
    return b2_metal_create(sources[kernel], entries[kernel], error, capacity);
}
void b2gpu_destroy(void *handle) { b2_metal_destroy(handle); }
int b2gpu_run(void *handle, float *values, uint64_t length, uint64_t *meta,
              uint32_t *faults, char *error, size_t capacity) {
    void *data[] = { values, meta, faults };
    uint64_t sizes[] = { length*4, 5*8, meta[0]*4 };
    double timings[4];
    return b2_metal_run(handle, data, sizes, 3, (uint32_t)meta[0], 5, timings, error, capacity);
}
'''.replace("SOURCES", sources).replace("ENTRIES", entries)
        (directory / "mpi_gpu_metal.m").write_text(wrapper)
    if any(b.startswith("cuda") for b in backends):
        sources = [CUDA_HEADER]
        for i, kernel in enumerate(planned):
            sources += [f"namespace mpi_k{i} {{", cuda_source(kernel).removeprefix(CUDA_HEADER), "}"]
        calls = "\n".join(f"case {i}: mpi_k{i}::{k.entry}<<<blocks,256>>>(h->values,h->meta,h->faults); break;"
                          for i, k in enumerate(planned))
        runtime = Path(__file__).with_name("mpi_runtime") / "gpu_cuda.cpp"
        sources.append(runtime.read_text().replace("KERNEL_COUNT", str(len(planned))).replace("KERNEL_CASES", calls))
        (directory / "mpi_gpu_cuda.cu").write_text("\n".join(sources))
    return source_inventory(backends)


def compile_libraries(directory, backends):
    from .distributed import _checked
    from .cuda_codegen import OPTIONS
    commands = []
    if "metal" in backends:
        if platform.system() != "Darwin":
            raise RuntimeError("MPI Metal ranks require macOS; use CUDA ranks on Linux")
        commands.append(("libmpi-metal.dylib", ["clang", "-O2", "-fobjc-arc", "-ffp-contract=off", "-dynamiclib",
            "-framework", "Foundation", "-framework", "Metal", str(directory / "mpi_gpu_metal.m")]))
    if any(b.startswith("cuda") for b in backends):
        commands.append(("libmpi-cuda.so", ["nvcc", *OPTIONS, "-O2", "--shared", "-Xcompiler=-fPIC",
            str(directory / "mpi_gpu_cuda.cu")]))
    result = {}
    for name, command in commands:
        output = directory / name
        _checked([*command, "-o", str(output)])
        result[name] = hashlib.sha256(output.read_bytes()).hexdigest()
    return result


def rust_runtime(backends):
    runtime = (Path(__file__).with_name("mpi_runtime") / "gpu.rs").read_text()
    # GPU entry count is supplied separately by each rank's generated call sites;
    # handles are lazily compiled for nodes actually executed by that rank.
    return runtime.replace("RANK_BACKENDS", json.dumps(backends))
