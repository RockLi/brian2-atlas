"""CUDA scalar/buffer lowering shared by the native CUDA execution plan."""
import re

OPTIONS = ("--std=c++17", "--fmad=false", "--ftz=false",
           "--prec-div=true", "--prec-sqrt=true")
CUDA_ATOMIC_LOAD = r'''__device__ inline uint atomic_load_explicit(uint *p,int) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600 && (__CUDACC_VER_MAJOR__ > 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 8))
    return __nv_atomic_load_n(p,__NV_ATOMIC_RELAXED,__NV_THREAD_SCOPE_DEVICE);
#else
    return atomicAdd(p,0u);
#endif
}'''
CUDA_HEADER = r'''
#include <cuda_runtime.h>
#include <math.h>
using uint = unsigned int;
using ulong = unsigned long;
using uchar = unsigned char;
using atomic_uint = uint;
static_assert(sizeof(long) == 8, "AtlasIR tick storage needs 64-bit long");
static_assert(sizeof(ulong) == 8, "AtlasIR counters need 64-bit ulong");
template<class T, class U> __device__ inline T as_type(U bits) {
    static_assert(sizeof(T)==sizeof(U), "bit-cast width mismatch");
    T result; memcpy(&result, &bits, sizeof(T)); return result;
}
__device__ inline float clamp(float x, float lo, float hi) { return fminf(fmaxf(x,lo),hi); }
__device__ inline float sign(float x) { return (x>0.0f)-(x<0.0f); }
constexpr int memory_order_relaxed = 0;
__device__ inline uint atomic_fetch_add_explicit(uint *p,uint x,int) { return atomicAdd(p,x); }
''' + CUDA_ATOMIC_LOAD + r'''
__device__ inline void atomic_store_explicit(uint *p,uint x,int) { atomicExch(p,x); }
'''


def cuda_source(kernel, *, chunked=False, advance=False):
    """Lower only our generated scalar/buffer ABI, never arbitrary Metal code."""
    source = kernel.source
    if source.count("kernel void ") != 1:
        raise ValueError("Expected one generated kernel entry")
    source = source.replace("#include <metal_stdlib>", "").replace("using namespace metal;", "")
    source = source.replace("#pragma clang fp contract(off)", "")
    source = re.sub(r"\bdevice\s+", "", source)
    source = re.sub(r"\bthread\s+", "", source)
    source = re.sub(r"\bconstant long &tick\b", "long tick", source)
    source = re.sub(r"\[\[buffer\(\d+\)\]\]", "", source)
    source, count = re.subn(r",\s*uint ([A-Za-z_]\w*)\s*\[\[thread_position_in_grid\]\]\s*\)\s*\{",
                            lambda m: f") {{\nuint {m[1]} = blockIdx.x * blockDim.x + threadIdx.x;", source)
    if count != 1 or "[[" in source or re.search(r"\b(?:constant|threadgroup)\b", source):
        raise ValueError("Unsupported generated Metal ABI")
    source = re.sub(r"\binline\b", "__device__ inline", source)
    source = source.replace("kernel void ", 'extern "C" __global__ void ')
    if chunked:
        source += chunk_variant(source,kernel.entry)
    if advance:
        if not chunked:raise ValueError('Cursor advancement requires the chunk ABI')
        source += CHUNK_ADVANCE_SOURCE
    fault_atomic = "__device__ inline uint atomic_fetch_or_explicit(uint *p,uint x,int) { return atomicOr(p,x); }\n" if "atomic_fetch_or_explicit(" in source else ""
    return CUDA_HEADER + fault_atomic + source


CUDA_CHUNK_ABI = "b2-cuda-chunk-ticks-v0"
CHUNK_ADVANCE_ENTRY = "b2_cuda_chunk_advance"
CHUNK_ADVANCE_SOURCE = r'''
extern "C" __global__ void b2_cuda_chunk_advance(
    unsigned long long* cursor, unsigned long long length,
    unsigned int count, unsigned int* fault) {
    const unsigned long long begin = *cursor;
    if (begin > length || count > length - begin) { *fault = 1; return; }
    *cursor = begin + count;
}
'''


def chunk_variant(source,entry):
    """Add a checked tick-table entry beside the unchanged direct CUDA kernel.

    The full generated variant is part of CudaPlan.source and the cubin key.
    The host computes clock coalescence; kernels only read exact int64 ticks.
    """
    marker='extern "C" __global__ void '+entry+'('
    if source.count(marker)!=1:raise ValueError('Expected one CUDA entry for chunk lowering')
    body=source[source.index(marker):]
    header,brace,implementation=body.partition('{')
    header,count=re.subn(r'\blong tick\b',
        'const long* b2_chunk_ticks, const unsigned long long* b2_chunk_cursor, '
        'unsigned long long b2_chunk_length, unsigned int b2_chunk_offset, unsigned int* b2_chunk_fault',header)
    if count!=1:raise ValueError('Chunk kernels require the generated int64 tick argument')
    header=header.replace(marker,'extern "C" __global__ void '+entry+'_chunk(')
    guard=r'''
    const unsigned long long b2_chunk_begin = *b2_chunk_cursor;
    if (b2_chunk_begin > b2_chunk_length || b2_chunk_offset >= b2_chunk_length-b2_chunk_begin) {
        if (blockIdx.x == 0 && threadIdx.x == 0) atomicExch(b2_chunk_fault,1u);
        return;
    }
    const long tick = b2_chunk_ticks[b2_chunk_begin+b2_chunk_offset];
'''
    return '\n'+header+brace+guard+implementation
