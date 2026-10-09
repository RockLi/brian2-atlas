"""Build the native CUDA training library; no tensor runtime or CPU fallback.

The arithmetic body is shared with Metal. Only kernel argument annotations,
thread indexing and scalar math spellings are translated, in a checked step.
"""
from pathlib import Path
import re
import shutil
import subprocess


def kernel_source():
    source='\n'.join((Path(__file__).parent/name).read_text() for name in
                     ('training_metal.metal','training_metal_mpi.metal','training_state.metal','training_clock.metal','training_poisson.metal','training_dynamic.metal','training_dynamic_mpi.metal'))
    source=source.replace('#include <metal_stdlib>\nusing namespace metal;',
                          '#include <cuda_runtime.h>\n#include <stdint.h>\n#include <math.h>')
    source=source.replace('kernel void ', '__global__ void ')
    # Every ordinary helper in a Metal shader executes on the GPU, including
    # scalar-equation adapters added independently of the dynamic/clock families.
    source=re.sub(r'(?m)^(int|float|bool|void|ulong|uint) (atlas_[a-z_][a-z_0-9]*)\(',r'__device__ \1 \2(',source)
    source=source.replace('atlas_dynamic_poisson_from_bits(uint x){return as_type<float>(x);}', 'atlas_dynamic_poisson_from_bits(uint x){return __uint_as_float(x);}')
    source=source.replace('as_type<int>(x)', '__float_as_int(x)').replace('as_type<float>(x)', '__int_as_float(x)').replace('as_type<uint>(x)', '__float_as_uint(x)')
    source=re.sub(r'\[\[buffer\(\d+\)\]\]', '', source)
    source=source.replace(', uint b [[thread_position_in_grid]]) {',
                          ') {\n uint32_t b=blockIdx.x*blockDim.x+threadIdx.x;')
    source=re.sub(r'\b(?:device|thread)\s+', '', source)
    source=re.sub(r'\bulong\b', 'uint64_t', source)
    source=re.sub(r'\blong\b', 'int64_t', source)
    for before,after in [('tan','tanf'),('cosh','coshf'),('sinh','sinhf'),('log10','log10f'),('expm1','expm1f'),('log1p','log1pf'),('acos','acosf'),('asin','asinf'),('atan','atanf'),('ceil','ceilf'),('exp','expf'),('log','logf'),('abs','fabsf'),('max','fmaxf'),('min','fminf'),('tanh','tanhf'),('sqrt','sqrtf'),('sin','sinf'),('cos','cosf'),('pow','powf'),('floor','floorf'),('round','roundf'),('fmod','fmodf')]:
        source=re.sub(r'\b'+before+r'\(',after+'(',source)
    if '[[' in source or source.count('__global__ void ')!=7:
        raise ValueError('native GPU shader ABI changed; update CUDA translation')
    return source


def build(directory):
    compiler=shutil.which('nvcc')
    if compiler is None:
        raise ValueError('native CUDA training requires nvcc and an actual NVIDIA GPU')
    directory=Path(directory)
    (directory/'training_kernels.cuh').write_text(kernel_source())
    library=directory/'libatlas-training-cuda.so'
    result=subprocess.run([compiler,'-O2','--fmad=false','-shared','-Xcompiler','-fPIC',
        '-std=c++17','-ldl','-I',str(directory),str(Path(__file__).with_suffix('.cu')),'-o',str(library)],
        capture_output=True,text=True,timeout=120)
    if result.returncode: raise RuntimeError(result.stderr)
    return library
