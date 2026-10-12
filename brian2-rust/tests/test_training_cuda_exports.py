"""Real C++ linkage and dlsym checks on CUDA host capability definitions.

The extracted definitions are compiled unchanged; no device runtime is tested.
"""
import ctypes
import importlib.util
import os
from pathlib import Path
import re
import shutil
import subprocess

import pytest


def translator():
    path = Path(os.environ.get('B2_CUDA_TRANSLATOR',
        Path(__file__).resolve().parents[1]/'python/brian2_rust/training_cuda.py'))
    spec = importlib.util.spec_from_file_location('cuda_abi_under_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def definitions():
    path = Path(os.environ.get('B2_CUDA_HOST_SOURCE',
        Path(__file__).resolve().parents[1]/'python/brian2_rust/training_cuda.cu'))
    return re.findall(r'(?m)^(?:extern "C" )?uint64_t b2_train_\w+_v1\(void\)\{return \d+;\}', path.read_text())


def compile_library(directory, functions):
    compiler = shutil.which('clang++') or shutil.which('g++')
    assert compiler, 'actual C++ compiler required for ABI verification'
    source = directory/'capabilities.cpp'
    source.write_text('#include <stdint.h>\n'+'\n'.join(functions)+'\n')
    library = directory/'capabilities.so'
    result = subprocess.run([compiler, '-std=c++17', '-shared', '-fPIC', str(source),
                             '-o', str(library)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    return library


@pytest.fixture(scope='module')
def host_library(tmp_path_factory):
    return ctypes.CDLL(str(compile_library(tmp_path_factory.mktemp('cuda-host-exports'), definitions())))


@pytest.mark.parametrize('name', [
    'math', 'poisson', 'poisson_vjp', 'poisson_boundary', 'vjp_activity',
    'static_vjp_activity', 'static_timed_input', 'static_poisson',
    'scalar_context', 'bitwise', 'sequence', 'boolean_eager', 'poisson_shared',
    'poisson_checkpoint', 'poisson_persistent', 'poisson_candidate',
])
def test_actual_cpp_capability_export_is_callable_from_c(host_library, name):
    function = getattr(host_library, 'b2_train_'+name+'_v1')
    function.argtypes = []
    function.restype = ctypes.c_uint64
    assert function() == 1


def test_preflight_accepts_all_actual_host_exports(host_library):
    translator().verify_capabilities(host_library._name)


def test_preflight_rejects_mangled_sequence_symbol(tmp_path):
    functions = definitions()
    assert len(functions) == len(translator().CAPABILITIES) == 16
    functions = [function.replace('extern "C" ', '')
                 if 'b2_train_sequence_v1(' in function else function for function in functions]
    library = compile_library(tmp_path, functions)
    with pytest.raises(ValueError, match='CUDA capability export missing: b2_train_sequence_v1'):
        translator().verify_capabilities(library)
