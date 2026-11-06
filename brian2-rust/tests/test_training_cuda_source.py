"""CUDA helper qualification, checked by a real CUDA semantic frontend."""
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
    spec = importlib.util.spec_from_file_location('cuda_translation_under_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def scalar_body(source):
    match = re.search(r'(?m)^(?:__device__ )?float atlas_scalar_equation\(', source)
    assert match
    opening = source.index('{', match.start())
    depth = 1
    end = opening+1
    while depth:
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    return source[match.start():end]


def test_all_cuda_shader_helpers_are_device_functions():
    source = translator().kernel_source()
    declarations = re.findall(r'(?m)^((?:__device__ |__global__ )?[A-Za-z_]\w*\s+)(atlas_\w+)\(', source)
    helpers = [(prefix, name) for prefix, name in declarations if '__global__' not in prefix]
    assert helpers and any(name == 'atlas_scalar_equation' for _, name in helpers)
    assert all('__device__' in prefix for prefix, name in helpers), helpers
    assert source.count('__global__ void ') == 7


@pytest.mark.parametrize('remove_device', [False, True])
def test_real_cuda_frontend_checks_scalar_helper_calls(tmp_path, remove_device):
    compiler = shutil.which('clang')
    assert compiler, 'clang CUDA semantic frontend required for source acceptance'
    source = translator().kernel_source()
    declarations = []
    for name in ('atlas_equation', 'atlas_state_equation'):
        match = re.search(r'(?m)^__device__ float '+name+r'\([\s\S]*?\)', source)
        assert match
        declarations.append(match.group(0)+';')
    helper = scalar_body(source)
    if remove_device:
        helper = helper.replace('__device__ ', '', 1)
    # Retain the actual generated scalar body and its actual callee signatures.
    # Builtins avoid a CUDA SDK dependency; clang still enforces host/device calls.
    text = ('#define __device__ __attribute__((device))\n'
            'using uint64_t = unsigned long long;\n'
            '#define isfinite __builtin_isfinite\n'
            '#define NAN __builtin_nanf("")\n'+
            '\n'.join(declarations)+'\n'+helper+'\n')
    path = tmp_path/'scalar.cu'
    path.write_text(text)
    result = subprocess.run([compiler, '-x', 'cuda', '--cuda-host-only',
        '-nocudainc', '-nocudalib', '-fsyntax-only', str(path)],
        capture_output=True, text=True, timeout=30)
    if remove_device:
        assert result.returncode != 0
        assert 'call to __device__ function from __host__ function' in result.stderr
    else:
        assert result.returncode == 0, result.stderr
