#! /usr/bin/env python
'''
Brian2 setup script
'''

# isort:skip_file

import os
import numpy
from setuptools import setup, Extension, find_packages
# PEP 517 frontends need not put the source root on sys.path.
import importlib.util
from pathlib import Path
_atlas_spec = importlib.util.spec_from_file_location("_atlas_build", Path(__file__).with_name("_atlas_build.py"))
_atlas_build = importlib.util.module_from_spec(_atlas_spec)
_atlas_spec.loader.exec_module(_atlas_build)
AtlasBuildPy = _atlas_build.AtlasBuildPy
from typing import List

# A Helper function to require cython extension
def require_cython_extension(module_path, module_name,extra_include_dirs=None):
    """
    Create a cythonized Extension object from a .pyx source.
    """
    # File paths
    base_path = os.path.join(*module_path)
    pyx_file = os.path.join(base_path, f"{module_name}.pyx")

    # Module name for setuptools
    full_module_name = ".".join(module_path + [module_name])

    include_dirs = [numpy.get_include()]
    if extra_include_dirs:
        include_dirs.extend(extra_include_dirs)

    ext = Extension(full_module_name, [pyx_file],include_dirs=include_dirs)
    return ext


# Collect Extensions
extensions : List[Extension]=[]

# Now Cython is required and no python fallback is possible
spike_queue_ext = require_cython_extension(
    module_path=["brian2", "synapses"],
    module_name="cythonspikequeue",
)

extensions.append(spike_queue_ext)

dynamic_array_ext = require_cython_extension(
    module_path=["brian2", "memory"],
    module_name="cythondynamicarray",
    extra_include_dirs=["brian2/devices/cpp_standalone/brianlib"]
)

extensions.append(dynamic_array_ext)


setup(
    ext_modules=extensions,
    packages=find_packages(include=["brian2", "brian2.*"])
             + find_packages(where="brian2-rust/python"),
    package_dir={"brian2_rust": "brian2-rust/python/brian2_rust",
                 "brian2_atlas": "brian2-rust/python/brian2_atlas"},
    package_data={"brian2_rust": ["*.c", "*.cu", "*.h", "*.m", "*.metal", "*.toml", "*.rs", "*.cpp",
                                "metal_runtime/*", "mpi_runtime/*"]},
    cmdclass={"build_py": AtlasBuildPy},
)
