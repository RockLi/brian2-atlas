"""CUDA capability names must be loadable through the native C ABI."""
import ctypes
import os

import pytest
from brian2_rust import training_cuda


@pytest.mark.skipif(os.environ.get("B2_TEST_CUDA_TRAIN") != "1",
                    reason="actual CUDA acceptance required")
def test_cuda_control_capabilities_have_c_linkage(tmp_path):
    # Resolve the compiled library exactly as Rust's dlsym capability checks do.
    library = ctypes.CDLL(str(training_cuda.build(tmp_path)))
    for capability in ("scalar_context", "bitwise", "sequence", "boolean_eager"):
        function = getattr(library, f"b2_train_{capability}_v1")
        function.argtypes = []
        function.restype = ctypes.c_uint64
        assert function() == 1, capability
