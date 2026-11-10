"""Keep implicit nvcc option injection outside the declared CUDA plan contract."""
import hashlib
import os
from .protocol import canonical_bytes

POLICY='b2-cuda-explicit-options-v1'
INJECTED_FLAGS=('NVCC_PREPEND_FLAGS','NVCC_APPEND_FLAGS')


def compilation_environment():
    """Snapshot subprocess environment without mutating the calling process.

    Toolchain discovery and unrelated environment remain available. Only nvcc's
    two implicit command-line injection variables are removed. Names, never their
    values, are exposed as diagnostics; options come from the validated plan.
    """
    environment=dict(os.environ)
    ignored=[name for name in INJECTED_FLAGS if name in environment]
    for name in INJECTED_FLAGS:environment.pop(name,None)
    return environment,dict(policy=POLICY,ignored_environment_variables=ignored)


def binary_identity(source,architecture,nvcc_version,options):
    # New namespace deliberately invalidates pre-policy binaries: old cache
    # entries may have been compiled with unreported inherited options.
    record=dict(policy=POLICY,source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                architecture=architecture,nvcc_version=nvcc_version,options=list(options))
    return hashlib.sha256(canonical_bytes(record)).hexdigest()
