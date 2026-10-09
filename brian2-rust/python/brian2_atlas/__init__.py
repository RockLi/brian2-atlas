"""Atlas execution backend for Brian2: CPU, GPU, MPI and export APIs.

Importing this package registers ``set_device("atlas", engine=...)``.
The implementation package ``brian2_rust`` remains available for compatibility.
"""

from brian2_rust import *  # noqa: F401,F403
from brian2_rust import __all__ as _implementation_exports

__all__ = list(_implementation_exports)
