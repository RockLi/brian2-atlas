"""Deduplicated TimedArray buffers shared by Metal, CUDA and CPU controls."""
import struct

import numpy as np

from .plan import PlanValidationError


def timed_nodes(tree, functions=(), seen=None):
    seen = set() if seen is None else seen
    if isinstance(tree, dict):
        if tree.get("op") == "timed_array":
            yield tree
        if tree.get("op") == "call" and tree["function"] not in seen:
            seen.add(tree["function"])
            yield from timed_nodes(functions[tree["function"]]["body"], functions, seen)
        for key, value in tree.items():
            if key != "values":
                yield from timed_nodes(value, functions, seen)
    elif isinstance(tree, (list, tuple)):
        for value in tree:
            yield from timed_nodes(value, functions, seen)


class TimedTables:
    def __init__(self, tree, functions, *, buffer, offset):
        self.buffer = buffer
        self.offsets = {}
        values = []
        for node in timed_nodes(tree, functions):
            key = tuple(node["values"])
            if key not in self.offsets:
                self.offsets[key] = offset + len(values)
                values.extend(key)
            epsilon = struct.unpack(">d", bytes.fromhex(node["epsilon"]))[0]
            with np.errstate(over="ignore", under="ignore"):
                eps = np.float32(epsilon)
            if not np.isfinite(eps) or eps < np.finfo(np.float32).tiny:
                raise PlanValidationError("GPU TimedArray epsilon must be a positive normal float32")
        with np.errstate(over="ignore"):
            self.values = np.asarray([struct.unpack(">d", bytes.fromhex(v))[0] for v in values], dtype=np.float32)
        if not np.isfinite(self.values).all():
            raise PlanValidationError("GPU TimedArray values must be finite float32")

    def address(self, node):
        return f"{self.buffer}, {self.offsets[tuple(node['values'])]}ul"


SOURCE = r'''
inline float b2_timed_array(device const float *values, ulong offset,
                            uint rows, uint width, float epsilon, float upsampling,
                            float time, float column, thread bool *fault) {
    if (!(isfinite(column) && column>=0.0f && column<float(width))) {
        *fault=true; return 0.0f;
    }
    float raw=(time/epsilon+0.5f)/upsampling;
    // Saturate before converting to an integer, including +/-infinity and NaN.
    // B2IR's saturating conversion maps NaN/negative time to the first row.
    uint row=!(raw>0.0f) ? 0u : (raw>=float(rows-1) ? rows-1 : uint(raw));
    return values[offset+ulong(row)*width+uint(column)];
}
'''
