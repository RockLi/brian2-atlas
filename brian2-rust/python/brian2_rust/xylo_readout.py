"""Fixed, untrained class-code decoding of physical output spike counts.

Integer count sums precede floating logit scaling, matching the native trainer.
The decoder runs on the host; it is not additional on-chip neural computation.
"""
import numpy as np


def validate_codes(codes, outputs):
    a = np.asarray(codes)
    if (a.ndim != 2 or not 2 <= len(a) <= 32 or a.shape[1] != outputs
            or not np.isin(a, [0, 1]).all() or not (a.sum(1) > 0).all()
            or not (a.sum(1) == a.sum(1)[0]).all()
            or len(np.unique(a, axis=0)) != len(a)):
        raise ValueError('class codes require distinct nonzero binary rows of equal Hamming weight')
    return a.astype(np.int64)


def decode_counts(counts, bridge, read_ticks):
    counts = np.asarray(counts, dtype=np.int64)
    if 'class_codes' in bridge:
        counts = counts @ validate_codes(bridge['class_codes'], counts.shape[-1]).T
    return counts * bridge['logit_scale'] / max(1, read_ticks)
