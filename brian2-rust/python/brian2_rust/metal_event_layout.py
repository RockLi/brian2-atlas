"""Stable event lanes in a population's flag and history buffers.

Keep the existing spike lane at zero, including populations without spike.
Declared custom events get distinct subsequent lanes in definition order.
"""
import numpy as np


def event_coordinates(flags):
    """Decode uint8 tick/neuron flags in canonical tick-major order.

    The byte storage also has a zero-copy boolean view. NumPy's one-dimensional
    boolean scan avoids the slower multidimensional uint8 nonzero path; divmod
    restores coordinates without sorting or changing any recorded event.
    """
    if flags.ndim != 2 or flags.dtype != np.uint8:
        raise ValueError('GPU event flags must be a two-dimensional uint8 array')
    positions = np.flatnonzero(flags.view(np.bool_))
    if not flags.shape[1]:
        return positions, positions.copy()
    return np.divmod(positions, flags.shape[1])


def spike_coordinates(ticks, counts, capacity):
    """Decode per-neuron recorded prefixes in canonical tick/neuron order.

    Sparse recordings gather only populated slots. Dense recordings retain the
    mask path to avoid constructing several event-sized indexing arrays when
    almost every capacity slot is occupied. Neither path reads unused values.
    """
    if type(capacity) is not int or capacity < 0:
        raise ValueError('GPU spike capacity must be a nonnegative integer')
    if counts.ndim != 1 or counts.dtype.kind not in 'iu':
        raise ValueError('GPU spike counts must be a one-dimensional integer array')
    if np.any(counts < 0) or np.any(counts > capacity):
        raise RuntimeError('Metal spike capacity invariant violated')
    n = len(counts)
    slots = n * capacity
    if slots and (ticks.ndim != 1 or ticks.size != slots):
        raise ValueError('GPU spike tick storage does not match capacity')
    repeats = counts.astype(np.intp, copy=False)
    total = int(repeats.sum(dtype=np.int64))
    if not total:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    if total * 4 < slots:
        indices = np.repeat(np.arange(n, dtype=np.int64), repeats)
        starts = np.cumsum(repeats, dtype=np.int64) - repeats
        positions = np.arange(total, dtype=np.int64)
        positions += np.repeat(np.arange(n, dtype=np.int64) * capacity - starts, repeats)
        event_ticks = ticks[positions]
    else:
        mask = np.arange(capacity)[None, :] < counts[:, None]
        indices = np.broadcast_to(np.arange(n)[:, None], mask.shape)[mask]
        event_ticks = ticks.reshape(n, capacity)[mask]
    order = np.lexsort((indices, event_ticks))
    return event_ticks[order], indices[order]


def event_lanes(population):
    return ('spike',)+tuple(event for event in population['events'] if event!='spike')


def event_offset(population,event):
    return event_lanes(population).index(event)*population['count']
