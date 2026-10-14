"""Bounded spike statistics for explicit observation windows.

These measurements do not establish scientific equivalence. Sampling, silent
cells, insufficient spike counts and zero-variance exclusions are reported.
The LvR expression follows Shinomoto et al. (2009), as used in the official
INM-6 multi-area model. Its zero-padding convention is reported separately
from the mean over cells with at least three observed spikes.
"""
import numpy as np


def mean_pairwise_correlation(counts):
    """Mean off-diagonal Pearson correlation without an S-by-S matrix."""
    x = np.asarray(counts, dtype=np.float64).copy()
    if x.ndim != 2 or x.shape[1] < 2 or not np.isfinite(x).all():
        raise ValueError('finite cell-by-time counts with at least two bins required')
    x -= x.mean(axis=1, keepdims=True)
    norms = np.sqrt(np.einsum('ij,ij->i', x, x))
    x = x[norms > 0]
    norms = norms[norms > 0]
    n = len(x)
    if n < 2:
        return None, n
    x /= norms[:, None]
    total = x.sum(axis=0)
    off_diagonal = np.dot(total, total) - np.einsum('ij,ij->', x, x)
    return float(off_diagonal / (n * (n - 1))), n


def sampled_lvr(ticks, cells, sample_count, *, dt_ms, refractory_ms):
    """LvR for sample-local cell IDs; insufficient cells retain zero padding."""
    order = np.argsort(cells, kind='stable')
    c, t = cells[order], ticks[order]
    counts = np.bincount(c, minlength=sample_count)
    eligible = counts >= 3
    values = np.zeros(sample_count)
    triples = (c[2:] == c[1:-1]) & (c[1:-1] == c[:-2])
    left = (t[1:-1] - t[:-2])[triples] * dt_ms
    right = (t[2:] - t[1:-1])[triples] * dt_ms
    if np.any(left <= 0) or np.any(right <= 0):
        raise ValueError('successive spikes of a cell must have positive intervals')
    summed = left + right
    terms = (1.0 - 4 * left * right / summed**2) * (1 + 4 * refractory_ms / summed)
    sums = np.bincount(c[2:][triples], weights=terms, minlength=sample_count)
    values[eligible] = 3 * sums[eligible] / (counts[eligible] - 2)
    return values, eligible


def population_activity(ticks, indices, neuron_count, *, start_tick, end_tick,
                        dt_seconds, bin_ticks, seed, sample_size=2000,
                        refractory_ms=2.0):
    """Analyze [start_tick, end_tick); no dense all-cell time matrix is built.

    Uniform, seeded sampling without replacement selects LvR cells from all
    cells and correlation cells from cells with at least one observed spike.
    Dense correlation storage is bounded to sample_size times window bins.
    """
    ticks, indices = np.asarray(ticks), np.asarray(indices)
    if (ticks.ndim != 1 or indices.shape != ticks.shape
            or ticks.dtype.kind not in 'iu' or indices.dtype.kind not in 'iu'
            or type(neuron_count) is not int or not 1 <= neuron_count <= 10_000_000
            or type(start_tick) is not int or type(end_tick) is not int
            or type(bin_ticks) is not int or bin_ticks < 1 or end_tick <= start_tick
            or (end_tick - start_tick) % bin_ticks
            or not np.isfinite(dt_seconds) or dt_seconds <= 0
            or type(sample_size) is not int or not 1 <= sample_size <= 2000
            or not 2 <= (end_tick - start_tick) // bin_ticks <= 20000
            or not np.isfinite(refractory_ms) or refractory_ms < 0):
        raise ValueError('invalid or excessive observation dimensions')
    if np.any(ticks[1:] < ticks[:-1]):
        raise ValueError('spikes must be ordered by time')
    lo, hi = np.searchsorted(ticks, [start_tick, end_tick])
    t, ids = ticks[lo:hi], indices[lo:hi]
    if len(ids) and (ids.min() < 0 or ids.max() >= neuron_count):
        raise ValueError('cell index outside population')
    ids = ids.astype(np.int64, copy=False)
    counts = np.bincount(ids, minlength=neuron_count)
    bins = ((t - start_tick) // bin_ticks).astype(np.int64, copy=False)
    n_bins = (end_tick - start_tick) // bin_ticks
    histogram = np.bincount(bins, minlength=n_bins)
    duration = (end_tick - start_tick) * dt_seconds
    rates = counts / duration
    rng = np.random.default_rng(seed)
    lvr_ids = np.sort(rng.choice(neuron_count, min(neuron_count, sample_size), replace=False))
    active = np.flatnonzero(counts)
    corr_ids = np.sort(rng.choice(active, min(len(active), sample_size), replace=False))

    def selected(sample):
        lookup = np.full(neuron_count, -1, dtype=np.int32)
        lookup[sample] = np.arange(len(sample), dtype=np.int32)
        local = lookup[ids]
        mask = local >= 0
        return local[mask], mask

    local, mask = selected(lvr_ids)
    lvr, eligible = sampled_lvr(t[mask], local, len(lvr_ids),
        dt_ms=dt_seconds * 1000, refractory_ms=refractory_ms)
    local, mask = selected(corr_ids)
    matrix = np.bincount(local.astype(np.int64) * n_bins + bins[mask],
        minlength=len(corr_ids) * n_bins).reshape(len(corr_ids), n_bins)
    correlation, variable = mean_pairwise_correlation(matrix)
    summary = dict(neurons=neuron_count, observed_spikes=int(histogram.sum()),
        mean_rate_hz=float(rates.mean()), rate_std_hz=float(rates.std()),
        rate_quantiles_hz=np.quantile(rates, [0, .25, .5, .75, .95, .99, 1]).tolist(),
        silent_fraction=float(np.count_nonzero(counts == 0) / neuron_count),
        lvr_zero_padded_mean=float(lvr.mean()),
        lvr_eligible_mean=float(lvr[eligible].mean()) if eligible.any() else None,
        lvr_sample_size=len(lvr_ids), lvr_eligible_cells=int(eligible.sum()),
        pairwise_corr_mean=correlation, corr_sample_size=len(corr_ids),
        corr_nonconstant_cells=variable, seed=int(seed))
    return summary, histogram, dict(lvr_ids=lvr_ids, lvr_values=lvr,
        lvr_eligible=eligible, corr_ids=corr_ids, single_cell_rates_hz=rates)
