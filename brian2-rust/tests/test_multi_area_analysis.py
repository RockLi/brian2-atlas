import numpy as np
import pytest

from brian2_rust.multi_area_analysis import (
    mean_pairwise_correlation, sampled_lvr, population_activity,
)


@pytest.mark.parametrize('n,bins', [(2,4),(9,31),(1,10),(0,10)])
def test_correlation_matches_explicit_pairwise_matrix(n, bins):
    x = np.random.default_rng(19).poisson(1.2, (n, bins))
    if n > 2:
        x[0] = 0  # Silent/constant cells do not have a Pearson coefficient.
    actual, count = mean_pairwise_correlation(x)
    x = x[x.std(axis=1) > 0]
    assert count == len(x)
    if count < 2:
        assert actual is None
    else:
        corr = np.corrcoef(x)
        assert actual == pytest.approx(corr[~np.eye(count, dtype=bool)].mean(), abs=1e-14)


def test_lvr_analytic_intervals_and_insufficient_cells():
    # Cell 0: intervals 10,20 ms. Cell 1: 10,10 ms. Cell 2: one
    # interval; cell 3 is silent. Padding and eligible means differ.
    ticks = np.array([0,0,0,10,10,10,20,30], dtype=np.int64)
    cells = np.array([0,1,2,0,1,2,1,0], dtype=np.int64)
    values, eligible = sampled_lvr(ticks, cells, 4, dt_ms=1, refractory_ms=2)
    assert eligible.tolist() == [True,True,False,False]
    assert values == pytest.approx([19/45,0,0,0])


def test_explicit_window_silence_rates_and_seeded_samples():
    ticks = np.array([9,10,10,20,20,30], dtype=np.int64)
    cells = np.array([2,0,1,0,1,2], dtype=np.int64)
    options = dict(start_tick=10,end_tick=30,dt_seconds=.001,bin_ticks=10,seed=7)
    summary, hist, samples = population_activity(ticks,cells,3,**options)
    assert hist.tolist() == [2,2] and summary['observed_spikes'] == 4
    assert samples['single_cell_rates_hz'].tolist() == [100,100,0]
    assert summary['silent_fraction'] == pytest.approx(1/3)
    assert summary['lvr_eligible_mean'] is None
    assert summary['pairwise_corr_mean'] is None  # Both active trains are constant.
    again = population_activity(ticks,cells,3,**options)[2]
    assert all(np.array_equal(samples[k],again[k]) for k in samples)


@pytest.mark.parametrize('change', [dict(bin_ticks=0),dict(sample_size=2001),
    dict(end_tick=200001),dict(dt_seconds=float('nan')),dict(end_tick=0)])
def test_resource_and_window_dimensions_reject_before_allocation(change):
    options = dict(start_tick=0,end_tick=100,dt_seconds=.001,bin_ticks=1,seed=7)
    options.update(change)
    with pytest.raises(ValueError):
        population_activity(np.array([],dtype=np.int64),np.array([],dtype=np.int64),3,**options)
