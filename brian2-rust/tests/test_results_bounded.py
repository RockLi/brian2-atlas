"""Bounded validation matches scalar rules and preserves mmap-backed results."""
import importlib.util
import json
import mmap
import os
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('bounded_result_reader', ROOT/'python/brian2_rust/results.py')
reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader)


def scalar_refractory(ticks, indices, initial, dt, period, full):
    last = {}
    for tick, neuron in zip(map(int, ticks), map(int, indices), strict=True):
        if period is not None:
            if neuron in last:
                assert tick-last[neuron] >= period
            elif full:
                assert np.trunc((tick*dt-initial[neuron]+1e-3*dt)/dt) >= period
        last[neuron] = tick
    neurons = np.array(sorted(last), dtype=np.int64)
    return neurons, np.array([last[n]*dt for n in neurons])


@pytest.mark.parametrize('dtype', ['<i8', '<u4'])
@pytest.mark.parametrize('chunk', [1, 2, 7, 131072])
@pytest.mark.parametrize('period,full', [(None, True), (3, True), (3, False)])
def test_chunk_refractory_matches_scalar_oracle(monkeypatch, dtype, chunk, period, full):
    monkeypatch.setattr(reader, 'VALIDATION_CHUNK', chunk)
    pairs = [(t,n) for t in range(200) for n in range(13) if (t+n*7)%17 == 0]
    ticks, indices = np.asarray(pairs, dtype=dtype).T
    initial = np.full(14, -1.0)
    expected = scalar_refractory(ticks, indices, initial, .0001, period, full)
    actual = reader._refractory_events(ticks, indices, initial, .0001, period, full, lambda _:None)
    for a,b in zip(actual,expected,strict=True):np.testing.assert_array_equal(a,b)


@pytest.mark.parametrize('dtype', ['<i8', '<u4'])
@pytest.mark.parametrize('chunk', [1,2,3])
def test_cross_chunk_refractory_violation_is_not_lost(monkeypatch,dtype,chunk):
    monkeypatch.setattr(reader,'VALIDATION_CHUNK',chunk)
    ticks=np.array([0,1,2,3],dtype=dtype);indices=np.array([0,1,2,0],dtype=dtype)
    with pytest.raises(ValueError,match='refractory'):
        reader._refractory_events(ticks,indices,np.full(3,-1.),.001,4,True,lambda _:None)


def test_first_appearance_in_later_chunk_uses_initial_state(monkeypatch):
    monkeypatch.setattr(reader,'VALIDATION_CHUNK',2)
    ticks=np.array([10,11,12,13]);indices=np.array([0,1,0,2])
    initial=np.array([-1.,-1.,.012])
    with pytest.raises(ValueError,match='refractory'):
        reader._refractory_events(ticks,indices,initial,.001,2,True,lambda _:None)
    # A bounded monitor must not apply its preceding hidden initial state.
    reader._refractory_events(ticks,indices,initial,.001,2,False,lambda _:None)


def test_empty_history_and_no_final_events():
    empty=np.array([],dtype='<u4')
    n,t=reader._refractory_events(empty,empty,np.zeros(3),.1,2,True,lambda _:None)
    assert len(n)==len(t)==0
    assert len(reader._last_events(empty,empty,9))==0
    assert len(reader._last_events(np.array([1,2]),np.array([0,1]),3))==0


@pytest.mark.parametrize('chunk',[1,3,8])
def test_last_event_suffix_and_chunk_ranges(monkeypatch,chunk):
    monkeypatch.setattr(reader,'VALIDATION_CHUNK',chunk)
    ticks=np.array([2,3,3,3,3,3],dtype='<u4');indices=np.array([0,0,1,2,3,4],dtype='<u4')
    assert reader._strict_event_order(ticks,indices)
    assert reader._event_range(ticks,indices,2,4,5,lambda _:None)
    np.testing.assert_array_equal(reader._last_events(ticks,indices,3),np.arange(5))
    np.testing.assert_array_equal(reader._event_counts(indices,5,lambda _:None),[2,1,1,1,1])
    ticks[3]=1
    assert not reader._strict_event_order(ticks,indices)
    assert not reader._event_range(ticks,indices,2,4,5,lambda _:None)


def test_sort_and_bincount_inputs_are_bounded(monkeypatch):
    monkeypatch.setattr(reader,'VALIDATION_CHUNK',7)
    sort=reader.np.argsort;bincount=reader.np.bincount
    def bounded_sort(a,*args,**kwargs):
        assert len(a)<=7
        return sort(a,*args,**kwargs)
    def bounded_count(a,*args,**kwargs):
        assert len(a)<=7
        return bincount(a,*args,**kwargs)
    monkeypatch.setattr(reader.np,'argsort',bounded_sort)
    monkeypatch.setattr(reader.np,'bincount',bounded_count)
    ticks=np.arange(1000,dtype='<u4');indices=ticks%5
    reader._refractory_events(ticks,indices,np.full(5,-1.),.001,5,True,lambda _:None)
    np.testing.assert_array_equal(reader._event_counts(indices,5,lambda _:None),np.full(5,200))


def _assert_equal(a,b):
    if isinstance(a,dict):
        assert set(a)==set(b)
        for key in a:_assert_equal(a[key],b[key])
    elif isinstance(a,list):
        assert len(a)==len(b)
        for x,y in zip(a,b,strict=True):_assert_equal(x,y)
    elif isinstance(a,np.ndarray):
        assert a.dtype==b.dtype
        np.testing.assert_array_equal(a,b)
    else:assert a==b


SUPPORTS_CACHE = (hasattr(mmap.mmap,'madvise') and hasattr(mmap,'MADV_DONTNEED')
                  and hasattr(os,'posix_fadvise') and hasattr(os,'POSIX_FADV_DONTNEED'))


@pytest.mark.skipif(not SUPPORTS_CACHE,reason='requires POSIX mmap/file cache advice')
@pytest.mark.parametrize('include_times',[True,False])
def test_cache_release_keeps_returned_views_and_closes_fd(monkeypatch,include_times):
    model=json.loads((ROOT/'tests/fixtures/reference-v1/model.json').read_text())
    folder=ROOT/'tests/fixtures/reference-v1/reference'
    expected=reader.load_results(model,folder,include_times=include_times)
    closed=[];close=reader.os.close
    def track(fd):closed.append(fd);close(fd)
    monkeypatch.setattr(reader.os,'close',track)
    actual=reader.load_results(model,folder,include_times=include_times,release_file_cache=True)
    assert len(closed)==2
    _assert_equal(actual,expected)
    assert not actual['populations'][0]['trace']['v'].flags.owndata


@pytest.mark.skipif(not SUPPORTS_CACHE,reason='requires POSIX mmap/file cache advice')
def test_cache_descriptors_close_on_corrupt_input(monkeypatch,tmp_path):
    model=json.loads((ROOT/'tests/fixtures/reference-v1/model.json').read_text())
    for n in ['results.bin','events.bin','summary.json']:shutil.copyfile(ROOT/'tests/fixtures/reference-v1/reference'/n,tmp_path/n)
    with (tmp_path/'results.bin').open('r+b') as f:f.write(b'BROKEN!!')
    closed=[];close=reader.os.close
    def track(fd):closed.append(fd);close(fd)
    monkeypatch.setattr(reader.os,'close',track)
    with pytest.raises(RuntimeError,match='marker'):
        reader.load_results(model,tmp_path,release_file_cache=True)
    assert len(closed)==1


def test_cache_release_requires_explicit_bool():
    with pytest.raises(TypeError,match='release_file_cache'):
        reader.load_results({},Path('/nonexistent'),release_file_cache=1)
