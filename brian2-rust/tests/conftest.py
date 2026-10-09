"""Keep standalone/backend preference changes local to each test."""
from copy import deepcopy
import brian2 as b
import pytest


@pytest.fixture(autouse=True)
def restore_brian_preferences():
    # Benchmark helpers deliberately select float32 and compiler options in
    # their isolated workers. Direct unit calls must not leak those choices
    # into later tests that construct the frozen f64 summed-variable contract.
    previous_device = b.get_device()
    before = deepcopy(dict(b.prefs))
    try:
        yield
    finally:
        if b.get_device() is not previous_device:
            b.set_device(previous_device)
        for key, value in before.items():
            b.prefs[key] = value
