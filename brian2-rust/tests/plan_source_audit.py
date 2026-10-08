"""Opt-in pytest plugin comparing optimized AOT source with a saved baseline.

Set B2_PLAN_BASELINE_NATIVE to the pre-extraction native.py and load with
``pytest -p plan_source_audit`` (this directory must be on PYTHONPATH).
Canonical-slot models are new functionality, excluded from old-emitter parity.
"""
import importlib.util
import os
from pathlib import Path
import sys


def pytest_configure(config):
    from brian2_rust import native, planner
    path = Path(os.environ["B2_PLAN_BASELINE_NATIVE"])
    spec = importlib.util.spec_from_file_location("brian2_rust._baseline_native", path)
    baseline = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = baseline
    spec.loader.exec_module(baseline)
    original = native.generate_source

    def compare(model, **kwargs):
        actual = original(model, **kwargs)
        if planner.fixed_phase_eligible(model):
            # The existing source-batching tests intentionally patch this knob.
            baseline.SOURCE_BATCH_MIN_EDGES = planner.SOURCE_BATCH_MIN_EDGES
            expected = baseline.generate_source(model)
            assert actual == expected, "plan extraction changed optimized Rust source"
        return actual

    native.generate_source = compare
