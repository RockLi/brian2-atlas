"""Generic deterministic Synapses.connect(j=...) generator regression."""

import sys
from pathlib import Path

import brian2 as b
import numpy as np
import pytest
from brian2.devices.device import all_devices

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import brian2_rust  # noqa: F401, E402


def test_post_index_generator_uses_source_indices_and_state(monkeypatch):
    previous = b.get_device()
    device = all_devices["rust_standalone"]
    device.reinit()
    b.start_scope()
    b.set_device("rust_standalone")
    try:
        source = b.NeuronGroup(
            3, "v : 1\nlabel : integer (constant)",
            threshold="v > 1", reset="v = 0")
        source.label = [0, 1, 0]
        target = b.NeuronGroup(2, "v : 1\nlabel : integer (constant)")
        diagonal = b.Synapses(source, source, on_pre="v_post += 1")
        diagonal.connect(j="i")
        np.testing.assert_array_equal(diagonal.i[:], [0, 1, 2])
        np.testing.assert_array_equal(diagonal.j[:], [0, 1, 2])

        labelled = b.Synapses(source, target, on_pre="v_post += 1")
        labelled.connect(j="label_pre")
        np.testing.assert_array_equal(labelled.i[:], [0, 1, 2])
        np.testing.assert_array_equal(labelled.j[:], [0, 1, 0])

        limit = 2
        filtered = b.Synapses(source, source, on_pre="v_post += 1")
        filtered.connect(j="i if i < limit")
        np.testing.assert_array_equal(filtered.i[:], [0, 1])
        np.testing.assert_array_equal(filtered.j[:], [0, 1])

        monkeypatch.setenv("B2_MAX_CANDIDATE_PAIRS", "1")
        target.label = [0, 1]
        post_label = b.Synapses(source, target)
        post_label.connect("i == label_post", n=2)
        np.testing.assert_array_equal(post_label.i[:], [0, 0, 1, 1])
        np.testing.assert_array_equal(post_label.j[:], [0, 0, 1, 1])

        pre_label = b.Synapses(source, target)
        pre_label.connect("label_pre == j")
        np.testing.assert_array_equal(pre_label.i[:], [0, 1, 2])
        np.testing.assert_array_equal(pre_label.j[:], [0, 1, 0])

        invalid = b.Synapses(source, target, on_pre="v_post += 1")
        with pytest.raises(NotImplementedError, match="generator"):
            invalid.connect(j="i // 2")
    finally:
        b.set_device(previous)
        device.reinit()
        b.start_scope()
