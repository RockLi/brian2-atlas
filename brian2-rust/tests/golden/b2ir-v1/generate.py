"""Regenerate the frozen B2IR v1 golden corpus intentionally.

Run from the brian2-rust repository root. Review every resulting diff; changing
these bytes after the v1 freeze requires either a bug-fix migration decision or
a new schema identifier.
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from brian2 import (Clock, Network, NeuronGroup, SpikeMonitor, StateMonitor,
                    ms, prefs, start_scope)

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python"))

from brian2_rust.export import lower_network  # noqa: E402
from brian2_rust.protocol import canonical_bytes, layer_hashes  # noqa: E402

HERE = Path(__file__).resolve().parent


def write(name: str, value: dict) -> None:
    (HERE / name).write_text(
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def old_envelope(model: dict) -> None:
    model["protocol"]["layers"] = layer_hashes(model)


def main() -> None:
    prefs.codegen.target = "numpy"
    start_scope()
    clock = Clock(dt=.25*ms, name="golden_clock")
    group = NeuronGroup(
        2,
        "dv/dt=(drive-v)/(2*ms) : 1 (unless refractory)\n"
        "drive : 1 (constant)",
        threshold="v >= 1", reset="v = 0", refractory=.5*ms,
        method="euler", clock=clock, name="golden_population",
    )
    group.v = np.array([.5, 1.25])
    group.drive = np.array([1.5, 2.0])
    state = StateMonitor(
        group, "v", record=[1, 0], name="golden_state_monitor")
    spike = SpikeMonitor(group, name="golden_spike_monitor")
    model = lower_network(
        Network(group, state, spike, name="golden_network"),
        1*ms, rng_seed=0x0123456789ABCDEF)
    write("minimal-v1.json", model)

    v37 = copy.deepcopy(model)
    v37["schema"] = "b2ir-gate0-probe-v37"
    old_envelope(v37)
    write("minimal-v37.json", v37)

    v36 = copy.deepcopy(model)
    v36["schema"] = "b2ir-gate0-probe-v36"
    for function in v36["definition"]["functions"]:
        function["abi"] = "b2ir-expression-v1"
        function.pop("backend_implementations")
    old_envelope(v36)
    write("minimal-v36.json", v36)

    v35 = copy.deepcopy(v36)
    v35["schema"] = "b2ir-gate0-probe-v35"
    for population in v35["definition"]["populations"]:
        population.pop("linked_variables")
    old_envelope(v35)
    write("minimal-v35.json", v35)

    v34 = copy.deepcopy(v35)
    v34["schema"] = "b2ir-gate0-probe-v34"
    v34.pop("protocol")
    write("minimal-v34.json", v34)

    hashes = {
        "schema": "b2ir-v1",
        "canonical_document_sha256": hashlib.sha256(
            canonical_bytes(model)).hexdigest(),
        "layers": layer_hashes(model),
    }
    write("hashes.json", hashes)


if __name__ == "__main__":
    main()
