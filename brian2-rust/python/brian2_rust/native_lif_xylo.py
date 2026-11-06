"""Explicit, approximate native LIF port to the existing exact Xylo profile.

No learned weights are changed here. Adding Xylo's synaptic current, integer
rounding, threshold equality and multi-spike behavior changes the model's
transition function. This is not an equivalence-preserving LIF lowering.
"""
import numpy as np
from .native_lif_conversion import validate_frozen, validate_spike_inputs, _require

def validate_xylo_frozen(frozen):
    sizes = validate_frozen(frozen)
    _require(len(sizes) == 3 and sizes[0] <= 16 and sizes[1] <= 64 and sizes[2] <= 8,
             'Xylo bridge requires input<=16, one hidden layer<=64, output<=8')
    for scale, threshold in zip(frozen['scales'], frozen['plan']['threshold']):
        t = scale * threshold
        _require(np.isfinite(t) and t == int(t) and 1 <= t <= 32767,
                 'Xylo scaled thresholds must be exact integers in [1,32767]')
    _require(all(np.any(w) for w in frozen['target_weights']),
             'Xylo bridge requires a nonzero input and readout projection')
    return sizes


def build_network(frozen, raster):
    """Create the adapted Brian/AtlasIR model for one fresh input raster."""
    sizes = validate_xylo_frozen(frozen)
    raster = validate_spike_inputs(frozen, np.asarray(raster)[None])[0]
    from .xylo_network import build_xylo_network
    return build_xylo_network(sizes, frozen['target_weights'],
        np.asarray(frozen['scales'])*frozen['plan']['threshold'], raster)
