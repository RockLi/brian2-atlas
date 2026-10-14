"""Fixed, untrained class-code decoding of physical output counts or currents.

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


def validate_code_rate_weight(weight):
    """Validate the optional training-only rate objective; inference is unchanged."""
    if type(weight) not in (int, float) or not np.isfinite(weight) or not 0 <= weight <= 100:
        raise ValueError('code_rate_weight must be finite in 0..100')
    return float(weight)


def decode_readout(counts, current_sums, bridge, read_ticks):
    """Decode a declared spike or integer-current readout on the host.

    Current sums use recorded output Isyn, normalized by the output threshold.
    They require state recording; no on-chip classifier or energy claim follows.
    """
    source=bridge.get('readout_source')
    if source is None:
        if 'readout_divisor' in bridge:
            raise ValueError('readout divisor requires a current source')
        return decode_counts(counts,bridge,read_ticks)
    divisor=bridge.get('readout_divisor')
    if source!='synaptic_current' or type(divisor) is not int or not 1<=divisor<=32767:
        raise ValueError('invalid current readout contract')
    if current_sums is None or np.asarray(current_sums).shape!=np.asarray(counts).shape:
        raise ValueError('current readout requires matching current sums')
    return decode_counts(current_sums,bridge,read_ticks)/divisor


def validate_readout_contract(bridge, output_thresholds):
    """Reject a decoder that disagrees with the exported neuron thresholds."""
    if bridge.get('readout_source') is None:
        if bridge.get('schema')=='atlas.native-xylo-current-bridge.v1':
            raise ValueError('current bridge requires an explicit readout source')
        if 'readout_divisor' in bridge:
            raise ValueError('readout divisor requires a current source')
        return
    divisor=bridge.get('readout_divisor')
    thresholds=np.asarray(output_thresholds)
    if (bridge['readout_source']!='synaptic_current'
            or bridge.get('schema')!='atlas.native-xylo-current-bridge.v1'
            or bridge.get('source_schema')!='atlas.native-xylo-frozen.v2'
            or bridge.get('dynamics_equivalent') is not True
            or type(divisor) is not int or not 1<=divisor<=32767
            or thresholds.size==0 or not np.all(thresholds==divisor)):
        raise ValueError('current readout must match the exact recurrent output threshold')
