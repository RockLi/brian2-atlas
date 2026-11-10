"""Exact static binary-input count lowering to stock Specksim IAF.

This is a conversion of final static values, not temporal spike equivalence.
The admitted input domain is binary and is checked both in the source instance
and at inference. No biases, signed weights, active clipping, leak, or nonidentity
readout are approximated. The legacy IAF/event profile is a separate experiment.
"""
import numpy as np


class B2IRSpecksimError(ValueError):
    """Source semantics cannot be preserved by this profile."""


def require(ok, message):
    if not ok:
        raise B2IRSpecksimError(message)


def graph_plan(source):
    """Consume a trusted, validated dense integer plan; prove binary equivalence."""
    require(source['schema'] == 'atlas.akida.dense-int8-plan.v1', 'requires dense integer plan')
    shifts = source['shift1']
    require(len(set(shifts)) == 1 and 1 <= shifts[0] <= 6,
            'requires a uniform power-of-two threshold in [2,64]')
    threshold = 1 << shifts[0]
    w = np.asarray(source['weights1'], dtype=np.int64)
    h = len(source['hidden_states'])
    require(w.shape == (len(source['inputs']), h), 'invalid weight shape')
    require(np.all((w >= 0) & (w <= threshold)), 'requires nonnegative weights <= threshold')
    bias = np.asarray(source['bias1'], dtype=np.int64) * (1 << np.asarray(source['bias_shift1']))
    require(np.all(bias + threshold//2 == 0), 'effective bias/rounding offset must be zero')
    # All binary inputs, not just the observed samples. Nonnegative increments
    # and w <= threshold guarantee at most one spike per serialized input event.
    peaks = w.sum(axis=0) // threshold
    require(np.all(peaks <= 127), 'source clipping can be active on the binary domain')
    outw = np.asarray(source['weights2'], dtype=np.int64)
    outb = np.asarray(source['bias2'], dtype=np.int64) * (1 << np.asarray(source['bias_shift2']))
    require(outw.shape == (h, len(source['outputs'])) and np.all(outb == 0),
            'requires zero output bias and an identity readout')
    indices = []
    for column in outw.T:
        nz = np.flatnonzero(column)
        require(len(nz) == 1 and column[nz[0]] == 1, 'requires identity output columns')
        indices.append(int(nz[0]))
    return dict(schema='atlas.specksim.binary-count-plan.v1',
                inputs=source['inputs'], outputs=source['outputs'], hidden_states=source['hidden_states'],
                weights=w.T.tolist(), threshold=threshold, output_indices=indices,
                binary_max_counts=peaks.tolist(), input_domain='binary',
                source_model_sha256=source['source_model_sha256'], source_hashes=source['source_hashes'],
                atlas_validator_sha256=source['atlas_validator_sha256'],
                observation='final output spike counts; static source has no spike-time contract',
                event_encoding='one event for each nonzero input, in increasing input-index order',
                proof='count=floor(sum(weights*binary_inputs)/threshold); source clipping inactive',
                physical_hardware=False, hardware_mapping_tested=False)


def encode_inputs(plan, samples):
    samples = np.asarray(samples)
    require(samples.ndim == 2 and len(samples) > 0 and samples.shape[1] == len(plan['inputs']),
            'incorrect input dimensions')
    require(np.isin(samples, [0, 1]).all(), 'inference input outside binary domain')
    return samples.astype(np.uint8)
