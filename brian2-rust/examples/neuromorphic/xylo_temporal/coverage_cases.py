"""Deterministic differential cases for the admitted temporal Xylo subset."""
import numpy as np

SEED = 20261013


def cases():
    rng = np.random.default_rng(SEED)
    result = {}

    def network(nin, hidden, output, steps, index):
        def weights(shape):
            w = rng.integers(-128, 128, shape)
            w[rng.random(shape) < (0.8 if index % 2 else 0.15)] = 0
            w.flat[0] = -128
            if w.size > 1:
                w.flat[-1] = 127
            return w.tolist()

        def parameters(n):
            # Rotate through every legal shift, including slow negative decay.
            syn = (np.arange(n) + index) % 16
            mem = (np.arange(n) + index) % 15 + 1
            threshold = rng.choice([1, 2, 31, 127, 1024, 32767], n)
            bias = rng.choice([-32768, -127, -1, 0, 1, 127, 32767], n)
            return dict(syn=syn.tolist(), mem=mem.tolist(),
                        threshold=threshold.tolist(), bias=bias.tolist())

        raster = rng.integers(0, 16, (steps, nin))
        raster[rng.random(raster.shape) < 0.7] = 0
        raster[0, 0] = 15
        return dict(raster=raster.tolist(), input_weights=weights((nin, hidden)),
                    recurrent_weights=weights((hidden, hidden)),
                    readout_weights=weights((hidden, output)),
                    hidden=parameters(hidden), output=parameters(output))

    for i in range(32):
        nin = int(rng.integers(1, 17))
        hidden = int(rng.integers(8, 33))
        output = int(rng.integers(1, 9))
        result[f'random_{i:02d}'] = network(nin, hidden, output, 97 + i, i)
    result['one_tick'] = network(1, 1, 1, 1, 36)
    result['one_neuron'] = network(1, 1, 1, 97, 32)
    result['maximum_profile_dimensions'] = network(16, 64, 8, 257, 33)
    result['maximum_duration'] = network(2, 3, 2, 10000, 34)
    c = network(1, 1, 1, 31, 35)
    # An impulse immediately before a 7-step boundary must reach the readout
    # in the next chunk; later recurrence must survive silent input chunks.
    c.update(raster=[[0] for _ in range(31)], input_weights=[[127]],
             recurrent_weights=[[4]], readout_weights=[[127]],
             hidden=dict(syn=[2], mem=[2], threshold=[8], bias=[0]),
             output=dict(syn=[2], mem=[2], threshold=[8], bias=[0]))
    c['raster'][6][0] = 15
    result['chunk_boundary_impulse'] = c
    return result


def parameter_coverage(networks):
    """Observed input values, not a percentage of all semantic combinations."""
    configs = list(networks.values())
    return dict(seed=SEED, generated_cases=len(configs),
                synaptic_shifts=sorted({v for c in configs for role in ('hidden', 'output') for v in c[role]['syn']}),
                membrane_shifts=sorted({v for c in configs for role in ('hidden', 'output') for v in c[role]['mem']}),
                input_counts=sorted({v for c in configs for row in c['raster'] for v in row}),
                weight_extrema=[min(v for c in configs for k in ('input_weights', 'recurrent_weights', 'readout_weights') for row in c[k] for v in row),
                                max(v for c in configs for k in ('input_weights', 'recurrent_weights', 'readout_weights') for row in c[k] for v in row)])
