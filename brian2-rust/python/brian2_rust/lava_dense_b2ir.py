"""Experimental static integer B2IR classifier -> stock Lava fixed-point graph.

Reuses the existing B2IR AST admission/algebra proof, not the Akida SDK.
All runtime operations use Lava Dense/Sparse/GradedReluVec processes. This
profile is a CPU SDK deployment, not a Loihi hardware compatibility claim.
"""
import numpy as np

class B2IRLavaDenseError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise B2IRLavaDenseError(message)


def graph_plan(source):
    require(source["schema"] == "atlas.akida.dense-int8-plan.v1", "requires the explicit quantized integer subset")
    shifts = source["shift1"]
    require(len(set(shifts)) == 1, "requires a uniform hidden right shift")
    shift = int(shifts[0])
    step = 1 << shift
    w1 = np.asarray(source["weights1"], dtype=np.int64)
    n_in, hidden = w1.shape
    bs1 = np.asarray(source["bias_shift1"], dtype=np.int64)
    b1 = np.asarray(source["bias1"], dtype=np.int64)
    bias_shifts = sorted(set(bs1.tolist()))
    # Constant channels represent static affine bias, half-up rounding and
    # the upper clip. No sample-dependent arithmetic happens on the host.
    constants = [1 << s for s in bias_shifts] + [step//2, step]
    first = np.zeros((2*hidden, n_in+len(constants)), dtype=np.int64)
    first[:hidden, :n_in] = w1.T
    first[hidden:, :n_in] = w1.T
    for j, bs in enumerate(bs1):
        column = n_in + bias_shifts.index(int(bs))
        first[j, column] = first[j+hidden, column] = b1[j]
    first[:, -2] = 1
    first[hidden:, -1] = -127
    score_columns = [i for i, name in enumerate(source["outputs"]) if name not in source["hidden_states"]]
    w2 = np.asarray(source["weights2"], dtype=np.int64)[:, score_columns].T
    second = np.concatenate((w2, -w2), axis=1)
    score_bias = (np.asarray(source["bias2"], dtype=np.int64) *
                  (1 << np.asarray(source["bias_shift2"], dtype=np.int64)))[score_columns]
    # Mixed-sign Lava weights discard one low bit. Doubling coefficients and
    # subtracting one from the exponent preserves every original odd weight.
    first *= 2
    second *= 2
    require(np.abs(first).max() <= 254 and np.abs(second).max() <= 254,
            "doubled weights exceed exact mixed-sign representation")
    input_bounds = np.asarray([127]*n_in+constants, dtype=np.int64)
    first_bounds = np.abs(first) @ input_bounds
    rectified_bounds = first_bounds // (1 << (shift+1))
    second_bounds = np.abs(second) @ rectified_bounds
    score_bounds = second_bounds//2 + np.abs(score_bias)
    hidden_bounds = 2*(rectified_bounds[:hidden]+rectified_bounds[hidden:])
    require(max(int(first_bounds.max()), int(second_bounds.max()), int(score_bounds.max()),
                int(hidden_bounds.max()), max(constants)) < 2**31,
            "Lava int32 intermediate overflow cannot be excluded")
    return dict(schema="atlas.lava.dense-int8-plan.v1", source_schema=source["source_schema"],
                source_model_sha256=source["source_model_sha256"], source_hashes=source["source_hashes"],
                atlas_validator_sha256=source["atlas_validator_sha256"],
                inputs=source["inputs"], outputs=source["outputs"], hidden_states=source["hidden_states"],
                score_states=[source["outputs"][i] for i in score_columns],
                input_constants=constants, first_weights=first.tolist(), first_weight_exp=-(shift+1),
                second_weights=second.tolist(), second_weight_exp=-1, score_bias=score_bias.tolist(),
                pipeline_delay_ticks=2, input_domain="integer [0,127]",
                bounds=dict(first_accumulator=int(first_bounds.max()), second_accumulator=int(second_bounds.max()),
                            score_with_bias=int(score_bounds.max()), hidden_readback_accumulator=int(hidden_bounds.max())),
                clip_identity="clip(q,0,127) = ReLU(q) - ReLU(q-127)",
                verification_inputs=source["verification_inputs"],
                verification_outputs=source["verification_outputs"],
                target="lava-nc 0.10.0 / Loihi2SimCfg fixed_pt / CPU",
                custom_process_models=False, hardware_mapping_tested=False, physical_hardware=False)


def lower_b2ir(model_path, *, runner):
    from .akida_b2ir import lower_b2ir as lower_integer_b2ir
    return graph_plan(lower_integer_b2ir(model_path, runner=runner, profile="dense-int8-v1"))


def encode_inputs(plan, samples):
    samples = np.asarray(samples)
    require(samples.ndim == 2 and len(samples) > 0 and samples.shape[1] == len(plan["inputs"]),
            "incorrect input dimensions")
    require(np.isfinite(samples).all() and (samples == np.floor(samples)).all() and
            ((samples >= 0) & (samples <= 127)).all(), "inference input outside integer [0,127] domain")
    steps = len(samples) + plan["pipeline_delay_ticks"]
    result = np.zeros((len(plan["inputs"])+len(plan["input_constants"]), steps), dtype=np.int32)
    result[:samples.shape[1], :len(samples)] = samples.T
    result[samples.shape[1]:, :] = np.asarray(plan["input_constants"], np.int32)[:, None]
    return result
