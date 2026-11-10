"""Input-domain and numerical-admission boundaries for the Lava dense graph."""
import numpy as np
import pytest
from brian2_rust.lava_dense_b2ir import B2IRLavaDenseError, encode_inputs, graph_plan


@pytest.mark.parametrize("invalid", [-1, 128, 0.5, float("nan"), float("inf"), 2**32])
def test_reject_outside_proven_domain_before_int_cast(invalid):
    plan = dict(inputs=["x"], input_constants=[64, 128], pipeline_delay_ticks=2)
    with pytest.raises(B2IRLavaDenseError, match="integer"):
        encode_inputs(plan, np.array([[invalid]]))


@pytest.mark.parametrize("invalid", [np.array([]), np.zeros((0, 1)), np.zeros((2, 2))])
def test_reject_input_shape(invalid):
    with pytest.raises(B2IRLavaDenseError, match="dimensions"):
        encode_inputs(dict(inputs=["x"]), invalid)


def test_preserve_boundary_pixels_and_flush_without_wrapping():
    plan = dict(inputs=["x"], input_constants=[64, 128], pipeline_delay_ticks=2)
    actual = encode_inputs(plan, np.array([[0], [127]]))
    np.testing.assert_array_equal(actual, [[0, 127, 0, 0], [64]*4, [128]*4])
    assert actual.dtype == np.int32


def test_reject_nonuniform_hidden_shifts():
    with pytest.raises(B2IRLavaDenseError, match="uniform"):
        graph_plan(dict(schema="atlas.akida.dense-int8-plan.v1", shift1=[6, 7]))


def test_reject_expanded_graph_overflow_despite_small_clipped_output():
    # Saturating the hidden state would bound the final answer, but the
    # expanded ReLU-difference graph must also keep *both* branches safe.
    hidden = [f"h{i}" for i in range(20)]
    source = dict(schema="atlas.akida.dense-int8-plan.v1", shift1=[1]*20,
                  weights1=np.full((1024, 20), 127).tolist(), bias1=[0]*20,
                  bias_shift1=[0]*20, outputs=hidden+["s0"], hidden_states=hidden,
                  weights2=np.column_stack((np.eye(20, dtype=int), np.full(20, 127))).tolist(),
                  bias2=[0]*21, bias_shift2=[0]*21)
    with pytest.raises(B2IRLavaDenseError, match="overflow"):
        graph_plan(source)
