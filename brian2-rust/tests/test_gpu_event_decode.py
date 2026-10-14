"""Event decoding must preserve byte truth, ordering and independent outputs."""
import numpy as np
import pytest
from brian2_rust.metal_event_layout import event_coordinates


@pytest.mark.parametrize('shape',[(0,0),(0,7),(4,0),(1,1),(7,13),(5,256)])
@pytest.mark.parametrize('layout',['contiguous','transpose','reverse','stride'])
def test_event_coordinates_match_numpy_for_all_byte_values(shape,layout):
    flags=(np.arange(np.prod(shape),dtype=np.uint64)%256).astype(np.uint8).reshape(shape)
    if layout=='transpose':flags=flags.T
    elif layout=='reverse':flags=flags[::-1,::-1]
    elif layout=='stride':flags=flags[::2,::2]
    flags.setflags(write=False)
    expected=np.nonzero(flags)
    actual=event_coordinates(flags)
    for a,e in zip(actual,expected,strict=True):
        assert a.dtype==e.dtype and a.shape==e.shape and a.tobytes()==e.tobytes()
        assert not np.shares_memory(a,flags)


@pytest.mark.parametrize('density',[0.,.001,.1,1.])
def test_sparse_and_dense_history_order_and_ownership(density):
    flags=(np.random.default_rng(1729).random((64,1024))<density).astype(np.uint8)
    expected=np.nonzero(flags);actual=event_coordinates(flags)
    flags.fill(0)
    for a,e in zip(actual,expected,strict=True):np.testing.assert_array_equal(a,e)


@pytest.mark.parametrize('flags',[np.zeros(3,np.uint8),np.zeros((2,3),np.uint32)])
def test_invalid_event_buffer_layout_is_rejected(flags):
    with pytest.raises(ValueError,match='two-dimensional uint8'):event_coordinates(flags)
