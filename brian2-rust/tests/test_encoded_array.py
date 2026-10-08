"""Packed private IR stays byte-identical to public list exports."""
import copy
import io
import json
import pickle
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from brian2_rust.encoded_array import (  # noqa: E402
    EncodedArray, IndexArray, index_array, packed_export)
from brian2_rust.export import _array_bits  # noqa: E402
from brian2_rust.native import write_streamed_instance  # noqa: E402
from brian2_rust.protocol import canonical_bytes, write_canonical  # noqa: E402


@pytest.mark.parametrize('dtype,npdtype,values', [
    ('f64', np.float64, [0., -0., 5e-324, -1.25, np.finfo('f8').max]),
    ('f32', np.float32, [0., -0., 2**-149, -1.25, np.finfo('f4').max]),
    ('i64', np.int64, [0, -1, -2**63, 2**63-1]),
    ('u64', np.uint64, [0, 1, 2**64-1]),
    ('i32', np.int32, [0, -1, -2**31, 2**31-1]),
    ('u32', np.uint32, [0, 1, 2**32-1]),
    ('bool', np.bool_, [False, True]),
])
def test_packed_scalars_match_list_json_binary_and_snapshot(tmp_path, dtype, npdtype, values):
    raw = np.resize(np.asarray(values, dtype=npdtype), 65539)
    public = _array_bits(raw, dtype)
    assert type(public) is list
    with packed_export():
        packed = _array_bits(raw, dtype)
    assert isinstance(packed, EncodedArray)
    assert list(packed) == public
    assert packed == public and public == packed
    assert packed[-1] == public[-1]
    assert list(packed[17:65538]) == public[17:65538]
    assert packed[::-2] == public[::-2]
    with pytest.raises(IndexError):
        packed[len(packed)]
    with pytest.raises(TypeError):
        packed[0] = public[0]
    assert copy.deepcopy(packed) is packed
    assert pickle.loads(pickle.dumps(packed)) == packed
    raw[:] = 0  # The snapshot must not alias Brian's mutable frontend storage.
    assert list(packed) == public
    for payload in [packed, {'array': packed, 'empty': []}]:
        expected = public if payload is packed else {'array': public, 'empty': []}
        stream = io.BytesIO()
        write_canonical(payload, stream)
        assert stream.getvalue() == canonical_bytes(expected) == canonical_bytes(payload)
        assert json.loads(stream.getvalue()) == expected
    model = {
        'definition': {'populations': [{'count': len(public), 'steps': 1,
            'dt': struct.pack('>d', .001).hex(), 'states': [{'name': 'v', 'dtype': dtype}],
            'parameters': []}], 'synapses': []},
        'instance': {'rng_seed': 0, 'populations': [{'initial_state': {'v': public},
            'parameters': {}, 'refractory': None}], 'synapses': []},
    }
    write_streamed_instance(model, tmp_path/'list.bin')
    model['instance']['populations'][0]['initial_state']['v'] = packed
    write_streamed_instance(model, tmp_path/'packed.bin')
    assert (tmp_path/'list.bin').read_bytes() == (tmp_path/'packed.bin').read_bytes()


def test_packing_context_is_nested_and_restored_after_failure():
    values = np.zeros(4096)
    assert type(_array_bits(values, 'f64')) is list
    with pytest.raises(RuntimeError):
        with packed_export():
            with packed_export():
                assert isinstance(_array_bits(values, 'f64'), EncodedArray)
            assert isinstance(_array_bits(values, 'f64'), EncodedArray)
            assert type(_array_bits(values[:4], 'f64')) is list
            raise RuntimeError('exercise context restoration')
    assert type(_array_bits(values, 'f64')) is list


def test_packed_input_rejects_invalid_payloads_and_canonicalizes_bool():
    for value in [float('nan'), float('inf'), -float('inf')]:
        with pytest.raises(ValueError, match='finite'):
            EncodedArray.from_values([value], 'f64')
    with pytest.raises(ValueError, match='immutable'):
        EncodedArray(bytearray(8), 'f64')
    with pytest.raises(ValueError, match='width'):
        EncodedArray(b'123', 'f64')
    with pytest.raises(ValueError, match='bool'):
        EncodedArray(b'\xff', 'bool')
    values = np.array([0, 1, 255], dtype=np.uint8).view(np.bool_)
    assert list(EncodedArray.from_values(values, 'bool')) == ['00', '01', '01']
    assert list(EncodedArray.from_values([], 'f64')) == []
    assert b''.join(EncodedArray.from_values([], 'f64').canonical_chunks()) == b'[]'


@pytest.mark.parametrize('value', [0., -0., 5e-324, 1.25])
def test_uniform_storage_preserves_bits_and_sequence_semantics(value):
    raw = np.full(65539, value)
    packed = EncodedArray.from_values(raw, 'f64')
    dense = EncodedArray(raw.astype('>f8').tobytes(), 'f64')
    assert len(packed._data) == 8
    assert packed == dense and dense == packed
    expected = struct.pack('>d', value).hex()
    assert list(packed) == [expected] * len(raw)
    assert packed[-1] == expected
    assert list(packed[65535:]) == [expected] * 4
    assert list(packed[20:10]) == []
    assert packed[::-10000] == list(dense)[::-10000]
    assert b''.join(packed.canonical_chunks()) == canonical_bytes(list(dense))
    assert b''.join(packed.little_endian_chunks()) == raw.astype('<f8').tobytes()
    raw[-1] = -1.
    assert packed == dense
    assert EncodedArray.from_values([0., -0.], 'f64')._uniform_count is None


def test_index_arrays_preserve_unsigned_values_and_snapshot():
    values = np.resize(np.array([0, 1, 2**32-1, 2**63, 2**64-1], dtype='u8'), 65539)
    public = values.tolist()
    assert type(index_array(values)) is list
    with packed_export():
        packed = index_array(values)
        assert type(index_array(values[:3])) is list
    assert isinstance(packed, IndexArray)
    assert packed == public and public == packed
    assert packed[-1] == public[-1]
    assert list(packed[65535:]) == public[65535:]
    assert list(packed[20:10]) == []
    assert packed[::-10000] == public[::-10000]
    assert b''.join(packed.canonical_chunks()) == canonical_bytes(public)
    stream = io.BytesIO()
    write_canonical({'indices': packed}, stream)
    assert stream.getvalue() == canonical_bytes({'indices': public})
    assert copy.deepcopy(packed) is packed
    assert pickle.loads(pickle.dumps(packed)) == packed
    view = np.asarray(packed)
    assert np.array_equal(view, values) and not view.flags.writeable
    with pytest.raises(ValueError):
        view.setflags(write=True)
    values[:] = 0
    assert list(packed) == public
    for invalid in [np.array([-1]), np.array([1.5])]:
        with pytest.raises(ValueError):
            IndexArray.from_values(invalid)
