"""Immutable encoded scalars for the device's private, large-array IR.

Public exports still use ordinary lists. This storage changes neither B2IR JSON
nor the binary instance format; it avoids one Python string per synaptic value.
"""
from collections.abc import Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import operator
import itertools
import json

import numpy as np

_PACKED_EXPORT = ContextVar("b2ir_packed_export", default=False)
_DTYPES = {"f64": ">f8", "f32": ">f4", "i64": ">i8", "u64": ">u8",
           "i32": ">i4", "u32": ">u4", "bool": "?"}


@contextmanager
def packed_export():
    token = _PACKED_EXPORT.set(True)
    try:
        yield
    finally:
        _PACKED_EXPORT.reset(token)


def should_pack(values):
    return _PACKED_EXPORT.get() and np.size(values) >= 4096


@dataclass(frozen=True, slots=True, eq=False)
class EncodedArray(Sequence):
    _data: bytes
    dtype: str
    _uniform_count: int | None = None

    def __post_init__(self):
        if not isinstance(self._data, bytes) or self.dtype not in _DTYPES:
            raise ValueError("encoded array requires immutable bytes and a storage dtype")
        if len(self._data) % self.width:
            raise ValueError("encoded array payload has an invalid width")
        if self._uniform_count is not None and (
                type(self._uniform_count) is not int or self._uniform_count < 1 or
                len(self._data) != self.width):
            raise ValueError("invalid uniform encoded array")
        values = np.frombuffer(self._data, dtype=_DTYPES[self.dtype])
        if values.dtype.kind == "f" and not np.isfinite(values).all():
            raise ValueError("B2IR values must be finite")
        if self.dtype == "bool" and np.any(np.frombuffer(self._data, dtype=np.uint8) > 1):
            raise ValueError("B2IR bool values must be 00 or 01")

    @classmethod
    def from_values(cls, values, dtype):
        original = np.asarray(values).reshape(-1)
        target = np.dtype(_DTYPES[dtype])
        if (original.size and original.flags.c_contiguous and
                original.dtype.kind == target.kind and
                original.dtype.itemsize == target.itemsize):
            # Check native bytes before any full-array endian conversion. Most
            # fresh synaptic state arrays contain a single repeated value.
            raw = original.view(f"V{original.dtype.itemsize}")
            if all(np.all(raw[start:start + 65536] == raw[0])
                   for start in range(0, len(raw), 65536)):
                first = np.asarray(original[:1], dtype=target)
                if dtype == "bool":
                    first = first.astype(np.uint8)
                return cls(first.tobytes(), dtype, len(original))
        array = np.asarray(values, dtype=_DTYPES[dtype]).reshape(-1)
        if dtype == "bool":
            array = array.astype(np.uint8)
        if array.size:
            raw = np.ascontiguousarray(array).view(f"V{array.dtype.itemsize}")
            if all(np.all(raw[start:start + 65536] == raw[0])
                   for start in range(0, len(raw), 65536)):
                return cls(array[:1].tobytes(), dtype, len(array))
        return cls(array.tobytes(), dtype)

    @property
    def width(self):
        return np.dtype(_DTYPES[self.dtype]).itemsize

    def __len__(self):
        return (self._uniform_count if self._uniform_count is not None
                else len(self._data) // self.width)

    def __getitem__(self, index):
        width = self.width
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            if step == 1:
                if self._uniform_count is not None and stop > start:
                    return type(self)(self._data, self.dtype, stop - start)
                if stop <= start:
                    return type(self)(b"", self.dtype)
                return type(self)(self._data[start * width:stop * width], self.dtype)
            # Strided access is uncommon in the IR but retains list semantics.
            return [self[i] for i in range(start, stop, step)]
        index = operator.index(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError("encoded array index out of range")
        if self._uniform_count is not None:
            return self._data.hex()
        return self._data[index * width:(index + 1) * width].hex()

    def __iter__(self):
        if self._uniform_count is not None:
            yield from itertools.repeat(self._data.hex(), self._uniform_count)
            return
        width = self.width * 2
        for start in range(0, len(self._data), 65536 * self.width):
            encoded = self._data[start:start + 65536 * self.width].hex()
            for index in range(0, len(encoded), width):
                yield encoded[index:index + width]

    def __eq__(self, other):
        if isinstance(other, EncodedArray):
            return (self.width == other.width and len(self) == len(other) and
                    all(a == b for a, b in zip(self._raw_chunks(), other._raw_chunks())))
        if not isinstance(other, Sequence):
            return NotImplemented
        return len(self) == len(other) and all(a == b for a, b in zip(self, other))

    def __deepcopy__(self, memo):
        # The copied byte payload and frozen metadata cannot alias mutable
        # frontend arrays. A build snapshot can safely share this storage.
        return self

    def canonical_chunks(self):
        yield b"["
        width = self.width
        for index, raw in enumerate(self._raw_chunks()):
            if index:
                yield b","
            count = len(raw) // width
            # Hex conversion and insertion into quoted rows happen in C;
            # no per-element Python strings survive or need to be allocated.
            encoded = np.frombuffer(raw.hex().encode("ascii"), dtype=np.uint8)
            rows = np.empty((count, width * 2 + 3), dtype=np.uint8)
            rows[:, 0] = ord('"')
            rows[:, 1:-2] = encoded.reshape(count, width * 2)
            rows[:, -2] = ord('"')
            rows[:, -1] = ord(',')
            yield rows.tobytes()[:-1]
        yield b"]"

    def little_endian_chunks(self):
        width = self.width
        for raw in self._raw_chunks():
            if width == 1:
                yield raw
            else:
                yield np.frombuffer(raw, dtype=np.uint8).reshape(-1, width)[:, ::-1].tobytes()

    def _raw_chunks(self):
        for start in range(0, len(self), 65536):
            if self._uniform_count is not None:
                yield self._data * min(65536, len(self) - start)
            else:
                yield self._data[start * self.width:(start + 65536) * self.width]


@dataclass(frozen=True, slots=True, eq=False)
class IndexArray(Sequence):
    """Private integer JSON arrays, copied into immutable little-endian storage."""

    _data: bytes
    _width: int = 8

    def __post_init__(self):
        if (self._width not in (4, 8) or not isinstance(self._data, bytes) or
                len(self._data) % self._width):
            raise ValueError("index array requires immutable whole unsigned integers")

    @classmethod
    def from_values(cls, values):
        values = np.asarray(values)
        if values.dtype.kind not in "ui" or np.any(values < 0):
            raise ValueError("index array requires nonnegative integers")
        width = 4 if values.size == 0 or values.max() <= 0xffffffff else 8
        return cls(np.asarray(values, dtype=f"<u{width}").tobytes(), width)

    def __array__(self, dtype=None, copy=None):
        result = np.frombuffer(self._data, dtype=f"<u{self._width}")
        if dtype is not None and np.dtype(dtype) != result.dtype:
            if copy is False:
                raise ValueError("dtype conversion requires a copy")
            result = result.astype(dtype)
        elif copy:
            result = result.copy()
        return result

    def __len__(self):
        return len(self._data) // self._width

    def __getitem__(self, index):
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            if step == 1:
                return type(self)(self._data[start * self._width:stop * self._width], self._width)
            return [self[i] for i in range(start, stop, step)]
        index = operator.index(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError("index array index out of range")
        return int.from_bytes(self._data[index * self._width:(index + 1) * self._width], "little")

    def __iter__(self):
        for start in range(0, len(self), 65536):
            yield from np.frombuffer(self._data[start * self._width:(start + 65536) * self._width],
                                     dtype=f"<u{self._width}").tolist()

    def __eq__(self, other):
        if isinstance(other, IndexArray):
            if self._width == other._width:
                return self._data == other._data
            return np.array_equal(np.asarray(self), np.asarray(other))
        if not isinstance(other, Sequence):
            return NotImplemented
        return len(self) == len(other) and all(a == b for a, b in zip(self, other))

    def __deepcopy__(self, memo):
        return self

    def canonical_chunks(self):
        yield b"["
        for start in range(0, len(self), 65536):
            if start:
                yield b","
            values = np.frombuffer(self._data[start * self._width:(start + 65536) * self._width],
                                   dtype=f"<u{self._width}").tolist()
            yield json.dumps(values, separators=(",", ":")).encode("ascii")[1:-1]
        yield b"]"


def index_array(values):
    return IndexArray.from_values(values) if should_pack(values) else values.tolist()
