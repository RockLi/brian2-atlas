"""Portable, source-major static CSR files for empirical connectivity.

Layout: B2CSR001, four little-endian u64 (source/target/edge/column counts),
source_count+1 u64 offsets, edge_count u32 targets, then column-major f64 data.
Column names/units belong to the Brian model, not this numeric payload.
"""
import hashlib
import struct
from pathlib import Path

import numpy as np

MAGIC = b'B2CSR001'
HEADER = struct.Struct('<8sQQQQ')
MAX_EDGES = 100_000_000


def inspect_csr(path, validate=True):
    path = Path(path).expanduser().resolve()
    with path.open('rb') as stream:
        header = stream.read(HEADER.size)
    if len(header) != HEADER.size:
        raise ValueError('truncated binary CSR header')
    magic, sources, targets, edges, columns = HEADER.unpack(header)
    if (magic != MAGIC or not 1 <= sources <= 1_000_000 or
            not 1 <= targets <= 1_000_000 or not 1 <= edges <= MAX_EDGES or
            not 0 <= columns <= 128):
        raise ValueError('invalid binary CSR header')
    target_offset = HEADER.size + (sources + 1) * 8
    parameter_offset = target_offset + edges * 4
    expected = parameter_offset + edges * columns * 8
    if path.stat().st_size != expected:
        raise ValueError('binary CSR file length mismatch')
    info = dict(path=str(path), source_count=sources, target_count=targets,
                edge_count=edges, column_count=columns,
                target_offset=target_offset, parameter_offset=parameter_offset,
                file_bytes=expected)
    if validate:
        offsets, target, values = csr_arrays(info)
        if offsets[0] != 0 or offsets[-1] != edges or np.any(offsets[1:] < offsets[:-1]):
            raise ValueError('invalid binary CSR offsets')
        for start in range(0, edges, 65536):
            end = min(start + 65536, edges)
            if np.any(target[start:end] >= targets):
                raise ValueError('binary CSR target outside population')
            if not np.isfinite(values[:, start:end]).all():
                raise ValueError('nonfinite binary CSR parameter')
    return info


def csr_arrays(info):
    path = info['path']
    offsets = np.memmap(path, mode='r', dtype='<u8', offset=HEADER.size,
                        shape=(info['source_count'] + 1,))
    targets = np.memmap(path, mode='r', dtype='<u4', offset=info['target_offset'],
                        shape=(info['edge_count'],))
    values = (np.memmap(path, mode='r', dtype='<f8', offset=info['parameter_offset'],
                        shape=(info['column_count'], info['edge_count']))
              if info['column_count'] else np.empty((0, info['edge_count'])))
    return offsets, targets, values


def file_hash(path, algorithm='sha256'):
    digest = hashlib.new(algorithm)
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def copy_region(source, destination, offset, count):
    with Path(source).open('rb') as stream:
        stream.seek(offset)
        while count:
            chunk = stream.read(min(count, 1024 * 1024))
            if not chunk:
                raise ValueError('binary CSR changed/truncated while copying')
            destination.write(chunk)
            count -= len(chunk)
