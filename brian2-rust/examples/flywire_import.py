"""Stream official FlyWire v783 Feather batches into a weighted pair CSR.

All proofread root IDs (including isolated nodes) are retained. No contact
threshold is applied. Neuropil rows for the same directed pair are summed.
This imports empirical connectivity, not neuron morphology or fitted dynamics.
"""
import argparse
import json
from pathlib import Path
import resource
import sys
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
from brian2_rust.binary_topology import HEADER, MAGIC, file_hash, inspect_csr

DATASET = 'https://doi.org/10.5281/zenodo.10676866'
OFFICIAL_MD5 = {
    'proofread_connections_783.feather': 'f48f972d262323a102aed49af1396b8a',
    'proofread_root_ids_783.npy': 'e0e6c19732fd8c7a4e39a2d170105421',
}


def batches(path):
    import pyarrow as pa
    import pyarrow.ipc as ipc
    # Seekable OSFile + IPC get_batch reads/decompresses one record batch. read_table(...).slice(...)
    # would still retain the entire decompressed Feather table in memory.
    with pa.OSFile(str(path), 'rb') as source:
        reader = ipc.open_file(source)
        required = ('pre_pt_root_id', 'post_pt_root_id', 'syn_count')
        positions = [reader.schema.get_field_index(name) for name in required]
        if min(positions) < 0:
            raise ValueError(f'missing connection columns: {reader.schema.names}')
        for index in range(reader.num_record_batches):
            batch = reader.get_batch(index)
            yield tuple(batch.column(position).to_numpy(zero_copy_only=False)
                        for position in positions)


def map_rows(ids, arrays):
    pre, post, count = arrays
    if any(array.dtype.kind not in 'iu' for array in arrays):
        raise ValueError('root IDs and syn_count must be integer columns')
    if np.any(pre < 0) or np.any(post < 0):
        raise ValueError('negative root ID')
    # Mixed uint64/int64 comparisons can promote to float64 and lose root-ID bits.
    pre, post = pre.astype(np.uint64), post.astype(np.uint64)
    source, target = np.searchsorted(ids, pre), np.searchsorted(ids, post)
    if (np.any(source >= len(ids)) or np.any(target >= len(ids)) or
            np.any(ids[source] != pre) or np.any(ids[target] != post)):
        raise ValueError('connection references a non-proofread root ID')
    if np.any(count <= 0) or np.any(count > 2**53):
        raise ValueError('contact counts must be positive, exactly representable integers')
    return source, target, count.astype(np.uint64)


def convert(connections, neurons, output, verify_official=True):
    started = time.perf_counter()
    connections, neurons, output = map(Path, (connections, neurons, output))
    checksums = {}
    for path in (connections, neurons):
        digest = file_hash(path, 'md5')
        if verify_official and digest != OFFICIAL_MD5.get(path.name):
            raise ValueError(f'official v783 checksum mismatch: {path.name}')
        checksums[path.name] = {'md5': digest, 'bytes': path.stat().st_size}
    output.mkdir(parents=True, exist_ok=False)
    ids = np.load(neurons, allow_pickle=False)
    if ids.ndim != 1 or ids.dtype.kind not in 'iu' or np.any(ids < 0):
        raise ValueError('invalid root ID vector')
    ids = np.sort(ids.astype(np.uint64))
    if len(ids) == 0 or np.any(ids[1:] == ids[:-1]):
        raise ValueError('root IDs must be unique and nonempty')
    n = len(ids)
    np.save(output / 'root_ids.npy', ids, allow_pickle=False)
    degrees = np.zeros(n, dtype=np.uint64)
    rows, contacts = 0, 0
    for arrays in batches(connections):
        source, target, counts = map_rows(ids, arrays)
        degrees += np.bincount(source, minlength=n).astype(np.uint64)
        rows += len(source)
        contacts += sum(map(int, counts))
    if rows == 0 or contacts > 2**53:
        raise ValueError('empty graph or contact sum outside exact f64 range')
    offsets = np.r_[np.uint64(0), np.cumsum(degrees, dtype=np.uint64)]
    print(f'[scan] {n:,} neurons, {rows:,} neuropil rows, {contacts:,} contacts', flush=True)
    # Disk-backed stable bucket sort by source, then aggregate by target within
    # each source. Peak Python work is O(neurons + batch + maximum out-degree).
    with tempfile.TemporaryDirectory(prefix='csr-staging-', dir=output) as staging:
        staging = Path(staging)
        raw = np.memmap(staging / 'rows.bin', mode='w+',
                        dtype=[('target', '<u4'), ('count', '<u8')], shape=(rows,))
        cursor = offsets[:-1].copy()
        for arrays in batches(connections):
            source, target, counts = map_rows(ids, arrays)
            order = np.argsort(source, kind='stable')
            source, target, counts = source[order], target[order], counts[order]
            starts = np.flatnonzero(np.r_[True, source[1:] != source[:-1]])
            for start, end in zip(starts, np.r_[starts[1:], len(source)], strict=True):
                neuron = int(source[start]); at = int(cursor[neuron]); size = int(end-start)
                raw['target'][at:at+size] = target[start:end]
                raw['count'][at:at+size] = counts[start:end]
                cursor[neuron] += size
        if not np.array_equal(cursor, offsets[1:]):
            raise ValueError('source bucket fill mismatch')
        raw.flush()
        final_offsets = np.zeros(n+1, dtype='<u8')
        in_degrees = np.zeros(n, dtype=np.uint64)
        out_strength = np.zeros(n, dtype=np.uint64)
        in_strength = np.zeros(n, dtype=np.uint64)
        total, autapses = 0, 0
        with (staging/'targets').open('wb') as targets_file, (staging/'weights').open('wb') as weights_file:
            for neuron in range(n):
                a, z = int(offsets[neuron]), int(offsets[neuron+1])
                if a < z:
                    chunk = raw[a:z]
                    order = np.argsort(chunk['target'], kind='stable')
                    targets, counts = chunk['target'][order], chunk['count'][order]
                    starts = np.flatnonzero(np.r_[True, targets[1:] != targets[:-1]])
                    targets = targets[starts]
                    counts = np.add.reduceat(counts, starts)
                    targets_file.write(targets.astype('<u4').tobytes())
                    weights_file.write(counts.astype('<f8').tobytes())
                    total += len(targets)
                    in_degrees[targets] += 1
                    in_strength[targets] += counts
                    out_strength[neuron] = counts.sum()
                    autapses += int(np.count_nonzero(targets == neuron))
                final_offsets[neuron+1] = total
        del raw
        with (output/'connectome.b2csr').open('wb') as dest:
            dest.write(HEADER.pack(MAGIC, n, n, total, 1))
            dest.write(final_offsets.tobytes())
            for name in ('targets', 'weights'):
                with (staging/name).open('rb') as source:
                    for block in iter(lambda: source.read(1024*1024), b''):
                        dest.write(block)
    if int(out_strength.sum()) != contacts or int(in_strength.sum()) != contacts:
        raise ValueError('aggregation lost biological contacts')
    info = inspect_csr(output/'connectome.b2csr')
    def stats(values):
        return dict(mean=float(np.mean(values)), maximum=int(np.max(values)),
                    quantiles=np.quantile(values, [0, .5, .9, .99, 1]).tolist())
    manifest = {
        'schema': 'b2-flywire-import-v1', 'dataset': DATASET if verify_official else 'custom',
        'release': '783.0' if verify_official else None,
        'license': 'CC-BY-4.0' if verify_official else None, 'official_checksums_verified': verify_official,
        'input_files': checksums, 'neurons': n, 'connection_rows': rows,
        'directed_pair_edges': total, 'biological_contacts': contacts,
        'autapse_pairs': autapses, 'contact_threshold': 1,
        'aggregation': 'sum syn_count across neuropils for each directed neuron pair',
        'node_order': 'ascending exact uint64 proofread root ID; isolated nodes retained',
        'columns': ['multiplicity'], 'csr_sha256': file_hash(output/'connectome.b2csr'),
        'csr_bytes': info['file_bytes'], 'out_degree': stats(np.diff(final_offsets)),
        'in_degree': stats(in_degrees), 'out_strength': stats(out_strength),
        'preprocess_seconds': time.perf_counter()-started,
        'peak_rss_bytes': int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) *
                          (1 if sys.platform == 'darwin' else 1024),
        'scope': 'empirical weighted topology only; no coordinates or biophysical parameters',
    }
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest, indent=2), flush=True)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--connections', type=Path, required=True)
    parser.add_argument('--neurons', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-custom-data', action='store_true',
                        help='disable official checksums for fixture/other data; recorded in manifest')
    args = parser.parse_args()
    convert(args.connections, args.neurons, args.output, not args.allow_custom_data)
