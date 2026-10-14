"""Attach auditable transmitter signs and olfactory cell sets to full v783 CSR."""
import argparse
import csv
import json
from pathlib import Path
import sys
import time
import resource

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'python'))
from brian2_rust.binary_topology import HEADER, MAGIC, inspect_csr, csr_arrays, copy_region, file_hash
from flywire_import import OFFICIAL_MD5, map_rows

ANNOTATION_SHA256 = '30be6c73975a70c56d930e27911f36455d3886e15abf383b78edd2a5d679e0b6'
NT = ['gaba', 'acetylcholine', 'glutamate', 'octopamine', 'serotonin', 'dopamine']
COLUMNS = ['gaba_avg', 'ach_avg', 'glut_avg', 'oct_avg', 'ser_avg', 'da_avg']
FAST_SIGN = {'acetylcholine': 1, 'gaba': -1, 'glutamate': -1, 'histamine': -1}


def transmitter_sign(predicted, known):
    """Unambiguous curated fast-transmitter evidence overrides EM predictions.

    Monoamines/neuropeptides alone have no fast current in this reduced model.
    Conflicting curated E/I evidence is disabled, rather than silently guessed.
    """
    signs = {FAST_SIGN[x.strip().lower()] for x in known.split(',')
             if x.strip().lower() in FAST_SIGN}
    if len(signs) > 1:
        return 0, 'curated_EI_conflict'
    if signs:
        return signs.pop(), 'curated_fast_transmitter'
    return FAST_SIGN.get(predicted, 0), 'EM_prediction'


def convert(graph, connections, annotations, output):
    import pyarrow as pa
    import pyarrow.ipc as ipc
    started=time.perf_counter()
    graph, connections, annotations, output = map(Path, (graph, connections, annotations, output))
    if file_hash(connections, 'md5') != OFFICIAL_MD5['proofread_connections_783.feather']:
        raise ValueError('official connection checksum mismatch')
    if file_hash(annotations) != ANNOTATION_SHA256:
        raise ValueError('expected official annotation release v2.1.0')
    original = json.loads((graph/'manifest.json').read_text())
    info = inspect_csr(graph/'connectome.b2csr')
    if file_hash(info['path']) != original['csr_sha256']:
        raise ValueError('original graph checksum mismatch')
    ids = np.load(graph/'root_ids.npy')
    n = len(ids)
    scores, contacts, predicted_contacts = np.zeros((n, 6)), np.zeros(n), np.zeros(n)
    missing_rows=0
    with pa.OSFile(str(connections), 'rb') as f:
        reader = ipc.open_file(f)
        names = ['pre_pt_root_id', 'post_pt_root_id', 'syn_count', *COLUMNS]
        positions = [reader.schema.get_field_index(x) for x in names]
        if min(positions) < 0:
            raise ValueError('missing transmitter score columns')
        for batch_id in range(reader.num_record_batches):
            batch = reader.get_batch(batch_id)
            values = [batch.column(i).to_numpy(zero_copy_only=False) for i in positions]
            source, _, count = map_rows(ids, values[:3])
            contacts += np.bincount(source, weights=count, minlength=n)
            valid=np.logical_and.reduce([np.isfinite(v) for v in values[3:]])
            missing_rows+=int(np.count_nonzero(~valid))
            predicted_contacts+=np.bincount(source[valid],weights=count[valid],minlength=n)
            for j, value in enumerate(values[3:]):
                if np.any((value[valid] < 0) | (value[valid] > 1)):
                    raise ValueError('invalid transmitter probability')
                scores[:, j] += np.bincount(source[valid], weights=value[valid]*count[valid], minlength=n)
    np.divide(scores, predicted_contacts[:, None], out=scores, where=predicted_contacts[:, None] > 0)
    predictions = np.array(NT)[scores.argmax(axis=1)]
    predictions[predicted_contacts == 0] = ''
    with annotations.open() as f:
        rows = list(csv.DictReader(f, delimiter='\t'))
    rows.sort(key=lambda row: int(row['root_id']))
    if len(rows) != n or not np.array_equal(ids, np.array([int(r['root_id']) for r in rows], dtype=np.uint64)):
        raise ValueError('annotation root IDs do not exactly match graph')
    decisions = [transmitter_sign(str(predictions[i]), r['known_nt']) for i, r in enumerate(rows)]
    signs = np.array([x[0] for x in decisions], dtype=np.int8)
    groups = {
        'sensory': np.array([i for i,r in enumerate(rows) if r['cell_class']=='olfactory' and
                              'ORN_DM1' in (r['cell_type'], r['hemibrain_type'])], dtype=np.int32),
        'pn': np.array([i for i,r in enumerate(rows) if r['cell_class']=='ALPN' and
                        'DM1_lPN' in (r['cell_type'],r['hemibrain_type'])], dtype=np.int32),
        'all_pn': np.array([i for i,r in enumerate(rows) if r['cell_class']=='ALPN'], dtype=np.int32),
        'kc': np.array([i for i,r in enumerate(rows) if r['cell_class']=='Kenyon_Cell'], dtype=np.int32),
        'mbon': np.array([i for i,r in enumerate(rows) if r['cell_class']=='MBON'], dtype=np.int32),
        'cx': np.array([i for i,r in enumerate(rows) if r['cell_class']=='CX'], dtype=np.int32),
    }
    if any(not len(x) for x in groups.values()) or not np.all(signs[groups['sensory']] == 1):
        raise ValueError('missing cell sets or non-cholinergic DM1 input')
    output.mkdir(parents=True, exist_ok=False)
    np.save(output/'root_ids.npy', ids)
    np.savez(output/'annotations.npz', root_ids=ids, signs=signs, scores=scores, **groups)
    offsets, _, weights = csr_arrays(info)
    counts = {'excitatory': 0, 'inhibitory': 0, 'unmodelled_fast': 0}
    with (output/'connectome.b2csr').open('wb') as out:
        out.write(HEADER.pack(MAGIC, n, n, info['edge_count'], 1))
        copy_region(info['path'], out, HEADER.size, (n+1)*8+info['edge_count']*4)
        for start in range(0, info['edge_count'], 65536):
            end = min(start+65536, info['edge_count'])
            source = np.searchsorted(offsets, np.arange(start, end), side='right')-1
            signed = signs[source]
            out.write((weights[0, start:end]*signed).astype('<f8').tobytes())
            for key, value in [('excitatory',1), ('inhibitory',-1), ('unmodelled_fast',0)]:
                counts[key] += int(np.count_nonzero(signed == value))
    with (output/'neurons.csv').open('w') as f:
        writer = csv.writer(f)
        writer.writerow(['index','root_id','raw_prediction','confidence','annotation_prediction',
                         'known_nt','sign','sign_source','cell_class','cell_type','hemibrain_type'])
        for i,r in enumerate(rows):
            writer.writerow([i,int(ids[i]),predictions[i],scores[i].max(),r['top_nt'],r['known_nt'],
                             int(signs[i]),decisions[i][1],r['cell_class'],r['cell_type'],r['hemibrain_type']])
    manifest = {**original, 'schema':'b2-flywire-ei-v1', 'columns':['signed_contacts'],
                'original_csr_sha256':original['csr_sha256'],
                'csr_sha256':file_hash(output/'connectome.b2csr'),
                'annotations_sha256':ANNOTATION_SHA256,
                'annotations_source':'https://github.com/flyconnectome/flywire_annotations/tree/ebd66db2596fcc39c6950fb54ea3efa00f7fe8a0',
                'transmitter_inference':'contact-weighted mean of six official probability columns per source; argmax; not a per-synapse majority vote',
                'sign_policy':'ACH +1; GABA/GLUT -1; unambiguous curated fast-transmitter evidence overrides; histamine -1; unknown/monoamine-only/conflicting E-I evidence 0',
                'edge_sign_counts':counts,
                'neuron_sign_counts':{str(s):int(np.count_nonzero(signs==s)) for s in (-1,0,1)},
                'curated_overrides':sum(x[1]=='curated_fast_transmitter' for x in decisions),
                'missing_prediction_rows':missing_rows,
                'contacts_without_prediction':int((contacts-predicted_contacts).sum()),
                'preprocess_seconds':time.perf_counter()-started,
                'peak_rss_bytes':int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*(1 if sys.platform=='darwin' else 1024),
                'cell_sets':{k:{'count':len(v),'root_ids':[int(x) for x in ids[v]]} for k,v in groups.items()},
                'scope':'transmitter-informed reduced LIF network, not receptor-resolved or experimentally validated whole-brain biophysics'}
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({k:manifest[k] for k in ('csr_sha256','edge_sign_counts','neuron_sign_counts','curated_overrides')}, indent=2))
    return manifest


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('graph','connections','annotations','output'):p.add_argument('--'+name, type=Path, required=True)
    a=p.parse_args();convert(a.graph,a.connections,a.annotations,a.output)
