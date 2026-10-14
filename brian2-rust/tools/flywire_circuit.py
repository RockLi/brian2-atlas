"""Extract an auditable DM1 olfactory induced subgraph for the browser.

All edges between the selected neurons are retained, including zero-fast-action
edges. Selection uses contact counts, with original root-ID order as tie-break.
Coordinates in the UI are schematic; this export contains no anatomical positions.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))
from brian2_rust.binary_topology import inspect_csr,csr_arrays,file_hash


def extract(graph,original,output,count=240):
    manifest=json.loads((graph/'manifest.json').read_text())
    if manifest['schema']!='b2-flywire-ei-v1':raise ValueError('Expected transmitter-informed FlyWire data')
    signed_info=inspect_csr(graph/'connectome.b2csr');raw_info=inspect_csr(original/'connectome.b2csr')
    if file_hash(signed_info['path'])!=manifest['csr_sha256']:raise ValueError('Signed graph hash mismatch')
    if file_hash(raw_info['path'])!=manifest['original_csr_sha256']:raise ValueError('Original graph hash mismatch')
    offsets,targets,signed=csr_arrays(signed_info)
    raw_offsets,raw_targets,contacts=csr_arrays(raw_info)
    if not np.array_equal(offsets,raw_offsets) or not np.array_equal(targets,raw_targets):raise ValueError('Graph topology mismatch')
    annotations=np.load(graph/'annotations.npz');root_ids=annotations['root_ids'];signs=annotations['signs']
    if not np.array_equal(root_ids,np.load(original/'root_ids.npy')):raise ValueError('Root-ID mismatch')
    with (graph/'neurons.csv').open() as file:rows=list(csv.DictReader(file))
    if len(rows)!=len(root_ids) or any(int(r['index'])!=i or int(r['root_id'])!=int(root_ids[i]) or int(r['sign'])!=signs[i] for i,r in enumerate(rows)):raise ValueError('Neuron annotations mismatch')
    def score(sources):
        result=np.zeros(len(root_ids))
        for source in sources:
            begin,end=offsets[source:source+2]
            np.add.at(result,targets[begin:end],contacts[0,begin:end])
        return result
    def strongest(sources,classes,count):
        scores=score(sources)
        candidates=[i for i,r in enumerate(rows) if r['cell_class'] in classes and scores[i]>0]
        selected=sorted(candidates,key=lambda i:(-scores[i],int(root_ids[i])))[:count]
        if len(selected)!=count:raise ValueError(f'Not enough connected cells in {classes}')
        return sorted(selected)
    sensory=sorted(annotations['sensory'].tolist());pn=sorted(annotations['pn'].tolist())
    if len(sensory)!=68 or len(pn)!=2:raise ValueError('Expected the pinned DM1 cell sets')
    local=strongest(sensory+pn,{'ALLN'},32)
    kc=strongest(pn,{'Kenyon_Cell'},96)
    lh=strongest(pn,{'LHLN','LHCENT'},32)
    mbon=strongest(kc,{'MBON'},10)
    groups=[('sensory','DM1 ORN',sensory),('local','Local neurons',local),('projection','DM1 PN',pn),('kenyon','Kenyon cells',kc),('lateral_horn','Lateral horn',lh),('output','MBON',mbon)]
    if count not in (240,1024,4096):raise ValueError('Supported sizes: 240, 1024, 4096')
    classes={'ALLN':local,'Kenyon_Cell':kc,'LHLN':lh,'LHCENT':lh,'MBON':mbon}
    chosen={i for _,_,items in groups for i in items}
    scores=score(sorted(chosen))
    # Grow one cell at a time so every larger selection contains the smaller one.
    candidates={i for i,r in enumerate(rows) if r['cell_class'] in classes}-chosen
    while len(chosen)<count:
        cell=min(candidates,key=lambda i:(-scores[i],int(root_ids[i])))
        if scores[cell]<=0:raise ValueError('No connected olfactory cells remain')
        chosen.add(cell);candidates.remove(cell);classes[rows[cell]['cell_class']].append(cell)
        begin,end=offsets[cell:cell+2]
        np.add.at(scores,targets[begin:end],contacts[0,begin:end])
    for _,_,items in groups:items.sort()
    selected=[i for _,_,items in groups for i in items];mapping={old:new for new,old in enumerate(selected)}
    if len(mapping)!=count:raise ValueError('Unexpected cell count')
    nodes=[]
    for key,label,items in groups:
        for old in items:
            row=rows[old]
            nodes.append(dict(root_id=str(root_ids[old]),original_index=int(old),group=key,cell_class=row['cell_class'],cell_type=row['cell_type'] or row['hemibrain_type'],sign=int(signs[old]),transmitter=row['raw_prediction'],known_nt=row['known_nt'],sign_source=row['sign_source']))
    edges=[]
    for old_source in selected:
        for edge in range(int(offsets[old_source]),int(offsets[old_source+1])):
            old_target=int(targets[edge])
            if old_target not in mapping:continue
            count=int(contacts[0,edge]);weight=int(signed[0,edge])
            if count<=0 or weight!=count*int(signs[old_source]):raise ValueError('Invalid contact count or sign')
            edges.append(dict(source=mapping[old_source],target=mapping[old_target],contacts=count,signed_contacts=weight))
    result=dict(schema='b2-flywire-circuit-v1',name='DM1 olfactory circuit',release='FlyWire v783',license='CC BY 4.0',
        source='https://zenodo.org/records/10676866',paper='https://www.nature.com/articles/s41586-024-07558-y',
        annotation_source=manifest['annotations_source'],annotations_sha256=manifest['annotations_sha256'],
        full_neurons=len(root_ids),full_weighted_edges=int(signed_info['edge_count']),source_csr_sha256=manifest['csr_sha256'],
        original_csr_sha256=manifest['original_csr_sha256'],local_annotations_sha256=file_hash(graph/'annotations.npz'),
        selection='All 68 annotated DM1 ORNs and both DM1 lPNs; strongest-contact 32 ALLNs from ORNs+PNs, 96 KCs and 32 LH cells from PNs, then 10 MBONs from selected KCs; root-ID tie-break. All induced edges retained.',
        scope='240-neuron induced subgraph; omitted neurons and their inputs are absent. Synthetic stimulation and reduced conductance LIF, not a full-brain or fitted odor simulation.',
        groups=[dict(id=key,label=label,count=len(items)) for key,label,items in groups],nodes=nodes,edges=edges)
    if count!=240:
        result['selection']+=' Expanded one cell at a time by summed incoming contacts from selected cells, restricted to ALLN, Kenyon_Cell, LHLN, LHCENT and MBON; root-ID tie-break. Nested selections; all induced edges retained.'
        result['scope']=result['scope'].replace('240-neuron',f'{count}-neuron')
    identity=json.dumps(dict(nodes=nodes,edges=edges),sort_keys=True,separators=(',',':')).encode()
    result['circuit_sha256']=hashlib.sha256(identity).hexdigest()
    result['weighted_edges']=len(edges);result['biological_contacts']=sum(e['contacts'] for e in edges)
    output.write_text(json.dumps(result,separators=(',',':'))+'\n')
    print(json.dumps({k:result[k] for k in ['name','weighted_edges','biological_contacts','circuit_sha256']},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--graph',type=Path,required=True);parser.add_argument('--original',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--count',type=int,default=240)
    args=parser.parse_args();extract(args.graph,args.original,args.output,args.count)
