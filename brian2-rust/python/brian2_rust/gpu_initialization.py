"""Canonical Rust host initialization for native GPU simulation plans.

The wire model stays frozen. This private runtime view contains compact NumPy
arrays and retains original layer identities; it must never be exported as IR.
"""
import copy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

import numpy as np

from .plan import PlanValidationError
from .protocol import canonical_bytes


@dataclass(frozen=True)
class Initialization:
    projection: int
    strategy: str
    edge_count: int
    bytes: int
    sha256: str


class _PreparedModel(dict):
    """Backend-only view, never accepted as an alternative wire schema."""


def prepare_model(model, *, runner=None, max_bytes=512*1024**2):
    if isinstance(model, _PreparedModel):
        return model
    procedural = [q for q, inst in enumerate(model['instance']['synapses'])
                  if inst.get('topology', {'kind':'explicit'})['kind']!='explicit']
    if not procedural:
        return model
    executable = (Path(runner) if runner is not None else
                  Path(os.environ.get('B2_RUNNER',
                       str(Path(__file__).resolve().parents[2]/'target/release/b2-runner'))))
    started=time.perf_counter()
    result=_PreparedModel(copy.deepcopy(model))
    records=[]
    with tempfile.TemporaryDirectory(prefix='b2-gpu-initialization-') as temporary:
        path=Path(temporary);(path/'model.json').write_bytes(canonical_bytes(model))
        run=subprocess.run([str(executable),'--gpu-initialization',str(path/'model.json'),
                            str(path/'output'),str(max_bytes)],capture_output=True,text=True)
        if run.returncode:
            raise PlanValidationError(run.stderr or run.stdout)
        manifest=json.loads((path/'output/manifest.json').read_text())
        if (manifest['schema']!='b2-gpu-initialization-v0' or
                [p['projection'] for p in manifest['projections']]!=procedural):
            raise PlanValidationError('GPU initialization manifest does not match projections')
        for item in manifest['projections']:
            q=item['projection'];inst=result['instance']['synapses'][q]
            edge_count=inst['topology']['edge_count']
            if item['file']!=f'projection-{q}.bin':
                raise PlanValidationError('GPU initialization file identity mismatch')
            data=(path/'output'/item['file']).read_bytes()
            if len(data)!=item['bytes'] or hashlib.sha256(data).hexdigest()!=item['sha256']:
                raise PlanValidationError('GPU initialization payload hash mismatch')
            cursor=0
            def array(info, dtype, length):
                nonlocal cursor
                if info['dtype']!=dtype or info['offset']!=cursor or info['length']!=length:
                    raise PlanValidationError('GPU initialization array layout mismatch')
                value=np.frombuffer(data,dtype=dtype,count=length,offset=cursor)
                cursor+=value.nbytes
                return value
            inst['source']=array(item['source'],'<u4',edge_count);inst['target']=array(item['target'],'<u4',edge_count)
            if item['edge_count']!=edge_count or set(item['parameters'])!=set(inst['topology']['initializers']):
                raise PlanValidationError('GPU initialization edge count mismatch')
            for name,info in sorted(item['parameters'].items()):
                inst['parameters'][name]=array(info,'<f8',edge_count)
            for pathway,info in zip(inst['pathways'],item['delays'],strict=True):
                count=edge_count if pathway.get('delay_initializer') is not None else len(pathway['delay_ticks'])
                pathway['delay_ticks']=array(info,'<u4',count)
                pathway['delay_initializer']=None
            if cursor!=len(data):
                raise PlanValidationError('GPU initialization payload has trailing data')
            records.append(Initialization(q,'rust-host-f64-v0',item['edge_count'],item['bytes'],item['sha256']))
            inst['topology']={'kind':'explicit'}
    result.initializations=tuple(records)
    result.initialization_seconds=time.perf_counter()-started
    return result


def initialization_records(model):
    return getattr(model,'initializations',())


def initialization_seconds(model):
    return getattr(model,'initialization_seconds',0.0)
