"""Ordered target bitmaps versus rank queues on native delayed STDP replay."""
from contextlib import contextmanager,nullcontext
from unittest.mock import patch
import argparse,json
from pathlib import Path


@contextmanager
def target_bitset():
    """Private single-threaded planner experiment, with the original buffer ABI.

    Only canonical target queues switch representation. Immutable sparse
    consumers, pending queues, scalar execution and stage barriers are unchanged.
    """
    from brian2_rust import metal_dag
    from brian2_rust.metal_synapses import canonical_projection
    producer,consumer=metal_dag.sparse_history_kernel,metal_dag.canonical_kernel
    def enqueue(model,logical,node,ordinal,**kwargs):
        if canonical_projection(model,node.owner_index):kwargs['bitset']=True
        return producer(model,logical,node,ordinal,**kwargs)
    def consume(*args,**kwargs):
        if kwargs.get('sparse_target'):kwargs['bitset']=True
        return consumer(*args,**kwargs)
    with patch.object(metal_dag,'sparse_history_kernel',enqueue),patch.object(metal_dag,'canonical_kernel',consume):yield


def compare(output,backend='cuda'):
    from gpu_sparse_saturation_compare import compare as paired
    return paired(output,backend,_variants=dict(baseline=nullcontext,bitset=target_bitset),
        _schema='b2-target-bitset-comparison-v0',
        _scope='separate retained continuation executors with rank queues / ordered target bitmaps; same buffers and dispatches; 128-tick native warm state independently checked; bootstrap compilation excluded; full fresh-reset to result timings; no external backend ranking')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--backend',choices=('metal','cuda'),default='cuda')
    a=p.parse_args();print(json.dumps(compare(a.output,a.backend),indent=2))
