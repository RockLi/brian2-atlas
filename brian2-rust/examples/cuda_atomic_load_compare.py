"""Matched 2x2 atomic-read and queue-saturation experiment on native CUDA."""
from contextlib import contextmanager
from unittest.mock import patch
from gpu_sparse_saturation_compare import queue_reservations


LEGACY_ATOMIC_LOAD='__device__ inline uint atomic_load_explicit(uint *p,int) { return atomicAdd(p,0u); }'


@contextmanager
def atomic_reads(*,legacy=False):
    """Single-threaded harness selection; production uses the guarded load."""
    from brian2_atlas import cuda_codegen
    header=cuda_codegen.CUDA_HEADER
    assert header.count(cuda_codegen.CUDA_ATOMIC_LOAD)==1
    if legacy:header=header.replace(cuda_codegen.CUDA_ATOMIC_LOAD,LEGACY_ATOMIC_LOAD)
    with patch.object(cuda_codegen,'CUDA_HEADER',header):yield


@contextmanager
def variant(mode):
    assert mode in ('legacy','load','bounded-legacy','bounded-load')
    with atomic_reads(legacy=mode.endswith('legacy')),queue_reservations(saturate=mode.startswith('bounded-')):
        yield


def compare(output):
    from gpu_sparse_saturation_compare import compare as replay
    return replay(output,'cuda',_variants={mode:lambda mode=mode:variant(mode)
        for mode in ('legacy','load','bounded-legacy','bounded-load')},
        _schema='b2-cuda-atomic-load-comparison-v0')


if __name__=='__main__':
    import argparse,json
    from pathlib import Path
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=compare(args.output)
    print(json.dumps(dict(passed=result['passed'],cases={c['name']:c['summary'] for c in result['cases']}),indent=2))
