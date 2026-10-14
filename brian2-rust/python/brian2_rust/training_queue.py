"""Read-only snapshots of Brian's native pending event queue.

The existing named capsule and installed C++ header provide typed access. No
pointer offsets, guessed layouts, queue.prepare(), or model execution are used.
"""
from functools import lru_cache
from pathlib import Path
import hashlib
import tempfile
import threading

_LOCK=threading.RLock()
_CODE='''
from cpython.pycapsule cimport PyCapsule_GetPointer
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libc.stdint cimport int32_t
cdef extern from "spikequeue.h":
    cdef cppclass CSpikeQueue:
        double dt
        int offset
        vector[vector[int32_t]] queue
        pair[int, vector[vector[int32_t]]] _full_state()

def snapshot(capsule, long long budget):
    cdef CSpikeQueue* q = <CSpikeQueue*>PyCapsule_GetPointer(capsule, "CSpikeQueue")
    cdef size_t count = 0
    cdef size_t i
    if q.queue.size() > budget // 64:
        raise ValueError("pending queue snapshot exceeds memory budget")
    for i in range(q.queue.size()):
        count += q.queue[i].size()
        if count > budget // 64:
            raise ValueError("pending queue snapshot exceeds memory budget")
    if (q.queue.size() + count) * 64 > budget:
        raise ValueError("pending queue snapshot exceeds memory budget")
    if count == 0:
        return q.dt, 0, []
    return q.dt, q.offset, q._full_state().second
'''


@lru_cache(maxsize=1)
def _reader():
    import brian2.synapses
    from brian2 import prefs
    from brian2.codegen.runtime.cython_rt.extension_manager import cython_extension_manager
    header=Path(brian2.synapses.__file__).with_name('spikequeue.h')
    code=_CODE+'\n# installed header sha256: '+hashlib.sha256(header.read_bytes()).hexdigest()+'\n'
    old=prefs.codegen.runtime.cython.cache_dir
    with tempfile.TemporaryDirectory(prefix='b2-queue-snapshot-') as directory:
        prefs.codegen.runtime.cython.cache_dir=directory
        try:module=cython_extension_manager.create_extension(code,owner_name='read-only training queue snapshot')
        finally:prefs.codegen.runtime.cython.cache_dir=old
    if module is None:raise ValueError('could not compile the read-only Brian queue snapshot bridge')
    return module


def pending_events(queue, new_dt, edges, *, max_bytes):
    """Return delivery bins as queue.prepare would, without mutating the queue."""
    import math
    from brian2.synapses.cythonspikequeue import SpikeQueue
    if not isinstance(queue,SpikeQueue):raise ValueError('a native Brian runtime event queue is required')
    with _LOCK:
        old_dt,offset,bins=_reader().snapshot(queue.get_capsule(),max_bytes)
    if not bins:return []
    if not math.isfinite(old_dt) or old_dt<0 or not 0<=offset<len(bins):
        raise ValueError('invalid pending queue clock or offset')
    if any(type(edge) is not int or not 0<=edge<edges for row in bins for edge in row):
        raise ValueError('pending queue contains an edge outside the snapshot topology')
    bins=bins[offset:]+bins[:offset]
    if old_dt and old_dt!=new_dt:
        ratio=old_dt/new_dt
        if not math.isfinite(ratio) or len(bins)*ratio>=max_bytes//64:
            raise ValueError('resampled pending queue exceeds memory budget')
        size=int(len(bins)*ratio)+1
        if size*64>max_bytes:raise ValueError('resampled pending queue exceeds memory budget')
        converted=[[] for _ in range(size)]
        for k,row in enumerate(bins):converted[int(k*ratio+.5)]=row
        bins=converted
    return bins
