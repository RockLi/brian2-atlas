"""Host result liveness for the canonical GPU DAG.

Writable inputs still reset on every replay. Only known internal scratch is
omitted from readback; unknown buffer kinds conservatively remain observable.
Population fault/refractory words and synaptic fault/event counters must return.
Pending Device events are reconstructed from the recorded event history, not
from the execution-local pathway cursor or delay ring.
"""
import re
import numpy as np


def readback_bindings(plan):
    writable = {i for d in plan.dispatches
                for i, dtype in zip(d.bindings, d.types, strict=True)
                if not dtype.startswith('const ')}
    scratch = re.compile(
        r'(?:population/\d+/(?:10|linked_values)'
        r'|synapse/\d+/(?:active_ranks|active_counts)'
        r'|synapse/\d+/pathway/\d+/(?:history|pending_cursor))\Z')
    return tuple(sorted(i for i in writable if not scratch.fullmatch(plan.buffers[i])))


def _spike_prefix_bindings(plan, arrays, readback):
    """Known DAG spike tick buffers and their uint32 per-neuron counts.

    Only the decoder-visible prefixes are observable. Everything else keeps the
    conservative full-buffer policy, including all fault and event-history data.
    This descriptor never depends on stale host counts from a previous replay.
    """
    names={name:i for i,name in enumerate(plan.buffers)}
    selected=set(readback);result={}
    for i in readback:
        match=re.fullmatch(r'population/(\d+)/4',plan.buffers[i])
        if not match:continue
        j=names.get('population/'+match[1]+'/5')
        if j not in selected:continue
        ticks,counts=arrays[i],arrays[j]
        if (ticks.dtype!=np.dtype(np.int64) or counts.dtype!=np.dtype(np.uint32)
                or ticks.ndim!=1 or counts.ndim!=1 or not counts.size
                or not ticks.flags.c_contiguous or not counts.flags.c_contiguous
                or not ticks.size or ticks.size%counts.size):continue
        result[i]=j
    return result


def spike_prefix_bindings(plan, arrays, readback):
    return _spike_prefix_bindings(plan,arrays,readback)


def cuda_readback(cp, stream, gpu, arrays, readback, prefixes=None):
    """Read completed device results into existing host arrays, without packing.

    GPU completion precedes this call. Counts are copied first; cudaMemcpy2D
    then transfers the maximum populated prefix width directly into strided
    rows of the original host allocation. No extra device buffer is allocated.
    """
    prefixes=prefixes or {};copied=set();transferred=0
    for j in sorted(set(prefixes.values())):
        gpu[j].get(out=arrays[j],stream=stream,blocking=True)
        transferred+=arrays[j].nbytes;copied.add(j)
    for i in readback:
        if i in copied:continue
        if i in prefixes:
            counts=arrays[prefixes[i]];rows=counts.size;capacity=arrays[i].size//rows
            width=int(counts.max(initial=0))
            if width>capacity:raise RuntimeError('GPU spike capacity invariant violated during readback')
            if width<capacity:
                if width:
                    pitch=capacity*8
                    cp.cuda.runtime.memcpy2D(arrays[i].ctypes.data,pitch,gpu[i].data.ptr,pitch,
                                             width*8,rows,cp.cuda.runtime.memcpyDeviceToHost)
                transferred+=width*rows*8
                continue
        gpu[i].get(out=arrays[i],stream=stream,blocking=True)
        transferred+=arrays[i].nbytes
    return int(transferred)


def spike_upload_omissions(arrays, prefixes):
    """Append-only tick outputs need no initial payload when counts reset to zero.

    The validated population kernel writes each tick before incrementing its
    count; no GPU stage consumes these output slots. History and pending queues
    use separate buffers. Require prefix readback so unused allocation contents
    are never interpreted as records. Nonzero initial counts retain full upload.
    """
    return frozenset(i for i,j in prefixes.items() if not np.any(arrays[j]))


def spike_host_copy_omissions(plan, arrays, writable, compute):
    """Fresh native DAG recording slots need allocation, but no initial copy.

    Counts are copied/reset separately. Every returned tick is written by the
    GPU and then read back before decoding. This remains safe with full upload
    and full readback too: the zero-count unused tail is never observable.
    CPU mirrors and unknown layouts keep the existing copying policy.
    """
    if compute not in {'metal','cuda'}:return frozenset()
    # Host storage liveness is independent of the selected GPU transfer policy.
    prefixes=_spike_prefix_bindings(plan,arrays,writable)
    return frozenset(i for i,j in prefixes.items() if not np.any(arrays[j]))


def fresh_dag_arrays(plan, initial, writable, compute, *, spike_cache=None):
    """Reset writable inputs; optionally reuse private append-only recording slots.

    Only zero-count native spike outputs can reuse host allocation. Their valid
    prefixes are rewritten/read back each run and the decoder publishes copies.
    No state, count, history, parameter or public result array enters this cache.
    A caller owns the cache for one executor and must release it on close.
    """
    omitted=spike_host_copy_omissions(plan,initial,writable,compute)
    if spike_cache is not None:
        for i in list(spike_cache):
            if i not in omitted:del spike_cache[i]
    arrays=[];reused=0
    for i,a in enumerate(initial):
        if i in omitted:
            cached=None if spike_cache is None else spike_cache.get(i)
            if (cached is not None and cached.shape==a.shape and cached.dtype==a.dtype
                    and cached.flags.c_contiguous and cached.flags.writeable
                    and not np.shares_memory(cached,a)):
                fresh=cached;reused+=a.nbytes
            else:fresh=np.empty_like(a)
            if spike_cache is not None:spike_cache[i]=fresh
        else:fresh=a.copy() if i in writable else a
        arrays.append(fresh)
    writable_bytes=sum(a.nbytes for i,a in enumerate(initial) if i in writable)
    saved=sum(initial[i].nbytes for i in omitted)
    return arrays,dict(allocated_bytes=writable_bytes-reused,copied_bytes=writable_bytes-saved,
                       omitted_spike_copy_bytes=saved,reused_spike_bytes=reused,
                       retained_spike_bytes=0 if spike_cache is None else sum(a.nbytes for a in spike_cache.values()))


def host_spike_cache(executor, compute, *, enabled=None):
    """Reuse CUDA records by default; Metal reuse remains an explicit experiment.

    ``enabled`` is an internal ablation control, not a numerical/plan policy.
    CPU mirrors always retain fresh independent storage.
    """
    if enabled is None:enabled=compute=='cuda'
    if compute not in {'metal','cuda'} or not enabled:return None
    if not hasattr(executor,'_dag_host_spike_cache'):executor._dag_host_spike_cache={}
    return executor._dag_host_spike_cache


def release_host_spike_cache(executor):
    cache=getattr(executor,'_dag_host_spike_cache',None)
    if cache is not None:cache.clear()
