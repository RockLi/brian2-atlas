"""Resident CUDA DAG buffers and bounded graph capture for immutable plans.

The captured schedule contains the same kernels, bindings and absolute int64
clock ticks as direct dispatch. Each replay resets writable buffers to the
validated host snapshot. Immutable topology/parameters stay on the device.
"""
import time
import numpy as np

MODES=frozenset(('auto','direct','resident','graph','chunked','workgroup','cooperative'))
MAX_GRAPH_LAUNCHES=65536


def validate_mode(mode):
    if not isinstance(mode,str) or mode not in MODES:
        raise ValueError('CUDA DAG execution must be auto, direct, resident, graph, chunked, workgroup or cooperative')
    return mode


def select_mode(requested,launches,*,replays=0):
    validate_mode(requested)
    if requested=='graph' and launches>MAX_GRAPH_LAUNCHES:
        raise ValueError(f'CUDA graph exceeds {MAX_GRAPH_LAUNCHES} captured kernel launches')
    if requested=='auto':
        if 1<launches<=MAX_GRAPH_LAUNCHES:
            if not replays:return 'resident','first-replay'
            return 'graph','bounded-replayed-dag'
        return 'resident','capture-limit' if launches>MAX_GRAPH_LAUNCHES else 'single-or-empty-dag'
    if requested=='graph' and not launches:return 'resident','empty-dag'
    return requested,'explicit'


def launch(kernels,gpu,stages,schedule,stream):
    for stage,tick in schedule:
        ordinal,lanes,bindings,has_tick=stages[stage]
        if not lanes:continue
        args=tuple(gpu[i] for i in bindings)
        if has_tick:args+=(np.int64(tick),)
        kernels[ordinal](((lanes+127)//128,),(128,),args,stream=stream)


class ResidentDag:
    def __init__(self,arrays,writable,cp,stream,*,omit_upload=()):
        self.cp,self.stream=cp,stream
        self.layout=tuple((a.shape,a.dtype.str,a.strides) for a in arrays)
        self.writable=tuple(sorted(writable))
        self.data_bytes=sum(a.nbytes for a in arrays)
        self.chunks=None
        self.graph=None
        self.gpu=[cp.empty(a.shape,dtype=a.dtype) if i in omit_upload else cp.asarray(a)
                  for i,a in enumerate(arrays)]
        self.replays=0

    @property
    def bytes(self):
        return self.data_bytes+(self.chunks.schedule.buffer_bytes if self.chunks is not None else 0)

    def close(self):
        # Destroy captured pointer references before releasing their buffers.
        self.graph=None
        if self.chunks is not None:self.chunks.close();self.chunks=None
        self.gpu=[]

    def reset(self,arrays,upload=None,*,omit_upload=()):
        if tuple((a.shape,a.dtype.str,a.strides) for a in arrays)!=self.layout:
            raise RuntimeError('CUDA resident buffer layout changed within an immutable executor')
        uploads=(set(self.writable)|set(upload or ()))-set(omit_upload)
        stats=dict(allocated_bytes=0,upload_bytes=0,reused_buffer_count=0,reused_buffer_bytes=0)
        for i,a in enumerate(arrays):
            if self.gpu[i] is None:
                self.gpu[i]=self.cp.empty(a.shape,dtype=a.dtype) if i in omit_upload else self.cp.asarray(a)
                stats['allocated_bytes']+=a.nbytes
                if i not in omit_upload:stats['upload_bytes']+=a.nbytes
            else:
                stats['reused_buffer_count']+=1;stats['reused_buffer_bytes']+=a.nbytes
                if i in uploads:self.gpu[i].set(a,stream=self.stream);stats['upload_bytes']+=a.nbytes
        if self.chunks is not None:self.chunks.reset()
        return stats

    def capture(self,kernels,stages,schedule):
        self.stream.begin_capture()
        try:
            launch(kernels,self.gpu,stages,schedule,self.stream)
        except BaseException:
            # Ending even an invalidated capture releases stream capture state.
            try:self.stream.end_capture()
            except Exception:pass
            raise
        self.graph=self.stream.end_capture()
        self.graph.upload(stream=self.stream)
        self.stream.synchronize()

    def execute(self,arrays,kernels,stages,schedule,mode,*,chunk_schedule=None,chunk_kernels=None,advance=None,readback=None,spike_prefixes=None):
        build_seconds=0.0;reused=self.graph is not None;chunk_input_seconds=0.;chunk_new_upload_bytes=0
        if mode=='chunked':
            from .cuda_chunks import ChunkGraphs
            reused=self.chunks is not None
            if self.chunks is None:
                started=time.perf_counter()
                self.chunks=ChunkGraphs(chunk_schedule,self.cp,self.stream)
                self.stream.synchronize();chunk_input_seconds=time.perf_counter()-started
                chunk_new_upload_bytes=chunk_schedule.buffer_bytes
                build_seconds=self.chunks.capture(chunk_kernels,advance,self.gpu,stages)
        if mode=='graph' and self.graph is None:
            started=time.perf_counter();self.capture(kernels,stages,schedule)
            build_seconds=time.perf_counter()-started
        begin,end=self.cp.cuda.Event(),self.cp.cuda.Event()
        started=time.perf_counter();begin.record(self.stream)
        if mode=='chunked':self.chunks.launch()
        elif mode=='graph':self.graph.launch(stream=self.stream)
        else:launch(kernels,self.gpu,stages,schedule,self.stream)
        end.record(self.stream);end.synchronize()
        command_seconds=time.perf_counter()-started
        gpu_seconds=self.cp.cuda.get_elapsed_time(begin,end)/1000
        started=time.perf_counter()
        if mode=='chunked':self.chunks.check()
        readback=self.writable if readback is None else readback
        from .gpu_readback import cuda_readback
        transferred=cuda_readback(self.cp,self.stream,self.gpu,arrays,readback,spike_prefixes)
        readback_seconds=time.perf_counter()-started
        self.replays+=1
        return (command_seconds,gpu_seconds,readback_seconds),dict(
            graph_build_seconds=build_seconds,graph_reused=mode in {'graph','chunked'} and reused,
            chunk_input_seconds=chunk_input_seconds,chunk_new_upload_bytes=chunk_new_upload_bytes,
            chunk_schedule_sha256=chunk_schedule.sha256 if mode=='chunked' else None,
            chunk_graph_count=len(chunk_schedule.patterns) if mode=='chunked' else 0,
            host_graph_launches=len(chunk_schedule.sequence) if mode=='chunked' else int(mode=='graph'),
            cursor_advance_launches=len(chunk_schedule.sequence) if mode=='chunked' else 0,
            resident_buffer_bytes=self.bytes,readback_bytes=transferred+(12 if mode=='chunked' else 0),
            replay_index=self.replays)
