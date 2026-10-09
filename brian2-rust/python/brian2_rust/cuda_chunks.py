"""Bounded reuse of captured stage patterns within one canonical activation."""
from dataclasses import dataclass
import hashlib
import json
import time

import numpy as np

CHUNK_SIZE = 64
MAX_PATTERNS = 64
MAX_TABLE_LAUNCHES = 1_000_000


@dataclass(frozen=True)
class ChunkSchedule:
    ticks: np.ndarray
    patterns: tuple[tuple[int, ...], ...]
    sequence: tuple[int, ...]
    sha256: str

    @property
    def buffer_bytes(self):return self.ticks.nbytes+12  # uint64 cursor + uint32 fault

    @property
    def capture_launches(self):return sum(len(p)+1 for p in self.patterns)

    @property
    def worthwhile(self):return self.capture_launches*4<=len(self.ticks)


def build_schedule(clocks,stages,launches):
    from .gpu_schedule import dispatch_ticks
    if not 0<launches<=MAX_TABLE_LAUNCHES:
        raise ValueError(f'CUDA chunk schedule requires 1..{MAX_TABLE_LAUNCHES} nonempty launches')
    ticks=np.empty(launches,np.int64);patterns=[];sequence=[];lookup={};pattern=[];position=0
    def finish():
        key=tuple(pattern)
        if key not in lookup:
            if len(patterns)>=MAX_PATTERNS:raise ValueError('Too many distinct CUDA graph chunks')
            lookup[key]=len(patterns);patterns.append(key)
        sequence.append(lookup[key]);pattern.clear()
    for ordinal,tick in dispatch_ticks(clocks,tuple(d.clock for d in stages)):
        if not stages[ordinal].lanes:continue
        if position>=launches:raise RuntimeError('CUDA chunk schedule count overflow')
        ticks[position]=tick;position+=1;pattern.append(ordinal)
        if len(pattern)==CHUNK_SIZE:finish()
    if pattern:finish()
    if position!=launches:raise RuntimeError('CUDA chunk schedule count mismatch')
    digest=hashlib.sha256(json.dumps(dict(schema='b2-cuda-chunk-schedule-v0',patterns=patterns,sequence=sequence),sort_keys=True).encode()+ticks.tobytes()).hexdigest()
    ticks.flags.writeable=False
    return ChunkSchedule(ticks,tuple(patterns),tuple(sequence),digest)


class ChunkGraphs:
    def __init__(self,schedule,cp,stream):
        self.schedule,self.cp,self.stream=schedule,cp,stream
        self.ticks=cp.asarray(schedule.ticks)
        # Keep async H2D source memory alive until the owning stream is drained.
        self._zero_cursor=np.zeros(1,np.uint64)
        self._zero_fault=np.zeros(1,np.uint32)
        self.cursor=cp.asarray(self._zero_cursor)
        self.fault=cp.asarray(self._zero_fault)
        self.graphs=[]

    def reset(self):
        self.cursor.set(self._zero_cursor,stream=self.stream)
        self.fault.set(self._zero_fault,stream=self.stream)

    def close(self):
        self.graphs=[]
        self.ticks,self.cursor,self.fault=None,None,None
        self._zero_cursor,self._zero_fault=None,None

    def capture(self,kernels,advance,gpu,stages):
        started=time.perf_counter();length=np.uint64(len(self.schedule.ticks))
        for pattern in self.schedule.patterns:
            self.stream.begin_capture()
            try:
                for offset,ordinal in enumerate(pattern):
                    _,lanes,bindings,_=stages[ordinal]
                    args=tuple(gpu[i] for i in bindings)+(self.ticks,self.cursor,length,np.uint32(offset),self.fault)
                    kernels[ordinal](((lanes+127)//128,),(128,),args,stream=self.stream)
                advance((1,),(1,),(self.cursor,length,np.uint32(len(pattern)),self.fault),stream=self.stream)
            except BaseException:
                try:self.stream.end_capture()
                except Exception:pass
                raise
            graph=self.stream.end_capture();graph.upload(stream=self.stream);self.graphs.append(graph)
        self.stream.synchronize()
        return time.perf_counter()-started

    def launch(self):
        for ordinal in self.schedule.sequence:self.graphs[ordinal].launch(stream=self.stream)

    def check(self):
        cursor=int(self.cursor.get(stream=self.stream,blocking=True)[0])
        fault=int(self.fault.get(stream=self.stream,blocking=True)[0])
        if fault or cursor!=len(self.schedule.ticks):
            raise RuntimeError('CUDA chunk tick cursor invariant failed; result not published')
