"""Explicit bounded single-workgroup replay of the canonical GPU stage schedule."""
import ctypes as c
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import time
from types import SimpleNamespace
import numpy as np

from .protocol import canonical_bytes
from .plan import PlanValidationError
from .gpu_schedule import dispatch_ticks

THREADS=128
MAX_BYTES=16*1024**2
MAX_LAUNCHES=8192
MAX_LANE_VISITS=16_000_000
ENTRY='b2_workgroup'


def sha(data):return hashlib.sha256(data).hexdigest()

def layout_of(arrays):return tuple((a.shape,a.dtype.str,a.strides) for a in arrays)


def prepare(owner,arrays,backend):
    plan=owner.plan
    if not plan.dispatches:raise PlanValidationError('Workgroup execution requires an explicit DAG')
    if len(plan.dispatches)>64:raise PlanValidationError('Workgroup execution supports at most 64 stages')
    launches=sum(plan.logical.clocks[d.clock].steps for d in plan.dispatches if d.lanes)
    visits=sum(plan.logical.clocks[d.clock].steps*d.lanes for d in plan.dispatches)
    if launches>MAX_LAUNCHES or visits>MAX_LANE_VISITS:
        raise PlanValidationError('Workgroup execution exceeds bounded stage/lane work budget')
    offset=[];size=0
    for a in arrays:
        if not a.flags.c_contiguous:raise ValueError('Workgroup buffers must be contiguous')
        size=(size+15)//16*16;offset.append(size);size+=a.nbytes
    offsets=np.asarray(offset,np.uint64)
    schedule=[(stage,tick) for stage,tick in dispatch_ticks(plan.logical.clocks,tuple(d.clock for d in plan.dispatches)) if plan.dispatches[stage].lanes]
    if len(schedule)!=launches:raise RuntimeError('Workgroup schedule count mismatch')
    ticks=np.asarray([tick for _,tick in schedule] or [0],np.int64)
    stages=np.asarray([stage for stage,_ in schedule] or [0],np.uint32)
    total=max(size,1)+offsets.nbytes+ticks.nbytes+stages.nbytes+4
    if total>MAX_BYTES:raise MemoryError('Workgroup packed buffers exceed 16 MiB budget')
    from .gpu_functions import source_block
    source=emit(plan,backend,launches,native=source_block(owner.model,backend))
    manifest=dict(schema='b2-single-workgroup-v0',backend=backend,base_plan_sha256=plan.sha256,
        source_sha256=sha(source.encode()),threads=THREADS,workgroups=1,logical_launches=launches,lane_visits=visits,
        offsets=offset,sizes=[a.nbytes for a in arrays],packed_bytes=max(size,1),total_buffer_bytes=total,
        schedule_sha256=sha(ticks.tobytes()+stages.tobytes()),barriers=launches)
    for a in (offsets,ticks,stages):a.flags.writeable=False
    return SimpleNamespace(source=source,manifest=manifest,sha256=sha(canonical_bytes(manifest)),offsets=offsets,ticks=ticks,stages=stages,layout=layout_of(arrays))


def emit(plan,backend,launches,*,native='',cooperative=False):
    if backend not in {'metal','cuda'}:raise ValueError('Unknown workgroup backend')
    if cooperative and backend!='cuda':raise ValueError('Cooperative grid requires CUDA')
    if backend=='metal':parts=['#include <metal_stdlib>\nusing namespace metal;\n']
    else:
        from .cuda_codegen import CUDA_HEADER
        parts=[CUDA_HEADER]
        if cooperative:parts.append('#include <cooperative_groups.h>\n')
    calls=[]
    for q,(kernel,dispatch) in enumerate(zip(plan.kernels,plan.dispatches,strict=True)):
        source=kernel.source
        # Only generated stage code may be rewritten. Native declarations are
        # reinserted verbatim before their callers inside each private stage.
        # This also protects user comments/attributes from builtin detection.
        if native:
            if source.count(native)!=1:
                raise PlanValidationError('Workgroup native source block does not match its validated stage')
            source=source.replace(native,'',1)
        if backend=='metal':
            source=source.replace('#include <metal_stdlib>','').replace('using namespace metal;','')
            source=source.replace('kernel void ','inline void ').replace('constant long &tick','long tick')
            source=re.sub(r'\[\[(?:buffer\(\d+\)|thread_position_in_grid)\]\]','',source)
            if '[[' in source or 'threadgroup_barrier(' in source:raise PlanValidationError('Unsupported stage builtin in workgroup source')
        else:
            source=source.removeprefix(CUDA_HEADER)
            marker='extern "C" __global__ void '+kernel.entry+'('
            chunk='extern "C" __global__ void '+kernel.entry+'_chunk('
            if marker not in source or chunk not in source:raise PlanValidationError('Workgroup requires generated CUDA DAG entries')
            source=source[:source.index(chunk)]
            source=source.replace(marker,'__device__ inline void '+kernel.entry+'(')
            source,count=re.subn(r'\)\s*\{\nuint ([A-Za-z_]\w*) = blockIdx.x \* blockDim.x \+ threadIdx.x;',lambda m:', uint '+m[1]+') {',source)
            if count!=1 or '__global__' in source:raise PlanValidationError('Unexpected CUDA stage entry')
        parts.extend([f'namespace b2_stage_{q} {{\n',native,source,'\n}\n'])
        space='device ' if backend=='metal' else ''
        args=', '.join(f'({space}{dtype}*)(arena+offsets[{binding}])' for binding,dtype in zip(dispatch.bindings,dispatch.types,strict=True))
        if cooperative:
            args=', '.join(f'({dtype}*)(addresses[{binding}])' for binding,dtype in zip(dispatch.bindings,dispatch.types,strict=True))
        calls.append(f'case {q}u: for(uint i=rank;i<{dispatch.lanes}u;i+=width) b2_stage_{q}::{kernel.entry}({args},tick,i); break;')
    if backend=='metal':
        parts.append(f'''kernel void {ENTRY}(device uchar *arena [[buffer(0)]], device const ulong *offsets [[buffer(1)]],
    device const long *ticks [[buffer(2)]], device const uint *stages [[buffer(3)]], device atomic_uint *fault [[buffer(4)]],
    uint rank [[thread_index_in_threadgroup]], uint3 shape [[threads_per_threadgroup]]) {{
    const uint width=shape.x;
''')
        barrier='threadgroup_barrier(mem_flags::mem_device);'
        fault='if(rank==0) atomic_store_explicit(fault,1u,memory_order_relaxed);'
    elif cooperative:
        parts.append('''extern "C" __global__ void b2_cooperative(const ulong *addresses,const long *ticks,const uint *stages,uint *fault) {
    auto grid=cooperative_groups::this_grid();
    const uint rank=blockIdx.x*blockDim.x+threadIdx.x,width=blockDim.x*gridDim.x;
''')
        barrier='grid.sync();';fault='if(rank==0) atomicExch(fault,1u);'
    else:
        parts.append(f'''extern "C" __global__ void {ENTRY}(uchar *arena,const ulong *offsets,const long *ticks,const uint *stages,uint *fault) {{
    const uint rank=threadIdx.x,width=blockDim.x;
''')
        barrier='__syncthreads();';fault='if(rank==0) atomicExch(fault,1u);'
    parts.append(f'''    for(uint step=0;step<{launches}u;++step) {{
        const long tick=ticks[step];
        switch(stages[step]) {{ {' '.join(calls)} default: {fault} break; }}
        {barrier}
    }}
}}
''')
    return ''.join(parts)


class Workgroup:
    def __init__(self,owner,arrays,backend,max_buffer_bytes):
        self.owner,self.backend=owner,backend;self.handle=None;self.module=None;self.function=None;self.replays=0
        self.program=prepare(owner,arrays,backend)
        if self.program.manifest['total_buffer_bytes']>max_buffer_bytes:raise MemoryError('Workgroup buffers exceed configured memory limit')
        p=self.program;directory=owner.directory
        source=directory/(ENTRY+('.metal' if backend=='metal' else '.cu'));source.write_text(p.source)
        (directory/'workgroup-plan.json').write_text(json.dumps(dict(**p.manifest,sha256=p.sha256),indent=2)+'\n')
        started=time.perf_counter()
        if backend=='metal':
            error=c.create_string_buffer(8192)
            self.handle=owner.bridge.b2_metal_create(p.source.encode(),ENTRY.encode(),error,len(error))
            if not self.handle:raise RuntimeError(error.value.decode())
        else:
            from .cuda_compiler import compilation_environment,binary_identity
            env,_=compilation_environment();nvcc=shutil.which('nvcc')
            identity=binary_identity(p.source,owner.architecture,owner.nvcc_version,owner.plan.compiler_options)
            binary=directory/(ENTRY+'-'+identity+'.cubin');temporary=binary.with_suffix('.building.cubin')
            result=subprocess.run([nvcc,'--cubin',f'--gpu-architecture={owner.architecture}',*owner.plan.compiler_options,str(source),'-o',str(temporary)],env=env,capture_output=True,text=True)
            if result.returncode:
                temporary.unlink(missing_ok=True);raise RuntimeError('Workgroup CUDA compile failed:\n'+result.stderr)
            temporary.replace(binary)
            with owner.device:
                self.module=owner.cp.RawModule(path=str(binary));self.function=self.module.get_function(ENTRY);self.function.attributes
        self.compile_seconds=time.perf_counter()-started

    def close(self):
        if self.handle is not None:self.owner.bridge.b2_metal_destroy(self.handle);self.handle=None
        self.function,self.module=None,None

    def run(self,arrays,max_buffer_bytes):
        p=self.program
        if layout_of(arrays)!=p.layout:raise ValueError('Workgroup buffer layout changed after planning')
        if p.manifest['total_buffer_bytes']>max_buffer_bytes:raise MemoryError('Workgroup buffers exceed configured memory limit')
        start=time.perf_counter();arena=np.zeros(p.manifest['packed_bytes'],np.uint8)
        for a,offset in zip(arrays,p.offsets,strict=True):arena[int(offset):int(offset)+a.nbytes]=a.view(np.uint8).reshape(-1)
        data=[arena,p.offsets.copy(),p.ticks.copy(),p.stages.copy(),np.zeros(1,np.uint32)]
        packed_seconds=time.perf_counter()-start
        if self.backend=='cuda':
            from .cuda import CudaExecutor
            runner=SimpleNamespace(cp=self.owner.cp,device=self.owner.device,stream=self.owner.stream,kernels=[self.function])
            timing=CudaExecutor._execute(runner,data,[(0,THREADS,range(5),False)],start_tick=0,steps=1,max_buffer_bytes=max_buffer_bytes)
        else:
            ptr=c.c_void_p;u64=c.c_uint64;u32=c.c_uint32
            fn=self.owner.bridge.b2_metal_run_workgroup
            fn.argtypes=[ptr,c.POINTER(ptr),c.POINTER(u64),u32,u32,u32,c.POINTER(c.c_double),ptr,c.c_size_t];fn.restype=c.c_int
            timing=(c.c_double*4)();error=c.create_string_buffer(8192)
            code=fn(self.handle,(ptr*5)(*(a.ctypes.data for a in data)),(u64*5)(*(a.nbytes for a in data)),5,THREADS,17,timing,error,len(error))
            if code:raise RuntimeError(error.value.decode())
        if data[-1][0]:raise RuntimeError('Workgroup stage schedule fault; result not published')
        start=time.perf_counter()
        writable={b for d in self.owner.plan.dispatches for b,t in zip(d.bindings,d.types,strict=True) if not t.startswith('const ')}
        for i in writable:
            a=arrays[i];offset=int(p.offsets[i]);a.view(np.uint8).reshape(-1)[:]=arena[offset:offset+a.nbytes]
        unpacked_seconds=time.perf_counter()-start;self.replays+=1
        report=dict(requested='workgroup',selected='workgroup',dag_execution='workgroup',workgroup_plan_sha256=p.sha256,
            threads=THREADS,workgroups=1,dispatches=1,logical_dispatches=p.manifest['logical_launches'],barriers=p.manifest['barriers'],
            compile_seconds=self.compile_seconds,compiled_program_reused=self.replays>1,replay_index=self.replays,
            total_buffer_bytes=p.manifest['total_buffer_bytes'],allocated_bytes=p.manifest['total_buffer_bytes'],
            packed_seconds=packed_seconds,unpacked_seconds=unpacked_seconds,buffer_reused=False,buffers_reused=False)
        if self.backend=='cuda':self.owner._dag_execution_report=report
        else:self.owner._metal_dag_report=report
        return tuple(timing)


def close(owner):
    workgroup=getattr(owner,'_workgroup',None)
    if workgroup is not None:workgroup.close();owner._workgroup=None


def execute(owner,arrays,*,max_buffer_bytes,backend):
    try:
        if getattr(owner,'_workgroup',None) is None:owner._workgroup=Workgroup(owner,arrays,backend,max_buffer_bytes)
        return owner._workgroup.run(arrays,max_buffer_bytes)
    except BaseException:
        close(owner);raise
