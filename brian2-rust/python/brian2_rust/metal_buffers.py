"""Bounded Metal DAG storage reuse; every execution resets the same snapshot."""
import ctypes as c
import numpy as np

from .gpu_schedule import clock_arrays

MAX_INDIRECT_DISPATCHES = 1_000_000


def validate_mode(mode):
    if type(mode) is not str or mode not in {'auto', 'direct', 'resident', 'workgroup', 'indirect'}:
        raise ValueError('Metal dag_execution must be auto, direct, resident, workgroup or indirect')
    return mode


def validate_synchronization(policy):
    if type(policy) is not str or policy not in {'tracked', 'explicit'}:
        raise ValueError('Metal dag_synchronization must be tracked or explicit')
    return policy


def execute_dag(executor, arrays, *, max_buffer_bytes):
    plan=executor.plan
    total=sum(a.nbytes for a in arrays)
    requested=executor._requested_dag_execution
    mode='resident' if requested=='auto' and total<=max_buffer_bytes else 'direct' if requested=='auto' else requested
    if mode=='indirect':
        if executor._requested_dag_synchronization!='explicit':
            executor._release_resident_dag()
            raise ValueError('Indirect Metal execution requires explicit synchronization')
        launches=sum(plan.logical.clocks[d.clock].steps for d in plan.dispatches)
        if len(plan.dispatches)>64 or launches>MAX_INDIRECT_DISPATCHES:
            executor._release_resident_dag()
            raise ValueError(
                f'Indirect Metal execution exceeds 64 stages or '
                f'{MAX_INDIRECT_DISPATCHES} dispatches'
            )
    if mode in {'resident','indirect'} and total>max_buffer_bytes:
        executor._release_resident_dag()
        raise MemoryError('Metal resident DAG exceeds configured total memory limit')
    if executor._metal_dag_metadata is None:
        u32,u64,pointer=c.c_uint32,c.c_uint64,c.c_void_p
        bindings=np.zeros((len(plan.dispatches),31),np.uint32)
        writable=np.zeros(len(arrays),np.uint8)
        for i,dispatch in enumerate(plan.dispatches):
            bindings[i,:len(dispatch.bindings)]=dispatch.bindings
            for binding,dtype in zip(dispatch.bindings,dispatch.types,strict=True):
                if not dtype.startswith('const '):writable[binding]=1
        starts,ends,dts=clock_arrays(plan.logical.clocks)
        count=len(starts)
        metadata=(
            (pointer*len(executor.handles))(*executor.handles), len(executor.handles),
            (u64*len(arrays))(*(a.nbytes for a in arrays)), len(arrays),
            bindings, (u32*len(plan.dispatches))(*(len(s.bindings) for s in plan.dispatches)),
            (u32*len(plan.dispatches))(*(s.lanes for s in plan.dispatches)),
            (u32*len(plan.dispatches))(*(s.clock for s in plan.dispatches)),
            (c.c_int64*count)(*starts),(c.c_int64*count)(*ends),(c.c_double*count)(*dts),count,
            writable)
        executor._metal_dag_metadata=metadata
    handles,stages,sizes,buffers,bindings,counts,lanes,stage_clocks,starts,ends,dts,clocks,writable=executor._metal_dag_metadata
    if len(arrays)!=buffers or any(a.nbytes!=size for a,size in zip(arrays,sizes,strict=True)):
        executor._release_resident_dag()
        raise ValueError('Metal DAG storage shape changed after planning')
    from .gpu_readback import readback_bindings,spike_prefix_bindings,spike_upload_omissions
    readback=np.zeros(len(arrays),np.uint8)
    readback[list(readback_bindings(plan))]=1
    prefixes=np.zeros(len(arrays),np.uint32)
    selected=spike_prefix_bindings(plan,arrays,np.flatnonzero(readback))
    for i,j in selected.items():prefixes[i]=j+1
    omitted=np.zeros(len(arrays),np.uint8)
    for i in spike_upload_omissions(arrays,selected):omitted[i]=1
    function=executor.bridge.b2_metal_run_dag
    pointer,u32,u64=c.c_void_p,c.c_uint32,c.c_uint64
    function.argtypes=[c.POINTER(pointer),u32,c.POINTER(pointer),c.POINTER(u64),u32,
        c.POINTER(u32),c.POINTER(u32),c.POINTER(u32),c.POINTER(u32),
        c.POINTER(c.c_int64),c.POINTER(c.c_int64),c.POINTER(c.c_double),u32,
        c.POINTER(c.c_uint8),c.POINTER(c.c_uint8),c.POINTER(c.c_uint8),c.POINTER(u32),c.POINTER(c.c_uint8),u32,u32,u64,c.POINTER(u64),c.POINTER(c.c_double),pointer,c.c_size_t]
    function.restype=c.c_int
    timing=(c.c_double*4)();stats=(u64*12)();error=c.create_string_buffer(8192)
    pointers=(pointer*len(arrays))(*(a.ctypes.data for a in arrays))
    upload=writable.copy()
    upload[getattr(executor,'_activation_upload_indices',[])]=1
    try:
        code=function(handles,stages,pointers,sizes,buffers,bindings.ctypes.data_as(c.POINTER(u32)),
            counts,lanes,stage_clocks,starts,ends,dts,clocks,writable.ctypes.data_as(c.POINTER(c.c_uint8)),upload.ctypes.data_as(c.POINTER(c.c_uint8)),readback.ctypes.data_as(c.POINTER(c.c_uint8)),prefixes.ctypes.data_as(c.POINTER(u32)),omitted.ctypes.data_as(c.POINTER(c.c_uint8)),
            2 if mode=='indirect' else int(mode=='resident'),int(executor._requested_dag_synchronization=='explicit'),max_buffer_bytes,stats,timing,error,len(error))
        if code:raise RuntimeError(error.value.decode())
    except BaseException:
        executor._release_resident_dag()
        raise
    executor._resident_dag_bytes=int(stats[3]+stats[9])
    adopted=bool(getattr(executor,'_activation_buffers_adopted',False)) and bool(stats[4])
    executor._activation_upload_indices=[];executor._activation_buffers_adopted=False
    executor._metal_dag_report=dict(dag_execution=mode,requested=requested,
        omitted_spike_upload_bytes=sum(a.nbytes for a,skip in zip(arrays,omitted,strict=True) if skip),
        readback_buffer_count=int(readback.sum()),allocated_bytes=int(stats[0]),uploaded_bytes=int(stats[1]),readback_bytes=int(stats[2]),
        resident_bytes=int(stats[3]+stats[9]),buffers_reused=bool(stats[4]),
        buffers_adopted=adopted,
        reused_buffer_count=int(stats[7]),reused_buffer_bytes=int(stats[8]),
        indirect_bytes=int(stats[9]),indirect_commands_reused=bool(stats[10]),indirect_commands_encoded=int(stats[11]),
        synchronization=executor._requested_dag_synchronization,dispatch_type='serial',hazard_tracking='tracked',
        dispatches=int(stats[5]),explicit_barriers=int(stats[6]),
        total_buffer_bytes=total,writable_buffer_bytes=sum(a.nbytes for a,w in zip(arrays,writable,strict=True) if w))
    return timing
