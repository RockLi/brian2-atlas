"""Transfer allocations between validated activations, never execution state."""
import ctypes
import time
from pathlib import Path
import numpy as np


def writable(plan):
    return {i for d in plan.dispatches for i,t in zip(d.bindings,d.types,strict=True)
            if not t.startswith('const ')}


def compatible_indices(old_arrays,new_arrays):
    return [i for i,(a,b) in enumerate(zip(old_arrays,new_arrays))
            if (a.shape,a.dtype.str,a.strides)==(b.shape,b.dtype.str,b.strides)]


def refresh_indices(old_arrays,new_arrays,old_writable,new_writable):
    """Return fresh/changed slots and all old/new writable slots still present.

    An old writable/new constant slot needs a refresh even when its two host
    initial snapshots match: its GPU contents may have changed during execution.
    """
    compatible=set(compatible_indices(old_arrays,new_arrays))
    return [i for i,b in enumerate(new_arrays) if i not in compatible or i in old_writable or i in new_writable
            or not np.array_equal(old_arrays[i].view(np.uint8),b.view(np.uint8))]


def adopt_buffers(current,previous,*,backend,max_buffer_bytes=512*1024**2):
    """Move sole buffer ownership; return whether the bounded allocation matched.

    Both executors must already have independently validated/compiled their plans.
    New kernels, clocks, bindings, pending events and host inputs remain current's.
    CUDA captures are always discarded because they retain old ticks/pointers.
    """
    if current is previous or type(current) is not type(previous):
        raise ValueError('buffer transfer requires distinct executors of the same backend')
    if type(max_buffer_bytes) is not int or max_buffer_bytes<=0:
        raise ValueError('max_buffer_bytes must be a positive integer')
    if backend not in {'metal','cuda'}:raise ValueError('unknown GPU backend')
    if not current.plan.dispatches or not previous.plan.dispatches:return False
    if current.dag_execution=='direct':return False
    if backend=='metal':
        if not current.handles or not previous.handles:raise RuntimeError('Metal executor is closed')
        # Handles belong to a particular bridge layout, including indirect caches.
        # Equal source/options hashes are encoded in the bridge library name.
        if Path(current.bridge._name).name != Path(previous.bridge._name).name:return False
        if current._resident_dag_bytes:raise ValueError('destination already owns resident buffers')
        if not previous._resident_dag_bytes:return False
    else:
        if current.closed or previous.closed:raise RuntimeError('CUDA executor is closed')
        if current._resident_dag is not None:raise ValueError('destination already owns resident buffers')
        if previous._resident_dag is None or current.device.id!=previous.device.id:return False
    from .metal_dag import _prepare_dag_storage
    initial=getattr(current,'_dag_initial_storage',None)
    if initial is None:initial=_prepare_dag_storage(current,max_buffer_bytes)
    arrays=initial[0]
    if sum(a.nbytes for a in arrays)>max_buffer_bytes:return False
    old_arrays=previous._dag_initial_storage[0]
    compatible=compatible_indices(old_arrays,arrays)
    if not compatible:return False
    uploads=refresh_indices(old_arrays,arrays,writable(previous.plan),writable(current.plan))
    if backend=='metal':
        function=current.bridge.b2_metal_move_dag
        function.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_uint32,ctypes.POINTER(ctypes.c_uint8)];function.restype=ctypes.c_int
        mask=np.zeros(len(arrays),np.uint8);mask[compatible]=1
        if not function(current.handles[0],previous.handles[0],len(arrays),mask.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))):return False
        current._resident_dag_bytes=sum(arrays[i].nbytes for i in compatible);previous._resident_dag_bytes=0
    else:
        with previous.device:
            previous.stream.synchronize()
            resident=previous._resident_dag
            resident.graph=None
            if resident.chunks is not None:resident.chunks.close();resident.chunks=None
            indices=set(compatible)
            resident.gpu=[resident.gpu[i] if i in indices else None for i in range(len(arrays))]
            resident.layout=tuple((a.shape,a.dtype.str,a.strides) for a in arrays)
            resident.data_bytes=sum(a.nbytes for a in arrays)
            resident.stream=current.stream;resident.writable=tuple(sorted(writable(current.plan)));resident.replays=0
            current._resident_dag=resident;previous._resident_dag=None
    current._dag_initial_storage=initial
    current._activation_upload_indices=uploads
    current._activation_buffers_adopted=True
    return True


def execute_device(device,model,directory,runner):
    """A Device retains at most one successful executor, with explicit cleanup."""
    from .gpu_autotune import DEFAULT_MAX_BUFFER_BYTES, run_with_buffer_budget
    engine=device.build_options['engine'];reuse=device.build_options.get('gpu_buffer_reuse',False)
    compile_reuse=device.build_options.get('gpu_compile_reuse',False)
    max_buffer_bytes=device.build_options.get('gpu_max_buffer_bytes',DEFAULT_MAX_BUFFER_BYTES)
    if engine=='metal':
        from .metal import MetalExecutor as Executor,write_metal_results as write_results
        options=dict(dag_execution=device.build_options.get('metal_dag_execution','auto' if reuse else 'direct'))
    else:
        from .cuda import CudaExecutor as Executor,write_cuda_results as write_results
        options=dict(dag_execution=device.build_options.get('cuda_dag_execution','auto'))
    previous=device._gpu_executor;device._gpu_executor=None;current=None
    device.last_gpu_tuning=None
    try:
        adopted=False;cache_entry=None
        if device.build_options.get('gpu_autotune',False):
            from .gpu_autotune import tune
            from .gpu_validation import ValidatedActivation
            adoptions={};activation=None
            def factories(root):
                prepared={};baseline=None
                def plan_for(name,prefix,sparse):
                    nonlocal activation
                    if activation is None:activation=ValidatedActivation(model,runner=runner)
                    prepared[name]=activation.plan(engine,numeric_mode='float32',runner=runner,
                        event_delivery=device.build_options.get('event_delivery','scan'),
                        synapse_prefix=prefix,synapse_fusion=False,synapse_sparse=sparse)
                    return prepared[name]
                def make_executor(name,prefix,sparse):
                    nonlocal previous,baseline
                    donor=baseline if baseline is not None else previous if compile_reuse else None
                    candidate=Executor(model,root/name,numeric_mode='float32',runner=runner,
                        event_delivery=device.build_options.get('event_delivery','scan'),
                        synapse_prefix=prefix,synapse_fusion=False,synapse_sparse=sparse,
                        plan=prepared[name],compile_reuse=True,reuse_from=donor,_validated_input=activation,**options)
                    adoptions[id(candidate)]=False
                    try:
                        if reuse and previous is not None and type(previous) is type(candidate):
                            adoptions[id(candidate)]=adopt_buffers(
                                candidate,previous,backend=engine,
                                max_buffer_bytes=max_buffer_bytes)
                        if previous is not None:previous.close();previous=None
                        if name=='baseline':baseline=candidate
                        return candidate
                    except BaseException:
                        candidate.close();raise
                return make_executor,plan_for
            if device.build_options.get('gpu_autotune_cache',False):
                from .gpu_tuning_cache import input_key,resolve
                started=time.perf_counter()
                key=input_key(model,runner,dict(engine=engine,**options,
                    event_delivery=device.build_options.get('event_delivery','scan'),
                    buffer_reuse=reuse,compile_reuse=compile_reuse,
                    max_buffer_bytes=max_buffer_bytes))
                current,result,tuning,record=resolve(factories,directory/'gpu-autotune',
                    device._gpu_tuning_cache,key,started=started,
                    max_buffer_bytes=max_buffer_bytes)
                cache_entry=(key,record)
            else:
                make_executor,plan_for=factories(directory/engine)
                current,result,tuning=tune(
                    make_executor,directory/'gpu-autotune',plan_for=plan_for,
                    max_buffer_bytes=max_buffer_bytes)
            # run() consumes the transfer marker. Keep the constructor outcome
            # independently so activation metadata describes the actual transfer.
            adopted=adoptions[id(current)]
            result.setdefault(engine+'_runtime',{})['autotune']=tuning
            device.last_gpu_tuning=tuning
        else:
            current=Executor(model,directory/engine,numeric_mode='float32',runner=runner,
                         event_delivery=device.build_options.get('event_delivery','scan'),
                         synapse_prefix=device.build_options.get('gpu_synapse_prefix',False),
                         synapse_fusion=device.build_options.get('gpu_synapse_fusion',False),
                         synapse_sparse=device.build_options.get('gpu_synapse_sparse',False),
                         compile_reuse=compile_reuse,reuse_from=previous if compile_reuse else None,**options)
            if reuse and previous is not None and type(previous) is type(current):
                adopted=adopt_buffers(current,previous,backend=engine,
                                      max_buffer_bytes=max_buffer_bytes)
            if previous is not None:previous.close();previous=None
            result=run_with_buffer_budget(current,max_buffer_bytes)
        result.setdefault(engine+'_runtime',{})['activation_buffer_reuse']=dict(requested=reuse,adopted=adopted)
        result[engine+'_runtime']['compilation']=dict(current.compilation_report)
        device.last_execution_plan=current.plan
        write_results(model,result,directory/'rust')
        if not reuse and compile_reuse:
            if engine=='cuda':
                with current.device:current._release_resident_dag()
            else:current._release_resident_dag()
        if reuse or compile_reuse:device._gpu_executor=current;current=None
        return cache_entry
    finally:
        if previous is not None:previous.close()
        if current is not None:current.close()
