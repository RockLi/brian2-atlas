"""Native NVIDIA CUDA execution, explicit f32 arithmetic and Brian result transport.

The current lowering shares the validated scalar schedule and storage layout
with Metal. CUDA owns its plan identity, compiler settings, device allocations,
stream and kernels. Unsupported AtlasIR features fail during planning.
"""
from dataclasses import asdict, dataclass, replace
import ctypes
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
from types import SimpleNamespace

import numpy as np

from .cuda_codegen import OPTIONS, cuda_source, CUDA_CHUNK_ABI, CHUNK_ADVANCE_ENTRY
from .cuda_compiler import compilation_environment, binary_identity
from .metal import (MetalKernel, _derive_metal_plan, population_arrays,
                    population_result, _write_gpu_results)
from .plan import LogicalPlan, PlanValidationError, validate_model
from .protocol import canonical_bytes

CUDA_PROFILE = "b2-cuda-f32-v0"


@dataclass(frozen=True)
class CudaPlan:
    schema: str
    numeric_profile: str
    definition_sha256: str
    instance_sha256: str
    run_sha256: str
    logical: LogicalPlan
    kernels: tuple[MetalKernel, ...]
    dispatches: tuple
    buffers: tuple
    strategy: str
    event_delivery: str
    elided_nodes: tuple[str, ...]
    compiler_options: tuple[str, ...] = OPTIONS
    rng_profile: str | None = None
    initializations: tuple = ()
    chunk_abi: str | None = None

    def to_dict(self):
        return asdict(self)

    def to_json(self):
        return json.dumps(self.to_dict(),sort_keys=True,indent=2)+"\n"

    @property
    def sha256(self):
        return hashlib.sha256(canonical_bytes(self.to_dict())).hexdigest()


def _derive_cuda_plan(model, *, numeric_mode, event_delivery="scan", synapse_prefix=False, synapse_fusion=False, synapse_sparse=False):
    if numeric_mode != "float32":
        raise PlanValidationError("CUDA requires explicit numeric_mode='float32'")
    from .gpu_functions import inject_sources
    source = _derive_metal_plan(model,numeric_mode=numeric_mode,event_delivery=event_delivery,
                                _native_backend='cuda',synapse_prefix=synapse_prefix,synapse_fusion=synapse_fusion,synapse_sparse=synapse_sparse)
    return CudaPlan("b2-cuda-plan-v0",CUDA_PROFILE,source.definition_sha256,
                    source.instance_sha256,source.run_sha256,source.logical,
                    tuple(replace(k,source=inject_sources(cuda_source(k,chunked=bool(source.dispatches),advance=bool(source.dispatches) and i==0), model, "cuda"))
                          for i,k in enumerate(source.kernels)),
                    source.dispatches,source.buffers,source.strategy,source.event_delivery,
                    source.elided_nodes,rng_profile=source.rng_profile,initializations=source.initializations,
                    chunk_abi=CUDA_CHUNK_ABI if source.dispatches else None)


def build_cuda_plan(model, *, numeric_mode, runner=None, event_delivery="scan", synapse_prefix=False, synapse_fusion=False, synapse_sparse=False):
    from .gpu_initialization import prepare_model
    return _derive_cuda_plan(prepare_model(validate_model(model,runner=runner),runner=runner),numeric_mode=numeric_mode,
                             event_delivery=event_delivery,synapse_prefix=synapse_prefix,synapse_fusion=synapse_fusion,synapse_sparse=synapse_sparse)


class CudaExecutor:
    """Compile once and replay a validated initial snapshot on one CUDA device."""
    def __init__(self,model,directory,*,numeric_mode,runner=None,plan=None,event_delivery="scan",dag_execution="auto",compile_reuse=False,reuse_from=None,synapse_prefix=False, synapse_fusion=False, synapse_sparse=False, _validated_input=None):
        self.closed = True
        from .cuda_graphs import validate_mode
        self.dag_execution=validate_mode(dag_execution)
        self._resident_dag=None
        self._dag_execution_report=None
        from .gpu_validation import executor_model
        self.model = executor_model(model,runner=runner,activation=_validated_input)
        self.plan = _derive_cuda_plan(self.model,numeric_mode=numeric_mode,event_delivery=event_delivery,synapse_prefix=synapse_prefix,synapse_fusion=synapse_fusion,synapse_sparse=synapse_sparse)
        if plan is not None and canonical_bytes(plan.to_dict()) != canonical_bytes(self.plan.to_dict()):
            raise PlanValidationError("CUDA plan does not match model, numeric mode or compiler options")
        from .gpu_compilation import request,compiler_context,cuda_dependencies,cached_binary,save_binary
        portable=request(self.model,compile_reuse,reuse_from,type(self))
        self._compiled_binaries={}
        self.compilation_report=dict(requested=compile_reuse,portable=portable,kernels_reused=0,kernels_compiled=0,dependency_files=0)
        # No GPU is needed for build_cuda_plan; execution requires the real runtime.
        try:
            import cupy as cp
        except ImportError as error:
            raise RuntimeError("CUDA execution requires CuPy and the NVIDIA CUDA toolkit") from error
        nvcc = shutil.which("nvcc")
        if nvcc is None:
            raise RuntimeError("CUDA execution requires nvcc on PATH")
        self.cp, self.directory = cp, Path(directory)
        self.directory.mkdir(parents=True,exist_ok=True)
        self.device = cp.cuda.Device()
        properties = cp.cuda.runtime.getDeviceProperties(self.device.id)
        name = properties["name"]
        self.device_name = name.decode() if isinstance(name,bytes) else str(name)
        self.architecture = f"sm_{properties['major']}{properties['minor']}"
        compile_environment, self.compiler_environment = compilation_environment()
        self.nvcc_version = subprocess.check_output([nvcc,"--version"],text=True,env=compile_environment)
        self.modules, self.kernels, self.chunk_kernels = [], [], []
        self.chunk_advance = None
        self.stream = cp.cuda.Stream(non_blocking=True)
        started = time.perf_counter()
        try:
            context=None
            if portable:
                source=self.directory/(self.plan.kernels[0].entry+'.cu');source.write_text(self.plan.kernels[0].source)
                dependencies,count=cuda_dependencies(nvcc,source,self.architecture,self.plan.compiler_options,compile_environment)
                context=compiler_context(nvcc,compile_environment,[self.architecture,self.nvcc_version,list(self.plan.compiler_options),dependencies])
                self.compilation_report.update(context_sha256=context,dependency_files=count)
            for kernel in self.plan.kernels:
                source = self.directory/(kernel.entry+".cu")
                source.write_text(kernel.source)
                digest = binary_identity(kernel.source,self.architecture,self.nvcc_version,self.plan.compiler_options)
                binary = self.directory/(kernel.entry+"-"+digest+".cubin")
                key=(context,kernel.entry,digest)
                data=cached_binary(reuse_from,key) if portable else None
                if data is not None:
                    binary.write_bytes(data);self.compilation_report['kernels_reused']+=1
                elif compile_reuse or not binary.exists():
                    temporary = binary.with_suffix(".building.cubin")
                    compiled = subprocess.run([nvcc,"--cubin",f"--gpu-architecture={self.architecture}",
                                               *self.plan.compiler_options,str(source),"-o",str(temporary)],
                                              text=True,capture_output=True,env=compile_environment)
                    if compiled.returncode:
                        temporary.unlink(missing_ok=True)
                        raise RuntimeError(f"CUDA compile failed ({kernel.entry}):\n{compiled.stderr}")
                    temporary.replace(binary)
                    self.compilation_report['kernels_compiled']+=1
                if portable:save_binary(self,key,binary.read_bytes())
                module = cp.RawModule(path=str(binary))
                function=module.get_function(kernel.entry)
                # Resolve lazy CUDA function loading before any stream capture.
                function.attributes
                self.kernels.append(function)
                self.modules.append(module)
                if self.plan.chunk_abi:
                    variant=module.get_function(kernel.entry+'_chunk');variant.attributes
                    self.chunk_kernels.append(variant)
                    if len(self.modules)==1:
                        self.chunk_advance=module.get_function(CHUNK_ADVANCE_ENTRY)
                        self.chunk_advance.attributes
        except BaseException:
            self.close()
            raise
        self.compile_seconds = time.perf_counter()-started
        (self.directory/"cuda-plan.json").write_text(self.plan.to_json())
        (self.directory/'compilation.json').write_text(json.dumps(self.compilation_report,indent=2)+'\n')
        self.closed = False

    def _release_resident_dag(self):
        self._activation_upload_indices=[];self._activation_buffers_adopted=False
        if self._resident_dag is not None:
            self._resident_dag.close()
            self._resident_dag=None

    def close(self):
        from .gpu_readback import release_host_spike_cache
        release_host_spike_cache(self)
        from .gpu_workgroup import close
        close(self)
        from .cuda_cooperative import close as close_cooperative
        try:
            if hasattr(self,"stream"):
                with self.device:
                    try:
                        close_cooperative(self)
                        self.stream.synchronize()
                    finally:self._release_resident_dag()
        finally:
            self.kernels, self.modules, self.chunk_kernels = [], [], []
            self.chunk_advance = None
            self._compiled_binaries={}
            self.closed = True

    def __enter__(self):
        return self

    def __exit__(self,*exc):
        self.close()

    def _execute(self,arrays,stages,*,start_tick,steps,max_buffer_bytes,schedule=None,readback=None,spike_prefixes=None,omit_upload=()):
        if sum(a.nbytes for a in arrays) > max_buffer_bytes:
            raise MemoryError("CUDA working buffers exceed configured total memory limit")
        cp = self.cp
        with self.device, self.stream:
            started = time.perf_counter()
            gpu = [cp.empty(a.shape,dtype=a.dtype) if i in omit_upload else cp.asarray(a)
                   for i,a in enumerate(arrays)]
            self.stream.synchronize()
            input_seconds = time.perf_counter()-started
            begin, end = cp.cuda.Event(), cp.cuda.Event()
            started = time.perf_counter()
            begin.record(self.stream)
            if schedule is None:
                schedule = ((s,tick) for tick in range(start_tick,start_tick+steps) for s in range(len(stages)))
            for stage,tick in schedule:
                ordinal,lanes,bindings,has_tick = stages[stage]
                if not lanes:
                    continue
                args = tuple(gpu[i] for i in bindings)
                if has_tick:
                    args += (np.int64(tick),)
                self.kernels[ordinal](((lanes+127)//128,),(128,),args,stream=self.stream)
            end.record(self.stream)
            end.synchronize()
            command_seconds = time.perf_counter()-started
            gpu_seconds = cp.cuda.get_elapsed_time(begin,end)/1000
            started = time.perf_counter()
            # Copy into existing host storage: result views and mutable synaptic
            # slices must retain their references to these arrays.
            from .gpu_readback import cuda_readback
            copied=cuda_readback(cp,self.stream,gpu,arrays,range(len(arrays)) if readback is None else readback,spike_prefixes)
            if spike_prefixes is not None:self._dag_execution_report['readback_bytes']=copied
            readback_seconds = time.perf_counter()-started
        return input_seconds,command_seconds,gpu_seconds,readback_seconds

    def _execute_dag(self,arrays,*,max_buffer_bytes):
        if self._requested_dag_execution=='cooperative':
            from .cuda_cooperative import execute
            with self.device:self._release_resident_dag()
            return execute(self,arrays,max_buffer_bytes=max_buffer_bytes)
        from .cuda_cooperative import close as close_cooperative
        close_cooperative(self)
        if self._requested_dag_execution=='workgroup':
            from .gpu_workgroup import execute
            with self.device:self._release_resident_dag()
            return execute(self,arrays,max_buffer_bytes=max_buffer_bytes,backend='cuda')
        from .gpu_schedule import dispatch_ticks
        from .gpu_readback import readback_bindings,spike_prefix_bindings,spike_upload_omissions
        readback=readback_bindings(self.plan)
        prefixes=spike_prefix_bindings(self.plan,arrays,readback)
        omit_upload=spike_upload_omissions(arrays,prefixes)
        omitted_bytes=sum(arrays[i].nbytes for i in omit_upload)
        from .cuda_graphs import ResidentDag,select_mode,MAX_GRAPH_LAUNCHES
        stages=[(i,d.lanes,d.bindings,True) for i,d in enumerate(self.plan.dispatches)]
        clocks=self.plan.logical.clocks
        launches=sum(clocks[d.clock].steps for d in self.plan.dispatches if d.lanes)
        mode,reason=select_mode(self._requested_dag_execution,launches,
                                replays=0 if self._resident_dag is None else self._resident_dag.replays)
        total=sum(a.nbytes for a in arrays)
        chunk_schedule=None;chunk_reason=None;chunk_preparation_seconds=0.
        if self._requested_dag_execution in {'auto','chunked'}:
            from .cuda_chunks import build_schedule
            started=time.perf_counter()
            if not hasattr(self,'_chunk_schedule'):
                try:self._chunk_schedule=build_schedule(clocks,self.plan.dispatches,launches);self._chunk_reason=None
                except ValueError as error:self._chunk_schedule=None;self._chunk_reason=str(error)
            chunk_preparation_seconds=time.perf_counter()-started
            chunk_schedule=self._chunk_schedule;chunk_reason=self._chunk_reason
            if self._requested_dag_execution=='chunked' and chunk_schedule is None:
                raise ValueError(chunk_reason)
            if chunk_schedule is not None:
                if total+chunk_schedule.buffer_bytes>max_buffer_bytes:
                    if self._requested_dag_execution=='chunked':
                        with self.device:self._release_resident_dag()
                        raise MemoryError('CUDA chunk schedule exceeds configured total memory limit')
                    chunk_schedule=None;chunk_reason='chunk-buffer-budget'
                elif self._requested_dag_execution=='auto' and not chunk_schedule.worthwhile:
                    chunk_schedule=None;chunk_reason='insufficient-pattern-reuse'
            if chunk_schedule is not None:
                mode,reason='chunked','reused-stage-patterns' if self._requested_dag_execution=='auto' else 'explicit'
        if total>max_buffer_bytes:
            with self.device:self._release_resident_dag()
            raise MemoryError('CUDA working buffers exceed configured total memory limit')
        report=dict(requested=self._requested_dag_execution,selected=mode,reason=reason,
                    kernel_launches=launches,capture_limit=MAX_GRAPH_LAUNCHES,
                    graph_build_seconds=0.0,graph_reused=False,resident_buffer_bytes=0,
                    buffer_reused=False,upload_bytes=total-omitted_bytes,readback_bytes=sum(arrays[i].nbytes for i in readback),
                    omitted_spike_upload_bytes=omitted_bytes,
                    readback_buffer_count=len(readback),
                    allocated_bytes=total,reused_buffer_count=0,reused_buffer_bytes=0,
                    chunk_schedule_preparation_seconds=chunk_preparation_seconds,chunk_fallback_reason=chunk_reason)
        self._dag_execution_report=report
        schedule=dispatch_ticks(clocks,tuple(d.clock for d in self.plan.dispatches))
        if mode=='direct':
            with self.device:self._release_resident_dag()
            return self._execute(arrays,stages,start_tick=0,steps=0,max_buffer_bytes=max_buffer_bytes,schedule=schedule,readback=readback,spike_prefixes=prefixes,omit_upload=omit_upload)
        writable={binding for d in self.plan.dispatches for binding,dtype in zip(d.bindings,d.types,strict=True)
                  if not dtype.startswith('const ')}
        with self.device,self.stream:
            started=time.perf_counter()
            try:
                if self._resident_dag is None:
                    self._resident_dag=ResidentDag(arrays,writable,self.cp,self.stream,omit_upload=omit_upload)
                else:
                    uploads=writable|set(getattr(self,'_activation_upload_indices',[]))
                    report.update(self._resident_dag.reset(arrays,uploads,omit_upload=omit_upload))
                    report.update(buffer_reused=bool(report['reused_buffer_count']))
                    report['upload_bytes']+=12 if self._resident_dag.chunks is not None else 0
                self.stream.synchronize()
                input_seconds=time.perf_counter()-started
                timing,details=self._resident_dag.execute(arrays,self.kernels,stages,schedule,mode,
                    chunk_schedule=chunk_schedule,chunk_kernels=self.chunk_kernels,advance=self.chunk_advance,readback=readback,spike_prefixes=prefixes)
                input_seconds+=details.get('chunk_input_seconds',0.)
                report['upload_bytes']+=details.get('chunk_new_upload_bytes',0)
                report.update(details)
                report['buffers_adopted']=bool(getattr(self,'_activation_buffers_adopted',False)) and report['buffer_reused']
                self._activation_upload_indices=[];self._activation_buffers_adopted=False
                return (input_seconds,*timing)
            except BaseException:
                # No half-captured graph or poisoned buffers survive a failure.
                # Drain queued work before its buffer ownership is released.
                try:self.stream.synchronize()
                except Exception:pass
                self._release_resident_dag()
                raise

    def _cpu_control(self,max_buffer_bytes,workers):
        from .gpu_functions import require_cpu_mirror
        require_cpu_mirror(self.model)
        from .metal import MetalExecutor
        from .metal_dag import run_dag
        if not hasattr(self,"control"):
            from .gpu_initialization import prepare_model
            model = prepare_model(self.model)
            plan = _derive_metal_plan(model,numeric_mode="float32",
                                      event_delivery=self.plan.event_delivery if self.plan.dispatches else "scan")
            self.control = SimpleNamespace(model=model,plan=plan,directory=self.directory,
                                            compile_seconds=0,device_name="CPU f32 control")
        if self.control.plan.dispatches:
            return run_dag(self.control,max_buffer_bytes=max_buffer_bytes,compute="cpu-f32",workers=workers)
        cpu = MetalExecutor._cpu_mirror(self.control)
        started = time.perf_counter()
        populations, timings = [], []
        for kernel in self.control.plan.kernels:
            arrays,last = population_arrays(self.model,kernel.population,kernel,max_buffer_bytes)
            pointers = (ctypes.c_void_p*len(arrays))(*(a.ctypes.data for a in arrays))
            begin = time.perf_counter()
            getattr(cpu,"cpu_"+kernel.entry)(pointers,workers)
            timings.append(dict(input_seconds=0,command_seconds=time.perf_counter()-begin,gpu_seconds=0,readback_seconds=0))
            populations.append(population_result(self.model,kernel.population,kernel,arrays,last))
        return dict(populations=populations,synapses=[],numeric_profile="b2-cpu-f32-mirror-v0",device="CPU f32 control",
                    timings=timings,run_seconds=time.perf_counter()-started,compile_seconds=0,
                    plan_sha256=self.control.plan.sha256,rng_profile=self.control.plan.rng_profile)

    def run(self,*,max_buffer_bytes=512*1024**2,compute="cuda",workers=1,dag_execution=None):
        if self.closed:
            raise RuntimeError("CUDA executor is closed")
        if type(max_buffer_bytes) is not int or max_buffer_bytes <= 0:
            raise ValueError("max_buffer_bytes must be a positive integer")
        if compute not in {"cuda","cpu-f32"}:
            raise ValueError("compute must be cuda or cpu-f32")
        if type(workers) is not int or not 1 <= workers <= 256:
            raise ValueError("workers must be within 1..256")
        from .cuda_graphs import validate_mode
        self._requested_dag_execution=validate_mode(self.dag_execution if dag_execution is None else dag_execution)
        if self._resident_dag is not None and self._resident_dag.bytes>max_buffer_bytes:
            with self.device:self._release_resident_dag()
        if compute=="cuda" and self._requested_dag_execution in {'workgroup','cooperative'} and not self.plan.dispatches:
            raise PlanValidationError('Persistent execution requires an explicit DAG')
        if compute == "cpu-f32":
            return self._cpu_control(max_buffer_bytes,workers)
        if self.plan.dispatches:
            from .metal_dag import run_dag
            result = run_dag(self,max_buffer_bytes=max_buffer_bytes,compute="cuda",workers=1)
        else:
            started = time.perf_counter()
            populations, timings = [], []
            for i,kernel in enumerate(self.plan.kernels):
                arrays,last = population_arrays(self.model,kernel.population,kernel,max_buffer_bytes)
                timing = self._execute(arrays,[(i,kernel.neurons,range(len(arrays)),False)],
                                       start_tick=0,steps=1,max_buffer_bytes=max_buffer_bytes)
                populations.append(population_result(self.model,kernel.population,kernel,arrays,last))
                timings.append(dict(zip(("input_seconds","command_seconds","gpu_seconds","readback_seconds"),timing,strict=True)))
            result = dict(populations=populations,synapses=[],numeric_profile=CUDA_PROFILE,device=self.device_name,
                          timings=timings,run_seconds=time.perf_counter()-started,
                          compile_seconds=self.compile_seconds,plan_sha256=self.plan.sha256,rng_profile=self.plan.rng_profile)
        result["cuda_runtime"] = dict(architecture=self.architecture,nvcc=self.nvcc_version,
                                      compilation=dict(self.compilation_report),
                                      cupy=self.cp.__version__,driver=self.cp.cuda.runtime.driverGetVersion(),
                                      compiler_options=list(self.plan.compiler_options),
                                      compiler_environment={**self.compiler_environment,
                                          "ignored_environment_variables":list(self.compiler_environment["ignored_environment_variables"])},
                                      timing_scope="CUDA event interval includes host submission gaps",
                                      dag_execution=self._dag_execution_report if self.plan.dispatches else
                                          dict(requested=self._requested_dag_execution,selected="fused-direct",reason="independent populations already fuse time"))
        return result


def write_cuda_results(model,result,directory):
    return _write_gpu_results(model,result,directory,engine="cuda",numeric_profile=CUDA_PROFILE)
