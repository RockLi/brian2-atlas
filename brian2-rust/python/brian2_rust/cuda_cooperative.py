"""Opt-in multi-block replay with occupancy-bounded cooperative grid barriers.

The pointer table refers only to buffers owned by this runtime. No host pointer
is accepted from a serialized plan. All stages retain the canonical clock order.
"""
import hashlib
import json
import shutil
import subprocess
import time
from types import SimpleNamespace
import numpy as np

from .plan import PlanValidationError
from .protocol import canonical_bytes
from .gpu_schedule import dispatch_ticks
from .gpu_workgroup import emit, layout_of, THREADS

ENTRY = 'b2_cooperative'
# Bound persistent execution independently of the single-workgroup experiment.
# These are admission limits, not guarantees about watchdog duration.
MAX_STAGES = 64
MAX_LAUNCHES = 65536
MAX_LANE_VISITS = 2**31


def prepare(owner, arrays):
    plan = owner.plan
    if not plan.dispatches or len(plan.dispatches) > MAX_STAGES:
        raise PlanValidationError('Cooperative execution requires 1..64 DAG stages')
    launches = sum(plan.logical.clocks[d.clock].steps for d in plan.dispatches if d.lanes)
    visits = sum(plan.logical.clocks[d.clock].steps*d.lanes for d in plan.dispatches)
    if launches > MAX_LAUNCHES or visits > MAX_LANE_VISITS:
        raise PlanValidationError('Cooperative execution exceeds bounded stage/lane work budget')
    if any(not a.flags.c_contiguous for a in arrays):
        raise ValueError('Cooperative buffers must be contiguous')
    schedule = [(s,t) for s,t in dispatch_ticks(plan.logical.clocks, tuple(d.clock for d in plan.dispatches))
                if plan.dispatches[s].lanes]
    if len(schedule) != launches:raise RuntimeError('Cooperative schedule count mismatch')
    ticks = np.asarray([t for _,t in schedule] or [0], np.int64)
    stages = np.asarray([s for s,_ in schedule] or [0], np.uint32)
    from .gpu_functions import source_block
    source = emit(plan, 'cuda', launches, native=source_block(owner.model,'cuda'), cooperative=True)
    metadata_bytes = 8*len(arrays)+ticks.nbytes+stages.nbytes+4
    manifest = dict(schema='b2-cooperative-grid-v0', base_plan_sha256=plan.sha256,
        source_sha256=hashlib.sha256(source.encode()).hexdigest(), threads=THREADS,
        logical_launches=launches, lane_visits=visits, metadata_bytes=metadata_bytes,
        total_buffer_bytes=sum(a.nbytes for a in arrays)+metadata_bytes,
        schedule_sha256=hashlib.sha256(ticks.tobytes()+stages.tobytes()).hexdigest(),
        barriers=launches, sizes=[a.nbytes for a in arrays])
    ticks.flags.writeable=False;stages.flags.writeable=False
    return SimpleNamespace(source=source, ticks=ticks, stages=stages, manifest=manifest,
        sha256=hashlib.sha256(canonical_bytes(manifest)).hexdigest(), layout=layout_of(arrays))


def grid_size(properties, active_blocks_per_sm, lanes):
    if not properties.get('cooperativeLaunch'):
        raise PlanValidationError('CUDA device does not support cooperative launch')
    sms = int(properties['multiProcessorCount'])
    if sms < 1 or active_blocks_per_sm < 1:
        raise PlanValidationError('Cooperative kernel has no resident block capacity')
    # One block per SM is the initial explicit policy. All lanes are covered by
    # grid-stride stage calls, even when the workload is larger than the grid.
    return min(max(1,(lanes+THREADS-1)//THREADS), sms, sms*active_blocks_per_sm)


def launch_binding(plan_sha256, binary_sha256, blocks, sms, active_blocks_per_sm):
    """Bind actual geometry to the exact compiled kernel and its residency limit."""
    if type(blocks) is not int or not 1 <= blocks <= sms*active_blocks_per_sm:
        raise PlanValidationError('Cooperative block count exceeds kernel residency or is not a positive integer')
    binding=dict(schema='b2-cooperative-launch-v0',plan_sha256=plan_sha256,
        binary_sha256=binary_sha256,blocks=blocks,threads=THREADS,shared_memory_bytes=0,
        multiprocessors=sms,active_blocks_per_sm=active_blocks_per_sm)
    return dict(binding,sha256=hashlib.sha256(canonical_bytes(binding)).hexdigest())


class CooperativeDag:
    def __init__(self, owner, arrays, max_buffer_bytes):
        self.owner=owner;self.resident=None;self.module=None;self.function=None;self.metadata=[]
        self.program=prepare(owner,arrays);p=self.program
        if p.manifest['total_buffer_bytes']>max_buffer_bytes:
            raise MemoryError('Cooperative buffers exceed configured total memory limit')
        properties=owner.cp.cuda.runtime.getDeviceProperties(owner.device.id)
        # Check hardware support before compiling or allocating the working set.
        grid_size(properties,1,1)
        source=owner.directory/(ENTRY+'.cu');source.write_text(p.source)
        from .cuda_compiler import compilation_environment,binary_identity
        env,_=compilation_environment()
        options=owner.plan.compiler_options
        identity=binary_identity(p.source,owner.architecture,owner.nvcc_version,options)
        binary=owner.directory/(ENTRY+'-'+identity+'.cubin')
        temporary=binary.with_suffix('.building.cubin')
        started=time.perf_counter()
        result=subprocess.run([shutil.which('nvcc'),'--cubin',f'--gpu-architecture={owner.architecture}',
            *options,str(source),'-o',str(temporary)],env=env,capture_output=True,text=True)
        if result.returncode:
            temporary.unlink(missing_ok=True)
            raise RuntimeError('Cooperative CUDA compile failed:\n'+result.stderr)
        temporary.replace(binary)
        # Use CuPy's driver Function: its public ptr lets us query occupancy for
        # this exact cubin before calling its cooperative launch implementation.
        self.module=owner.cp.cuda.function.Module();self.module.load_file(str(binary))
        self.function=self.module.get_function(ENTRY)
        self.active_blocks_per_sm=owner.cp.cuda.driver.occupancyMaxActiveBlocksPerMultiprocessor(self.function.ptr,THREADS,0)
        self.blocks=grid_size(properties,self.active_blocks_per_sm,max(d.lanes for d in owner.plan.dispatches))
        self.sms=int(properties['multiProcessorCount'])
        self.compile_seconds=time.perf_counter()-started
        self.binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest()
        (owner.directory/'cooperative-plan.json').write_text(json.dumps(dict(p.manifest,
            sha256=p.sha256,blocks=self.blocks,multiprocessors=self.sms,
            active_blocks_per_sm=self.active_blocks_per_sm,binary_sha256=self.binary_sha256),indent=2)+'\n')

    def configure_grid(self, blocks):
        """Internal explicit tuning hook; never silently clamp an unsafe grid."""
        if self.function is None:raise RuntimeError('Cooperative runtime is closed')
        binding=launch_binding(self.program.sha256,self.binary_sha256,blocks,self.sms,self.active_blocks_per_sm)
        self.blocks=blocks
        return binding

    def close(self):
        if self.resident is not None:self.resident.close();self.resident=None
        self.metadata=[];self.function=None;self.module=None

    def run(self, arrays, max_buffer_bytes):
        p=self.program;owner=self.owner;cp=owner.cp;stream=owner.stream
        binding=launch_binding(p.sha256,self.binary_sha256,self.blocks,self.sms,self.active_blocks_per_sm)
        if layout_of(arrays)!=p.layout:raise ValueError('Cooperative buffer layout changed')
        if p.manifest['total_buffer_bytes']>max_buffer_bytes:
            raise MemoryError('Cooperative buffers exceed configured total memory limit')
        from .gpu_readback import readback_bindings,spike_prefix_bindings,spike_upload_omissions,cuda_readback
        from .cuda_graphs import ResidentDag
        readback=readback_bindings(owner.plan)
        prefixes=spike_prefix_bindings(owner.plan,arrays,readback)
        omitted=spike_upload_omissions(arrays,prefixes)
        writable={b for d in owner.plan.dispatches for b,t in zip(d.bindings,d.types,strict=True) if not t.startswith('const ')}
        started=time.perf_counter();reused=self.resident is not None
        if not reused:
            self.resident=ResidentDag(arrays,writable,cp,stream,omit_upload=omitted)
            addresses=np.asarray([a.data.ptr for a in self.resident.gpu],np.uint64)
            self.metadata=[cp.asarray(addresses),cp.asarray(p.ticks),cp.asarray(p.stages),cp.zeros(1,np.uint32)]
            upload=sum(a.nbytes for i,a in enumerate(arrays) if i not in omitted)+p.manifest['metadata_bytes']
        else:
            stats=self.resident.reset(arrays,omit_upload=omitted)
            self.metadata[-1].fill(0);upload=stats['upload_bytes']+4
        stream.synchronize();input_seconds=time.perf_counter()-started
        begin,end=cp.cuda.Event(),cp.cuda.Event();started=time.perf_counter()
        begin.record(stream)
        self.function((binding['blocks'],),(THREADS,),tuple(self.metadata),stream=stream,enable_cooperative_groups=True)
        end.record(stream);end.synchronize()
        command_seconds=time.perf_counter()-started
        gpu_seconds=cp.cuda.get_elapsed_time(begin,end)/1000
        started=time.perf_counter()
        if int(self.metadata[-1].get(stream=stream)[0]):
            raise RuntimeError('Cooperative stage schedule fault; result not published')
        copied=cuda_readback(cp,stream,self.resident.gpu,arrays,readback,prefixes)
        readback_seconds=time.perf_counter()-started;self.resident.replays+=1
        owner._dag_execution_report=dict(requested='cooperative',selected='cooperative',reason='explicit',
            cooperative_plan_sha256=p.sha256,binary_sha256=self.binary_sha256,threads=THREADS,
            workgroups=binding['blocks'],multiprocessors=self.sms,active_blocks_per_sm=self.active_blocks_per_sm,
            launch_binding=binding,
            kernel_launches=1,logical_dispatches=p.manifest['logical_launches'],barriers=p.manifest['barriers'],
            compile_seconds=self.compile_seconds,buffer_reused=reused,compiled_program_reused=reused,
            replay_index=self.resident.replays,resident_buffer_bytes=p.manifest['total_buffer_bytes'],
            allocated_bytes=0 if reused else p.manifest['total_buffer_bytes'],upload_bytes=upload,
            readback_bytes=copied+4,omitted_spike_upload_bytes=sum(arrays[i].nbytes for i in omitted))
        return input_seconds,command_seconds,gpu_seconds,readback_seconds


def close(owner):
    runtime=getattr(owner,'_cooperative_dag',None)
    if runtime is not None:
        with owner.device:
            try:owner.stream.synchronize()
            finally:runtime.close();owner._cooperative_dag=None


def execute(owner, arrays, *, max_buffer_bytes):
    try:
        with owner.device,owner.stream:
            if getattr(owner,'_cooperative_dag',None) is None:
                owner._cooperative_dag=CooperativeDag(owner,arrays,max_buffer_bytes)
            return owner._cooperative_dag.run(arrays,max_buffer_bytes)
    except BaseException:
        close(owner);raise
