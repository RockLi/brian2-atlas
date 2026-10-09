"""Persistent delayed-STDP worker: reset and replay all states, queues and traces."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

import numpy as np

BACKENDS=('rust-f64','cpu-f32','cuda','brian2cuda','genn','brian2genn-corrected')
PREFIX_BACKENDS=('cuda-prefix','metal-prefix')
BITSET_BACKENDS=('cuda-bitset','metal-bitset','cuda-prefix-bitset','metal-prefix-bitset')
CORRECTED_GENN_BACKENDS=('genn-barrier',)
GATHER_GENN_BACKENDS=('genn-barrier-gather',)
F32_GENN_BACKENDS=('brian2genn-f32-factor',)
MAX_REPLAY_STEPS=4096
CPP_BACKENDS=('cpp-f64-t1','cpp-f64-t2','cpp-f64-t4')
COMPILERS={'nvcc','rustc','cc','c++','gcc','g++','clang','clang++','make','cmake','ninja','genn-buildmodel.sh'}


@contextmanager
def forbid_compilation():
    original=subprocess.Popen
    def checked(command,*args,**kwargs):
        tokens=shlex.split(command) if isinstance(command,str) else [os.fsdecode(t) for t in command]
        if any(Path(t).name in COMPILERS for t in tokens):
            raise RuntimeError('Compilation attempted inside a precompiled replay: '+repr(tokens))
        return original(command,*args,**kwargs)
    subprocess.Popen=checked
    try:yield
    finally:subprocess.Popen=original


def artifact_hashes(directory):
    return {str(p.relative_to(directory)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(directory.rglob('*')) if p.is_file() and not any(part.startswith('._') for part in p.relative_to(directory).parts) and
        (p.name in {'main','b2-native'} or p.suffix in {'.so','.cubin','.dylib'})}


def native_arrays(raw):
    from gpu_stdp_compare import FIELDS
    p=raw['populations'][0]
    result=dict(v=np.asarray(p['states']['v']).copy(),trace=np.asarray(p['trace']['v']).copy(),
                ticks=np.asarray(p['spike_ticks'],np.int64).copy(),indices=np.asarray(p['indices'],np.int64).copy())
    for key in FIELDS:result[key]=np.asarray(raw['synapses'][0]['states'][key],dtype=np.float64 if key=='lastupdate' else None).copy()
    return result


def configuration(neurons,steps,degree,**workload):
    from gpu_stdp_compare import configuration as config
    return config(neurons,degree,steps,**workload)


class Replay:
    def __init__(self,backend,neurons,steps,degree,output,*,drive=.125,delay_span=4,post_delay=2,topology_kind="ring",topology_seed=0):
        from gpu_stdp_compare import brian_run,genn_run,workload_options
        workload=workload_options(drive,delay_span,post_delay,topology_kind,topology_seed)
        self.requested_backend=backend
        self.genn_f32_factor=backend in F32_GENN_BACKENDS
        if self.genn_f32_factor:backend="brian2genn-corrected"
        self.genn_gather=backend in GATHER_GENN_BACKENDS
        self.genn_barrier=backend in CORRECTED_GENN_BACKENDS+GATHER_GENN_BACKENDS
        if self.genn_barrier:backend='genn'
        self.synapse_sparse='bitset' if backend in BITSET_BACKENDS else False
        if self.synapse_sparse:backend=backend.removesuffix('-bitset')
        self.synapse_prefix=backend in PREFIX_BACKENDS
        if self.synapse_prefix:backend=backend.removesuffix('-prefix')
        self.backend,self.steps,self.output=backend,steps,output
        self.context={};self.count=0;self.executor=None
        started=time.perf_counter()
        if backend in CPP_BACKENDS:
            result,timing=brian_run('cpp-f64',neurons,degree,steps,output,prepared=self.context,cpp_threads=int(backend[-1]),**workload)
        elif backend=='brian2genn-corrected':
            from gpu_brian2genn_stdp_adapter import run
            result,timing=run(neurons,degree,steps,output,prepared=self.context,
                decay_mode="f32-factor" if self.genn_f32_factor else "generated",**workload)
        elif backend=='genn':
            if self.genn_barrier:
                from gpu_genn_barrier_adapter import run
                result,timing=run(neurons,degree,steps,output,prepared=self.context,**workload)
            else:result,timing=genn_run(neurons,degree,steps,output,prepared=self.context,trace_mode='device',**workload)
        else:
            result,timing=brian_run('rust' if backend=='rust-f64' else backend,neurons,degree,steps,output,prepared=self.context,**workload)
        if backend in {'cuda','metal','rust-f64'}:
            self.model=json.loads((output/'project/model.json').read_text())
        if backend in {'cuda','metal'}:
            from brian2_rust.cuda import CudaExecutor
            from brian2_rust.metal import MetalExecutor
            self.executor=(CudaExecutor if backend=='cuda' else MetalExecutor)(self.model,output/'project'/backend,numeric_mode='float32',event_delivery='sparse',synapse_prefix=self.synapse_prefix,synapse_sparse=self.synapse_sparse)
        if self.synapse_prefix:
            if not any(d.role=='edge-synapse-prefix' for d in self.executor.plan.dispatches):
                raise ValueError('Prefix comparison requires an actually selected prefix stage')
        if self.synapse_sparse:
            if not any(d.role=='target-owned-bitset-synapse-pathway' for d in self.executor.plan.dispatches):
                raise ValueError('Bitset comparison requires an actually selected bitmap pathway')
        if self.synapse_prefix or self.synapse_sparse:
            result=native_arrays(self.executor.run())
        gather_evidence=None
        if self.genn_gather:
            from gpu_genn_readback import prepare,replay
            gather_evidence=prepare(self.context)
            result=replay(self.context,steps,verify=True)
            gather_evidence['bootstrap']=dict(self.context['last_gather'])
        self.setup_seconds=time.perf_counter()-started;self.bootstrap=result;self.setup_timing=timing
        self.adapter_evidence=None
        if self.genn_barrier:
            self.adapter_evidence=dict(label=timing['label'],correction=timing['generated_code_correction'],
                sources={name:(output/'project'/name).read_text() for name in
                    ('synapseUpdate.original.cc','postsynaptic-barrier.patch','barrier-build.log','b2_delayed_stdp_CODE/synapseUpdate.cc')})
            if self.genn_gather:self.adapter_evidence['host_gather']=gather_evidence
        if backend in {'cuda','metal'}:
            self.adapter_evidence=dict(execution_plan=self.executor.plan.to_dict(),synapse_prefix=self.synapse_prefix,synapse_sparse=self.synapse_sparse)
        if backend=='brian2genn-corrected':
            self.adapter_evidence={name:(output/name).read_text() for name in ('transformations.json',
                'magicnetwork_model.cpp.original','engine.cpp.original','magicnetwork_model.cpp.patch','engine.cpp.patch',
                'project/magicnetwork_model.cpp','project/engine.cpp')}
        if backend in CPP_BACKENDS:
            self.adapter_evidence={str(p.relative_to(output)):p.read_text() for p in sorted((output/'project/code_objects').glob('*.cpp')) if not p.name.startswith('._')}
            timing['observed_openmp_threads']=int((Path(self.context['device'].results_dir)/'openmp_threads.txt').read_text())
            if timing['observed_openmp_threads']!=int(backend[-1]):raise RuntimeError('Unexpected OpenMP team size')
        if backend=='rust-f64':
            self.adapter_evidence={'execution_plan':json.loads((output/'project/native/execution-plan.json').read_text()),
                'main_rs':(output/'project/native/main.rs').read_text()}
        self.artifacts=artifact_hashes(output)
        if not self.artifacts:raise RuntimeError('No compiled artifact found after preparation')

    def _brian_arrays(self):
        import brian2 as b
        from gpu_stdp_compare import DT,FIELDS
        c=self.context;p=c['population'];m=c['monitor'];dev=c['device'];trace=c['trace']
        variables=[p.variables['v'],m.variables['t'],m.variables['i'],trace.variables['v']]
        for syn,edges in c['groups']:variables.extend(syn.variables[k] for k in FIELDS)
        for var in variables:dev.array_cache[var]=None
        ticks=np.rint(np.asarray(m.t[:]/b.second)/DT).astype(np.int64);indices=np.asarray(m.i[:],np.int64)
        order=np.lexsort((indices,ticks));result=dict(v=np.asarray(p.v[:]).copy(),ticks=ticks[order],indices=indices[order],trace=np.asarray(trace.v).T.copy())
        for key in FIELDS:
            syn=c['groups'][0][0];values=getattr(syn,key)[:]
            result[key]=np.asarray(values/b.second if key=='lastupdate' else values,dtype=np.float64 if key=='lastupdate' else None).copy()
        return result

    def run(self,*,rust_threads=1):
        if type(rust_threads) is not int or not 1<=rust_threads<=256:
            raise ValueError('rust_threads must be an integer within 1..256')
        if self.backend!='rust-f64' and rust_threads!=1:
            raise ValueError('Only Rust f64 accepts rust_threads')
        self.count+=1;start=time.perf_counter();details={};result_directory=None
        with forbid_compilation():
            if self.backend in {'cuda','metal'}:
                raw=self.executor.run();result=native_arrays(raw)
                details=dict(runtime=raw.get('cuda_runtime',raw.get('metal_runtime')),timings=raw['timings'])
                if 'host_storage' in raw:details['host_storage']=raw['host_storage']
            elif self.backend=='cpu-f32':
                from brian2_rust.cuda import CudaExecutor
                raw=CudaExecutor._cpu_control(self.context['control'],512*1024**2,1)
                result=native_arrays(raw);details=dict(native_run_seconds=raw['run_seconds'])
            elif self.backend=='rust-f64':
                from brian2_rust.results import load_results
                artifact=self.context['device'].native_artifact
                directory=self.output/f'native-replay-{self.count}'
                if directory.exists():raise RuntimeError('Result directory already exists')
                subprocess.run([str(artifact['binary']),str(artifact['instance']),str(directory)],check=True,
                    stdout=subprocess.DEVNULL,env={**os.environ,'B2_NUM_THREADS':str(rust_threads),'B2_THREAD_AFFINITY':'auto'})
                result=native_arrays(load_results(self.model,directory));result_directory=directory
                details['rust_runtime']=json.loads((directory/'summary.json').read_text())
                details['requested_rust_threads']=rust_threads
            elif self.backend=='brian2cuda' or self.backend in CPP_BACKENDS:
                dev=self.context['device'];project=self.output/'project';name=f'replay-{self.count}'
                if (project/name).exists():raise RuntimeError('Result directory already exists')
                dev.run(directory=str(project),results_directory=name,with_output=False);result_directory=project/name
                result=self._brian_arrays()
                details=dict(backend_last_run_seconds=float(dev._last_run_time),scope='backend-specific diagnostic only')
                if self.backend in CPP_BACKENDS:
                    details['observed_openmp_threads']=int((result_directory/'openmp_threads.txt').read_text())
                    if details['observed_openmp_threads']!=int(self.backend[-1]):raise RuntimeError('Unexpected OpenMP team size')
            elif self.backend=='brian2genn-corrected':
                from gpu_brian2genn_stdp_adapter import replay
                result,phases=replay(self.context,self.count)
                result_directory=self.output/'project/results'
                details['brian2genn_phases']=phases
            else:
                if self.genn_gather:
                    from gpu_genn_readback import replay
                    result=replay(self.context,self.steps)
                    details['genn_readback']=dict(self.context['last_gather'])
                else:
                    from gpu_stdp_compare import genn_replay
                    result=genn_replay(self.context,self.steps)
                details['genn_phases']=self.context['last_replay_timings']
        elapsed=time.perf_counter()-start
        if result_directory is not None:
            details['output_file_bytes']=sum(p.stat().st_size for p in result_directory.rglob('*') if p.is_file())
        details['result_array_bytes']=sum(a.nbytes for a in result.values())
        return result,dict(wall_seconds=elapsed,details=details)

    def close(self):
        if self.executor is not None:self.executor.close()


def write_result(output,name,result):
    path=output/(name+'.npz');np.savez_compressed(path,**result)
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        arrays={k:dict(dtype=str(v.dtype),shape=list(v.shape),sha256=hashlib.sha256(v.tobytes()).hexdigest()) for k,v in result.items()})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend',choices=BACKENDS+CPP_BACKENDS+PREFIX_BACKENDS+BITSET_BACKENDS+CORRECTED_GENN_BACKENDS+GATHER_GENN_BACKENDS+F32_GENN_BACKENDS+('metal',),required=True);parser.add_argument('--neurons',type=int,default=4096)
    parser.add_argument('--steps',type=int,default=512);parser.add_argument('--degree',type=int,default=32)
    parser.add_argument('--drive',type=float,default=.125);parser.add_argument('--delay-span',type=int,default=4);parser.add_argument('--post-delay',type=int,default=2)
    parser.add_argument('--topology-kind',choices=('ring','random-fixed-outdegree'),default='ring');parser.add_argument('--topology-seed',type=int,default=0)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    from gpu_stdp_compare import workload_options
    workload=workload_options(args.drive,args.delay_span,args.post_delay,args.topology_kind,args.topology_seed)
    from gpu_stdp_compare import topology,MAX_NEURONS
    if not 2<=args.neurons<=MAX_NEURONS or not 1<=args.steps<=MAX_REPLAY_STEPS:parser.error('Require 2..16384 neurons and 1..4096 replay steps')
    try:topology(args.neurons,args.degree,delay_span=args.delay_span,topology_kind=args.topology_kind,topology_seed=args.topology_seed)
    except ValueError as error:parser.error(str(error))
    args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=False)
    # Preserve a dedicated protocol descriptor, then direct all Python/native
    # compiler stdout to stderr. Compiler output cannot impersonate JSON replies.
    protocol=os.fdopen(os.dup(sys.stdout.fileno()),'w',buffering=1);os.dup2(sys.stderr.fileno(),sys.stdout.fileno())
    replay=None
    def emit(value):protocol.write(json.dumps(value,allow_nan=False)+'\n');protocol.flush()
    try:
        replay=Replay(args.backend,args.neurons,args.steps,args.degree,args.output,**workload)
        emit(dict(event='ready',backend=args.backend,configuration=configuration(args.neurons,args.steps,args.degree,**workload),
            arithmetic=('float32 states and decay factor/product; double time and exponential; schedule-corrected Brian2GeNN variant' if replay.genn_f32_factor else 'full f64 storage/expressions' if args.backend=='rust-f64' or args.backend in CPP_BACKENDS else 'float32; explicit delayed STDP mapping' if replay.backend in {'genn','brian2genn-corrected'} else 'float32'),
            lifecycle={**{b:'fresh compiled Brian2 C++ OpenMP process and output directory' for b in CPP_BACKENDS},'rust-f64':'fresh compiled native process','cpu-f32':'retained C++ library and initial topology, fresh writable arrays','cuda':'retained executor and device buffers, reset writable initial state','brian2cuda':'fresh standalone process and output directory','metal':'retained executor, reset writable state and queues','genn':'reload compiled model, run with GPU trace recording, synchronized full pull, unload','brian2genn-corrected':'rotate previous outputs, clear result caches, fresh corrected standalone process, read all arrays'}[replay.backend],
            synapse_prefix=replay.synapse_prefix,synapse_sparse=replay.synapse_sparse,genn_barrier=replay.genn_barrier,genn_gather=replay.genn_gather,adapter_evidence=replay.adapter_evidence,
            setup_seconds=replay.setup_seconds,setup_timing=replay.setup_timing,artifacts=replay.artifacts,
            bootstrap=write_result(args.output,'bootstrap',replay.bootstrap)))
        for line in sys.stdin:
            request=json.loads(line)
            if request['command']=='close':
                replay.close();replay=None;emit(dict(event='closed'));break
            if request['command']!='run':raise ValueError('Unknown worker command')
            result,timing=replay.run(rust_threads=request.get('rust_threads',1))
            # Hashing, NPZ compression and artifact verification are outside the
            # measured reset-to-result interval, and cannot hide a changed binary.
            if artifact_hashes(args.output)!=replay.artifacts:raise RuntimeError('Compiled artifact changed during replay')
            artifact=write_result(args.output,f'replay-{replay.count}',result)
            emit(dict(event='result',sample=request['sample'],**timing,result=artifact))
    except BaseException as error:
        emit(dict(event='error',error_type=type(error).__name__,error=str(error)))
        raise
    finally:
        if replay is not None:replay.close()
        protocol.close()


if __name__=='__main__':main()
