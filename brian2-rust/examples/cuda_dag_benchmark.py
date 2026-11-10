"""Matched direct/resident/graph CUDA replay ablation on recurrent CUBA.

Cold first replay and NVCC setup are separate from seven interleaved warm
replays. Every replay must match the direct f32 result bit for bit. This is a
CUDA runtime ablation, not an f64 or competing-backend speedup claim.
"""
import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import statistics
import time

import numpy as np


def workload(neurons,steps,degree,directory):
    import brian2 as b
    import brian2_rust
    from brian2_rust.export import lower_network
    from gpu_recurrent import arrays,DT_MS,REF_TICKS,DELAY_TICKS
    b.set_device('atlas',engine='reference',directory=directory,
                 runner=Path(__file__).resolve().parents[1]/'target/release/b2-runner')
    v,drive,projections=arrays(neurons,degree)
    pop=b.NeuronGroup(neurons,'''dv/dt=(drive-v+I_syn)/(20*ms):1 (unless refractory)
        dI_syn/dt=-I_syn/(5*ms):1
        drive:1 (constant)''',threshold='v>1',reset='v=0',
        refractory=REF_TICKS*DT_MS*b.ms,method='euler',dt=DT_MS*b.ms,dtype=dict(v=np.float32,I_syn=np.float32,drive=np.float32),name='population')
    pop.v=v;pop.drive=drive
    synapses=[]
    for label,(sources,targets,weight) in zip(('exc','inh'),projections,strict=True):
        syn=b.Synapses(pop,pop,'w:1 (constant)',on_pre='I_syn_post+=w',
                      delay=DELAY_TICKS*DT_MS*b.ms,clock=pop.clock,name=label)
        syn.connect(i=sources.astype(int),j=targets.astype(int));syn.w=float(weight);synapses.append(syn)
    return lower_network(b.Network(pop,*synapses,b.SpikeMonitor(pop)),steps*DT_MS*b.ms)


def result_digest(result):
    digest=hashlib.sha256()
    def visit(value):
        if isinstance(value,np.ndarray):
            digest.update(json.dumps([value.dtype.str,value.shape]).encode());digest.update(value.tobytes())
        elif isinstance(value,dict):
            for key in sorted(value):digest.update(json.dumps(key).encode());visit(value[key])
        elif isinstance(value,(tuple,list)):
            digest.update(str(len(value)).encode())
            for item in value:visit(item)
        elif isinstance(value,np.generic):visit(value.item())
        else:digest.update(json.dumps(value,allow_nan=False).encode())
    visit({key:result[key] for key in ('populations','synapses')})
    return digest.hexdigest()


def measure(executor,expected=None):
    started=time.perf_counter();result=executor.run();wall=time.perf_counter()-started
    digest=result_digest(result)
    if expected is not None and digest!=expected:raise AssertionError('CUDA DAG replay differs from direct f32')
    return result,dict(wall_seconds=wall,run_seconds=result['run_seconds'],
        gpu_seconds=sum(t['gpu_seconds'] for t in result['timings']),
        command_seconds=sum(t['command_seconds'] for t in result['timings']),
        input_seconds=sum(t['input_seconds'] for t in result['timings']),
        readback_seconds=sum(t['readback_seconds'] for t in result['timings']),
        execution=result['cuda_runtime']['dag_execution'],result_sha256=digest)


def benchmark(model,directory,repeats):
    from brian2_rust.cuda import CudaExecutor
    rows=[];reference=None;runtime=None
    for route in ('scan','sparse'):
        with ExitStack() as stack:
            executors={};rows_by_mode={}
            for mode in ('direct','resident','graph'):
                start=time.perf_counter()
                ex=stack.enter_context(CudaExecutor(model,directory/route/mode,numeric_mode='float32',event_delivery=route,dag_execution=mode))
                setup=time.perf_counter()-start
                result,cold=measure(ex,reference)
                if reference is None:reference=cold['result_sha256']
                runtime=result['cuda_runtime']
                row=dict(route=route,mode=mode,plan_sha256=ex.plan.sha256,setup_seconds=setup,
                         compile_seconds=ex.compile_seconds,cold=cold,warm=[],numeric_profile=result['numeric_profile'])
                # A warm-up replay is explicitly excluded from the sample set.
                _,row['warmup']=measure(ex,reference)
                rows.append(row);rows_by_mode[mode]=row;executors[mode]=ex
            rng=np.random.default_rng(1729)
            for round_index in range(repeats):
                for ordinal,mode in enumerate(rng.permutation(['direct','resident','graph'])):
                    _,sample=measure(executors[mode],reference)
                    sample.update(round=round_index,order=ordinal);rows_by_mode[mode]['warm'].append(sample)
            for row in rows_by_mode.values():
                row['median']={key:statistics.median(sample[key] for sample in row['warm']) for key in
                               ('wall_seconds','run_seconds','gpu_seconds','command_seconds','input_seconds','readback_seconds')}
    return dict(rows=rows,result_sha256=reference,cuda_runtime=runtime,
                correctness='all cold, warmup and measured outputs bitwise equal to direct f32 across both routes')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--neurons',type=int,default=512)
    parser.add_argument('--steps',type=int,default=512)
    parser.add_argument('--degree',type=int,default=32)
    parser.add_argument('--repeats',type=int,default=7)
    args=parser.parse_args()
    if not 1<=args.repeats<=20 or not 1<=args.steps<=4096 or not 8<=args.neurons<=8192:parser.error('bounded benchmark size exceeded')
    args.output.mkdir(parents=True,exist_ok=False)
    from gpu_recurrent import configuration
    model=workload(args.neurons,args.steps,args.degree,args.output/'model')
    source=json.dumps(model,sort_keys=True,indent=2)+'\n';(args.output/'model.json').write_text(source)
    report=dict(schema='b2-cuda-dag-ablation-v0',configuration=configuration(args.neurons,args.steps,args.degree),
                repeats=args.repeats,model_sha256=hashlib.sha256(source.encode()).hexdigest(),
                timing_scope='same CUDA kernels and recording; wall includes reset, dispatch, synchronized readback and result decoding; excludes construction/NVCC and correctness hashing',
                limitations=['immutable snapshot replay; no between-replay learning',
                             'f32 direct equivalence does not establish reference-f64 trajectory equivalence',
                             'single GPU and workload family; not competing-backend performance'],
                **benchmark(model,args.output/'executors',args.repeats))
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps([{key:row[key] for key in ('route','mode','median')} for row in report['rows']],indent=2))


if __name__=='__main__':main()
