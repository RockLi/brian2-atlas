"""Paired dense-target saturation of sparse event expansion on a naturally continued GPU activation."""
from contextlib import contextmanager
from unittest.mock import patch
import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics
import time
import numpy as np

CASES=(('quiet',4096,8,1/64,16,16),('low',4096,8,17/512,16,16),('dense',1024,128,1/16,8,3))


@contextmanager
def queue_reservations(*, saturate):
    """Select an experimental producer for plan construction in this process.

    Production keeps unbounded reservations: the bounded producer passed native
    correctness checks but did not improve dense replay on the tested devices.
    This scoped patch is only for the single-threaded comparison/test harness.
    """
    from brian2_atlas import metal_dag
    current=metal_dag.sparse_history_kernel
    def original(*args,**kwargs):
        kwargs['saturate']=saturate
        return current(*args,**kwargs)
    with patch.object(metal_dag,'sparse_history_kernel',original):yield


def original_queue():
    return queue_reservations(saturate=False)


def bounded_queue():
    return queue_reservations(saturate=True)


def compare(output,backend="cuda",*,_variants=None,_schema='b2-sparse-saturation-comparison-v0',_scope=None):
    from gpu_stdp_compare import brian_run,oracle,checks,configuration,activity
    from gpu_stdp_precompiled import native_arrays,forbid_compilation,artifact_hashes
    from brian2_rust.cuda import CudaExecutor
    from brian2_rust.metal import MetalExecutor
    if backend not in {"metal","cuda"}:raise ValueError("Native backend required")
    Executor=MetalExecutor if backend=="metal" else CudaExecutor
    variants=_variants or dict(baseline=original_queue,bounded=bounded_queue)
    if len(variants)<2:raise ValueError('Comparison needs at least two variants')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    report=dict(schema=_schema,backend=backend,cases=[],passed=False,repeats=5,seed=1729,variants=list(variants),
        scope=_scope or 'separate retained continuation executors with full / saturation-bounded target queues; 128-tick native warm state independently checked; bootstrap compilation excluded; full fresh-reset to result timings; no external backend ranking')
    (output/'declared-protocol.json').write_text(json.dumps(dict(cases=CASES,variants=list(variants),warm_steps=128,steps=1024,repeats=5,warmups=1,seed=1729,numeric_contract='explicit-f32-v1',backend=backend,required_event_count='independent emitted-spike topology recurrence'),indent=2)+'\n')
    def save(): (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    try:
        for name,n,degree,drive,delay,post in CASES:
            directory=output/name;directory.mkdir()
            opts=dict(drive=drive,delay_span=delay,post_delay=post,topology_kind='random-fixed-outdegree',topology_seed=42)
            row=dict(name=name,configuration=configuration(n,degree,1024,**opts),samples=[],passed=False)
            report['cases'].append(row);save()
            warm=128;total=warm+1024
            whole={p:oracle(n,degree,total,dtype=dtype,**opts) for p,dtype in [('f32',np.float32),('f64',np.float64)]}
            warm_ref=oracle(n,degree,warm,dtype=np.float32,**opts)
            refs={}
            for precision,result in whole.items():
                mask=result['ticks']>=warm
                refs[precision]={k:(v[warm:].copy() if k=='trace' else v[mask].copy() if k in {'ticks','indices'} else v.copy()) for k,v in result.items()}
            context={};(directory/'warm-native').mkdir()
            warm_actual,_=brian_run(backend,n,degree,warm,directory/'warm-native',prepared=context,**opts)
            from brian2_rust.export import lower_network
            import brian2 as b
            model=lower_network(context['network'],1024*b.second/1024)
            # Match Device.network_run's continuation preparation: Brian arrays
            # alone do not contain delayed events retained by the native Device.
            context['device']._inject_pending_events(model)
            from brian2_rust.protocol import attach_protocol
            attach_protocol(model)
            context['device'].close_gpu()
            (directory/'warm-cpu').mkdir()
            warm_cpu,_=brian_run('cpu-f32',n,degree,warm,directory/'warm-cpu',**opts)
            assert all(np.array_equal(warm_actual[k],warm_cpu[k]) for k in warm_actual)
            row['warm_gate']=checks(warm_actual,warm_ref);assert row['warm_gate']['passed']
            for label,values in [('native',warm_actual),('reference',warm_ref),('compiled',warm_cpu)]:
                np.savez_compressed(directory/('warm-'+label+'.npz'),**values)
            from brian2_rust.metal import build_metal_plan
            from brian2_rust.metal_dag import run_dag
            from types import SimpleNamespace
            cpu_path=directory/'cpu';cpu_path.mkdir()
            control=SimpleNamespace(model=model,plan=build_metal_plan(model,numeric_mode='float32',event_delivery='sparse'),directory=cpu_path,compile_seconds=0,device_name='CPU f32')
            cpu=native_arrays(run_dag(control,max_buffer_bytes=512*1024**2,compute='cpu-f32',workers=3))
            refs['compiled_f32']=cpu
            full_events=activity(n,degree,total,whole['f32'],**opts)
            warm_events=activity(n,degree,warm,warm_ref,**opts)
            row['expected_delivered_events']=sum(full_events[k]-warm_events[k] for k in ('delivered_pre_events','delivered_post_events'))
            row['warm_steps']=warm
            row['pending_entries']={p['name']:len(p['pending']) for p in model['instance']['synapses'][0]['pathways']}
            if name!='quiet':assert sum(row['pending_entries'].values())>0
            for label,values in refs.items():np.savez_compressed(directory/('reference-'+label+'.npz'),**values)
            if not checks(cpu,refs['f32'])['passed']:raise RuntimeError('Continuation CPU f32 fails independent reference')
            (directory/'model.json').write_text(json.dumps(model)+'\n')
            executors={}
            try:
                for mode,context in variants.items():
                    with context():
                        ex=Executor(model,directory/mode,numeric_mode='float32',event_delivery='sparse',synapse_sparse=True)
                    executors[mode]=ex
                before=next(iter(executors.values())).plan
                for ex in executors.values():
                    after=ex.plan
                    assert before.logical==after.logical and before.buffers==after.buffers
                    assert before.dispatches==after.dispatches
                    assert sum(d.role=='target-owned-sparse-synapse-pathway' for d in after.dispatches)==1
                row['new_buffers']=[]
                row['plans']={mode:ex.plan.to_dict() for mode,ex in executors.items()}
                def run(mode,round_index,position):
                    started=time.perf_counter()
                    raw=executors[mode].run();values=native_arrays(raw)
                    elapsed=time.perf_counter()-started
                    gates={p:checks(values,ref) for p,ref in refs.items()}
                    file=f'{mode}-{round_index}.npz';np.savez_compressed(directory/file,**values)
                    sample=dict(mode=mode,round=round_index,position=position,wall_seconds=elapsed,
                        timings=raw['timings'],runtime=raw.get('cuda_runtime',raw.get('metal_runtime')),delivered_events=sum(s['events'] for s in raw['synapses']),host_storage=raw.get('host_storage'),gates=gates,
                        result=file,sha256=hashlib.sha256((directory/file).read_bytes()).hexdigest())
                    row['samples'].append(sample);save()
                    if not (gates['f32']['passed'] and gates['compiled_f32']['passed']):raise RuntimeError('Full f32 gate failed')
                    if any(not np.array_equal(values[k],cpu[k]) for k in values):raise RuntimeError('Compiled f32 bitwise gate failed')
                    if sample['delivered_events']!=row['expected_delivered_events']:
                        raise RuntimeError('Sparse event counter does not match independent event count')
                for mode in executors:run(mode,-2,0)
                before={mode:artifact_hashes(directory/mode) for mode in executors}
                rng=random.Random(1729)
                for i in range(-1,5):
                    modes=list(executors);rng.shuffle(modes)
                    for position,mode in enumerate(modes):
                        with forbid_compilation():run(mode,i,position)
                    print(name,'round',i,'complete',flush=True)
                after={mode:artifact_hashes(directory/mode) for mode in executors}
                if before!=after:raise RuntimeError('Compiled binaries changed during replay')
                row['compiled']=after
                row['summary']={mode:{'median_ms':statistics.median(values)*1000,'min_ms':min(values)*1000,'max_ms':max(values)*1000}
                    for mode in executors for values in [[s['wall_seconds'] for s in row['samples'] if s['mode']==mode and s['round']>=0]]}
                row['passed']=True;save()
            finally:
                for ex in executors.values():ex.close()
        report['passed']=True
    finally:save()
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--backend',choices=('metal','cuda'),default='cuda')
    args=p.parse_args();print(json.dumps(compare(args.output,args.backend),indent=2))
