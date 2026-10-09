"""Explicit full-activation tuning: report calibration cost as well as replay time."""
import argparse
import json
from pathlib import Path
import statistics
import time
import numpy as np

CASES=(('quiet',4096,8,1/64,16,16),('wide',4096,32,1/16,8,3))


def save_observables(path,result):
    arrays={}
    def encode(value):
        if isinstance(value,np.ndarray):
            name='array'+str(len(arrays));arrays[name]=value
            return {'array':name}
        if isinstance(value,np.generic):return value.item()
        if isinstance(value,dict):return {k:encode(v) for k,v in value.items()}
        if isinstance(value,(tuple,list)):return [encode(v) for v in value]
        return value
    tree=encode({k:result.get(k) for k in ('populations','synapses','numeric_profile','rng_profile')})
    path.with_suffix('.json').write_text(json.dumps(tree)+'\n')
    np.savez_compressed(path.with_suffix('.npz'),**arrays)


def benchmark(output,backend):
    from brian2_rust.gpu_autotune import tune,observable_fingerprint
    from brian2_rust.cuda import CudaExecutor,build_cuda_plan
    from brian2_rust.metal import MetalExecutor,build_metal_plan
    from gpu_stdp_compare import brian_run,oracle,checks,configuration,activity
    from gpu_stdp_precompiled import native_arrays,forbid_compilation,artifact_hashes
    cls=MetalExecutor if backend=='metal' else CudaExecutor
    build=build_metal_plan if backend=='metal' else build_cuda_plan
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    report=dict(schema='b2-gpu-autotune-benchmark-v1',backend=backend,passed=False,cases=[],
        scope='full tuning includes compilation, warmup, selection, hashing, final replay and loser cleanup; replay samples exclude those preparation costs')
    def save():(output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    try:
        for name,n,degree,drive,delay,post in CASES:
            directory=output/name;directory.mkdir()
            opts=dict(drive=drive,delay_span=delay,post_delay=post,topology_kind='random-fixed-outdegree',topology_seed=42)
            context={};cpu_dir=directory/'cpu';cpu_dir.mkdir()
            cpu,_=brian_run('cpu-f32',n,degree,1024,cpu_dir,prepared=context,**opts)
            model=context['control'].model
            control=CudaExecutor._cpu_control(context['control'],512*1024**2,1)
            refs={p:oracle(n,degree,1024,dtype=d,**opts) for p,d in [('f32',np.float32),('f64',np.float64)]}
            assert checks(cpu,refs['f32'])['passed']
            expected_events=activity(n,degree,1024,refs['f32'],**opts)
            expected_events=sum(expected_events[k] for k in ('delivered_pre_events','delivered_post_events'))
            row=dict(name=name,configuration=configuration(n,degree,1024,**opts),expected_events=expected_events,
                replay_samples=[],passed=False)
            report['cases'].append(row);save()
            (directory/'model.json').write_text(json.dumps(model)+'\n')
            for p,values in refs.items():np.savez_compressed(directory/('reference-'+p+'.npz'),**values)
            plans={};prepared={};baseline=None
            def plan_for(name,prefix,sparse):
                plan=build(model,numeric_mode='float32',event_delivery='sparse',synapse_prefix=prefix,synapse_sparse=sparse)
                prepared[name]=plan;plans[name]=plan.to_dict()
                return plan
            def make(name,prefix,sparse):
                nonlocal baseline
                ex=cls(model,directory/name,numeric_mode='float32',event_delivery='sparse',
                    synapse_prefix=prefix,synapse_sparse=sparse,plan=prepared[name],
                    compile_reuse=True,reuse_from=baseline)
                if name=='baseline':baseline=ex
                return ex
            start=time.perf_counter()
            winner,actual,tuning=tune(make,directory/'tuning',plan_for=plan_for)
            row['tuning_wall_seconds']=time.perf_counter()-start
            try:
                row['tuning']=tuning;row['plans']=plans
                # Normalize only the declared backend profile for cross-platform
                # control comparison; population/synapse leaves remain exact.
                control['numeric_profile']=actual.get('numeric_profile')
                assert control.get('rng_profile')==actual.get('rng_profile')
                expected_hash=observable_fingerprint(control)
                assert expected_hash==tuning['reference_observable_sha256']
                save_observables(directory/'control-observables',control)
                save_observables(directory/'selected-observables',actual)
                def validate(raw,label):
                    assert observable_fingerprint(raw)==expected_hash
                    values=native_arrays(raw)
                    gates={p:checks(values,ref) for p,ref in refs.items()}
                    assert gates['f32']['passed']
                    assert sum(s['events'] for s in raw['synapses'])==expected_events
                    np.savez_compressed(directory/(label+'.npz'),**values)
                    return gates
                row['selected_gates']=validate(actual,'selected')
                before=artifact_hashes(directory/tuning['selected'])
                for i in range(5):
                    with forbid_compilation():
                        start=time.perf_counter();raw=winner.run();elapsed=time.perf_counter()-start
                    gates=validate(raw,'replay-'+str(i))
                    row['replay_samples'].append(dict(seconds=elapsed,gates=gates,
                        observable_sha256=observable_fingerprint(raw)))
                    save()
                assert before==artifact_hashes(directory/tuning['selected'])
                row['compiled_artifacts']=before
                row['selected_replay_median_seconds']=statistics.median(s['seconds'] for s in row['replay_samples'])
                row['baseline_profiling_median_seconds']=statistics.median(tuning['candidates']['baseline']['seconds'])
                # A new activation currently tunes again; this is not an amortized
                # Device speedup, regardless of a faster selected replay.
                row['passed']=True;save()
            finally:winner.close()
            print(name,'selected',tuning['selected'],'tuning seconds',row['tuning_wall_seconds'],flush=True)
        report['passed']=True
    finally:save()
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend',choices=('metal','cuda'),required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();benchmark(args.output,args.backend)
