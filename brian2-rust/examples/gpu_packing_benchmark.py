"""Randomized paired scalar/bulk packing through full GPU cache-hit activations."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import time
import numpy as np

from gpu_autotune_benchmark import CASES, save_observables
import struct
from brian2_rust.gpu_types import width
from brian2_rust.plan import PlanValidationError


def scalar_pack(values, dtype):
    if dtype in {'f32','f64'}:
        values = [struct.unpack('>f' if len(v)==8 else '>d', bytes.fromhex(v))[0]
                  if isinstance(v,str) else v for v in values]
        with np.errstate(over='ignore'): result = np.asarray(values, np.float32)
        if not np.isfinite(result).all():
            raise PlanValidationError('GPU initial value cannot be represented as finite float32')
        return result
    raw = np.asarray([int(v,16) if isinstance(v,str) else int(v) for v in values],
                     dtype=np.uint64 if width(dtype)==2 else np.uint32)
    if dtype=='bool': raw=(raw!=0).astype(np.uint32)
    if width(dtype)==2:
        return np.concatenate((raw.astype(np.uint32),(raw >> np.uint64(32)).astype(np.uint32))).view(np.float32)
    return raw.view(np.float32)


def benchmark(output, backend):
    from brian2_atlas import gpu_types
    bulk_pack=gpu_types.pack
    from brian2_rust.gpu_autotune import observable_fingerprint
    from brian2_rust.gpu_tuning_cache import TuningCache
    from brian2_rust.gpu_buffer_transfer import execute_device
    from brian2_rust.results import load_results
    from brian2_rust.cuda import CudaExecutor
    from gpu_stdp_compare import brian_run, oracle, checks, configuration, activity
    from gpu_stdp_precompiled import native_arrays
    root=Path(__file__).resolve().parents[1]
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    report=dict(schema='b2-gpu-packing-benchmark-v1',backend=backend,passed=False,cases=[],
        scope='identical complete model; fresh executor, input hashing, plan validation, compiler-context checks, execution, full-result hashing, result write/read and cache publication; Brian lowering and model export excluded')
    def save():(output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    try:
        for name,n,degree,drive,delay,post in CASES:
            directory=output/name;directory.mkdir()
            opts=dict(drive=drive,delay_span=delay,post_delay=post,topology_kind='random-fixed-outdegree',topology_seed=42)
            prepared={};cpu_dir=directory/'cpu';cpu_dir.mkdir()
            cpu,_=brian_run('cpu-f32',n,degree,1024,cpu_dir,prepared=prepared,**opts)
            model=prepared['control'].model
            control=CudaExecutor._cpu_control(prepared['control'],512*1024**2,1)
            refs={p:oracle(n,degree,1024,dtype=d,**opts) for p,d in [('f32',np.float32),('f64',np.float64)]}
            assert checks(cpu,refs['f32'])['passed']
            activities=activity(n,degree,1024,refs['f32'],**opts)
            events=sum(activities[k] for k in ('delivered_pre_events','delivered_post_events'))
            (directory/'model.json').write_text(json.dumps(model)+'\n')
            for p,values in refs.items():np.savez_compressed(directory/('reference-'+p+'.npz'),**values)
            device=SimpleNamespace(build_options=dict(engine=backend,event_delivery='sparse',
                gpu_autotune=True,gpu_autotune_cache=True,gpu_compile_reuse=True,gpu_buffer_reuse=True),
                _gpu_executor=None,_gpu_tuning_cache=TuningCache(),last_gpu_tuning=None)
            row=dict(name=name,configuration=configuration(n,degree,1024,**opts),expected_events=events,
                     activations=[],passed=False)
            report['cases'].append(row);save()
            try:
                order=[('bulk',-2),('scalar',-1),('bulk',-1)]
                rng=np.random.default_rng(2718)
                for pair in range(5):
                    order.extend((str(p),pair) for p in rng.permutation(['scalar','bulk']))
                row['declared_order']=[dict(policy=p,pair=k) for p,k in order];save()
                for i,(policy,pair) in enumerate(order):
                    path=directory/('activation-'+str(i));path.mkdir()
                    gpu_types.pack=scalar_pack if policy=='scalar' else bulk_pack
                    try:
                        start=time.perf_counter()
                        pending=execute_device(device,model,path,root/'target/release/b2-runner')
                        actual=load_results(model,path/'rust')
                        device._gpu_tuning_cache.publish(*pending)
                        wall=time.perf_counter()-start
                    finally:gpu_types.pack=bulk_pack
                    tuning=device.last_gpu_tuning
                    assert tuning['cache']['status']==('miss' if i==0 else 'hit')
                    # The transport adds seconds arrays; compare the complete raw
                    # semantic contract separately using a fresh retained replay.
                    raw=device._gpu_executor.run()
                    control['numeric_profile']=raw['numeric_profile']
                    assert control.get('rng_profile')==raw.get('rng_profile')
                    expected_hash=observable_fingerprint(control)
                    assert observable_fingerprint(raw)==expected_hash==tuning['reference_observable_sha256']
                    save_observables(path/'raw-observables',raw)
                    save_observables(path/'transport-observables',actual)
                    if i==0:save_observables(directory/'control-observables',control)
                    values=native_arrays(actual)
                    gates={p:checks(values,ref) for p,ref in refs.items()}
                    assert gates['f32']['passed']
                    assert sum(s['events'] for s in actual['synapses'])==events
                    np.savez_compressed(path/'snapshot.npz',**values)
                    row['activations'].append(dict(index=i,policy=policy,pair=pair,wall_seconds=wall,tuning=tuning,gates=gates,
                        compilation=actual['metadata'][backend+'_runtime']['compilation'],
                        buffer_reuse=actual['metadata'][backend+'_runtime']['activation_buffer_reuse'],
                        observable_sha256=expected_hash))
                    save()
                assert len(device._gpu_tuning_cache)==1
                row['passed']=True;save()
            finally:
                if device._gpu_executor is not None:device._gpu_executor.close()
            print(name,[(a['tuning']['cache']['status'],a['wall_seconds']) for a in row['activations']],flush=True)
        report['passed']=True
    finally:save()
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend',choices=('metal','cuda'),required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();benchmark(args.output,args.backend)
