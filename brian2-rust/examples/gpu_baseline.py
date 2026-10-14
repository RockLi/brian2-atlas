"""Common IF, recurrent LIF and HH ionic GPU comparison workloads.

Run each adapter in its pinned environment on the same physical GPU. The default
case has no synapses; recurrent-cuba-v0 adds fixed delayed E/I feedback.
Compiler/startup-inclusive wall time and backend-specific timers are kept separate.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np


def initial_arrays(neurons):
    index=np.arange(neurons)
    return ((index%13)/64).astype(np.float32),((index%17+1)/16).astype(np.float32)


def oracle(neurons,steps):
    """Independent exact dyadic recurrence: v += drive/16; spike if v>1; v=0."""
    v,drive=initial_arrays(neurons)
    ticks,indices=[],[]
    for tick in range(steps):
        v+=drive/np.float32(16)
        fired=np.flatnonzero(v>1)
        ticks.extend([tick]*len(fired));indices.extend(fired)
        v[fired]=0
    return dict(v=v,ticks=np.asarray(ticks,np.int64),indices=np.asarray(indices,np.int64))


def canonical_result(v,times_ms,indices,dt_ms):
    tick_values=np.asarray(times_ms,dtype=np.float64)/dt_ms
    ticks=np.rint(tick_values).astype(np.int64)
    if not np.allclose(tick_values,ticks,rtol=0,atol=1e-5):
        raise ValueError("Backend reported spike times off the declared tick grid")
    indices=np.asarray(indices,np.int64)
    order=np.lexsort((indices,ticks))
    return dict(v=np.asarray(v).reshape(-1),ticks=ticks[order],indices=indices[order])


def cpu_f32_control(net,duration,output,*,event_delivery="scan",prepared=None):
    """Run the GPU expression/control program on one compiled C++ CPU worker."""
    from types import SimpleNamespace
    from brian2_rust.export import lower_network
    from brian2_rust.cuda import CudaExecutor,build_cuda_plan
    # This adapter bypasses Network.run's lifecycle. Align shared Clock
    # objects to this Network's absolute time before direct lowering, just as
    # Brian's normal activation does; a previous comparison may have used the
    # same defaultclock with a different backend.
    for clock in {obj.clock for obj in net.sorted_objects}:
        clock.set_interval(net.t,net.t+duration)
    model=lower_network(net,duration)
    control=SimpleNamespace(model=model,directory=output/'control',
        plan=build_cuda_plan(model,numeric_mode='float32',event_delivery=event_delivery))
    control.directory.mkdir()
    if prepared is not None:prepared['control']=control
    return CudaExecutor._cpu_control(control,512*1024**2,1)


def brian_run(backend,neurons,steps,output,profile):
    import brian2 as b
    if backend in {"cuda","metal","rust","cpu-f32"}:
        import brian2_rust
        options={"engine":"aot" if backend in {"rust","cpu-f32"} else backend,"directory":output/"project"}
        runner=Path(__file__).resolve().parents[1]/"target/release/b2-runner"
        if runner.is_file():options["runner"]=runner
        if backend in {"cuda","metal"}:options["numeric_mode"]="float32"
        b.set_device("atlas",**options)
    elif backend=="brian2cuda":
        import brian2cuda
        b.set_device("cuda_standalone",directory=str(output/"project"))
    else:
        import brian2genn
        b.set_device("genn",directory=str(output/"project"),use_GPU=True)
    b.prefs.core.default_float_dtype=np.float32
    b.defaultclock.dt=.0625*b.ms
    pop=b.NeuronGroup(neurons,"dv/dt=drive/ms:1\ndrive:1 (constant)",threshold="v>1",reset="v=0",
                      method="euler",name="population")
    pop.v,pop.drive=initial_arrays(neurons)
    spikes=b.SpikeMonitor(pop)
    net=b.Network(pop,spikes)
    start=time.perf_counter()
    if backend=="cpu-f32":
        raw=cpu_f32_control(net,steps*.0625*b.ms,output)
        pop_result=raw["populations"][0]
        return dict(v=pop_result["states"]["v"],ticks=pop_result["spike_ticks"],indices=pop_result["indices"]),dict(
            build_run_seconds=time.perf_counter()-start,simulation_seconds=raw["run_seconds"],
            scope="one-worker compiled C++ GPU-f32 expression control")
    net.run(steps*.0625*b.ms,profile=profile)
    build_run_seconds=time.perf_counter()-start
    start=time.perf_counter()
    result=canonical_result(pop.v[:],spikes.t[:]/b.ms,spikes.i[:],.0625)
    readback_seconds=time.perf_counter()-start
    dev=b.get_device()
    timing=dict(build_run_seconds=build_run_seconds,final_array_read_seconds=readback_seconds,
                backend_reported_last_run_seconds=float(dev._last_run_time),
                last_run_scope="backend-specific; do not compare this field across implementations")
    if backend in {"cuda","metal","rust"}:
        metadata=json.loads((dev.last_run_directory/"rust/summary.json").read_text())
        timing["native_metadata"]=metadata
    if profile:
        timing["profile"]={name:float(value/b.second) for name,value in net.profiling_info}
    return result,timing


def genn_run(neurons,steps,output,profile):
    from pygenn import GeNNModel,create_neuron_model
    model=GeNNModel("float","b2_if_comparison",backend="cuda")
    model.dt=.0625
    model.timing_enabled=profile
    neuron=create_neuron_model("ExactIF",vars=[("v","scalar"),("drive","scalar")],
                               sim_code="v += dt * drive;",threshold_condition_code="v > 1.0",
                               reset_code="v = 0.0;")
    v,drive=initial_arrays(neurons)
    pop=model.add_neuron_population("population",neurons,neuron,{},dict(v=v,drive=drive))
    pop.spike_recording_enabled=True
    (output/"project").mkdir(parents=True,exist_ok=True)
    # Bound build parallelism to this runner's CPU allocation. Restore the
    # library helper immediately; this does not change simulation execution.
    import pygenn.genn_model as genn_module
    original_cpu_count=genn_module.cpu_count
    start=time.perf_counter()
    try:
        genn_module.cpu_count=lambda logical=True: 2
        model.build(path_to_model=str(output/"project"))
    finally:
        genn_module.cpu_count=original_cpu_count
    compile_seconds=time.perf_counter()-start
    start=time.perf_counter();model.load(num_recording_timesteps=steps)
    initialization_seconds=time.perf_counter()-start
    start=time.perf_counter()
    for _ in range(steps):model.step_time()
    # The loop is asynchronous. Readback synchronizes completion and is included
    # in this measured scope; launch-only wall time is not simulation time.
    model.pull_recording_buffers_from_device()
    pop.vars["v"].pull_from_device()
    simulation_readback_seconds=time.perf_counter()-start
    times,indices=pop.spike_recording_data[0]
    result=canonical_result(pop.vars["v"].current_values,times,indices,.0625)
    timing=dict(compile_seconds=compile_seconds,initialization_seconds=initialization_seconds,
                simulation_and_readback_seconds=simulation_readback_seconds)
    if profile:timing["neuron_kernel_seconds"]=model.neuron_update_time
    model.unload()
    return result,timing


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend",choices=("oracle","rust","cpu-f32","metal","cuda","brian2cuda","brian2genn","genn"),required=True)
    parser.add_argument("--neurons",type=int,default=1024)
    parser.add_argument("--steps",type=int,default=128)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--profile",action="store_true")
    parser.add_argument("--case",choices=("independent-if-v0","recurrent-cuba-v0","hh-ionic-v0"),default="independent-if-v0")
    parser.add_argument("--degree",type=int,default=32)
    parser.add_argument("--genn-recurrent-policy",choices=("legacy","brian-euler","projection-inputs","brian-euler-projection-inputs"),default="legacy")
    args=parser.parse_args()
    if not 1<=args.neurons<=1000000 or not 1<=args.steps<=100000:
        parser.error("neurons must be 1..1000000 and steps 1..100000")
    if args.case=='recurrent-cuba-v0':
        import gpu_recurrent as recurrent
        try:recurrent.validate_size(args.neurons,args.degree)
        except ValueError as error:parser.error(str(error))
    if args.genn_recurrent_policy!='legacy' and (args.backend!='genn' or args.case!='recurrent-cuba-v0'):
        parser.error('GeNN recurrent policy requires backend=genn and case=recurrent-cuba-v0')
    args.output.mkdir(parents=True,exist_ok=False)
    config=dict(case="independent-if-v0",neurons=args.neurons,steps=args.steps,dt_ms=.0625,
                equation="dv/dt=drive/ms",threshold="v>1",reset="v=0",method="euler",
                state_dtype="float32",recording="all spikes and final v",profile=args.profile)
    initial,drive=initial_arrays(args.neurons)
    topology_bytes=b''
    if args.case=='recurrent-cuba-v0':
        import gpu_recurrent as recurrent
        config=recurrent.configuration(args.neurons,args.steps,args.degree);config['profile']=args.profile
        initial,drive,projections=recurrent.arrays(args.neurons,args.degree)
        topology_bytes=b''.join(s.tobytes()+t.tobytes()+w.tobytes() for s,t,w in projections)
    if args.case=='hh-ionic-v0':
        import gpu_hh as hh
        config=hh.configuration(args.neurons,args.steps);config['profile']=args.profile
        fields=hh.arrays(args.neurons)
        initial,drive=fields['v'],fields['I_ext']
        topology_bytes=b''.join(fields[k].tobytes() for k in hh.FIELDS if k!='v')
    fingerprint=hashlib.sha256(json.dumps(config,sort_keys=True).encode()+initial.tobytes()+drive.tobytes()+topology_bytes).hexdigest()
    report=dict(schema="b2-gpu-baseline-v0",backend=args.backend,config=config,model_sha256=fingerprint,
                arithmetic="reference-f64" if args.backend=="rust" or (args.backend=="oracle" and args.case!="independent-if-v0") else "float32",
                host=platform.platform(),status="running")
    if args.case=='hh-ionic-v0' and args.backend in {'brian2cuda','brian2genn','genn'}:
        report['arithmetic']='float32 state storage; native backend expression precision'
    if args.backend=='genn' and args.case=='recurrent-cuba-v0':report['genn_recurrent_policy']=args.genn_recurrent_policy
    report["compiler_environment"]={name:os.environ.get(name,"") for name in ("NVCC_PREPEND_FLAGS","NVCC_APPEND_FLAGS")}
    report["versions"]={}
    for name in ("numpy","Brian2","Brian2CUDA","Brian2GeNN","pygenn"):
        try:report["versions"][name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:pass
    try:
        report["nvidia_smi"]=subprocess.check_output(["nvidia-smi","--query-gpu=name,uuid,driver_version","--format=csv,noheader"],text=True)
    except FileNotFoundError:report["nvidia_smi"]=None
    start=time.perf_counter()
    try:
        if args.case=='hh-ionic-v0':
            if args.backend=='oracle':result,timing=hh.oracle(args.neurons,args.steps),{}
            elif args.backend=='genn':result,timing=hh.genn_run(args.neurons,args.steps,args.output,args.profile)
            else:result,timing=hh.brian_run(args.backend,args.neurons,args.steps,args.output,args.profile)
        elif args.case=='recurrent-cuba-v0':
            if args.backend=='oracle':result,timing=recurrent.oracle(args.neurons,args.steps,args.degree),{}
            elif args.backend=='genn':result,timing=recurrent.genn_run(args.neurons,args.steps,args.degree,args.output,args.profile,policy=args.genn_recurrent_policy)
            else:result,timing=recurrent.brian_run(args.backend,args.neurons,args.steps,args.degree,args.output,args.profile)
        elif args.backend=="oracle":result,timing=oracle(args.neurons,args.steps),{}
        elif args.backend=="genn":result,timing=genn_run(args.neurons,args.steps,args.output,args.profile)
        else:result,timing=brian_run(args.backend,args.neurons,args.steps,args.output,args.profile)
    except Exception as error:
        report.update(status="execution_error",error_type=type(error).__name__,error=str(error),
                      total_backend_wall_seconds=time.perf_counter()-start)
        (args.output/"report.json").write_text(json.dumps(report,indent=2)+"\n")
        raise
    report.update(total_backend_wall_seconds=time.perf_counter()-start,timing=timing,
        total_backend_wall_scope="adapter entry through completed result arrays, including imports, model construction, compilation, initialization, simulation and readback; excludes oracle, validation, serialization and process startup")
    expected=(hh.oracle(args.neurons,args.steps) if args.case=='hh-ionic-v0' else recurrent.oracle(args.neurons,args.steps,args.degree) if args.case=='recurrent-cuba-v0'
              else oracle(args.neurons,args.steps))
    report["checks"]={key:bool(np.array_equal(result[key],value)) for key,value in expected.items()}
    report['exact_checks']=dict(report['checks'])
    if args.case=='recurrent-cuba-v0':
        report['checks']['v']=bool(np.allclose(result['v'],expected['v'],rtol=config['acceptance']['v_rtol'],atol=config['acceptance']['v_atol']))
        report['checks'].pop('synaptic_current')
        report['diagnostic_scope']='raw synaptic current retained; GeNN applies pending input at next neuron update, so its readback phase differs'
    if args.case=='hh-ionic-v0':
        report['checks'],report['acceptance_details']=hh.checks(result,expected,config)
    report["result_arrays"]={key:dict(shape=list(value.shape),dtype=str(value.dtype),
                                    sha256=hashlib.sha256(value.tobytes()).hexdigest())
                             for key,value in result.items()}
    report["differences"]={}
    for key,value in expected.items():
        actual=result[key]
        if actual.shape!=value.shape:
            report["differences"][key]=dict(actual_shape=list(actual.shape),expected_shape=list(value.shape))
        elif not np.array_equal(actual,value):
            indices=np.flatnonzero(actual!=value)
            report["differences"][key]=dict(count=int(len(indices)),
                max_abs=float(np.max(np.abs(actual.astype(np.float64)-value.astype(np.float64)))),
                first_indices=indices[:8].tolist(),actual=actual.reshape(-1)[indices[:8]].tolist(),expected=value.reshape(-1)[indices[:8]].tolist())
    report["status"]="passed" if all(report["checks"].values()) else "correctness_failed"
    np.savez_compressed(args.output/"result.npz",**result)
    (args.output/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps(report,indent=2))
    if report["status"]!="passed":raise SystemExit(1)


if __name__=="__main__":
    main()
