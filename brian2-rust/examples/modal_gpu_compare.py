"""Compare isolated backend processes on the same allocated NVIDIA GPU."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import time
import random
import statistics

ROOT=Path(__file__).resolve().parents[1]
BACKENDS=("rust","cpu-f32","cuda","brian2cuda","brian2genn","genn")
PRECISE_BASIC_FLAGS="--fmad=false --ftz=false --prec-div=true --prec-sqrt=true"



def completed_trial(row):
    report=row.get('report')
    return (row.get('exit_code') in (0,1) and report is not None and
            report.get('status') in {'passed','correctness_failed'} and 'result_npz' in row)


def paired_checks(actual,expected,case,neurons,steps):
    """Keep f32 comparison separate from the unchanged reference-f64 gate."""
    import numpy as np
    exact={k:k in actual and actual[k].dtype==v.dtype and actual[k].shape==v.shape and
           actual[k].tobytes()==v.tobytes() for k,v in expected.items()}
    checks={k:k in actual and np.array_equal(actual[k],v) for k,v in expected.items()}
    if case=='recurrent-cuba-v0':
        from gpu_recurrent import configuration
        tolerance=configuration(neurons,steps,1)['acceptance']
        checks['v']=bool(actual['v'].shape==expected['v'].shape and np.allclose(actual['v'],expected['v'],
            rtol=tolerance['v_rtol'],atol=tolerance['v_atol']))
        checks.pop('synaptic_current',None)  # GeNN's pending current has a different readback phase.
    elif case=='hh-ionic-v0':
        from gpu_hh import checks as hh_checks,configuration
        checks,_=hh_checks(actual,expected,configuration(neurons,steps))
    return dict(checks=checks,passed=all(checks.values()),bitwise=exact,
        max_abs={k:float(np.max(np.abs(actual[k].astype(np.float64)-v.astype(np.float64)),initial=0))
                 for k,v in expected.items() if k in actual and actual[k].shape==v.shape})


def summarize_trials(report):
    """Same-scope wall samples and distinct acceptance gates; never mix kernel timers."""
    summaries=[]
    for backend in BACKENDS:
        rows=[r for r in report['trials'] if r['backend']==backend]
        complete=len(rows)==report['requested_repeats'] and all(completed_trial(r) for r in rows)
        pairs=[r for r in report['pairwise_against_cpu_f32'] if r['backend']==backend]
        f32_ok=len(pairs)==report['requested_repeats'] and all(r['passed'] for r in pairs)
        f64_ok=complete and all(all(r['report']['checks'].values()) for r in rows)
        values=[r['report']['total_backend_wall_seconds'] for r in rows if completed_trial(r)]
        identities={json.dumps(r['report']['result_arrays'],sort_keys=True) for r in rows if completed_trial(r)}
        summaries.append(dict(backend=backend,completed_all_trials=complete,
            unchanged_reference_gate_passed=f64_ok,matched_cpu_f32_gate_passed=f32_ok,
            repeat_output_hashes_equal=complete and len(identities)==1,
            # Acceptance is per trial; byte variation is diagnostic when all
            # samples remain inside the declared state/spike tolerance.
            timing_eligible=complete and (f64_ok if backend=='rust' else f32_ok),
            samples_seconds=values,median_seconds=statistics.median(values) if values else None,
            min_seconds=min(values) if values else None,max_seconds=max(values) if values else None))
    return summaries

def compare(neurons,steps,repeats,math_policy,case,degree,genn_policy='legacy'):
    import os
    import signal
    import subprocess
    import sys
    import tempfile
    sys.path.insert(0,"/workspace/brian2-rust/examples")
    interpreters=dict(rust=sys.executable,cuda=sys.executable,**{"cpu-f32":sys.executable},
                      brian2cuda="/opt/brian2cuda-env/bin/python",
                      brian2genn="/opt/brian2genn-env/bin/python",genn="/opt/genn5-env/bin/python")
    report=dict(schema="b2-modal-comparison-v1",status="running",neurons=neurons,steps=steps,
                math_policy=math_policy,case=case,degree=degree,requested_repeats=repeats,order_seed=1729,
                genn_recurrent_policy=genn_policy,
                timing_scope="fresh process/project per trial; total_backend_wall_seconds spans adapter entry through completed arrays, including compilation/initialization/readback, excluding process startup and correctness checks; no warm simulation speedup claim",
                host_cpu=Path('/proc/cpuinfo').read_text(),cpu_affinity=sorted(os.sched_getaffinity(0)),allocated_cpu_cores=4,
                process_wall_scope="also includes interpreter startup, oracle, checks and output serialization; diagnostic only",
                nvidia_smi=subprocess.check_output(["nvidia-smi","--query-gpu=name,uuid,driver_version","--format=csv,noheader"],text=True),
                locks={p.name:p.read_text() for p in Path("/opt/baseline-locks").glob("*.txt")},trials=[])
    backends=BACKENDS
    rng=random.Random(1729)
    with tempfile.TemporaryDirectory(prefix="b2-compare-") as temporary:
        for repeat in range(repeats):
            order=list(backends);rng.shuffle(order)
            for position,backend in enumerate(order):
                output=Path(temporary)/f"{repeat}-{backend}"
                env={**os.environ,"CUDA_PATH":"/usr/local/cuda"}
                # Explicit, recorded basic arithmetic control. This does not
                # undo fast transcendental substitutions in general models.
                env.pop("NVCC_PREPEND_FLAGS",None)
                env.pop("NVCC_APPEND_FLAGS",None)
                if math_policy=="precise-basic":env["NVCC_APPEND_FLAGS"]=PRECISE_BASIC_FLAGS
                if backend not in {"rust","cuda","cpu-f32"}:env["PYTHONPATH"]=""
                if backend=="brian2genn":
                    env["PATH"]="/opt/genn4/bin:"+env["PATH"]
                start=time.perf_counter()
                try:
                    run=subprocess.Popen([interpreters[backend],"/workspace/brian2-rust/examples/gpu_baseline.py",
                                        "--backend",backend,"--neurons",str(neurons),"--steps",str(steps),
                                        "--case",case,"--degree",str(degree),
                                        "--genn-recurrent-policy",genn_policy if backend=="genn" else "legacy",
                                        "--output",str(output)],cwd=temporary,env=env,
                                         stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,start_new_session=True)
                    stdout,stderr=run.communicate(timeout=300)
                    row=dict(backend=backend,repeat=repeat,exit_code=run.returncode,stdout=stdout,stderr=stderr)
                except subprocess.TimeoutExpired:
                    os.killpg(run.pid,signal.SIGKILL)
                    stdout,stderr=run.communicate()
                    row=dict(backend=backend,repeat=repeat,exit_code=None,error="300 second process timeout",
                             stdout=stdout,stderr=stderr)
                row["order"]=position
                row["process_wall_seconds"]=time.perf_counter()-start
                row["report"]=json.loads((output/"report.json").read_text()) if (output/"report.json").exists() else None
                if (output/"result.npz").exists():row["result_npz"]=(output/"result.npz").read_bytes()
                row["generated_build_files"]={str(p.relative_to(output)):p.read_text()
                    for p in output.rglob("*") if p.is_file() and p.name.lower()=="makefile" and p.stat().st_size<100000}
                row["generated_neuron_sources"]={str(p.relative_to(output)):p.read_text()
                    for p in output.rglob("*") if p.is_file() and
                    (p.name in {"neuronUpdate.cc","neuronUpdate.cu","magicnetwork_model.cpp"} or p.name.endswith("_stateupdater_codeobject.cu")) and p.stat().st_size<200000}
                report["trials"].append(row)
                print(backend,repeat,row["exit_code"],flush=True)
            # Finish one diagnostic round so a failed adapter cannot hide others.
            if any(not completed_trial(t) for t in report["trials"]):break
    import numpy as np
    report['pairwise_against_cpu_f32']=[]
    for repeat in range(repeats):
        rows=[r for r in report['trials'] if r['repeat']==repeat and completed_trial(r)]
        control=next((r for r in rows if r['backend']=='cpu-f32'),None)
        if control is None:continue
        with np.load(io.BytesIO(control['result_npz'])) as data:expected=dict(data)
        for row in rows:
            with np.load(io.BytesIO(row['result_npz'])) as data:actual=dict(data)
            report['pairwise_against_cpu_f32'].append(dict(backend=row['backend'],repeat=repeat,
                **paired_checks(actual,expected,case,neurons,steps)))
    report['summary']=summarize_trials(report)
    report['execution_completed']=len(report['trials'])==repeats*len(backends) and all(completed_trial(r) for r in report['trials'])
    reports=[t["report"] for t in report["trials"] if t["report"]]
    report["identity_checks"]=dict(
        all_reports_present=len(reports)==len(report["trials"]),
        same_model=len({r["model_sha256"] for r in reports})==1,
        same_gpu={r["nvidia_smi"] for r in reports}=={report["nvidia_smi"]})
    if not all(report['identity_checks'].values()):
        for summary in report['summary']:summary['timing_eligible']=False
    report["status"]="passed" if all(t["exit_code"]==0 and t["report"] and t["report"]["status"]=="passed"
                                    for t in report["trials"]) else "failed"
    if not all(report["identity_checks"].values()):report["status"]="identity_mismatch"
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--gpu",choices=("L4","A100-40GB","H100!"),default="L4")
    parser.add_argument("--neurons",type=int,default=32)
    parser.add_argument("--steps",type=int,default=32)
    parser.add_argument("--repeats",type=int,default=1)
    parser.add_argument("--math-policy",choices=("native-default","precise-basic"),default="native-default")
    parser.add_argument("--case",choices=("independent-if-v0","recurrent-cuba-v0","hh-ionic-v0"),default="independent-if-v0")
    parser.add_argument("--degree",type=int,default=32)
    parser.add_argument("--genn-recurrent-policy",choices=("legacy","brian-euler","projection-inputs","brian-euler-projection-inputs"),default="legacy")
    args=parser.parse_args()
    if not 1<=args.repeats<=5 or not 1<=args.neurons<=65536 or not 1<=args.steps<=4096:
        parser.error("bounded positive workload and 1..5 repeats required")
    if args.case=='recurrent-cuba-v0':
        from gpu_recurrent import validate_size
        try:validate_size(args.neurons,args.degree)
        except ValueError as error:parser.error(str(error))
    if args.genn_recurrent_policy!='legacy' and args.case!='recurrent-cuba-v0':
        parser.error('GeNN recurrent policy requires case=recurrent-cuba-v0')
    args.output.mkdir(parents=True,exist_ok=False)
    import modal
    from modal_cuda_tests import native_image
    sources={}
    for folder in (ROOT/"python",ROOT.parent/"brian2",ROOT/"src"):
        for p in sorted(folder.rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts and p.suffix not in {".pyc",".so",".dylib"}:
                sources[str(p.relative_to(ROOT.parent))]=hashlib.sha256(p.read_bytes()).hexdigest()
    for name in ("modal_gpu_compare.py","modal_cuda_tests.py","gpu_baseline.py","gpu_recurrent.py","gpu_hh.py","install_gpu_baselines.sh"):
        p=ROOT/"examples"/name;sources[str(p.relative_to(ROOT.parent))]=hashlib.sha256(p.read_bytes()).hexdigest()
    for p in (ROOT/"Cargo.toml",ROOT/"Cargo.lock",*(ROOT.parent/name for name in
               ("setup.py","pyproject.toml","README.md","LICENSE","AUTHORS","CONTRIBUTORS"))):
        sources[str(p.relative_to(ROOT.parent))]=hashlib.sha256(p.read_bytes()).hexdigest()
    (args.output/"source-hashes.json").write_text(json.dumps(sources,sort_keys=True,indent=2)+"\n")
    image=(native_image().apt_install("libffi-dev","pkg-config")
           .pip_install("uv==0.8.22")
           .add_local_file(ROOT/"examples/install_gpu_baselines.sh","/opt/install_gpu_baselines.sh",copy=True)
           .run_commands("sh /opt/install_gpu_baselines.sh")
           .add_local_dir(ROOT/"python","/workspace/brian2-rust/python",ignore=["**/__pycache__/**","**/*.pyc"])
           .add_local_file(ROOT/"examples/gpu_baseline.py","/workspace/brian2-rust/examples/gpu_baseline.py")
           .add_local_file(ROOT/"examples/gpu_recurrent.py","/workspace/brian2-rust/examples/gpu_recurrent.py")
           .add_local_file(ROOT/"examples/gpu_hh.py","/workspace/brian2-rust/examples/gpu_hh.py")
           .add_local_file(Path(__file__),"/root/modal_gpu_compare.py"))
    app=modal.App("brian2-gpu-backend-comparison",include_source=False)
    run=app.function(image=image,gpu=args.gpu,cpu=4,memory=8192,max_containers=1,timeout=1800,retries=0)(compare)
    started=time.perf_counter()
    with modal.enable_output(),app.run():report=run.remote(args.neurons,args.steps,args.repeats,args.math_policy,args.case,args.degree,args.genn_recurrent_policy)
    report.update(requested_gpu=args.gpu,remote_wall_seconds=time.perf_counter()-started,
                  source_manifest_sha256=hashlib.sha256((args.output/"source-hashes.json").read_bytes()).hexdigest())
    for row in report['trials']:
        payload=row.pop('result_npz',None)
        if payload is not None:
            name=f"{row['repeat']}-{row['backend']}.npz"
            (args.output/name).write_bytes(payload)
            row['result_artifact']=dict(path=name,sha256=hashlib.sha256(payload).hexdigest(),bytes=len(payload))
    (args.output/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(report["status"],[(t["backend"],t["exit_code"]) for t in report["trials"]])
    if report["status"]!="passed":raise SystemExit(1)


if __name__=="__main__":main()
