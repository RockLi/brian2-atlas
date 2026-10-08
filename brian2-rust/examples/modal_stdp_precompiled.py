"""Interleave fully gated delayed-STDP replays across six explicitly labeled precompiled backends."""
import argparse
import hashlib
import json
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[1]
NUMERIC_CONTRACTS=('f64-compatible','explicit-f32-v1')


def qualified(checks,backend,numeric_contract='f64-compatible'):
    if numeric_contract not in NUMERIC_CONTRACTS:
        raise ValueError('Unknown numerical contract')
    if numeric_contract=='f64-compatible' or backend=='rust-f64' or backend.startswith('cpp-f64-'):
        return checks['passed'] and all(checks['original_f64_gate'].values())
    # Explicit f32 qualification requires BOTH the compiled CPU control and
    # the independent f32 recurrence. f64 equivalence remains separately visible.
    return checks['passed'] and checks['independent_f32_gate']['passed']


def retain_gate_failure(report,backend,checks,phase,continue_on_gate_failure):
    report['excluded_by_gate'][backend]=checks
    report.setdefault('exclusion_phase',{})[backend]=phase
    if not continue_on_gate_failure:
        raise RuntimeError(backend+' '+phase+' violates declared numerical gate')


def timing_summary(backends,samples,repeats,excluded,numeric_contract='f64-compatible'):
    import statistics
    summary=[]
    for backend in backends:
        rows=[] if backend in excluded else [r for r in samples if r['backend']==backend and r['round']>=0]
        values=[r['wall_seconds'] for r in rows]
        eligible=len(rows)==repeats and all(qualified(r['checks'],backend,numeric_contract) for r in rows)
        summary.append(dict(backend=backend,samples_seconds=values,
            complete=len(rows)==repeats,matched_gate_passed=eligible,
            median_seconds=statistics.median(values) if values else None,
            min_seconds=min(values) if values else None,max_seconds=max(values) if values else None))
    return summary


def compare(neurons,steps,degree,repeats,requested_backends=None,workload=None,continue_on_gate_failure=False,numeric_contract='f64-compatible'):
    if numeric_contract not in NUMERIC_CONTRACTS:
        raise ValueError('Unknown numerical contract')
    import os,random,select,signal,subprocess,sys,tempfile,statistics
    import numpy as np
    sys.path.insert(0,'/workspace/brian2-rust/examples')
    from gpu_stdp_precompiled import BACKENDS as DEFAULT_BACKENDS,CPP_BACKENDS,PREFIX_BACKENDS,BITSET_BACKENDS,CORRECTED_GENN_BACKENDS,GATHER_GENN_BACKENDS,F32_GENN_BACKENDS
    worker_script=Path(sys.modules['gpu_stdp_precompiled'].__file__).resolve()
    from gpu_stdp_compare import configuration,oracle,activity,checks as stdp_checks
    BACKENDS=tuple(requested_backends) if requested_backends else DEFAULT_BACKENDS
    if ('cpu-f32' not in BACKENDS or len(set(BACKENDS))!=len(BACKENDS)
            or any(b not in (*DEFAULT_BACKENDS,*CPP_BACKENDS,*PREFIX_BACKENDS,*BITSET_BACKENDS,*CORRECTED_GENN_BACKENDS,*GATHER_GENN_BACKENDS,*F32_GENN_BACKENDS,'metal') for b in BACKENDS)):
        raise ValueError('Unique supported comparison backends including cpu-f32 are required')
    PRECISE_BASIC_FLAGS='--fmad=false --ftz=false --prec-div=true --prec-sqrt=true'
    workload=workload or {}
    config=configuration(neurons,degree,steps,**workload)
    fingerprint=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    report=dict(schema='b2-stdp-precompiled-v0',configuration=config,declared_input_sha256=fingerprint,
        repeats=repeats,order_seed=1729,backends=list(BACKENDS),excluded_backends={'brian2genn':'Original/grouped full delayed-STDP adapter fails semantic gates; see stdp-comparison evidence.'},workers={},samples=[],payloads={},status='running',excluded_by_gate={},
        scope='precompiled reset/initialization through completed host result arrays; includes process launch/file IO where required and GeNN unload; excludes compilation, frontend/model construction, protocol, hashes and NPZ serialization',
        nvidia_smi=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version','--format=csv,noheader'],text=True) if __import__('shutil').which('nvidia-smi') else None,
        host_cpu=Path('/proc/cpuinfo').read_text() if Path('/proc/cpuinfo').exists() else __import__('platform').platform(),allocated_cpu_cores=4 if Path('/proc/cpuinfo').exists() else None,
        locks={p.name:p.read_text() for p in Path('/opt/baseline-locks').glob('*.txt')})
    report['numeric_contract']=numeric_contract
    report['precision_by_backend']={b:'float64' if b=='rust-f64' or b.startswith('cpp-f64-') else 'float32' for b in BACKENDS}
    interpreters={b:sys.executable for b in BACKENDS}
    interpreters['brian2genn-corrected']='/opt/brian2genn-env/bin/python'
    interpreters.update({b:'/opt/brian2genn-env/bin/python' for b in F32_GENN_BACKENDS})
    interpreters['genn-barrier']='/opt/genn5-env/bin/python'
    interpreters['genn-barrier-gather']='/opt/genn5-env/bin/python'
    interpreters.update(brian2cuda='/opt/brian2cuda-env/bin/python',brian2genn='/opt/brian2genn-env/bin/python',genn='/opt/genn5-env/bin/python')
    workers={};logs={};rng=random.Random(1729)
    def receive(backend,timeout=300):
        process=workers[backend]
        if not select.select([process.stdout],[],[],timeout)[0]:raise TimeoutError(backend+' worker response timeout')
        line=process.stdout.readline()
        if not line:raise RuntimeError(backend+' worker exited before response')
        value=json.loads(line)
        if value['event']=='error':raise RuntimeError(backend+': '+value['error_type']+': '+value['error'])
        return value
    def request(backend,message):
        p=workers[backend];p.stdin.write(json.dumps(message)+'\n');p.stdin.flush();return receive(backend)
    def capture(artifact,name):
        payload=Path(artifact['path']).read_bytes();assert hashlib.sha256(payload).hexdigest()==artifact['sha256']
        result=dict(np.load(artifact['path']));report['payloads'][name]=payload
        artifact['path']=name
        return result
    reference=None;f64_reference=oracle(neurons,degree,steps,**workload)
    f32_reference=oracle(neurons,degree,steps,dtype=np.float32,**workload)
    # Retain the actual in-process references, including platform-specific last
    # bits, so future audits can reproduce every reported diagnostic exactly.
    import io
    report['references']={}
    for precision,values in [('f64',f64_reference),('f32',f32_reference)]:
        stream=io.BytesIO();np.savez_compressed(stream,**values);payload=stream.getvalue()
        name='reference-'+precision+'.npz';report['payloads'][name]=payload
        report['references'][precision]=dict(path=name,sha256=hashlib.sha256(payload).hexdigest(),
            arrays={k:dict(shape=list(v.shape),dtype=str(v.dtype),sha256=hashlib.sha256(v.tobytes()).hexdigest()) for k,v in values.items()})
    report['oracle_activity']=activity(neurons,degree,steps,f64_reference,**workload)
    def checks(backend,actual):
        f64_control=backend=='rust-f64' or (numeric_contract=='explicit-f32-v1' and backend.startswith('cpp-f64-'))
        control=f64_reference if f64_control else reference
        result=stdp_checks(actual,control)
        result['control']='independent-f64' if f64_control else 'compiled-cpu-f32'
        result['original_f64_gate']={k:v['passed'] for k,v in stdp_checks(actual,f64_reference)['fields'].items()}
        # Diagnostic in the default contract; independently required in the
        # explicitly selected f32 contract. CPU f32 self-equality is insufficient.
        result['independent_f32_gate']=stdp_checks(actual,f32_reference)
        return result
    with tempfile.TemporaryDirectory(prefix='b2-stdp-precompiled-') as temporary:
        try:
            # Prepare CPU f32 first to establish the numerical gate. All six
            # workers stay alive; no GPU simulation runs concurrently.
            preparation_order=['cpu-f32']+[b for b in BACKENDS if b!='cpu-f32']
            for backend in preparation_order:
                output=Path(temporary)/backend;log=Path(temporary)/(backend+'.log');logs[backend]=log
                env={**os.environ,'CUDA_PATH':'/usr/local/cuda','NVCC_APPEND_FLAGS':PRECISE_BASIC_FLAGS}
                env.pop('NVCC_PREPEND_FLAGS',None)
                if backend in {'genn','genn-barrier','genn-barrier-gather','brian2cuda','brian2genn','brian2genn-corrected',*F32_GENN_BACKENDS}:env['PYTHONPATH']=''
                if backend in {'brian2genn','brian2genn-corrected',*F32_GENN_BACKENDS}:env['PATH']='/opt/genn4/bin:'+env['PATH']
                with log.open('w') as stderr:
                    workers[backend]=subprocess.Popen([interpreters[backend],str(worker_script),
                        '--backend',backend,'--neurons',str(neurons),'--steps',str(steps),'--degree',str(degree),'--output',str(output),
                        *[item for key,value in workload.items() for item in ('--'+key.replace('_','-'),str(value))]],
                        cwd=temporary,env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=stderr,text=True,start_new_session=True)
                ready=receive(backend);assert ready['event']=='ready' and ready['configuration']==config
                actual=capture(ready['bootstrap'],backend+'-bootstrap.npz')
                if backend=='cpu-f32':reference=actual
                ready['checks']=checks(backend,actual);ready['observed_activity']=activity(neurons,degree,steps,actual,**workload);report['workers'][backend]=ready
                ready['timing_eligible']=qualified(ready['checks'],backend,numeric_contract)
                if not ready['timing_eligible']:
                    retain_gate_failure(report,backend,ready['checks'],'bootstrap',continue_on_gate_failure)
                print('prepared',backend,flush=True)
            # One explicit unmeasured-for-summary warmup per backend, then
            # randomized paired rounds on the same physical GPU allocation.
            for round_index in range(-1,repeats):
                order=[b for b in BACKENDS if b not in report['excluded_by_gate']];rng.shuffle(order)
                for position,backend in enumerate(order):
                    sample=request(backend,dict(command='run',sample=round_index))
                    assert sample['event']=='result' and sample['sample']==round_index
                    actual=capture(sample['result'],f'{backend}-{round_index}.npz')
                    sample.update(backend=backend,round=round_index,order=position,checks=checks(backend,actual))
                    report['samples'].append(sample)
                    if not qualified(sample['checks'],backend,numeric_contract):
                        report['workers'][backend]['timing_eligible']=False
                        retain_gate_failure(report,backend,sample['checks'],'replay',continue_on_gate_failure)
                print('round',round_index,'complete',flush=True)
            report['status']='completed-with-gate-failures' if report['excluded_by_gate'] else 'passed-matched-numeric-gates'
        except BaseException as error:
            report.update(status='failed',error_type=type(error).__name__,error=str(error))
        finally:
            for backend,p in workers.items():
                try:
                    if p.poll() is None:
                        p.stdin.write('{"command":"close"}\n');p.stdin.flush()
                        reply=receive(backend,30);assert reply['event']=='closed'
                    code=p.wait(timeout=10)
                    report.setdefault('worker_exit_codes',{})[backend]=code
                except BaseException as error:
                    if p.poll() is None:os.killpg(p.pid,signal.SIGKILL)
                    p.wait(timeout=10);report.setdefault('shutdown_errors',{})[backend]=str(error)
            for backend,log in logs.items():report.setdefault('worker_logs',{})[backend]=log.read_text()
    if report.get('shutdown_errors') or any(code!=0 for code in report.get('worker_exit_codes',{}).values()):report['status']='failed'
    report['original_f64_gate_passed_all_samples']=bool(report['samples']) and all(all(r['checks']['original_f64_gate'].values()) for r in report['samples'])
    report['summary']=timing_summary(BACKENDS,report['samples'],repeats,report['excluded_by_gate'],numeric_contract)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu',choices=('L4','A100-40GB'),default='L4')
    parser.add_argument('--neurons',type=int,default=256);parser.add_argument('--steps',type=int,default=128)
    parser.add_argument('--degree',type=int,default=128);parser.add_argument('--repeats',type=int,default=5)
    parser.add_argument('--drive',type=float,default=.125);parser.add_argument('--delay-span',type=int,default=4);parser.add_argument('--post-delay',type=int,default=2)
    parser.add_argument('--topology-kind',choices=('ring','random-fixed-outdegree'),default='ring');parser.add_argument('--topology-seed',type=int,default=0)
    parser.add_argument('--continue-on-gate-failure',action='store_true',help='Capture every bootstrap; exclude failed backends from timed replays without relaxing gates.')
    parser.add_argument('--numeric-contract',choices=NUMERIC_CONTRACTS,default='f64-compatible',help='Explicit f32 uses independent and compiled f32 controls; f64 divergence remains reported.')
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    from gpu_stdp_compare import workload_options
    workload=workload_options(args.drive,args.delay_span,args.post_delay,args.topology_kind,args.topology_seed)
    from gpu_stdp_compare import topology,MAX_NEURONS
    from gpu_stdp_precompiled import MAX_REPLAY_STEPS
    if not 2<=args.neurons<=MAX_NEURONS or not 1<=args.steps<=MAX_REPLAY_STEPS or not 1<=args.repeats<=10:parser.error('bounded workload, 1..4096 steps and 1..10 repeats required')
    try:topology(args.neurons,args.degree,delay_span=args.delay_span,topology_kind=args.topology_kind,topology_seed=args.topology_seed)
    except ValueError as error:parser.error(str(error))
    args.output.mkdir(parents=True,exist_ok=False)
    import modal
    from modal_cuda_tests import native_image
    names=('gpu_brian2genn_stdp_adapter.py','gpu_baseline.py','gpu_stdp_compare.py','gpu_stdp_precompiled.py','modal_stdp_precompiled.py','modal_cuda_tests.py','install_gpu_baselines.sh')
    hashes={str(p.relative_to(ROOT.parent)):hashlib.sha256(p.read_bytes()).hexdigest()
        for folder in (ROOT/'python',ROOT/'src',ROOT.parent/'brian2') for p in sorted(folder.rglob('*'))
        if p.is_file() and '__pycache__' not in p.parts and p.suffix not in {'.pyc','.so','.dylib'}}
    for p in [*(ROOT/'examples'/n for n in names),ROOT/'Cargo.toml',ROOT/'Cargo.lock',
              *(ROOT.parent/n for n in ('setup.py','pyproject.toml','README.md','LICENSE','AUTHORS','CONTRIBUTORS'))]:
        hashes[str(p.relative_to(ROOT.parent))]=hashlib.sha256(p.read_bytes()).hexdigest()
    (args.output/'source-hashes.json').write_text(json.dumps(hashes,sort_keys=True,indent=2)+'\n')
    image=(native_image().apt_install('libffi-dev','pkg-config').pip_install('uv==0.8.22')
        .add_local_file(ROOT/'examples/install_gpu_baselines.sh','/opt/install_gpu_baselines.sh',copy=True)
        .run_commands('sh /opt/install_gpu_baselines.sh')
        .add_local_dir(ROOT/'python','/workspace/brian2-rust/python',ignore=['**/__pycache__/**','**/*.pyc']))
    for n in names:
        if n!='install_gpu_baselines.sh':image=image.add_local_file(ROOT/'examples'/n,'/workspace/brian2-rust/examples/'+n)
    image=image.add_local_file(Path(__file__),'/root/modal_stdp_precompiled.py')
    app=modal.App('brian2-stdp-precompiled',include_source=False)
    run=app.function(image=image,gpu=args.gpu,cpu=4,memory=8192,max_containers=1,timeout=1200,retries=0)(compare)
    start=time.perf_counter()
    with modal.enable_output(),app.run():report=run.remote(args.neurons,args.steps,args.degree,args.repeats,workload=workload,continue_on_gate_failure=args.continue_on_gate_failure,numeric_contract=args.numeric_contract)
    report.update(requested_gpu=args.gpu,remote_wall_seconds=time.perf_counter()-start,
        source_manifest_sha256=hashlib.sha256((args.output/'source-hashes.json').read_bytes()).hexdigest())
    for name,payload in report.pop('payloads').items():(args.output/name).write_bytes(payload)
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(report['status'],report['summary'])
    if report['status']!='passed-matched-numeric-gates':raise SystemExit(1)


if __name__=='__main__':main()
