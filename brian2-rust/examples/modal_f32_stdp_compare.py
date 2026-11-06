"""Prospective STDP scenarios under the explicitly selected float32 contract."""
import argparse,hashlib,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


CASES={
    'long-4096':dict(neurons=4096,degree=8,steps=4096,workload=dict(drive=10/256,delay_span=16,post_delay=16,topology_kind='random-fixed-outdegree',topology_seed=42)),
}
SCENARIOS={
    'long':CASES,
    'population-scale':{'ring-'+str(n):dict(neurons=n,degree=8,steps=256,workload=dict(
        drive=1/16,delay_span=8,post_delay=3,topology_kind='ring',topology_seed=0))
        for n in (4096,16384)},
    'wide':{'wide-4096':dict(neurons=4096,degree=32,steps=1024,workload=dict(
        drive=1/16,delay_span=8,post_delay=3,topology_kind='random-fixed-outdegree',topology_seed=42))},
    'dense':{'dense-1024':dict(neurons=1024,degree=128,steps=1024,workload=dict(
        drive=1/16,delay_span=8,post_delay=3,topology_kind='random-fixed-outdegree',topology_seed=42))},
    'activity':{name:dict(neurons=4096,degree=8,steps=1024,workload=dict(
        drive=drive,delay_span=16,post_delay=16,topology_kind='random-fixed-outdegree',topology_seed=42))
        for name,drive in [('quiet-4096',1/64),('low-4096',17/512)]},
}
NUMERIC_CONTRACT='explicit-f32-v1'
BACKENDS=('rust-f64','cpu-f32','cuda','cuda-prefix','brian2cuda','genn','genn-barrier','genn-barrier-gather','brian2genn-corrected')


def verify(requested_backends=None,scenario='long'):
    import sys
    sys.path.insert(0,'/workspace/brian2-rust/examples')
    from modal_stdp_precompiled import compare
    from gpu_stdp_precompiled import BACKENDS as ordinary,PREFIX_BACKENDS,BITSET_BACKENDS,CORRECTED_GENN_BACKENDS,GATHER_GENN_BACKENDS,F32_GENN_BACKENDS
    if scenario not in SCENARIOS:raise ValueError('Unknown declared STDP scenario')
    cases=SCENARIOS[scenario]
    backends=list(BACKENDS if requested_backends is None else requested_backends)
    if ('cpu-f32' not in backends or len(set(backends))!=len(backends)
            or any(b not in (*ordinary,*PREFIX_BACKENDS,*BITSET_BACKENDS,*CORRECTED_GENN_BACKENDS,*GATHER_GENN_BACKENDS,*F32_GENN_BACKENDS,'metal') for b in backends)):
        raise ValueError('Unique supported backends including cpu-f32 required')
    report=dict(schema='b2-f32-stdp-v1',numeric_contract=NUMERIC_CONTRACT,scenario=scenario,declared_cases=cases,cases={},case_order=list(cases),
        backends=backends,repeats=5,payloads={},passed=False,status='running',
        scope='Declared scenario on one allocation; five randomized rounds; float32 workers require independent and compiled f32 controls, Rust f64 is a separately labeled f64 reference; f64 divergence is diagnostic')
    gpu=None
    for name,case in cases.items():
        print('case',name,'starting',flush=True)
        result=compare(case['neurons'],case['steps'],case['degree'],5,
            requested_backends=backends,workload=dict(case['workload']),continue_on_gate_failure=True,numeric_contract=NUMERIC_CONTRACT)
        if not report['cases']:gpu=result['nvidia_smi']
        elif gpu!=result['nvidia_smi']:raise RuntimeError('GPU identity changed across cases')
        for key,value in result.pop('payloads').items():
            if Path(key).name!=key or key in {'.','..'}:raise ValueError('Invalid payload path')
            report['payloads'][name+'/'+key]=value
        report['cases'][name]=result
        print('case',name,result['status'],flush=True)
    statuses={c['status'] for c in report['cases'].values()}
    report['passed']=statuses=={'passed-matched-numeric-gates'}
    report['status']='passed-matched-numeric-gates' if report['passed'] else 'failed' if 'failed' in statuses else 'completed-with-gate-failures'
    report['stdout']=report['status'];report['nvidia_smi']=gpu
    return report


def write_json(path,value):
    temporary=path.with_suffix(path.suffix+'.partial')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def save_result(report,output,metadata):
    # Save the small verdict first. A failed large payload write must never
    # erase the test counts, hardware inventory or the retained call identity.
    for name in report.get('payloads',{}):
        relative=Path(name)
        if relative.is_absolute() or '..' in relative.parts or str(relative) in {'report.json','call.json','source-hashes.json'}:
            raise ValueError('Result payload collides with provenance or escapes output')
    result={k:v for k,v in report.items() if k!='payloads'}
    result.update(metadata)
    result['artifact_status']='pending'
    result['result_artifacts']={name:dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest()) for name,data in report.get('payloads',{}).items()}
    write_json(output/'report.json',result)
    for name,data in report.get('payloads',{}).items():
        relative=Path(name)
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('Invalid result path')
        path=output/relative;path.parent.mkdir(parents=True,exist_ok=True)
        temporary=path.with_suffix(path.suffix+'.partial')
        temporary.write_bytes(data);temporary.replace(path)
    result['artifact_status']='complete'
    write_json(output/'report.json',result)
    print(result['stdout'])
    if not result['passed']:raise SystemExit(1)


def resume_result(output):
    import modal
    checkpoint=json.loads((output/'call.json').read_text())
    manifest=(output/'source-hashes.json').read_bytes()
    if hashlib.sha256(manifest).hexdigest()!=checkpoint['source_manifest_sha256']:
        raise ValueError('Saved call manifest changed')
    # Reading the retained call is the only cloud operation in this path.
    # Never invoke verify, create an App or retry an expired/missing call.
    report=modal.FunctionCall.from_id(checkpoint['function_call_id']).get(timeout=0)
    save_result(report,output,dict(checkpoint,remote_wall_seconds=None,recovered_existing_call=True))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--gpu',choices=('L4','A100-40GB'),default='L4');p.add_argument('--output',type=Path,required=True);p.add_argument('--scenario',choices=tuple(SCENARIOS),default='long');p.add_argument('--include-brian2genn-f32-factor',action='store_true',help='Add the explicitly labeled schedule and decay-factor corrected comparator; retain the existing adapter.');p.add_argument('--include-bitset',action='store_true',help='Add explicit CUDA bitset and prefix+bitset workers; retain existing workers.');p.add_argument('--resume',action='store_true',help='Fetch the saved call again without launching GPU work');a=p.parse_args()
    if a.resume:return resume_result(a.output)
    a.output.mkdir(parents=True,exist_ok=False)
    backends=list(BACKENDS)+(['brian2genn-f32-factor'] if a.include_brian2genn_f32_factor else [])
    if a.include_bitset:backends+=['cuda-bitset','cuda-prefix-bitset']
    write_json(a.output/'declared-protocol.json',dict(schema='b2-f32-stdp-v1',numeric_contract=NUMERIC_CONTRACT,
        scenario=a.scenario,cases=SCENARIOS[a.scenario],backends=backends,repeats=5,order_seed=1729,
        f32_acceptance='compiled CPU f32 and independent f32: floats rtol=2e-5 atol=2e-6; spikes and lastupdate exact',
        f64_reference='Rust f64 requires independent f64; label separately, do not call it matched precision',
        compatibility='retain f64 gate and all spikes; no retrospective changes to preceding f64-compatible experiment',
        timeout_seconds=1200,retries=0))
    import modal
    from modal_cuda_tests import native_image
    names=('gpu_brian2genn_stdp_adapter.py','gpu_baseline.py','gpu_stdp_compare.py','gpu_stdp_precompiled.py','modal_stdp_precompiled.py','modal_f32_stdp_compare.py','gpu_genn_readback.py','gpu_genn_barrier_adapter.py','genn_postsynaptic_barrier.py','modal_cuda_tests.py','install_gpu_baselines.sh')
    hashes={str(p.relative_to(ROOT.parent)):hashlib.sha256(p.read_bytes()).hexdigest()
        for folder in (ROOT/'python',ROOT/'src',ROOT.parent/'brian2') for p in sorted(folder.rglob('*'))
        if p.is_file() and '__pycache__' not in p.parts and p.suffix not in {'.pyc','.so','.dylib'}}
    for path in [*(ROOT/'examples'/n for n in names),ROOT/'Cargo.toml',ROOT/'Cargo.lock',
        *(ROOT.parent/n for n in ('setup.py','pyproject.toml','README.md','LICENSE','AUTHORS','CONTRIBUTORS'))]:
        hashes[str(path.relative_to(ROOT.parent))]=hashlib.sha256(path.read_bytes()).hexdigest()
    manifest=json.dumps(hashes,sort_keys=True,indent=2)+'\n';(a.output/'source-hashes.json').write_text(manifest)
    ignore=['**/__pycache__/**','**/*.pyc','**/*.so','**/*.dylib']
    image=(native_image().apt_install('libffi-dev','pkg-config').pip_install('uv==0.8.22')
        .add_local_file(ROOT/'examples/install_gpu_baselines.sh','/opt/install_gpu_baselines.sh',copy=True)
        .run_commands('sh /opt/install_gpu_baselines.sh')
        .add_local_dir(ROOT/'python','/workspace/brian2-rust/python',ignore=ignore))
    for name in names:
        if name!='install_gpu_baselines.sh':image=image.add_local_file(ROOT/'examples'/name,'/workspace/brian2-rust/examples/'+name)
    image=image.add_local_file(Path(__file__),'/root/modal_f32_stdp_compare.py')
    app=modal.App('brian2-f32-stdp-comparison',include_source=False)
    run=app.function(image=image,gpu=a.gpu,cpu=4,memory=8192,max_containers=1,timeout=1200,retries=0)(verify)
    started=time.perf_counter()
    with modal.enable_output(),app.run():
        call=run.spawn(scenario=a.scenario,requested_backends=backends)
        checkpoint=dict(requested_gpu=a.gpu,function_call_id=call.object_id,
            source_manifest_sha256=hashlib.sha256(manifest.encode()).hexdigest())
        write_json(a.output/'call.json',checkpoint)
        report=call.get()
    save_result(report,a.output,dict(checkpoint,remote_wall_seconds=time.perf_counter()-started,
        recovered_existing_call=False))



if __name__=='__main__':main()
