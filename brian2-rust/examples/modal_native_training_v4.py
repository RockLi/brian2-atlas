"""One bounded L4 acceptance run for native CUDA MPI and vector-state BPTT.

Run only after authorization for a new cloud run. The previous L4 acceptance
was a different, completed run. No automatic retries or deployment are used.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[1]
TESTS=['test_native_training_cuda.py','test_native_training_equations.py',
       'test_native_training_gpu_mpi.py','test_native_training_multistate.py',
       'test_training_brian_multistate.py','test_native_training_multistate_gpu.py',
       'test_training_brian_integrators.py','test_training_refractory.py',
       'test_training_rk_refractory.py','test_training_heterogeneous.py','test_training_time.py','test_training_stochastic.py']
DEPENDENCIES=['test_native_training.py','test_native_training_graph.py','test_training_brian.py']


def claim_attempt(guard):
    """Atomically refuse a second test execution, including platform replays."""
    if guard is None:raise ValueError('a cross-container attempt guard is required')
    if not guard.put('attempt',dict(started_unix=time.time()),skip_if_exists=True):
        raise RuntimeError('test execution already claimed; refusing platform replay after interruption')


def execute_tests(command,env,timeout=720):
    """Stream per-test evidence to platform logs while retaining the full report."""
    import os
    import signal
    import subprocess
    import threading
    lines=[]
    with subprocess.Popen(command,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                          text=True,start_new_session=True) as process:
        def capture():
            for line in process.stdout:
                lines.append(line);print(line,end='',flush=True)
        reader=threading.Thread(target=capture,daemon=True);reader.start()
        timed_out=False
        try:process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out=True;os.killpg(process.pid,signal.SIGKILL);process.wait()
        reader.join(timeout=10)
        if reader.is_alive():raise RuntimeError('test log pipe did not close after process termination')
    return dict(passed=process.returncode==0 and not timed_out,exit_code=process.returncode,
                stdout=''.join(lines),stderr='',timeout=timed_out)


def run_tests(test_names,attempt_guard):
    # Modal preemption may replay an input even with retries=0. This claim is
    # made before any test process starts and survives container replacement.
    claim_attempt(attempt_guard)
    import os
    import platform
    import subprocess
    import sys
    os.chdir('/workspace')
    env=dict(os.environ,B2_TRAIN_RUNNER='/workspace/brian2-rust/target/release/b2-train',
             B2_TEST_CUDA_TRAIN='1',B2_TEST_MPI='1',B2_TRAIN_CUDA_DEVICE='0')
    # MPICH uses ordinary host buffers; no CUDA-aware MPI dependency.
    paths=['brian2-rust/tests/'+name for name in (test_names or TESTS)]
    command=[sys.executable,'-u','-m','pytest','-v',*paths,'--junitxml=/tmp/native-v4-cuda.xml']
    started=time.monotonic()
    report=execute_tests(command,env)
    junit=Path('/tmp/native-v4-cuda.xml')
    report.update(schema='b2-cuda-mpi-multistate-acceptance-v1',tests=paths,
        junit=junit.read_text() if junit.exists() else '',test_seconds=time.monotonic()-started,
        gpu=subprocess.check_output(['nvidia-smi'],text=True),compiler=subprocess.check_output(['nvcc','--version'],text=True),
        mpi=subprocess.check_output(['mpiexec','--version'],text=True),python=sys.version,host=platform.platform(),
        native_binary_sha256=hashlib.sha256(Path(env['B2_TRAIN_RUNNER']).read_bytes()).hexdigest(),
        scope='one L4; multiple MPI ranks share logical CUDA device 0; no cross-host or multi-physical-GPU claim',
        mpi_preflight=json.loads(Path('/opt/mpi-preflight.json').read_text()))
    attempt_guard.put('report',report)
    return report


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--suite',choices=['full','refractory','rk-refractory','heterogeneous','time','stochastic'],default='full',
        help='refractory selects the affected v4 GPU, RK frontend and refractory suites within the same time cap')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    import modal
    from modal_cuda_tests import native_image
    files=[p for d in ('src','python') for p in (ROOT/d).rglob('*')
           if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.pyc','.so','.dylib')]
    files += [ROOT/'tests'/name for name in TESTS+DEPENDENCIES]
    files += [Path(__file__),ROOT/'examples/modal_cuda_tests.py',ROOT/'Cargo.toml',ROOT/'Cargo.lock',ROOT/'tools/verify_native_training_mpi_bootstrap.py']
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (args.output/'source-hashes.json').write_text(json.dumps(hashes,sort_keys=True,indent=2)+'\n')
    # Ubuntu 24.04's distro MPICH 4.2.0 can pair a PMIx library with Hydra,
    # producing singleton ranks (pmodels/mpich#7064). Install a matched native
    # distribution and test the exact shim/runner during CPU image construction.
    image=(native_image().pip_install('mpich==5.0.1')
        .add_local_file(ROOT/'tools/verify_native_training_mpi_bootstrap.py','/opt/verify_native_training_mpi_bootstrap.py',copy=True)
        .add_local_file(ROOT/'python/brian2_rust/training_mpi.c','/opt/training_mpi.c',copy=True)
        .run_commands('python /opt/verify_native_training_mpi_bootstrap.py --runner /workspace/brian2-rust/target/release/b2-train --shim-source /opt/training_mpi.c > /opt/mpi-preflight.json')
        .add_local_dir(ROOT/'python','/workspace/brian2-rust/python',ignore=['**/__pycache__/**','**/*.pyc'])
        .add_local_file(Path(__file__),'/root/modal_native_training_v4.py'))
    for name in TESTS+DEPENDENCIES:
        image=image.add_local_file(ROOT/'tests'/name,'/workspace/brian2-rust/tests/'+name)
    app=modal.App('atlas-native-training-v4-acceptance',include_source=False)
    run=app.function(image=image,gpu='L4',cpu=4,memory=8192,max_containers=1,timeout=900,retries=0)(run_tests)
    started=time.monotonic()
    selected=TESTS if args.suite=='full' else ['test_native_training_multistate_gpu.py',
        'test_training_brian_integrators.py','test_training_refractory.py']
    if args.suite=='rk-refractory':selected.append('test_training_rk_refractory.py')
    if args.suite=='heterogeneous':selected=['test_native_training_multistate_gpu.py','test_training_heterogeneous.py']
    if args.suite=='time':selected=['test_native_training_multistate_gpu.py','test_training_time.py']
    if args.suite=='stochastic':selected=['test_native_training_multistate_gpu.py','test_training_time.py','test_training_stochastic.py']
    with modal.enable_output(),modal.Dict.ephemeral() as attempt_guard:
        try:
            with app.run():report=run.remote(selected,attempt_guard)
        except Exception as error:
            # Preserve interruption evidence even when the function cannot
            # return a report. A fully persisted report may survive preemption.
            report=attempt_guard.get('report')
            if report is None:
                report=dict(passed=False,exit_code=None,junit='',stdout='',stderr='',
                    interrupted=True,error_type=type(error).__name__,error=str(error),
                    attempt=attempt_guard.get('attempt'))
    report['elapsed_seconds']=time.monotonic()-started
    report['source_manifest_sha256']=hashlib.sha256((args.output/'source-hashes.json').read_bytes()).hexdigest()
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    (args.output/'junit.xml').write_text(report['junit'])
    print(json.dumps({key:report.get(key) for key in ('passed','exit_code','interrupted','error','test_seconds')}))
    if not report['passed']:raise SystemExit(1)


if __name__=='__main__':main()
