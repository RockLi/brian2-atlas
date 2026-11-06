"""One bounded L4 native-training verification, with source-only upload."""
import argparse
import hashlib
import json
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[1]


def run_tests():
    import os
    import subprocess
    import sys
    os.chdir('/workspace')
    env=dict(os.environ,B2_TRAIN_RUNNER='/workspace/brian2-rust/target/release/b2-train',
             B2_TEST_CUDA_TRAIN='1',B2_RUNNER='/workspace/brian2-rust/target/release/b2-runner')
    result=subprocess.run([sys.executable,'-m','pytest','-q',
        'brian2-rust/tests/test_native_training_cuda.py','brian2-rust/tests/test_native_training_equations.py','--junitxml=/tmp/native-cuda.xml'],
        env=env,capture_output=True,text=True,timeout=600)
    examples=[]
    if result.returncode==0:
        for backend in ['cpu','cuda']:
            checkpoint='/tmp/equations-'+backend+'.json'
            for resume in [False,True]:
                command=[sys.executable,'brian2-rust/examples/native_graph_training.py',
                    '--backend',backend,'--equations','--steps','0' if resume else '80',
                    '--checkpoint',checkpoint]+(['--resume'] if resume else [])
                example=subprocess.run(command,env=env,capture_output=True,text=True,timeout=180)
                examples.append(dict(argv=command,code=example.returncode,stdout=example.stdout,stderr=example.stderr))
    return dict(passed=result.returncode==0 and all(e['code']==0 for e in examples),examples=examples,
                stdout=result.stdout,stderr=result.stderr,
                junit=Path('/tmp/native-cuda.xml').read_text(),
                gpu=subprocess.check_output(['nvidia-smi'],text=True),
                compiler=subprocess.check_output(['nvcc','--version'],text=True))


def main():
    import modal
    from modal_cuda_tests import native_image
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    files=[p for d in ('src','python','tests') for p in (ROOT/d).rglob('*')
           if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.pyc','.so','.dylib')]
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (args.output/'source-hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
    image=(native_image().add_local_dir(ROOT/'python','/workspace/brian2-rust/python',
            ignore=['**/__pycache__/**','**/*.pyc'])
           .add_local_dir(ROOT/'tests','/workspace/brian2-rust/tests',ignore=['**/__pycache__/**','**/*.pyc'])
           .add_local_file(ROOT/'examples/native_graph_training.py','/workspace/brian2-rust/examples/native_graph_training.py')
           .add_local_file(Path(__file__),'/root/modal_native_training.py'))
    app=modal.App('atlas-native-training-acceptance',include_source=False)
    run=app.function(image=image,gpu='L4',cpu=2,memory=4096,max_containers=1,
                     timeout=900,retries=0)(run_tests)
    started=time.monotonic()
    with modal.enable_output(),app.run():report=run.remote()
    report['elapsed_seconds']=time.monotonic()-started
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    (args.output/'junit.xml').write_text(report['junit'])
    print(report['stdout']);print(report['stderr'])
    if not report['passed']:raise SystemExit(1)


if __name__=='__main__':main()
