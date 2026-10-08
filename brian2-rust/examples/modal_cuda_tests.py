"""Run the native CUDA Device conformance suite on one Modal GPU."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def run_tests(test_paths=None,dag_benchmark=False,chunk_benchmark=False):
    """Imported from uploaded source; no serialized cross-version function."""
    import os
    import platform
    import sys
    os.chdir("/workspace")
    start = time.perf_counter()
    selected_tests=test_paths or ["brian2-rust/tests/test_cuda.py",
                                "brian2-rust/tests/test_cuda_compiler.py",
                                "brian2-rust/tests/test_cuda_graphs.py",
                                "brian2-rust/tests/test_cuda_chunks.py",
                                "brian2-rust/tests/test_gpu_spike_generator.py",
                                "brian2-rust/tests/test_gpu_refractory.py",
                                "brian2-rust/tests/test_gpu_random.py",
                                "brian2-rust/tests/test_gpu_timed_array.py",
                                "brian2-rust/tests/test_gpu_initialization.py",
                                "brian2-rust/tests/test_gpu_multiclock.py",
                                "brian2-rust/tests/test_gpu_monitors.py",
                                "brian2-rust/tests/test_gpu_custom_events.py",
                                "brian2-rust/tests/test_gpu_typed_storage.py",
                                "brian2-rust/tests/test_gpu_links.py",
                                "brian2-rust/tests/test_gpu_poisson.py",
                                "brian2-rust/tests/test_gpu_pathway_order.py",
                                "brian2-rust/tests/test_gpu_binomial.py",
                                "brian2-rust/tests/test_gpu_ticks.py"]
    completed = subprocess.run([sys.executable,"-m","pytest",*selected_tests,
                               "-q","--tb=short","--junitxml=/tmp/cuda-tests.xml"],
                               capture_output=True,text=True,env={**os.environ,"B2_TEST_CUDA":"1"})
    baselines={}
    if completed.returncode==0:
        for backend in ("rust","cuda"):
            output=Path("/tmp")/("baseline-"+backend)
            check=subprocess.run([sys.executable,"brian2-rust/examples/gpu_baseline.py","--backend",backend,
                                   "--neurons","32","--steps","32","--output",str(output)],
                                  text=True,capture_output=True)
            baselines[backend]=dict(exit_code=check.returncode,stdout=check.stdout,stderr=check.stderr,
                                    report=json.loads((output/"report.json").read_text()) if (output/"report.json").exists() else None)
    ablations={}
    if (dag_benchmark or chunk_benchmark) and completed.returncode==0 and all(b['exit_code']==0 for b in baselines.values()):
        for neurons,steps,degree in ((512,512,32),(4096,2048,64)):
            label=f'n{neurons}-t{steps}-k{degree}';output=Path('/tmp')/label
            script='cuda_chunk_benchmark.py' if chunk_benchmark else 'cuda_dag_benchmark.py'
            repeat=['--trials','5'] if chunk_benchmark else ['--repeats','7']
            check=subprocess.run([sys.executable,'brian2-rust/examples/'+script,
                '--neurons',str(neurons),'--steps',str(steps),'--degree',str(degree),
                *repeat,'--output',str(output)],text=True,capture_output=True)
            ablations[label]=dict(exit_code=check.returncode,stdout=check.stdout,stderr=check.stderr,
                report=json.loads((output/'report.json').read_text()) if (output/'report.json').exists() else None,
                model=(output/'model.json').read_text() if (output/'model.json').exists() else None)
    return dict(schema="b2-cuda-device-tests-v0",passed=completed.returncode==0 and all(b["exit_code"]==0 for b in baselines.values()) and all(b['exit_code']==0 for b in ablations.values()),
                dag_ablations=ablations,benchmark_kind="chunks" if chunk_benchmark else "replay" if dag_benchmark else None,
                baseline_smoke=baselines,selected_tests=selected_tests,
                compiler_diagnostics={str(p.relative_to("/tmp/pytest-of-root")):json.loads(p.read_text())
                    for p in Path("/tmp/pytest-of-root").rglob("compiler-isolation.json")},
                poisson_diagnostics={str(p.relative_to("/tmp/pytest-of-root")):json.loads(p.read_text())
                    for pattern in ("*-distribution.json","log-mass-diagnostics.json","binomial-probability-diagnostics.json")
                    for p in Path("/tmp/pytest-of-root").rglob(pattern)},
                exit_code=completed.returncode,stdout=completed.stdout,stderr=completed.stderr,
                junit=Path("/tmp/cuda-tests.xml").read_text(),elapsed_seconds=time.perf_counter()-start,
                host=platform.platform(),python=sys.version,
                nvidia_smi=subprocess.check_output(["nvidia-smi"],text=True),
                packages=subprocess.check_output([sys.executable,"-m","pip","freeze"],text=True),
                rustc=subprocess.check_output(["rustc","-vV"],text=True))


def native_image():
    import modal
    ignore=["**/__pycache__/**","**/*.pyc","**/*.so","**/*.dylib"]
    frontend_files=[ROOT.parent/name for name in ("setup.py","pyproject.toml","README.md","LICENSE","AUTHORS","CONTRIBUTORS")]
    image=(modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04",add_python="3.12")
           .entrypoint([]).apt_install("curl","clang","build-essential")
           .run_commands("curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain 1.98.1")
           .env({"PATH":"/root/.cargo/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin",
                 "PYTHONPATH":"/workspace/brian2-rust/python:/workspace",
                 "MPLCONFIGDIR":"/tmp/matplotlib"})
           .pip_install("numpy==2.2.6","cupy-cuda12x==13.6.0","pytest==8.4.1",
                        "cython==3.0.12","sympy==1.14.0","pyparsing==3.2.3",
                        "jinja2==3.1.6","setuptools==80.9.0","packaging==25.0")
           .add_local_file(ROOT/"Cargo.toml","/workspace/brian2-rust/Cargo.toml",copy=True)
           .add_local_file(ROOT/"Cargo.lock","/workspace/brian2-rust/Cargo.lock",copy=True)
           .add_local_dir(ROOT/"src","/workspace/brian2-rust/src",copy=True)
           .run_commands("cargo build --release --locked --manifest-path /workspace/brian2-rust/Cargo.toml"))
    # This Brian checkout requires Cython extensions; compile the exact uploaded
    # frontend source for Linux instead of copying macOS binaries or substituting
    # a released Brian version for our source tree.
    image=image.add_local_dir(ROOT.parent/"brian2","/workspace/brian2",ignore=ignore,copy=True)
    for path in frontend_files:
        image=image.add_local_file(path,"/workspace/"+path.name,copy=True)
    import brian2
    image=(image.pip_install("setuptools_scm==8.3.1","wheel==0.45.1")
           .run_commands("python -m pip install --no-deps --no-build-isolation -e /workspace",
                         env={"SETUPTOOLS_SCM_PRETEND_VERSION":brian2.__version__}))
    return image


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--gpu",choices=("L4","A100-40GB","A100-80GB","H100!"),default="L4")
    parser.add_argument("--test-path",action="append",help="Select a pytest path/node; repeat for targeted follow-up checks. Default: full CUDA suite.")
    benchmarks=parser.add_mutually_exclusive_group()
    benchmarks.add_argument("--dag-benchmark",action="store_true",help="After passing tests, run bounded matched direct/resident/graph CUBA ablations.")
    benchmarks.add_argument("--chunk-benchmark",action="store_true",help="After passing tests, measure five fresh executors per mode for two bounded CUBA workloads.")
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    import modal
    # Only source/test directories enter the container; no .git, credentials,
    # local native binaries, output models, or benchmark data are mounted.
    ignore=["**/__pycache__/**","**/*.pyc","**/*.so","**/*.dylib"]
    source_dirs=[ROOT/"python",ROOT/"tests",ROOT.parent/"brian2",ROOT/"src"]
    hashes={str(p.relative_to(ROOT.parent)):hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in source_dirs for p in sorted(folder.rglob("*"))
            if p.is_file() and "__pycache__" not in p.parts and p.suffix not in {".pyc",".so",".dylib"}}
    frontend_files=[ROOT.parent/name for name in ("setup.py","pyproject.toml","README.md","LICENSE","AUTHORS","CONTRIBUTORS")]
    for p in (ROOT/"Cargo.toml",ROOT/"Cargo.lock",Path(__file__),ROOT/"examples/gpu_baseline.py",ROOT/"examples/cuda_dag_benchmark.py",ROOT/"examples/gpu_recurrent.py",ROOT/"examples/cuda_chunk_benchmark.py",*frontend_files):
        hashes[str(p.relative_to(ROOT.parent))]=hashlib.sha256(p.read_bytes()).hexdigest()
    (args.output/"source-hashes.json").write_text(json.dumps(hashes,sort_keys=True,indent=2)+"\n")
    image=(native_image()
           .add_local_dir(ROOT/"python","/workspace/brian2-rust/python",ignore=ignore)
           .add_local_dir(ROOT/"tests","/workspace/brian2-rust/tests",ignore=ignore)
           .add_local_file(ROOT/"examples/gpu_baseline.py","/workspace/brian2-rust/examples/gpu_baseline.py")
           .add_local_file(ROOT/"examples/cuda_dag_benchmark.py","/workspace/brian2-rust/examples/cuda_dag_benchmark.py")
           .add_local_file(ROOT/"examples/gpu_recurrent.py","/workspace/brian2-rust/examples/gpu_recurrent.py")
           .add_local_file(ROOT/"examples/cuda_chunk_benchmark.py","/workspace/brian2-rust/examples/cuda_chunk_benchmark.py")
           .add_local_file(Path(__file__),"/root/modal_cuda_tests.py"))
    app=modal.App("brian2-native-cuda-tests",include_source=False)
    run=app.function(image=image,gpu=args.gpu,cpu=4,memory=8192,
                     max_containers=1,timeout=900,retries=0)(run_tests)
    started=time.perf_counter()
    with modal.enable_output(),app.run():
        report=run.remote(args.test_path,args.dag_benchmark,args.chunk_benchmark)
    report.update(requested_gpu=args.gpu,remote_wall_seconds=time.perf_counter()-started,
                  source_manifest_sha256=hashlib.sha256((args.output/"source-hashes.json").read_bytes()).hexdigest())
    (args.output/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    (args.output/"pytest.txt").write_text(report["stdout"]+report["stderr"])
    print(report["stdout"])
    if not report["passed"]:raise SystemExit(1)


if __name__=="__main__":
    main()
