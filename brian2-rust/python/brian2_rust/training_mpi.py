"""Build the small native MPI training ABI in an isolated directory."""
from pathlib import Path
import platform
import shutil
import subprocess


def build(directory):
    compiler=shutil.which('mpicc')
    if compiler is None:raise ValueError('native MPI training requires mpicc')
    library=Path(directory)/('libatlas-training-mpi.dylib' if platform.system()=='Darwin' else 'libatlas-training-mpi.so')
    result=subprocess.run([compiler,'-O2','-fPIC','-ffp-contract=off',
        '-dynamiclib' if platform.system()=='Darwin' else '-shared',
        str(Path(__file__).with_suffix('.c')),'-o',str(library)],capture_output=True,text=True)
    if result.returncode:raise RuntimeError(result.stderr)
    return library
