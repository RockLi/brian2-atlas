"""Console entry points for the native executables bundled with Atlas."""
import subprocess
import sys
from ._runtime import executable_path


def runner():
    return subprocess.call([str(executable_path("b2-runner")), *sys.argv[1:]])


def train():
    return subprocess.call([str(executable_path("b2-train")), *sys.argv[1:]])
