"""Compile Atlas's native Metal BPTT library in an isolated directory."""
import json
from pathlib import Path
import platform
import subprocess


def build(directory):
    if platform.system()!='Darwin':
        raise ValueError('native Metal training requires macOS')
    package=Path(__file__).parent;directory=Path(directory)
    shader='\n'.join((package/name).read_text() for name in
                    ('training_metal.metal','training_metal_mpi.metal','training_state.metal','training_clock.metal','training_poisson.metal','training_dynamic.metal','training_dynamic_mpi.metal'))
    (directory/'training_shader.h').write_text('static const char* ATLAS_TRAIN_SHADER='+json.dumps(shader)+';\n')
    library=directory/'libatlas-training-metal.dylib'
    result=subprocess.run(['clang','-O2','-fobjc-arc','-ffp-contract=off','-dynamiclib',
        '-framework','Foundation','-framework','Metal','-framework','CoreGraphics','-I',str(directory),
        str(package/'training_metal.m'),'-o',str(library)],capture_output=True,text=True)
    if result.returncode: raise RuntimeError(result.stderr)
    return library
