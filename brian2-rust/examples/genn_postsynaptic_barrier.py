"""Fail-closed, diagnostic-only patch for the pinned GeNN generated kernel.

This is not a stock GeNN result or a change to our Metal/CUDA backend.
"""
import difflib
import hashlib
import re
import subprocess


def insert_barriers(source,*,groups=16):
    if type(groups) is not int or not 1<=groups<=16:
        raise ValueError('Require 1..16 declared delay groups')
    start = source.index('extern "C" __global__ void updatePostsynapticKernel(float t)')
    end = source.index('void updateSynapses(float t)', start)
    kernel = source[start:end]
    block = re.search(r'const unsigned int id = (\d+) \* blockIdx.x', kernel)
    if block is None:
        raise ValueError('Unrecognized postsynaptic block layout')
    width = int(block[1])
    if not width:raise ValueError('Invalid postsynaptic block width')
    bounds = re.findall(r'if\(id (?:>= (\d+) && id )?< (\d+)\)', kernel)
    if len(bounds) != groups or any(int(v) % width for pair in bounds for v in pair if v):
        raise ValueError('Barrier requires the declared number of block-uniform groups')
    previous=0
    for lower,upper in bounds:
        if int(lower or 0)!=previous or int(upper)<=previous:
            raise ValueError('Unexpected postsynaptic group ranges')
        previous=int(upper)
    needle = re.compile(r'(const unsigned int numSpikesInBlock = [^;]+;\n)( +)(if \(threadIdx.x < numSpikesInBlock\))')
    if len(needle.findall(kernel)) != groups or kernel.count('__syncthreads();') != groups:
        raise ValueError('Unexpected GeNN spike-block synchronization layout')
    patched = needle.sub(r'\1\2__syncthreads(); // diagnostic: finish previous shared-buffer readers\n\2\3', kernel)
    return source[:start] + patched + source[end:]


def patch_and_build(project,*,groups=16):
    path = project / 'b2_delayed_stdp_CODE' / 'synapseUpdate.cc'
    source = path.read_text()
    patched = insert_barriers(source,groups=groups)
    (project / 'synapseUpdate.original.cc').write_text(source)
    (project / 'postsynaptic-barrier.patch').write_text(''.join(difflib.unified_diff(
        source.splitlines(True), patched.splitlines(True), fromfile='stock/synapseUpdate.cc', tofile='barrier/synapseUpdate.cc')))
    path.write_text(patched)
    done = subprocess.run(['make', '-j2'], cwd=path.parent, capture_output=True, text=True, timeout=300)
    (project / 'barrier-build.log').write_text(done.stdout + done.stderr)
    done.check_returncode()
    return dict(kind='diagnostic-generated-source-barrier', barriers_added=groups,
                original_sha256=hashlib.sha256(source.encode()).hexdigest(),
                patched_sha256=hashlib.sha256(patched.encode()).hexdigest())
