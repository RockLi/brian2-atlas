"""Bounded predecessor-only compilation reuse; models/plans are never cached."""
import hashlib
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess

from .protocol import canonical_bytes

MAX_CUBIN_BYTES=32*1024**2

def sha(data):return hashlib.sha256(data).hexdigest()

def request(model,enabled,previous,cls):
    if type(enabled) is not bool:raise ValueError('compile_reuse must be boolean')
    if previous is not None:
        if not enabled or type(previous) is not cls:raise ValueError('compile reuse requires an enabled same-backend predecessor')
        if getattr(previous,'closed',False) or (hasattr(previous,'handles') and not previous.handles):
            raise RuntimeError('compilation predecessor is closed')
    # Opaque native source may depend on files/preprocessor behavior outside the
    # generated portable includes. It still compiles normally, without reuse.
    return enabled and not any(f['body'] is None for f in model['definition']['functions'])


def compiler_context(executable,environment,extra):
    path=Path(shutil.which(executable) or executable).resolve()
    files={str(path):sha(path.read_bytes())}
    if path.name=='clang' and platform.system()=='Darwin':
        actual=Path(subprocess.check_output(['xcrun','--find','clang'],env=environment,text=True).strip()).resolve()
        files[str(actual)]=sha(actual.read_bytes())
        sdk=Path(subprocess.check_output(['xcrun','--show-sdk-path'],env=environment,text=True).strip())
        for p in (sdk/'SDKSettings.json',sdk/'SDKSettings.plist'):
            if p.is_file():files[str(p)]=sha(p.read_bytes())
    if path.name=='nvcc':
        for p in [*(path.parent/n for n in ('ptxas','nvlink','fatbinary','cudafe++')),path.parent.parent/'nvvm/bin/cicc']:
            if p.is_file():files[str(p)]=sha(p.read_bytes())
        host=environment.get('NVCC_CCBIN','g++')
        p=Path(shutil.which(host,path=environment.get('PATH')) or host)
        if p.is_file():files[str(p.resolve())]=sha(p.read_bytes())
    # Environment values are never published, only the combined digest.
    return sha(canonical_bytes(dict(files=files,environment=environment,extra=extra,
        platform=platform.platform(),machine=platform.machine(),cwd=os.getcwd())))


def dependency_files(output,source):
    target,separator,body=output.replace('\\\n',' ').partition(':')
    if not separator or target.strip()!='b2':raise RuntimeError('Unexpected nvcc dependency format')
    paths=sorted({Path(p).resolve() for p in shlex.split(body)}-{Path(source).resolve()})
    if not paths or not all(p.is_file() for p in paths):raise RuntimeError('Incomplete nvcc dependency set')
    return paths


def cuda_dependencies(nvcc,source,architecture,options,environment):
    result=subprocess.run([nvcc,'-M','-MT','b2',f'--gpu-architecture={architecture}',*options,str(source)],
                          env=environment,text=True,capture_output=True)
    if result.returncode:raise RuntimeError('CUDA dependency scan failed:\n'+result.stderr)
    paths=dependency_files(result.stdout,source)
    return sha(canonical_bytes({str(p):sha(p.read_bytes()) for p in paths})),len(paths)


def cached_binary(previous,key):
    record=getattr(previous,'_compiled_binaries',{}).get(key)
    if record is None:return None
    digest,data=record
    return data if isinstance(data,bytes) and sha(data)==digest else None


def save_binary(executor,key,data):
    used=sum(len(record[1]) for record in executor._compiled_binaries.values())
    if len(data)+used<=MAX_CUBIN_BYTES:executor._compiled_binaries[key]=(sha(data),data)
