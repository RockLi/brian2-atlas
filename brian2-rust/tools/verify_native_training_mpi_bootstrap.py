"""CPU-only MPI bootstrap gate: exact shim, rank identity and native execution.

Suitable for an image-build step before allocating any GPU. A mismatched MPI
launcher/library cannot silently pass by starting independent singleton ranks.
"""
import argparse
import copy
import ctypes
import json
import math
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import tempfile


def run(command,env):
    with subprocess.Popen(command,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                          start_new_session=True) as process:
        try:stdout,stderr=process.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGKILL);process.communicate();raise
    if process.returncode:raise RuntimeError(f'MPI preflight command failed: {command}\n{stderr}\n{stdout}')
    return stdout


def child(library):
    shim=ctypes.CDLL(str(library))
    rank=ctypes.c_int();size=ctypes.c_int()
    init=shim.b2_train_mpi_init;init.argtypes=[ctypes.POINTER(ctypes.c_int)]*2;init.restype=ctypes.c_int
    if init(ctypes.byref(rank),ctypes.byref(size))!=0:raise RuntimeError('MPI_Init failed')
    total=ctypes.c_double(rank.value+1)
    reduce=shim.b2_train_mpi_sum;reduce.argtypes=[ctypes.POINTER(ctypes.c_double),ctypes.c_uint64]
    if reduce(ctypes.byref(total),1)!=0:shim.b2_train_mpi_abort();raise RuntimeError('MPI sum failed')
    print(json.dumps(dict(rank=rank.value,size=size.value,total=total.value)),flush=True)
    if shim.b2_train_mpi_finish()!=0:raise RuntimeError('MPI finalization failed')


def verify(runner,source):
    env=os.environ.copy()
    if platform.system()=='Darwin':env.setdefault('FI_PROVIDER','tcp')
    with tempfile.TemporaryDirectory(prefix='b2-mpi-preflight-') as directory:
        d=Path(directory);lib=d/('shim.dylib' if platform.system()=='Darwin' else 'shim.so')
        run(['mpicc','-O2','-fPIC','-ffp-contract=off','-dynamiclib' if platform.system()=='Darwin' else '-shared',str(source),'-o',str(lib)],env)
        results=[]
        for ranks in [2,4]:
            text=run(['mpiexec','-n',str(ranks),sys.executable,str(Path(__file__).resolve()),'--child',str(lib)],env)
            rows=[json.loads(line) for line in text.splitlines() if line.startswith('{')]
            expected=[dict(rank=r,size=ranks,total=ranks*(ranks+1)/2) for r in range(ranks)]
            if sorted(rows,key=lambda row:row['rank'])!=expected:
                raise RuntimeError(f'MPI bootstrap contract failed: expected={expected}, observed={rows}')
            results.append(dict(ranks=ranks,identities=rows))
        plan=dict(schema='b2-lif-training-plan-v1',backend='cpu',sizes=[1,1,2],beta=[.9,.9],threshold=[1.,1.],
            reset='subtract',detach_reset=False,surrogate=dict(kind='fast_sigmoid',slope=2.,scale=1.),
            optimizer=dict(kind='adam',learning_rate=.001,beta1=.9,beta2=.999,epsilon=1e-8),
            trainable=[True,True],masks=[[1.],[1.,1.]],seed=1,logit_scale=5.,max_tape_bytes=1024**2,tbptt_window=None)
        request=dict(plan=plan,state=None,operation='train',inputs=[[[1.],[0.],[1.],[1.]],[[0.],[1.],[1.],[0.]]],
                     labels=[0,1],initial=[[1.2,1.4,.1],[.4,.2,1.2]])
        inp=d/'request.json';out=d/'result.json'
        inp.write_text(json.dumps(request));serial_env=env.copy();serial_env.pop('B2_TRAIN_MPI_LIB',None)
        run([str(runner),str(inp),str(out)],serial_env);baseline=json.loads(out.read_text())
        distributed_env=dict(env,B2_TRAIN_MPI_LIB=str(lib))
        def compare(a,b):
            if isinstance(a,list):
                assert len(a)==len(b)
                for x,y in zip(a,b):compare(x,y)
            elif isinstance(a,dict):
                assert a.keys()==b.keys()
                for key in a:compare(a[key],b[key])
            elif isinstance(a,(float,int)):
                assert math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12),(a,b)
            else:assert a==b,(a,b)
        for result in results:
            request['plan']['mpi_ranks']=result['ranks'];inp.write_text(json.dumps(request));out.unlink(missing_ok=True)
            run(['mpiexec','-n',str(result['ranks']),str(runner),str(inp),str(out)],distributed_env)
            output=json.loads(out.read_text())
            assert output['spikes']==baseline['spikes']
            for key in ['state','loss','gradients','initial_gradients','final_membrane','logits']:compare(baseline[key],output[key])
            result['native_matches_serial']=True
        return dict(passed=True,results=results,
            compiler=run(['mpicc','-show'],env).strip(),launcher=run(['mpiexec','--version'],env),
            runner=str(runner),shim_source=str(source))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--child',type=Path)
    parser.add_argument('--runner',type=Path);parser.add_argument('--shim-source',type=Path)
    args=parser.parse_args()
    if args.child:child(args.child)
    else:
        if not args.runner or not args.shim_source:parser.error('--runner and --shim-source are required')
        print(json.dumps(verify(args.runner.resolve(),args.shim_source.resolve()),indent=2))


if __name__=='__main__':main()
