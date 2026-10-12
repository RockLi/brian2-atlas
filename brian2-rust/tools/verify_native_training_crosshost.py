"""Verify pre-staged native BPTT on two hosts using existing Teleport MPI.

Requires an isolated remote directory with matching b2-train/shim/toy requests.
Does not install software, alter services, or copy authentication material.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import shlex
import subprocess

from mpi_teleport_launch import launch


def continuation_request(original, result, ranks=None):
    """Retain every v4 state, including auxiliary states and refractory ticks."""
    request=copy.deepcopy(original)
    vector=request['plan'].get('state_equations') is not None
    field='final_state' if vector else 'final_membrane'
    if result.get(field) is None:
        raise ValueError('continuation requires '+field)
    request.update(state=copy.deepcopy(result['state']),initial=copy.deepcopy(result[field]))
    if request['plan'].get('clock') is not None:
        tick=result.get('final_tick')
        if type(tick) is not int or not 0<=tick<=2**53:
            raise ValueError('continuation requires final_tick')
        request['start_tick']=tick
    if request['plan'].get('noise_streams') is not None:
        sequence=result.get('noise_sequence')
        if type(sequence) is not int or not 0<=sequence<2**64-1:
            raise ValueError('continuation requires noise_sequence')
        request['noise_sequence']=sequence
    if ranks is not None:request['plan']['mpi_ranks']=ranks
    return request


def compare_outputs(a,b,require_full_state=False):
    """CPU cross-host contract, including complete v4 state and its VJP."""
    maximum=0.
    def recurse(x,y):
        nonlocal maximum
        if isinstance(x,dict):
            assert isinstance(y,dict) and x.keys()==y.keys(),'result fields differ'
            for key in x:recurse(x[key],y[key])
        elif isinstance(x,list):
            assert isinstance(y,list) and len(x)==len(y),'result shape differs'
            for xv,yv in zip(x,y):recurse(xv,yv)
        elif isinstance(x,bool) or isinstance(x,int):
            assert type(x) is type(y) and x==y,'discrete result differs'
        elif isinstance(x,float):
            assert type(y) in (int,float) and math.isfinite(x) and math.isfinite(y),'nonfinite or nonnumeric result'
            maximum=max(maximum,abs(x-y))
            assert math.isclose(x,y,rel_tol=3e-12,abs_tol=3e-13),(x,y)
        else:assert x==y
    assert a['spikes']==b['spikes'],'spikes differ'
    for key in ['state','loss','gradients','initial_gradients','final_membrane','logits']:
        recurse(a[key],b[key])
    for key in ['final_state','initial_state_gradients']:
        if require_full_state or a.get(key) is not None or b.get(key) is not None:
            assert a.get(key) is not None and b.get(key) is not None,'missing '+key
            recurse(a[key],b[key])
    for key in ['final_tick','noise_sequence']:
        if key in a or key in b:
            assert key in a and key in b,'missing '+key
            recurse(a[key],b[key])
    return maximum


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--nodes',nargs=2,required=True)
    parser.add_argument('--ips',nargs=2,required=True)
    parser.add_argument('--remote-base',required=True)
    parser.add_argument('--mpi-prefix',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    base=args.remote_base
    def remote(node,argv):
        return subprocess.run(['tsh','ssh',node,shlex.join(argv)],stdin=subprocess.DEVNULL,
                               capture_output=True,text=True,check=True,timeout=40).stdout
    def fetch(name):
        data=remote(args.nodes[0],['python3','-c','import pathlib,sys;print(pathlib.Path(sys.argv[1]).read_text())',base+'/'+name])
        (args.output/name).write_text(data);return json.loads(data)
    def stage(name,value):
        path=args.output/name;path.write_text(json.dumps(value))
        for node in args.nodes:
            subprocess.run(['tsh','scp',str(path),node+':'+base+'/'+name],
                           stdin=subprocess.DEVNULL,capture_output=True,text=True,check=True,timeout=40)
    facts=[]
    code='''import hashlib,json,pathlib,socket,sys
base=pathlib.Path(sys.argv[1])
print(json.dumps(dict(host=socket.gethostname(),sha256={n:hashlib.sha256((base/n).read_bytes()).hexdigest() for n in ['target/release/b2-train','libatlas-training-mpi.so','request-2.json','request-4.json','rank_launch.py']})))
'''
    for node in args.nodes:facts.append(json.loads(remote(node,['python3','-c',code,base])))
    assert facts[0]['sha256']==facts[1]['sha256'],'host payload mismatch'
    assert {f['host'] for f in facts}==set(args.nodes),'unexpected hosts'
    (args.output/'host-facts.json').write_text(json.dumps(facts,indent=2)+'\n')
    serial=fetch('serial.json');original=fetch('request-0.json')
    second=continuation_request(original,serial)
    stage('serial-next-request.json',second)
    remote(args.nodes[0],['timeout','-k','5','30',base+'/target/release/b2-train',base+'/serial-next-request.json',base+'/serial-next.json'])
    serial_next=fetch('serial-next.json')
    def compare(a,b):
        return compare_outputs(a,b,require_full_state=original['plan'].get('state_equations') is not None)
    results=[]
    for per_node in [1,2]:
        ranks=per_node*2
        def run(request,name):
            report=launch(nodes=args.nodes,ips=args.ips,ranks_per_node=per_node,
                remote_base=base,mpi_prefix=args.mpi_prefix,timeout=60,output=args.output/name,
                runtime_environment={'B2_TRAIN_MPI_LIB':base+'/libatlas-training-mpi.so'},
                application=['python3',base+'/rank_launch.py',base+'/target/release/b2-train',base+'/'+request,base+'/'+name+'.json'])
            assert report['error'] is None and all(code==0 for code in report['returncodes'].values()),report
            inventory=[]
            for line in (args.output/name/'controller.log').read_text().splitlines():
                if line.startswith('{"hostname"'):inventory.append(json.loads(line))
            assert len(inventory)==ranks and {v['hostname'] for v in inventory}==set(args.nodes)
            assert sorted(int(v['rank']) for v in inventory)==list(range(ranks))
            return fetch(name+'.json'),inventory
        first,hosts=run(f'request-{ranks}.json',f'first-r{ranks}')
        first_error=compare(first,serial)
        request=continuation_request(original,first,ranks)
        name=f'next-request-r{ranks}.json';stage(name,request)
        continued,_=run(name,f'next-r{ranks}');replay,_=run(name,f'replay-r{ranks}')
        assert continued==replay,'fresh process serialized-state replay mismatch'
        next_error=compare(continued,serial_next)
        row=dict(ranks=ranks,ranks_per_node=per_node,hosts=hosts,
                 first_max_error=first_error,continued_max_error=next_error,
                 exact_replay=True,optimizer_step=continued['state']['step'])
        results.append(row);print(json.dumps(row),flush=True)
    (args.output/'verification.json').write_text(json.dumps(dict(passed=True,results=results,hosts=facts),indent=2)+'\n')


if __name__=='__main__':main()
