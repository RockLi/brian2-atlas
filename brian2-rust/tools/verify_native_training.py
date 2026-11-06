"""Bounded equation-learning and fresh-process checkpoint acceptance."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--backend',choices=['cpu','metal','cuda','mpi2'],action='append')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    example=Path(__file__).resolve().parents[1]/'examples/native_graph_training.py'
    records=[]
    for label in args.backend or ['cpu']:
        backend='cpu' if label=='mpi2' else label
        checkpoint=args.output/(label+'.checkpoint')
        pair=[]
        for resume in [False,True]:
            command=[sys.executable,str(example),'--backend',backend,'--equations',
                '--steps','0' if resume else '80','--checkpoint',str(checkpoint)]
            if label=='mpi2':command+=['--mpi-ranks','2']
            if resume:command+=['--resume']
            process=subprocess.run(command,capture_output=True,text=True,check=True,timeout=240)
            result=json.loads(process.stdout);pair.append(result)
            records.append(dict(label=label,resume=resume,argv=command,result=result))
        trained,restored=pair
        assert trained['loss_after']<trained['loss_before'] and all(trained['banks_changed'])
        assert trained['heldout_predictions']==[0,0,1,1]
        for key in ['weights','loss_after','heldout_predictions','optimizer_step']:
            assert trained[key]==restored[key],key
        print(label,trained['loss_before'],trained['loss_after'],flush=True)
    report=dict(runtime_sha256=hashlib.sha256(Path(os.environ['B2_TRAIN_RUNNER']).read_bytes()).hexdigest(),records=records)
    (args.output/'examples.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
