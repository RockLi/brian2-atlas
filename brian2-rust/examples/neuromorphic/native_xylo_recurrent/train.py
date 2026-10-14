"""Native recurrent integer training on delayed AB/BA discrimination.

Each pair has the same per-channel counts and identical query/suffix. Only
symbol order differs. Timing templates are split before creating class pairs.
No test templates or labels participate in training or checkpoint selection.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import time
import numpy as np
from brian2_atlas import NativeXyloTrainer,xylo_recurrent_training_plan
from brian2_atlas.neuromorphic import freeze_native_xylo
from brian2_rust.protocol import canonical_bytes

STEPS=40
READOUT_START=32
SEED=20261014


def data(split):
    patterns=list(itertools.combinations(range(8),3))
    templates=list(itertools.product(patterns,patterns))
    order=np.random.default_rng(SEED).permutation(len(templates))
    bounds={'train':(0,1800),'validation':(1800,2200),'test':(2200,3136)}
    lo,hi=bounds[split];ids=order[lo:hi]
    x=np.zeros((len(ids)*2,STEPS,3),np.uint8);labels=np.tile([0,1],len(ids))
    for i,identity in enumerate(ids):
        a,c=templates[identity]
        for label in (0,1):
            x[2*i+label,2+np.array(a),label]=1
            x[2*i+label,14+np.array(c),1-label]=1
            x[2*i+label,32:36,2]=1
    return x,labels,np.repeat(ids,2)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evaluate(trainer,x,y):
    correct=0
    for i in range(0,len(x),64):
        r=trainer.evaluate(x[i:i+64],y[i:i+64])
        correct+=np.count_nonzero(np.argmax(r['logits'],axis=1)==y[i:i+64])
    return float(correct/len(y))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runner',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--updates',type=int,default=1200)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    x,y,ids=data('train');vx,vy,vi=data('validation');rng=np.random.default_rng(SEED)
    plan=xylo_recurrent_training_plan([3,24,2],readout_start=READOUT_START,
        thresholds=(8,8),weight_scale=16.,learning_rate=.001,logit_scale=8.)
    initial=[rng.normal(.15,.25,(3,24)).ravel(),rng.normal(.02,.1,(24,2)).ravel(),
             (np.eye(24)*.5+rng.normal(0,.01,(24,24))).ravel()]
    trainer=NativeXyloTrainer(plan,runner=a.runner,weights=initial)
    baseline=evaluate(trainer,vx,vy);history=[];best=-1.;start=time.monotonic()
    print('Initial validation',baseline,flush=True)
    for step in range(1,a.updates+1):
        ix=rng.integers(0,len(x),32);result=trainer.step(x[ix],y[ix])
        if step%100 and step!=a.updates:continue
        score=evaluate(trainer,vx,vy)
        item=dict(step=step,validation_accuracy=score,loss=result['loss']);history.append(item)
        print(json.dumps(item),flush=True)
        if score>best:
            best=score;trainer.store(a.output/'checkpoint.json')
            f=freeze_native_xylo(trainer);(a.output/'frozen.json').write_bytes(canonical_bytes(f))
            selected=dict(item,changed_latent_weights=[int(np.count_nonzero(np.asarray(w)!=old)) for w,old in zip(trainer.state['weights'],initial)])
    report=dict(schema='atlas.native-xylo.recurrent-training.v1',task='Delayed AB/BA discrimination',
        sizes=plan['sizes'],steps=STEPS,readout_start=READOUT_START,seed=SEED,updates=a.updates,
        training_samples=len(y),validation_samples=len(vy),training_templates=len(set(ids)),validation_templates=len(set(vi)),
        test_split_read=False,initial_validation_accuracy=baseline,selected=selected,history=history,
        selection='highest native validation accuracy; earliest tie; no test/SDK selection',
        training='native CPU integer forward and full-sequence surrogate BPTT/STE with Adam; all three banks train',
        training_seconds=time.monotonic()-start,training_script_sha256=sha(__file__),runner_sha256=sha(a.runner),
        checkpoint_sha256=sha(a.output/'checkpoint.json'),frozen_file_sha256=sha(a.output/'frozen.json'))
    (a.output/'training.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
