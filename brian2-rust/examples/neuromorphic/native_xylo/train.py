"""Train one small native LIF model for a same-weight two-target comparison.

No TensorFlow/PyTorch dependency. This program never reads the test arrays.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from brian2_atlas import NativeLIFTrainer, lif_training_plan
from brian2_atlas.neuromorphic import freeze_native_lif
from brian2_rust.protocol import canonical_bytes

DATASET_SHA256='731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1'


def preprocess(images):
    return (images.reshape(-1,4,7,4,7).mean((2,4)).reshape(-1,16)>=50).astype(np.uint8)


def raster(images, steps=16):
    return np.repeat(images[:,None,:],steps,axis=1)


def evaluate(trainer, x, y, batch_size=128):
    correct=0
    for start in range(0,len(x),batch_size):
        r=trainer.evaluate(raster(x[start:start+batch_size]),y[start:start+batch_size])
        correct+=int(np.count_nonzero(np.argmax(r['logits'],axis=1)==y[start:start+batch_size]))
    return correct/len(x)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--runner',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=900)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    assert hashlib.sha256(a.dataset.read_bytes()).hexdigest()==DATASET_SHA256
    data=np.load(a.dataset,allow_pickle=False)
    all_y=data['y_train']
    train_ix=np.flatnonzero((np.arange(len(all_y))<54000)&(all_y<4))
    validation_ix=np.flatnonzero((np.arange(len(all_y))>=54000)&(all_y<4))
    x=preprocess(data['x_train'][train_ix]);y=all_y[train_ix]
    vx=preprocess(data['x_train'][validation_ix]);vy=all_y[validation_ix]
    rng=np.random.default_rng(20261014)
    plan=lif_training_plan([16,32,4],beta=.5,threshold=1.,learning_rate=.005,
        logit_scale=8.,surrogate_slope=3.,max_tape_bytes=128*1024**2,seed=20261014)
    initial=[rng.normal(.15,.35,(16,32)).ravel().tolist(),rng.normal(.1,.3,(32,4)).ravel().tolist()]
    trainer=NativeLIFTrainer(plan,runner=a.runner,weights=initial)
    history=[];best=-1.;start_time=time.monotonic()
    baseline=evaluate(trainer,vx,vy)
    print(json.dumps(dict(initial_validation_accuracy=baseline)),flush=True)
    for step in range(1,a.steps+1):
        ix=rng.integers(0,len(x),128)
        result=trainer.step(raster(x[ix]),y[ix])
        if step%50==0:print(json.dumps(dict(step=step,loss=result['loss'],elapsed_seconds=time.monotonic()-start_time)),flush=True)
        if step%150 and step!=a.steps:continue
        scales=[float(2**np.floor(np.log2(15/(beta*np.abs(w).max())))) for w,beta in zip(trainer.state['weights'],plan['beta'])]
        frozen=freeze_native_lif(trainer,scales=scales,quantize=True)
        quantized=NativeLIFTrainer(plan,runner=a.runner,weights=frozen['weights'])
        before=evaluate(trainer,vx,vy);after=evaluate(quantized,vx,vy)
        item=dict(step=step,float_validation_accuracy=before,quantized_validation_accuracy=after,scales=scales)
        history.append(item);print(json.dumps(item),flush=True)
        if after>best:
            best=after;trainer.store(a.output/'checkpoint.json')
            (a.output/'frozen.json').write_bytes(canonical_bytes(frozen))
            selected=dict(item,weight_changes=[int(np.count_nonzero(np.asarray(w)!=old))
                for w,old in zip(trainer.state['weights'],initial)])
    report=dict(schema='atlas.native-lif.xylo-training.v1',dataset_sha256=DATASET_SHA256,
        seed=20261014,training_backend='native CPU b2-train',external_training_framework=False,
        train_original_index_range=[0,54000],validation_original_index_range=[54000,60000],
        classes=[0,1,2,3],training_samples=len(x),validation_samples=len(vx),test_arrays_read=False,
        preprocessing='7x7 mean pooling to 4x4; pixel >= 50; repeat binary spikes for 16 ticks',
        sizes=plan['sizes'],trainable_layers=[True,True],optimizer_steps=a.steps,
        selection='highest quantized native accuracy on validation digits 0..3; earliest tie; no target tuning',
        initial_validation_accuracy=baseline,selected=selected,history=history,
        training_seconds=time.monotonic()-start_time,
        runner_sha256=hashlib.sha256(a.runner.read_bytes()).hexdigest(),
        training_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        checkpoint_sha256=hashlib.sha256((a.output/'checkpoint.json').read_bytes()).hexdigest(),
        frozen_file_sha256=hashlib.sha256((a.output/'frozen.json').read_bytes()).hexdigest())
    (a.output/'training.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
