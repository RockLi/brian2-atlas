"""Train Xylo integer dynamics natively; choose checkpoints on validation only."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from brian2_atlas import NativeXyloTrainer,xylo_training_plan
from brian2_atlas.neuromorphic import freeze_native_xylo
from brian2_rust.protocol import canonical_bytes

DATASET_SHA256='731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1'


def preprocess(images,encoding='rate'):
    pooled=images.reshape(-1,4,7,4,7).mean((2,4)).reshape(-1,16)
    return ((pooled>=50)*16 if encoding=='binary' else np.rint(pooled*16/255)).astype(np.uint8)


def raster(levels,steps=32):
    t=np.arange(steps)[None,:,None]
    values=levels[:,None,:].astype(int)
    return (((t+1)*values//16)-(t*values//16)).astype(np.uint8)


def evaluate(trainer,levels,labels,steps):
    correct=0
    for i in range(0,len(levels),64):
        result=trainer.evaluate(raster(levels[i:i+64],steps),labels[i:i+64])
        correct+=np.count_nonzero(np.argmax(result['logits'],1)==labels[i:i+64])
    return float(correct/len(levels))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('dataset','runner','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--updates',type=int,default=1800)
    p.add_argument('--encoding',choices=('rate','binary'),default='rate')
    p.add_argument('--hidden',type=int,default=64)
    p.add_argument('--ticks',type=int,default=32)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    assert hashlib.sha256(a.dataset.read_bytes()).hexdigest()==DATASET_SHA256
    data=np.load(a.dataset,allow_pickle=False);all_y=data['y_train'];ix=np.arange(len(all_y))
    train_ix=np.flatnonzero((ix<54000)&(all_y<4));validation_ix=np.flatnonzero((ix>=54000)&(all_y<4))
    x=preprocess(data['x_train'][train_ix],a.encoding);y=all_y[train_ix]
    vx=preprocess(data['x_train'][validation_ix],a.encoding);vy=all_y[validation_ix]
    rng=np.random.default_rng(20261014)
    plan=xylo_training_plan([16,a.hidden,4])
    initial=[rng.normal(.02,.18,(16,a.hidden)).ravel().tolist(),rng.normal(.02,.12,(a.hidden,4)).ravel().tolist()]
    trainer=NativeXyloTrainer(plan,runner=a.runner,weights=initial)
    baseline=evaluate(trainer,vx,vy,a.ticks);print('Initial validation:',baseline,flush=True)
    history=[];best=-1.;start=time.monotonic()
    for step in range(1,a.updates+1):
        batch=rng.integers(0,len(x),64)
        result=trainer.step(raster(x[batch],a.ticks),y[batch])
        if step%100==0:print(json.dumps(dict(step=step,loss=result['loss'],seconds=time.monotonic()-start)),flush=True)
        if step%300 and step!=a.updates:continue
        accuracy=evaluate(trainer,vx,vy,a.ticks)
        item=dict(step=step,validation_accuracy=accuracy);history.append(item);print(json.dumps(item),flush=True)
        if accuracy>best:
            best=accuracy;trainer.store(a.output/'checkpoint.json')
            frozen=freeze_native_xylo(trainer)
            (a.output/'frozen.json').write_bytes(canonical_bytes(frozen))
            selected=dict(item,changed_latent_weights=[int(np.count_nonzero(np.asarray(w)!=old)) for w,old in zip(trainer.state['weights'],initial)])
    report=dict(schema='atlas.native-xylo.integer-training.v1',dataset_sha256=DATASET_SHA256,
        encoding=a.encoding,encoding_description='7x7 average pool to 4x4; '+('threshold 50' if a.encoding=='binary' else 'round intensity to 0..16 levels; deterministic pulse-density encoding'),
        ticks=a.ticks,sizes=plan['sizes'],classes=[0,1,2,3],training_samples=len(x),validation_samples=len(vx),
        train_original_index_range=[0,54000],validation_original_index_range=[54000,60000],test_arrays_read=False,
        initial_validation_accuracy=baseline,selected=selected,history=history,updates=a.updates,
        selection='highest native integer validation accuracy; earliest tie; no SDK or test selection',
        training='CPU b2-train integer forward, surrogate BPTT, quantization STE, Adam',
        seed=20261014,external_training_framework=False,training_seconds=time.monotonic()-start,
        runner_sha256=hashlib.sha256(a.runner.read_bytes()).hexdigest(),
        training_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        checkpoint_sha256=hashlib.sha256((a.output/'checkpoint.json').read_bytes()).hexdigest(),
        frozen_sha256=hashlib.sha256((a.output/'frozen.json').read_bytes()).hexdigest())
    (a.output/'training.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
