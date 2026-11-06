"""Train a four-character-context LIF language model entirely in native Atlas."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from brian2_atlas import NativeLIFTrainer, lif_training_plan
from brian2_atlas.neuromorphic import freeze_native_lif
from brian2_rust.protocol import canonical_bytes
from common import CONTEXT, DATASET_SHA256, SEED, SPLITS, TICKS, VOCAB, baselines, corpus, evaluate, metrics, raster, sha, windows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--runner', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--steps', type=int, default=3000)
    a=p.parse_args()
    if a.steps < 1: p.error('--steps must be positive')
    a.output.mkdir(parents=True, exist_ok=False)
    tokens=corpus(a.dataset)
    x,y,_=windows(tokens,'train');vx,vy,_=windows(tokens,'validation')
    rng=np.random.default_rng(SEED)
    plan=lif_training_plan([CONTEXT*len(VOCAB)+1,128,len(VOCAB)], beta=.5, threshold=1.,
        learning_rate=.005, logit_scale=8., surrogate_slope=3., max_tape_bytes=128*1024**2, seed=SEED)
    initial=[rng.normal(.25,.35,(plan['sizes'][0],128)).ravel().tolist(),
             rng.normal(.015,.1,(128,len(VOCAB))).ravel().tolist()]
    trainer=NativeLIFTrainer(plan, runner=a.runner, weights=initial)
    start=time.monotonic();best=float('inf');history=[]
    baseline=baselines(tokens,vx,vy)
    untrained=metrics(evaluate(trainer,vx,vy),vy)
    print(json.dumps(dict(validation_baselines=baseline,untrained=untrained)),flush=True)
    for step in range(1,a.steps+1):
        ix=rng.integers(0,len(y),128)
        result=trainer.step(raster(x[ix]),y[ix])
        if step%100==0:
            print(json.dumps(dict(step=step,loss=result['loss'],elapsed_seconds=time.monotonic()-start)),flush=True)
        if step%500 and step!=a.steps: continue
        scales=[float(2**np.floor(np.log2(15/(beta*np.abs(w).max()))))
                for w,beta in zip(trainer.state['weights'],plan['beta'])]
        frozen=freeze_native_lif(trainer,scales=scales,quantize=True)
        quantized=NativeLIFTrainer(plan,runner=a.runner,weights=frozen['weights'])
        item=dict(step=step,scales=scales,float_validation=metrics(evaluate(trainer,vx,vy),vy),
                  quantized_validation=metrics(evaluate(quantized,vx,vy),vy))
        history.append(item);print(json.dumps(item),flush=True)
        if item['quantized_validation']['perplexity'] < best:
            best=item['quantized_validation']['perplexity']
            trainer.store(a.output/'checkpoint.json')
            (a.output/'frozen.json').write_bytes(canonical_bytes(frozen))
            selected=dict(item,weight_changes=[int(np.count_nonzero(np.asarray(w)!=old))
                for w,old in zip(trainer.state['weights'],initial)])
    report=dict(schema='atlas.character-lm.training.v1',dataset_sha256=DATASET_SHA256,
        normalization='lowercase; characters outside fixed vocabulary replaced with space; first 200000 characters',
        vocabulary=VOCAB,context_characters=CONTEXT,inference_ticks=TICKS,splits=SPLITS,
        sizes=plan['sizes'],seed=SEED,training_backend='native CPU b2-train',external_training_framework=False,
        trainable_layers=[True,True],optimizer_steps=a.steps,batch_size=128,test_evaluated=False,
        selection='minimum perplexity on all 9996 validation targets after explicit quantization; earliest tie',
        validation_baselines=baseline,untrained_validation=untrained,selected=selected,history=history,
        training_seconds=time.monotonic()-start,runner_sha256=sha(a.runner),
        source_sha256={p.name:sha(p) for p in (Path(__file__),Path(__file__).with_name('common.py'))},
        checkpoint_sha256=sha(a.output/'checkpoint.json'),frozen_file_sha256=sha(a.output/'frozen.json'))
    (a.output/'training.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(selected=selected,training_seconds=report['training_seconds'])),flush=True)


if __name__=='__main__': main()
