"""Native Atlas classification; forward, surrogate BPTT and Adam run natively."""
import argparse
import json
from pathlib import Path
import numpy as np
from brian2_rust import NativeLIFTrainer,lif_training_plan


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--backend',choices=['cpu','metal'],default='cpu')
    parser.add_argument('--steps',type=int,default=60)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    plan=lif_training_plan([2,2,2],backend=args.backend,beta=0.8,learning_rate=0.04,
                           surrogate_slope=3,logit_scale=3)
    trainer=NativeLIFTrainer(plan,weights=[[1.2,.4,.4,1.2],[.2,.8,.8,.2]])
    if args.resume:trainer.restore(args.checkpoint)
    train=np.zeros((4,24,2));train[:2,:,0]=1;train[2:,:,1]=1;labels=[0,0,1,1]
    initial=trainer.evaluate(train,labels)['loss']
    for _ in range(args.steps):trainer.step(train,labels)
    final=trainer.evaluate(train,labels)
    heldout=train.copy();heldout[:,::3]=0
    validation=trainer.evaluate(heldout,labels)
    if args.checkpoint:trainer.store(args.checkpoint)
    print(json.dumps(dict(backend=final['backend'],numeric_profile=final['numeric_profile'],
                          optimizer_step=trainer.state['step'],loss_before=initial,loss_after=final['loss'],
                          heldout_predictions=np.argmax(validation['logits'],axis=1).tolist(),
                          hidden_weights=trainer.state['weights'][0],output_weights=trainer.state['weights'][1])))


if __name__=='__main__':main()
