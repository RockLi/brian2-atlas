"""Train shared convolution, recurrent LIF and readout parameters natively."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
from brian2_rust import (NativeLIFTrainer,lif_training_plan,
                        dense_training_projection,conv2d_training_projection)


def demo(backend='cpu', *, equations=False, mpi_ranks=None):
    convolution,shape=conv2d_training_projection(0,1,(2,2,2),2,1)
    recurrent=dense_training_projection(1,1,8,8)
    readout=dense_training_projection(1,2,8,2)
    masks=[[1.]*4,np.eye(8).ravel().tolist(),[1.]*16]
    plan=lif_training_plan([8,8,2],backend=backend,
                          projections=[convolution,recurrent,readout],masks=masks,
                          beta=.8,learning_rate=.02,surrogate_slope=3,logit_scale=3)
    weights=[[1.2,.4,.4,1.2],(np.eye(8)*.05).ravel().tolist(),
             np.array([[.05,.2]]*4+[[.2,.05]]*4).ravel().tolist()]
    if equations:
        from brian2_rust import compile_training_equation,neuron_parameter_bank
        projections=[convolution,recurrent,readout,neuron_parameter_bank(2)]
        programs=[compile_training_equation('v*(1-dt/(.1+exp(log_tau)))',
                  parameters={'dt':.1,'log_tau':(3,l)}) for l in range(2)]
        plan=lif_training_plan([8,8,2],backend=backend,projections=projections,
             masks=[*masks,[1.,1.]],equations=programs,mpi_ranks=mpi_ranks,
             learning_rate=.02,surrogate_slope=3,logit_scale=3)
        weights.append([float(np.log(.4))]*2)
    elif mpi_ranks is not None:plan['mpi_ranks']=mpi_ranks
    inputs=np.zeros((4,24,8));inputs[:2,:,:4]=1;inputs[2:,:,4:]=1
    heldout=inputs.copy();heldout[:,::3]=0;heldout[:,:,[0,4]]=0
    return plan,weights,inputs,[0,0,1,1],heldout


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--backend',choices=['cpu','metal','cuda'],default='cpu')
    parser.add_argument('--equations',action='store_true')
    parser.add_argument('--mpi-ranks',type=int)
    parser.add_argument('--steps',type=int,default=80)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    plan,weights,x,y,heldout=demo(args.backend,equations=args.equations,mpi_ranks=args.mpi_ranks)
    trainer=NativeLIFTrainer(plan,weights=weights)
    if args.resume:
        if args.checkpoint is None:parser.error('--resume requires --checkpoint')
        trainer.restore(args.checkpoint)
    before=trainer.evaluate(x,y)['loss'];initial=copy.deepcopy(trainer.state)
    for _ in range(args.steps):trainer.step(x,y)
    final=trainer.evaluate(x,y)
    state=copy.deepcopy(trainer.state);validation=trainer.evaluate(heldout,y)
    assert trainer.state==state
    if args.checkpoint:trainer.store(args.checkpoint)
    print(json.dumps(dict(backend=final['backend'],numeric_profile=final['numeric_profile'],
                          gpu_dispatches=final['gpu_dispatches'],optimizer_step=trainer.state['step'],
                          schema=plan['schema'],parameter_counts=list(map(len,trainer.state['weights'])),
                          edge_counts=[len(p['sources']) for p in plan['projections']],
                          loss_before=before,loss_after=final['loss'],
                          heldout_predictions=np.argmax(validation['logits'],axis=1).tolist(),
                          banks_changed=[a!=b for a,b in zip(initial['weights'],trainer.state['weights'])],
                          weights=trainer.state['weights'])))


if __name__=='__main__':main()
