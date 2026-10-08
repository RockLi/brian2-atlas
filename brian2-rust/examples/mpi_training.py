"""Segmented native MPI STDP with boundary checkpoint and bounded growth.

Run once from Python, not under mpiexec. --resume builds the same identities in
a fresh process and resumes a complete checkpoint with the same rank policy.
"""
import argparse
import json
from pathlib import Path
import brian2 as b
import brian2_rust
import numpy as np


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--ranks',type=int,default=2)
    parser.add_argument('--rank-backends',default=None)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    gpu={} if args.rank_backends is None else dict(rank_backends=args.rank_backends.split(','),numeric_mode='mixed-f32')
    b.set_device('rust_standalone',engine='mpi',ranks=args.ranks,directory=args.directory,**gpu)
    b.seed(123)
    clock=b.Clock(dt=b.second/1024,name='learning_clock')
    group=b.NeuronGroup(3,'dv/dt=drive*1024*Hz:1\ndrive:1\nx:1',
                        threshold='v>1',reset='v=0',clock=clock,method='euler',name='learning_neurons')
    group.drive=[0.5,0.25,0.5]
    syn=b.Synapses(group,group,'w:1\nlive:integer\nborn:integer\nactive_after:second\n'
                   'dapre/dt=-apre/(8*ms):1 (event-driven)\n'
                   'dapost/dt=-apost/(11*ms):1 (event-driven)',
                   on_pre='apre+=live*0.125; w=clip(w+live*apost,0,1); x_post+=live*w*int(t>=active_after)',
                   on_post='apost-=live*0.0625; w=clip(w+live*apre,0,1)',
                   clock=clock,name='learning_synapses')
    syn.connect(i=[2,0,1,0,2,1],j=[0,2,0,2,1,2])
    syn.w=0.25;syn.live=1;syn.delay=np.array([3,4,1,2,5,3])*clock.dt
    spikes=b.SpikeMonitor(group,name='learning_spikes')
    net=b.Network(group,syn,spikes)
    if args.resume:
        net.restore('training',filename=args.checkpoint,restore_random_state=True)
    else:
        net.run(0*clock.dt)
        net.run(6*clock.dt)
        # Snapshot includes events still awaiting their cross-rank delays.
        net.store('training',filename=args.checkpoint)
    # Host updates, including weights/readout consolidation, are exported on
    # every segment. A candidate generation starts only at this boundary.
    group.drive=[0.25,0.5,0.25]
    net.run(7*clock.dt)
    print(json.dumps(dict(time_seconds=float(net.t/b.second),weights=np.asarray(syn.w[:]).tolist(),
                         spikes=len(spikes.i),pending_events=sum(map(len,b.get_device()._pending_events.values())))))


if __name__=='__main__':main()
