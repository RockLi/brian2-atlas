"""Bounded fixed-total E/I construction pilot, not the scientific MAM model.

Four recurrent projections use clipped-normal weights and heterogeneous delays.
The export remains a recipe. All local topology/parameter construction is Rust.
"""
import argparse
import json
from pathlib import Path
import sys

import brian2 as b
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))
import brian2_rust
from brian2_rust.export import lower_network
from brian2_rust.distributed import write_mpi_project,compile_mpi_project


def make_model(edges_per_projection,steps=128):
    clock=b.Clock(dt=0.1*b.ms)
    equation='dv/dt=(drive-v+current)/(10*ms):1 (unless refractory)\ndcurrent/dt=-current/(5*ms):1\ndrive:1 (constant)'
    pops=[]
    for name,n in [('exc',800),('inh',200)]:
        p=b.NeuronGroup(n,equation,threshold='v>1',reset='v=0',refractory=0.2*b.ms,method='euler',clock=clock,name=name)
        p.v=np.linspace(0,0.95,n);p.drive=np.linspace(1.2,1.8,n);pops.append(p)
    synapses=[]
    for source,a in enumerate(pops):
        for target,c in enumerate(pops):
            syn=b.Synapses(a,c,'w:1 (constant)',on_pre='current_post += w',clock=clock,name=f'projection_{source}_{target}')
            sign=1 if source==0 else -1
            # Keep expected drive comparable while changing edge count.
            mean=sign*0.01*65536/edges_per_projection
            weight=brian2_rust.ClippedNormal(mean,abs(mean)*0.1,minimum=0.0 if sign==1 else None,maximum=0.0 if sign==-1 else None)
            delay=brian2_rust.ClippedNormal((1.5 if source==0 else 0.75)*b.ms,0.25*b.ms,minimum=0.1*b.ms,maximum=3*b.ms)
            brian2_rust.connect_fixed_total(syn,edges_per_projection,seed=1729+source*2+target,initializers={'w':weight},delay_initializer=delay)
            synapses.append(syn)
    monitors=[b.SpikeMonitor(p) for p in pops]+[b.StateMonitor(p,['v','current'],record=[0,len(p)-1]) for p in pops]
    return lower_network(b.Network(*pops,*synapses,*monitors),steps*clock.dt,rng_seed=1729)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--edges-per-projection',type=int,nargs='+',default=[65536,1048576])
    p.add_argument('--mpicc',default='mpicc');p.add_argument('--rustc',default='rustc')
    a=p.parse_args()
    if any(not 1<=n<=1048576 for n in a.edges_per_projection):p.error('pilot limited to 1,048,576 edges per projection')
    a.output.mkdir(parents=True,exist_ok=False)
    report={'schema':'b2-mpi-procedural-pilot-v1','scientific_scope':'synthetic E/I construction pilot; not MAM','models':[],'complete':False}
    for edges in a.edges_per_projection:
        b.get_device().reinit();b.set_device('atlas',runner=ROOT/'target/release/b2-runner')
        model=make_model(edges);folder=a.output/f'edges-{edges*4}';folder.mkdir()
        (folder/'model.json').write_text(json.dumps(model)+'\n')
        row={'global_edges':edges*4,'neurons':1000,'projects':{}}
        for ranks in [1,2,4]:
            project=folder/f'rank-{ranks}'
            write_mpi_project(model,project,ranks=ranks)
            compile_mpi_project(project,mpicc=a.mpicc,rustc=a.rustc)
            row['projects'][str(ranks)]={'instance_bytes':[x.stat().st_size for x in sorted(project.glob('instance.rank-*.bin'))],'executable_bytes':(project/'b2-mpi').stat().st_size}
        report['models'].append(row)
    report['complete']=True;(a.output/'build-report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
