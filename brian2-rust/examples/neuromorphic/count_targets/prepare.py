"""Convert the existing common AtlasIR and prepare simulator-only inputs."""
import argparse
import os
from pathlib import Path
import numpy as np
from brian2_atlas.neuromorphic import export_ir


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('common','output','runner'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    os.environ.update(B2_MAX_POPULATION_PARAMETERS='1024',B2_MAX_POPULATION_STATES='512',B2_MAX_STATEMENTS='512')
    for target in ('spinnaker2','xylosim'):
        bundle=export_ir(a.common/'source.json',a.output/target,target=target,
                         profile='binary-count-v1',runner=a.runner)
        print(target,bundle['plan']['resources'])
    with np.load(a.common/'inputs.npz',allow_pickle=False) as data:
        x=data['inputs'][:20]
    with np.load(a.common/'probes/inputs.npz',allow_pickle=False) as data:
        probes=data['inputs']
    np.savez_compressed(a.output/'smoke.npz',inputs=np.concatenate([x,probes]))


if __name__=='__main__':main()
