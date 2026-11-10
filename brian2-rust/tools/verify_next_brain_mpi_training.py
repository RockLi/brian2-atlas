"""Read-only compatibility probe against a supplied Next Brain checkout.

Uses its actual builders/present/consolidation/readout/freeze/structural_step;
only the isolated Device configuration is changed to MPI after initialization.
No platform files, installed Atlas packages or running services are modified.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import tempfile


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--platform-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--families',default=None)
    parser.add_argument('--structure',choices=['fixed','random-rewire','weight'],default='random-rewire')
    parser.add_argument('--dynamics',choices=['lif','adex','izhikevich','custom'],default='lif')
    args=parser.parse_args()
    sys.path.insert(0,str(args.platform_root/'model-platform/backend'))
    import brian2 as b
    import numpy as np
    import brian2_rust.distributed as distributed
    from app.training_engine import (build_network,present,learned_state,portable_store,
        portable_restore,learn_readout,frozen,assert_frozen,structural_step)
    compiler=distributed.compile_mpi_project
    distributed.compile_mpi_project=lambda path:compiler(path,opt_level=0)
    common=dict(neurons=16,membraneTauMs=20,synapticGain=4,stdpRate=.001,
                inhibition=2,delayMs=1,learningRate=.05,l2=.0001)
    cases=[('diehl-cook',dict(neurons=16,membraneTauMs=50,thresholdIncrementMv=.02)),
           ('reservoir',dict(inputProbability=.3,recurrentProbability=.1,excitatoryFraction=.8,recurrentGain=.5)),
           ('local-receptive',dict(neurons=4,receptiveField=3,stride=2)),
           ('feedforward',dict(hiddenWidths=[8],connectionProbability=.5)),
           ('convolutional',dict(channels=[2],kernelSize=2,stride=1,pooling='none',poolSize=2)),
           ('recurrent',dict(hiddenWidths=[8],connectionProbability=.5,recurrentProbability=.2,recurrentGain=.2))]
    results=[]
    with tempfile.TemporaryDirectory(prefix='b2-next-brain-mpi-') as directory:
        for family,options in cases:
            if args.families and family not in args.families.split(','):continue
            config=dict(profile='compact',inputNeurons=64,seed=7,encoding='poisson',
                        intensity=300,stimulusMs=60,restMs=2,sampleState='reset',
                        _classes=2,_shape=[8,8,1],network=dict(common,family=family,**options))
            # Enable the exact bounded-candidate fields for every projection.
            config['network']['structure']=dict(mode=args.structure,initialDensity=.7,maxDensity=.9,
                intervalSamples=1,pruneFraction=.1,growthFraction=.05,growthWeight=.5,
                graceSamples=0,pruneBelow=.1)
            if args.dynamics!='lif':
                config['network']['neuron']=dict(kind=args.dynamics,
                    voltageEquation='(-65-x-y)/tau',recoveryEquation='-y/100')
            os.environ['NEXT_BRAIN_TRAINING_WORKDIR']=str(Path(directory)/family)
            os.environ['NEXT_BRAIN_TRAINING_ENGINE']='reference'
            row=dict(family=family,ranks=2,backend='cpu',structure=args.structure,dynamics=args.dynamics)
            try:
                net=build_network(config)
                b.get_device().build_options.update(engine='mpi',ranks=2)
                first=present(net,config,np.ones(64))
                structural_step(net,config,1)
                path=Path(directory)/(family+'.checkpoint')
                portable_store(net,path)
                before=learned_state(net)
                frozen(net,True);present(net,config,np.ones(64));assert_frozen(net,before)
                portable_restore(net,path)
                sample=np.linspace(.2,1,64)
                counts=present(net,config,sample);learn_readout(net,config,counts,1,2)
                expected=learned_state(net)
                portable_restore(net,path)
                replay=present(net,config,sample);learn_readout(net,config,replay,1,2)
                assert np.array_equal(counts,replay)
                assert all(np.array_equal(value,learned_state(net)[key]) for key,value in expected.items())
                row.update(status='passed',first_spikes=int(first.sum()),replay_spikes=int(replay.sum()),
                           clock_seconds=float(net.t/b.second),learned_fields=sorted(expected))
            except Exception as error:
                row.update(status='failed',error=str(error))
            results.append(row)
            print(json.dumps(row),flush=True)
    args.output.write_text(json.dumps(results,indent=2)+'\n')
    if any(row['status']!='passed' for row in results):raise SystemExit(1)


if __name__=='__main__':main()
