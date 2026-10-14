"""Verify held-out native integer inference against XyloSim and AtlasIR."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import brian2 as b
import numpy as np
from brian2_atlas import NativeXyloTrainer, export_network
from brian2_atlas.neuromorphic import export_native_xylo
from brian2_rust.native_xylo_conversion import build_native_xylo_network, validate_frozen
from brian2_rust.results import load_results
from train import DATASET_SHA256, preprocess, raster


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def native(trainer, x, y):
    arrays={k:[] for k in ('spikes','logits','currents','membranes')}
    for i in range(0,len(x),64):
        result=trainer.evaluate(x[i:i+64],y[i:i+64])
        for key in arrays:arrays[key].append(np.asarray(result[key]))
    return {key:np.concatenate(parts) for key,parts in arrays.items()}


def accuracy(logits,y):
    prediction=np.argmax(logits,axis=1);confusion=np.zeros((4,4),int)
    np.add.at(confusion,(y,prediction),1)
    return dict(correct=int(np.count_nonzero(prediction==y)),total=len(y),
                accuracy=float(np.mean(prediction==y)),confusion_matrix=confusion.tolist())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','trained','runner','train-runner','xylo-python','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False)
    assert sha(a.dataset)==DATASET_SHA256
    f=json.loads((a.trained/'frozen.json').read_text());sizes=validate_frozen(f)
    training=json.loads((a.trained/'training.json').read_text());steps=training['ticks']
    assert sha(a.trained/'frozen.json')==training['frozen_sha256']
    assert sha(a.trained/'checkpoint.json')==training['checkpoint_sha256']
    learned=NativeXyloTrainer(f['plan'],runner=a.train_runner);learned.restore(a.trained/'checkpoint.json')
    frozen=NativeXyloTrainer(f['plan'],runner=a.train_runner,
        weights=[np.asarray(w)/f['plan']['weight_scale'] for w in f['target_weights']])
    data=np.load(a.dataset,allow_pickle=False);indices=np.flatnonzero(data['y_test']<4)
    x=raster(preprocess(data['x_test'][indices],training['encoding']),steps);y=data['y_test'][indices]
    np.savez_compressed(a.output/'inputs.npz',spikes=x)
    before=native(learned,x,y);after=native(frozen,x,y)
    for key in before:np.testing.assert_array_equal(before[key],after[key])
    del after
    print('Native held-out accuracy:',accuracy(before['logits'],y),flush=True)
    b.prefs.codegen.target='numpy';b.prefs.codegen.loop_invariant_optimisations=False
    bundle=export_native_xylo(f,a.output/'xylosim',steps=steps,runner=a.runner)
    assert bundle['native_training_bridge']['dynamics_equivalent'] is True
    for name,bank,shape in zip(('input','readout'),f['target_weights'],zip(sizes,sizes[1:])):
        np.testing.assert_array_equal(bundle['plan']['weights'][name],np.asarray(bank).reshape(shape))
    runtime=Path(__file__).resolve().parents[3]/'python/brian2_rust/neuromorphic_runtime.py'
    worker=Path(__file__).resolve().parents[1]/'native_xylo/sdk_worker.py'
    output=a.output/'xylosim/sdk'
    with (a.output/'xylosim.log').open('w') as log:
        subprocess.run([str(a.xylo_python.absolute()),str(worker),'--runtime',str(runtime),
            '--bundle',str(a.output/'xylosim/converted/bundle.json'),'--inputs',str(a.output/'inputs.npz'),
            '--output',str(output)],cwd=a.output,env={**os.environ,'PYTHONPATH':'','PYTHONNOUSERSITE':'1'},
            stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
    execution=json.loads((output/'execution.json').read_text())
    assert execution['official_configuration_valid']
    assert execution['inputs_sha256']==sha(a.output/'inputs.npz')
    assert execution['results_sha256']==sha(output/'results.npz')
    xy=np.load(output/'results.npz',allow_pickle=False)
    checks={}
    for field,target in (('spikes','spike_raster'),('membranes','voltage'),('logits','logits')):
        np.testing.assert_array_equal(before[field],xy[target])
        checks[field]=dict(values=int(before[field].size),differences=0)
    np.testing.assert_array_equal(before['currents'],np.concatenate((xy['hidden_I'],xy['output_I']),axis=2))
    checks['currents']=dict(values=int(before['currents'].size),differences=0)
    replay=32;receipts=[];state_values=0
    for i in range(replay):
        dest=a.output/'replay'/f'{i:03d}';dest.mkdir(parents=True)
        model=export_network(build_native_xylo_network(f,x[i]),steps*b.ms,dest/'source.json')
        subprocess.run([str(a.runner.resolve()),str(dest/'source.json'),str(dest/'results')],check=True,capture_output=True)
        recorded=load_results(model,dest/'results')
        for role in ('hidden','output'):
            pop=recorded['populations'][bundle['plan']['roles'][role]]
            for key in ('I','v','q'):
                np.testing.assert_array_equal(pop['trace'][key],xy[role+'_'+key][i]);state_values+=pop['trace'][key].size
            np.testing.assert_array_equal(np.column_stack((pop['spike_ticks'],pop['indices'])),
                                          np.column_stack(np.nonzero(xy[role+'_q'][i])))
        receipts.append(dict(original_test_index=int(indices[i]),source_sha256=sha(dest/'source.json'),
                             results_sha256=sha(dest/'results/results.bin'),events_sha256=sha(dest/'results/events.bin')))
    np.savez_compressed(a.output/'per_sample.npz',original_test_index=indices,labels=y,
        native_prediction=np.argmax(before['logits'],1),xylo_prediction=np.argmax(xy['logits'],1),
        native_counts=before['spikes'][:,:,-sizes[-1]:].sum(1),xylo_counts=xy['counts'])
    source_files=[runtime,runtime.with_name('xylo_runtime.py'),runtime.with_name('training_xylo.py'),
        runtime.with_name('native_xylo_conversion.py'),runtime.with_name('xylo_network.py'),
        runtime.parents[2]/'src/xylo_training.rs',Path(__file__),worker]
    report=dict(schema='atlas.native-xylo.integer-comparison.v1',validation_passed=True,
        dynamics_equivalent=True,same_integer_weight_banks=True,freeze_differences=0,
        dataset_sha256=DATASET_SHA256,classes=[0,1,2,3],full_official_test_subset=True,full_ten_class_mnist=False,
        training=training,native=accuracy(before['logits'],y),xylosim=accuracy(xy['logits'],y),
        native_xylosim=checks,prediction_differences=0,timing_alignment='none',
        atlas_ir_replay=dict(samples=replay,state_values=state_values,differences=0,receipts=receipts),
        observed=dict(max_hidden_spikes_per_tick=int(xy['hidden_q'].max()),
            min_current=float(before['currents'].min()),max_current=float(before['currents'].max()),
            min_membrane=float(before['membranes'].min()),max_membrane=float(before['membranes'].max())),
        frozen_sha256=f['sha256'],frozen_file_sha256=sha(a.trained/'frozen.json'),
        checkpoint_sha256=sha(a.trained/'checkpoint.json'),input_sha256=sha(a.output/'inputs.npz'),
        per_sample_sha256=sha(a.output/'per_sample.npz'),execution=execution,
        physical_hardware=False,hardware_mapping_tested=False,
        source_hashes={p.name:sha(p) for p in source_files})
    (a.output/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('validation_passed','native','xylosim','native_xylosim')}),flush=True)


if __name__=='__main__':main()
