"""Verify native-trained MNIST on the test split, AtlasIR and stock SpiNNaker2."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import brian2 as b
import numpy as np
from brian2_atlas import NativeLIFTrainer, export_network
from brian2_atlas.neuromorphic import export_native_lif
from brian2_rust.native_lif_conversion import build_native_lif_network, validate_frozen
from brian2_rust.results import load_results
from train import DATASET_SHA256, preprocess, raster


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def native(trainer,x,y):
    spikes=[];logits=[];membrane=[]
    for k in range(0,len(x),128):
        result=trainer.evaluate(x[k:k+128],y[k:k+128])
        spikes.append(np.asarray(result['spikes'],dtype=np.uint8));logits.extend(result['logits']);membrane.extend(result['final_membrane'])
    return dict(spikes=np.concatenate(spikes),logits=np.asarray(logits),final_membrane=np.asarray(membrane))


def accuracy(logits,y):
    pred=np.argmax(logits,axis=1);conf=np.zeros((10,10),dtype=int)
    np.add.at(conf,(y,pred),1)
    return dict(correct=int(np.count_nonzero(pred==y)),total=len(y),accuracy=float(np.mean(pred==y)),confusion_matrix=conf.tolist())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--trained',type=Path,required=True)
    p.add_argument('--runner',type=Path,required=True)
    p.add_argument('--train-runner',type=Path,required=True)
    p.add_argument('--sdk-python',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--limit',type=int,default=10000)
    a=p.parse_args();a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False)
    assert 1<=a.limit<=10000 and sha(a.dataset)==DATASET_SHA256
    f=json.loads((a.trained/'frozen.json').read_text());validate_frozen(f)
    training=json.loads((a.trained/'training.json').read_text())
    assert sha(a.trained/'frozen.json')==training['frozen_file_sha256']
    checkpoint=json.loads(json.loads((a.trained/'checkpoint.json').read_text())['payload'])
    float_trainer=NativeLIFTrainer(checkpoint['plan'],runner=a.train_runner)
    float_trainer.restore(a.trained/'checkpoint.json')
    quantized=NativeLIFTrainer(f['plan'],weights=f['weights'],runner=a.train_runner)
    data=np.load(a.dataset,allow_pickle=False);x=raster(preprocess(data['x_test'][:a.limit]));y=data['y_test'][:a.limit]
    np.savez_compressed(a.output/'inputs.npz',spikes=x)
    before=native(float_trainer,x,y);after=native(quantized,x,y)
    np.savez_compressed(a.output/'native.npz',**after,float_logits=before['logits'])
    print('Native inference finished:',accuracy(before['logits'],y)['accuracy'],accuracy(after['logits'],y)['accuracy'],flush=True)
    b.prefs.codegen.target='numpy';b.prefs.codegen.loop_invariant_optimisations=False
    bundle=export_native_lif(f,a.output/'export',steps=x.shape[1],runner=a.runner)
    # Concrete native AtlasIR replay on a declared subset; full-set native SNN
    # and SDK comparison below is separate from this bounded IR replay.
    replay_samples=min(64,len(x));atlas_spikes=[];atlas_voltages=[];receipts=[]
    cells=sum(f['plan']['sizes'][1:]);steps=x.shape[1]
    for k in range(0,replay_samples,8):
        chunk=x[k:min(k+8,replay_samples)];dest=a.output/f'atlas-{k:05d}';dest.mkdir()
        model=export_network(build_native_lif_network(f,chunk),steps*b.ms,dest/'source.json')
        subprocess.run([str(a.runner.resolve()),str(dest/'source.json'),str(dest/'results')],check=True,capture_output=True)
        state=load_results(model,dest/'results')['populations'][bundle['plan']['lif_population']]
        sp=np.zeros((len(chunk),steps,cells),dtype=np.uint8)
        idx=np.asarray(state['indices'],dtype=int);ticks=np.asarray(state['spike_ticks'],dtype=int)
        sp[idx//cells,ticks,idx%cells]=1
        atlas_spikes.append(sp);atlas_voltages.append(state['trace']['v'].reshape(steps,len(chunk),cells).transpose(1,0,2))
        receipts.append(dict(first_test_index=k,samples=len(chunk),source_sha256=sha(dest/'source.json')))
    atlas_spikes=np.concatenate(atlas_spikes);atlas_voltages=np.concatenate(atlas_voltages)
    np.testing.assert_array_equal(atlas_spikes,after['spikes'][:replay_samples])
    print('Native AtlasIR replay finished:',replay_samples,'images',flush=True)
    runtime=Path(__file__).resolve().parents[3]/'python/brian2_rust/neuromorphic_runtime.py'
    worker=Path(__file__).with_name('sdk_worker.py')
    with (a.output/'sdk.log').open('w') as log:
        subprocess.run([str(a.sdk_python.absolute()),str(worker),'--runtime',str(runtime),'--bundle',str(a.output/'export/converted/bundle.json'),
            '--inputs',str(a.output/'inputs.npz'),'--output',str(a.output/'sdk')],cwd=a.output,
            env={**os.environ,'PYTHONPATH':'','PYTHONNOUSERSITE':'1'},stdout=log,stderr=subprocess.STDOUT,check=True)
    sdk=np.load(a.output/'sdk/results.npz',allow_pickle=False)
    np.testing.assert_array_equal(sdk['spike_raster'],after['spikes'])
    native_counts=after['spikes'][:,:,-10:].sum(1,dtype=np.int64)
    np.testing.assert_array_equal(sdk['counts'],native_counts)
    np.testing.assert_array_equal(np.argmax(sdk['logits'],axis=1),np.argmax(after['logits'],axis=1))
    np.testing.assert_allclose(sdk['voltage_f64'][:replay_samples],atlas_voltages,rtol=0,atol=1e-12)
    # Restore the native end-of-step state using the declared state transform.
    sizes=f['plan']['sizes'];last=after['spikes'][:,-1];currents=[];offset=0
    for l,(a_size,b_size,w) in enumerate(zip(sizes,sizes[1:],f['weights'])):
        pre=x[:,-1] if l==0 else last[:,offset-sizes[l]:offset]
        currents.append(pre@np.asarray(w).reshape(a_size,b_size));offset+=b_size
    reconstructed=sdk['final_v']/np.repeat(f['scales'],sizes[1:])+np.concatenate(currents,axis=1)
    np.testing.assert_allclose(reconstructed,after['final_membrane'],rtol=0,atol=1e-12)
    report=dict(schema='atlas.native-lif.mnist-audit.v1',passed=True,dataset_sha256=DATASET_SHA256,
        training=training,full_official_test_split=a.limit==10000,
        float_native=accuracy(before['logits'],y),quantized_native=accuracy(after['logits'],y),sdk=accuracy(sdk['logits'],y),
        quantization_accuracy_change_percentage_points=100*(accuracy(after['logits'],y)['accuracy']-accuracy(before['logits'],y)['accuracy']),
        conversion_prediction_differences=0,spike_raster_values=int(after['spikes'].size),spike_raster_differences=0,
        readout_count_values=int(native_counts.size),readout_count_differences=0,
        final_native_state_values=int(reconstructed.size),max_native_state_error=float(np.max(np.abs(reconstructed-after['final_membrane']))),
        atlas_ir_replay_samples=replay_samples,atlas_ir_voltage_values=int(atlas_voltages.size),
        atlas_sdk_voltage_max_error=float(np.max(np.abs(sdk['voltage_f64'][:replay_samples]-atlas_voltages))),
        atlas_ir_receipts=receipts,frozen_sha256=f['sha256'],frozen_file_sha256=sha(a.trained/'frozen.json'),
        checkpoint_sha256=sha(a.trained/'checkpoint.json'),input_file_sha256=sha(a.output/'inputs.npz'),
        native_results_sha256=sha(a.output/'native.npz'),atlas_runner_sha256=sha(a.runner),
        sdk_execution=json.loads((a.output/'sdk/execution.json').read_text()),
        physical_hardware=False,hardware_mapping_tested=False,
        runtime_source_sha256={p.name:sha(p) for p in [runtime,runtime.with_name('spinnaker2_runtime.py'),runtime.with_name('native_lif_conversion.py')]})
    (a.output/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k in ('passed','float_native','quantized_native','sdk','quantization_accuracy_change_percentage_points','spike_raster_values','max_native_state_error')}),flush=True)


if __name__=='__main__':main()
