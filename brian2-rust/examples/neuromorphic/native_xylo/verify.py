"""Compare one frozen model on native Atlas, SpiNNaker2 and adapted Xylo.

Only Atlas-to-SpiNNaker2 and adapted-AtlasIR-to-Xylo are asserted equivalent.
Cross-target differences are measured without shifts, calibration or relabeling.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

import brian2 as b
import numpy as np
from brian2_atlas import NativeLIFTrainer, export_network
from brian2_atlas.neuromorphic import export_native_lif
from brian2_rust.native_lif_conversion import build_native_lif_network, validate_frozen
from brian2_rust.native_lif_xylo import build_network as build_xylo
from brian2_rust.results import load_results
from train import DATASET_SHA256,preprocess,raster


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def native(trainer,x,y):
    results=[trainer.evaluate(x[i:i+128],y[i:i+128]) for i in range(0,len(x),128)]
    return {k:np.concatenate([np.asarray(r[k]) for r in results]) for k in ('spikes','logits','final_membrane')}


def accuracy(logits,y):
    prediction=np.argmax(logits,axis=1);confusion=np.zeros((4,4),int)
    np.add.at(confusion,(y,prediction),1)
    return dict(correct=int(np.count_nonzero(prediction==y)),total=len(y),accuracy=float(np.mean(prediction==y)),
                confusion_matrix=confusion.tolist())


def difference(a,b):
    assert a.shape==b.shape
    delta=a.astype(float)-b.astype(float)
    return dict(values=int(a.size),different_values=int(np.count_nonzero(delta)),
                max_abs_error=float(np.abs(delta).max()),mean_abs_error=float(np.abs(delta).mean()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','trained','runner','train-runner','spinnaker-python','xylo-python','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False)
    assert sha(a.dataset)==DATASET_SHA256
    f=json.loads((a.trained/'frozen.json').read_text());sizes=validate_frozen(f)
    training=json.loads((a.trained/'training.json').read_text())
    assert sha(a.trained/'frozen.json')==training['frozen_file_sha256']
    assert sha(a.trained/'checkpoint.json')==training['checkpoint_sha256']
    checkpoint=json.loads(json.loads((a.trained/'checkpoint.json').read_text())['payload'])
    learned=NativeLIFTrainer(checkpoint['plan'],runner=a.train_runner)
    learned.restore(a.trained/'checkpoint.json')
    quantized=NativeLIFTrainer(f['plan'],weights=f['weights'],runner=a.train_runner)
    data=np.load(a.dataset,allow_pickle=False);indices=np.flatnonzero(data['y_test']<4)
    x=raster(preprocess(data['x_test'][indices]));y=data['y_test'][indices]
    np.savez_compressed(a.output/'inputs.npz',spikes=x)
    before=native(learned,x,y);after=native(quantized,x,y)
    print('Native test accuracy',accuracy(before['logits'],y),accuracy(after['logits'],y),flush=True)
    b.prefs.codegen.target='numpy';b.prefs.codegen.loop_invariant_optimisations=False
    bundles={}
    for target in ('spinnaker2','xylosim'):
        bundles[target]=export_native_lif(f,a.output/target,steps=16,runner=a.runner,target=target,
                                         allow_dynamics_approximation=target=='xylosim')
        assert bundles[target]['native_training_bridge']['frozen_sha256']==f['sha256']
    for name,bank,shape in zip(('input','readout'),f['target_weights'],zip(sizes,sizes[1:])):
        np.testing.assert_array_equal(bundles['xylosim']['plan']['weights'][name],np.asarray(bank).reshape(shape))
    runtime=Path(__file__).resolve().parents[3]/'python/brian2_rust/neuromorphic_runtime.py'
    env={**os.environ,'PYTHONPATH':'','PYTHONNOUSERSITE':'1'}
    results={};executions={}
    for target,python in (('spinnaker2',a.spinnaker_python),('xylosim',a.xylo_python)):
        output=a.output/target/'sdk'
        with (a.output/(target+'.log')).open('w') as log:
            subprocess.run([str(python.absolute()),str(Path(__file__).with_name('sdk_worker.py')),
                '--runtime',str(runtime),'--bundle',str(a.output/target/'converted/bundle.json'),
                '--inputs',str(a.output/'inputs.npz'),'--output',str(output)],cwd=a.output,env=env,
                stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
        results[target]=np.load(output/'results.npz',allow_pickle=False)
        executions[target]=json.loads((output/'execution.json').read_text())
        assert executions[target]['inputs_sha256']==sha(a.output/'inputs.npz')
        assert executions[target]['results_sha256']==sha(output/'results.npz')
        print(target,'accuracy',accuracy(results[target]['logits'],y),flush=True)
    assert executions['xylosim']['official_configuration_valid']
    sp,xy=results['spinnaker2'],results['xylosim']
    np.testing.assert_array_equal(after['spikes'],sp['spike_raster'])
    np.testing.assert_array_equal(after['logits'],sp['logits'])
    currents=[];offset=0
    for l,(n,m,w) in enumerate(zip(sizes,sizes[1:],f['weights'])):
        pre=x[:,-1] if l==0 else after['spikes'][:,-1,offset-n:offset]
        currents.append(pre@np.asarray(w).reshape(n,m));offset+=m
    reconstructed=sp['final_v']/np.repeat(f['scales'],sizes[1:])+np.concatenate(currents,axis=1)
    np.testing.assert_array_equal(reconstructed,after['final_membrane'])
    # Concrete AtlasIR replays use the same held-out input indices for both paths.
    replay=32;receipts=[];xy_values=0;sp_values=0
    for i in range(replay):
        for target,build in (('spinnaker2',build_native_lif_network),('xylosim',build_xylo)):
            dest=a.output/'replay'/f'{target}-{i:03d}';dest.mkdir(parents=True)
            inputs=x[i:i+1] if target=='spinnaker2' else x[i]
            model=export_network(build(f,inputs),16*b.ms,dest/'source.json')
            subprocess.run([str(a.runner.resolve()),str(dest/'source.json'),str(dest/'results')],
                           check=True,capture_output=True)
            recorded=load_results(model,dest/'results');plan=bundles[target]['plan']
            if target=='spinnaker2':
                pop=recorded['populations'][plan['lif_population']]
                np.testing.assert_array_equal(pop['trace']['v'],sp['voltage_f64'][i]);sp_values+=pop['trace']['v'].size
                events=np.column_stack((pop['spike_ticks'],pop['indices']))
                np.testing.assert_array_equal(events,np.column_stack(np.nonzero(sp['spike_raster'][i])))
            else:
                for role in ('hidden','output'):
                    pop=recorded['populations'][plan['roles'][role]]
                    for key in ('I','v','q'):
                        np.testing.assert_array_equal(pop['trace'][key],xy[role+'_'+key][i]);xy_values+=pop['trace'][key].size
                    np.testing.assert_array_equal(np.column_stack((pop['spike_ticks'],pop['indices'])),
                                                  np.column_stack(np.nonzero(xy[role+'_q'][i])))
            receipts.append(dict(target=target,original_test_index=int(indices[i]),source_sha256=sha(dest/'source.json'),
                                 results_sha256=sha(dest/'results/results.bin'),events_sha256=sha(dest/'results/events.bin')))
    cross=dict(spike_counts=difference(sp['spike_raster'],xy['spike_raster']),
        spike_presence=difference(sp['spike_raster']>0,xy['spike_raster']>0),
        membrane_target_units=difference(sp['voltage_f64'],xy['voltage']),
        readout_counts=difference(sp['counts'],xy['counts']),
        prediction_differences=int(np.count_nonzero(np.argmax(sp['logits'],1)!=np.argmax(xy['logits'],1))),
        spinnaker_spike_count=int(sp['spike_raster'].sum()),xylo_spike_count=int(xy['spike_raster'].sum()),
        xylo_max_hidden_spikes_per_tick=int(xy['hidden_q'].max()),timing_alignment='none',
        extra_xylo_current_state='recorded and checked against adapted AtlasIR; absent in native LIF')
    predictions=np.stack((indices,y,np.argmax(before['logits'],1),np.argmax(after['logits'],1),
                          np.argmax(sp['logits'],1),np.argmax(xy['logits'],1)),axis=1)
    np.savez_compressed(a.output/'per_sample.npz',predictions=predictions,
        native_counts=sp['counts'],xylo_counts=xy['counts'],
        spike_count_differences=np.count_nonzero(sp['spike_raster']!=xy['spike_raster'],axis=(1,2)),
        max_membrane_difference=np.abs(sp['voltage_f64']-xy['voltage']).max(axis=(1,2)))
    report=dict(schema='atlas.native-lif.xylo-comparison.v1',validation_passed=True,
        cross_target_equivalent=False,same_frozen_artifact=True,same_integer_weight_banks=True,
        same_transition_function=False,xylo_adaptation=bundles['xylosim']['native_training_bridge']['adaptation'],
        dataset_sha256=DATASET_SHA256,classes=[0,1,2,3],full_official_test_subset=True,
        full_ten_class_mnist=False,training=training,
        native_float=accuracy(before['logits'],y),native_quantized=accuracy(after['logits'],y),
        spinnaker2=accuracy(sp['logits'],y),xylosim=accuracy(xy['logits'],y),cross_target=cross,
        native_spinnaker2=dict(spike_values=int(after['spikes'].size),spike_differences=0,
            final_state_values=int(reconstructed.size),final_state_differences=0),
        atlas_ir_replay=dict(samples_per_target=replay,spinnaker2_voltage_values=sp_values,
            xylo_state_values=xy_values,differences=0,receipts=receipts),
        frozen_sha256=f['sha256'],frozen_file_sha256=sha(a.trained/'frozen.json'),
        checkpoint_sha256=sha(a.trained/'checkpoint.json'),input_sha256=sha(a.output/'inputs.npz'),
        per_sample_sha256=sha(a.output/'per_sample.npz'),executions=executions,
        physical_hardware=False,hardware_mapping_tested=False,
        source_hashes={p.name:sha(p) for p in (runtime,runtime.with_name('native_lif_conversion.py'),
            runtime.with_name('native_lif_xylo.py'),runtime.with_name('xylo_runtime.py'),
            runtime.with_name('spinnaker2_runtime.py'),Path(__file__),
            Path(__file__).with_name('sdk_worker.py'))})
    (a.output/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k in ('validation_passed','native_float','native_quantized','spinnaker2','xylosim','cross_target')}),flush=True)


if __name__=='__main__':main()
