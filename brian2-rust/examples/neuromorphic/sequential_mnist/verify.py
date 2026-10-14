"""Frozen full-test MNIST accuracy and exact native/Xylo/custom-C trace audit."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import numpy as np
import brian2 as b
from brian2_atlas import NativeXyloTrainer, export_network
from brian2_atlas.neuromorphic import export_native_xylo
from brian2_rust.native_xylo_conversion import validate_frozen, build_native_xylo_network
from brian2_rust.results import load_results
from train import DATASET_SHA256, encode, sha
from sdk_worker import integer_trace


def accuracy(logits,y):
    pred=np.argmax(logits,axis=1);confusion=np.zeros((10,10),dtype=int)
    np.add.at(confusion,(y,pred),1)
    return dict(correct=int(np.count_nonzero(pred==y)),total=len(y),accuracy=float(np.mean(pred==y)),
        confusion_matrix=confusion.tolist(),ties=int(np.count_nonzero((logits==logits.max(1,keepdims=True)).sum(1)>1)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('dataset','trained','runner','train-runner','xylo-python','spinnaker2-python','output'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--portable-train-runner',type=Path,help='also compare a separately built native trainer on every test image')
    p.add_argument('--frozen-only',action='store_true',help='skip latent-checkpoint comparison when using a different trainer binary')
    a=p.parse_args();a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False)
    assert sha(a.dataset)==DATASET_SHA256
    training=json.loads((a.trained/'training.json').read_text())
    f=json.loads((a.trained/'frozen.json').read_text());sizes=validate_frozen(f)
    assert sha(a.trained/'frozen.json')==training['frozen_file_sha256']
    with np.load(a.dataset,allow_pickle=False) as data:
        x=encode(data['x_test'],training['encoding'],training['repeats']);y=data['y_test']
    assert len(y)==10000 and set(y)==set(range(10))
    assert training['test_arrays_read'] is False and training['training_samples']==54000
    assert training['validation_samples']==6000 and x.shape[1]==training['steps']
    np.savez_compressed(a.output/'inputs.npz',spikes=x)
    weights=[np.asarray(w)/f['plan']['weight_scale'] for w in f['target_weights']]
    trainer=NativeXyloTrainer(f['plan'],runner=a.train_runner,weights=weights)
    portable=(NativeXyloTrainer(f['plan'],runner=a.portable_train_runner,weights=weights)
        if a.portable_train_runner else None)
    portable_values=0
    trained=None
    if not a.frozen_only:
        trained=NativeXyloTrainer(f['plan'],runner=a.train_runner)
        trained.restore(a.trained/'checkpoint.json')
    zero=[w.copy() for w in weights];zero[2][:]=0
    no_rec=NativeXyloTrainer(f['plan'],runner=a.train_runner,weights=zero)
    logits=[];ablated=[];reset_logits=[];replay_indices=[int(np.flatnonzero(y==c)[0]) for c in range(10)]
    replay_reference={};freeze_values=0
    (a.output/'native').mkdir()
    boundary=f['plan']['readout_start'] or x.shape[1]//2
    for i in range(0,len(x),64):
        raster=x[i:i+64];labels=y[i:i+64]
        result=trainer.evaluate(raster,labels)
        # Frozen integer execution must also equal the selected latent checkpoint.
        arrays={k:(np.asarray(result[k],dtype=np.float64) if k=='logits' else integer_trace(result[k]))
            for k in ('spikes','currents','membranes','logits')}
        if trained is not None:
            before=trained.evaluate(raster,labels)
            for key in arrays:
                np.testing.assert_array_equal(arrays[key],before[key]);freeze_values+=arrays[key].size
        if portable is not None:
            other=portable.evaluate(raster,labels)
            for key in arrays:
                np.testing.assert_array_equal(arrays[key],other[key]);portable_values+=arrays[key].size
        np.savez_compressed(a.output/'native'/f'{i:05d}.npz',**arrays);logits.extend(arrays['logits'])
        ablated.extend(no_rec.evaluate(raster,labels)['logits'])
        prefix=trainer.evaluate(raster[:,:boundary],labels);state=copy.deepcopy(prefix['continuation'])
        for key in ('currents','membranes','spikes'):state[key]=np.zeros_like(state[key]).tolist()
        # Preserve any accumulated readout counts: change neural memory only.
        reset_logits.extend(trainer.evaluate(raster[:,boundary:],labels,initial=state)['logits'])
        for index in replay_indices:
            if i<=index<i+len(raster):replay_reference[index]={k:v[index-i].copy() for k,v in arrays.items()}
        if i%1024==0:print(json.dumps(dict(stage='native',samples=min(i+64,len(x)))),flush=True)
    logits=np.asarray(logits);ablated=np.asarray(ablated);reset_logits=np.asarray(reset_logits)
    print(json.dumps(dict(native=accuracy(logits,y),zero_recurrence=accuracy(ablated,y),reset_state=accuracy(reset_logits,y))),flush=True)
    runtime=Path(__file__).resolve().parents[3]/'python/brian2_rust/neuromorphic_runtime.py'
    b.prefs.codegen.target='numpy';b.prefs.codegen.loop_invariant_optimisations=False
    comparisons={};executions={};target_logits={}
    for target,python in [('xylosim',a.xylo_python),('spinnaker2',a.spinnaker2_python)]:
        bundle=export_native_xylo(f,a.output/target,steps=x.shape[1],runner=a.runner,target=target)
        for bank,role,shape in zip(f['target_weights'],('input','readout','recurrent'),
                ((sizes[0],sizes[1]),(sizes[1],sizes[2]),(sizes[1],sizes[1]))):
            np.testing.assert_array_equal(bundle['plan']['weights'][role],np.asarray(bank).reshape(shape))
        assert bundle['native_training_bridge']['class_codes']==f['plan']['class_codes']
        dest=a.output/(target+'-results')
        with (a.output/(target+'.log')).open('w') as log:
            subprocess.run([str(python.absolute()),str(Path(__file__).with_name('sdk_worker.py')),
                '--runtime',str(runtime),'--bundle',str(a.output/target/'converted/bundle.json'),
                '--inputs',str(a.output/'inputs.npz'),'--output',str(dest)],cwd=a.output,
                env={**os.environ,'PYTHONPATH':'','PYTHONNOUSERSITE':'1'},stdout=log,stderr=subprocess.STDOUT,check=True,timeout=3600)
        execution=json.loads((dest/'execution.json').read_text())
        assert execution['inputs_sha256']==sha(a.output/'inputs.npz') and execution['samples']==10000
        diffs={k:dict(values=0,differences=0,max_absolute_error=0.) for k in ('spikes','currents','membranes','logits')}
        predictions=[]
        for shard in execution['shards']:
            name=f"{shard['start']:05d}.npz";assert sha(dest/name)==shard['sha256']
            with np.load(a.output/'native'/name,allow_pickle=False) as ref, np.load(dest/name,allow_pickle=False) as actual:
                for key,item in diffs.items():
                    assert ref[key].shape==actual[key].shape
                    item['values']+=int(ref[key].size)
                    item['differences']+=int(np.count_nonzero(ref[key]!=actual[key]))
                    item['max_absolute_error']=max(item['max_absolute_error'],float(np.abs(ref[key]-actual[key]).max()))
                predictions.extend(actual['logits'])
        comparisons[target]=diffs;executions[target]=execution;target_logits[target]=np.asarray(predictions)
        assert all(v['differences']==0 for v in diffs.values()),diffs
    receipts=[];state_values=0
    for index in replay_indices:
        dest=a.output/'atlas-ir-replay'/str(index);dest.mkdir(parents=True)
        model=export_network(build_native_xylo_network(f,x[index]),x.shape[1]*b.ms,dest/'source.json')
        subprocess.run([str(a.runner.resolve()),str(dest/'source.json'),str(dest/'results')],capture_output=True,check=True)
        replay=load_results(model,dest/'results');ref=replay_reference[index]
        for role,sl in [('hidden',slice(0,sizes[1])),('output',slice(sizes[1],sum(sizes[1:])) )]:
            pop=replay['populations'][bundle['plan']['roles'][role]]
            for key,field in [('I','currents'),('v','membranes'),('q','spikes')]:
                np.testing.assert_array_equal(pop['trace'][key],ref[field][:,sl]);state_values+=pop['trace'][key].size
            np.testing.assert_array_equal(np.column_stack((pop['spike_ticks'],pop['indices'])),np.column_stack(np.nonzero(ref['spikes'][:,sl])))
        receipts.append(dict(test_index=index,label=int(y[index]),source_sha256=sha(dest/'source.json'),
            results_sha256=sha(dest/'results/results.bin'),events_sha256=sha(dest/'results/events.bin')))
    np.savez_compressed(a.output/'per_sample.npz',test_index=np.arange(10000),labels=y,native_logits=logits,
        xylo_logits=target_logits['xylosim'],spinnaker2_logits=target_logits['spinnaker2'],
        zero_recurrence_logits=ablated,reset_neural_state_logits=reset_logits)
    report=dict(schema='atlas.sequential-mnist.audit.v1',validation_passed=True,full_official_test_set=True,
        dataset_sha256=DATASET_SHA256,samples=10000,classes=list(range(10)),sizes=sizes,steps=x.shape[1],
        encoding=training['encoding'],readout_start=f['plan']['readout_start'],class_codes=f['plan']['class_codes'],
        class_decoder_execution='host-fixed-integer-codebook',same_model=True,same_weight_banks=True,
        same_class_codebook=True,retraining=False,clock_shift=0,numerical_tolerance=0,
        native=accuracy(logits,y),xylosim=accuracy(target_logits['xylosim'],y),spinnaker2_integer=accuracy(target_logits['spinnaker2'],y),
        accuracy_target=.98,accuracy_target_met=bool(accuracy(logits,y)['accuracy']>=.98),
        memory_controls=dict(zero_recurrence=accuracy(ablated,y),reset_neural_state=accuracy(reset_logits,y),
            reset_before_tick=boundary,accumulated_counts_preserved=True),comparisons=comparisons,
        portable_native_comparison=dict(checked=portable is not None,values=portable_values,
            differences=0 if portable is not None else None,runner_sha256=sha(a.portable_train_runner) if portable is not None else None),
        freeze_comparison=dict(checked=not a.frozen_only,values=freeze_values,differences=0 if not a.frozen_only else None),sdk_executions=executions,
        atlas_ir_replay=dict(samples=10,state_values=state_values,differences=0,receipts=receipts),
        frozen_sha256=f['sha256'],frozen_file_sha256=sha(a.trained/'frozen.json'),training_report_sha256=sha(a.trained/'training.json'),
        inputs_sha256=sha(a.output/'inputs.npz'),per_sample_sha256=sha(a.output/'per_sample.npz'),
        train_runner_sha256=sha(a.train_runner),atlas_runner_sha256=sha(a.runner),
        physical_hardware=False,hardware_mapping_tested=False,spinnaker2_execution_kind='custom-c-host-kernel',
        spinnaker2_firmware_adapter_implemented=False,
        source_hashes={path.name:sha(path) for path in (Path(__file__),Path(__file__).with_name('sdk_worker.py'),
            Path(__file__).with_name('train.py'),runtime,runtime.with_name('xylo_readout.py'),
            runtime.with_name('xylo_runtime.py'),runtime.with_name('spinnaker2_integer_runtime.py'),
            runtime.with_name('native_xylo_conversion.py'),runtime.parents[2]/'src/xylo_training.rs')})
    (a.output/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('validation_passed','native','xylosim','spinnaker2_integer','accuracy_target_met','comparisons')},indent=2),flush=True)


if __name__=='__main__':main()
