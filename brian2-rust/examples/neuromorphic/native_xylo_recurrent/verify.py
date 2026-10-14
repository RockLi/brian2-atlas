"""Held-out order task, memory controls and native/AtlasIR/XyloSim parity."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import brian2 as b
import numpy as np
from brian2_atlas import NativeXyloTrainer,export_network
from brian2_atlas.neuromorphic import export_native_xylo
from brian2_rust.native_xylo_conversion import build_native_xylo_network,validate_frozen
from brian2_rust.results import load_results
from train import data,sha,STEPS,READOUT_START


def accuracy(logits,y):
    pred=np.argmax(logits,1);confusion=np.zeros((2,2),int);np.add.at(confusion,(y,pred),1)
    return dict(correct=int(np.count_nonzero(pred==y)),total=len(y),accuracy=float(np.mean(pred==y)),confusion_matrix=confusion.tolist())


def native(trainer,x,y,chunk_steps=None):
    arrays={key:[] for key in ('spikes','currents','membranes','logits')}
    for i in range(0,len(x),64):
        result=(trainer.evaluate(x[i:i+64],y[i:i+64]) if chunk_steps is None else
                trainer.evaluate_chunks(x[i:i+64],y[i:i+64],chunk_steps=chunk_steps))
        for key in arrays:arrays[key].append(np.asarray(result[key]))
    return {key:np.concatenate(value) for key,value in arrays.items()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('trained','runner','train-runner','xylo-python','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False)
    f=json.loads((a.trained/'frozen.json').read_text());sizes=validate_frozen(f)
    training=json.loads((a.trained/'training.json').read_text())
    assert sha(a.trained/'frozen.json')==training['frozen_file_sha256']
    assert sha(a.trained/'checkpoint.json')==training['checkpoint_sha256']
    trainer=NativeXyloTrainer(f['plan'],runner=a.train_runner);trainer.restore(a.trained/'checkpoint.json')
    quantized=[np.asarray(w)/f['plan']['weight_scale'] for w in f['target_weights']]
    frozen=NativeXyloTrainer(f['plan'],runner=a.train_runner,weights=quantized)
    x,y,ids=data('test');_,_,train_ids=data('train');_,_,val_ids=data('validation')
    assert not set(ids)&set(train_ids) and not set(ids)&set(val_ids) and not set(train_ids)&set(val_ids)
    assert len(set(hashlib.sha256(r.tobytes()).hexdigest() for r in x))==len(x)
    np.testing.assert_array_equal(x[::2].sum(1),x[1::2].sum(1))
    np.testing.assert_array_equal(x[:,22:],np.broadcast_to(x[0,22:],x[:,22:].shape))
    np.testing.assert_array_equal(x[::2,:,[1,0,2]],x[1::2])
    np.savez_compressed(a.output/'inputs.npz',spikes=x)
    whole=native(trainer,x,y);after=native(frozen,x,y)
    for key in whole:np.testing.assert_array_equal(whole[key],after[key])
    del after
    chunks=[]
    for width in (7,13):
        part=native(frozen,x,y,width)
        for key in whole:np.testing.assert_array_equal(whole[key],part[key])
        chunks.append(dict(chunk_steps=width,differences=0,samples=len(x)))
    print('Native test:',accuracy(whole['logits'],y),flush=True)
    # Controls are evaluated only after selection; they never tune the checkpoint.
    ablated=[w.copy() for w in quantized];ablated[2][:]=0
    no_rec=NativeXyloTrainer(f['plan'],runner=a.train_runner,weights=ablated)
    without=native(no_rec,x,y)
    reset_logits=[]
    for i in range(0,len(x),64):
        prefix=frozen.evaluate(x[i:i+64,:READOUT_START],y[i:i+64]);state=copy.deepcopy(prefix['continuation'])
        for key in ('currents','membranes','spikes','counts'):
            state[key]=np.zeros_like(state[key]).tolist()
        result=frozen.evaluate(x[i:i+64,READOUT_START:],y[i:i+64],initial=state)
        reset_logits.extend(result['logits'])
    reset_logits=np.asarray(reset_logits)
    controls=dict(zero_recurrent_weights=accuracy(without['logits'],y),reset_all_neural_state_at_query=accuracy(reset_logits,y),
        count_only_upper_bound=.5,unique_count_vectors=len(np.unique(x.sum(1),axis=0)),
        reversed_pair_prediction_changes=int(np.count_nonzero(np.argmax(whole['logits'][::2],1)!=np.argmax(whole['logits'][1::2],1))))
    print('Memory controls:',controls,flush=True)
    b.prefs.codegen.target='numpy';b.prefs.codegen.loop_invariant_optimisations=False
    bundle=export_native_xylo(f,a.output/'xylosim',steps=STEPS,runner=a.runner)
    np.testing.assert_array_equal(bundle['plan']['weights']['recurrent'],np.asarray(f['target_weights'][2]).reshape(sizes[1],sizes[1]))
    runtime=Path(__file__).resolve().parents[3]/'python/brian2_rust/neuromorphic_runtime.py'
    worker=Path(__file__).with_name('sdk_worker.py');sdk_dir=a.output/'sdk'
    with (a.output/'sdk.log').open('w') as log:
        subprocess.run([str(a.xylo_python.absolute()),str(worker),'--runtime',str(runtime),
            '--bundle',str(a.output/'xylosim/converted/bundle.json'),'--inputs',str(a.output/'inputs.npz'),'--output',str(sdk_dir)],
            cwd=a.output,env={**os.environ,'PYTHONPATH':'','PYTHONNOUSERSITE':'1'},stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
    execution=json.loads((sdk_dir/'execution.json').read_text());sdk=np.load(sdk_dir/'results.npz')
    assert execution['inputs_sha256']==sha(a.output/'inputs.npz') and execution['results_sha256']==sha(sdk_dir/'results.npz')
    comparisons={}
    for field,key in [('spikes','spike_raster'),('membranes','voltage'),('logits','logits')]:
        np.testing.assert_array_equal(whole[field],sdk[key]);comparisons[field]=dict(values=int(whole[field].size),differences=0)
    np.testing.assert_array_equal(whole['currents'],np.concatenate((sdk['hidden_I'],sdk['output_I']),axis=2))
    comparisons['currents']=dict(values=int(whole['currents'].size),differences=0)
    receipts=[];state_values=0
    for i in range(32):
        dest=a.output/'replay'/str(i);dest.mkdir(parents=True)
        model=export_network(build_native_xylo_network(f,x[i]),STEPS*b.ms,dest/'source.json')
        subprocess.run([str(a.runner.resolve()),str(dest/'source.json'),str(dest/'results')],capture_output=True,check=True)
        r=load_results(model,dest/'results')
        for role in ('hidden','output'):
            pop=r['populations'][bundle['plan']['roles'][role]]
            for key in ('I','v','q'):
                np.testing.assert_array_equal(pop['trace'][key],sdk[role+'_'+key][i]);state_values+=pop['trace'][key].size
            np.testing.assert_array_equal(np.column_stack((pop['spike_ticks'],pop['indices'])),np.column_stack(np.nonzero(sdk[role+'_q'][i])))
        receipts.append(dict(test_sample=i,source_sha256=sha(dest/'source.json'),results_sha256=sha(dest/'results/results.bin'),events_sha256=sha(dest/'results/events.bin')))
    np.savez_compressed(a.output/'per_sample.npz',template_ids=ids,labels=y,native_logits=whole['logits'],xylo_logits=sdk['logits'],
        native_counts=whole['spikes'][:,READOUT_START:,-2:].sum(1),xylo_counts=sdk['counts'],
        zero_recurrence_logits=without['logits'],reset_at_query_logits=reset_logits)
    files=[Path(__file__),worker,Path(__file__).with_name('train.py'),runtime,
        runtime.with_name('training_xylo.py'),runtime.with_name('native_xylo_conversion.py'),
        runtime.with_name('xylo_runtime.py'),runtime.with_name('xylo_network.py'),runtime.with_name('xylo_b2ir.py'),
        runtime.parents[2]/'src/xylo_training.rs']
    report=dict(schema='atlas.native-xylo.recurrent-audit.v1',validation_passed=True,training=training,
        test_samples=len(y),test_templates=len(set(ids)),split_templates_disjoint=True,pairs_equal_event_counts=True,
        identical_suffix_from_tick=22,readout_start=READOUT_START,query_starts_at_tick=32,
        native=accuracy(whole['logits'],y),xylosim=accuracy(sdk['logits'],y),memory_controls=controls,
        freeze_differences=0,native_vs_xylosim=comparisons,native_full_vs_chunks=chunks,
        atlas_ir_replay=dict(samples=32,state_values=state_values,differences=0,receipts=receipts),
        sdk_execution=execution,frozen_sha256=f['sha256'],frozen_file_sha256=sha(a.trained/'frozen.json'),
        checkpoint_sha256=sha(a.trained/'checkpoint.json'),inputs_sha256=sha(a.output/'inputs.npz'),
        per_sample_sha256=sha(a.output/'per_sample.npz'),physical_hardware=False,
        source_hashes={path.name:sha(path) for path in files})
    (a.output/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({key:report[key] for key in ('validation_passed','native','xylosim','memory_controls','native_vs_xylosim')}),flush=True)


if __name__=='__main__':main()
