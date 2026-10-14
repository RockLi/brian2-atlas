"""Audit held-out character perplexity and per-token native/AtlasIR/SDK parity."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import numpy as np
import brian2 as b
from brian2_atlas import NativeLIFTrainer, export_network
from brian2_atlas.neuromorphic import export_native_lif
from brian2_rust.native_lif_conversion import build_native_lif_network, validate_frozen
from brian2_rust.results import load_results
from common import CONTEXT, DATASET_SHA256, TICKS, VOCAB, baselines, corpus, log_probs, metrics, normalize, raster, sha, windows


def native(trainer, contexts, labels):
    spikes=[];logits=[];membrane=[]
    for k in range(0,len(labels),128):
        r=trainer.evaluate(raster(contexts[k:k+128]),labels[k:k+128])
        spikes.append(np.asarray(r['spikes'],dtype=np.uint8));logits.extend(r['logits']);membrane.extend(r['final_membrane'])
    return dict(spikes=np.concatenate(spikes),logits=np.asarray(logits),final_membrane=np.asarray(membrane))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--trained',type=Path,required=True)
    p.add_argument('--runner',type=Path,required=True)
    p.add_argument('--train-runner',type=Path,required=True)
    p.add_argument('--sdk-python',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--limit',type=int,default=9996)
    a=p.parse_args()
    if not 1<=a.limit<=9996: p.error('--limit must be in 1..9996')
    a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False)
    f=json.loads((a.trained/'frozen.json').read_text());validate_frozen(f)
    training=json.loads((a.trained/'training.json').read_text())
    assert sha(a.trained/'frozen.json')==training['frozen_file_sha256']
    assert sha(a.trained/'checkpoint.json')==training['checkpoint_sha256']
    assert training['vocabulary']==VOCAB and training['context_characters']==CONTEXT and training['inference_ticks']==TICKS
    checkpoint=json.loads(json.loads((a.trained/'checkpoint.json').read_text())['payload'])
    float_trainer=NativeLIFTrainer(checkpoint['plan'],runner=a.train_runner)
    float_trainer.restore(a.trained/'checkpoint.json')
    quantized=NativeLIFTrainer(f['plan'],weights=f['weights'],runner=a.train_runner)
    tokens=corpus(a.dataset);x,y,positions=windows(tokens,'test')
    x=x[:a.limit];y=y[:a.limit];positions=positions[:a.limit]
    prompt=normalize('the ');generation_length=32
    # Inputs contain contexts only: no next-token labels or reference outputs.
    np.savez_compressed(a.output/'inputs.npz',contexts=x,prompt=prompt,generation_length=generation_length)
    before=native(float_trainer,x,y);after=native(quantized,x,y)
    np.savez_compressed(a.output/'native.npz',**after,float_logits=before['logits'])
    print(json.dumps(dict(float_native=metrics(before['logits'],y),quantized_native=metrics(after['logits'],y))),flush=True)
    b.prefs.codegen.target='numpy';b.prefs.codegen.loop_invariant_optimisations=False
    bundle=export_native_lif(f,a.output/'export',steps=TICKS,runner=a.runner)
    replay_samples=min(64,len(x));atlas_spikes=[];atlas_voltages=[];receipts=[]
    cells=sum(f['plan']['sizes'][1:])
    for k in range(0,replay_samples,8):
        chunk=raster(x[k:min(k+8,replay_samples)]);dest=a.output/f'atlas-{k:05d}';dest.mkdir()
        model=export_network(build_native_lif_network(f,chunk),TICKS*b.ms,dest/'source.json')
        subprocess.run([str(a.runner.resolve()),str(dest/'source.json'),str(dest/'results')],check=True,capture_output=True)
        state=load_results(model,dest/'results')['populations'][bundle['plan']['lif_population']]
        sp=np.zeros((len(chunk),TICKS,cells),dtype=np.uint8)
        idx=np.asarray(state['indices'],dtype=int);ticks=np.asarray(state['spike_ticks'],dtype=int)
        sp[idx//cells,ticks,idx%cells]=1
        atlas_spikes.append(sp)
        atlas_voltages.append(state['trace']['v'].reshape(TICKS,len(chunk),cells).transpose(1,0,2))
        receipts.append(dict(first_corpus_position=int(positions[k]),tokens=len(chunk),source_sha256=sha(dest/'source.json')))
    atlas_spikes=np.concatenate(atlas_spikes);atlas_voltages=np.concatenate(atlas_voltages)
    np.testing.assert_array_equal(atlas_spikes,after['spikes'][:replay_samples])
    print('Concrete AtlasIR replay passed for',replay_samples,'token contexts',flush=True)
    runtime=Path(__file__).resolve().parents[3]/'python/brian2_rust/neuromorphic_runtime.py'
    worker=Path(__file__).with_name('sdk_worker.py')
    with (a.output/'sdk.log').open('w') as log:
        subprocess.run([str(a.sdk_python.absolute()),str(worker),'--runtime',str(runtime),'--bundle',str(a.output/'export/converted/bundle.json'),
            '--inputs',str(a.output/'inputs.npz'),'--output',str(a.output/'sdk')],cwd=a.output,
            env={**os.environ,'PYTHONPATH':'','PYTHONNOUSERSITE':'1'},stdout=log,stderr=subprocess.STDOUT,check=True)
    sdk=np.load(a.output/'sdk/results.npz',allow_pickle=False)
    np.testing.assert_array_equal(sdk['spike_raster'],after['spikes'])
    counts=after['spikes'][:,:,-len(VOCAB):].sum(1,dtype=np.int64)
    np.testing.assert_array_equal(sdk['counts'],counts)
    np.testing.assert_array_equal(sdk['logits'],after['logits'])
    np.testing.assert_array_equal(log_probs(sdk['logits']),log_probs(after['logits']))
    np.testing.assert_array_equal(sdk['voltage_f64'][:replay_samples],atlas_voltages)
    # Invert z=scale*(v-J) for every test context, not just the IR replay subset.
    sizes=f['plan']['sizes'];last=after['spikes'][:,-1];currents=[];offset=0
    for layer,(na,nb,w) in enumerate(zip(sizes,sizes[1:],f['weights'])):
        pre=raster(x)[:,-1] if layer==0 else last[:,offset-sizes[layer]:offset]
        currents.append(pre@np.asarray(w).reshape(na,nb));offset+=nb
    reconstructed=sdk['final_v']/np.repeat(f['scales'],sizes[1:])+np.concatenate(currents,axis=1)
    np.testing.assert_array_equal(reconstructed,after['final_membrane'])
    # A second, independent autoregressive loop; no SDK token drives native input.
    context=prompt.copy();generated=[];gen_logits=[];gen_spikes=[]
    for _ in range(generation_length):
        r=quantized.evaluate(raster(context[None,:]),[0])
        token=int(np.argmax(r['logits'][0]));generated.append(token)
        gen_logits.append(r['logits'][0]);gen_spikes.append(r['spikes'][0])
        context=np.concatenate((context[1:],np.asarray([token],dtype=np.uint8)))
    sg=np.load(a.output/'sdk/generation.npz',allow_pickle=False)
    np.testing.assert_array_equal(sg['tokens'],generated)
    np.testing.assert_array_equal(sg['logits'],gen_logits)
    np.testing.assert_array_equal(sg['spikes'],gen_spikes)
    bline=baselines(tokens,x,y)
    float_metrics=metrics(before['logits'],y);quant_metrics=metrics(after['logits'],y);sdk_metrics=metrics(sdk['logits'],y)
    assert quant_metrics['perplexity'] < bline['unigram']['perplexity'], 'trained model did not beat the unigram baseline'
    nll=lambda z: -log_probs(z)[np.arange(len(y)),y]
    np.savez_compressed(a.output/'token-audit.npz',corpus_positions=positions,targets=y,
        float_nll=nll(before['logits']),quantized_nll=nll(after['logits']),sdk_nll=nll(sdk['logits']),
        native_predictions=np.argmax(after['logits'],axis=1),sdk_predictions=np.argmax(sdk['logits'],axis=1),
        spike_differences=np.count_nonzero(sdk['spike_raster']!=after['spikes'],axis=(1,2)))
    report=dict(schema='atlas.character-lm.audit.v1',passed=True,dataset_sha256=DATASET_SHA256,
        scope='fixed four-character context; fresh LIF state per prediction; no inter-token neural state or Transformer',
        vocabulary=VOCAB,context_characters=CONTEXT,ticks_per_token=TICKS,full_declared_test_split=a.limit==9996,
        test_positions=[int(positions[0]),int(positions[-1])+1],float_native=float_metrics,quantized_native=quant_metrics,sdk=sdk_metrics,
        baselines=bline,quantization_perplexity_change=quant_metrics['perplexity']-float_metrics['perplexity'],
        conversion_perplexity_change=sdk_metrics['perplexity']-quant_metrics['perplexity'],
        per_token_logit_values=int(sdk['logits'].size),max_logit_error=0.,max_log_probability_error=0.,prediction_differences=0,
        spike_raster_values=int(after['spikes'].size),spike_raster_differences=0,
        final_native_state_values=int(reconstructed.size),max_native_state_error=0.,
        atlas_ir_replay_tokens=replay_samples,atlas_ir_voltage_values=int(atlas_voltages.size),atlas_sdk_voltage_max_error=0.,
        atlas_ir_receipts=receipts,autoregressive=dict(prompt='the ',tokens=generation_length,prediction_differences=0,
            max_logit_error=0.,spike_differences=0,text='the '+''.join(VOCAB[i] for i in generated),
            tie_breaking='lowest vocabulary index; numpy argmax'),
        checkpoint_sha256=sha(a.trained/'checkpoint.json'),frozen_sha256=f['sha256'],frozen_file_sha256=sha(a.trained/'frozen.json'),
        input_file_sha256=sha(a.output/'inputs.npz'),native_results_sha256=sha(a.output/'native.npz'),
        token_audit_sha256=sha(a.output/'token-audit.npz'),atlas_runner_sha256=sha(a.runner),
        sdk_execution=json.loads((a.output/'sdk/execution.json').read_text()),physical_hardware=False,hardware_mapping_tested=False,
        source_sha256={p.name:sha(p) for p in (Path(__file__),worker,Path(__file__).with_name('common.py'),runtime,runtime.with_name('spinnaker2_runtime.py'))})
    (a.output/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('passed','float_native','quantized_native','sdk','baselines','autoregressive')}),flush=True)


if __name__=='__main__':main()
