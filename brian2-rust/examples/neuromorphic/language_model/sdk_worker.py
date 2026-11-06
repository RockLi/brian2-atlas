"""Stock SDK inference with inputs/model only; no labels or reference outputs."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import time
import numpy as np
from common import CONTEXT, VOCAB, raster, sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime',type=Path,required=True)
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--batch-size',type=int,default=100)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    spec=importlib.util.spec_from_file_location('atlas_neuro_runtime',a.runtime)
    runtime=importlib.util.module_from_spec(spec);spec.loader.exec_module(runtime)
    bundle=json.loads(a.bundle.read_text());data=np.load(a.inputs,allow_pickle=False)
    x=data['contexts'];prompt=data['prompt'];length=int(data['generation_length'])
    if prompt.shape != (CONTEXT,) or length != 32:
        raise ValueError('unexpected generation contract')
    start=time.monotonic()
    result=runtime.run_simulator(bundle,raster(x),batch_size=a.batch_size)
    np.savez_compressed(a.output/'results.npz',**{k:v for k,v in result.items() if isinstance(v,np.ndarray)})
    inference_seconds=time.monotonic()-start
    print(json.dumps(dict(teacher_forced_tokens=len(x),inference_seconds=inference_seconds)),flush=True)
    # Autoregressive: only this worker's preceding predictions feed its context.
    context=prompt.copy();ids=[];scores=[];spikes=[];states=[]
    for _ in range(length):
        r=runtime.run_simulator(bundle,raster(context[None,:]),batch_size=1)
        token=int(np.argmax(r['logits'][0]))
        ids.append(token);scores.append(r['logits'][0]);spikes.append(r['spike_raster'][0]);states.append(r['final_v'][0])
        context=np.concatenate((context[1:],np.asarray([token],dtype=np.uint8)))
    np.savez_compressed(a.output/'generation.npz',tokens=ids,logits=scores,spikes=spikes,final_v=states)
    report=dict(versions={n:importlib.metadata.version(n) for n in ('spinnaker2','brian2','numpy','scipy')},
        sdk_simulator_sha256=result['sdk_simulator_sha256'],batch_size=a.batch_size,
        inference_seconds=inference_seconds,total_seconds=time.monotonic()-start,
        bundle_sha256=sha(a.bundle),inputs_sha256=sha(a.inputs),results_sha256=sha(a.output/'results.npz'),
        generation_sha256=sha(a.output/'generation.npz'),worker_sha256=sha(__file__),
        physical_hardware=False,reference_or_labels_read=False,
        host_operations=['context one-hot encoding','independent replica batching','count-to-logit scaling','argmax','context window update'])
    (a.output/'execution.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
