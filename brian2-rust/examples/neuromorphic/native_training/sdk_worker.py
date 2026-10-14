"""Isolated stock SDK execution; receives no labels or reference outputs."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import time
import numpy as np

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--runtime',type=Path,required=True)
p.add_argument('--bundle',type=Path,required=True)
p.add_argument('--inputs',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--batch-size',type=int,default=100)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
spec=importlib.util.spec_from_file_location('atlas_neuro_runtime',a.runtime)
runtime=importlib.util.module_from_spec(spec);spec.loader.exec_module(runtime)
bundle=json.loads(a.bundle.read_text());x=np.load(a.inputs,allow_pickle=False)['spikes']
start=time.monotonic();result=runtime.run_simulator(bundle,x,batch_size=a.batch_size)
np.savez_compressed(a.output/'results.npz',**{k:v for k,v in result.items() if isinstance(v,np.ndarray)})
report=dict(versions={n:importlib.metadata.version(n) for n in ('spinnaker2','brian2','numpy','scipy')},
    sdk_simulator_sha256=result['sdk_simulator_sha256'],batch_size=a.batch_size,
    execution_seconds=time.monotonic()-start,
    bundle_sha256=hashlib.sha256(a.bundle.read_bytes()).hexdigest(),
    inputs_sha256=hashlib.sha256(a.inputs.read_bytes()).hexdigest(),
    results_sha256=hashlib.sha256((a.output/'results.npz').read_bytes()).hexdigest(),
    physical_hardware=False,reference_or_labels_read=False)
(a.output/'execution.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
