"""Run one SDK in isolation; this worker never receives labels or references."""
import argparse
import hashlib
import importlib.metadata as md
import importlib.util
import json
from pathlib import Path
import time
import numpy as np

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--bundle',type=Path,required=True)
p.add_argument('--inputs',type=Path,required=True)
p.add_argument('--runtime',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--batch-size',type=int,default=50)
a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
bundle=json.loads(a.bundle.read_text())
spec=importlib.util.spec_from_file_location('atlas_sdk_runtime',a.runtime)
api=importlib.util.module_from_spec(spec);spec.loader.exec_module(api)
with np.load(a.inputs,allow_pickle=False) as data:
    x=data['inputs']
started=time.monotonic()
result=api.run_simulator(bundle,x,batch_size=a.batch_size)
np.savez_compressed(a.output/'results.npz',**result)
packages={'akida':['akida','numpy'],'lava':['lava-nc','numpy'],'specksim':['sinabs','samna','torch','numpy'],'spinnaker2':['spinnaker2','brian2','numpy'],'xylosim':['xylosim','numpy']}[bundle['target']]
sha=lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
report=dict(target=bundle['target'],samples=len(x),states=int(result['states'].size),
            versions={p:md.version(p) for p in packages},bundle_sha256=sha(a.bundle),inputs_sha256=sha(a.inputs),
            source_model_sha256=bundle['source_model_sha256'],runtime_sha256=sha(a.runtime),
            results_sha256=sha(a.output/'results.npz'),elapsed_seconds=time.monotonic()-started,
            physical_hardware=False,hardware_mapping_tested=False,
            implementation_sha256={p.name:sha(p) for p in a.runtime.parent.glob('*.py')
                                   if p.name in ('count_targets.py','neuromorphic_runtime.py','spinnaker2_dense.py')},
            note='CPU simulator wall time is not chip latency or energy.')
(a.output/'execution.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
