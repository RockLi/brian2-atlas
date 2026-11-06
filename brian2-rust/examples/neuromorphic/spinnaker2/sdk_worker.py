import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser()
p.add_argument('--runtime',type=Path,required=True)
p.add_argument('--bundles',type=Path,nargs='+',required=True)
a=p.parse_args()
spec=importlib.util.spec_from_file_location('atlas_neuro_runtime',a.runtime)
runtime=importlib.util.module_from_spec(spec);spec.loader.exec_module(runtime)
for path in a.bundles:
 bundle=json.loads(path.read_text())
 result=runtime.run_simulator(bundle)
 dest=path.parent/'sdk-results.npz'
 if dest.exists():raise FileExistsError(dest)
 np.savez_compressed(dest,**{k:v for k,v in result.items() if isinstance(v,np.ndarray)})
 report=dict(versions={n:importlib.metadata.version(n) for n in ('spinnaker2','brian2','numpy','scipy')},
             sdk_simulator_sha256=result['sdk_simulator_sha256'],
             bundle_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
             results_sha256=hashlib.sha256(dest.read_bytes()).hexdigest())
 (path.parent/'sdk-execution.json').write_text(json.dumps(report,indent=2)+'\n')
 print(path.parent.name, 'SDK finished',flush=True)
