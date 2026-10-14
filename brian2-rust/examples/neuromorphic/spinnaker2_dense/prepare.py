"""Export the unchanged source and replay every input with the native Atlas runner."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np
from brian2_atlas.neuromorphic import export_ir
from brian2_rust.protocol import attach_protocol, canonical_bytes
from brian2_rust.results import load_results

p = argparse.ArgumentParser(description=__doc__)
for key in ('source','inputs','runner','output'):
    p.add_argument('--'+key,type=Path,required=True)
a = p.parse_args()
os.environ.update(B2_MAX_POPULATION_STATES='512',B2_MAX_POPULATION_PARAMETERS='1024',
                  B2_MAX_STATEMENTS='512',B2_MAX_SCRATCH_VALUES='128000000')
a.output.mkdir(parents=True,exist_ok=False)
bundle = export_ir(a.source,a.output/'converted',target='spinnaker2',
                   profile='dense-int8-v1',runner=a.runner)
plan = bundle['plan']
x = np.load(a.inputs,allow_pickle=False)['inputs']
np.savez_compressed(a.output/'inputs.npz',inputs=x)
model = json.loads(a.source.read_text())
count = model['definition']['populations'][0]['count']
if len(x) % count:
    raise ValueError('this replay fixture requires an exact number of source-sized batches')
states, batches = [], []
started = time.monotonic()
for start in range(0,len(x),count):
    for i,name in enumerate(plan['inputs']):
        raw = x[start:start+count,i].astype('>f8').tobytes().hex()
        model['instance']['populations'][0]['parameters'][name] = [raw[j:j+16] for j in range(0,len(raw),16)]
    attach_protocol(model)
    assert model['protocol']['layers']['definition'] == bundle['source_hashes']['definition']
    encoded = canonical_bytes(model)
    path = a.output/'replay-batch.json'
    path.write_bytes(encoded)
    dest = a.output/f'atlas-{start:05d}'
    run = subprocess.run([str(a.runner.absolute()),str(path),str(dest)],capture_output=True,text=True)
    if run.returncode:
        raise RuntimeError(run.stderr)
    result = load_results(model,dest)['populations'][0]['states']
    states.append(np.stack([result[n] for n in plan['outputs']],axis=1))
    batches.append(dict(start=start,count=count,source_sha256=hashlib.sha256(encoded).hexdigest()))
    print('Atlas',start+count,'/',len(x),flush=True)
np.savez_compressed(a.output/'atlas.npz',states=np.concatenate(states))
sha = lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
report = dict(reference_kind='fresh native Atlas replay',samples=len(x),batches=batches,
    elapsed_seconds=time.monotonic()-started,source_model_sha256=bundle['source_model_sha256'],
    definition_sha256=bundle['source_hashes']['definition'],runner_sha256=sha(a.runner),
    inputs_sha256=sha(a.output/'inputs.npz'),states_sha256=sha(a.output/'atlas.npz'),
    resource_budgets={k:v for k,v in os.environ.items() if k.startswith('B2_MAX_')})
(a.output/'atlas-execution.json').write_text(json.dumps(report,indent=2)+'\n')
