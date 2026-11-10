"""An exhaustive two-input signed/biased MLP probe, not a public converter."""
import argparse,importlib.util,json,hashlib,subprocess
from pathlib import Path
import numpy as np
import brian2 as b
from brian2_atlas import export_network
from brian2_atlas.neuromorphic import export_ir
from brian2_rust.results import load_results
from brian2_rust.protocol import attach_protocol,canonical_bytes
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--runner',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
spec=importlib.util.spec_from_file_location('coverage_cases',Path(__file__).resolve().parents[1]/'coverage/cases.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
x=np.array([(i,j) for i in range(128) for j in range(128)],dtype=np.uint8)
c=dict(profile='dense-int8-v1',weights=[[127,-127,3],[127,127,-7]],thresholds=[128]*3,
       bias=[-1,65,130],readout=[[127,-3],[-61,7],[5,-127]],output_bias=[-100,17],inputs=x.tolist())
source=a.output/'source.json';model=export_network(module.build({**c,'inputs':x[:8192].tolist()}),b.ms,source)
bundle=export_ir(source,a.output/'validated-dense',target='akida',profile='dense-int8-v1',runner=a.runner)
all_states=[]
for index,start in enumerate((0,8192)):
    for i,name in enumerate(bundle['plan']['inputs']):
        raw=x[start:start+8192,i].astype('>f8').tobytes().hex()
        model['instance']['populations'][0]['parameters'][name]=[raw[j:j+16] for j in range(0,len(raw),16)]
    attach_protocol(model);part=a.output/f'part-{index}.json';part.write_bytes(canonical_bytes(model))
    subprocess.run([str(a.runner.absolute()),str(part),str(a.output/f'atlas-{index}')],check=True,capture_output=True,text=True)
    states=load_results(model,a.output/f'atlas-{index}')['populations'][0]['states']
    all_states.append(np.stack([states[k] for k in bundle['plan']['outputs']],axis=1))
np.savez_compressed(a.output/'atlas.npz',states=np.concatenate(all_states))
np.savez_compressed(a.output/'inputs.npz',inputs=x)
(a.output/'identity.json').write_text(json.dumps(dict(runner_sha256=hashlib.sha256(a.runner.read_bytes()).hexdigest(),
    source_sha256=bundle['source_model_sha256'],scope='Primitive feasibility only; no SpiNNaker2 public dense profile'),indent=2)+'\n')
print('Prepared',len(x),'exhaustive input pairs')
