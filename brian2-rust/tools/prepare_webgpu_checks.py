"""Generate independent native CPU-f32 controls for the browser WGSL checks.

Run after build_wasm.py. Outputs are local test artifacts, not shipped assets.
"""
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'));sys.path.insert(0,str(ROOT/'tests'))
from brian2_rust.protocol import attach_protocol
from test_gpu_spike_generator import execute

script='''
import {readFileSync} from 'node:fs';
import {models} from './brian2-rust/wasm/models.js';
import {makeDraft,bits} from './brian2-rust/wasm/experiment.js';
const cases=[];
for(const [id,meta]of Object.entries(models)){
 if(id==='flywire')continue;
 const template=JSON.parse(readFileSync('brian2-rust/output/wasm/'+meta.template));
 const configs=Object.entries(meta.presets).map(([name,p])=>[name,p.config]);configs.push(['maximum',{...meta.defaults,neurons:meta.scales.at(-1)}]);
 if(id==='adaptive_lif')configs.push(['exact-binary-clock',{...meta.defaults,neurons:8,duration_ms:64,dt_ms:1,tau_ms:16,drive:2,spread:0,adaptation:0,synchronize:true}]);
 if(id==='hodgkin_huxley')configs.push(['exprel-poles',{...meta.defaults,neurons:8}]);
 for(const [name,c]of configs){const m=makeDraft(template,c);if(name==='exprel-poles')m.instance.populations[0].initial_state.v=[-40,-55,-40.00001,-39.99999,-55.00001,-54.99999,-65,-65].map(bits);cases.push({name:id+'/'+name,model:m});}
}
console.log(JSON.stringify(cases));
'''
items=json.loads(subprocess.check_output(['node','--input-type=module','-e',script],cwd=ROOT.parent,text=True))
output=ROOT/'output/webgpu-cpu-f32';output.mkdir(exist_ok=True)
for item in items:
    name=item['name'].replace('/','-');model=item['model'];attach_protocol(model)
    directory=output/name
    if directory.exists():
        import shutil
        shutil.rmtree(directory)
    result=execute(model,directory,'cpu-f32')['populations'][0]
    control=dict(n=model['definition']['populations'][0]['count'],counts=result['counts'].tolist(),indices=result['indices'].tolist(),ticks=result['spike_ticks'].tolist(),trace={k:v.reshape(-1).tolist() for k,v in result['trace'].items()})
    (ROOT/f'output/wasm/oracle-{name}.json').write_text(json.dumps(control,separators=(',',':')))
    (directory/'model.json').write_text(json.dumps(model))
    print(name,flush=True)
