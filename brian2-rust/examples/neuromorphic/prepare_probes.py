"""Boundary and reset probes for an already frozen shared conversion bundle."""
import argparse,json,os,subprocess,tempfile,hashlib
from pathlib import Path
import numpy as np
from brian2_rust.protocol import attach_protocol,canonical_bytes
from brian2_rust.results import load_results
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--run',type=Path,required=True)
p.add_argument('--runner',type=Path,required=True)
a=p.parse_args();dest=a.run/'probes';dest.mkdir(exist_ok=False)
bundle=json.loads((a.run/'akida/bundle.json').read_text());plan=bundle['plan'];n=len(plan['inputs'])
x=np.concatenate([np.zeros((1,n)),np.ones((1,n)),np.zeros((2,n)),np.eye(1,n),np.eye(1,n,k=n-1),
                  (np.arange(n)%2)[None,:],(1-np.arange(n)%2)[None,:],np.random.default_rng(42).integers(0,2,(4,n)),
                  np.ones((1,n)),np.zeros((1,n))]).astype(np.uint8)
np.savez_compressed(dest/'inputs.npz',inputs=x)
m=json.loads((a.run/'source.json').read_text());m['definition']['populations'][0]['count']=len(x);m['instance']['neuron_count']=len(x)
inst=m['instance']['populations'][0]
for key in inst['initial_state']:inst['initial_state'][key]=['0000000000000000']*len(x)
for i,name in enumerate(plan['inputs']):
    raw=x[:,i].astype('>f8').tobytes().hex()
    inst['parameters'][name]=[raw[k:k+16] for k in range(0,len(raw),16)]
attach_protocol(m);(dest/'source.json').write_bytes(canonical_bytes(m))
os.environ['B2_MAX_POPULATION_PARAMETERS']='1024'
with tempfile.TemporaryDirectory(prefix='atlas-neuro-probes-') as temp:
    result=subprocess.run([str(a.runner.resolve()),str(dest/'source.json'),temp+'/result'],capture_output=True,text=True)
    if result.returncode:raise RuntimeError(result.stderr)
    states=load_results(m,Path(temp)/'result')['populations'][0]['states']
    np.savez_compressed(dest/'atlas.npz',states=np.stack([states[n] for n in plan['outputs']],axis=1))
print('Prepared',len(x),'boundary/reset probes')
