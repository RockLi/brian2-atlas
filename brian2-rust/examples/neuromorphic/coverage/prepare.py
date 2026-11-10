"""Export each coverage model and run its reference with native Atlas."""
import argparse,hashlib,json,os,subprocess
from pathlib import Path
import numpy as np
import brian2 as b
from brian2_atlas import export_network
from brian2_atlas.neuromorphic import export_ir
from brian2_rust.results import load_results
from cases import cases,build,SEED

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True);p.add_argument('--runner',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
os.environ.update(B2_MAX_POPULATION_PARAMETERS='1024',B2_MAX_POPULATION_STATES='512',B2_MAX_STATEMENTS='512')
reports=[]
for name,c in cases().items():
    dest=a.output/name;dest.mkdir();source=dest/'source.json'
    model=export_network(build(c),b.ms,source)
    targets=('akida','lava','specksim','spinnaker2','xylosim') if c['profile']=='binary-count-v1' else ('akida','lava')
    bundles={t:export_ir(source,dest/t,target=t,runner=a.runner,profile=c['profile']) for t in targets}
    assert len({v['source_model_sha256'] for v in bundles.values()})==1
    subprocess.run([str(a.runner.absolute()),str(source),str(dest/'atlas')],check=True,capture_output=True,text=True)
    states=load_results(model,dest/'atlas')['populations'][0]['states'];plan=bundles['akida']['plan']
    np.savez_compressed(dest/'atlas.npz',states=np.stack([states[n] for n in plan['outputs']],axis=1))
    np.savez_compressed(dest/'inputs.npz',inputs=np.asarray(c['inputs'],np.uint8))
    reports.append(dict(case=name,profile=c['profile'],targets=list(targets),samples=len(c['inputs']),
                        input_size=len(c['weights']),hidden_size=len(c['weights'][0]),
                        source_model_sha256=bundles['akida']['source_model_sha256']))
    print(name,'Atlas complete',flush=True)
(a.output/'preparation.json').write_text(json.dumps(dict(seed=SEED,cases=reports,runner_sha256=hashlib.sha256(a.runner.read_bytes()).hexdigest()),indent=2)+'\n')
