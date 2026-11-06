"""Execute SDK batches without reading reference results or labels."""
import argparse,hashlib,importlib.metadata as md,importlib.util,json,time
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--run',type=Path,required=True);p.add_argument('--runtime',type=Path,required=True)
p.add_argument('--target',choices=('akida','lava','specksim','spinnaker2','xylosim'),required=True)
a=p.parse_args();spec=importlib.util.spec_from_file_location('atlas_runtime',a.runtime)
api=importlib.util.module_from_spec(spec);spec.loader.exec_module(api)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
packages={'akida':['akida','numpy'],'lava':['lava-nc','numpy'],'specksim':['sinabs','samna','torch','numpy'],
          'spinnaker2':['spinnaker2','brian2','numpy'],'xylosim':['xylosim','numpy']}[a.target]
for path in sorted(a.run.glob('*/'+a.target+'/bundle.json')):
    dest=path.parent/'coverage-run';dest.mkdir(exist_ok=False)
    bundle=json.loads(path.read_text());x=np.load(path.parent.parent/'inputs.npz')['inputs']
    started=time.monotonic();result=api.run_simulator(bundle,x,batch_size=2)
    np.savez_compressed(dest/'results.npz',**result)
    # A different batch width forces different reset/rebuild boundaries in Specksim.
    batch_check=None
    if a.target=='specksim':
        alternate=api.run_simulator(bundle,x,batch_size=5)
        for key in result:np.testing.assert_array_equal(result[key],alternate[key])
        np.savez_compressed(dest/'alternate.npz',**alternate)
        batch_check=dict(batch_sizes=[2,5],differences=0,result_sha256=sha(dest/'alternate.npz'))
    report=dict(target=a.target,samples=len(x),state_values=int(result['states'].size),versions={p:md.version(p) for p in packages},
                bundle_sha256=sha(path),inputs_sha256=sha(path.parent.parent/'inputs.npz'),results_sha256=sha(dest/'results.npz'),
                implementation_sha256={p.name:sha(p) for p in a.runtime.parent.glob('*.py') if p.name in
                    ('neuromorphic_runtime.py','akida_b2ir.py','lava_dense_b2ir.py','count_targets.py')},
                batch_check=batch_check,elapsed_seconds=time.monotonic()-started,physical_hardware=False)
    (dest/'execution.json').write_text(json.dumps(report,indent=2)+'\n')
    print(path.parent.parent.name,a.target,'complete',flush=True)
