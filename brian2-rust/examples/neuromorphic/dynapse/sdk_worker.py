import argparse,hashlib,importlib.metadata as md,importlib.util,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--runtime',type=Path,required=True);p.add_argument('--bundles',type=Path,nargs='+',required=True)
a=p.parse_args();spec=importlib.util.spec_from_file_location('api',a.runtime);api=importlib.util.module_from_spec(spec);spec.loader.exec_module(api)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
for path in a.bundles:
    if (path.parent/'sdk.npz').exists():raise FileExistsError(path.parent/'sdk.npz')
    bundle=json.loads(path.read_text());result=api.run_simulator(bundle)
    np.savez_compressed(path.parent/'sdk.npz',**result)
    report=dict(versions={p:md.version(p) for p in ('rockpool','jax','jaxlib','numpy')},
        sdk_source_sha256=result['sdk_source_sha256'],bundle_sha256=sha(path),result_sha256=sha(path.parent/'sdk.npz'),
        runtime_sha256=sha(a.runtime.with_name('dynapse_runtime.py')),physical_hardware=False)
    (path.parent/'execution.json').write_text(json.dumps(report,indent=2)+'\n')
    print(path.parent.parent.name,'SDK complete',flush=True)
