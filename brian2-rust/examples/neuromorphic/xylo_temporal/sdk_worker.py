"""Execute actual SDK configs, with no Atlas reference or labels as input."""
import argparse
import hashlib
import importlib.metadata as md
import importlib.util
import json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--runtime',type=Path,required=True);p.add_argument('--bundles',type=Path,nargs='+',required=True)
a=p.parse_args();spec=importlib.util.spec_from_file_location('api',a.runtime)
api=importlib.util.module_from_spec(spec);spec.loader.exec_module(api)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
for path in a.bundles:
    if (path.parent/'sdk.npz').exists():raise FileExistsError(path.parent/'sdk.npz')
    bundle=json.loads(path.read_text());result=api.run_simulator(bundle)
    configuration=result.pop('configuration_json')
    (path.parent/'hardware-config.json').write_text(configuration)
    np.savez_compressed(path.parent/'sdk.npz',**result)
    report=dict(official_configuration_valid=bool(result['configuration_valid']),
        bundle_sha256=sha(path),result_sha256=sha(path.parent/'sdk.npz'),config_sha256=sha(path.parent/'hardware-config.json'),
        versions={p:md.version(p) for p in ('xylosim','samna','rockpool','numpy','bitstruct','packaging')},
        sdk_binary_sha256=str(result['sdk_binary_sha256']),
        runtime_sha256=sha(a.runtime),xylo_runtime_sha256=sha(a.runtime.with_name('xylo_runtime.py')),
        physical_hardware=False)
    (path.parent/'execution.json').write_text(json.dumps(report,indent=2)+'\n')
    print(path.parent.parent.name,'SDK complete',flush=True)
