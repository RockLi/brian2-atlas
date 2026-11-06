"""Execute an installed target SDK, without labels or reference arrays."""
import argparse
import hashlib
import importlib.metadata as md
import importlib.util
import json
from pathlib import Path
import time
import numpy as np


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('runtime','bundle','inputs','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    spec=importlib.util.spec_from_file_location('runtime',a.runtime)
    runtime=importlib.util.module_from_spec(spec);spec.loader.exec_module(runtime)
    bundle=json.loads(a.bundle.read_text());x=np.load(a.inputs,allow_pickle=False)['spikes']
    start=time.monotonic();r=runtime.run_simulator(bundle,x,batch_size=100)
    np.savez_compressed(a.output/'results.npz',**{k:v for k,v in r.items() if isinstance(v,np.ndarray)})
    names=('spinnaker2','brian2','numpy','scipy') if bundle['target']=='spinnaker2' else ('rockpool','xylosim','samna','numpy')
    report=dict(target=bundle['target'],versions={n:md.version(n) for n in names},
        bundle_sha256=sha(a.bundle),inputs_sha256=sha(a.inputs),results_sha256=sha(a.output/'results.npz'),
        execution_seconds=time.monotonic()-start,reference_or_labels_read=False,physical_hardware=False)
    if bundle['target']=='xylosim':
        (a.output/'hardware-config.json').write_text(r['configuration_json'])
        report.update(official_configuration_valid=bool(r['configuration_valid']),
                      configuration_sha256=sha(a.output/'hardware-config.json'),sdk_sha256=r['sdk_binary_sha256'])
    else: report['sdk_sha256']=r['sdk_simulator_sha256']
    (a.output/'execution.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
