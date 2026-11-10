"""Stock XyloSim full, chunked and persistent-session execution; no labels."""
import argparse
import hashlib
import importlib.metadata as md
import importlib.util
import json
from pathlib import Path
import time
import numpy as np


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('runtime','bundle','inputs','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    spec=importlib.util.spec_from_file_location('runtime',a.runtime)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    bundle=json.loads(a.bundle.read_text());x=np.load(a.inputs,allow_pickle=False)['spikes']
    start=time.monotonic();full=m.run_simulator(bundle,x)
    assert full['dynamics_equivalent'] and full['configuration_valid']
    keys=('hidden_I','hidden_v','hidden_q','output_I','output_v','output_q','counts','logits')
    checks=[]
    for width in (7,13):
        chunked=m.run_simulator(bundle,x,chunk_steps=width)
        for key in keys:np.testing.assert_array_equal(full[key],chunked[key])
        checks.append(dict(chunk_steps=width,samples=len(x),differences=0))
    session=m._module('xylo_runtime').XyloInferenceSession(bundle)
    boundaries=(1,9,14,22,32,33,40)
    for i,raster in enumerate(x):
        session.reset();chunks=[];cursor=0
        for end in boundaries:
            part=session.push(raster[cursor:end]);chunks.append(part);cursor=end
            counts=full['output_q'][i,bundle['native_training_bridge']['readout_start']:end].sum(0)
            np.testing.assert_array_equal(part['counts'],counts)
        for key in keys[:6]:np.testing.assert_array_equal(np.concatenate([c[key] for c in chunks]),full[key][i])
        np.testing.assert_array_equal(part['logits'],full['logits'][i])
    np.savez_compressed(a.output/'results.npz',**{k:v for k,v in full.items() if isinstance(v,np.ndarray)})
    (a.output/'hardware-config.json').write_text(full['configuration_json'])
    report=dict(versions={name:md.version(name) for name in ('rockpool','xylosim','samna','numpy')},
        sdk_binary_sha256=full['sdk_binary_sha256'],bundle_sha256=sha(a.bundle),inputs_sha256=sha(a.inputs),
        results_sha256=sha(a.output/'results.npz'),configuration_sha256=sha(a.output/'hardware-config.json'),
        official_configuration_valid=True,full_vs_chunked=checks,
        persistent_session=dict(samples=len(x),boundaries=boundaries,differences=0),
        execution_seconds=time.monotonic()-start,labels_or_native_reference_read=False,physical_hardware=False)
    (a.output/'execution.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
