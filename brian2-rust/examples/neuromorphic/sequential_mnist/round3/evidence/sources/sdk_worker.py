"""Label-free full-test execution, with bounded-size trace shards."""
import argparse
import hashlib
import importlib.metadata as md
import importlib.util
import json
from pathlib import Path
import time
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def integer_trace(value):
    value=np.asarray(value)
    assert np.isfinite(value).all() and (value==np.rint(value)).all()
    assert ((value>=-32768)&(value<=32767)).all()
    return value.astype(np.int32)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('runtime','bundle','inputs','output'):
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    spec=importlib.util.spec_from_file_location('runtime',a.runtime)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    bundle=json.loads(a.bundle.read_text());x=np.load(a.inputs,allow_pickle=False)['spikes']
    start=time.monotonic();receipts=[];checks=[];execution={}
    for i in range(0,len(x),64):
        result=m.run_simulator(bundle,x[i:i+64],batch_size=64)
        arrays=dict(spikes=integer_trace(result['spike_raster']),
            currents=integer_trace(np.concatenate((result['hidden_I'],result['output_I']),axis=2)),
            membranes=integer_trace(result['voltage']),logits=result['logits'])
        # Full-set baseline, plus bounded boundary checks for the first batch.
        if i==0:
            for width in (1,7,13):
                chunk=m.run_simulator(bundle,x[:64],batch_size=37,chunk_steps=width)
                for key in ('spike_raster','voltage','hidden_I','output_I','logits'):
                    np.testing.assert_array_equal(result[key],chunk[key])
                checks.append(dict(samples=min(64,len(x)),chunk_steps=width,differences=0))
            if bundle['target']=='xylosim':
                assert result['configuration_valid'] and result['dynamics_equivalent']
                (a.output/'hardware-config.json').write_text(result['configuration_json'])
                execution=dict(configuration_valid=True,sdk_binary_sha256=result['sdk_binary_sha256'],
                    configuration_sha256=sha(a.output/'hardware-config.json'),
                    versions={k:md.version(k) for k in ('rockpool','xylosim','samna','numpy')})
            else:
                assert result['execution_kind']=='custom-c-host-kernel'
                assert not result['physical_hardware'] and not result['firmware_adapter_implemented']
                execution=dict(kernel_receipt=result['kernel_receipt'],execution_kind=result['execution_kind'],
                    stock_sdk_neuron_model=False,firmware_adapter_implemented=False,
                    versions={k:md.version(k) for k in ('spinnaker2','numpy')})
        dest=a.output/f'{i:05d}.npz';np.savez_compressed(dest,**arrays)
        receipts.append(dict(start=i,samples=len(arrays['logits']),sha256=sha(dest)))
        if i%1024==0:print(json.dumps(dict(samples=min(i+64,len(x)),seconds=time.monotonic()-start)),flush=True)
    report=dict(schema='atlas.sequential-mnist.sdk-execution.v1',target=bundle['target'],samples=len(x),
        labels_or_native_reference_read=False,physical_hardware=False,hardware_mapping_tested=False,
        bundle_sha256=sha(a.bundle),inputs_sha256=sha(a.inputs),shards=receipts,chunk_checks=checks,
        execution_seconds=time.monotonic()-start,worker_sha256=sha(__file__),runtime_sha256=sha(a.runtime),
        readout_sha256=sha(a.runtime.with_name('xylo_readout.py')),**execution)
    (a.output/'execution.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('target','samples','execution_seconds')}),flush=True)


if __name__=='__main__':main()
