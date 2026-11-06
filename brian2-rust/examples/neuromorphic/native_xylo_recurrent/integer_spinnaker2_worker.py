"""Label-free execution of the custom SpiNNaker2 SDK C host kernel."""
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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('runtime', 'bundle', 'inputs', 'output'):
        p.add_argument('--' + key, type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location('runtime', a.runtime)
    runtime = importlib.util.module_from_spec(spec); spec.loader.exec_module(runtime)
    bundle = json.loads(a.bundle.read_text())
    x = np.load(a.inputs, allow_pickle=False)['spikes']
    start = time.monotonic()
    full = runtime.run_simulator(bundle, x, batch_size=64)
    assert full['execution_kind'] == 'custom-c-host-kernel' and full['dynamics_equivalent']
    assert not full['stock_sdk_neuron_model'] and not full['firmware_adapter_implemented']
    checks = []
    for width in (1, 7, 13):
        chunked = runtime.run_simulator(bundle, x, batch_size=37, chunk_steps=width)
        for key in ('spikes', 'currents', 'membranes', 'counts', 'logits'):
            np.testing.assert_array_equal(full[key], chunked[key])
        checks.append(dict(chunk_steps=width, batch_size=37, samples=len(x), differences=0))
    np.savez_compressed(a.output / 'results.npz', **{k:v for k,v in full.items() if isinstance(v, np.ndarray)})
    report = dict(schema='atlas.spinnaker2.integer-kernel-execution.v1',
        execution_kind=full['execution_kind'], stock_sdk_neuron_model=False,
        firmware_adapter_implemented=False, physical_hardware=False, hardware_mapping_tested=False,
        labels_or_native_reference_read=False, baseline_batch_size=64, full_vs_chunks=checks,
        kernel_receipt=full['kernel_receipt'], versions={name:md.version(name) for name in ('spinnaker2','numpy')},
        bundle_sha256=sha(a.bundle), inputs_sha256=sha(a.inputs), results_sha256=sha(a.output/'results.npz'),
        worker_sha256=sha(__file__), runtime_sha256=sha(a.runtime),
        integer_runtime_sha256=sha(a.runtime.with_name('spinnaker2_integer_runtime.py')),
        execution_seconds=time.monotonic()-start)
    (a.output/'execution.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__ == '__main__':
    main()
