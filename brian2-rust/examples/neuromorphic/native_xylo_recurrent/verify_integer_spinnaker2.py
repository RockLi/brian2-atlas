"""Same frozen recurrent model: native Rust, XyloSim, custom SpiNNaker2 C kernel."""
import argparse
import json
import subprocess
from pathlib import Path

import brian2 as b
import numpy as np
from brian2_atlas import NativeXyloTrainer, export_network
from brian2_atlas.neuromorphic import export_native_xylo
from brian2_rust.native_xylo_conversion import validate_frozen, build_native_xylo_network
from brian2_rust.results import load_results
from train import data, sha, STEPS, READOUT_START
from verify import accuracy, native
from probe_spinnaker2 import difference, sdk_run


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('train-runner','runner','xylo-python','spinnaker2-python','output'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--trained',type=Path,default=Path(__file__).with_name('fixtures'))
    a = p.parse_args(); a.output = a.output.resolve(); a.output.mkdir(parents=True,exist_ok=False)
    here = Path(__file__).resolve().parent
    f = json.loads((a.trained/'frozen.json').read_text()); sizes = validate_frozen(f)
    training = json.loads((a.trained/'training.json').read_text())
    assert sha(a.trained/'frozen.json') == training['frozen_file_sha256']
    assert f['schema'] == 'atlas.native-xylo-frozen.v2' and f['plan']['readout_start'] == READOUT_START
    x,y,ids = data('test')
    for split in ('train','validation'):
        assert not set(ids).intersection(data(split)[2])
    np.savez_compressed(a.output/'inputs.npz',spikes=x)
    trainer = NativeXyloTrainer(f['plan'],runner=a.train_runner,
        weights=[np.asarray(w)/f['plan']['weight_scale'] for w in f['target_weights']])
    reference = native(trainer,x,y)
    b.prefs.codegen.target = 'numpy'
    runtime = here.parents[2]/'python/brian2_rust/neuromorphic_runtime.py'
    comparisons, executions, logits = {}, {}, {}
    for target, python, worker in [('xylosim',a.xylo_python,'sdk_worker.py'),
                                  ('spinnaker2',a.spinnaker2_python,'integer_spinnaker2_worker.py')]:
        bundle = export_native_xylo(f,a.output/target,steps=STEPS,runner=a.runner,target=target)
        for bank,role,shape in zip(f['target_weights'],('input','readout','recurrent'),
                                  ((sizes[0],sizes[1]),(sizes[1],sizes[2]),(sizes[1],sizes[1]))):
            np.testing.assert_array_equal(bundle['plan']['weights'][role],np.asarray(bank).reshape(shape))
        sdk_run(python,here/worker,['--runtime',runtime,'--bundle',a.output/target/'converted/bundle.json',
            '--inputs',a.output/'inputs.npz','--output',a.output/(target+'-results')],a.output/(target+'.log'))
        dest = a.output/(target+'-results')
        execution = json.loads((dest/'execution.json').read_text())
        assert execution['inputs_sha256'] == sha(a.output/'inputs.npz')
        assert execution['results_sha256'] == sha(dest/'results.npz')
        result = np.load(dest/'results.npz',allow_pickle=False)
        arrays = dict(spikes=result['spike_raster'],membranes=result['voltage'],logits=result['logits'],
            currents=np.concatenate((result['hidden_I'],result['output_I']),axis=2))
        comparisons[target] = {key:difference(reference[key],arrays[key]) for key in reference}
        assert all(item['differences'] == 0 for item in comparisons[target].values())
        executions[target] = execution; logits[target] = arrays['logits']
    receipts = []; state_values = 0
    for index in range(32):
        dest = a.output/'atlas-ir-replay'/str(index); dest.mkdir(parents=True)
        previous = b.prefs.codegen.loop_invariant_optimisations
        try:
            b.prefs.codegen.loop_invariant_optimisations = False
            model = export_network(build_native_xylo_network(f,x[index]),STEPS*b.ms,dest/'source.json')
        finally:
            b.prefs.codegen.loop_invariant_optimisations = previous
        subprocess.run([str(a.runner.resolve()),str(dest/'source.json'),str(dest/'results')],capture_output=True,check=True)
        replay = load_results(model,dest/'results')
        for role in ('hidden','output'):
            pop = replay['populations'][bundle['plan']['roles'][role]]
            for key in ('I','v','q'):
                np.testing.assert_array_equal(pop['trace'][key],result[role+'_'+key][index])
                state_values += pop['trace'][key].size
            np.testing.assert_array_equal(np.column_stack((pop['spike_ticks'],pop['indices'])),
                                          np.column_stack(np.nonzero(result[role+'_q'][index])))
        receipts.append(dict(sample=index,source_sha256=sha(dest/'source.json'),
            results_sha256=sha(dest/'results/results.bin'),events_sha256=sha(dest/'results/events.bin')))
    np.savez_compressed(a.output/'per_sample.npz',labels=y,template_ids=ids,native_logits=reference['logits'],
                        xylo_logits=logits['xylosim'],spinnaker2_integer_logits=logits['spinnaker2'])
    report = dict(schema='atlas.recurrent-spinnaker2-integer-audit.v1',validation_passed=True,
        execution_kind='custom-c-host-kernel',stock_sdk_neuron_model=False,firmware_adapter_implemented=False,
        same_model=True,same_weight_banks=True,retraining=False,clock_shift=0,numerical_tolerance=0,
        task='Delayed AB/BA discrimination',samples=len(y),templates=len(set(ids)),sizes=sizes,
        steps=STEPS,readout_start=READOUT_START,native=accuracy(reference['logits'],y),
        xylosim=accuracy(logits['xylosim'],y),spinnaker2_integer=accuracy(logits['spinnaker2'],y),
        comparisons=comparisons,sdk_executions=executions,
        atlas_ir_replay=dict(samples=32,state_values=state_values,differences=0,receipts=receipts),
        frozen_sha256=f['sha256'],frozen_file_sha256=sha(a.trained/'frozen.json'),
        inputs_sha256=sha(a.output/'inputs.npz'),per_sample_sha256=sha(a.output/'per_sample.npz'),
        train_runner_sha256=sha(a.train_runner),atlas_runner_sha256=sha(a.runner),
        physical_hardware=False,hardware_mapping_tested=False,
        source_hashes={path.name:sha(path) for path in (Path(__file__),here/'integer_spinnaker2_worker.py',
            here/'train.py',here/'verify.py',here/'probe_spinnaker2.py',runtime,
            runtime.with_name('spinnaker2_integer_runtime.py'),runtime.with_name('spinnaker2_integer_b2ir.py'),
            runtime.with_name('native_xylo_conversion.py'))})
    (a.output/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({key:report[key] for key in ('validation_passed','native','xylosim','spinnaker2_integer','comparisons')},indent=2))


if __name__ == '__main__':
    main()
