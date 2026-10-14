"""Replay native Atlas and audit the two real SDK runs against frozen TF.

The SDK workers receive only a converted bundle and inputs. This audit reads
labels and references afterwards. It requires the reproducible common example.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import numpy as np
from brian2_rust.protocol import attach_protocol, canonical_bytes
from brian2_rust.results import load_results


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('common','run','trained','runner'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    output=a.run/'verified';output.mkdir(exist_ok=False)
    prep=json.loads((a.common/'preparation.json').read_text())
    for name,key in [('inputs.npz','inputs_sha256'),('labels.npz','labels_sha256'),
                     ('atlas.npz','atlas_results_sha256'),('source.json','source_model_sha256')]:
        actual = (hashlib.sha256(canonical_bytes(json.loads((a.common/name).read_text()))).hexdigest()
                  if name=='source.json' else sha(a.common/name))
        assert actual==prep[key], name
    assert sha(a.trained/'weights.npz')==prep['weights_sha256']
    assert sha(a.runner)==prep['runner_sha256']
    tfr=json.loads((a.common/'tf/execution.json').read_text())
    assert sha(a.common/'tf/results.npz')==tfr['results_sha256']
    assert tfr['inputs_sha256']==prep['inputs_sha256']
    for name,identity in tfr['model_files'].items():
        assert sha(a.trained/'saved_model'/name)==identity
    x=np.load(a.common/'inputs.npz')['inputs']
    probes=np.load(a.common/'probes/inputs.npz')['inputs']
    labels=np.load(a.common/'labels.npz')['labels']
    tf=np.load(a.common/'tf/results.npz')['counts']
    assert len(x)==len(labels)==10000
    model=json.loads((a.common/'source.json').read_text())
    names=prep['state_names'];scores=[names.index(n) for n in prep['score_names']]
    inputs=sorted(model['instance']['populations'][0]['parameters'])
    size=model['definition']['populations'][0]['count']
    batches=[];all_states=[]
    combined=np.concatenate([x,probes])
    env={**os.environ,'B2_MAX_POPULATION_PARAMETERS':'1024',
         'B2_MAX_POPULATION_STATES':'512','B2_MAX_STATEMENTS':'512'}
    for start in range(0,len(combined),size):
        rows=combined[start:start+size]
        padded=np.zeros((size,len(inputs)),dtype=np.uint8);padded[:len(rows)]=rows
        for i,name in enumerate(inputs):
            raw=padded[:,i].astype('>f8').tobytes().hex()
            model['instance']['populations'][0]['parameters'][name]=[raw[k:k+16] for k in range(0,len(raw),16)]
        attach_protocol(model); encoded=canonical_bytes(model)
        with tempfile.TemporaryDirectory(prefix='atlas-count-replay-') as directory:
            source=Path(directory)/'source.json';source.write_bytes(encoded)
            dest=Path(directory)/'results'
            run=subprocess.run([str(a.runner.resolve()),str(source),str(dest)],env=env,capture_output=True,text=True)
            if run.returncode:raise RuntimeError(run.stderr)
            states=load_results(model,dest)['populations'][0]['states']
            all_states.append(np.stack([states[n] for n in names],axis=1)[:len(rows)])
        batches.append(dict(start=start,count=len(rows),source_sha256=hashlib.sha256(encoded).hexdigest()))
        print('Atlas replay',start+len(rows),'/',len(combined),flush=True)
    states=np.concatenate(all_states)
    np.savez_compressed(output/'atlas.npz',states=states)
    np.testing.assert_array_equal(states[:len(x)],np.load(a.common/'atlas.npz')['states'])
    np.testing.assert_array_equal(states[:len(x),scores],tf)
    # Independent integer equations are diagnostic only, never SDK inference.
    checkpoint=np.load(a.trained/'weights.npz');w=checkpoint['weights'];theta=int(checkpoint['threshold'])
    np.testing.assert_array_equal(tf,x.astype(np.int64)@w//theta)
    smoke=np.load(a.run/'smoke.npz')['inputs']
    np.testing.assert_array_equal(smoke,np.concatenate([x[:20],probes]))
    expected_smoke=np.concatenate([states[:20],states[len(x):]])
    reports=[]
    for target in ('spinnaker2','xylosim'):
        bundle_path=a.run/target/'bundle.json';bundle=json.loads(bundle_path.read_text());plan=bundle['plan']
        assert bundle['source_model_sha256']==prep['source_model_sha256']
        assert sha(a.run/target/'source.json')==prep['source_model_sha256']
        np.testing.assert_array_equal(np.asarray(plan['weights']).T,w)
        assert plan['threshold']==theta
        executions={}
        for kind,inputs_path,expected in [('full',a.common/'inputs.npz',states[:len(x)]),
                                          ('smoke',a.run/'smoke.npz',expected_smoke)]:
            directory=a.run/(target+'-'+kind)
            report=json.loads((directory/'execution.json').read_text())
            assert report['bundle_sha256']==sha(bundle_path)
            assert report['inputs_sha256']==sha(inputs_path)
            assert report['results_sha256']==sha(directory/'results.npz')
            api=Path(__file__).resolve().parents[3]/'python/brian2_rust'
            for filename,identity in report['implementation_sha256'].items():
                assert sha(api/filename)==identity
            data=np.load(directory/'results.npz')
            np.testing.assert_array_equal(data['states'],expected)
            np.testing.assert_array_equal(data['counts'],expected[:,scores])
            if target=='xylosim':
                rows=np.load(inputs_path)['inputs'];summed=rows.astype(np.int64)@w
                bank_expected=[];current_expected=[];voltage_expected=[]
                for (c,_),bias in zip(plan['banks'],plan['biases']):
                    current=summed[:,c];voltage=current+bias
                    count=np.clip(voltage//theta,0,31)
                    bank_expected.append(count);current_expected.append(current)
                    voltage_expected.append(voltage-count*theta)
                for name,values in [('bank_spikes',bank_expected),('bank_current',current_expected),('bank_voltage',voltage_expected)]:
                    np.testing.assert_array_equal(data[name],np.array(values).T)
            executions[kind]=report
        prediction=np.load(a.run/(target+'-full')/'results.npz')['counts'].argmax(1)
        np.testing.assert_array_equal(prediction,tf.argmax(1))
        confusion=np.zeros((10,10),np.int64);np.add.at(confusion,(labels,prediction),1)
        reports.append(dict(target=target,samples=len(x),correct=int((prediction==labels).sum()),
            accuracy=float((prediction==labels).mean()),differing_states=0,state_values_compared=int(states[:len(x)].size),
            boundary_samples=len(probes),boundary_differing_states=0,resources=plan['resources'],
            hardware_input_limit_exceeded=plan.get('hardware_input_limit_exceeded'),
            confusion_matrix=confusion.tolist(),executions=executions))
    report=dict(schema='atlas.count-targets.audit.v1',passed=True,targets=reports,
        source_model_sha256=prep['source_model_sha256'],weights_sha256=prep['weights_sha256'],
        dataset_sha256=prep['dataset_sha256'],runner_sha256=sha(a.runner),
        fresh_atlas_results_sha256=sha(output/'atlas.npz'),batches=batches,
        same_frozen_tf_model=True,same_weights=True,same_source_ir=True,
        tensorflow_reference='Previously executed TF SavedModel; artifacts and model identities rechecked',
        fresh_atlas_execution=True,physical_hardware=False,hardware_mapping_tested=False,
        temporal_spike_equivalence_claimed=False,
        note='90.03% shared count classifier; not the 98.74% signed biased MLP.')
    (output/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    for row in reports:print(row['target'],row['correct'],row['samples'],row['accuracy'])


if __name__=='__main__':main()
