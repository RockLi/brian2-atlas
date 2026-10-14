"""Fail on any TF/Atlas/SDK mismatch; record full-set classification accuracy."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--trained',type=Path,required=True)
    a=p.parse_args();root=a.run
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    prep=json.loads((root/'preparation.json').read_text())
    assert sha(root/'inputs.npz')==prep['inputs_sha256']
    assert sha(root/'labels.npz')==prep['labels_sha256']
    assert sha(root/'atlas.npz')==prep['atlas_results_sha256']
    assert sha(a.trained/'weights.npz')==prep['weights_sha256']
    with np.load(root/'inputs.npz',allow_pickle=False) as d:x=d['inputs']
    with np.load(root/'labels.npz',allow_pickle=False) as d:labels=d['labels']
    with np.load(root/'atlas.npz',allow_pickle=False) as d:atlas=d['states']
    with np.load(a.trained/'weights.npz',allow_pickle=False) as d:w,theta=d['weights'],int(d['threshold'])
    with np.load(root/'tf/results.npz',allow_pickle=False) as d:tf=d['counts']
    tfr=json.loads((root/'tf/execution.json').read_text())
    assert tfr['inputs_sha256']==sha(root/'inputs.npz') and tfr['results_sha256']==sha(root/'tf/results.npz')
    for rel,expected in tfr['model_files'].items():assert sha(a.trained/'saved_model'/rel)==expected
    indices=[prep['state_names'].index(n) for n in prep['score_names']]
    np.testing.assert_array_equal(tf,x.astype(np.int64)@w//theta)
    np.testing.assert_array_equal(atlas[:,indices],tf)
    reports=[]
    for target in ('akida','lava','specksim'):
        bundle_path=root/target/'bundle.json';bundle=json.loads(bundle_path.read_text())
        assert sha(root/target/'source.json')==prep['source_model_sha256']
        execution=json.loads((root/(target+'-run')/'execution.json').read_text())
        assert execution['bundle_sha256']==sha(bundle_path)
        assert execution['inputs_sha256']==sha(root/'inputs.npz')
        assert execution['source_model_sha256']==bundle['source_model_sha256']==prep['source_model_sha256']
        result_path=root/(target+'-run')/'results.npz'
        assert execution['results_sha256']==sha(result_path)
        with np.load(result_path,allow_pickle=False) as d:
            states=d['states']
            np.testing.assert_array_equal(states,atlas)
            if target=='specksim':
                np.testing.assert_array_equal(d['counts'],tf)
                np.testing.assert_array_equal(d['voltage'],x.astype(np.int64)@w%theta)
        prediction=states[:,indices].argmax(1)
        np.testing.assert_array_equal(prediction,tf.argmax(1))
        confusion=np.zeros((10,10),np.int64);np.add.at(confusion,(labels,prediction),1)
        reports.append(dict(target=target,samples=len(labels),correct=int((prediction==labels).sum()),
                            accuracy=float((prediction==labels).mean()),state_values_compared=int(atlas.size),
                            differing_states=0,prediction_disagreements=0,versions=execution['versions'],
                            confusion_matrix=confusion.tolist()))
    report=dict(schema='atlas.neuromorphic.same-tf-model-audit.v1',source_model_sha256=prep['source_model_sha256'],
                weights_sha256=prep['weights_sha256'],dataset_sha256=prep['dataset_sha256'],
                atlas_runner_sha256=prep['runner_sha256'],tensorflow_version=tfr['tensorflow_version'],targets=reports,
                same_frozen_model=True,same_weights=True,same_source_ir=True,fresh_atlas_execution=True,
                physical_hardware=False,hardware_mapping_tested=False,
                temporal_spike_equivalence_claimed=False,
                note='Restricted quantized TF linear model; not the historical 98.74% signed biased MLP.')
    (root/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='targets'},indent=2))
    for row in reports:print(row['target'],row['correct'],row['samples'],row['accuracy'])

if __name__=='__main__':main()
