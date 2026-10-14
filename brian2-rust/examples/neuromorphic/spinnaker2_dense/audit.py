"""Bind full-test SDK results to fresh Atlas, original frozen weights and MNIST."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser(description=__doc__)
for key in ('run','trained','dataset','historical'):
    p.add_argument('--'+key,type=Path,required=True)
a = p.parse_args()
sha = lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
bundle = json.loads((a.run/'converted/bundle.json').read_text())
plan = bundle['plan']
sdk = json.loads((a.run/'sdk/execution.json').read_text())
atlas_execution = json.loads((a.run/'atlas-execution.json').read_text())
training = json.loads((a.trained/'training.json').read_text())
assert sha(a.trained/'weights.npz') == training['weights_sha256']
assert sha(a.dataset) == training['dataset_sha256']
assert sha(a.run/'converted/bundle.json') == sdk['bundle_sha256']
assert sha(a.run/'converted/source.json') == bundle['source_model_sha256'] == sdk['source_model_sha256'] == atlas_execution['source_model_sha256']
assert sha(a.run/'sdk/results.npz') == sdk['results_sha256']
assert sha(a.run/'atlas.npz') == atlas_execution['states_sha256']
assert sha(a.run/'inputs.npz') == sdk['inputs_sha256'] == atlas_execution['inputs_sha256']
w = np.load(a.trained/'weights.npz',allow_pickle=False)
data = np.load(a.dataset,allow_pickle=False)
x = np.load(a.run/'inputs.npz',allow_pickle=False)['inputs']
labels = data['y_test']
np.testing.assert_array_equal(x,(data['x_test']//2).reshape(-1,784))
assert x.shape == (10000,784)
indices = [plan['outputs'].index(f's{i}') for i in range(10)]
np.testing.assert_array_equal(plan['weights1'],w['w1'])
np.testing.assert_array_equal(np.array(plan['weights2'])[:,indices],w['w2'])
np.testing.assert_array_equal(plan['shift1'],np.full(256,int(w['shift'])))
np.testing.assert_array_equal(np.array(plan['bias1'])*(1<<np.array(plan['bias_shift1'])),w['b1']*(1<<w['bs1']))
np.testing.assert_array_equal((np.array(plan['bias2'])*(1<<np.array(plan['bias_shift2'])))[indices],w['b2']*(1<<w['bs2']))
actual = np.load(a.run/'sdk/results.npz')['states']
atlas = np.load(a.run/'atlas.npz')['states']
history = np.load(a.historical)['atlas_states']
np.testing.assert_array_equal(actual.astype('<f8').view('<u8'),atlas.astype('<f8').view('<u8'))
np.testing.assert_array_equal(actual,history)
assert actual.shape == (10000,266)
predictions = actual[:,indices].argmax(1)
correct = int(np.sum(predictions == labels))
assert correct == 9874
confusion = np.zeros((10,10),dtype=np.int64)
np.add.at(confusion,(labels,predictions),1)
resources = plan['resources']
report = dict(schema='atlas.spinnaker2.dense-mnist-audit.v1',passed=True,
    samples=len(x),correct=correct,accuracy=correct/len(x),accuracy_loss_percentage_points=0.,
    compared_states=actual.size,differing_states=0,prediction_disagreements=0,
    hidden_state_values=256*len(x),score_values=10*len(x),
    exact_as_float64_bits=True,same_frozen_weights_and_biases=True,same_input_preprocessing=True,
    reference_kind=atlas_execution['reference_kind'],training_framework='external TensorFlow',
    training_repeated=False,confusion_matrix=confusion.tolist(),
    source_model_sha256=bundle['source_model_sha256'],weights_sha256=sha(a.trained/'weights.npz'),
    dataset_sha256=sha(a.dataset),historical_reference_sha256=sha(a.historical),
    resources=resources,total_connections=sum(v for k,v in resources.items() if k.endswith('_connections')),
    naive_comparator_neurons=256*127,naive_first_layer_connections=int(np.count_nonzero(w['w1']))*7*127,
    sdk=sdk,atlas=atlas_execution,physical_hardware=False,hardware_mapping_tested=False,
    scope='Exact static integer MLP conversion in the stock SDK CPU simulator. '
          'Internal spikes encode arithmetic; no temporal-SNN portability or chip power claim.')
(a.run/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:report[k] for k in ('samples','correct','accuracy','compared_states','differing_states','resources','total_connections')},indent=2))
