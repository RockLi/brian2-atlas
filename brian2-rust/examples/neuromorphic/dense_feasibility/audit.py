"""Audit the bounded feasibility probe and inspect the original MLP's bounds."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser()
for key in ('run','legacy-bundle','speck-probe'):p.add_argument('--'+key,type=Path,required=True)
a=p.parse_args();dest=a.run/'audit.json'
if dest.exists():raise FileExistsError(dest)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
atlas=np.load(a.run/'atlas.npz')['states'];sdk=np.load(a.run/'sdk/results.npz')['states'];np.testing.assert_array_equal(atlas,sdk)
execution=json.loads((a.run/'sdk/execution.json').read_text())
assert execution['bundle_sha256']==sha(a.run/'validated-dense/bundle.json')
assert execution['results_sha256']==sha(a.run/'sdk/results.npz')
plan=json.loads((a.run/'validated-dense/bundle.json').read_text())['plan']
assert len(atlas)==128**2 and len(plan['inputs'])==2
hidden=[plan['outputs'].index(k) for k in plan['hidden_states']]
assert atlas[:,hidden].max()==127 and atlas[:,hidden].min()==0 and atlas.min()<0
legacy=json.loads(a.legacy_bundle.read_text())['plan'];w=np.array(legacy['weights1'])
bias=np.array(legacy['bias1'])*(1<<np.array(legacy['bias_shift1']));step=1<<np.array(legacy['shift1'])
lo=np.minimum(w,0).sum(0)*127+bias+step//2;hi=np.maximum(w,0).sum(0)*127+bias+step//2
original=dict(shape=[w.shape[0],w.shape[1],len(legacy['outputs'])-len(legacy['hidden_states'])],
    weights1_range=[int(w.min()),int(w.max())],weights2_range=[int(np.array(legacy['weights2']).min()),int(np.array(legacy['weights2']).max())],
    hidden_shifts=sorted(set(legacy['shift1'])),first_affine_attainable_extrema=[int(lo.min()),int(hi.max())],
    hidden_units_exceeding_int16=int(((lo< -32768)|(hi>32767)).sum()),output_absolute_bound=max(legacy['output_absolute_bounds']),
    naive_bitplane_inputs=w.shape[0]*7,naive_hidden_comparators=w.shape[1]*127,
    naive_first_layer_connections=int(np.count_nonzero(w)*7*127),bundle_sha256=sha(a.legacy_bundle))
report=dict(schema='atlas.dense-feasibility.v1',passed=True,samples=len(atlas),compared_states=atlas.size,differences=0,
    scope='Exhaustive small-model mechanism validation, not full MNIST conversion or a public SpiNNaker2 dense profile',
    atlas_sha256=sha(a.run/'atlas.npz'),inputs_sha256=sha(a.run/'inputs.npz'),sdk=execution,
    original_mnist_model=original,specksim=json.loads(a.speck_probe.read_text()),physical_hardware=False)
dest.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(original,indent=2))
