from pathlib import Path
import os,sys,json,subprocess,numpy as np
repo=Path('/Users/rock/workspace/bettiai/brian2-atlas');example=repo/'brian2-rust/examples/neuromorphic/sequential_mnist';root=Path('/Volumes/T7/atlas-sequential-mnist-round3-20261014');out=root/'wide-preflight';out.mkdir(exist_ok=False)
sys.path.insert(0,str(example));from train import widen_weights,recurrent_group_mask,project_recurrence,encode,sha
from brian2_atlas import NativeXyloTrainer,xylo_recurrent_training_plan
from brian2_atlas.neuromorphic import freeze_native_xylo,export_native_xylo
parent=json.loads((example/'refinement/fixtures/frozen.json').read_text());plan=parent['plan'].copy();plan['sizes']=[16,126,5]
t=NativeXyloTrainer(plan,runner=root/'b2-train-wide',weights=widen_weights(parent,plan,2,.03,np.random.default_rng(123)))
with np.load('/Users/rock/workspace/bettiai/brian2/brian2-rust/experiments/akida-mnist/output/cache/datasets/mnist.npz') as d:x=encode(d['x_train'][:4],'rows-full-counts',deskew=True);y=d['y_train'][:4]
mask=recurrent_group_mask(126,2);t.step(x,y);project_recurrence(t,mask);f=freeze_native_xylo(t);ref=t.evaluate(x,y)
np.savez_compressed(out/'inputs.npz',spikes=x)
import brian2 as b
b.prefs.codegen.target='numpy';b.prefs.codegen.loop_invariant_optimisations=False
for target,py in [('xylosim','/private/tmp/atlas-xylo-env/bin/python'),('spinnaker2','/private/tmp/atlas-spinnaker2-env312/bin/python')]:
 export_native_xylo(f,out/target,steps=36,runner=root/'b2-runner-wide',target=target)
 with (out/(target+'.log')).open('w') as log:
  subprocess.run([py,str(example/'sdk_worker.py'),'--runtime',str(repo/'brian2-rust/python/brian2_rust/neuromorphic_runtime.py'),'--bundle',str(out/target/'converted/bundle.json'),'--inputs',str(out/'inputs.npz'),'--output',str(out/(target+'-results'))],env={**os.environ,'PYTHONPATH':'','PYTHONNOUSERSITE':'1'},cwd=out,stdout=log,stderr=subprocess.STDOUT,check=True)
 with np.load(out/(target+'-results/00000.npz')) as z:
  for k in ('spikes','currents','membranes','logits'):np.testing.assert_array_equal(ref[k],z[k])
 print(target,'exact',flush=True)
(out/'receipt.json').write_text(json.dumps(dict(samples=4,source_split='training',sizes=[16,126,5],groups=2,max_recurrent_fanout=63,neural_values=4*36*131*3,differences=0,trainer_sha256=sha(root/'b2-train-wide'),physical_hardware=False),indent=2)+'\n')
