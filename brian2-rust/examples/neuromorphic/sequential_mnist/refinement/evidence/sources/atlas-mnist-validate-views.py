from pathlib import Path
import sys,json,numpy as np,time
repo=Path('/Users/rock/workspace/bettiai/brian2-atlas');root=Path('/Volumes/T7/atlas-sequential-mnist-round2-20261014')
sys.path.insert(0,str(repo/'brian2-rust/examples/neuromorphic/sequential_mnist'))
from train import deskew_images,encode,sha
from brian2_atlas import NativeXyloTrainer
source=root/'precision';f=json.loads((source/'frozen.json').read_text());r=json.loads((source/'training.json').read_text())
t=NativeXyloTrainer(f['plan'],runner='/private/tmp/atlas-smnist-count-trainer',weights=[np.asarray(w)/f['plan']['weight_scale'] for w in f['target_weights']])
with np.load('/Users/rock/workspace/bettiai/brian2/brian2-rust/experiments/akida-mnist/output/cache/datasets/mnist.npz') as d:images=deskew_images(d['x_train'][54000:]);labels=d['y_train'][54000:]
offsets=[(0,0),(0,-1),(0,1),(-1,0),(1,0),(-1,-1),(-1,1),(1,-1),(1,1)];all_logits=[];start=time.monotonic()
for dy,dx in offsets:
 shifted=np.zeros_like(images)
 shifted[:,max(0,dy):min(28,28+dy),max(0,dx):min(28,28+dx)]=images[:,max(0,-dy):min(28,28-dy),max(0,-dx):min(28,28-dx)]
 x=encode(shifted,r['encoding'],r['repeats'],query_ticks=r.get('query_ticks',8));scores=[]
 for i in range(0,len(x),64):scores.extend(t.evaluate(x[i:i+64],labels[i:i+64])['logits'])
 all_logits.append(scores);print(json.dumps(dict(offset=[dy,dx],accuracy=float(np.mean(np.argmax(scores,1)==labels)),seconds=time.monotonic()-start)),flush=True)
scores=np.asarray(all_logits);metrics={str(n):float(np.mean(scores[:n].sum(0).argmax(1)==labels)) for n in [1,3,5,9]}
report=dict(schema='atlas.sequential-mnist.validation-views.v1',samples=6000,test_data_read=False,model_sha256=sha(source/'frozen.json'),offsets=offsets,aggregation='sum class scores; fixed equal weights; lowest-index tie',validation_accuracy=metrics,seconds=time.monotonic()-start)
(root/'validation-views.json').write_text(json.dumps(report,indent=2)+'\n');np.savez_compressed(root/'validation-views.npz',logits=scores,labels=labels)
print(json.dumps(report),flush=True)
