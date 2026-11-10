from pathlib import Path
import sys,json,numpy as np,time,argparse
repo=Path('/Users/rock/workspace/bettiai/brian2-atlas');sys.path.insert(0,str(repo/'brian2-rust/examples/neuromorphic/sequential_mnist'))
from train import deskew_images,encode,sha
from brian2_atlas import NativeXyloTrainer
p=argparse.ArgumentParser();p.add_argument('--trained',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False,parents=True)
f=json.loads((a.trained/'frozen.json').read_text());r=json.loads((a.trained/'training.json').read_text());assert sha(a.trained/'frozen.json')==r['frozen_file_sha256']
t=NativeXyloTrainer(f['plan'],runner='/private/tmp/atlas-smnist-count-trainer',weights=[np.asarray(w)/f['plan']['weight_scale'] for w in f['target_weights']])
with np.load('/Users/rock/workspace/bettiai/brian2/brian2-rust/experiments/akida-mnist/output/cache/datasets/mnist.npz') as d:images=d['x_train'][54000:];labels=d['y_train'][54000:]
if r.get('deskew'):images=deskew_images(images)
offsets=[(0,0),(0,-1),(0,1),(-1,0),(1,0),(-1,-1),(-1,1),(1,-1),(1,1)];all_counts=[];start=time.monotonic()
for dy,dx in offsets:
 shifted=np.zeros_like(images)
 shifted[:,max(0,dy):min(28,28+dy),max(0,dx):min(28,28+dx)]=images[:,max(0,-dy):min(28,28-dy),max(0,-dx):min(28,28-dx)]
 x=encode(shifted,r['encoding'],r['repeats'],query_ticks=r.get('query_ticks',8));counts=[]
 for i in range(0,len(x),64):
  result=t.evaluate(x[i:i+64],labels[i:i+64]);physical=np.array(result['spikes'],dtype=np.int64)[:,f['plan']['readout_start']:,-f['plan']['sizes'][-1]:].sum(1)
  cc=physical@np.asarray(f['plan']['class_codes'],dtype=np.int64).T
  np.testing.assert_array_equal(cc.astype(float)*f['plan']['logit_scale']/(r['steps']-f['plan']['readout_start']),result['logits']);counts.extend(cc)
 all_counts.append(counts);print(json.dumps(dict(offset=[dy,dx],accuracy=float(np.mean(np.argmax(counts,1)==labels)),seconds=time.monotonic()-start)),flush=True)
counts=np.asarray(all_counts);metrics={str(n):float(np.mean(counts[:n].sum(0).argmax(1)==labels)) for n in [1,3,5,9]}
report=dict(schema='atlas.sequential-mnist.validation-views.v2',samples=6000,test_data_read=False,model_sha256=sha(a.trained/'frozen.json'),offsets=offsets,aggregation='sum integer class counts; same weights; independent zero state for every view; lowest-index tie',validation_accuracy=metrics,seconds=time.monotonic()-start,source_sha256=sha(__file__))
np.savez_compressed(a.output/'predictions.npz',class_counts=counts,labels=labels);report['predictions_sha256']=sha(a.output/'predictions.npz')
(a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
