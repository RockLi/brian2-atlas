from pathlib import Path
import importlib.util,json,numpy as np,time
from brian2_atlas import NativeXyloTrainer
r=Path('/Users/rock/workspace/bettiai/brian2-atlas/brian2-rust/examples/neuromorphic/sequential_mnist');spec=importlib.util.spec_from_file_location('train',r/'train.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
out=Path('/Volumes/T7/atlas-sequential-mnist-round4-20261014/readout-probe');out.mkdir()
methods=['counts','current','positive_current','pre_reset_voltage','counts_plus_final_residual'];windows=[8,12,20]
(out/'protocol.json').write_text(json.dumps(dict(methods=methods,readout_ticks=windows,samples='original train[54000:60000]',test_arrays_read=False),indent=2)+'\n')
with np.load('/Users/rock/workspace/bettiai/brian2/brian2-rust/experiments/akida-mnist/output/cache/datasets/mnist.npz') as d:x=m.encode(d['x_train'][54000:],'rows-full-counts',deskew=True);y=d['y_train'][54000:]
reports=[]
for name,path in [('compact',r/'round3/candidates/averaged-refine'),('wide',r/'round3/fixtures')]:
 f=json.loads((path/'frozen.json').read_text());p=f['plan'];trainer=NativeXyloTrainer(p,runner='/Volumes/T7/atlas-sequential-mnist-round4-20261014/b2-train',weights=[np.asarray(w)/p['weight_scale'] for w in f['target_weights']]);no=p['sizes'][-1];c=np.asarray(p['class_codes']);theta=p['threshold'][-1]
 arrays={k:[] for k in ('spikes','currents','membranes')}
 for i in range(0,len(x),64):
  rr=trainer.evaluate(x[i:i+64],y[i:i+64])
  for k in arrays:arrays[k].append(np.asarray(rr[k])[:,:,-no:])
 a={k:np.concatenate(v) for k,v in arrays.items()};np.savez_compressed(out/(name+'.npz'),**a)
 for ticks in windows:
  q=a['spikes'][:,-ticks:];v=a['membranes'][:,-ticks:];cur=a['currents'][:,-ticks:]
  features=dict(counts=q.sum(1),current=cur.sum(1),positive_current=np.maximum(cur,0).sum(1),pre_reset_voltage=(v+q*theta).sum(1),counts_plus_final_residual=q.sum(1)+np.clip(v[:,-1]/theta,0,1))
  for method,feature in features.items():
   logits=feature@c.T;pred=np.argmax(logits,1)
   item=dict(parent=name,readout_ticks=ticks,method=method,validation_accuracy=float(np.mean(pred==y)),ties=int(np.sum((logits==logits.max(1,keepdims=True)).sum(1)>1)))
   reports.append(item);print(json.dumps(item),flush=True)
(out/'report.json').write_text(json.dumps(dict(results=reports,test_arrays_read=False),indent=2)+'\n')
