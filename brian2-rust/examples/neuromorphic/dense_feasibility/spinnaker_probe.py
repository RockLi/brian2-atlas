"""Execute a bit-plane/comparator MLP using stock SpiNNaker2 SDK neurons."""
import argparse,hashlib,importlib.metadata as md,json
from pathlib import Path
import numpy as np
from spinnaker2 import snn,brian2_sim
from spinnaker2.neuron_models import NEURON_MODEL_TO_APPLICATION
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
dest=a.run/'sdk';dest.mkdir(exist_ok=False)
path=a.run/'validated-dense/bundle.json';plan=json.loads(path.read_text())['plan']
x=np.load(a.run/'inputs.npz')['inputs'];w=np.array(plan['weights1']);h=w.shape[1]
bias=np.array(plan['bias1'])*(1<<np.array(plan['bias_shift1']))
step=1<<np.array(plan['shift1']);assert np.all(step==128)
bits=7;banks=[(j,k) for j in range(h) for k in range(1,128)]
source=snn.Population(x.shape[1]*bits,'spike_list',
    {i*bits+b:np.flatnonzero((x[:,i]>>b)&1).tolist() for i in range(x.shape[1]) for b in range(bits)},name='bitplanes')
post=snn.Population(len(banks),'lif_32bit',dict(alpha_decay=0.,
    threshold=[k*step[j]-.5 for j,k in banks],i_offset=[int(bias[j]+step[j]//2) for j,k in banks],
    reset='reset_to_v_reset',v_reset=0.),name='hidden_comparators',record=['spikes'])
# Reading the final voltage is observation; the affine sum is executed by SDK synapses.
final_bias=np.array(plan['bias2'])*(1<<np.array(plan['bias_shift2']))
readout=snn.Population(len(plan['outputs']),'lif_32bit',dict(alpha_decay=0.,threshold=2.**23,
    i_offset=final_bias.tolist(),reset='reset_to_v_reset',v_reset=0.),name='affine',record=['v'])
conn=[[i*bits+b,k,int(w[i,j])*(1<<b),0] for k,(j,_) in enumerate(banks) for i in range(w.shape[0]) for b in range(bits) if w[i,j]]
w2=np.asarray(plan['weights2']);out_conn=[[k,i,int(v),0] for k,(j,_) in enumerate(banks) for i,v in enumerate(w2[j]) if v]
max_weight=NEURON_MODEL_TO_APPLICATION['lif_32bit'].sw_spec.max_weight()
assert max(abs(r[2]) for r in conn+out_conn)<=max_weight
net=snn.Network('dense_feasibility');net.add(source,post,readout,snn.Projection(source,post,conn),snn.Projection(post,readout,out_conn))
backend=brian2_sim.Brian2Backend();backend.run(net,len(x)+2,quantize_weights=True)
# Fixed two-layer pipeline latency follows the SDK schedule, not fitted output alignment.
states=np.stack([readout.get_voltages()[i] for i in range(readout.size)],axis=1)[2:len(x)+2]
np.savez_compressed(dest/'results.npz',states=states)
report=dict(samples=len(x),input_neurons=source.size,comparator_neurons=post.size,readout_neurons=readout.size,
    connections=len(conn)+len(out_conn),max_sdk_weight=max_weight,neuron_model='lif_32bit',pipeline_ticks=2,
    versions={k:md.version(k) for k in ('spinnaker2','brian2','numpy')},
    bundle_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    sdk_source_sha256=hashlib.sha256(Path(brian2_sim.__file__).read_bytes()).hexdigest(),
    results_sha256=hashlib.sha256((dest/'results.npz').read_bytes()).hexdigest(),physical_hardware=False)
(dest/'execution.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
