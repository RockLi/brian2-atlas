"""Observe direct-conversion bias rejection and signed event-order sensitivity."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sinabs.layers import IAF
from sinabs.backend.dynapcnn.specksim import from_sequential
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
try:
    from_sequential(nn.Sequential(nn.Conv2d(1,1,(1,2),bias=True),IAF(spike_threshold=2,min_v_mem=-100)),input_shape=(1,1,2))
except ValueError as e:bias_error=str(e)
else:raise AssertionError('expected bias rejection')
conv=nn.Conv2d(1,1,(1,2),bias=False)
with torch.no_grad():conv.weight.copy_(torch.tensor([[[[2.,-1.]]]]))
net=from_sequential(nn.Sequential(conv,IAF(spike_threshold=2,min_v_mem=-100)),input_shape=(1,1,2))
results=[]
for order in ([0,1],[1,0]):
    net.reset_states();events=np.zeros(2,dtype=net.output_dtype);events['x']=order;events['t']=[0,1]
    out=net(events);results.append(dict(input_order=order,output_spikes=len(out),final_membrane=np.asarray(net.read_spiking_layer_states(0)).tolist()))
assert results[0]['output_spikes']!=results[1]['output_spikes']
a.output.write_text(json.dumps(dict(bias_error=bias_error,signed_order_cases=results,
    static_reference_count=0,scope='Counterexample to direct event-order-independent count mapping, not a proof that every encoding is impossible.'),indent=2)+'\n')
print(a.output.read_text())
