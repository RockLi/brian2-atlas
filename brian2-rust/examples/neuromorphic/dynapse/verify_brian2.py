"""Independent source execution for representative circuit-model cases."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import brian2 as b
from compare import cases
from brian2_rust.dynapse_model import build,STATES
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
dest=a.run/'brian2';dest.mkdir(exist_ok=False);reports=[]
for name in ('constant','adaptation','eight_neurons'):
    c=cases()[name];net=build(**c);net.run(len(c['raster'])*b.ms,namespace={})
    atlas=np.load(a.run/name/'atlas.npz');values={}
    for key in STATES:
        values[key]=np.asarray(getattr(net['dynapse_state'],key)).T
        np.testing.assert_allclose(values[key],atlas[key],rtol=1e-12,atol=1e-24)
    mon=net['dynapse_events'];events=np.column_stack((np.rint(mon.t/b.ms).astype(int),np.asarray(mon.i)))
    np.testing.assert_array_equal(events,atlas['spikes'])
    path=dest/(name+'.npz');np.savez_compressed(path,**values)
    reports.append(dict(case=name,state_values=sum(v.size for v in values.values()),
        spike_events=len(events),result_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
(dest/'audit.json').write_text(json.dumps(dict(backend='Brian2 NumPy',cases=reports,rtol=1e-12,atol=1e-24),indent=2)+'\n')
print('Independent Brian2 source comparisons passed:',len(reports))
