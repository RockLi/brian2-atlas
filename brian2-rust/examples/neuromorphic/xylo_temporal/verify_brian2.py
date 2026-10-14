"""Independently compare source networks with Brian2 NumPy."""
import argparse
import hashlib
import json
from pathlib import Path
import brian2 as b
import numpy as np
from compare import build,cases
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
p.add_argument('--coverage',action='store_true')
a=p.parse_args();output=a.run/'brian2-check';output.mkdir(exist_ok=False)
reports=[]
networks=cases();names=['recurrent','mixed_saturation_order']
if a.coverage:
    from coverage_cases import cases as coverage_cases
    networks.update(coverage_cases())
    names += ['random_00','random_31','one_tick','one_neuron','maximum_profile_dimensions','chunk_boundary_impulse']
for name in names:
    c=networks[name];net=build(c);net.run(len(c['raster'])*b.ms,namespace={})
    atlas=np.load(a.run/name/'atlas.npz');values={}
    for role in ('hidden','output'):
        mon=net[role+'_state']
        for key in ('I','v','q'):
            values[role+'_'+key]=np.asarray(getattr(mon,key)).T
            np.testing.assert_array_equal(values[role+'_'+key],atlas[role+'_'+key])
    for role in ('hidden','output'):
        mon=net[role+'_spikes']
        events=np.column_stack((np.rint(np.asarray(mon.t/b.ms)).astype(int),np.asarray(mon.i)))
        np.testing.assert_array_equal(events,atlas[role+'_events'])
    path=output/(name+'.npz');np.savez_compressed(path,**values)
    reports.append(dict(case=name,state_values=sum(v.size for v in values.values()),differences=0,
                        result_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
(output/'audit.json').write_text(json.dumps(dict(backend='Brian2 NumPy',brian2_version=b.__version__,cases=reports),indent=2)+'\n')
print(f'Brian2 NumPy matched native Atlas states and events in {len(reports)} cases.')
