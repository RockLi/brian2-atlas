"""Independently run two order-sensitive source networks with Brian2 NumPy."""
import argparse
import hashlib
import json
from pathlib import Path
import brian2 as b
import numpy as np
from compare import build,cases
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
a=p.parse_args();output=a.run/'brian2-check';output.mkdir(exist_ok=False)
reports=[]
for name in ('recurrent','mixed_saturation_order'):
    c=cases()[name];net=build(c);net.run(len(c['raster'])*b.ms,namespace={})
    atlas=np.load(a.run/name/'atlas.npz');values={}
    for role in ('hidden','output'):
        mon=net[role+'_state']
        for key in ('I','v','q'):
            values[role+'_'+key]=np.asarray(getattr(mon,key)).T
            np.testing.assert_array_equal(values[role+'_'+key],atlas[role+'_'+key])
    path=output/(name+'.npz');np.savez_compressed(path,**values)
    reports.append(dict(case=name,state_values=sum(v.size for v in values.values()),differences=0,
                        result_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
(output/'audit.json').write_text(json.dumps(dict(backend='Brian2 NumPy',brian2_version=b.__version__,cases=reports),indent=2)+'\n')
print('Brian2 NumPy matched native Atlas in both order-sensitive cases.')
