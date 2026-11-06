from pathlib import Path
import importlib.util,json,hashlib,time
import numpy as np
import argparse
p=argparse.ArgumentParser(description='Execute and compare prepared small SDK regression cases.')
p.add_argument('--run',type=Path,required=True);p.add_argument('--runtime',type=Path,required=True)
a=p.parse_args();dest=a.run
spec=importlib.util.spec_from_file_location('runtime',a.runtime);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
report=json.loads((dest/'preparation.json').read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
for row in report:
 if row['status']=='rejected':continue
 path=dest/row['case']/'bundle.json';bundle=json.loads(path.read_text());x=np.load(row['inputs'])['inputs'];t=time.monotonic()
 result=mod.run_simulator(bundle,x);outfile=dest/row['case']/'results.npz';np.savez_compressed(outfile,**result)
 ref=np.load(row['reference'])['states'];np.testing.assert_array_equal(result['states'],ref)
 row.update(samples=len(x),state_values=ref.size,differences=0,elapsed_seconds=time.monotonic()-t,bundle_sha256=sha(path),results_sha256=sha(outfile),reference_sha256=sha(row['reference']),inputs_sha256=sha(row['inputs']))
 print(row['case'],'exact',ref.size,flush=True)
(dest/'audit.json').write_text(json.dumps(dict(passed=True,cases=report),indent=2)+'\n')
