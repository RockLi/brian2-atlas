from pathlib import Path
import json
from brian2_atlas.neuromorphic import export_ir
import argparse
p=argparse.ArgumentParser(description='Compile existing Atlas coverage fixtures for SpiNNaker2 dense.')
for key in ('static-run','exhaustive-run','runner','output'):p.add_argument('--'+key,type=Path,required=True)
a=p.parse_args();dest=a.output;dest.mkdir(parents=True,exist_ok=False)
runner=a.runner;report=[]
for case in sorted(a.static_run.glob('dense_*')):
 try:
  b=export_ir(case/'source.json',dest/case.name,target='spinnaker2',runner=runner)
 except ValueError as e:
  assert case.name in ['dense_13','dense_14','dense_15'] and '20-bit' in str(e)
  report.append(dict(case=case.name,status='rejected',reason=str(e)))
 else:
  report.append(dict(case=case.name,status='admitted',reference=str(case/'atlas.npz'),inputs=str(case/'inputs.npz')))
small=a.exhaustive_run
export_ir(small/'source.json',dest/'exhaustive',target='spinnaker2',runner=runner)
report.append(dict(case='exhaustive',status='admitted',reference=str(small/'atlas.npz'),inputs=str(small/'inputs.npz')))
(dest/'preparation.json').write_text(json.dumps(report,indent=2)+'\n')
print('prepared',len(report))
