"""Check an existing signed/biased TF MLP without silently replacing its model."""
import argparse,json,os
from pathlib import Path
from brian2_atlas.neuromorphic import export_ir
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--source',type=Path,required=True)
p.add_argument('--runner',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
os.environ.update(B2_MAX_POPULATION_STATES='512',B2_MAX_POPULATION_PARAMETERS='1024',B2_MAX_STATEMENTS='512')
report={}
for target in ('akida','lava','specksim'):
    try:
        bundle=export_ir(a.source,a.output/target,target=target,runner=a.runner,
                         profile='binary-count-v1' if target=='specksim' else 'dense-int8-v1')
        report[target]=dict(status='converted',source_model_sha256=bundle['source_model_sha256'])
    except ValueError as e:
        if target!='specksim':raise
        report[target]=dict(status='unsupported',reason=str(e))
(a.output/'preflight.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
