"""Check every public target/profile pairing against a valid source fixture."""
import argparse,hashlib,json,os
from pathlib import Path
from brian2_atlas.neuromorphic import convert_ir,TARGETS,PROFILES

p=argparse.ArgumentParser(description=__doc__)
for n in ('static-run','spinnaker-run','xylo-run','dynapse-run','runner','output'):p.add_argument('--'+n,type=Path,required=True)
p.add_argument('--integer-source',type=Path,required=True)
a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
os.environ.update(B2_MAX_POPULATION_PARAMETERS='1024',B2_MAX_POPULATION_STATES='512',B2_MAX_STATEMENTS='512')
sources={'dense-int8-v1':a.static_run/'dense_clip_round/source.json',
         'binary-count-v1':a.static_run/'count_00/source.json',
         'spinnaker2-lif-v1':a.spinnaker_run/'recurrent_64/model.json',
         'xylo-audio2-temporal-v1':a.xylo_run/'recurrent/source.json',
         'dynapse-dpi-v1':a.dynapse_run/'constant/source.json',
         'spinnaker2-xylo-integer-v1':a.integer_source}
supported={'dense-int8-v1':{'akida','lava','spinnaker2'},'binary-count-v1':set(TARGETS)-{'dynapse'},
           'spinnaker2-lif-v1':{'spinnaker2'},'xylo-audio2-temporal-v1':{'xylosim'},'dynapse-dpi-v1':{'dynapse'},
           'spinnaker2-xylo-integer-v1':{'spinnaker2'}}
rows=[]
for profile in PROFILES:
    for target in TARGETS:
        expected=target in supported[profile]
        try:
            bundle=convert_ir(sources[profile],target=target,profile=profile,runner=a.runner)
        except ValueError as e:
            if expected:raise
            reason=str(e)
            assert ('only supported by' in reason or 'requires binary-count-v1' in reason or 'requires dynapse-dpi-v1' in reason),reason
            rows.append(dict(profile=profile,target=target,status='rejected',reason=reason))
        else:
            assert expected,(profile,target)
            rows.append(dict(profile=profile,target=target,status='admitted',source_model_sha256=bundle['source_model_sha256']))
report=dict(schema='atlas.neuromorphic.compatibility-check.v1',passed=True,cells=len(rows),
            admitted=sum(r['status']=='admitted' for r in rows),rejected=sum(r['status']=='rejected' for r in rows),
            rows=rows,runner_sha256=hashlib.sha256(a.runner.read_bytes()).hexdigest(),
            scope='Public conversion routing only. Simulator execution evidence is reported separately.')
a.output.write_text(json.dumps(report,indent=2)+'\n')
print(f"Compatibility matrix checked: {report['admitted']} admitted, {report['rejected']} rejected.")
