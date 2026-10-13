"""Require complete, exact native Atlas/SDK results for all prepared cases."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
a=p.parse_args();preparation=json.loads((a.run/'preparation.json').read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
reports=[];totals={}
for case in preparation['cases']:
    root=a.run/case['case'];atlas=np.load(root/'atlas.npz')['states'];x=np.load(root/'inputs.npz')['inputs']
    for target in case['targets']:
        dest=root/target/'coverage-run';bundle_path=root/target/'bundle.json'
        bundle=json.loads(bundle_path.read_text());execution=json.loads((dest/'execution.json').read_text())
        assert execution['bundle_sha256']==sha(bundle_path)
        assert execution['inputs_sha256']==sha(root/'inputs.npz')
        assert execution['results_sha256']==sha(dest/'results.npz')
        assert bundle['source_model_sha256']==case['source_model_sha256']
        with np.load(dest/'results.npz') as sdk:
            np.testing.assert_array_equal(sdk['states'],atlas,err_msg=case['case']+': '+target)
            voltage_values=0
            if target=='specksim':
                plan=bundle['plan'];sums=x.astype(np.int64)@np.asarray(plan['weights']).T
                np.testing.assert_array_equal(sdk['voltage'],sums%plan['threshold'])
                np.testing.assert_array_equal(sdk['counts'],sums//plan['threshold'])
                voltage_values=sdk['voltage'].size
                assert execution['batch_check']['result_sha256']==sha(dest/'alternate.npz')
                with np.load(dest/'alternate.npz') as alt:
                    for key in sdk.files:np.testing.assert_array_equal(alt[key],sdk[key])
        reports.append(dict(case=case['case'],profile=case['profile'],target=target,samples=len(x),
            compared_state_values=int(atlas.size),membrane_values=int(voltage_values),differences=0,
            sdk_execution_skipped=(target=='spinnaker2' and not bundle['plan']['comparators']),
            atlas_result_sha256=sha(root/'atlas.npz'),source_model_sha256=case['source_model_sha256'],execution=execution))
        t=totals.setdefault(target,dict(cases=0,samples=0,state_values=0,membrane_values=0,differences=0))
        t['cases']+=1;t['samples']+=len(x);t['state_values']+=int(atlas.size);t['membrane_values']+=int(voltage_values)
# Explicitly ensure the high-count and active-clip cases are non-vacuous.
assert np.load(a.run/'count_cap_127/atlas.npz')['states'].max()==127
clip=np.load(a.run/'dense_clip_round/atlas.npz')['states']
assert 0 in clip and 127 in clip and clip.max()>127
report=dict(schema='atlas.neuromorphic.coverage-audit.v1',passed=True,seed=preparation['seed'],
            runner_sha256=preparation['runner_sha256'],models=len(preparation['cases']),totals=totals,cases=reports,
            same_source_ir_per_model=True,physical_hardware=False,classification_accuracy_tested=False,
            static_comparison='final hidden and readout values, not source spike timing')
(a.run/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(totals,indent=2))
