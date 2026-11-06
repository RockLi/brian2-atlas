"""Rescore the saved validation-only view probe in integer count space."""
from pathlib import Path
import hashlib,json,numpy as np
r=Path('/Volumes/T7/atlas-sequential-mnist-round2-20261014')
f=json.loads((r/'precision/frozen.json').read_text());p=json.loads((r/'validation-views.json').read_text())
a=np.load(r/'validation-views.npz',allow_pickle=False);scores=a['logits'];labels=a['labels'];ticks=22-f['plan']['readout_start'];scale=f['plan']['logit_scale']
counts=np.rint(scores*ticks/scale).astype(np.int64)
np.testing.assert_array_equal(counts.astype(float)*scale/ticks,scores)
p.update(aggregation='sum recovered integer class counts; exact score round-trip verified; equal weights; lowest-index ties',
 validation_accuracy={str(n):float(np.mean(counts[:n].sum(0).argmax(1)==labels)) for n in [1,3,5,9]},
 test_data_read=False,adopted_for_final_audit=False,
 decision='The small validation gain requires 3–9 separate neural executions; retain single-view inference for the primary result.',
 prediction_script_sha256=hashlib.sha256(Path('/private/tmp/atlas-mnist-validate-views.py').read_bytes()).hexdigest(),
 scoring_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
 predictions_sha256=hashlib.sha256((r/'validation-views.npz').read_bytes()).hexdigest())
(r/'validation-views.json').write_text(json.dumps(p,indent=2)+'\n')
print(json.dumps(p['validation_accuracy']))
