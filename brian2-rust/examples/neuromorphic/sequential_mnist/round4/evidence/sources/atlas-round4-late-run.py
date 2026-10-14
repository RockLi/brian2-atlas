from pathlib import Path
import hashlib,json,os,subprocess,time
repo=Path('/Users/rock/workspace/bettiai/brian2-atlas');root=Path('/Volumes/T7/atlas-sequential-mnist-round4-20261014');ex=repo/'brian2-rust/examples/neuromorphic/sequential_mnist'
parent=root/'compact-cosine';d=json.loads((parent/'training.json').read_text());assert d['updates_completed']==d['requested_updates']
args=['/Users/rock/workspace/bettiai/brian2/.venv/bin/python',str(ex/'train.py'),'--dataset',d['arguments']['dataset'],'--runner',str(root/'b2-train-current'),'--warm-start',str(parent),'--source-runner',str(root/'b2-train'),'--output',str(root/'compact-late'),'--encoding','rows-full-counts','--deskew','--threshold','64','--output-threshold','256','--hidden','63','--readout-ticks','12','--augmentation-shift','1','--augmentation-affine','4','--evaluation-round','4','--prior-test-accuracy','.9747','--lr-schedule','cosine','--lr','.0003','--updates','8000','--validate-every','1000','--seed','20261126']
protocol=dict(created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),parent_validation_accuracy=d['selected']['validation_accuracy'],
 rationale='Second and final refinement: retrain a later [24,36) spike readout window instead of [16,36), retaining parent augmentation and network dimensions.',
 parent_report_sha256=hashlib.sha256((parent/'training.json').read_bytes()).hexdigest(),arguments=args,test_arrays_read=False)
(root/'late-protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
while True:
 d=json.loads((root/'code8-ce/training.json').read_text())
 if d['updates_completed']==d['requested_updates']:break
 time.sleep(10)
env={**os.environ,'PYTHONPATH':str(repo/'brian2-rust/python')+':'+str(repo),'MPLCONFIGDIR':'/private/tmp/atlas-mpl','OPENBLAS_NUM_THREADS':'1'}
with (root/'compact-late.log').open('w') as log:p=subprocess.run(args,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT)
raise SystemExit(p.returncode)
