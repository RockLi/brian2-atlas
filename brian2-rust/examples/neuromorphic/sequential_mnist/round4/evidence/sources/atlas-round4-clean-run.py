from pathlib import Path
import hashlib,json,os,subprocess,time
repo=Path('/Users/rock/workspace/bettiai/brian2-atlas');root=Path('/Volumes/T7/atlas-sequential-mnist-round4-20261014');ex=repo/'brian2-rust/examples/neuromorphic/sequential_mnist'
parent=root/'compact-cosine';d=json.loads((parent/'training.json').read_text());assert d['updates_completed']==d['requested_updates']
args=['/Users/rock/workspace/bettiai/brian2/.venv/bin/python',str(ex/'train.py'),'--dataset',d['arguments']['dataset'],'--runner',str(root/'b2-train-current'),'--warm-start',str(parent),'--source-runner',str(root/'b2-train'),'--output',str(root/'compact-clean'),'--encoding','rows-full-counts','--deskew','--threshold','64','--output-threshold','256','--hidden','63','--readout-ticks','20','--evaluation-round','4','--prior-test-accuracy','.9747','--lr-schedule','cosine','--lr','.0001','--updates','6000','--validate-every','1000','--seed','20261125']
protocol=dict(created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),parent_validation_accuracy=d['selected']['validation_accuracy'],
 rationale='Best completed new candidate; end-stage training on clean samples to test whether continued augmentation limits fit. One of at most two refinement stages.',
 parent_report_sha256=hashlib.sha256((parent/'training.json').read_bytes()).hexdigest(),arguments=args,test_arrays_read=False)
(root/'clean-protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
env={**os.environ,'PYTHONPATH':str(repo/'brian2-rust/python')+':'+str(repo),'MPLCONFIGDIR':'/private/tmp/atlas-mpl','OPENBLAS_NUM_THREADS':'1'}
with (root/'compact-clean.log').open('w') as log:p=subprocess.run(args,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT)
raise SystemExit(p.returncode)
