from pathlib import Path
import hashlib,json,os,subprocess,time
repo=Path('/Users/rock/workspace/bettiai/brian2-atlas');root=Path('/Volumes/T7/atlas-sequential-mnist-round4-20261014');old=Path('/Volumes/T7/atlas-sequential-mnist-round3-20261014');ex=repo/'brian2-rust/examples/neuromorphic/sequential_mnist'
names=('compact-cosine','compact-rate','code8-ce','code8-rate','wide-rate','current5','current8','compact-clean','compact-late')
(root/'final-selection-amended-protocol.json').write_text(json.dumps(dict(created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),candidates=list(names),early_stop='wide-rate may be excluded if still below its original validation baseline at update 2000; record its incomplete run separately',baseline=str(old/'selected'),selection='maximum validation accuracy among completed runs; baseline first, listed order breaks ties; no further tuning after test',test='one selected model, full official 10000 test images; exact neural traces and logits on both targets; independent native build; checkpoint and AtlasIR replay'),indent=2)+'\n')
while True:
 try:
  active=[n for n in names if not (n=='wide-rate' and (root/'wide-rate-early-stop.json').exists())]
  reports=[json.loads((root/n/'training.json').read_text()) for n in active]
  if all(d['updates_completed']==d['requested_updates'] for d in reports):break
 except (FileNotFoundError,json.JSONDecodeError):pass
 time.sleep(10)
env={**os.environ,'PYTHONPATH':str(repo/'brian2-rust/python')+':'+str(repo),'MPLCONFIGDIR':'/private/tmp/atlas-mpl','OPENBLAS_NUM_THREADS':'1'}
py='/Users/rock/workspace/bettiai/brian2/.venv/bin/python'
with (root/'selection.log').open('w') as log:subprocess.run([py,str(ex/'select_model.py'),'--runs',str(old/'selected'),*[str(root/n) for n in active],'--output',str(root/'selected')],cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
selection=json.loads((root/'selected/selection.json').read_text());selection['excluded_incomplete_runs']=[json.loads((root/'wide-rate-early-stop.json').read_text())] if (root/'wide-rate-early-stop.json').exists() else [];(root/'selected/selection.json').write_text(json.dumps(selection,indent=2)+'\n');t=json.loads((root/'selected/training.json').read_text())
runners=[root/'b2-train',root/'b2-train-current',old/'b2-train-wide']
runner=next(p for p in runners if hashlib.sha256(p.read_bytes()).hexdigest()==t['runner_sha256'])
print(json.dumps(dict(stage='selected-before-test',candidate=selection['selected']['run'],validation=selection['selected']['validation_accuracy'],runner=str(runner))),flush=True)
args=[py,str(ex/'verify.py'),'--dataset',t['arguments']['dataset'],'--trained',str(root/'selected'),'--runner',str(old/'b2-runner-wide'),'--train-runner',str(runner),'--portable-train-runner',str(root/'b2-train-current-portable'),'--xylo-python','/private/tmp/atlas-xylo-env/bin/python','--spinnaker2-python','/private/tmp/atlas-spinnaker2-env312/bin/python','--output',str(root/'audit')]
(root/'audit-command.json').write_text(json.dumps(dict(arguments=args,selection_sha256=hashlib.sha256((root/'selected/selection.json').read_bytes()).hexdigest()),indent=2)+'\n')
with (root/'audit.log').open('w') as log:p=subprocess.run(args,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT)
print(json.dumps(dict(stage='audit-finished',exit_code=p.returncode)),flush=True)
raise SystemExit(p.returncode)
