from pathlib import Path
import hashlib,json,os,shlex,signal,subprocess,time
root=Path('/Volumes/T7/atlas-sequential-mnist-round4-20261014');name='wide-rate'
while True:
 try:
  d=json.loads((root/name/'training.json').read_text())
  if d['updates_completed']>=2000:break
 except (FileNotFoundError,json.JSONDecodeError):pass
 time.sleep(5)
if d['selected']['validation_accuracy']>=.981:
 print('Wide candidate reached its baseline; retain the original full training budget.',flush=True)
 raise SystemExit(0)
# Stop only this task's explicitly launched training parent, never a process group.
lines=subprocess.check_output(['ps','-axo','pid=,args='],text=True).splitlines();matches=[]
for line in lines:
 pid,command=line.strip().split(None,1)
 try:args=shlex.split(command)
 except ValueError:continue
 if '--output' in args and args[args.index('--output')+1]==str(root/name) and any(a.endswith('/sequential_mnist/train.py') for a in args):matches.append(int(pid))
assert len(matches)==1,matches
os.kill(matches[0],signal.SIGTERM)
receipt=dict(candidate=name,status='stopped-after-validation',validated_updates=d['updates_completed'],requested_updates=d['requested_updates'],best_validation=d['selected']['validation_accuracy'],test_arrays_read=False,
 reason='The wide rate-loss experiment is much slower and has not beaten the 98.10% wide baseline; completed compact rate-loss training also failed to improve its parent. Prioritize the requested readout and compact training experiments.',
 excluded_from_completed_candidate_selection=True,training_report_sha256=hashlib.sha256((root/name/'training.json').read_bytes()).hexdigest(),stopped_pid=matches[0])
(root/'wide-rate-early-stop.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
