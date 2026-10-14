from pathlib import Path
import concurrent.futures,hashlib,json,os,subprocess,time
repo=Path('/Users/rock/workspace/bettiai/brian2-atlas');root=Path('/Volumes/T7/atlas-sequential-mnist-round4-20261014');old=Path('/Volumes/T7/atlas-sequential-mnist-round3-20261014');ex=repo/'brian2-rust/examples/neuromorphic/sequential_mnist'
py='/Users/rock/workspace/bettiai/brian2/.venv/bin/python';dataset='/Users/rock/workspace/bettiai/brian2/brian2-rust/experiments/akida-mnist/output/cache/datasets/mnist.npz'
env={**os.environ,'PYTHONPATH':str(repo/'brian2-rust/python')+':'+str(repo),'MPLCONFIGDIR':'/private/tmp/atlas-mpl','OPENBLAS_NUM_THREADS':'1'}
base=[py,str(ex/'train.py'),'--dataset',dataset,'--runner',str(root/'b2-train-current'),'--encoding','rows-full-counts','--deskew','--threshold','64','--output-threshold','256','--hidden','63','--readout-ticks','20','--augmentation-shift','1','--augmentation-affine','4','--evaluation-round','4','--prior-test-accuracy','.9747','--lr-schedule','cosine','--validate-every','1000','--readout-source','synaptic_current','--lr','.0004','--updates','12000','--seed','20261124']
commands={
 'current5':base+['--warm-start',str(old/'averaged-refine'),'--source-runner','/private/tmp/atlas-smnist-count-trainer'],
 'current8':base+['--recode-from',str(old/'averaged-refine'),'--codebook','hadamard8']}
commands={k:v+['--output',str(root/k)] for k,v in commands.items()}
protocol=dict(schema='atlas.sequential-mnist.round4.current-protocol.v1',created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
 rationale='Validation-only direct readout probes underperform without retraining; train the integer current readout natively.',
 primary_protocol_sha256=hashlib.sha256((root/'protocol.json').read_bytes()).hexdigest(),probe_sha256=hashlib.sha256((root/'readout-probe/report.json').read_bytes()).hexdigest(),
 candidates=commands,official_test_access=False,selection='compete with completed primary candidates by validation only',
 inference='signed output-current sum, fixed codebook and threshold normalization on host; needs state recording; not on-chip classifier',
 runner_sha256=hashlib.sha256((root/'b2-train-current').read_bytes()).hexdigest(),training_source_sha256=hashlib.sha256((ex/'train.py').read_bytes()).hexdigest())
(root/'current-protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
def run(item):
 name,args=item;print(json.dumps(dict(stage='start',candidate=name)),flush=True)
 with (root/(name+'.log')).open('w') as log:p=subprocess.run(args,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT)
 print(json.dumps(dict(stage='finished',candidate=name,exit=p.returncode)),flush=True);return name,p.returncode
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(run,commands.items()))
(root/'current-exit-codes.json').write_text(json.dumps(dict(results),indent=2)+'\n')
raise SystemExit(any(code for _,code in results))
