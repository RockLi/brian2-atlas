from pathlib import Path
import concurrent.futures,hashlib,json,os,shutil,subprocess,time
repo=Path('/Users/rock/workspace/bettiai/brian2-atlas');root=Path('/Volumes/T7/atlas-sequential-mnist-round4-20261014');root.mkdir(exist_ok=False)
old=Path('/Volumes/T7/atlas-sequential-mnist-round3-20261014');ex=repo/'brian2-rust/examples/neuromorphic/sequential_mnist'
runner=root/'b2-train';shutil.copy2(repo/'brian2-rust/target/release/b2-train',runner)
py='/Users/rock/workspace/bettiai/brian2/.venv/bin/python';dataset='/Users/rock/workspace/bettiai/brian2/brian2-rust/experiments/akida-mnist/output/cache/datasets/mnist.npz'
env={**os.environ,'PYTHONPATH':str(repo/'brian2-rust/python')+':'+str(repo),'MPLCONFIGDIR':'/private/tmp/atlas-mpl','OPENBLAS_NUM_THREADS':'1'}
base=[py,str(ex/'train.py'),'--dataset',dataset,'--runner',str(runner),'--encoding','rows-full-counts','--deskew','--threshold','64','--readout-ticks','20','--augmentation-shift','1','--augmentation-affine','4','--evaluation-round','4','--prior-test-accuracy','.9747','--lr-schedule','cosine','--validate-every','1000']
compact=['--hidden','63','--output-threshold','256','--warm-start',str(old/'averaged-refine'),'--source-runner','/private/tmp/atlas-smnist-count-trainer','--lr','.0003','--updates','8000','--seed','20261121']
code8=['--hidden','63','--output-threshold','256','--recode-from',str(old/'averaged-refine'),'--codebook','hadamard8','--lr','.0006','--updates','12000','--seed','20261122']
wide=['--hidden','126','--recurrent-groups','2','--output-threshold','512','--warm-start',str(old/'selected'),'--source-runner',str(old/'b2-train-wide'),'--lr','.0003','--updates','8000','--seed','20261123']
candidates={'compact-cosine':compact,'compact-rate':compact+['--code-rate-weight','.5'],'code8-ce':code8,'code8-rate':code8+['--code-rate-weight','1'],'wide-rate':wide+['--code-rate-weight','.5']}
commands={name:base+args+['--output',str(root/name)] for name,args in candidates.items()}
protocol=dict(schema='atlas.sequential-mnist.round4.protocol.v1',created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),prior_test_accuracy=.9747,
 training_original_indices=[0,54000],validation_original_indices=[54000,60000],official_test_access='only after completed candidate selection; no test-based tuning',
 method='native integer BPTT; fixed host binary codebook; unchanged per-image encoding and neuron transition',
 primary_candidates=commands,baseline=str(old/'selected'),selection='highest validation accuracy; earliest checkpoint tie; baseline listed first',
 possible_followup='at most two validation-selected candidates may receive a pre-test refinement stage; record commands before running',
 runner_sha256=hashlib.sha256(runner.read_bytes()).hexdigest(),training_source_sha256=hashlib.sha256((ex/'train.py').read_bytes()).hexdigest())
(root/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
def run(item):
 name,args=item
 print(json.dumps(dict(stage='start',candidate=name)),flush=True)
 with (root/(name+'.log')).open('w') as log:
  p=subprocess.run(args,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT)
 print(json.dumps(dict(stage='finished',candidate=name,exit=p.returncode)),flush=True)
 return name,p.returncode
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:results=list(pool.map(run,commands.items()))
(root/'candidate-exit-codes.json').write_text(json.dumps(dict(results),indent=2)+'\n')
raise SystemExit(any(code for _,code in results))
