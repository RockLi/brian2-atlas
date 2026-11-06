"""Export a frozen TF count model through Brian2/AtlasIR and rerun Atlas.

The TF checkpoint is fixed before reading test labels. All simulator bundles
are derived from the same canonical IR, rather than directly from TF weights.
"""
import argparse,hashlib,json,os,subprocess,tempfile,time
from pathlib import Path
import numpy as np
import brian2 as b
from brian2_atlas import export_network
from brian2_atlas.neuromorphic import export_ir
from brian2_rust.protocol import attach_protocol,canonical_bytes
from brian2_rust.results import load_results


def balanced(terms):
    if not terms:return '0'
    if len(terms)==1:return terms[0]
    mid=len(terms)//2
    return f'({balanced(terms[:mid])}+{balanced(terms[mid:])})'


def affine(names,weights):
    return balanced([f'{int(w)}*({balanced([n for n,v in zip(names,weights) if v==w])})'
                     for w in sorted(set(weights.tolist())) if w])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','trained','runner','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--limit',type=int,default=10000)
    p.add_argument('--batch-size',type=int,default=500)
    a=p.parse_args()
    if not 1 <= a.limit <= 10000 or not 1 <= a.batch_size <= a.limit or a.limit%a.batch_size:
        p.error('limit must be in [1,10000] and divisible by batch-size')
    a.output.mkdir(parents=True,exist_ok=False)
    os.environ.update(B2_MAX_POPULATION_PARAMETERS='1024',B2_MAX_POPULATION_STATES='512',B2_MAX_STATEMENTS='512')
    sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    training=json.loads((a.trained/'training.json').read_text())
    assert sha(a.dataset)==training['dataset_sha256']
    assert sha(a.trained/'weights.npz')==training['weights_sha256']
    with np.load(a.trained/'weights.npz',allow_pickle=False) as data:
        w,theta=data['weights'],int(data['threshold'])
    assert w.shape==(784,10) and np.all(w>=0) and np.all(w<=theta) and np.all(w.sum(0)//theta<=127)
    with np.load(a.dataset,allow_pickle=False) as data:
        x=(data['x_test'][:a.limit].reshape(-1,784)>=128).astype(np.uint8)
        labels=data['y_test'][:a.limit]
    np.savez_compressed(a.output/'inputs.npz',inputs=x)
    np.savez_compressed(a.output/'labels.npz',labels=labels)
    ins=[f'x{i:03d}' for i in range(784)]
    hidden=[f'h{i}' for i in range(10)];scores=[f's{i}' for i in range(10)]
    lines=[f'{name}=clip(floor(({affine(ins,w[:,i])})/{theta}),0,127)' for i,name in enumerate(hidden)]
    lines += [f'{s}={h}' for s,h in zip(scores,hidden)]
    b.start_scope();b.set_device('runtime');b.prefs.codegen.target='numpy'
    b.prefs.codegen.loop_invariant_optimisations=False
    group=b.NeuronGroup(a.batch_size,'\n'.join([f'{n}:1 (constant)' for n in ins]+[f'{n}:1' for n in hidden+scores]),dt=b.ms)
    for i,n in enumerate(ins):setattr(group,n,x[:a.batch_size,i])
    group.run_regularly('\n'.join(lines),when='start')
    source=a.output/'source.json'
    model=export_network(b.Network(group),b.ms,source)
    bundles={target:export_ir(source,a.output/target,runner=a.runner,target=target,profile='binary-count-v1')
             for target in ('akida','lava','specksim')}
    plan=bundles['akida']['plan']
    assert len({v['source_model_sha256'] for v in bundles.values()})==1
    counts=x.astype(np.int64)@w//theta
    expected={**{n:counts[:,i] for i,n in enumerate(hidden)},**{n:counts[:,i] for i,n in enumerate(scores)}}
    truth=np.stack([expected[n] for n in plan['outputs']],axis=1)
    all_states=[];batches=[];start_time=time.monotonic()
    for start in range(0,len(x),a.batch_size):
        rows=x[start:start+a.batch_size]
        for i,n in enumerate(ins):
            raw=rows[:,i].astype('>f8').tobytes().hex()
            model['instance']['populations'][0]['parameters'][n]=[raw[k:k+16] for k in range(0,len(raw),16)]
        attach_protocol(model)
        encoded=canonical_bytes(model)
        with tempfile.TemporaryDirectory(prefix='atlas-neuro-batch-') as directory:
            path=Path(directory)/'source.json';path.write_bytes(encoded)
            output=Path(directory)/'result'
            run=subprocess.run([str(a.runner.resolve()),str(path),str(output)],capture_output=True,text=True)
            if run.returncode:raise RuntimeError(run.stderr)
            values=load_results(model,output)['populations'][0]['states']
            states=np.stack([values[n] for n in plan['outputs']],axis=1)
        np.testing.assert_array_equal(states,truth[start:start+a.batch_size])
        all_states.append(states)
        batches.append(dict(start=start,count=len(rows),source_sha256=hashlib.sha256(encoded).hexdigest(),
                            source_hashes=model['protocol']['layers']))
        print(f'Atlas {start+len(rows)}/{len(x)}',flush=True)
    states=np.concatenate(all_states)
    np.savez_compressed(a.output/'atlas.npz',states=states)
    report=dict(schema='atlas.neuromorphic.preparation.v1',samples=len(x),batch_size=a.batch_size,batches=batches,
                source_model_sha256=bundles['akida']['source_model_sha256'],
                definition_sha256=model['protocol']['layers']['definition'],
                runner_sha256=sha(a.runner),runner_path=str(a.runner.resolve()),
                dataset_sha256=sha(a.dataset),weights_sha256=sha(a.trained/'weights.npz'),
                inputs_sha256=sha(a.output/'inputs.npz'),labels_sha256=sha(a.output/'labels.npz'),
                atlas_results_sha256=sha(a.output/'atlas.npz'),state_names=plan['outputs'],score_names=scores,
                training=training,elapsed_seconds=time.monotonic()-start_time,
                source_packages={'brian2':b.__file__,'brian2_atlas':__import__('brian2_atlas').__file__})
    (a.output/'preparation.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
