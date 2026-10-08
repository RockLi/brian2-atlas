"""Same-executor ablation of capacity-mask versus populated-prefix spike decoding."""
import argparse
from contextlib import contextmanager
import hashlib,json,random,statistics,time
from pathlib import Path
import numpy as np

MODES=('capacity-mask','count-gather')
OPTIONS=dict(drive=10/256,delay_span=16,post_delay=16,topology_kind='random-fixed-outdegree',topology_seed=42)


def capacity_mask(ticks,counts,capacity):
    """The spike extraction block from metal.py at ea4918c8, unchanged."""
    if np.any(counts>capacity):raise RuntimeError('Metal spike capacity invariant violated')
    indices,event_ticks=np.empty(0,np.int64),np.empty(0,np.int64)
    if capacity:
        n=len(counts)
        mask=np.arange(capacity)[None,:]<counts[:,None]
        indices=np.broadcast_to(np.arange(n)[:,None],mask.shape)[mask]
        event_ticks=ticks.reshape(n,capacity)[mask]
        order=np.lexsort((indices,event_ticks))
        indices,event_ticks=indices[order],event_ticks[order]
    return event_ticks,indices


@contextmanager
def decode_mode(mode):
    from brian2_rust import metal_event_layout as module
    if mode not in MODES:raise ValueError('Unknown spike decoder')
    original=module.spike_coordinates;times=[]
    implementation=capacity_mask if mode=='capacity-mask' else original
    def measured(*args):
        started=time.perf_counter()
        result=implementation(*args)
        times.append(time.perf_counter()-started)
        return result
    module.spike_coordinates=measured
    try:yield times
    finally:module.spike_coordinates=original


def compare(backend,output):
    from gpu_stdp_compare import oracle,checks,configuration
    from gpu_stdp_precompiled import Replay,write_result,artifact_hashes
    if backend not in {'metal','cuda'}:raise ValueError('Native GPU backend required')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    report=dict(schema='b2-spike-decode-v1',backend=backend,numeric_contract='explicit-f32-v1',
        configuration=configuration(4096,8,4096,**OPTIONS),model_preparations=1,repeats=5,order_seed=1729,
        scope='same compiled executor, fresh reset per replay; only host spike decoder changes; full result and decoder intervals both include identical instrumentation',
        bootstraps={},samples=[],artifact_files=[],passed=False,status='running')
    (output/'declared-protocol.json').write_text(json.dumps({k:v for k,v in report.items() if k not in {'bootstraps','samples','artifact_files','passed','status'}},indent=2)+'\n')
    report['artifact_files'].append('declared-protocol.json')
    def save(name,arrays):
        artifact=write_result(output,name,arrays);artifact['path']=Path(artifact['path']).name
        report['artifact_files'].append(artifact['path']);return artifact
    references={p:oracle(4096,8,4096,dtype=dtype,**OPTIONS) for p,dtype in [('f64',np.float64),('f32',np.float32)]}
    report['references']={p:save('reference-'+p,a) for p,a in references.items()}
    (output/'cpu').mkdir()
    cpu=Replay('cpu-f32',4096,4096,8,output/'cpu',**OPTIONS)
    try:control=cpu.bootstrap;report['cpu_control']=save('cpu-control',control)
    finally:cpu.close()
    if not checks(control,references['f32'])['passed']:
        report.update(status='failed-cpu-control')
        (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        raise RuntimeError('CPU control failed independent f32 gate')
    (output/'gpu').mkdir()
    replay=Replay(backend,4096,4096,8,output/'gpu',**OPTIONS)
    try:
        executor=replay.executor;report['compiled_before']=replay.artifacts
        for name,digest in replay.artifacts.items():
            data=(output/'gpu'/name).read_bytes();assert hashlib.sha256(data).hexdigest()==digest
            target=output/'compiled'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
            report['artifact_files'].append(str(target.relative_to(output)))
        initial=None
        def run(mode,label,round_index=None,position=None):
            nonlocal initial
            with decode_mode(mode) as decoding:
                result,timing=replay.run()
            assert replay.executor is executor and len(decoding)==1
            compiled=artifact_hashes(output/'gpu');assert compiled==replay.artifacts
            gates={p:checks(result,a) for p,a in references.items()}
            gates['compiled_f32']=checks(result,control)
            row=dict(mode=mode,round=round_index,position=position,**timing,decoder_seconds=sum(decoding),
                decoder_calls=len(decoding),compiled=compiled,gates=gates,result=save(label,result))
            exact=initial is None or all(np.array_equal(v,initial[k]) for k,v in result.items())
            if not (gates['f32']['passed'] and gates['compiled_f32']['passed'] and exact):
                report.update(status='failed-numeric-gate',failed_sample=row)
                raise RuntimeError('Decoder result failed numerical or bitwise gate')
            if initial is None:initial=result
            return row
        for mode in MODES:report['bootstraps'][mode]=run(mode,mode+'-bootstrap')
        rng=random.Random(1729)
        for i in range(-1,5):
            modes=list(MODES);rng.shuffle(modes)
            for position,mode in enumerate(modes):
                report['samples'].append(run(mode,f'{mode}-{i}',i,position))
            print('round',i,'complete',flush=True)
        report['compiled_after']=artifact_hashes(output/'gpu')
        report['summary']={}
        for mode in MODES:
            rows=[r for r in report['samples'] if r['mode']==mode and r['round']>=0]
            report['summary'][mode]={}
            for metric in ['wall_seconds','decoder_seconds']:
                values=[r[metric] for r in rows]
                report['summary'][mode][metric]=dict(values=values,median=statistics.median(values),min=min(values),max=max(values))
        report.update(passed=True,status='passed')
    finally:
        replay.close()
        (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--backend',choices=('metal','cuda'),required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    print(json.dumps(compare(a.backend,a.output)['summary'],indent=2))
