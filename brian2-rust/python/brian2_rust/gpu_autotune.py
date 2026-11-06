"""Explicit, bounded per-activation GPU policy selection with full-result gates."""
import hashlib
import json
import random
import statistics
import time
import numpy as np

POLICIES=(('baseline',False,False),('prefix',True,False),
          ('bitset',False,'bitset'),('prefix-bitset',True,'bitset'))
ROUNDS=3
MIN_GAIN=.05
DEFAULT_MAX_BUFFER_BYTES=512*1024**2


def run_with_buffer_budget(executor,max_buffer_bytes):
    """Preserve the legacy no-argument executor contract at the default."""
    if max_buffer_bytes==DEFAULT_MAX_BUFFER_BYTES:return executor.run()
    return executor.run(max_buffer_bytes=max_buffer_bytes)


def observable_fingerprint(result):
    """Hash every semantic population/synapse leaf, including scalar event counts."""
    def encode(value):
        if isinstance(value,np.ndarray):
            if value.dtype.hasobject:raise ValueError('Object arrays cannot be tuned')
            return dict(array_sha256=hashlib.sha256(value.tobytes(order='C')).hexdigest(),
                        shape=list(value.shape),dtype=value.dtype.str)
        if isinstance(value,np.generic):return encode(value.item())
        if isinstance(value,dict):return {str(k):encode(v) for k,v in value.items()}
        if isinstance(value,(tuple,list)):return [encode(v) for v in value]
        if value is None or type(value) in (bool,int,float,str):return value
        raise ValueError('Unsupported observable type: '+type(value).__name__)
    payload={key:encode(result[key]) for key in ('populations','synapses')}
    payload.update({key:encode(result.get(key)) for key in ('numeric_profile','rng_profile')})
    encoded=json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def execution_identity(plan):
    """Deduplicate only byte-equivalent complete validated physical plans."""
    return json.dumps(plan.to_dict(),sort_keys=True,separators=(',',':'))


def select_candidate(rows):
    """Require separated ranges and at least 5% median gain against baseline."""
    baseline=rows['baseline']['seconds'];median=statistics.median(baseline)
    eligible=[name for name,row in rows.items() if row.get('status')=='eligible'
        and len(row['seconds'])==ROUNDS and max(row['seconds'])<min(baseline)
        and statistics.median(row['seconds']) <= (1-MIN_GAIN)*median]
    return min(eligible,key=lambda n:statistics.median(rows[n]['seconds'])) if eligible else 'baseline'


def tune(make_executor,directory,*,plan_for=None,max_buffer_bytes=DEFAULT_MAX_BUFFER_BYTES):
    """Return sole ownership of the selected executor and its verified final run.

    make_executor(name, prefix, sparse) must create an independent validated
    initial snapshot. Every candidate has one full warmup and three full replays;
    a final selected replay is the sole result delivered to the Device.
    No calibration result updates Brian state, pending events or clocks.
    Optional plan_for derives a validated plan before construction so identical
    candidates need neither compilation nor a second executor.
    """
    directory.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter();executors={};reference=None;identities={}
    report=dict(schema='b2-gpu-autotune-v1',rounds=ROUNDS,order_seed=1729,
        min_median_gain=MIN_GAIN,scope='complete reset-to-result replays after full warmup; compilation and fingerprinting excluded from samples',
        cache='none; current full activation is profiled each time',candidates={},samples=[],status='running')
    def save():
        report['total_seconds']=time.perf_counter()-started
        (directory/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    def close_except(keep=None):
        errors=[]
        for name in list(executors):
            if name==keep:continue
            ex=executors.pop(name)
            try:ex.close()
            except Exception as error:errors.append(name+': '+str(error))
        if errors:raise RuntimeError('GPU executor cleanup failed: '+'; '.join(errors))
    def execute(name,round_index):
        start=time.perf_counter()
        result=run_with_buffer_budget(executors[name],max_buffer_bytes)
        elapsed=time.perf_counter()-start
        fingerprint=observable_fingerprint(result)
        if not np.isfinite(elapsed) or elapsed<=0:raise RuntimeError('Invalid GPU profiling duration')
        report['samples'].append(dict(candidate=name,round=round_index,seconds=elapsed,
                                      observable_sha256=fingerprint,plan_sha256=executors[name].plan.sha256))
        if reference is not None and fingerprint!=reference:
            raise RuntimeError('Complete GPU result differs from baseline')
        return result,fingerprint,elapsed
    try:
        for name,prefix,sparse in POLICIES:
            row=dict(synapse_prefix=prefix,synapse_sparse=sparse,status='preparing',seconds=[])
            report['candidates'][name]=row
            try:
                prepared=plan_for(name,prefix,sparse) if plan_for is not None else None
                if prepared is not None:
                    row['plan_sha256']=prepared.sha256
                    identity=execution_identity(prepared)
                    if identity in identities:
                        row.update(status='duplicate',equivalent_to=identities[identity],
                            device=report['candidates'][identities[identity]]['device'],compiled=False)
                        save();continue
                ex=make_executor(name,prefix,sparse);executors[name]=ex
                if prepared is not None and execution_identity(prepared)!=execution_identity(ex.plan):
                    raise ValueError('Executor does not match the prepared tuning plan')
                row['plan_sha256']=ex.plan.sha256
                row['device']=ex.device_name
                row['compilation']=dict(getattr(ex,'compilation_report',{}))
                identity=execution_identity(ex.plan)
                if identity in identities:
                    row.update(status='duplicate',equivalent_to=identities[identity])
                    ex.close();del executors[name];continue
                result,fingerprint,_=execute(name,-1)
                if name=='baseline':reference=fingerprint;report['reference_observable_sha256']=reference
                identities[identity]=name
                row['status']='eligible'
                del result
            except Exception as error:
                row.update(status='rejected',reason=type(error).__name__+': '+str(error))
                if name=='baseline':raise
                if name in executors:executors.pop(name).close()
            save()
        rng=random.Random(1729)
        for round_index in range(ROUNDS):
            names=list(executors);rng.shuffle(names)
            for name in names:
                row=report['candidates'][name]
                try:
                    result,_,elapsed=execute(name,round_index);del result
                    row['seconds'].append(elapsed)
                except Exception as error:
                    row.update(status='rejected',reason=type(error).__name__+': '+str(error))
                    if name=='baseline':raise
                    executors.pop(name).close()
                save()
        chosen=select_candidate(report['candidates'])
        result,_,elapsed=execute(chosen,ROUNDS)
        report.update(status='passed',selected=chosen,selected_plan_sha256=executors[chosen].plan.sha256,
            selected_run_seconds=elapsed,selection_reason='measured separated-range improvement' if chosen!='baseline' else 'no verified candidate clears the conservative improvement gate')
        close_except(chosen)
        save();winner=executors.pop(chosen)
        return winner,result,report
    except BaseException as error:
        report.update(status='failed',error=type(error).__name__+': '+str(error));save();raise
    finally:
        close_except()
