"""Run temporal AtlasIR plans through validated official Xylo Audio 2 configs."""
import hashlib
from pathlib import Path
import numpy as np


def _model(plan):
    from rockpool.devices.xylo.syns61201 import config_from_specification,XyloSim
    import samna
    import xylosim.v2
    if plan.get('schema')!='atlas.xylo-audio2.temporal-plan.v1':
        raise ValueError('unknown Xylo temporal plan')
    hidden,output=plan['parameters']['hidden'],plan['parameters']['output']
    config,ok,message=config_from_specification(
        weights_in=np.array(plan['weights']['input'],dtype=int),
        weights_rec=np.array(plan['weights']['recurrent'],dtype=int),
        weights_out=np.array(plan['weights']['readout'],dtype=int),
        dash_syn=hidden['dash_syn'],dash_mem=hidden['dash_mem'],dash_syn_2=[0]*plan['sizes']['hidden'],
        threshold=hidden['threshold'],bias_hidden=hidden['bias'],
        dash_syn_out=output['dash_syn'],dash_mem_out=output['dash_mem'],
        threshold_out=output['threshold'],bias_out=output['bias'])
    if not ok:
        raise ValueError('Xylo official configuration validation failed: '+str(message))
    # Build from the validated hardware config, not a separately constructed
    # unrestricted simulator specification. No hardware device is opened.
    model=XyloSim.from_config(config,dt=plan['dt_seconds'],output_mode='Spike')
    sdk_hash=hashlib.sha256(Path(xylosim.v2.__file__).read_bytes()).hexdigest()
    return model,config.to_json(),sdk_hash


def _records(model, raster):
    spikes,_,record=model.evolve(raster,record=True)
    return dict(hidden_I=record['Isyn'],hidden_v=record['Vmem'],hidden_q=record['Spikes'],
                output_I=record['Isyn_out'],output_v=record['Vmem_out'],output_q=spikes)


def run(plan, *, chunk_steps=None):
    if chunk_steps is not None and (type(chunk_steps) is not int or chunk_steps < 1):
        raise ValueError('chunk_steps must be a positive integer')
    model,configuration,sdk_hash=_model(plan)
    raster=np.array(plan['input_raster'],dtype=int)
    chunks=[]
    width=chunk_steps if chunk_steps is not None else len(raster)
    for start in range(0,len(raster),width):
        # Keep the same model: recurrence and neuron states cross chunk boundaries.
        chunks.append(_records(model,raster[start:start+width]))
    result={key:np.concatenate([chunk[key] for chunk in chunks],axis=0) for key in chunks[0]}
    result['configuration_json']=configuration
    result['configuration_valid']=np.array(True)
    result['sdk_binary_sha256']=sdk_hash
    return result


def run_native_batch(plan, bridge, samples):
    """Execute unchanged integer weights with explicitly adapted dynamics.

    Reset the stock simulator between samples; never replicate the graph past
    its validated hardware dimensions. No labels or reference states enter.
    """
    x=np.asarray(samples)
    sizes=bridge.get('sizes', [])
    if (bridge.get('schema') != 'atlas.native-lif-frozen.v1'
            or bridge.get('dynamics_equivalent') is not False
            or bridge.get('adaptation', {}).get('schema') != 'atlas.native-lif.xylo-adaptation.v1'
            or not bridge.get('fresh_state_per_sample') or not bridge.get('source_is_zero_input_template')
            or len(sizes) != 3 or sizes != [plan['sizes'][k] for k in ('input','hidden','output')]
            or np.any(plan['input_raster'])):
        raise ValueError('unknown native Xylo bridge contract')
    if (x.ndim != 3 or len(x) < 1 or x.shape[1:] != (plan['steps'], sizes[0])
            or not np.isfinite(x).all() or not ((x == 0) | (x == 1)).all()):
        raise ValueError('native Xylo inference requires binary batch × time × input')
    model,configuration,sdk_hash=_model(plan)
    records=[]
    for raster in x:
        model.reset_state()
        records.append(_records(model,raster.astype(int)))
    result={key:np.stack([r[key] for r in records]) for key in records[0]}
    result['spike_raster']=np.concatenate((result['hidden_q'],result['output_q']),axis=2)
    result['voltage']=np.concatenate((result['hidden_v'],result['output_v']),axis=2)
    result['counts']=result['output_q'].sum(axis=1,dtype=np.int64)
    result['logits']=result['counts']*bridge['logit_scale']/plan['steps']
    result.update(configuration_json=configuration,configuration_valid=np.array(True),
                  sdk_binary_sha256=sdk_hash,dynamics_equivalent=False)
    return result
