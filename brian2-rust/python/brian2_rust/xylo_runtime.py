"""Run temporal AtlasIR plans through validated official Xylo Audio 2 configs."""
import hashlib
from pathlib import Path
import numpy as np


def run(plan, *, chunk_steps=None):
    if chunk_steps is not None and (type(chunk_steps) is not int or chunk_steps < 1):
        raise ValueError('chunk_steps must be a positive integer')
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
    raster=np.array(plan['input_raster'],dtype=int)
    chunks=[]
    width=chunk_steps if chunk_steps is not None else len(raster)
    for start in range(0,len(raster),width):
        # Keep the same model: recurrence and neuron states cross chunk boundaries.
        spikes,_,record=model.evolve(raster[start:start+width],record=True)
        chunks.append(dict(hidden_I=record['Isyn'],hidden_v=record['Vmem'],hidden_q=record['Spikes'],
                           output_I=record['Isyn_out'],output_v=record['Vmem_out'],output_q=spikes))
    result={key:np.concatenate([chunk[key] for chunk in chunks],axis=0) for key in chunks[0]}
    result['configuration_json']=config.to_json()
    result['configuration_valid']=np.array(True)
    result['sdk_binary_sha256']=hashlib.sha256(Path(xylosim.v2.__file__).read_bytes()).hexdigest()
    return result
