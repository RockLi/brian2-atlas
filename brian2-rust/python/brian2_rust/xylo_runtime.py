"""Run temporal AtlasIR plans through validated official Xylo Audio 2 configs."""
import hashlib
from pathlib import Path
import numpy as np


def run(plan):
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
    spikes,_,record=model.evolve(np.array(plan['input_raster'],dtype=int),record=True)
    result=dict(hidden_I=record['Isyn'],hidden_v=record['Vmem'],hidden_q=record['Spikes'],
                output_I=record['Isyn_out'],output_v=record['Vmem_out'],output_q=spikes)
    result['configuration_json']=config.to_json()
    result['configuration_valid']=np.array(True)
    result['sdk_binary_sha256']=hashlib.sha256(Path(xylosim.v2.__file__).read_bytes()).hexdigest()
    return result
