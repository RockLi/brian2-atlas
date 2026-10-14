"""Run temporal AtlasIR plans through validated official Xylo Audio 2 configs."""
import copy
import hashlib
from pathlib import Path
import numpy as np

if __package__:
    from .xylo_readout import decode_readout, validate_native_readout
else:
    import importlib.util
    _spec = importlib.util.spec_from_file_location('atlas_xylo_readout', Path(__file__).with_name('xylo_readout.py'))
    _readout = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_readout)
    decode_readout = _readout.decode_readout
    validate_native_readout = _readout.validate_native_readout


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


def run_native_batch(plan, bridge, samples, *, chunk_steps=None):
    """Execute frozen integer weights using the declared exact or adapted bridge.

    Reset the stock simulator between samples; never replicate the graph past
    its validated hardware dimensions. No labels or reference states enter.
    """
    x=np.asarray(samples)
    sizes=bridge.get('sizes', [])
    exact=(bridge.get('schema') in ('atlas.native-xylo-frozen.v1','atlas.native-xylo-frozen.v2','atlas.native-xylo-current-bridge.v1')
           and bridge.get('dynamics_equivalent') is True)
    approximate=(bridge.get('schema') == 'atlas.native-lif-frozen.v1'
                 and bridge.get('dynamics_equivalent') is False
                 and bridge.get('adaptation', {}).get('schema') == 'atlas.native-lif.xylo-adaptation.v1')
    if (not (exact or approximate)
            or not bridge.get('fresh_state_per_sample') or not bridge.get('source_is_zero_input_template')
            or len(sizes) != 3 or sizes != [plan['sizes'][k] for k in ('input','hidden','output')]
            or np.any(plan['input_raster'])):
        raise ValueError('unknown native Xylo bridge contract')
    upper=bridge.get('input_max',1)
    if type(upper) is not int or not 1<=upper<=15:
        raise ValueError('invalid native Xylo input_max')
    if (x.ndim != 3 or len(x) < 1 or x.shape[1:] != (plan['steps'], sizes[0])
            or not np.isfinite(x).all() or not ((x>=0)&(x<=upper)&(x==np.floor(x))).all()):
        raise ValueError('native Xylo inference requires '+('binary batch' if upper==1 else 'bounded integer counts, batch')+' × time × input')
    readout_start=bridge.get('readout_start',0)
    recurrent=bridge.get('schema') in ('atlas.native-xylo-frozen.v2','atlas.native-xylo-current-bridge.v1')
    if (type(readout_start) is not int or not 0<=readout_start<plan['steps']
            or (recurrent and bridge.get('stateful_chunks') is not True)):
        raise ValueError('invalid recurrent Xylo readout contract')
    if chunk_steps is not None and (not recurrent or type(chunk_steps) is not int or chunk_steps<1):
        raise ValueError('native chunking requires v2 and positive integer chunk_steps')
    validate_native_readout(bridge, plan)
    model,configuration,sdk_hash=_model(plan)
    records=[]
    for raster in x:
        model.reset_state()
        width=chunk_steps or len(raster)
        chunks=[_records(model,raster[start:start+width].astype(int)) for start in range(0,len(raster),width)]
        records.append({k:np.concatenate([r[k] for r in chunks]) for k in chunks[0]})
    result={key:np.stack([r[key] for r in records]) for key in records[0]}
    result['spike_raster']=np.concatenate((result['hidden_q'],result['output_q']),axis=2)
    result['voltage']=np.concatenate((result['hidden_v'],result['output_v']),axis=2)
    result['counts']=result['output_q'][:,readout_start:].sum(axis=1,dtype=np.int64)
    result['current_sums']=result['output_I'][:,readout_start:].sum(axis=1,dtype=np.int64)
    result['logits']=decode_readout(result['counts'],result['current_sums'],bridge,plan['steps']-readout_start)
    result.update(configuration_json=configuration,configuration_valid=np.array(True),
                  sdk_binary_sha256=sdk_hash,dynamics_equivalent=exact)
    return result


class XyloInferenceSession:
    """One v2 sequence on a persistent stock XyloSim model, bounded by export.

    push() accepts the next bounded-count time x input chunk. Counts and currents accumulate
    only within the absolute readout window. reset() starts a new independent
    sequence. No arbitrary SDK-state injection or on-chip learning is provided.
    """
    def __init__(self,bundle):
        if (bundle.get('schema')!='atlas.neuromorphic.simulator-bundle.v1'
                or bundle.get('target')!='xylosim' or bundle.get('profile')!='xylo-audio2-temporal-v1'):
            raise ValueError('XyloInferenceSession requires an exported Xylo bundle')
        self._plan=copy.deepcopy(bundle['plan']);self._bridge=copy.deepcopy(bundle.get('native_training_bridge',{}))
        p=self._plan;d=self._bridge
        if (d.get('schema') not in ('atlas.native-xylo-frozen.v2','atlas.native-xylo-current-bridge.v1') or d.get('dynamics_equivalent') is not True
                or d.get('stateful_chunks') is not True or d.get('fresh_state_per_sample') is not True
                or d.get('source_is_zero_input_template') is not True or np.any(p['input_raster'])
                or d.get('sizes')!=[p['sizes'][k] for k in ('input','hidden','output')]
                or type(d.get('readout_start')) is not int or not 0<=d['readout_start']<p['steps']
                or not 1<=p['steps']<=128):
            raise ValueError('invalid native recurrent Xylo session contract')
        validate_native_readout(d, p)
        self._model,self.configuration_json,self.sdk_binary_sha256=_model(p)
        self.reset()

    def reset(self):
        self._model.reset_state();self.elapsed_ticks=0
        self._counts=np.zeros(self._plan['sizes']['output'],dtype=np.int64)
        self._current_sums=np.zeros_like(self._counts)

    def push(self,inputs):
        x=np.asarray(inputs);p=self._plan;d=self._bridge
        upper=d.get('input_max',1)
        if type(upper) is not int or not 1<=upper<=15:raise ValueError('invalid input_max')
        if (x.ndim!=2 or not 1<=len(x)<=p['steps']-self.elapsed_ticks or x.shape[1]!=p['sizes']['input']
                or not np.isfinite(x).all() or not ((x>=0)&(x<=upper)&(x==np.floor(x))).all()):
            raise ValueError('Xylo chunk requires bounded integer counts, time x input within the remaining horizon')
        result=_records(self._model,x.astype(int))
        start=max(0,d['readout_start']-self.elapsed_ticks)
        self._counts+=result['output_q'][start:].sum(axis=0,dtype=np.int64)
        self._current_sums+=result['output_I'][start:].sum(axis=0,dtype=np.int64)
        self.elapsed_ticks+=len(x);read_ticks=max(0,self.elapsed_ticks-d['readout_start'])
        result.update(spike_raster=np.concatenate((result['hidden_q'],result['output_q']),axis=1),
            voltage=np.concatenate((result['hidden_v'],result['output_v']),axis=1),
            counts=self._counts.copy(),current_sums=self._current_sums.copy(),
            logits=decode_readout(self._counts,self._current_sums,d,read_ticks),
            elapsed_ticks=self.elapsed_ticks,readout_ticks=read_ticks)
        return result
