"""Optional SDK execution, also loadable in isolated Python 3.10 SDK workers.

This module imports no Brian2 or SDK at import time. Plans are trusted compiler
artifacts; workers receive inputs only, never labels or reference outputs.
"""
import importlib.util
from pathlib import Path
import numpy as np


def _module(name):
    spec = importlib.util.spec_from_file_location('atlas_neuro_'+name, Path(__file__).with_name(name+'.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_inputs(bundle, samples):
    if bundle.get('schema') != 'atlas.neuromorphic.simulator-bundle.v1':
        raise ValueError('unknown simulator bundle schema')
    if bundle.get('profile') not in ('binary-count-v1', 'dense-int8-v1'):
        raise ValueError('unknown conversion profile')
    x = np.asarray(samples)
    if x.ndim != 2 or len(x) == 0 or x.shape[1] != len(bundle['plan']['inputs']):
        raise ValueError('incorrect input dimensions')
    upper = 1 if bundle['profile'] == 'binary-count-v1' else 127
    if not (np.isfinite(x).all() and (x == np.floor(x)).all() and ((x >= 0) & (x <= upper)).all()):
        raise ValueError(f'inference input outside {bundle["input_domain"]} domain')
    return x.astype(np.uint8)


def run_simulator(bundle, samples=None, *, batch_size=50):
    if bundle.get('profile') == 'dynapse-dpi-v1':
        if (bundle.get('schema') != 'atlas.neuromorphic.simulator-bundle.v1'
                or bundle.get('target') != 'dynapse' or samples is not None):
            raise ValueError('DynapSim requires dynapse and its embedded spike raster')
        return _module('dynapse_runtime').run(bundle['plan'])
    if bundle.get('profile') == 'xylo-audio2-temporal-v1':
        if (bundle.get('schema') != 'atlas.neuromorphic.simulator-bundle.v1'
                or bundle.get('target') != 'xylosim' or samples is not None):
            raise ValueError('Xylo temporal requires xylosim and its embedded spike-count raster')
        return _module('xylo_runtime').run(bundle['plan'])
    if bundle.get('target') == 'spinnaker2' and bundle.get('profile') == 'spinnaker2-lif-v1':
        if 'native_training_bridge' in bundle and samples is not None:
            if bundle.get('schema') != 'atlas.neuromorphic.simulator-bundle.v1':
                raise ValueError('unknown simulator bundle schema')
            return _module('spinnaker2_runtime').run_native_batch(
                bundle['plan'], bundle['native_training_bridge'], samples, batch_size)
        if (bundle.get('schema') != 'atlas.neuromorphic.simulator-bundle.v1'
                or bundle.get('profile') != 'spinnaker2-lif-v1' or samples is not None):
            raise ValueError('SpiNNaker2 requires spinnaker2-lif-v1 and its embedded spike schedule')
        return _module('spinnaker2_runtime').run(bundle['plan'])
    x = validate_inputs(bundle, samples)
    plan = bundle['plan']
    target = bundle['target']
    if target == 'spinnaker2' and bundle['profile'] == 'dense-int8-v1':
        return _module('spinnaker2_dense').run(plan, x)
    if target in ('spinnaker2', 'xylosim'):
        if bundle['profile'] != 'binary-count-v1':
            raise ValueError(f'{target} requires binary-count-v1')
        return getattr(_module('count_targets'), 'run_'+target)(plan, x)
    if target == 'akida':
        api = _module('akida_b2ir')
        model = api.build_akida_model(plan)
        states = api.predict_quantized(model, plan, x)
        return {'states': states}
    if target == 'lava':
        return {'states': _lava(plan, x)}
    if target == 'specksim':
        if bundle['profile'] != 'binary-count-v1':
            raise ValueError('Specksim requires binary-count-v1')
        return _specksim(plan, x, batch_size)
    raise ValueError(f'unknown simulator target: {target!r}')


def _lava(plan, x):
    from scipy.sparse import csr_matrix
    from lava.magma.core.run_conditions import RunSteps
    from lava.magma.core.run_configs import Loihi2SimCfg
    from lava.proc.dense.process import Dense
    from lava.proc.sparse.process import Sparse
    from lava.proc.graded.process import GradedReluVec
    from lava.proc.io.source import RingBuffer as Source
    from lava.proc.io.sink import RingBuffer as Sink
    from lava.utils.weightutils import SignMode
    h, classes = len(plan['hidden_states']), len(plan['score_states'])
    ticks = len(x)+plan['pipeline_delay_ticks']
    source = Source(data=_module('lava_dense_b2ir').encode_inputs(plan, x))
    first = Dense(weights=np.asarray(plan['first_weights'], np.int32), num_message_bits=24,
                  sign_mode=SignMode.MIXED, num_weight_bits=8, weight_exp=plan['first_weight_exp'])
    relu = GradedReluVec(shape=(2*h,), vth=0)
    scores = Dense(weights=np.asarray(plan['second_weights'], np.int32), num_message_bits=24,
                   sign_mode=SignMode.MIXED, num_weight_bits=8, weight_exp=plan['second_weight_exp'])
    readback = Sparse(weights=csr_matrix(2*np.concatenate((np.eye(h,dtype=np.int32),-np.eye(h,dtype=np.int32)),axis=1)),
                      num_message_bits=24, sign_mode=SignMode.MIXED, num_weight_bits=8, weight_exp=-1)
    bias = Source(data=np.broadcast_to(np.asarray(plan['score_bias'],np.int32)[:,None],(classes,ticks)).copy())
    score_sink, hidden_sink = Sink(shape=(classes,),buffer=ticks), Sink(shape=(h,),buffer=ticks)
    source.s_out.connect(first.s_in)
    first.a_out.connect(relu.a_in)
    relu.s_out.connect(scores.s_in)
    relu.s_out.connect(readback.s_in)
    scores.a_out.connect(score_sink.a_in)
    bias.s_out.connect(score_sink.a_in)
    readback.a_out.connect(hidden_sink.a_in)
    try:
        source.run(condition=RunSteps(num_steps=ticks),run_cfg=Loihi2SimCfg(select_tag='fixed_pt'))
        delay = plan['pipeline_delay_ticks']
        y = score_sink.data.get().T[delay:delay+len(x)].copy()
        hidden = hidden_sink.data.get().T[delay:delay+len(x)].copy()
        values = {**{n:hidden[:,i] for i,n in enumerate(plan['hidden_states'])},
                  **{n:y[:,i] for i,n in enumerate(plan['score_states'])}}
        return np.stack([values[n] for n in plan['outputs']],axis=1)
    finally:
        source.stop()


def _specksim(plan, x, batch_size):
    import torch
    from torch import nn
    from sinabs.layers import IAF
    from sinabs.backend.dynapcnn.specksim import from_sequential
    if not isinstance(batch_size, int) or not 1 <= batch_size <= 50:
        raise ValueError('Specksim batch_size must be in [1,50]')
    inputs, outputs = len(plan['inputs']), len(plan['hidden_states'])
    height, width = (28,28) if inputs == 784 else (1,inputs)
    all_counts, all_voltages = [], []
    net = None
    previous_n = None
    for start in range(0,len(x),batch_size):
        rows = x[start:start+batch_size]
        n = len(rows)
        if net is None or n != previous_n:
            conv = nn.Conv2d(1,outputs,(height,width),stride=(height,width),bias=False)
            with torch.no_grad():
                conv.weight.copy_(torch.tensor(plan['weights'],dtype=torch.float32).reshape(outputs,1,height,width))
            net = from_sequential(nn.Sequential(conv,IAF(spike_threshold=plan['threshold'],min_v_mem=0)),
                                  input_shape=(1,height,width*n))
            net.initial_sleep_duration = 1.0
            net.subsequent_sleep_duration = .1
            previous_n = n
        else:
            net.reset_states()
        np.testing.assert_array_equal(np.asarray(net.read_spiking_layer_states(0)),0)
        lanes, pixels = np.nonzero(rows)
        order = np.lexsort((lanes,pixels))
        lanes, pixels = lanes[order], pixels[order]
        events = np.empty(len(pixels),dtype=net.output_dtype)
        events['x'] = lanes*width+pixels%width
        events['y'] = pixels//width
        # One event at each input-index tick for each independent sample.
        events['t'],events['p'] = pixels,0
        if len(events):
            output = net(events)
            if (np.any(output['y']) or np.any(output['x'] >= n) or np.any(output['p'] >= outputs)):
                raise RuntimeError('Specksim produced out-of-domain output events')
            ids = output['x'].astype(np.int64)*outputs+output['p']
            counts = np.bincount(ids,minlength=n*outputs).reshape(n,outputs)
        else:
            counts = np.zeros((n,outputs),np.int64)
        voltage = np.asarray(net.read_spiking_layer_states(0),dtype=np.float64).reshape(outputs,n).T.copy()
        all_counts.append(counts)
        all_voltages.append(voltage)
        if start % (batch_size*20) == 0:
            print(f'Specksim {start+n}/{len(x)} samples',flush=True)
    counts = np.concatenate(all_counts)
    return {'states': counts[:,plan['output_indices']], 'counts':counts,
            'voltage':np.concatenate(all_voltages)}
