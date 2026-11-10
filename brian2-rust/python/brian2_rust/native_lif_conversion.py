"""Bounded native-training bridges with explicit target semantics.

Only dense v1, subtract-reset, fresh-state inference is admitted. Quantization
is opt-in and produces a distinct frozen model before any target conversion.
"""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from .protocol import canonical_bytes

SCHEMA = 'atlas.native-lif-frozen.v1'
MAX_BRIDGE_STEPS = 16


def _digest(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _require(ok, message):
    if not ok:
        raise ValueError('Native LIF export: ' + message)


def _validate_plan(plan):
    keys = {'schema', 'backend', 'sizes', 'beta', 'threshold', 'reset', 'detach_reset',
            'surrogate', 'optimizer', 'trainable', 'masks', 'seed', 'logit_scale',
            'max_tape_bytes', 'tbptt_window'}
    _require(set(plan) == keys and plan['schema'] == 'b2-lif-training-plan-v1',
             'requires the dense v1 training plan without extensions')
    sizes = plan['sizes']
    _require(3 <= len(sizes) <= 17 and all(type(n) is int and n > 0 for n in sizes)
             and sizes[0] <= 256 and sum(sizes[1:]) <= 256, 'population budget exceeded')
    layers = len(sizes)-1
    _require(all(v == .5 for v in plan['beta']), 'first bridge requires beta=0.5 for every layer')
    _require(plan['reset'] == 'subtract', 'only subtract reset is supported')
    _require(len(plan['beta']) == len(plan['threshold']) == layers, 'layer parameter shape mismatch')
    beta, threshold = np.asarray(plan['beta']), np.asarray(plan['threshold'])
    _require(np.isfinite(beta).all() and ((beta > 0) & (beta < 1)).all(),
             'decay must lie strictly between zero and one')
    _require(np.isfinite(threshold).all() and (threshold > 0).all(), 'invalid threshold')
    _require(np.isfinite(plan['logit_scale']) and plan['logit_scale'] > 0, 'invalid logit scale')
    _require(len(plan['masks']) == layers and all(len(m) == a*b and all(v == 1 for v in m)
             for m,a,b in zip(plan['masks'],sizes,sizes[1:])), 'only unmasked dense weights are supported')
    return sizes


def freeze_native_lif(trainer, *, scales, quantize=False):
    """Freeze a trained v1 model; explicitly opt into nearest-even quantization.

    Target weights are ``scale[layer] * beta[layer] * native_weight``. They
    must fit signed integer [-15,15]. No clipping or optimizer update occurs.
    The returned ``weights`` can be evaluated by a fresh NativeLIFTrainer.
    Inference always starts at zero; live training state is not exported.
    """
    _require(type(quantize) is bool, 'quantize must be boolean')
    plan = copy.deepcopy(trainer.plan)
    sizes = _validate_plan(plan)
    _require(trainer.state is not None and trainer.state['step'] > 0, 'requires a trained checkpoint')
    scales = np.asarray(scales, dtype=float)
    _require(scales.shape == (len(sizes)-1,) and np.isfinite(scales).all()
             and (scales > 0).all(), 'provide one finite positive scale per layer')
    original = trainer.state['weights']
    _require(len(original) == len(sizes)-1, 'weight bank count mismatch')
    weights, integers, errors = [], [], []
    for l,(raw,a,b) in enumerate(zip(original,sizes,sizes[1:])):
        w = np.asarray(raw, dtype=float)
        _require(w.shape == (a*b,) and np.isfinite(w).all(), 'invalid weight bank')
        multiplier = scales[l]*plan['beta'][l]
        q = np.rint(w*multiplier)
        _require(quantize or np.array_equal(w*multiplier,q), 'noninteger target weights; opt into quantization')
        _require((np.abs(q) <= 15).all(), 'target weights exceed [-15,15]; choose smaller scales')
        frozen = q/multiplier if quantize else w
        weights.append(frozen.tolist()); integers.append(q.astype(int).tolist())
        errors.append(float(np.max(np.abs(w-frozen))))
    frozen = dict(schema=SCHEMA, plan=plan, weights=weights, target_weights=integers,
                  scales=scales.tolist(), quantization=dict(enabled=quantize,
                  rounding='nearest-even' if quantize else 'none', clipping=False,
                  max_abs_weight_error=errors), provenance=dict(
                  training_plan_sha256=_digest(plan), training_state_sha256=_digest(trainer.state),
                  training_runner_sha256=hashlib.sha256(trainer.runner.read_bytes()).hexdigest(),
                  optimizer_step=trainer.state['step']), inference_initial_state='zero')
    frozen['sha256'] = _digest(frozen)
    validate_frozen(frozen)
    return frozen


def validate_frozen(frozen):
    _require(frozen.get('schema') == SCHEMA, 'unknown frozen model schema')
    _require(frozen.get('sha256') == _digest({k:v for k,v in frozen.items() if k != 'sha256'}),
             'frozen model integrity mismatch')
    sizes = _validate_plan(frozen['plan'])
    _require(frozen['inference_initial_state'] == 'zero', 'only fresh-state inference is supported')
    layers = len(sizes)-1
    _require(len(frozen['scales']) == len(frozen['weights']) == len(frozen['target_weights']) == layers,
             'frozen layer shape mismatch')
    for l,(a,b) in enumerate(zip(sizes,sizes[1:])):
        scale = frozen['scales'][l]
        beta = frozen['plan']['beta'][l]
        _require(np.isfinite(scale) and scale > 0 and np.log2(scale).is_integer(),
                 'scale must be an exact power of two')
        threshold = scale*frozen['plan']['threshold'][l]
        _require(np.isfinite(scale) and scale > 0 and 0 < threshold <= 1e6
                 and float(np.float32(threshold)) == threshold and float(np.float32(beta)) == beta,
                 'scaled threshold and decay must be exactly float32 representable')
        w,q = np.asarray(frozen['weights'][l]),np.asarray(frozen['target_weights'][l])
        _require(w.shape == q.shape == (a*b,) and np.isfinite(w).all() and np.isfinite(q).all()
                 and (q == np.rint(q)).all() and (np.abs(q) <= 15).all(), 'invalid frozen weights')
        _require(np.allclose(w*(scale*beta),q,rtol=0,atol=4e-15), 'inconsistent native and target weights')
    return sizes


def validate_spike_inputs(frozen, inputs):
    sizes = validate_frozen(frozen)
    x = np.asarray(inputs)
    _require(x.ndim == 3 and 1 <= len(x) and x.shape[2] == sizes[0]
             and 1 <= x.shape[1] <= MAX_BRIDGE_STEPS, 'inputs must be batch × time × input')
    _require(np.isfinite(x).all() and ((x == 0) | (x == 1)).all(), 'input spikes must be binary')
    return x.astype(np.uint8)


def build_native_lif_network(frozen, inputs):
    """Construct a concrete Brian model of the transformed inference state.

    Let native v(t)=beta*v(t-1)-theta*s(t)+J(t). The SDK state is
    z(t)=scale*(v(t)-J(t)); SDK weights are scale*beta*W. Its zero-delay
    synapses consume the previous tick's spikes. Hence thresholds, spike ticks
    and count logits agree in real arithmetic; floating point is audited.
    """
    import brian2 as b
    x = validate_spike_inputs(frozen, inputs)
    batch,steps,inputs_n = x.shape
    plan, sizes = frozen['plan'],frozen['plan']['sizes']
    cells = sum(sizes[1:]); offsets=np.cumsum([0]+sizes[1:])
    clock=b.Clock(dt=b.ms)
    sample,tick,neuron=np.nonzero(x)
    order=np.lexsort((sample*inputs_n+neuron,tick))
    source=b.SpikeGeneratorGroup(batch*inputs_n,(sample*inputs_n+neuron)[order],
                                tick[order]*b.ms,clock=clock,name='native_input*')
    group=b.NeuronGroup(batch*cells,
        'dv/dt = -v/tau + i_offset/dt : 1\n'
        'tau : second (constant)\ni_offset : 1 (constant)\n'
        'v_reset : 1 (constant)\nthreshold : 1 (constant)',
        threshold='v > threshold',reset='v -= threshold',method='euler',clock=clock,name='native_lif*')
    group.tau=np.tile(np.repeat(.001/(1-np.asarray(plan['beta'])),sizes[1:]),batch)*b.second
    group.threshold=np.tile(np.repeat(np.asarray(plan['threshold'])*frozen['scales'],sizes[1:]),batch)
    group.i_offset=0;group.v_reset=0
    input_rows=[];internal_rows=[]
    for l,q in enumerate(frozen['target_weights']):
        w=np.asarray(q).reshape(sizes[l],sizes[l+1]);src,dst=np.nonzero(w)
        for k in range(batch):
            rows=input_rows if l==0 else internal_rows
            rows.extend(zip((src+(k*inputs_n if l==0 else k*cells+offsets[l-1])).tolist(),
                            (dst+k*cells+offsets[l]).tolist(),w[src,dst].tolist()))
    _require(input_rows, 'all input weights are zero')
    objects=[source,group]
    for pre,rows,name in [(source,input_rows,'native_input_projection*'),(group,internal_rows,'native_internal_projection*')]:
        if not rows: continue
        s=b.Synapses(pre,group,'w : 1 (constant)',on_pre='v_post += w',clock=clock,name=name)
        s.connect(i=[r[0] for r in rows],j=[r[1] for r in rows]);s.w=[r[2] for r in rows];s.delay=0*b.ms
        objects.append(s)
    objects += [b.StateMonitor(group,'v',record=True,when='end',name='native_voltage*'),
                b.SpikeMonitor(group,name='native_spikes*')]
    net=b.Network(*objects)
    net.schedule=['start','groups','synapses','thresholds','resets','end']
    return net


def export_native_lif(frozen, directory, *, steps, runner, target='spinnaker2',
                      allow_dynamics_approximation=False):
    """Export a zero-input template via AtlasIR and the existing LIF converter.

    Xylo requires explicit ``allow_dynamics_approximation=True``: its integer
    current/membrane dynamics are not equivalent to the source float LIF.
    The same frozen integer weight banks are retained without retraining.
    Call run_simulator(bundle, binary_rasters) for fresh independent samples.
    The saved source is the zero-input template, not a receipt for later inputs.
    """
    import brian2 as b
    from .export import export_network
    from brian2_atlas.neuromorphic import export_ir
    _require(type(allow_dynamics_approximation) is bool, 'approximation opt-in must be boolean')
    _require(target in ('spinnaker2', 'xylosim'), 'unsupported native LIF target')
    if target == 'xylosim':
        _require(allow_dynamics_approximation,
                 'Xylo changes neuron dynamics; set allow_dynamics_approximation=True explicitly')
        from .native_lif_xylo import build_network, validate_xylo_frozen
        validate_xylo_frozen(frozen)
    else:
        _require(not allow_dynamics_approximation, 'SpiNNaker2 does not require dynamics approximation')
    sizes=validate_frozen(frozen)
    _require(type(steps) is int and 1 <= steps <= MAX_BRIDGE_STEPS, 'inference duration must be 1..16 ticks')
    directory=Path(directory)
    _require(not directory.exists(), 'destination already exists')
    previous=b.prefs.codegen.loop_invariant_optimisations
    try:
        b.prefs.codegen.loop_invariant_optimisations=False
        inputs=np.zeros((1,steps,sizes[0]),dtype=np.uint8)
        net=(build_network(frozen,inputs[0]) if target == 'xylosim'
             else build_native_lif_network(frozen,inputs))
        directory.mkdir(parents=True)
        source=directory/'template.json'
        export_network(net,steps*b.ms,source)
        bundle=export_ir(source,directory/'converted',target=target,profile=('xylo-audio2-temporal-v1' if target == 'xylosim' else 'spinnaker2-lif-v1'),runner=runner)
    finally:
        b.prefs.codegen.loop_invariant_optimisations=previous
    bundle['native_training_bridge']=dict(schema=SCHEMA,frozen_sha256=frozen['sha256'],
        sizes=sizes,logit_scale=frozen['plan']['logit_scale'],fresh_state_per_sample=True,
        source_is_zero_input_template=True,quantization=frozen['quantization'])
    bundle['native_training_bridge']['dynamics_equivalent'] = target == 'spinnaker2'
    if target == 'xylosim':
        bundle['native_training_bridge']['adaptation'] = dict(
            schema='atlas.native-lif.xylo-adaptation.v1', same_integer_weight_banks=True,
            synaptic_shift=1, membrane_shift=1, threshold_rounding='none',
            threshold_rule='greater-or-equal', hidden_spike_cap=31, output_spike_cap=1,
            state_arithmetic='signed-int16-saturating-minimum-decay',
            timing_alignment='none', retrained=False)
    (directory/'frozen.json').write_bytes(canonical_bytes(frozen))
    (directory/'converted/bundle.json').write_bytes(canonical_bytes(bundle))
    return bundle
