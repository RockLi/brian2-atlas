"""Experimental Akida lowering of bounded, exact-integer B2IR v1 subsets.

One population, one tick, binary neuron parameters, a single straight-line
run_regularly block: independent clip(affine(inputs), 0, upper) assignments,
followed by affine(hidden) assignments. Every state is overwritten exactly once
and returned. No B2IR fields are added and no temporal/event semantics erased.
The opt-in binary-classifier-v1 profile admits 64 inputs, 32 states, 10000
population elements and output bias. It uses algebraic interval bounds rather
than exhaustive domain enumeration, plus explicit SDK regression probes.
binary-classifier-v2 extends only the input limit to 128 (Atlas's default parameter budget).
binary-classifier-v3 additionally admits 128 states with an updated Atlas runner.
The opt-in dense-int8-v1 profile admits integer [0,127] inputs and explicit
round-half-up power-of-two scaling and saturation in the source B2IR, with up
to 1024 inputs and 512 states. Larger Atlas budgets must be selected explicitly.
"""
import hashlib
import itertools
import json
import math
from pathlib import Path
import struct
import subprocess
import tempfile

import numpy as np


class B2IRAkidaError(ValueError):
    """Invalid B2IR or semantics outside the selected exact-integer subset."""


def _require(condition, reason):
    if not condition:
        raise B2IRAkidaError(reason)


def _float(bits):
    return struct.unpack('>d', bytes.fromhex(bits))[0]


def _constant(expr, limit=127):
    op = expr['op']
    if op == 'literal':
        value = _float(expr['bits'])
    elif op == 'integer':
        value = int(expr['value'])
    elif op == 'cast' and expr['dtype'] == 'f64':
        return _constant(expr['arg'], limit)
    elif op == 'neg':
        value = -_constant(expr['arg'], limit)
    else:
        raise B2IRAkidaError('expected an integer constant, not a state/time/function read')
    _require(math.isfinite(value) and int(value) == value and abs(value) <= limit,
             f'constants must be exact integers in [-{limit},{limit}]')
    return int(value)


def _affine(expr, names, maxima, constant_limit=127):
    """Extract coefficients from the AST, preserving exact intermediate bounds."""
    op = expr['op']
    if op == 'load':
        _require(expr['name'] in names, f"unsupported read before assignment or external read: {expr['name']}")
        coefficients = np.zeros(len(names), dtype=np.int64)
        coefficients[names.index(expr['name'])] = 1
        return coefficients, 0
    if op in ('literal', 'integer', 'cast'):
        return np.zeros(len(names), dtype=np.int64), _constant(expr, constant_limit)
    if op == 'neg':
        coefficients, bias = _affine(expr['arg'], names, maxima, constant_limit)
        coefficients, bias = -coefficients, -bias
    elif op in ('add', 'sub'):
        a, b = _affine(expr['left'], names, maxima, constant_limit)
        c, d = _affine(expr['right'], names, maxima, constant_limit)
        sign = 1 if op == 'add' else -1
        coefficients, bias = a + sign * c, b + sign * d
    elif op == 'mul':
        a, b = _affine(expr['left'], names, maxima, constant_limit)
        c, d = _affine(expr['right'], names, maxima, constant_limit)
        _require(not a.any() or not c.any(), 'nonlinear multiplication is unsupported')
        coefficients, bias = (c * b, d * b) if not a.any() else (a * d, b * d)
    else:
        raise B2IRAkidaError(f'unsupported expression operation: {op}')
    # All accepted operations remain exact integers in f64 and f32; no rounding
    # tolerance is used for this backend profile. The bound also avoids int64 overflow.
    _require(int(np.abs(coefficients) @ maxima) + abs(bias) < 2**24,
             'intermediate arithmetic exceeds exact float32 integer range')
    return coefficients, bias


def _evaluate(expr, values):
    """Direct f64 AST evaluation used to check the bounded domain proof."""
    op = expr['op']
    if op == 'load':
        return values[expr['name']]
    if op == 'literal':
        return np.float64(_float(expr['bits']))
    if op == 'integer':
        return np.float64(int(expr['value']))
    if op == 'cast':
        return np.asarray(_evaluate(expr['arg'], values), dtype=np.float64)
    if op == 'neg':
        return -_evaluate(expr['arg'], values)
    if op == 'floor':
        return np.floor(_evaluate(expr['arg'], values))
    if op == 'clip':
        return np.clip(_evaluate(expr['value'], values), _evaluate(expr['min'], values), _evaluate(expr['max'], values))
    left, right = _evaluate(expr['left'], values), _evaluate(expr['right'], values)
    return {'add': np.add, 'sub': np.subtract, 'mul': np.multiply, 'div': np.divide}[op](left, right)


def _lower_validated(model, profile='binary-affine-v1'):
    """Capability analysis only; public entry point first invokes Rust validation."""
    _require(profile in ('binary-affine-v1', 'binary-classifier-v1', 'binary-classifier-v2', 'binary-classifier-v3', 'dense-int8-v1'),
             'unknown Akida capability profile')
    wide = profile != 'binary-affine-v1'
    input_limit = {'binary-affine-v1': 8, 'binary-classifier-v1': 64, 'binary-classifier-v2': 128,
                   'binary-classifier-v3': 128, 'dense-int8-v1': 1024}[profile]
    state_limit = 512 if profile == 'dense-int8-v1' else 128 if profile == 'binary-classifier-v3' else 32 if wide else 16
    d, instance, run = model['definition'], model['instance'], model['run']
    _require(model['schema'] == 'b2ir-v1', 'only b2ir-v1 is supported')
    _require(d['numeric_profile'] == 'reference-f64', 'requires reference-f64 semantics')
    _require(len(d['populations']) == 1 and len(d['clocks']) == 1, 'requires one population and one clock')
    _require(not d['synapses'] and not instance['synapses'] and not d['functions'],
             'synapses, events and user functions are unsupported')
    _require(run['clocks'] == [dict(start_tick=0, steps=1)] and _float(run['start']) == 0,
             'requires exactly one tick from time zero; continuation is unsupported')
    p, state = d['populations'][0], instance['populations'][0]
    _require(1 <= p['count'] <= (10000 if wide else 256), 'population size exceeds profile limit')
    _require(p['steps'] == 1 and p['clock'] == 0 and p['offset'] == 0, 'unsupported population schedule')
    _require(p['refractory'] is None and state['refractory'] is None and state['spike_generator'] is None,
             'refractory and spike-generator state are unsupported')
    for name in ['events', 'linked_variables', 'event_monitors', 'rate_monitors', 'state_monitors', 'monitor_expressions']:
        _require(not p.get(name), f'{name} is unsupported')
    _require(p['spike_monitor'] is None and not p['monitor']['record'] and not p['monitor']['variables'],
             'monitors are unsupported; all final states are returned instead')
    _require(not p.get('spatial'), 'spatial dynamics are unsupported')
    inputs = [v['name'] for v in p['parameters']]
    outputs = [v['name'] for v in p['states']]
    _require(1 <= len(inputs) <= input_limit and 1 <= len(outputs) <= state_limit,
             'input/state count exceeds profile limit')
    for v in p['parameters'] + p['states']:
        _require(v['dtype'] == 'f64' and v['index_domain'] == 'neuron' and not any(v['dimensions']),
                 f"{v['name']}: requires dimensionless neuron-domain f64 storage")
    samples = np.array([[_float(x) for x in state['parameters'][name]] for name in inputs]).T
    if profile == 'dense-int8-v1':
        _require(samples.shape == (p['count'], len(inputs)) and np.isfinite(samples).all() and
                 (samples == np.floor(samples)).all() and ((0 <= samples) & (samples <= 127)).all(),
                 'dense-int8 requires integer inputs in [0,127]')
    else:
        _require(samples.shape == (p['count'], len(inputs)) and np.isin(samples, [0, 1]).all(),
                 'every input parameter must contain only binary values 0 or 1')
    active = []
    for code in p['code_objects']:
        _require(not code['scalar'], 'scalar statements are outside this minimal profile')
        if code['vector']:
            active.append(code)
        else:
            _require(code['kind'] == 'state_update', 'only an empty state_update may accompany the static block')
    _require(len(active) == 1, 'requires exactly one nonempty code object')
    code = active[0]
    _require(code['kind'] == 'run_regularly' and code['iteration_domain'] == 'all_neurons' and
             code['clock'] == 0 and code['when'] == 'start', 'requires run_regularly over all neurons at start')
    _require(all(n['operation'] == 'code_object' for n in d['schedule']['nodes']),
             'schedule contains non-code operations')
    if profile == 'dense-int8-v1':
        return _lower_quantized(model, inputs, outputs, samples, code)
    hidden, kernels, biases, maxima, final, final_bias = [], [], [], [], {}, {}
    seen = set()
    for statement in code['vector']:
        target, expr = statement['target'], statement['value']
        _require(statement['condition'] is None and statement['dtype'] == 'f64' and not any(statement['dimensions']),
                 'conditional writes, units and non-f64 statements are unsupported')
        _require(target in outputs and target not in seen, 'each state must be assigned exactly once; temporaries are unsupported')
        seen.add(target)
        if expr['op'] == 'clip':
            _require(not final, 'hidden clip assignments must precede all output assignments')
            _require(_constant(expr['min']) == 0, 'clip lower bound must be zero')
            upper = _constant(expr['max'])
            w, b = _affine(expr['value'], inputs, np.ones(len(inputs), dtype=np.int64))
            peak = max(0, int(np.maximum(w, 0).sum()) + b)
            _require(0 <= peak <= upper <= 127, 'clip upper bound can be active; cannot replace with ReLU')
            _require(np.abs(w).max() <= 127 and abs(b) <= 127, 'hidden weights/bias do not fit int8')
            hidden.append(target)
            kernels.append(w)
            biases.append(b)
            maxima.append(peak)
        else:
            _require(hidden, 'requires at least one hidden clip assignment')
            w, b = _affine(expr, hidden, np.asarray(maxima, dtype=np.int64))
            _require(abs(b) <= 127 if wide else b == 0, 'output bias is outside this profile')
            _require(np.abs(w).max() <= 127, 'output weights do not fit int8')
            final[target] = w
            final_bias[target] = b
    _require(seen == set(outputs) and final, 'all states must be overwritten; requires an affine output')
    # Include hidden states in the final Dense outputs: they are observable B2IR
    # state too, and must not silently disappear during deployment.
    out_weights = np.stack([np.eye(len(hidden), dtype=np.int64)[hidden.index(name)]
                            if name in hidden else final[name] for name in outputs], axis=1)
    w1 = np.stack(kernels, axis=1)
    out_bias = np.array([0 if name in hidden else final_bias[name] for name in outputs], dtype=np.int64)
    if wide:
        # Sound interval bound, independent of the sampled checks below. ReLU
        # inputs fit 8 bits; output accumulators fit signed 16 bits and exact f32.
        bound = np.asarray(maxima, dtype=np.int64) @ np.abs(out_weights) + np.abs(out_bias)
        _require((bound <= 32767).all(), 'output accumulator bound exceeds signed 16-bit profile')
        # The all-zero hidden vector checks signed zero separately from magnitude.
        zero_values = {name: np.float64(0) for name in hidden}
        for statement in code['vector'][len(hidden):]:
            zero = _evaluate(statement['value'], zero_values)
            _require(not (zero == 0 and np.signbit(zero)), 'observable negative zero is unsupported')
        rng = np.random.default_rng(0)
        domain = np.unique(np.concatenate([
            np.zeros((1, len(inputs)), dtype=np.int64), np.ones((1, len(inputs)), dtype=np.int64),
            np.eye(len(inputs), dtype=np.int64), 1 - np.eye(len(inputs), dtype=np.int64),
            rng.integers(0, 2, size=(1024, len(inputs))), samples.astype(np.int64),
        ]), axis=0)
    else:
        domain = np.array(list(itertools.product([0, 1], repeat=len(inputs))), dtype=np.int64)
    truth = np.maximum(domain @ w1 + biases, 0) @ out_weights + out_bias
    _require(np.abs(truth).max() <= (32767 if wide else 127), 'outputs exceed the bounded integer profile')
    values = {name: domain[:, index].astype(np.float64) for index, name in enumerate(inputs)}
    for statement in code['vector']:
        values[statement['target']] = np.broadcast_to(_evaluate(statement['value'], values), (len(domain),))
    direct = np.stack([values[name] for name in outputs], axis=1)
    _require(np.array_equal(truth, direct), 'affine lowering differs from direct B2IR expression evaluation')
    _require(not ((direct == 0) & np.signbit(direct)).any(), 'observable negative zero is outside this integer profile')
    plan = dict(schema='atlas.akida.binary-affine-plan.v1', source_schema=model['schema'],
                source_hashes=model['protocol']['layers'], inputs=inputs, outputs=outputs,
                hidden_states=hidden, weights1=w1.tolist(), bias1=biases,
                weights2=out_weights.tolist(), source_inputs=samples.astype(int).tolist(),
                input_domain='each input is exactly 0 or 1', exact_integer_arithmetic=True,
                )
    if wide:
        plan.update(schema='atlas.akida.binary-classifier-plan.' + profile.rsplit('-', 1)[1],
                    capability_profile=profile, bias2=out_bias.tolist(),
                    proof='affine algebra and interval bounds; SDK checks are not exhaustive',
                    output_absolute_bounds=bound.tolist(), verification_inputs=domain.tolist(),
                    verification_outputs=truth.tolist())
    else:
        plan.update(exhaustive_inputs=domain.tolist(), exhaustive_outputs=truth.tolist())
    return plan


def lower_b2ir(model_path, *, runner, profile='binary-affine-v1'):
    """Validate a real B2IR snapshot with Atlas, then automatically derive weights.

    No Akida SDK or generated artifacts are needed for capability analysis.
    The runner must be a trusted Atlas binary supplied by the caller.
    """
    from .protocol import canonical_bytes, verify_protocol
    model = json.loads(Path(model_path).read_text())
    _require(model.get('schema') == 'b2ir-v1', 'only b2ir-v1 is supported')
    verify_protocol(model)
    encoded = canonical_bytes(model)
    with tempfile.TemporaryDirectory(prefix='atlas-akida-validate-') as temporary:
        snapshot = Path(temporary) / 'model.json'
        snapshot.write_bytes(encoded)
        validation = subprocess.run([str(Path(runner).resolve()), '--validate', str(snapshot)],
                                    capture_output=True, text=True, timeout=60)
    _require(validation.returncode == 0, 'Atlas B2IR validation failed: ' + validation.stderr[-2000:])
    plan = _lower_validated(model, profile=profile)
    plan['source_model_sha256'] = hashlib.sha256(encoded).hexdigest()
    plan['atlas_validator_sha256'] = hashlib.sha256(Path(runner).read_bytes()).hexdigest()
    return plan


def export_b2ir_plan(model_path, directory, *, runner, profile='binary-affine-v1'):
    """Write a derived backend-private plan; the B2IR v1 format stays unchanged."""
    plan = lower_b2ir(model_path, runner=runner, profile=profile)
    from .protocol import canonical_bytes
    source = canonical_bytes(json.loads(Path(model_path).read_text()))
    _require(hashlib.sha256(source).hexdigest() == plan['source_model_sha256'],
             'B2IR source changed during export')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    (directory / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    (directory / 'source-b2ir.json').write_bytes(source)
    return plan


def build_akida_model(plan):
    """Materialize the compiler-derived matrices using the optional Akida SDK.

    The plan is a trusted compiler artifact, not a replacement for B2IR validation.
    """
    import akida
    _require(plan['schema'] in ('atlas.akida.binary-affine-plan.v1', 'atlas.akida.binary-classifier-plan.v1',
                               'atlas.akida.binary-classifier-plan.v2', 'atlas.akida.binary-classifier-plan.v3',
                               'atlas.akida.dense-int8-plan.v1'),
             'unsupported plan schema')
    quantized = plan['schema'] == 'atlas.akida.dense-int8-plan.v1'
    emitted = ([i for i,n in enumerate(plan['outputs']) if n not in plan['hidden_states']]
               if quantized else list(range(len(plan['outputs']))))
    output_name = 'affine_outputs' if quantized else 'all_final_states'
    model = akida.Model([
        akida.InputData(input_shape=(1, 1, len(plan['inputs'])), input_bits=8),
        akida.Dense1D(units=len(plan['hidden_states']), activation=akida.ActivationType.ReLU, name='hidden'),
        # Binary classifiers prove a signed 16-bit range; dense-int8 proves
        # exact integer magnitudes below 2**24. Unactivated hardware outputs
        # allow 8, 28 or 32 bits, not 16. Use 32-bit
        # storage to avoid the SDK default's 8-bit score saturation.
        akida.Dense1D(units=len(emitted), name=output_name,
                      output_bits=8 if plan['schema'] == 'atlas.akida.binary-affine-plan.v1' else 32),
        akida.Dequantizer(),
    ])
    model.get_layer('hidden').variables['weights'] = np.asarray(plan['weights1'], dtype=np.int8)
    model.get_layer('hidden').variables['bias'] = np.asarray(plan['bias1'], dtype=np.int8)
    if plan['schema'] == 'atlas.akida.dense-int8-plan.v1':
        model.get_layer('hidden').variables['bias_shift'] = np.asarray(plan['bias_shift1'], dtype=np.uint8)
        model.get_layer('hidden').variables['output_shift'] = -np.asarray(plan['shift1'], dtype=np.int8)
        model.get_layer(output_name).variables['bias_shift'] = np.asarray(plan['bias_shift2'], dtype=np.uint8)[emitted]
    model.get_layer(output_name).variables['weights'] = np.asarray(plan['weights2'], dtype=np.int8)[:,emitted]
    if 'bias2' in plan:
        model.get_layer(output_name).variables['bias'] = np.asarray(plan['bias2'], dtype=np.int8)[emitted]
    check_inputs = plan.get('verification_inputs', plan.get('exhaustive_inputs'))
    check_outputs = plan.get('verification_outputs', plan.get('exhaustive_outputs'))
    predictor = predict_quantized if plan['schema'] == 'atlas.akida.dense-int8-plan.v1' else predict_binary
    actual = predictor(model, plan, check_inputs)
    _require(np.array_equal(actual, check_outputs), 'Akida differs from the proven-domain verification outputs')
    _require(not ((actual == 0) & np.signbit(actual)).any(), 'Akida produced observable negative zero')
    return model


def predict_binary(model, plan, samples):
    """Enforce the proven domain at the inference boundary before any int8 cast."""
    samples = np.asarray(samples)
    _require(samples.ndim == 2 and samples.shape[1] == len(plan['inputs']) and len(samples) > 0,
             'incorrect input dimensions')
    _require(np.isin(samples, [0, 1]).all(), 'inference input is outside proven binary domain')
    return model.predict(samples.astype(np.int8).reshape(-1, 1, 1, len(plan['inputs']))).reshape(-1, len(plan['outputs']))


def _bias_encoding(value):
    for shift in range(16):
        if value % (1 << shift) == 0 and abs(value // (1 << shift)) <= 127:
            return value // (1 << shift), shift
    raise B2IRAkidaError('bias cannot be represented exactly as int8 times a power of two')


def _lower_quantized(model, inputs, outputs, samples, code):
    """Match explicit round-half-up, saturating integer Dense semantics in B2IR.

    No rounding is inserted by the compiler. The B2IR must already contain
    clip(floor((affine + half_step)/step), 0, 127) for every hidden state.
    """
    hidden, kernels, biases, shifts, finals, final_bias = [], [], [], [], {}, {}
    seen = set()
    for statement in code['vector']:
        target, expr = statement['target'], statement['value']
        _require(statement['condition'] is None and statement['dtype'] == 'f64' and not any(statement['dimensions']),
                 'requires unconditional dimensionless f64 assignments')
        _require(target in outputs and target not in seen, 'each state must be assigned exactly once')
        seen.add(target)
        if expr['op'] == 'clip':
            _require(not finals, 'all hidden assignments must precede outputs')
            _require(_constant(expr['min']) == 0 and _constant(expr['max']) == 127,
                     'quantized activation must saturate to [0,127]')
            rounded = expr['value']
            _require(rounded['op'] == 'floor' and rounded['arg']['op'] == 'div',
                     'requires explicit floor of a power-of-two division')
            quotient = rounded['arg']
            step = _constant(quotient['right'], 2**16)
            _require(2 <= step <= 2**16 and step & (step-1) == 0, 'shift divisor must be a power of two in [2,65536]')
            w, b = _affine(quotient['left'], inputs, np.full(len(inputs),127,dtype=np.int64), 2**23)
            b -= step//2
            _require(np.abs(w).max() <= 127, 'weights must fit signed int8')
            _bias_encoding(b)
            _require(int(np.abs(w).sum())*127+abs(b)+step//2 < 2**24,
                     'hidden accumulator exceeds exact float32 range')
            hidden.append(target); kernels.append(w); biases.append(b); shifts.append(step.bit_length()-1)
        else:
            _require(hidden, 'requires hidden activation before affine outputs')
            w,b=_affine(expr,hidden,np.full(len(hidden),127,dtype=np.int64),2**23)
            _require(np.abs(w).max() <= 127, 'weights must fit signed int8')
            _bias_encoding(b)
            finals[target],final_bias[target]=w,b
    _require(seen == set(outputs) and finals, 'all states must be overwritten; requires affine outputs')
    w1=np.stack(kernels,axis=1)
    w2=np.stack([np.eye(len(hidden),dtype=np.int64)[hidden.index(name)] if name in hidden else finals[name]
                 for name in outputs],axis=1)
    b2=np.array([0 if name in hidden else final_bias[name] for name in outputs],dtype=np.int64)
    bounds=127*np.abs(w2).sum(0)+np.abs(b2)
    _require((bounds < 2**24).all(), 'output accumulator exceeds exact float32 range')
    rng=np.random.default_rng(42)
    # Algebra proves the full bounded domain. These finite AST/SDK probes are
    # supplemental; the experiment compares every actual Atlas state separately.
    domain=np.unique(np.concatenate([np.zeros((1,len(inputs))),np.full((1,len(inputs)),127),
                    rng.integers(0,128,size=(64,len(inputs))),samples[:64]]),axis=0)
    h=np.clip(np.floor((domain@w1+np.array(biases)+2.**(np.array(shifts)-1))/2.**np.array(shifts)),0,127)
    truth=h@w2+b2
    values={name:domain[:,i] for i,name in enumerate(inputs)}
    for statement in code['vector']:
        values[statement['target']]=np.broadcast_to(_evaluate(statement['value'],values),(len(domain),))
    direct=np.stack([values[name] for name in outputs],axis=1)
    _require(np.array_equal(direct,truth), 'quantized lowering differs from direct B2IR expression evaluation')
    _require(not ((direct==0)&np.signbit(direct)).any(), 'observable negative zero is unsupported')
    eb1,bs1=zip(*[_bias_encoding(int(v)) for v in biases])
    eb2,bs2=zip(*[_bias_encoding(int(v)) for v in b2])
    return dict(schema='atlas.akida.dense-int8-plan.v1',capability_profile='dense-int8-v1',source_schema=model['schema'],
                source_hashes=model['protocol']['layers'],inputs=inputs,outputs=outputs,hidden_states=hidden,
                weights1=w1.tolist(),bias1=list(eb1),bias_shift1=list(bs1),shift1=shifts,
                weights2=w2.tolist(),bias2=list(eb2),bias_shift2=list(bs2),
                input_domain='integer [0,127]',exact_integer_arithmetic=True,
                state_readback='hidden states via copied SDK model prefix; affine states via deployed output layer',
                proof='affine algebra and exact power-of-two rounding/saturation; sampled checks are not exhaustive',
                output_absolute_bounds=bounds.tolist(),verification_inputs=domain.astype(int).tolist(),
                verification_outputs=truth.astype(int).tolist())


def predict_quantized(model, plan, samples):
    samples=np.asarray(samples)
    _require(samples.ndim==2 and samples.shape[1]==len(plan['inputs']) and len(samples)>0, 'incorrect input dimensions')
    _require(np.isfinite(samples).all() and (samples==np.floor(samples)).all() and
             ((samples>=0)&(samples<=127)).all(), 'inference input is outside proven integer [0,127] domain')
    import akida
    encoded=samples.astype(np.int8).reshape(-1,1,1,len(plan['inputs']))
    finals=[n for n in plan['outputs'] if n not in plan['hidden_states']]
    scores=model.predict(encoded).reshape(-1,len(finals))
    # The SDK copies layers from the actual model (including after .fbz reload).
    # This diagnostic pass preserves observable B2IR hidden states without
    # adding identity outputs to the deployable classifier's hardware head.
    prefix=akida.Model(model.layers[:2])
    hidden=prefix.forward(encoded).reshape(-1,len(plan['hidden_states']))
    values={**{n:hidden[:,i] for i,n in enumerate(plan['hidden_states'])},
            **{n:scores[:,i] for i,n in enumerate(finals)}}
    return np.stack([values[n] for n in plan['outputs']],axis=1)
