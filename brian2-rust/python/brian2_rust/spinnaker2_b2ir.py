"""Bounded AtlasIR LIF lowering to the optional SpiNNaker2 Brian2 backend.

This profile targets SDK software semantics, not chip arithmetic or mapping.
It recognizes the Euler AST literally; it does not infer LIF from a name.
"""
import hashlib
import json
import struct
import subprocess
import tempfile
from pathlib import Path

import numpy as np

PROFILE = 'spinnaker2-lif-v1'
SLOTS = ['start', 'groups', 'synapses', 'thresholds', 'resets', 'end']


def require(condition, reason):
    if not condition:
        raise ValueError('SpiNNaker2: ' + reason)


def number(bits):
    return struct.unpack('>d', bytes.fromhex(bits))[0]


def load(name):
    return dict(op='load', name=name)


def op(kind, left, right):
    return dict(op=kind, left=left, right=right)


# Retain operation order, including the source dt, rather than reassociating
# floating-point arithmetic into alpha*v + bias.
EULER = op('add', op('mul', load('dt'), op('add',
    op('div', dict(op='neg', arg=load('v')), load('tau')),
    op('div', load('i_offset'), load('dt')))), load('v'))
THRESHOLD = op('gt', load('v'), load('threshold'))
RESETS = {'reset_to_v_reset': load('v_reset'),
          'reset_by_subtraction': op('sub', load('v'), load('threshold'))}


def _code(code, *, kind, when, domain, target, value, clock, order=0):
    require(code['kind'] == kind and code['when'] == when and code['order'] == order
            and code['iteration_domain'] == domain and code['clock'] == clock,
            'unsupported code schedule/domain')
    require(not code['scalar'] and len(code['vector']) == 1,
            'requires one vector assignment without scalar temporaries')
    s = code['vector'][0]
    require(s['target'] == target and s['condition'] is None and not any(s['dimensions'])
            and s['dtype'] == ('bool' if kind == 'threshold' else 'f64')
            and s['value'] == value, 'unsupported ' + kind + ' AST')
    if kind in ('threshold', 'reset', 'synapses'):
        require(code['event_name'] == 'spike', 'only spike events are supported')


def _lower_validated(model):
    d, instance, run = model['definition'], model['instance'], model['run']
    require(model['schema'] == 'b2ir-v1' and d['numeric_profile'] == 'reference-f64',
            'requires reference-f64 b2ir-v1')
    require(not d['functions'], 'user functions are unsupported')
    require(d['schedule']['base_slots'] == SLOTS,
            'requires groups → synapses → thresholds → resets scheduling')
    require(number(run['start']) == 0 and run['clocks'], 'requires a fresh run at time zero')
    steps = run['clocks'][0]['steps']
    require(1 <= steps <= 10000 and all(c == dict(start_tick=0, steps=steps) for c in run['clocks'])
            and all(number(c['dt']) == 0.001 for c in d['clocks']),
            'all clocks must start at zero with dt=1 ms and the same bounded duration')
    require(len(d['populations']) == 2, 'pilot requires one spike source and one LIF population')
    populations, lif_index, source_index = [], None, None
    for index, (p, state) in enumerate(zip(d['populations'], instance['populations'])):
        require(1 <= p['count'] <= 256 and p['steps'] == steps and number(p['dt']) == 0.001,
                'population exceeds the pilot size/time limits')
        for key in ('linked_variables', 'event_monitors', 'rate_monitors', 'monitor_expressions', 'spatial'):
            require(not p.get(key), key + ' is unsupported')
        require(p['refractory'] is None and state['refractory'] is None
                and p['events'] == ['spike'], 'refractory/custom events are unsupported')
        if state['spike_generator'] is not None:
            require(source_index is None and not p['states'] and not p['parameters']
                    and not p['code_objects'] and not p['state_monitors'] and p['spike_monitor'] is None
                    and not p['monitor']['record'] and not p['monitor']['variables'],
                    'requires one unmonitored static spike source')
            source_index = index
            schedule = state['spike_generator']
            params = {str(i): [] for i in range(p['count'])}
            for i, tick in zip(schedule['spike_indices'], schedule['spike_ticks']):
                params[str(i)].append(tick)
            populations.append(dict(name=p['name'], size=p['count'], neuron_model='spike_list', params=params))
            continue
        require(lif_index is None, 'only one LIF population is supported')
        lif_index = index
        require([s['name'] for s in p['states']] == ['v']
                and {s['name'] for s in p['parameters']} == {'tau', 'threshold', 'v_reset', 'i_offset'},
                'requires v and constant tau/threshold/v_reset/i_offset')
        for symbol in p['states'] + p['parameters']:
            dims = [0., 0., 1., 0., 0., 0., 0.] if symbol['name'] == 'tau' else [0.] * 7
            require(symbol['dtype'] == 'f64' and symbol['index_domain'] == 'neuron'
                    and symbol['dimensions'] == dims, 'unsupported neuron storage/units')
        require(all(bits == '0000000000000000' for bits in state['initial_state']['v']),
                'nonzero initial voltage is unsupported by the SDK lif backend')
        codes = p['code_objects']
        require(len(codes) == 3 and {c['kind'] for c in codes} == {'state_update', 'threshold', 'reset'},
                'requires only Euler update, threshold and reset')
        codes = {c['kind']: c for c in codes}
        _code(codes['state_update'], kind='state_update', when='groups', domain='all_neurons',
              target='v', value=EULER, clock=p['clock'])
        _code(codes['threshold'], kind='threshold', when='thresholds', domain='all_neurons',
              target='_cond', value=THRESHOLD, clock=p['clock'])
        reset = next((name for name, expr in RESETS.items()
                      if len(codes['reset']['vector']) == 1 and codes['reset']['vector'][0]['value'] == expr), None)
        require(reset is not None, 'unsupported reset AST')
        _code(codes['reset'], kind='reset', when='resets', domain='spiking_neurons',
              target='v', value=RESETS[reset], clock=p['clock'])
        values = {name: np.array([number(v) for v in data]) for name, data in state['parameters'].items()}
        tau = values.pop('tau')
        alpha = 1 - 0.001 / tau
        require(np.isfinite(tau).all() and (tau >= 0.001).all() and ((alpha >= 0) & (alpha < 1)).all()
                and np.array_equal(0.001 / np.maximum(1 - alpha, 1e-9), tau),
                'tau cannot round-trip exactly through SDK alpha_decay')
        values['alpha_decay'] = alpha
        require(all(np.isfinite(v).all() and (np.abs(v) <= 1e6).all()
                    and np.array_equal(v.astype(np.float32).astype(np.float64), v) for v in values.values()),
                'parameters must be finite and exactly representable in float32 within pilot bounds')
        require((values['threshold'] > 0).all(), 'threshold must be positive')
        require(p['spike_monitor'] is not None and p['monitor'] == dict(
                    variables=['v'], record=list(range(p['count'])), window_steps=steps)
                and len(p['state_monitors']) == 1, 'requires full voltage and spike recording')
        monitor = p['state_monitors'][0]
        require(monitor['when'] == 'end' and monitor['clock'] == p['clock']
                and monitor['variables'] == ['v'] and monitor['output_variables'] == ['v']
                and monitor['record'] == list(range(p['count'])), 'requires end-of-step voltage recording')
        populations.append(dict(name=p['name'], size=p['count'], neuron_model='lif',
                                params={**{k: v.tolist() for k, v in values.items()}, 'reset': reset}))
    require(lif_index is not None and source_index is not None, 'requires a spike source and LIF population')
    projections = []
    require(1 <= len(d['synapses']) <= 2, 'requires an input projection and optional recurrent projection')
    endpoints = set()
    for s, state in zip(d['synapses'], instance['synapses']):
        src, dst = s['source_population'], s['target_population']
        require(dst == lif_index and (src, dst) not in endpoints, 'unsupported/duplicate projection endpoints')
        endpoints.add((src, dst))
        require(s['source_synapse'] is None and s['target_synapse'] is None
                and s['source_start'] == 0 and s['target_start'] == 0
                and s['source_count'] == d['populations'][src]['count']
                and s['target_count'] == d['populations'][dst]['count'], 'subgroups and synapse targets are unsupported')
        require(not s['states'] and not s['clock_driven_states'] and not s['linked_variables']
                and not s['state_monitors'] and not s['monitor_expressions'], 'plasticity/extra synapse state is unsupported')
        require(s['post_state_aliases'].get('v_post') == 'v' and len(s['code_objects']) == 1,
                'requires one static voltage-increment pathway')
        _code(s['code_objects'][0], kind='synapses', when='synapses', domain='active_synapses',
              target='v_post', value=op('add', load('v_post'), load('w')),
              clock=d['populations'][src]['clock'], order=-1)
        weight_symbol = next((v for v in s['parameters'] if v['name'] == 'w'), None)
        require(weight_symbol is not None and weight_symbol['dtype'] == 'f64'
                and weight_symbol['index_domain'] == 'synapse' and not any(weight_symbol['dimensions']),
                'weights must be dimensionless per-synapse constants')
        require(state['topology'] == {'kind': 'explicit'} and len(state['pathways']) == 1,
                'requires explicit connectivity and one pathway')
        pathway = state['pathways'][0]
        require(pathway['kind'] == 'pre' and pathway['event'] == 'spike' and not pathway['pending']
                and pathway['delay_initializer'] is None, 'pending events/dynamic delays are unsupported')
        w = [number(v) for v in state['parameters']['w']]
        require(all(np.isfinite(v) and v == int(v) and 0 < abs(v) <= 15 for v in w),
                'lif weights must be nonzero integers in [-15,15]; SDK quantization must be lossless')
        delays = pathway['delay_ticks']
        require(all(0 <= v <= 7 for v in delays), 'lif delays must fit 3 bits (0..7 ticks)')
        require(all(number(bits) == ticks * 0.001 for bits, ticks in zip(pathway['delay'], delays)),
                'delay must already be an exact integer number of ticks')
        pairs = list(zip(state['source'], state['target']))
        require(len(set(pairs)) == len(pairs), 'multapses would be merged by the SDK sparse matrices')
        projections.append(dict(name=s['name'], source=src, target=dst,
                                connections=[[i, j, int(weight), delay] for (i, j), weight, delay in zip(pairs, w, delays)]))
    require((source_index, lif_index) in endpoints, 'requires a spike-input projection')
    return dict(schema='atlas.spinnaker2.lif-plan.v1', populations=populations, projections=projections,
                lif_population=lif_index, steps=steps, dt_seconds=0.001,
                schedule=SLOTS, source_hashes=model['protocol']['layers'])


def lower_b2ir(model_path, *, runner):
    from .protocol import canonical_bytes, verify_protocol
    model = json.loads(Path(model_path).read_text())
    verify_protocol(model)
    encoded = canonical_bytes(model)
    with tempfile.TemporaryDirectory(prefix='atlas-spinnaker2-validate-') as tmp:
        snapshot = Path(tmp) / 'model.json'
        snapshot.write_bytes(encoded)
        result = subprocess.run([str(Path(runner).resolve()), '--validate', str(snapshot)],
                                capture_output=True, text=True, timeout=60)
    require(result.returncode == 0, 'Atlas validation failed: ' + result.stderr[-2000:])
    plan = _lower_validated(model)
    plan['source_model_sha256'] = hashlib.sha256(encoded).hexdigest()
    plan['atlas_validator_sha256'] = hashlib.sha256(Path(runner).read_bytes()).hexdigest()
    return plan
