"""Strict admission of an integer, temporal Xylo Audio 2 subset of AtlasIR."""
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import tempfile
import numpy as np

from .protocol import SUPPORTED_V1_SCHEMAS
from .spec import expression

PROFILE = 'xylo-audio2-temporal-v1'
SLOTS = ['start', 'synapses', 'groups', 'thresholds', 'resets', 'end']


def require(ok, reason):
    if not ok:
        raise ValueError('Xylo temporal: '+reason)


def number(bits):
    return struct.unpack('>d',bytes.fromhex(bits))[0]


def _expr(source):
    names=['I','v','q','syn_div','mem_div','bias','threshold','spike_cap','I_post','q_pre','w']
    return expression(source,symbol_dtypes={n:'f64' for n in names})


def _decay(x,d):
    return f'(floor({x}/{d})+int({x}>0)*int({x}<{d}))'


def _clip(x):
    return f'clip({x},-32768,32767)'


UPDATE = [
    ('I', _expr('I + -'+_decay('I','syn_div'))),
    ('v', _expr(_clip('v+'+_clip(_clip('-'+_decay('v','mem_div')+'+I')+'+bias')))),
    ('q', _expr('clip(floor(v/threshold),0,spike_cap)')),
]
THRESHOLD = [('_cond',_expr('q>0'))]
RESET = [('v',_expr('v + -q*threshold'))]
SYNAPSE = [('I_post',_expr(_clip('I_post+w*q_pre')))]


def _code(c,kind,when,order,expected,clock,domain='all_neurons'):
    require(c.get('adaptive') is None,'adaptive integrators unsupported')
    require(c['kind']==kind and c['when']==when and c['order']==order
            and c['clock']==clock and c['iteration_domain']==domain,'unsupported code schedule/domain')
    require(not c['scalar'] and len(c['vector'])==len(expected),'unsupported scalar/extra statements')
    for statement,(name,value) in zip(c['vector'],expected):
        require(statement['target']==name and statement['value']==value
                and statement['condition'] is None and not any(statement['dimensions'])
                and statement['dtype']==('bool' if kind=='threshold' else 'f64'),
                'unsupported '+kind+' AST')
    if kind in ('threshold','reset','synapses'):
        require(c['event_name']=='spike','only spike events are supported')


def _lower_validated(model):
    d,instance,run=model['definition'],model['instance'],model['run']
    require(model['schema'] in SUPPORTED_V1_SCHEMAS and d['numeric_profile']=='reference-f64' and not d['functions'],
            'requires reference-f64 AtlasIR without user functions')
    require(d['schedule']['base_slots']==SLOTS,'requires start → synapses → groups → thresholds → resets')
    require(number(run['start'])==0 and len(d['clocks'])==len(run['clocks'])==1,'requires one fresh clock')
    steps=run['clocks'][0]['steps']
    require(run['clocks'][0]['start_tick']==0 and 1<=steps<=10000
            and number(d['clocks'][0]['dt'])==.001,'requires 1 ms and 1..10000 steps')
    require(len(d['populations'])==3,'requires input, hidden and output populations')
    roles={};params={};raster=None
    for index,(p,state) in enumerate(zip(d['populations'],instance['populations'])):
        require(p['clock']==0 and p['steps']==steps and number(p['dt'])==.001,'unaligned population')
        require(p['events']==['spike'] and p['refractory'] is None and state['refractory'] is None
                and state['spike_generator'] is None,'custom events/refractory/static generators unsupported')
        require(all(not p.get(k) for k in ('linked_variables','event_monitors','rate_monitors','monitor_expressions','spatial')),
                'extra population features unsupported')
        require(all(v=='0000000000000000' for a in state['initial_state'].values() for v in a),
                'requires zero initial states')
        require(all(s['dtype']=='f64' and s['index_domain']=='neuron' and not any(s['dimensions'])
                    for s in p['states']+p['parameters']),'requires dimensionless f64 integer storage')
        codes={c['kind']:c for c in p['code_objects']}
        require(len(codes)==len(p['code_objects']),'duplicate code kinds unsupported')
        _code(codes['state_update'],'state_update','groups',0,[],0)
        if [s['name'] for s in p['states']]==['q']:
            role='input'
            require(not p['parameters'] and set(codes)=={'state_update','run_regularly','threshold'}
                    and 1<=p['count']<=16,'input must have 1..16 channels and only a spike-count raster')
            require(not p['state_monitors'] and p['spike_monitor'] is None
                    and not p['monitor']['record'] and not p['monitor']['variables'],'input recording unsupported')
            _code(codes['threshold'],'threshold','start',1,THRESHOLD,0)
            statements=codes['run_regularly']['vector']
            require(len(statements)==1,'requires one raster assignment')
            table=statements[0]['value']
            require(table['op']=='timed_array' and table['time']==dict(op='load',name='t')
                    and table['index']==dict(op='load',name='i') and table['columns']==p['count']
                    and table['rows']==steps and table['upsampling']==8
                    and number(table['epsilon'])==.001/8 and not any(table['dimensions']),
                    'requires an aligned 1 ms raster covering the full run')
            _code(codes['run_regularly'],'run_regularly','start',0,[('q',table)],0)
            raster=np.array([number(v) for v in table['values']]).reshape(steps,p['count'])
            require(np.isfinite(raster).all() and ((raster>=0)&(raster<=15)&(raster==np.floor(raster))).all(),
                    'input counts must be integers in [0,15]')
        else:
            require([s['name'] for s in p['states']]==['I','q','v']
                    and {s['name'] for s in p['parameters']}=={'syn_div','mem_div','bias','threshold','spike_cap'}
                    and set(codes)=={'state_update','run_regularly','threshold','reset'},'unsupported neuron state/parameters')
            values={k:np.array([number(v) for v in a]) for k,a in state['parameters'].items()}
            require(all(np.isfinite(v).all() and (v==np.floor(v)).all() for v in values.values()),'integer parameters required')
            cap=values['spike_cap'];role='hidden' if (cap==31).all() else 'output'
            require((cap==(31 if role=='hidden' else 1)).all(),'hidden/output spike caps must be 31/1')
            require(1<=p['count']<=(64 if role=='hidden' else 8),'pilot permits 1..64 hidden and 1..8 output neurons')
            require(np.isin(values['syn_div'],2**np.arange(16)).all()
                    and np.isin(values['mem_div'],2**np.arange(1,16)).all(),
                    'decay divisors must be powers of two; membrane shifts 1..15, synaptic shifts 0..15')
            require(((values['threshold']>=1)&(values['threshold']<=32767)).all()
                    and ((values['bias']>=-32768)&(values['bias']<=32767)).all(),'threshold/bias must fit signed 16 bits')
            _code(codes['run_regularly'],'run_regularly','groups',0,UPDATE,0)
            _code(codes['threshold'],'threshold','thresholds',0,THRESHOLD,0)
            _code(codes['reset'],'reset','resets',0,RESET,0,'spiking_neurons')
            require(p['spike_monitor'] is not None and len(p['state_monitors'])==1
                    and p['monitor']==dict(variables=['I','v','q'],record=list(range(p['count'])),window_steps=steps),
                    'requires full state and spike recording')
            mon=p['state_monitors'][0]
            require(mon['when']=='end' and mon.get('order',0)==0 and mon['clock']==0
                    and mon['variables']==mon['output_variables']==['I','v','q']
                    and mon['record']==list(range(p['count'])),'requires full end-of-step I/v/q records')
            params[role]=dict(threshold=values['threshold'].astype(int).tolist(),bias=values['bias'].astype(int).tolist(),
                dash_syn=np.log2(values['syn_div']).astype(int).tolist(),dash_mem=np.log2(values['mem_div']).astype(int).tolist())
        require(role not in roles,'duplicate population role')
        roles[role]=index
    require(set(roles)=={'input','hidden','output'},'missing input/hidden/output population')
    sizes={role:d['populations'][idx]['count'] for role,idx in roles.items()}
    endpoints={(roles['input'],roles['hidden']):'input',(roles['hidden'],roles['hidden']):'recurrent',
               (roles['hidden'],roles['output']):'readout'}
    projections={};syn_roles={}
    require(2<=len(d['synapses'])<=3,'requires input and readout, with optional recurrent projection')
    for index,(s,state) in enumerate(zip(d['synapses'],instance['synapses'])):
        endpoint=(s['source_population'],s['target_population']);role=endpoints.get(endpoint)
        require(role is not None and role not in projections,'unsupported/duplicate projection')
        src,dst=endpoint
        require(s['source_synapse'] is None and s['target_synapse'] is None
                and s['source_start']==s['target_start']==0
                and s['source_count']==d['populations'][src]['count'] and s['target_count']==d['populations'][dst]['count'],
                'subgroups and synapse endpoints unsupported')
        require(all(not s.get(k) for k in ('states','clock_driven_states','linked_variables','state_monitors','monitor_expressions')),
                'plasticity/extra synapse state unsupported')
        require(s['pre_state_aliases'].get('q_pre')=='q' and s['post_state_aliases'].get('I_post')=='I'
                and len(s['code_objects'])==1,'requires one q-weighted current pathway')
        _code(s['code_objects'][0],'synapses','synapses',-1,SYNAPSE,0,'active_synapses')
        require(state['topology']=={'kind':'explicit'} and len(state['pathways'])==1,'requires explicit static connectivity')
        path=state['pathways'][0]
        require(path['kind']=='pre' and path['event']=='spike' and not path['pending'] and path['delay_initializer'] is None
                and all(v==0 for v in path['delay_ticks']) and all(number(v)==0 for v in path['delay']),
                'explicit delays/pending events unsupported; recurrence uses previous-tick spikes')
        weights=np.array([number(v) for v in state['parameters']['w']])
        require(np.isfinite(weights).all() and ((weights>=-128)&(weights<=127)&(weights!=0)&(weights==np.floor(weights))).all(),
                'weights must be nonzero signed 8-bit integers')
        symbol=next(x for x in s['parameters'] if x['name']=='w')
        require(symbol['dtype']=='f64' and symbol['index_domain']=='synapse' and not any(symbol['dimensions']),
                'unsupported weight type/units')
        pairs=list(zip(state['source'],state['target']))
        require(pairs==sorted(set(pairs)),'requires sorted unique connections to preserve saturation order')
        matrix=np.zeros((s['source_count'],s['target_count']),dtype=int)
        for (i,j),w in zip(pairs,weights):matrix[i,j]=int(w)
        projections[role]=matrix.tolist();syn_roles[index]=role
    require({'input','readout'}<=projections.keys(),'missing input/readout projection')
    delivery=[syn_roles[n['owner_index']] for n in d['schedule']['nodes']
              if n['owner_kind']=='synapse' and n['operation']=='code_object']
    require(delivery==[r for r in ('input','recurrent','readout') if r in projections],
            'synapse delivery must be input, recurrent, then readout')
    projections.setdefault('recurrent',np.zeros((sizes['hidden'],sizes['hidden']),dtype=int).tolist())
    require(np.all(np.count_nonzero(np.asarray(projections['recurrent']),axis=1)<=63),
            'recurrent fanout must be at most 63 per hidden neuron')
    return dict(schema='atlas.xylo-audio2.temporal-plan.v1',roles=roles,sizes=sizes,parameters=params,
                weights=projections,input_raster=raster.astype(int).tolist(),steps=steps,dt_seconds=.001,
                source_hashes=model['protocol']['layers'],hardware_dimension_limits_passed=True,
                hardware_config_validation_required=True,physical_hardware=False,hardware_mapping_tested=False)


def lower_b2ir(model_path,*,runner):
    from .protocol import canonical_bytes,verify_protocol
    model=json.loads(Path(model_path).read_text());verify_protocol(model);encoded=canonical_bytes(model)
    with tempfile.TemporaryDirectory(prefix='atlas-xylo-validate-') as tmp:
        path=Path(tmp)/'source.json';path.write_bytes(encoded)
        result=subprocess.run([str(Path(runner).resolve()),'--validate',str(path)],capture_output=True,text=True,timeout=60)
    require(result.returncode==0,'Atlas validation failed: '+result.stderr[-2000:])
    plan=_lower_validated(model)
    plan['source_model_sha256']=hashlib.sha256(encoded).hexdigest()
    plan['atlas_validator_sha256']=hashlib.sha256(Path(runner).read_bytes()).hexdigest()
    return plan
