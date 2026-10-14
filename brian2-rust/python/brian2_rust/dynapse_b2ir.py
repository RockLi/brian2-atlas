"""Strict, fingerprinted admission of the explicit DynapSim circuit subset."""
import copy,hashlib,json,struct,subprocess,tempfile
from pathlib import Path
import numpy as np

from .protocol import SUPPORTED_V1_SCHEMAS
from .dynapse_model import PARAMETERS,STATES

PROFILE='dynapse-dpi-v1'
DEFINITION_SHA256='e6e5e9549f5b1b2ccd9ef484ad032d2b296cdddbe9c09229833a8951370ec942'


def require(ok,message):
    if not ok:raise ValueError('DynapSim: '+message)


def number(value):return struct.unpack('>d',bytes.fromhex(value))[0]


def tables(value):
    if isinstance(value,dict):
        if value.get('op')=='timed_array':yield value
        else:
            for v in value.values():yield from tables(v)
    elif isinstance(value,list):
        for v in value:yield from tables(v)


def signature(definition):
    from .protocol import canonical_bytes
    d=copy.deepcopy(definition);p=d['populations'][0]
    require(len(p.get('state_monitors',[]))==1,'requires one state monitor')
    p['count']=0;p['steps']=0;p['monitor']['window_steps']=0;p['monitor']['record']=[]
    p['state_monitors'][0]['record']=[]
    for table in tables(d):table.update(rows=0,columns=0,values=[])
    return hashlib.sha256(canonical_bytes(d)).hexdigest()


def _lower_validated(model):
    d,inst,run=model['definition'],model['instance'],model['run']
    require(model['schema'] in SUPPORTED_V1_SCHEMAS and len(d['populations'])==len(inst['populations'])==1,'requires one DPI population')
    p=d['populations'][0];state=inst['populations'][0];n=p['count'];steps=p['steps']
    require(1<=n<=8 and 1<=steps<=512,'pilot admits 1..8 neurons and 1..512 steps')
    require(signature(d)==DEFINITION_SHA256,'unsupported circuit AST, schedule, monitoring or feature')
    require(run['clocks']==[dict(start_tick=0,steps=steps)] and number(run['start'])==0
            and abs(number(run['duration'])-steps*.001)<1e-12,'requires a fresh 1 ms run')
    require(inst['neuron_count']==n and not inst['synapses'],'no synapses or recurrence in this pilot')
    require(p['monitor']['record']==list(range(n)) and p['state_monitors'][0]['record']==list(range(n))
            and p['monitor']['window_steps']==steps,'requires all neuron traces')
    require(state['refractory'] is None and state['spike_generator'] is None,'no external refractory or spike generator')
    require(set(state['initial_state'])==set(STATES),'unexpected initial states')
    for key,values in state['initial_state'].items():
        expected=PARAMETERS['Io'] if key in ('iahp','imem','isyn') else 0
        require(len(values)==n and all(number(v)==expected for v in values),'requires default initial state')
    params={k:np.array([number(v) for v in values]) for k,values in state['parameters'].items()}
    require(set(params)==set(PARAMETERS)|{'weight'},'unexpected parameters')
    allowed=dict(weight=[0,.5,1,2,4],Idc=[0,2.**-40,2.**-30,2.**-28],
                 Ispkthr=[2.**-26,2.**-25,2.**-24],Iw_ahp=[2.**-40,2.**-34,2.**-32,2.**-24],Iref=[2.**-31,2.**-30,2.**-29])
    for key,values in params.items():
        require(len(values)==n and np.isfinite(values).all() and np.isin(values,allowed.get(key,[PARAMETERS.get(key)])).all(),
                'parameter outside pilot domain: '+key)
    nodes=list(tables(d));require(nodes,'missing input raster');table=nodes[0]
    require(all(t==table for t in nodes),'inconsistent input rasters')
    require(table['rows']==steps and table['columns']==n and len(table['values'])==steps*n,'raster dimensions')
    raster=np.array([number(v) for v in table['values']]).reshape(steps,n)
    require(np.isin(raster,[0,1]).all(),'requires binary spike raster')
    return dict(schema='atlas.dynapse.dpi-plan.v1',steps=steps,neurons=n,dt_seconds=.001,
                parameters={k:v.tolist() for k,v in params.items()},input_raster=raster.astype(int).tolist(),
                source_hashes=model['protocol']['layers'],physical_hardware=False,hardware_mapping_tested=False,
                mismatch_enabled=False,recurrence=False,precision='stock JAX float32; Atlas reference-f64')


def lower_b2ir(path,*,runner):
    from .protocol import canonical_bytes,verify_protocol
    model=json.loads(Path(path).read_text());verify_protocol(model);encoded=canonical_bytes(model)
    with tempfile.TemporaryDirectory(prefix='atlas-dynapse-validate-') as tmp:
        source=Path(tmp)/'source.json';source.write_bytes(encoded)
        result=subprocess.run([str(Path(runner).resolve()),'--validate',str(source)],capture_output=True,text=True,timeout=60)
    require(result.returncode==0,'native validation failed: '+result.stderr[-1500:])
    plan=_lower_validated(model);plan['source_model_sha256']=hashlib.sha256(encoded).hexdigest()
    plan['atlas_validator_sha256']=hashlib.sha256(Path(runner).read_bytes()).hexdigest()
    return plan
