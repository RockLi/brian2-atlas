"""Portable floating expressions, ordered clip and observable numeric faults."""
from pathlib import Path
import hashlib,json,subprocess
from copy import deepcopy
import brian2 as b
import numpy as np
import pytest
from brian2_rust.export import lower_network
from brian2_rust.spec import bits
from brian2_rust.protocol import attach_protocol
from brian2_rust.results import load_results
from brian2_rust.metal import build_metal_plan
from test_gpu_spike_generator import BACKENDS,execute
from test_gpu_monitors import setup,DT
from test_gpu_refractory import refresh_code
from test_metal_delays import device,ROOT
from test_metal_plasticity import equivalent


def lit(x):return dict(op='literal',bits=bits(x))
def load(n):return dict(op='load',name=n)
def binary(op,a,b):return dict(op=op,left=a,right=b)
def unary(op,x):return dict(op=op,arg=x)
def statement(name,value,dtype='f64',condition=None):return dict(target=name,dtype=dtype,dimensions=[0.]*7,condition=condition,value=value)

def oracle(model,path,success=True):
    path.mkdir();p=path/'model.json';p.write_text(json.dumps(model))
    r=subprocess.run([str(ROOT/'target/release/b2-runner'),str(p),str(path/'result')],capture_output=True,text=True)
    if success:
        assert r.returncode==0,r.stderr
        return load_results(model,path/'result')
    assert r.returncode!=0 and 'non-finite' in r.stderr,r.stderr


def portable(model,name,body):
    model['definition']['functions'].append(dict(name=name,semantic_version='1.0.0',abi='b2ir-function-v1',
        effects=dict(stateful=False,deterministic=True,thread_safe=True,rng=False),
        implementations={'b2ir-expression-v1':hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':')).encode()).hexdigest()},
        backend_implementations={},arguments=[dict(name='arg',dtype='f64',dimensions=[0.]*7)],return_dtype='f64',return_dimensions=[0.]*7,body=body))


def base(path,count=5,dag=False):
    setup(path);p=b.NeuronGroup(count,'x:1\ny:1\nlo:1\nhi:1\nmask:boolean',threshold='False',reset='',dt=DT,name='population')
    p.x=np.linspace(-1,1,count);p.lo=1;p.hi=-1;p.run_regularly('y=x',name='update')
    objects=[p]
    if dag:
        s=b.Synapses(p,p,'w:1',on_pre='y_post+=w',name='connection',clock=p.clock);s.connect(i=np.array([],dtype=np.int32),j=np.array([],dtype=np.int32));objects.append(s)
    m=lower_network(b.Network(*objects),DT)
    c=next(c for c in m['definition']['populations'][0]['code_objects'] if c['kind']=='run_regularly')
    return m,c


@pytest.mark.parametrize('backend',BACKENDS)
@pytest.mark.parametrize('dag',[False,True])
def test_reversed_equal_and_regular_clip_bounds(device,tmp_path,backend,dag):
    m,c=base(tmp_path/'ref',dag=dag)
    m['instance']['populations'][0]['initial_state']['lo']=[bits(x) for x in (1,1,0,-1,-1)]
    m['instance']['populations'][0]['initial_state']['hi']=[bits(x) for x in (-1,-1,0,1,1)]
    c['vector']=[statement('y',dict(op='clip',value=load('x'),min=load('lo'),max=load('hi')))]
    c['effects']['reads']=['x','lo','hi'];refresh_code(m,c)
    expected=oracle(m,tmp_path/'oracle');np.testing.assert_array_equal(expected['populations'][0]['states']['y'],[-1,-1,0,.5,1])
    equivalent(execute(m,tmp_path/backend,backend),expected,exact=True)


@pytest.mark.parametrize('backend',BACKENDS)
@pytest.mark.parametrize('kind',['hidden-nan','overwritten-inf','eager-and','ignored-function-argument','nested-invalid-exp'])
def test_intermediate_faults_cannot_be_hidden(device,tmp_path,backend,kind):
    m,c=base(tmp_path/'ref')
    bad=unary('sqrt',lit(-1))
    if kind=='hidden-nan':rhs=dict(op='clip',value=bad,min=lit(0),max=lit(1))
    elif kind=='overwritten-inf':rhs=binary('div',lit(1),lit(0))
    elif kind=='nested-invalid-exp':rhs=unary('exp',bad)
    elif kind=='eager-and':rhs=dict(op='bool_to_f64',arg=binary('and',dict(op='boolean',value=False),binary('gt',bad,lit(0))))
    else:
        portable(m,'ignore_argument',lit(0));rhs=dict(op='call',function='ignore_argument',arguments=[bad])
    c['vector']=[statement('y',rhs),statement('y',lit(0))];refresh_code(m,c)
    oracle(m,tmp_path/'oracle',success=False)
    with pytest.raises(FloatingPointError):execute(m,tmp_path/backend,backend)
    assert not (tmp_path/backend/'transport/summary.json').exists()


@pytest.mark.parametrize('backend',BACKENDS)
def test_f32_overflow_has_explicit_failure_even_if_f64_is_finite(device,tmp_path,backend):
    m,c=base(tmp_path/'ref');c['vector']=[statement('y',unary('exp',lit(90))),statement('y',lit(0))];refresh_code(m,c)
    expected=oracle(m,tmp_path/'oracle');np.testing.assert_array_equal(expected['populations'][0]['states']['y'],0)
    with pytest.raises(FloatingPointError):execute(m,tmp_path/backend,backend)


@pytest.mark.parametrize('backend',BACKENDS)
def test_false_statement_mask_suppresses_full_rhs_fault(device,tmp_path,backend):
    from test_gpu_refractory import make_model
    m=make_model(device,tmp_path,'masked')
    c=next(c for c in m['definition']['populations'][0]['code_objects'] if c['kind']=='state_update')
    stmt=next(s for s in c['vector'] if s['target']=='v')
    assert stmt['condition']=='not_refractory'
    stmt['value']=unary('sqrt',lit(-1));refresh_code(m,c)
    equivalent(execute(m,tmp_path/backend,backend),oracle(m,tmp_path/'oracle'),exact=True)


@pytest.mark.parametrize('backend',BACKENDS)
@pytest.mark.parametrize('where',['static-pathway','mutable-pathway','quiet-scalar','summed-overflow','static-summed-overflow'])
def test_synaptic_intermediate_and_accumulation_faults(device,tmp_path,backend,where):
    setup(tmp_path/'ref');p=b.NeuronGroup(3,'v:1\nu:1',threshold='v>0.5',reset='',dt=DT);p.v=0 if where=='quiet-scalar' else 1
    equations='w:1\nu_post=w:1 (summed)' if 'summed-overflow' in where else 'w:1'
    body='w+=0.125' if where in {'mutable-pathway','summed-overflow'} else 'u_post+=1'
    syn=b.Synapses(p,p,equations,on_pre=body,clock=p.clock);syn.connect(i=[0,1,2],j=[0,0,0]);syn.w=.5
    m=lower_network(b.Network(p,syn),DT);s=m['definition']['synapses'][0]
    if 'summed-overflow' in where:
        m['instance']['synapses'][0]['initial_state']['w']=[bits(float(np.float32(3e38)))]*3
        # f64 sum is finite; native f32 accumulation must fail even if later reset.
        code=next(c for c in m['definition']['populations'][0]['code_objects'] if c['kind']=='reset')
        code['vector'].append(statement('u',lit(0)));code['effects']['writes'].append('u');refresh_code(m,code)
        oracle(m,tmp_path/'oracle')
    else:
        code=next(c for c in s['code_objects'] if c['kind']=='synapses')
        code['scalar' if where=='quiet-scalar' else 'vector'].insert(0,statement('_discarded',unary('sqrt',lit(-1))))
        refresh_code(m,code);oracle(m,tmp_path/'oracle',success=False)
        if where=='static-pathway':assert any(x.role=='target-delivery' for x in build_metal_plan(m,numeric_mode='float32').dispatches)
    with pytest.raises(FloatingPointError):execute(m,tmp_path/backend,backend)


@pytest.mark.parametrize('backend',BACKENDS)
@pytest.mark.parametrize('family',[0,1])
def test_portable_float_operator_grid_and_nested_call(device,tmp_path,backend,family):
    setup(tmp_path/'ref');x=load('x');q=binary('add',lit(1),x)
    expressions={op:unary(op,x if op in {'abs','arccos','arcsin','arctan','ceil','cos','cosh','exp','expm1','exprel','floor','trunc','sign','sin','sinh','tan','tanh'} else q)
        for op in ('abs','arccos','arcsin','arctan','ceil','cos','cosh','exp','expm1','exprel','floor','trunc','log','log10','log1p','sign','sin','sinh','sqrt','tan','tanh')}
    expressions.update({op:binary(op,x,lit(.3125)) for op in ('add','sub','mul','div','mod','floor_div')})
    expressions.update(power=binary('pow',q,x),integer_power=binary('pow',q,lit(-3)),exp_power=binary('pow',unary('exp',x),lit(.25)),negative=unary('neg',x))
    expressions=dict(list(expressions.items())[family*16:(family+1)*16])
    values=np.asarray(np.concatenate((np.linspace(-.875,.875,257),[-1e-6,0,1e-6])),np.float32)
    names={name:'out'+str(i) for i,name in enumerate(expressions)}
    pop=b.NeuronGroup(len(values),'x:1\n'+'\n'.join(n+':1' for n in names.values())+'\ncalled:1',dt=DT);pop.x=values
    pop.run_regularly('; '.join(n+'=x' for n in [*names.values(),'called']))
    m=lower_network(b.Network(pop),DT);code=next(c for c in m['definition']['populations'][0]['code_objects'] if c['kind']=='run_regularly')
    portable(m,'a_square',binary('mul',load('arg'),load('arg')))
    portable(m,'z_nested',binary('add',load('arg'),lit(.5)))
    m['definition']['functions'].sort(key=lambda f:f['name'])
    code['vector']=[statement(names[n],v) for n,v in expressions.items()]+[statement('called',dict(op='call',function='z_nested',arguments=[dict(op='call',function='a_square',arguments=[x])]))]
    refresh_code(m,code)
    actual=execute(m,tmp_path/backend,backend);expected=oracle(m,tmp_path/'oracle')
    arrays={'input':values}
    for name in [*names.values(),'called']:
        a=actual['populations'][0]['states'][name];e=expected['populations'][0]['states'][name]
        np.testing.assert_allclose(a,e,rtol=2e-5,atol=2e-6,err_msg=name)
        arrays['actual/'+name]=a;arrays['reference/'+name]=e
    np.savez_compressed(tmp_path/'expression-grid.npz',**arrays)
    (tmp_path/'expression-grid.json').write_text(json.dumps(dict(backend=backend,family=family,names=names,rtol=2e-5,atol=2e-6,
        model_sha256=hashlib.sha256(json.dumps(m,sort_keys=True).encode()).hexdigest(),arrays={k:dict(dtype=v.dtype.str,shape=list(v.shape),sha256=hashlib.sha256(v.tobytes()).hexdigest()) for k,v in arrays.items()}),indent=2)+'\n')
    (tmp_path/'expression-grid-model.json').write_text(json.dumps(m,sort_keys=True)+'\n')
