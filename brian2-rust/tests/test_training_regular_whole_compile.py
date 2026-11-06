"""Fixed-column rebinding, broadcasting and scalar extraction against NumPy."""
import numpy as np
import pytest
from brian2_rust.training_regular_whole import compile_whole_regular,cached_poisson_code


def evaluate(program,context):
    values=[]
    for node in program:
        op=node['op']
        if op in ('state','integer_state'):value=context[node['index']]
        elif op in ('constant','integer_constant'):value=node['value']
        elif op in ('sequence','integer_sequence'):value=values[node['right']]
        elif op in ('float_cast','integer_cast','integer_float'):value=(int if op=='integer_cast' else float)(values[node['arg']])
        elif op in ('add','sub','mul','div','pow'):value={'add':lambda a,c:a+c,'sub':lambda a,c:a-c,'mul':lambda a,c:a*c,'div':lambda a,c:a/c,'pow':lambda a,c:a**c}[op](values[node['left']],values[node['right']])
        else:raise AssertionError(op)
        values.append(value)
    return values[-1]


@pytest.mark.parametrize('code',['u=u+.1*s\nv+=.01*n','alias=u\nu=u+.1*s\nalias*=.8','tmp=u+.1*s\nu=tmp+tmp','u+=.1*s\nu*=.8'])
def test_whole_regular_array_lifetimes(code):
    z=np.array([.173,.219,.59,.38,.217,.2,.3]);bindings={name:dict(values=values,dtype='float',array=array,writable=True) for name,values,array in [('u',[0,1],True),('v',[2,3],True),('s',[4],False),('n',[5,6],True)]}
    scope=dict(u=z[:2].copy(),v=z[2:4].copy(),s=z[4],n=z[5:7].copy());exec(code,scope)
    for column in range(2):
        out=compile_whole_regular(code,bindings,{},column=column);context=z[out['reads']]
        for name in ('u','v'):
            if name not in out['names']:continue
            slot=out['names'][name]
            if slot in out['programs']:assert evaluate(out['programs'][slot],context)==pytest.approx(scope[name][column])


def test_whole_regular_integer_parameter_is_not_state_address():
    out=compile_whole_regular('u=u+number',dict(u=dict(values=[0,1],dtype='float',array=True,writable=True),number=dict(values=[('parameter',2)],dtype='integer',array=False)),{},column=1)
    assert out['reads']==[0,1];assert evaluate(out['programs'][out['names']['u']],[.173,.219])==pytest.approx(2.219)


def test_cached_poisson_prefix_order():
    code='a=p0(rate)\nb=p1(a+.1)\nu+=a+b';records=[dict(name='p0',cache_name='c0'),dict(name='p1',cache_name='c1')]
    for row in records:row['kind']='poisson'
    result,prefixes=cached_poisson_code(code,records)
    assert len(prefixes)==2
    assert 'c0' not in prefixes[0][1] and 'c0' in prefixes[1][1] and 'c1' not in prefixes[1][1]
    assert result.count('_b2_scheduled_poisson')==2
