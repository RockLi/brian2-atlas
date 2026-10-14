"""Signed division, Cython remainder semantics and native piecewise VJPs."""
import copy
import os
import numpy as np
import pytest
import brian2 as b
from brian2_rust import NativeLIFTrainer, lower_brian_dynamic_training
from brian2_rust.training_dynamic import compile_dynamic_transform
from brian2_rust.training_equations import NeuronParameter, StateSlot
from test_native_training import RUNNER
from test_training_integer_ir import engine, model, wrap
from test_training_linked import cython_cache


@pytest.mark.parametrize('kind',['floor_div','mod'])
def test_signed_integer_division_full_range(engine,kind):
    rng=np.random.default_rng(146)
    pairs=[(-2147483648,-1),(-2147483648,1),(2147483647,-2147483648),(-2147483647,2147483647),
           (-7,3),(7,-3),(-7,-3),(7,3),(0,-1),(2147483646,2147483647),(-2147483648,2),(1,-2147483648)]
    pairs+=list(zip(rng.integers(-2**31,2**31,size=20).tolist(),rng.integers(-2**31,2**31,size=20).tolist()))
    for start in range(0,len(pairs),4):
        part=pairs[start:start+4];p,w,x=model(kind,engine=engine)
        p['dynamic']['initial'][4:8]=[a for a,_ in part];w[1]=[d for _,d in part]
        result=NativeLIFTrainer(p,runner=RUNNER,weights=w).gradients(x[:,:1],[0])
        expected=[wrap(a//d if kind=='floor_div' else a%d) for a,d in part]
        np.testing.assert_array_equal(result['final_state'][0][4:8],expected)
        np.testing.assert_array_equal(result['gradients'][1],0)


@pytest.mark.parametrize('kind',['floor_div','mod'])
@pytest.mark.parametrize('ranks',[None,2,8])
@pytest.mark.parametrize('selected',[False,True])
@pytest.mark.parametrize('dtype',['integer','float'])
def test_divide_by_zero_is_lazy_and_atomic(engine,kind,ranks,selected,dtype):
    if ranks and os.environ.get('B2_TEST_MPI')!='1':pytest.skip('local MPI required')
    p,w,x=model(engine=engine,ranks=ranks)
    p['dynamic']['program_sets'][0][0]=[
        dict(op='constant',value=float(selected)),dict(op='integer_constant',value=1),dict(op='integer_constant',value=0),
        dict(op='integer_binary',kind=kind,left=1,right=2),dict(op='integer_constant',value=-1),
        dict(op='integer_select',condition=0,yes=3,no=4)]
    if dtype=='float':
        p['dynamic']['program_sets'][0][0]=[
            dict(op='constant',value=float(selected)),dict(op='constant',value=1.),dict(op='constant',value=0.),
            dict(op='floor_div' if kind=='floor_div' else 'modulo',left=1,right=2),dict(op='constant',value=-1.),
            dict(op='select',condition=0,yes=3,no=4),dict(op='integer_cast',arg=5)]
    trainer=NativeLIFTrainer(p,runner=RUNNER,weights=w);before=copy.deepcopy(trainer.state)
    if selected:
        with pytest.raises((ValueError,RuntimeError)):trainer.step(x[:,:1],[0])
        assert trainer.state==before and trainer.neuron_state is None
    else:
        result=trainer.gradients(x[:,:1],[0]);assert result['final_state'][0][4]==-1


@pytest.mark.parametrize('kind',['floor_div','modulo'])
@pytest.mark.parametrize('ranks',[None,2,8])
@pytest.mark.parametrize('syntax',['ssa','expression','augmented'])
def test_float_division_piecewise_vjp(engine,kind,ranks,syntax):
    if ranks and os.environ.get('B2_TEST_MPI')!='1':pytest.skip('local MPI required')
    p,w,x=model(engine=engine,ranks=ranks);a=np.array([1.3,-1.3,7.1,-7.1]);d=np.array([.7,.7,-3.,-3.])
    w[0]=a.tolist();p['dynamic']['initial'][:4]=d.tolist()
    p['dynamic']['program_sets']=[[[dict(op='neuron_parameter',bank=0,index=0),dict(op='state',index=0),dict(op=kind,left=0,right=1)]]]
    if syntax!='ssa':
        code=('v=floor(a/v)' if kind=='floor_div' else 'v=a%v') if syntax=='expression' else ('v=a\nv//=den' if kind=='floor_div' else 'v=a\nv%=den')
        p['dynamic']['program_sets']=[compile_dynamic_transform(code,states={'v':0},parameters={'a':NeuronParameter(0),'den':StateSlot(0)},state_types={0:'float'})['programs']]
    p['dynamic']['actions']=[dict(owner=j,reads=[j],writes=[j],program_set=0,threshold=None,trigger=None,parameter_index=j) for j in range(4)]
    p['dynamic']['actions'] += [dict(owner=j,reads=[j],writes=[],program_set=None,threshold=j,trigger=None) for j in range(4)]
    result=NativeLIFTrainer(p,runner=RUNNER,weights=w).gradients(x[:,:1],[0])
    value=np.floor(a/d) if kind=='floor_div' else a%d
    np.testing.assert_allclose(result['final_state'][0][:4],value,rtol=3e-6,atol=3e-6)
    anchors=value-.5;hard=(anchors>0).astype(float)
    def loss(aa,dd):
        vv=np.floor(aa/dd) if kind=='floor_div' else aa%dd
        spikes=hard+p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(anchors))**2*(vv-value)
        z=spikes[2:]*p['logit_scale'];return np.log(np.exp(z-z.max()).sum())+z.max()-z[0]
    for operand,key in [('a','gradients'),('d','initial_state_gradients')]:
        expected=[]
        for j in range(4):
            hi=(a if operand=='a' else d).copy();lo=hi.copy();hi[j]+=1e-6;lo[j]-=1e-6
            expected.append(((loss(hi,d)-loss(lo,d)) if operand=='a' else (loss(a,hi)-loss(a,lo)))/2e-6)
        np.testing.assert_allclose(result[key][0][:4],expected,rtol=4e-4,atol=4e-6)


def brian_model(kind='integer',noisy=False,**options):
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='cython';dt=.1*b.ms
    inp=b.SpikeGeneratorGroup(1,[0,0],[0,2]*dt,dt=dt)
    h=b.NeuronGroup(1,'dv/dt=0/second:1',threshold='v>1',reset='v=0',method='euler',dt=dt)
    typ='integer' if kind=='integer' else '1'
    eq='dv/dt=(.1+((t/ms)%1.3))/ms'+('+.01*(1+int(count%2))*xi/sqrt(ms)' if noisy else '')+':1\n'
    g=b.NeuronGroup(5,eq+f'a:{typ}\nden:{typ}\nq:{typ}\nr:{typ}\ncount:integer',threshold='v>-.5',
                    reset='q=a//den\nr=a%den\nq//=2\nr%=den\ncount+=1',method='euler',dt=dt)
    aa=[2147483646,-2147483647,-7,7,-2147483648] if kind=='integer' else [1.,1e-20,-1e-20,-7.1,7.1]
    dd=[2147483647,-2147483648,3,-3,2] if kind=='integer' else [.1,1.,1.,3.,-3.]
    g.a=aa;g.den=dd
    s=b.Synapses(inp,g,'w:1\nquota:integer\nrest:integer',on_pre='quota//=2\nrest=quota%3\nrest%=2\nv_post+=w*(1+rest)',dt=dt)
    s.connect();s.quota=[-7,7,-8,8,-1];s.w=[.1,.12,.13,.14,.15]
    net=b.Network(inp,h,g,s);bundle=lower_brian_dynamic_training(net,input_group=inp,layers=[h,g],**options)
    return net,g,s,bundle,np.array([[[1],[0],[1],[0]]],float)


@pytest.mark.parametrize('kind',['integer','float'])
def test_frontend_sequential_division_matches_actual_cython(kind):
    net,g,s,bundle,x=brian_model(kind)
    result=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights).evaluate(x,[0]);net.run(.4*b.ms)
    assert g.resetter['spike'].codeobj.compiled_code['run'] is not None
    assert s.pre.codeobj.compiled_code['run'] is not None
    for obj,layout in [(g,'neuron_state_layout'),(s,'dynamic_state_layout')]:
        for name,indices in bundle.provenance[layout][obj.name].items():
            np.testing.assert_allclose(np.array(result['final_state'])[0,indices],np.asarray(getattr(obj,name)[:]),rtol=1e-13,atol=1e-15)


@pytest.mark.parametrize('window',[None,2])
@pytest.mark.parametrize('ranks',[None,2,8])
def test_stochastic_division_events_gradient_and_checkpoint(engine,window,ranks,tmp_path):
    if ranks and os.environ.get('B2_TEST_MPI')!='1':pytest.skip('local MPI required')
    _,_,_,bundle,x=brian_model(noisy=True,backend=engine,mpi_ranks=ranks,tbptt_window=window)
    p=copy.deepcopy(bundle.plan);p['backend']='cpu';p['mpi_ranks']=None
    ref=NativeLIFTrainer(p,runner=RUNNER,weights=bundle.weights).gradients(x,[0],noise_sequence=7)
    trainer=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights)
    actual=trainer.gradients(x,[0],noise_sequence=7)
    for key in ('final_state','spikes','initial_state_gradients','logits'):
        np.testing.assert_allclose(actual[key],ref[key],rtol=4e-5,atol=4e-6)
    for a,c in zip(actual['gradients'],ref['gradients']):np.testing.assert_allclose(a,c,rtol=5e-4,atol=5e-6)
    bundle.plan['trainable']=[False]*len(bundle.weights)
    trainer=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights);trainer.step(x[:,:2],[0],noise_sequence=7)
    checkpoint=tmp_path/'division.json';trainer.store(checkpoint)
    restored=NativeLIFTrainer(bundle.plan,runner=RUNNER);restored.restore(checkpoint)
    tail=restored.evaluate(x[:,2:],[0],initial='carry')
    np.testing.assert_allclose(tail['final_state'],ref['final_state'],rtol=4e-5,atol=4e-6)


def test_augmented_division_units_and_rhs_grouping():
    from brian2.equations.unitcheck import check_units_statements
    from brian2.core.variables import Constant
    variables={name:Constant(name,float(value),dimensions=b.get_dimensions(value)) for name,value in {'x':3.,'v':3*b.volt,'w':2*b.volt,'t':2*b.second}.items()}
    for code in ('x//=2', 'x//=1+2', 'v%=w', 'v*=w/w+1'):
        check_units_statements(code,variables)
    for code in ('v//=w','x//=t'):
        with pytest.raises(SyntaxError,match='dimensionless'):check_units_statements(code,variables)
    for code in ('v%=t','v*=w+1'):
        with pytest.raises(b.DimensionMismatchError):check_units_statements(code,variables)
