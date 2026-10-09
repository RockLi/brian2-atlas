"""Transitive NumPy selectors are snapshots, including their writeback locals."""
import copy
import brian2 as b
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer, lower_brian_dynamic_training
from test_native_training import RUNNER
from test_training_integer_ir import engine
from test_training_poisson_zero_vjp import mpi
from test_training_callback_effects import curve

CODES = {
    'scatter': 'peer=curve(peer)+gain;v=.8*v+.1*peer',
    'root_change': 'pick=(pick+1)%4;peer=curve(peer)+gain;v=.8*v+.1*peer',
    'intermediate_change': 'hop=(hop+1)%4;peer=curve(peer)+gain;v=.8*v+.1*peer',
    'repeat': 'peer+=gain*curve(v);peer+=.05*curve(v)',
    'repeat_route': 'peer+=gain*curve(v);route=(route+1)%4;peer+=.05*curve(v)',
    'repeat_root': 'peer+=gain*curve(v);pick=(pick+1)%4;peer+=.05*curve(v)',
}


def model(kind, depth, ranks, window, backend, boundary=False):
    from brian2.codegen.translation import make_statements
    from brian2.codegen.generators.numpy_generator import NumpyCodeGenerator
    from brian2.codegen.runtime.numpy_rt import NumpyCodeObject
    b.set_device('runtime'); b.start_scope(); b.prefs.codegen.target='numpy'
    dt=.2*b.ms
    inp=b.SpikeGeneratorGroup(1, [], np.array([])*b.second, dt=dt)
    a=b.NeuronGroup(1, 'dv/dt=0/second:1', threshold='v>100', reset='v=0', method='euler', dt=dt)
    g=b.NeuronGroup(4, 'dv/dt=0/second:1\npick:integer\nroute:integer\nlast:integer\ngain:1 (constant)',
                    threshold='v>.5', reset='v-=.5', method='euler', dt=dt)
    g.v=[.1,.4,.7,.2] if boundary else [.117,.413,.731,.229]
    g.pick=[1,1,0,2];g.route=[2,0,1,3];g.last=[1,3,0,2]
    g.gain=[.1,.2,.3,.4] if boundary else [.113,.217,.331,.419]
    g.variables.add_reference('hop',g,'route',index='pick')
    if depth==3:
        g.variables.add_reference('tip',g,'last',index='hop')
    if depth==4:
        g.variables.add_array('fixed_map',size=4,values=np.array([2,0,3,1],dtype=np.int64),
                              dtype=np.int64,constant=True,read_only=True,index='hop')
    g.variables.add_reference('peer',g,'v',index='fixed_map' if depth==4 else 'tip' if depth==3 else 'hop')
    g.namespace['curve']=b.Function(curve,arg_units=[1],arg_names=['x'],return_unit=1,stateless=False)
    runner=g.run_regularly(CODES[kind],when='start');net=b.Network(inp,a,g)
    bundle=lower_brian_dynamic_training(net,input_group=inp,layers=[a,g],backend=backend,mpi_ranks=ranks,
                                       tbptt_window=window,detach_reset=False,trainable_neuron_parameters={g.name:['gain']})
    variables={**b.core.functions.DEFAULT_FUNCTIONS,**g.variables,**g.namespace}
    # create_runner_codeobj inserts resolved identifiers in sorted order.
    # NumPy write_arrays preserves that set's iteration order for alias ties.
    variables={name:variables[name] for name in sorted(variables)}
    scalar,vector=make_statements(CODES[kind],variables,np.float64,optimise=True)
    gen=NumpyCodeGenerator(variables,g.variables.indices,runner,{'_idx'},NumpyCodeObject,runner.name,'stateupdate',allows_scalar_write=True)
    lines=gen.translate_one_statement_sequence(vector)
    assert not scalar
    return net,g,runner,dt,bundle,variables,gen,compile('\n'.join(lines),'<original nested NumPy generator>','exec')


def oracle(bundle,g,variables,gen,code,weights,initial=None,anchors=None,precision=np.float64):
    z=np.array(bundle.initial_state if initial is None else initial,float)
    layout=bundle.provenance['neuron_state_layout'][g.name];p=bundle.plan
    bank=next(r['bank'] for r in bundle.provenance['bindings'] if r['object']==g.name and r['variables']==['gain'])
    before=[];margins=[];hard=[];spikes=[]
    for tick in range(4):
        if anchors is not None and p['tbptt_window'] and tick and tick%p['tbptt_window']==0:z=anchors['before'][tick].copy()
        before.append(z.copy());env=dict(curve=curve,_numpy=np,_vectorisation_idx=np.arange(4),logical_not=np.logical_not)
        for name,var in variables.items():
            if not isinstance(var,b.core.variables.ArrayVariable):continue
            key=gen.get_array_name(var)
            if key in env:continue
            field=next((f for f in ['v','pick','route','last'] if var is g.variables[f]),None)
            value=z[layout[field]] if field else weights[bank] if var is g.variables['gain'] else var.get_value()
            env[key]=np.array(value,dtype=precision if np.dtype(var.dtype).kind=='f' else var.dtype)
        exec(code,env)
        for field in ['v','pick','route','last']:z[layout[field]]=env[gen.get_array_name(g.variables[field])]
        margin=z[layout['v']]-.5;event=(margin>0).astype(float)
        margins.append(margin.copy());hard.append(event.copy())
        if anchors is not None:
            old=anchors['margins'][tick]
            event=anchors['hard'][tick]+p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(old))**2*(margin-old)
        spikes.append(event.copy());z[layout['v']]-=.5*event
    logits=np.array(spikes).mean(0)*p['logit_scale']
    loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
    return loss,z,np.array(spikes),dict(before=before,margins=margins,hard=hard)


@pytest.mark.parametrize('kind',list(CODES))
@pytest.mark.parametrize('depth',[2,3,4])
@pytest.mark.parametrize('ranks',[None,2])
@pytest.mark.parametrize('window',[None,2])
def test_nested_regular_original_and_all_vjps(engine,kind,depth,ranks,window):
    mpi(ranks);net,g,runner,dt,bundle,variables,gen,code=model(kind,depth,ranks,window,engine)
    out=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).gradients(np.zeros((1,4,1)),[0])
    def ref(weights,initial=None,anchors=None):return oracle(bundle,g,variables,gen,code,weights,initial,anchors)
    loss,z,spikes,anchors=ref(bundle.weights);layout=bundle.provenance['neuron_state_layout'][g.name]
    for field in ['v','pick','route','last']:
        np.testing.assert_allclose(np.asarray(out['final_state'])[0,layout[field]],z[layout[field]],rtol=8e-5,atol=8e-6)
    np.testing.assert_array_equal(np.array(out['spikes'])[0,:,1:],spikes);assert out['loss']==pytest.approx(loss,abs=8e-6)
    for bank,row in enumerate(bundle.weights):
        for index in range(len(row)):
            hi=copy.deepcopy(bundle.weights);lo=copy.deepcopy(bundle.weights);hi[bank][index]+=1e-6;lo[bank][index]-=1e-6
            fd=(ref(hi,anchors=anchors)[0]-ref(lo,anchors=anchors)[0])/2e-6
            assert out['gradients'][bank][index]==pytest.approx(fd,rel=1e-3,abs=1e-5),(bank,index)
    for index in range(len(bundle.initial_state)):
        if bundle.plan['dynamic']['detached'][index] or index in bundle.plan['dynamic']['integer_states']:
            assert out['initial_state_gradients'][0][index]==0.;continue
        hi=np.array(bundle.initial_state);lo=hi.copy();hi[index]+=1e-6;lo[index]-=1e-6
        fd=(ref(bundle.weights,hi,anchors)[0]-ref(bundle.weights,lo,anchors)[0])/2e-6
        assert out['initial_state_gradients'][0][index]==pytest.approx(fd,rel=1e-3,abs=1e-5),index
    net.run(4*dt,namespace={})
    for field in ['v','pick','route','last']:
        np.testing.assert_allclose(np.asarray(out['final_state'])[0,layout[field]],g.variables[field].get_value(),rtol=8e-5,atol=8e-6)
    assert (out['gpu_dispatches']>0)==(engine!='cpu')


@pytest.mark.parametrize('kind',list(CODES))
@pytest.mark.parametrize('depth',[2,3,4])
@pytest.mark.parametrize('ranks',[None,2])
def test_nested_regular_carry_and_restore(engine,kind,depth,ranks,tmp_path):
    mpi(ranks);_,_,_,_,bundle,_,_,_=model(kind,depth,ranks,None,engine)
    p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(p['trainable']);x=np.zeros((1,4,1))
    whole=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER).evaluate(x,[0])
    t=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER);first=t.step(x[:,:2],[0]);path=tmp_path/'nested';t.store(path)
    t=NativeLIFTrainer(p,runner=RUNNER);t.restore(path);last=t.step(x[:,2:],[0],initial='carry')
    np.testing.assert_allclose(last['final_state'],whole['final_state'],rtol=8e-5,atol=8e-6)
    np.testing.assert_array_equal(np.concatenate([first['spikes'],last['spikes']],axis=1),whole['spikes'])
    assert t.clock_tick==4 and (last['gpu_dispatches']>0)==(engine!='cpu')


@pytest.mark.parametrize('depth,field',[(2,'pick'),(2,'route'),(3,'pick'),(3,'route'),(3,'last'),(4,'pick'),(4,'route')])
@pytest.mark.parametrize('ranks',[None,2])
def test_nested_regular_invalid_selector_is_atomic(engine,depth,field,ranks):
    mpi(ranks);net,g,runner,dt,_,_,_,_=model('repeat',depth,ranks,None,engine)
    runner.abstract_code='peer+=gain*curve(v);'+field+'='+field+'+4;peer+=.05*curve(v)'
    inp=next(o for o in net.objects if isinstance(o,b.SpikeGeneratorGroup))
    hidden=next(o for o in net.objects if isinstance(o,b.NeuronGroup) and len(o)==1)
    bundle=lower_brian_dynamic_training(net,input_group=inp,layers=[hidden,g],backend=engine,mpi_ranks=ranks)
    trainer=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER);before=copy.deepcopy(trainer.state)
    with pytest.raises(ValueError,match='[Ii]ndex|outside|domain|nonfinite dynamic GPU result|nonfinite or invalid dynamic (?:Metal|CUDA)(?: MPI)? action'):
        trainer.step(np.zeros((1,1,1)),[0])
    assert trainer.state==before and trainer.neuron_state is None and trainer.clock_tick==0 and trainer.next_noise_sequence==0
    with pytest.raises(b.core.base.BrianObjectException) as caught:net.run(dt,namespace={})
    assert isinstance(caught.value.__cause__,IndexError)


@pytest.mark.parametrize('ranks',[None,2])
def test_nested_threshold_boundary_follows_backend_arithmetic(engine,ranks):
    """Retain the former exact-threshold case with an independent typed oracle."""
    mpi(ranks);net,g,runner,dt,bundle,variables,gen,code=model('intermediate_change',2,ranks,None,engine,boundary=True)
    out=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).evaluate(np.zeros((1,4,1)),[0])
    precision=np.float64 if engine=='cpu' else np.float32
    _,z,spikes,_=oracle(bundle,g,variables,gen,code,bundle.weights,precision=precision)
    layout=bundle.provenance['neuron_state_layout'][g.name]
    for field in ['v','pick','route','last']:
        np.testing.assert_allclose(np.asarray(out['final_state'])[0,layout[field]],z[layout[field]],rtol=8e-5,atol=8e-6)
    np.testing.assert_array_equal(np.asarray(out['spikes'])[0,:,1:],spikes)
    if engine=='cpu':
        net.run(4*dt,namespace={});np.testing.assert_allclose(z[layout['v']],g.v[:],rtol=8e-5,atol=8e-6)
    assert (out['gpu_dispatches']>0)==(engine!='cpu')
