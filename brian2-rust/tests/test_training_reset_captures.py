"""Whole closure writes run independently of the reset event selection."""
import copy
import brian2 as b
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer, lower_brian_dynamic_training
from test_training_integer_ir import engine
from test_training_poisson_zero_vjp import mpi
from test_native_training import RUNNER


def reset_capture(array):
    def f(x):
        captured = array
        captured *= .8
        x *= .7
        return x
    return f


def model(selection,cross,ranks,window,backend,capture_voltage=False):
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='numpy'
    dt=.2*b.ms
    source=b.SpikeGeneratorGroup(1,[],np.array([])*b.second,dt=dt)
    hidden=b.NeuronGroup(2,'dv/dt=0/second:1\ngain:1 (constant)',threshold='v>100',reset='v=0',method='euler',dt=dt)
    g=b.NeuronGroup(2,'dv/dt=gain/ms:1\ngain:1 (constant)',threshold='v>.5',reset='v=f(v)',method='euler',dt=dt)
    g.v={'empty':[.02,.07],'partial':[.02,.73],'full':[.71,.83]}[selection]
    g.gain=[.09,.17];hidden.gain=[.11,.19]
    physical=hidden if cross else g;array=physical.variables['v' if capture_voltage else 'gain'].get_value()
    g.namespace['f']=b.Function(reset_capture(array),arg_units=[1],arg_names=['x'],return_unit=1,stateless=False)
    net=b.Network(source,hidden,g)
    bundle=lower_brian_dynamic_training(net,input_group=source,layers=[hidden,g],backend=backend,mpi_ranks=ranks,tbptt_window=window,detach_reset=False,
        trainable_neuron_parameters={g.name:['gain'],**({hidden.name:['gain']} if cross else {})})
    return net,g,physical,dt,bundle


@pytest.mark.parametrize('selection',['empty','partial','full'])
@pytest.mark.parametrize('cross',[False,True])
@pytest.mark.parametrize('ranks',[None,2])
def test_reset_capture_original_restore(engine,selection,cross,ranks,tmp_path):
    mpi(ranks);net,g,physical,dt,bundle=model(selection,cross,ranks,None,engine)
    p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights)
    t=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER)
    for tick in range(4):
        out=t.step(np.zeros((1,1,1)),[0],initial='carry' if tick else None)
        net.run(dt,namespace={})
        for obj,name in [(g,'v'),(physical,'gain')]:
            cells=bundle.provenance['neuron_state_layout'][obj.name][name]
            np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],obj.variables[name].get_value(),rtol=8e-5,atol=8e-6)
        path=tmp_path/str(tick);t.store(path);t=NativeLIFTrainer(p,runner=RUNNER);t.restore(path)
    assert (out['gpu_dispatches']>0)==(engine!='cpu')


def reference(data,weights,initial=None,anchors=None):
    _,g,physical,_,bundle=data;p=bundle.plan
    z=np.array(bundle.initial_state if initial is None else initial,float)
    if initial is None:
        for i,ref in enumerate(p['dynamic']['initial_parameters']):
            if ref is not None:z[i]=weights[ref[0]][ref[1]]
    layout=bundle.provenance['neuron_state_layout'];v=layout[g.name]['v'];gain=layout[physical.name]['gain']
    g_gain=layout[g.name].get('gain');bank=next(row['bank'] for row in bundle.provenance['bindings'] if row['object']==g.name and 'gain' in row['variables'])
    before=[];margins=[];hard=[];spikes=[]
    for tick in range(4):
        if anchors is not None and p['tbptt_window'] and tick and tick%p['tbptt_window']==0:z=anchors['before'][tick].copy()
        before.append(z.copy());z[v]+=.2*(np.array(weights[bank]) if g_gain is None else z[g_gain])
        margin=z[v]-.5;event=(margin>0).astype(float);margins.append(margin.copy());hard.append(event.copy())
        if anchors is not None:
            old=anchors['margins'][tick];event=anchors['hard'][tick]+p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(old))**2*(margin-old)
        spikes.append(event.copy());z[gain]*=.8;z[v]+=event*(-.3*z[v])
    logits=np.array(spikes).mean(0)*p['logit_scale'];loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
    return loss,z,dict(before=before,margins=margins,hard=hard)


@pytest.mark.parametrize('selection',['empty','partial','full'])
@pytest.mark.parametrize('cross',[False,True])
@pytest.mark.parametrize('ranks',[None,2])
@pytest.mark.parametrize('window',[None,2])
def test_reset_capture_all_bank_initial_vjps(engine,selection,cross,ranks,window):
    mpi(ranks);data=model(selection,cross,ranks,window,engine);bundle=data[4]
    out=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).gradients(np.zeros((1,4,1)),[0])
    loss,z,anchors=reference(data,bundle.weights)
    assert out['loss']==pytest.approx(loss,abs=8e-6)
    for obj,name in [(data[1],'v'),(data[2],'gain')]:
        cells=bundle.provenance['neuron_state_layout'][obj.name][name]
        np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],z[cells],rtol=8e-5,atol=8e-6)
    for bank,row in enumerate(bundle.weights):
        for j in range(len(row)):
            hi=copy.deepcopy(bundle.weights);lo=copy.deepcopy(bundle.weights);hi[bank][j]+=1e-6;lo[bank][j]-=1e-6
            fd=(reference(data,hi,anchors=anchors)[0]-reference(data,lo,anchors=anchors)[0])/2e-6
            assert out['gradients'][bank][j]==pytest.approx(fd,rel=1e-3,abs=1e-5),(bank,j)
    for j in range(len(bundle.initial_state)):
        if bundle.plan['dynamic']['detached'][j]:continue
        hi=np.array(bundle.initial_state);lo=hi.copy();hi[j]+=1e-6;lo[j]-=1e-6
        fd=(reference(data,bundle.weights,hi,anchors)[0]-reference(data,bundle.weights,lo,anchors=anchors)[0])/2e-6
        assert out['initial_state_gradients'][0][j]==pytest.approx(fd,rel=1e-3,abs=1e-5),j


@pytest.mark.parametrize('selection',['empty','partial','full'])
@pytest.mark.parametrize('ranks',[None,2])
def test_reset_capture_aliases_selected_voltage(engine,selection,ranks):
    mpi(ranks);net,g,physical,dt,bundle=model(selection,False,ranks,None,engine,capture_voltage=True)
    p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights)
    t=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER)
    for tick in range(4):
        out=t.step(np.zeros((1,1,1)),[0],initial='carry' if tick else None);net.run(dt,namespace={})
        cells=bundle.provenance['neuron_state_layout'][g.name]['v']
        np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],g.v[:],rtol=8e-5,atol=8e-6)


def mixed_reset_capture(array):
    def f(x):
        captured=array
        captured*=.8
        return x+captured
    return f


def test_reset_mixed_selection_shape_rejected_without_execution():
    from brian2_rust.training_effects import lower_state_effect_function,bind_state_effect_captures,compile_state_effect_transform
    array=np.array([.1,.2]);f=bind_state_effect_captures(lower_state_effect_function(mixed_reset_capture(array)),{'array':'captured'})
    with pytest.raises(ValueError,match='whole arrays|broadcast'):
        compile_state_effect_transform('v=f(v)',states={'v':0,'captured':1},parameters={'f':f},
            array_states={'v','captured'},writable_states={'v','captured'},copied_array_states={'v'},unconditional_states={'captured'})
    np.testing.assert_array_equal(array,[.1,.2])


def typed_reset_capture(array):
    def f(x):
        captured=array
        captured+=1
        x*=.7
        return x
    return f


def boolean_reset_capture(array):
    def f(x):
        captured=array
        captured+=True
        x*=.7
        return x
    return f


@pytest.mark.parametrize('dtype',['integer','boolean'])
@pytest.mark.parametrize('selection',['empty','partial','full'])
@pytest.mark.parametrize('ranks',[None,2])
def test_reset_typed_capture_and_selected_coefficient(engine,dtype,selection,ranks):
    mpi(ranks);b.set_device('runtime');b.start_scope();b.prefs.codegen.target='numpy';dt=.2*b.ms
    source=b.SpikeGeneratorGroup(1,[],np.array([])*b.second,dt=dt)
    hidden=b.NeuronGroup(1,'dv/dt=0/second:1',threshold='v>100',reset='v=0',method='euler',dt=dt)
    g=b.NeuronGroup(2,'dv/dt=0/second:1\nk:'+dtype+' (constant)',threshold='v>.5',reset='v=f(v)-.1*int(k)',method='euler',dt=dt)
    g.v={'empty':[.02,.07],'partial':[.02,.73],'full':[.71,.83]}[selection];g.k=[1,2] if dtype=='integer' else [False,True]
    array=g.variables['k'].get_value();g.namespace['f']=b.Function((typed_reset_capture if dtype=='integer' else boolean_reset_capture)(array),arg_units=[1],arg_names=['x'],return_unit=1,stateless=False)
    net=b.Network(source,hidden,g);bundle=lower_brian_dynamic_training(net,input_group=source,layers=[hidden,g],backend=engine,mpi_ranks=ranks)
    t=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER)
    for tick in range(3):
        out=t.step(np.zeros((1,1,1)),[0],initial='carry' if tick else None);net.run(dt,namespace={})
        for name in ['v','k']:
            cells=bundle.provenance['neuron_state_layout'][g.name][name]
            np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],g.variables[name].get_value(),rtol=8e-5,atol=8e-6)
        for name in ['k']:
            cells=bundle.provenance['neuron_state_layout'][g.name][name]
            np.testing.assert_array_equal(np.asarray(out['initial_state_gradients'])[0,cells],0.)
