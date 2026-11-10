"""Selected reset RNG keeps physical neuron identity and copy writeback."""
import copy
import brian2 as b
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer,lower_brian_dynamic_training
import test_training_scalar_event_rng_functions as factories
from test_training_stochastic import normal
from test_training_uniform import uniform
from test_training_regular_callback_rng import count
from test_training_integer_ir import engine
from test_training_poisson_zero_vjp import mpi
from test_native_training import RUNNER


def model(backend,kind,capture,selection,window=None,reset_prefix=False,capture_voltage=False):
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='numpy';dt=.2*b.ms
    source=b.SpikeGeneratorGroup(1,[],np.array([])*dt,dt=dt)
    hidden=b.NeuronGroup(1,'dv/dt=0/second:1',threshold='v>100',reset='v=0',method='euler',dt=dt)
    reset=('v+=.01*randn()\n' if reset_prefix else '')+'v=.2*f(a,mean,spread)+v-.5'
    g=b.NeuronGroup(2,'dv/dt=gain/ms:1\na:1\nmean:1 (constant)\nspread:1 (constant)\ngain:1 (constant)',method='euler',threshold='v>.5',reset=reset,dt=dt)
    g.v={'subset':[.103,.713],'all':[.713,.937],'empty':[-4.013,-3.017]}[selection]
    g.a=[.3,.6];g.mean=[1.3,.7] if kind in ('poisson','mixed') else [.04,.07];g.spread=[.13,.11];g.gain=[.217,.319]
    function=getattr(factories,kind)(g.variables['v' if capture_voltage else 'a'].get_value()) if capture else getattr(factories,'pure_'+kind)
    g.namespace['f']=b.Function(function,arg_units=[1,1,1],return_unit=1,stateless=False)
    net=b.Network(source,hidden,g)
    bundle=lower_brian_dynamic_training(net,input_group=source,layers=[hidden,g],backend=backend,mpi_ranks=2,tbptt_window=window,detach_reset=False,trainable_neuron_parameters={g.name:['gain','mean','spread']})
    return net,g,dt,bundle


@pytest.mark.parametrize('kind,capture',[('normal',False),('normal',True),('uniform',False),('poisson',False),('mixed',True)])
@pytest.mark.parametrize('selection',['subset','all','empty'])
def test_reset_callback_original_restore(engine,kind,capture,selection,monkeypatch,tmp_path,reset_prefix=False,capture_voltage=False):
    mpi(2);net,g,dt,bundle=model(engine,kind,capture,selection,reset_prefix=reset_prefix,capture_voltage=capture_voltage)
    p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights)
    trainer=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER)
    kinds=['normal','uniform','poisson'] if kind=='mixed' else [kind];calls=[]
    offset=int(reset_prefix)
    assert [r['stream'] for r in bundle.provenance['neuron_draws'][1]['reset']]==list(range(offset+len(kinds)))
    for tick in range(4):
        selected=np.flatnonzero(np.asarray(g.v[:])+.2*np.asarray(g.gain[:])>.5)
        serial=offset
        def sample(which,*args,**kwargs):
            nonlocal serial
            stream=serial;assert which==kinds[(serial-offset)%len(kinds)];serial+=1
            mean=np.asarray(g.mean[:])[selected];spread=np.asarray(g.spread[:])[selected]
            np.testing.assert_allclose(args,(mean,) if which=='poisson' else (mean,spread if which=='normal' else mean+spread))
            calls.append((tick,stream,tuple(selected)))
            if which=='poisson':return np.array([count(p['seed'],9,1,int(j),tick,float(mean[k]),stream) for k,j in enumerate(selected)])
            raw=np.array([(normal if which=='normal' else uniform)(p['seed'],9,0,1,int(j),tick,stream) for j in selected])
            return mean+spread*raw
        for which in kinds:monkeypatch.setattr(np.random,which,lambda *args,_which=which,**kwargs:sample(_which,*args,**kwargs))
        if reset_prefix:
            def standard(n):
                assert n==len(selected)
                return np.array([normal(p['seed'],9,0,1,int(j),tick,0) for j in selected])
            monkeypatch.setattr(np.random,'randn',standard)
        out=trainer.step(np.zeros((1,1,1)),[0],**({'initial':'carry'} if tick else {'noise_sequence':9}));net.run(dt,namespace={})
        for name in ('v','a'):
            cells=bundle.provenance['neuron_state_layout'][g.name][name]
            np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],g.variables[name].get_value(),rtol=8e-5,atol=8e-6)
        path=tmp_path/str(tick);trainer.store(path);trainer=NativeLIFTrainer(p,runner=RUNNER);trainer.restore(path)
    assert len(calls)==4*len(kinds)
    assert (out['gpu_dispatches']>0)==(engine!='cpu')


@pytest.mark.parametrize('scenario',['explicit_stream','capture_voltage'])
def test_reset_callback_stream_and_alias(engine,scenario,monkeypatch,tmp_path):
    test_reset_callback_original_restore(engine,'normal',True,'subset',monkeypatch,tmp_path,
        reset_prefix=scenario=='explicit_stream',capture_voltage=scenario=='capture_voltage')


def reference(data,kind,capture,weights,initial=None,anchors=None):
    from unittest.mock import patch
    _,g,_,bundle=data;p=bundle.plan
    z=np.array(bundle.initial_state if initial is None else initial,dtype=float)
    if initial is None:
        for cell,ref in enumerate(p['dynamic']['initial_parameters']):
            if ref is not None:z[cell]=weights[ref[0]][ref[1]]
    layout=bundle.provenance['neuron_state_layout'][g.name];v=layout['v'];a=layout['a']
    def parameter(name):
        row=next(row for row in bundle.provenance['bindings'] if row['object']==g.name and row['variables']==[name])
        return np.asarray(weights[row['bank']])
    before=[];margins=[];hard=[];events=[];samples=[]
    for tick in range(4):
        if anchors is not None and p['tbptt_window'] and tick and tick%p['tbptt_window']==0:z=anchors['before'][tick].copy()
        before.append(z.copy());z[v]+=.2*parameter('gain')
        margin=z[v]-.5;event=(margin>0).astype(float);margins.append(margin.copy());hard.append(event.copy())
        if anchors is not None:event=anchors['hard'][tick]+p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(anchors['margins'][tick]))**2*(margin-anchors['margins'][tick])
        events.append(event.copy());physical=z[a].copy();function=getattr(factories,kind)(physical) if capture else getattr(factories,'pure_'+kind);serial=0
        def sample(which,*args,**kwargs):
            nonlocal serial
            stream=serial;serial+=1
            if which=='poisson':
                values=[]
                for j,rate in enumerate(args[0]):
                    value=count(p['seed'],9,1,j,tick,float(rate),stream)
                    if anchors is not None:value=next(c for t,s,k,r,c,active in anchors['samples'] if (t,s,k)==(tick,stream,j))
                    samples.append((tick,stream,j,float(rate),value,bool(hard[-1][j])));values.append(value)
                return np.array(values)
            raw=np.array([(normal if which=='normal' else uniform)(p['seed'],9,0,1,j,tick,stream) for j in range(2)])
            return args[0]+(args[1] if which=='normal' else args[1]-args[0])*raw
        with patch.object(np.random,'normal',lambda *args,**kwargs:sample('normal',*args,**kwargs)),patch.object(np.random,'uniform',lambda *args,**kwargs:sample('uniform',*args,**kwargs)),patch.object(np.random,'poisson',lambda *args,**kwargs:sample('poisson',*args,**kwargs)):
            candidate=.2*function(z[a].copy(),parameter('mean'),parameter('spread'))+z[v]-.5
        if capture:z[a]=physical
        z[v]+=event*(candidate-z[v])
    logits=np.array(events).mean(0)*p['logit_scale'];loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
    return loss,z,dict(before=before,margins=margins,hard=hard,samples=samples)


@pytest.mark.parametrize('kind,capture',[('normal',True),('uniform',False),('poisson',False),('mixed',True)])
@pytest.mark.parametrize('window',[None,2])
def test_reset_callback_all_bank_initial_vjps(engine,kind,capture,window):
    mpi(2);data=model(engine,kind,capture,'subset',window);bundle=data[3];x=np.zeros((1,4,1))
    loss,z,anchors=reference(data,kind,capture,bundle.weights)
    out=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).gradients(x,[0],noise_sequence=9)
    assert out['loss']==pytest.approx(loss,abs=8e-6)
    layout=bundle.provenance['neuron_state_layout'][data[1].name];cells=layout['v']+layout['a']
    np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],z[cells],rtol=8e-5,atol=8e-6)
    np.testing.assert_array_equal(np.asarray(out['spikes'])[0,:,1:],anchors['hard'])
    mean=next(row['bank'] for row in bundle.provenance['bindings'] if row['object']==data[1].name and row['variables']==['mean'])
    for bank,values in enumerate(bundle.weights):
        for index in range(len(values)):
            hi=copy.deepcopy(bundle.weights);lo=copy.deepcopy(bundle.weights);hi[bank][index]+=1e-6;lo[bank][index]-=1e-6
            fd=(reference(data,kind,capture,hi,anchors=anchors)[0]-reference(data,kind,capture,lo,anchors=anchors)[0])/2e-6
            score=sum(loss*(c/r-1.) for _,_,j,r,c,active in anchors['samples'] if active and j==index) if bank==mean else 0.
            assert out['gradients'][bank][index]==pytest.approx(fd+score,rel=3e-3,abs=2e-5),(bank,index)
    initial=np.array(bundle.initial_state,dtype=float)
    explicit=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).gradients(x,[0],initial=initial[None],noise_sequence=9)
    _,_,anchors=reference(data,kind,capture,bundle.weights,initial=initial)
    for index in range(len(initial)):
        if bundle.plan['dynamic']['detached'][index] or index in bundle.plan['dynamic']['integer_states']:
            assert explicit['initial_state_gradients'][0][index]==0.;continue
        hi=initial.copy();lo=initial.copy();hi[index]+=1e-6;lo[index]-=1e-6
        fd=(reference(data,kind,capture,bundle.weights,initial=hi,anchors=anchors)[0]-reference(data,kind,capture,bundle.weights,initial=lo,anchors=anchors)[0])/2e-6
        assert explicit['initial_state_gradients'][0][index]==pytest.approx(fd,rel=3e-3,abs=2e-5),index


@pytest.mark.parametrize('kind',[ 'normal','mixed','poisson'])
def test_reset_callback_invalid_restore_atomic(engine,kind,tmp_path):
    mpi(2);data=model(engine,kind,kind!='poisson','all');bundle=data[3];x=np.zeros((1,1,1))
    p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights)
    trainer=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER);trainer.step(x,[0],noise_sequence=9)
    path=tmp_path/'prefix';trainer.store(path);trainer=NativeLIFTrainer(p,runner=RUNNER);trainer.restore(path)
    # Force both neurons to trigger so validation visits the second bad row.
    cells=bundle.provenance['neuron_state_layout'][data[1].name]['v']
    for cell in cells:trainer.neuron_state[0][cell]=.713
    name='mean' if kind=='poisson' else 'spread';row=next(row for row in bundle.provenance['bindings'] if row['object']==data[1].name and row['variables']==[name])
    trainer.state['weights'][row['bank']][1]=-1.
    before=copy.deepcopy((trainer.state,trainer.neuron_state,trainer.clock_state,trainer.poisson_state,trainer.elapsed_ticks,trainer.next_noise_sequence))
    with pytest.raises(ValueError):trainer.step(x,[0],initial='carry')
    assert (trainer.state,trainer.neuron_state,trainer.clock_state,trainer.poisson_state,trainer.elapsed_ticks,trainer.next_noise_sequence)==before
