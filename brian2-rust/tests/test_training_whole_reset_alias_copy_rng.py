"""Foreign whole reset capture and physical voltage alias vs caller copies."""
import copy
import brian2 as b
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer,lower_brian_dynamic_training
from test_training_integer_ir import engine
from test_training_poisson_zero_vjp import mpi
from test_training_stochastic import normal
from test_native_training import RUNNER


def factory(foreign,voltage):
    def f(x,k,mean,spread):
        a=foreign
        v=voltage
        a*=.8
        v*=.7
        v+=.03
        draws=np.random.normal(mean+.1*a,spread,size=len(a))
        return x+.1*draws[k]
    return f


def model(backend,selection,window):
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='numpy';dt=.2*b.ms
    source=b.SpikeGeneratorGroup(1,[],np.array([])*dt,dt=dt)
    storage=b.NeuronGroup(4,'dv/dt=0/second:1\na:1',threshold='v>100',reset='v=0',method='euler',dt=dt);storage.a=[.1,.2,.3,.4]
    g=b.NeuronGroup(2,'dv/dt=gain/ms:1\nk:integer\nmean:1 (constant,shared)\nspread:1 (constant,shared)\ngain:1 (constant)',threshold='v>.5',reset='v=.2*f(v,k,mean,spread)+v-.5',method='euler',dt=dt)
    g.v={'empty':[.1,.2],'partial':[.1,.7],'full':[.6,.7]}[selection];g.k=[3,0];g.mean=.043;g.spread=.13;g.gain=[.217,.319]
    g.namespace['f']=b.Function(factory(storage.variables['a'].get_value(),g.variables['v'].get_value()),arg_units=[1,1,1,1],arg_types=['float','integer','float','float'],return_unit=1,stateless=False)
    net=b.Network(source,storage,g)
    bundle=lower_brian_dynamic_training(net,input_group=source,layers=[storage,g],backend=backend,mpi_ranks=2,tbptt_window=window,detach_reset=False,trainable_neuron_parameters={g.name:['gain','mean','spread']})
    return net,storage,g,dt,bundle


def oracle(data,weights,initial=None,anchors=None):
    _,storage,g,_,bundle=data;p=bundle.plan;z=np.array(bundle.initial_state if initial is None else initial,float)
    if initial is None:
        for cell,ref in enumerate(p['dynamic']['initial_parameters']):
            if ref is not None:z[cell]=weights[ref[0]][ref[1]]
    layout=bundle.provenance['neuron_state_layout'];a=layout[storage.name]['a'];v=layout[g.name]['v'];k=layout[g.name]['k']
    def parameter(name):
        row=next(row for row in bundle.provenance['bindings'] if row['object']==g.name and name in row['variables'])
        return weights[row['bank']][row['variables'].index(name)] if g.variables[name].scalar else np.asarray(weights[row['bank']])
    stream=bundle.provenance['whole_neuron_reset_layout'][g.name]['draws'][0]['stream'];before=[];margins=[];hard=[];events=[]
    for tick in range(6):
        if anchors is not None and p['tbptt_window'] and tick and tick%p['tbptt_window']==0:z=anchors['before'][tick].copy()
        before.append(z.copy());z[v]+=.2*parameter('gain');caller=z[v].copy();margin=caller-.5;event=(margin>0).astype(float)
        margins.append(margin.copy());hard.append(event.copy())
        if anchors is not None:event=anchors['hard'][tick]+p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(anchors['margins'][tick]))**2*(margin-anchors['margins'][tick])
        events.append(event.copy());z[a]*=.8;z[v]=.7*caller+.03
        draws=parameter('mean')+.1*z[a]+parameter('spread')*np.array([normal(p['seed'],9,0,1,j,tick,stream) for j in range(4)])
        candidate=1.2*caller+.02*draws[z[k].astype(int)]-.5
        z[v]+=event*(candidate-z[v])
    logits=np.array(events).mean(0)*p['logit_scale'];loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
    return loss,z,dict(before=before,margins=margins,hard=hard,events=events)


@pytest.mark.parametrize('selection',['empty','partial','full'])
def test_whole_reset_voltage_alias_original_restore(engine,selection,monkeypatch,tmp_path):
    mpi(2);data=model(engine,selection,None);net,storage,g,dt,bundle=data
    p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights);trainer=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER)
    stream=bundle.provenance['whole_neuron_reset_layout'][g.name]['draws'][0]['stream'];calls=[]
    for tick in range(4):
        def sample(mean,spread,*,size):
            assert size==4;calls.append(tick)
            return mean+spread*np.array([normal(p['seed'],9,0,1,j,tick,stream) for j in range(4)])
        monkeypatch.setattr(np.random,'normal',sample)
        out=trainer.step(np.zeros((1,1,1)),[0],**({'initial':'carry'} if tick else {'noise_sequence':9}));net.run(dt,namespace={})
        for obj,names in [(storage,('a',)),(g,('v','k'))]:
            for name in names:
                cells=bundle.provenance['neuron_state_layout'][obj.name][name]
                np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],obj.variables[name].get_value(),rtol=8e-5,atol=8e-6)
        path=tmp_path/str(tick);trainer.store(path);trainer=NativeLIFTrainer(p,runner=RUNNER);trainer.restore(path)
    assert calls==list(range(4))


@pytest.mark.parametrize('window',[None,2])
def test_whole_reset_voltage_alias_all_vjps(engine,window):
    mpi(2);data=model(engine,'partial',window);bundle=data[4];x=np.zeros((1,6,1));loss,z,anchors=oracle(data,bundle.weights)
    out=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).gradients(x,[0],noise_sequence=9)
    assert out['loss']==pytest.approx(loss,abs=8e-6)
    cells=[*bundle.provenance['neuron_state_layout'][data[1].name]['a'],*bundle.provenance['neuron_state_layout'][data[2].name]['v']]
    np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],z[cells],rtol=8e-5,atol=8e-6)
    for bank,values in enumerate(bundle.weights):
        for index in range(len(values)):
            hi=copy.deepcopy(bundle.weights);lo=copy.deepcopy(bundle.weights);hi[bank][index]+=1e-6;lo[bank][index]-=1e-6
            fd=(oracle(data,hi,anchors=anchors)[0]-oracle(data,lo,anchors=anchors)[0])/2e-6
            assert out['gradients'][bank][index]==pytest.approx(fd,rel=3e-3,abs=2e-5),(bank,index)
    initial=np.array(bundle.initial_state,float)
    explicit=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).gradients(x,[0],initial=initial[None],noise_sequence=9)
    _,_,anchors=oracle(data,bundle.weights,initial=initial)
    for index in range(len(initial)):
        if bundle.plan['dynamic']['detached'][index] or index in bundle.plan['dynamic']['integer_states']:
            assert explicit['initial_state_gradients'][0][index]==0.;continue
        hi=initial.copy();lo=initial.copy();hi[index]+=1e-6;lo[index]-=1e-6
        fd=(oracle(data,bundle.weights,initial=hi,anchors=anchors)[0]-oracle(data,bundle.weights,initial=lo,anchors=anchors)[0])/2e-6
        assert explicit['initial_state_gradients'][0][index]==pytest.approx(fd,rel=3e-3,abs=2e-5),index
