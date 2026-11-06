"""Independent Brian integration and sequential reset semantics for v4."""
import os

import brian2 as b
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer
from brian2_rust.training_brian import lower_brian_training,TrainingConversionError
from test_native_training import RUNNER


def model(reset='subtract',units=False):
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='numpy'
    dt=.2*b.ms
    x=np.array([[1,0],[0,1],[1,1],[0,0],[1,0],[1,1],[0,1],[1,0],[0,0],[1,1],[1,0],[1,1]],float)
    ticks,ids=np.nonzero(x);source=b.SpikeGeneratorGroup(2,ids,ticks*dt,dt=dt,name='multi_input')
    groups=[];dim='volt' if units else '1';scale=b.mV if units else 1.
    for l,name in enumerate(['multi_hidden','multi_output']):
        eq=f'''dv/dt = (-v+.3*a{'+.1*c' if l else ''})/tau : {dim}
        da/dt = (.15*v-.4*a)/tau : {dim}
        tau : second (shared,constant)
        theta : {dim} (shared,constant)
        kick : {dim} (shared,constant)'''
        if l:eq+=f'\ndc/dt=(.1*a-.3*c)/tau : {dim}'
        reset_code='a+=kick+.1*v\n'+('v=0*volt' if units else 'v=0' if reset=='zero' else 'v-=theta')
        if units and reset!='zero':reset_code='a+=kick+.1*v\nv-=theta'
        if l:reset_code+='\nc=.9*c+.05*a'
        g=b.NeuronGroup(2,eq,threshold='v>theta',reset=reset_code,method='euler',dt=dt,name=name)
        g.tau=(1+l*.1)*b.ms;g.theta=1.0625*scale;g.kick=.07*scale
        g.v=np.array([.2,1.8])*scale;g.a=np.array([.1,.3])*scale
        if l:g.c=np.array([.05,.15])*scale
        groups.append(g)
    synapses=[]
    for q,(src,dst) in enumerate([(source,groups[0]),(groups[0],groups[1]),(groups[1],groups[0])]):
        syn=b.Synapses(src,dst,'w:'+dim,on_pre='v_post+=w',name=f'multi_syn_{q}')
        syn.connect(i=[1,0,1,0],j=[0,0,1,1]);syn.w=np.array([1.2,.4,.3,1.1])*scale;synapses.append(syn)
    return b.Network(source,*groups,*synapses),source,groups,x,dt


@pytest.mark.parametrize('reset',['zero','subtract'])
@pytest.mark.parametrize('units',[False,True])
def test_multistate_brian_forward(reset,units):
    net,source,groups,x,dt=model(reset,units)
    bundle=lower_brian_training(net,input_group=source,layers=groups,
        trainable_neuron_parameters={g.name:['tau','theta','kick'] for g in groups})
    assert bundle.plan['schema']=='b2-state-training-plan-v4'
    assert bundle.provenance['state_names']==[['v','a'],['v','a','c']]
    assert len(bundle.initial_state)==10 and len(bundle.initial_membrane)==4
    trainer=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights)
    result=trainer.evaluate(x[None],[0],initial=[bundle.initial_state])
    monitors=[b.SpikeMonitor(g) for g in groups];net.add(*monitors);net.run(len(x)*dt,namespace={})
    spikes=np.zeros((len(x),4))
    for l,monitor in enumerate(monitors):
        ticks=np.rint(np.asarray(monitor.t/b.second)/float(dt/b.second)).astype(int)
        spikes[ticks,2*l+np.asarray(monitor.i)]=1
    np.testing.assert_array_equal(result['spikes'][0],spikes)
    expected=np.concatenate([np.asarray(g.variables[name].get_value()) for g,names in zip(groups,bundle.provenance['state_names']) for name in names])
    np.testing.assert_allclose(result['final_state'][0],expected,rtol=5e-13,atol=2e-15)


@pytest.mark.parametrize('mpi',[False,True])
def test_multistate_lowered_training_checkpoint(mpi,tmp_path):
    if mpi and os.environ.get('B2_TEST_MPI')!='1':pytest.skip('MPI required')
    net,source,groups,x,dt=model()
    bundle=lower_brian_training(net,input_group=source,layers=groups,mpi_ranks=2 if mpi else None,
        trainable_neuron_parameters={g.name:['tau','theta','kick'] for g in groups},learning_rate=1e-7)
    trainer=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights)
    first=trainer.step(x[None],[0],initial=[bundle.initial_state]);assert trainer.state['weights']!=bundle.weights
    assert any(abs(v)>1e-8 for row in first['initial_state_gradients'] for v in row)
    path=tmp_path/'checkpoint';trainer.store(path);restored=NativeLIFTrainer(bundle.plan,runner=RUNNER);restored.restore(path)
    assert restored.neuron_state==first['final_state']
    assert trainer.step(x[None],[0],initial='carry')==restored.step(x[None],[0],initial='carry')


def test_multistate_unsupported_reset():
    net,source,groups,x,dt=model()
    groups[0].event_codes['spike']='kick+=.1\nv=0'
    with pytest.raises(TrainingConversionError,match='scalar/shared storage'):
        lower_brian_training(net,input_group=source,layers=groups)
