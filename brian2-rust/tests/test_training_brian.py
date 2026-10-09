"""Brian Network lowering: independent forward and explicit capability gates."""
import copy
import os

import brian2 as b
import numpy as np
import pytest

from brian2_rust import NativeLIFTrainer
from brian2_rust.training_brian import lower_brian_training,TrainingConversionError
from test_native_training import RUNNER


def model(reset='zero',units=False,shared=False):
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='numpy'
    dt=.25*b.ms;steps=12
    x=np.array([[1,0],[0,1],[1,1],[0,1],[1,0],[0,0],[1,1],[1,0],[0,1],[1,1],[0,1],[1,0]],float)
    ticks,ids=np.nonzero(x)
    source=b.SpikeGeneratorGroup(2,ids,ticks*dt,dt=dt,name='train_input')
    dim='volt' if units else '1';scale=b.mV if units else 1.
    equations=f'dv/dt = (leak-v)/tau : {dim}\nleak = .02*v : {dim}\ntau : second (constant,shared)\ntheta : {dim} (constant,shared)'
    groups=[]
    for name in ['train_hidden','train_output']:
        g=b.NeuronGroup(2,equations,threshold='v>theta',reset=('v=0*volt' if units else 'v=0') if reset=='zero' else 'v-=theta',
                        method='euler',dt=dt,name=name)
        g.tau=1*b.ms;g.theta=1.0625*scale;g.v=np.array([.3,1.7])*scale;groups.append(g)
    connections=[]
    for q,(src,dst) in enumerate([(source,groups[0]),(groups[0],groups[1]),(groups[1],groups[0])]):
        syn=b.Synapses(src,dst,'w:'+dim+(' (shared)' if shared else ''),on_pre='v_post+=w',name=f'train_syn_{q}')
        syn.connect(i=[1,0,1,0],j=[0,0,1,1])
        syn.w=(.6 if shared else np.array([1.2,.4,.3,1.1]))*scale
        connections.append(syn)
    net=b.Network(source,*groups,*connections)
    return net,source,groups,connections,x,dt


@pytest.mark.parametrize('reset',['zero','subtract'])
@pytest.mark.parametrize('units',[False,True])
@pytest.mark.parametrize('shared',[False,True])
def test_brian_forward_snapshot_matches_native(reset,units,shared):
    net,source,groups,synapses,x,dt=model(reset,units,shared)
    before=[g.v[:].copy() for g in groups];times=float(net.t/b.second)
    bundle=lower_brian_training(net,input_group=source,layers=groups,
        trainable_neuron_parameters={groups[0].name:['tau','theta']})
    assert float(net.t/b.second)==times
    for g,v in zip(groups,before):np.testing.assert_array_equal(g.v[:],v)
    assert len(bundle.weights[0])==(1 if shared else 4)
    native=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights)
    result=native.evaluate(x[None],[0],initial=[bundle.initial_membrane])
    monitors=[b.SpikeMonitor(g) for g in groups];net.add(*monitors)
    net.run(len(x)*dt,namespace={})
    spikes=np.zeros((len(x),4))
    for l,m in enumerate(monitors):
        ticks=np.rint(np.asarray(m.t/b.second)/float(dt/b.second)).astype(int)
        spikes[ticks,l*2+np.asarray(m.i)]=1
    np.testing.assert_array_equal(result['spikes'][0],spikes)
    np.testing.assert_allclose(result['final_membrane'][0],np.concatenate([np.asarray(g.v[:]) for g in groups]),rtol=2e-13,atol=1e-15)
    assert bundle.provenance['dt_seconds']==float(dt/b.second)
    # Snapshot is detached from later changes to the source model.
    for g in groups:g.v=0
    assert bundle.initial_membrane==np.concatenate([np.asarray(v) for v in before]).tolist()


@pytest.mark.parametrize('backend',['cpu','metal','mpi'])
def test_lowered_network_training_and_restart(backend,tmp_path):
    if backend=='metal' and os.environ.get('B2_TEST_GPU')!='1':pytest.skip('real Metal required')
    if backend=='mpi' and os.environ.get('B2_TEST_MPI')!='1':pytest.skip('real MPI required')
    net,source,groups,synapses,x,dt=model()
    bundle=lower_brian_training(net,input_group=source,layers=groups,
        trainable_neuron_parameters={g.name:['tau','theta'] for g in groups},
        backend='cpu' if backend=='mpi' else backend,mpi_ranks=2 if backend=='mpi' else None,
        learning_rate=1e-6)
    trainer=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights)
    before=copy.deepcopy(bundle.weights)
    result=trainer.step(x[None],[0],initial=[bundle.initial_membrane])
    assert any(abs(v)>1e-8 for row in result['gradients'] for v in row)
    assert trainer.state['weights']!=before
    for q,syn in enumerate(synapses):np.testing.assert_array_equal(np.asarray(syn.w[:]),before[q])
    checkpoint=tmp_path/'checkpoint';trainer.store(checkpoint)
    restored=NativeLIFTrainer(bundle.plan,runner=RUNNER);restored.restore(checkpoint)
    assert trainer.step(x[None],[0],initial='carry')==restored.step(x[None],[0],initial='carry')


@pytest.mark.parametrize('change,code',[
    ('delay','delay'),('plasticity','pathway-code'),('refractory','refractory'),
    ('integrator','integrator'),('clock','clock'),('runner','runner'),
    ('schedule','schedule'),('heterogeneous','coefficient'),('units','units'),
    ('budget','budget'),('parameter','parameter'),('threshold','threshold'),
])
def test_unsupported_model_is_rejected(change,code):
    net,source,groups,synapses,x,dt=model();options={}
    if change=='delay':synapses[0].delay=.25*b.ms
    elif change=='plasticity':synapses[0].pre.code='v_post+=w\nw+=.1'
    elif change=='refractory':groups[0]._refractory=1*b.ms
    elif change=='integrator':groups[0].state_updater.method_choice='exponential_euler'
    elif change=='clock':groups[0].clock.dt=.5*b.ms
    elif change=='runner':groups[0].run_regularly('v+=.1')
    elif change=='schedule':net.schedule=['start','thresholds','groups','synapses','resets','end']
    elif change=='heterogeneous':
        # A supplied external vector cannot be folded into one scalar constant.
        groups[0].namespace['coefficient']=np.array([1.,2.])
        groups[0].events['spike']='v>coefficient'
    elif change=='units':groups[0].namespace['wrong']=1*b.ms;groups[0].events['spike']='v>wrong'
    elif change=='budget':options['max_tape_bytes']=100
    elif change=='parameter':options['trainable_neuron_parameters']={groups[0].name:['leak']}
    elif change=='threshold':groups[0].events['spike']='v>=theta'
    with pytest.raises(TrainingConversionError) as raised:
        lower_brian_training(net,input_group=source,layers=groups,**options)
    assert raised.value.code==code
