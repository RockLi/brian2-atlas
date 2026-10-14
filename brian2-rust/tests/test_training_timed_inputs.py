"""Native TimedArray sampling, input VJP and sequence-boundary replacement."""
import copy
import os
import numpy as np
import pytest
import brian2 as b
from brian2_rust import NativeLIFTrainer, lower_brian_dynamic_training
from test_native_training import RUNNER
from test_training_linked import cython_cache
from test_training_dynamic_gpu import backend, compare
from test_training_stochastic import normal


def bank(bundle,owner,name):
    return next(e['bank'] for e in bundle.provenance['bindings'] if e['object']==owner.name and name in e['variables'])


def model(dimensions=2,method='euler',noisy=False,syn_dt=.2,refractory=False,**options):
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='cython'
    values=np.array([[.4,.7],[1.2,.6],[.8,1.3],[1.4,.5]])
    if dimensions==1:values=values[:,0].copy()
    drive=b.TimedArray(values,dt=(.4 if noisy and method=='euler' else .3)*b.ms,name='timed_drive')
    call=lambda t,i:f'drive({t}'+(f', {i}' if dimensions==2 else '')+')'
    x=np.array([[1],[0],[1],[1],[0],[1],[0],[1]],float);ticks=np.nonzero(x[:,0])[0]
    inp=b.SpikeGeneratorGroup(1,np.zeros(len(ticks),int),ticks*.2*b.ms,dt=.2*b.ms,name='timed_input')
    g=b.NeuronGroup(2,'dv/dt=(-v+gain*'+call('t','i')+')/ms'+('+.08*'+call('t','i')+'*xi/ms**.5' if noisy else '')+':1'+(' (unless refractory)' if refractory else '')+'\ngain:1 (constant)',
        threshold='v>1',reset='v-=.7+.02*'+call('t','i'),method=method,dt=.2*b.ms,refractory=.4*b.ms if refractory else False,namespace={'drive':drive},name='timed_neurons')
    g.v=[.8,1.15];g.gain=[1.8,1.6]
    syn=b.Synapses(inp,g,('dz/dt=(-z+.1*'+call('t','j')+')/ms'+('+.06*'+call('t','j')+'*xi/ms**.5' if noisy else '')+':1 (clock-driven)' if syn_dt==.2 else 'z:1')+'\nw:1',
        on_pre='v_post+=w*(1+z)+.03*'+call('t-.1*ms' if syn_dt==.2 else 't+.09*ms','j'),
        on_post='z+=.015*'+call('t','j'),method=method,dt=syn_dt*b.ms,namespace={'drive':drive},name='timed_synapses')
    syn.connect();syn.w=[.35,.4];syn.z=[.1,.15]
    hidden=b.NeuronGroup(1,"dv/dt=-v/ms:1",threshold="v>1",reset="v=0",method="euler",dt=.2*b.ms,name="timed_hidden")
    net=b.Network(inp,hidden,g,syn)
    bundle=lower_brian_dynamic_training(net,input_group=inp,layers=[hidden,g],
        trainable_neuron_parameters={g.name:['gain']},trainable_synapse_parameters={syn.name:['w','z']},**options)
    return net,g,syn,drive,x,bundle


@pytest.mark.parametrize('dimensions',[1,2])
@pytest.mark.parametrize('method',['euler','rk2','rk4'])
@pytest.mark.parametrize('syn_dt',[.2,.4])
def test_timed_inputs_match_brian_cython(dimensions,method,syn_dt):
    net,g,syn,drive,x,bundle=model(dimensions,method,syn_dt=syn_dt,refractory=method=='rk2' and dimensions==2 and syn_dt==.2)
    out=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights).evaluate(x[None],[0])
    monitor=b.SpikeMonitor(g);net.add(monitor);net.run(len(x)*.2*b.ms,namespace={})
    expected=np.zeros((len(x),2));expected[np.rint(monitor.t/(.2*b.ms)).astype(int),monitor.i[:]]=1
    np.testing.assert_array_equal(np.array(out['spikes'][0])[:,1:],expected)
    np.testing.assert_allclose(out['final_membrane'][0][1:],g.v[:],rtol=2e-13,atol=2e-13)
    layout=bundle.provenance['dynamic_state_layout'][syn.name]
    np.testing.assert_allclose(np.array(out['final_state'][0])[layout['z']],syn.z[:],rtol=2e-13,atol=2e-13)
    assert len(bundle.provenance['timed_inputs'])==1
    source=bundle.provenance['timed_inputs'][0]
    assert len(source['aliases'])==2 and not bundle.plan['trainable'][source['bank']]


def oracle(bundle,g,syn,weights,x,*,anchors=None,noisy=False,sequence=0,start_tick=0,change=None):
    p=bundle.plan;entry=bundle.provenance['timed_inputs'][0];table=np.array(weights[entry['bank']]).reshape(entry['shape'])
    gain=np.array(weights[bank(bundle,g,'gain')]);w=np.array(weights[bank(bundle,syn,'w')]);z=np.array(weights[bank(bundle,syn,'z')])
    v=np.array([.8,1.15]);history=[];spikes=[]
    array_dt=entry['dt_seconds'];k=max(1,int(2**np.ceil(np.log2(8*array_dt/.0002))))
    def sample(t):
        row=int(np.clip((t/(array_dt/k)+.5)/k,0,len(table)-1))
        return np.full(2,table[row]) if table.ndim==1 else table[row]
    for tick,external in enumerate(x[:,0]):
        if change is not None and tick==change[0]:table=np.array(change[1]).reshape(entry['shape'])
        time=(tick+start_tick)*.0002;u=sample(time)
        v=.8*v+.2*gain*u;z=.8*z+.02*u
        if noisy:
            amplitude=(u+sample(time+.0002))/2 if bundle.provenance['integrators'][-1]=='heun' else u
            v+=.08*amplitude*np.sqrt(.2)*np.array([normal(p['seed'],sequence,0,1,j,tick+start_tick,0) for j in range(2)])
            z+=.06*amplitude*np.sqrt(.2)*np.array([normal(p['seed'],sequence,0,2,j,tick+start_tick,0) for j in range(2)])
        pre=v.copy();hard=(pre>1).astype(float)
        s=hard if anchors is None else (anchors[tick]>1).astype(float)+p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(anchors[tick]-1))**2*(pre-anchors[tick])
        history.append(pre);spikes.append(s)
        v+=external*(w*(1+z)+.03*sample(time-.0001))
        z+=s*.015*u
        reset=hard if anchors is None or p['detach_reset'] else s
        if anchors is not None and p['detach_reset']:reset=(anchors[tick]>1).astype(float)
        v-=reset*(.7+.02*u)
    logits=np.array(spikes).mean(axis=0)*p['logit_scale']
    loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
    return loss,history,v,z


@pytest.mark.parametrize('method,noisy',[('euler',False),('euler',True),('heun',True),('milstein',True)])
@pytest.mark.parametrize('detach',[False,True])
def test_table_and_parameter_gradients_independent(method,noisy,detach):
    _,g,syn,_,x,bundle=model(method=method,noisy=noisy,detach_reset=detach)
    trainer=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights)
    out=trainer.gradients(x[None],[0],**(dict(noise_sequence=7) if noisy else {}))
    loss,anchors,v,z=oracle(bundle,g,syn,bundle.weights,x,noisy=noisy,sequence=7)
    assert out['loss']==pytest.approx(loss,abs=2e-13)
    np.testing.assert_allclose(out['final_membrane'][0][1:],v,atol=2e-13)
    for bidx,values in enumerate(bundle.weights):
        for j in range(len(values)):
            hi=copy.deepcopy(bundle.weights);lo=copy.deepcopy(bundle.weights);hi[bidx][j]+=1e-6;lo[bidx][j]-=1e-6
            fd=(oracle(bundle,g,syn,hi,x,anchors=anchors,noisy=noisy,sequence=7)[0]-oracle(bundle,g,syn,lo,x,anchors=anchors,noisy=noisy,sequence=7)[0])/2e-6
            assert out['gradients'][bidx][j]==pytest.approx(fd,rel=4e-4,abs=3e-7)


@pytest.mark.parametrize('method,noisy',[('euler',False),('euler',True),('heun',True),('milstein',True)])
@pytest.mark.parametrize('ranks',[None,2,8])
def test_timed_input_gpu_and_mpi(method,noisy,ranks,backend):
    if ranks is not None and os.environ.get('B2_TEST_MPI')!='1':pytest.skip('local MPI required')
    *_,x,bundle=model(method=method,noisy=noisy,refractory=method=='heun',detach_reset=False,tbptt_window=3)
    kw=dict(noise_sequence=9) if noisy else {}
    cpu=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights).gradients(x[None],[0],**kw)
    bundle.plan.update(backend=backend,mpi_ranks=ranks)
    out=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights).gradients(x[None],[0],**kw)
    compare(out,cpu,backend)


@pytest.mark.parametrize('engine,ranks',[('cpu',None),('cpu',2),('metal',None),('metal',2)])
@pytest.mark.parametrize('noisy',[False,True])
def test_input_replacement_preserves_carry_optimizer_and_checkpoint(engine,ranks,noisy,tmp_path):
    if engine=='metal' and os.environ.get('B2_TEST_GPU')!='1':pytest.skip('actual Metal required')
    if ranks is not None and os.environ.get('B2_TEST_MPI')!='1':pytest.skip('local MPI required')
    _,g,syn,drive,x,bundle=model(noisy=noisy,backend=engine,mpi_ranks=ranks)
    bundle.plan['trainable']=[False]*len(bundle.weights)
    trainer=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights)
    trainer.step(x[None,:3],[0],**(dict(noise_sequence=7) if noisy else {}))
    values=np.array(drive.values)*.7+.2;source=bundle.provenance['timed_inputs'][0]['bank']
    before=copy.deepcopy((trainer.state,trainer.neuron_state,trainer.clock_tick,trainer.elapsed_ticks,trainer.noise_sequence,trainer.next_noise_sequence))
    trainer.update_timed_input(source,values)
    expected=copy.deepcopy(before[0]);expected['weights'][source]=values.reshape(-1).tolist()
    assert trainer.state==expected
    assert (trainer.neuron_state,trainer.clock_tick,trainer.elapsed_ticks,trainer.noise_sequence,trainer.next_noise_sequence)==before[1:]
    path=tmp_path/'timed.json';trainer.store(path)
    restored=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights);restored.restore(path)
    assert restored.state==trainer.state
    tail=restored.step(x[None,3:],[0],initial='carry')
    _,_,v,z=oracle(bundle,g,syn,bundle.weights,x,noisy=noisy,sequence=7,change=(3,values))
    np.testing.assert_allclose(tail['final_membrane'][0][1:],v,rtol=1e-5,atol=3e-6)
    slots=bundle.provenance['dynamic_state_layout'][syn.name]['z']
    np.testing.assert_allclose(np.array(tail['final_state'][0])[slots],z,rtol=1e-5,atol=3e-6)


def test_input_update_before_first_execution_and_unchanged_topology():
    *_,x,bundle=model()
    source=bundle.provenance['timed_inputs'][0]['bank'];weights=copy.deepcopy(bundle.weights);weights[source]=[.5]*len(weights[source])
    trainer=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights)
    trainer.update_timed_input(source,weights[source])
    assert trainer.neuron_state is None and trainer.clock_tick==0 and trainer.elapsed_ticks==0
    actual=trainer.evaluate(x[None],[0])
    reference=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=weights).evaluate(x[None],[0])
    assert actual==reference


@pytest.mark.parametrize('issue',['size','bank','trainable','nonfinite','zero_epsilon','bad_k','bad_shape','bad_dependency'])
def test_invalid_timed_inputs_fail_atomically(issue):
    *_,x,bundle=model();p=bundle.plan;source=bundle.provenance['timed_inputs'][0]['bank']
    values=copy.deepcopy(bundle.weights[source]);target=source
    if issue=='size':values.pop()
    elif issue=='bank':target=len(bundle.weights)
    elif issue=='trainable':p['trainable'][source]=True
    elif issue=='nonfinite':values[0]=float('nan')
    else:
        node=next(n for programs in p['dynamic']['program_sets'] for program in programs for n in program if n['op']=='timed_parameter')
        if issue=='zero_epsilon':node['epsilon']=0
        elif issue=='bad_k':node['k']=3
        elif issue=='bad_shape':node['columns']+=1
        else:node['time']=128
    trainer=NativeLIFTrainer(p,runner=RUNNER,weights=bundle.weights);before=copy.deepcopy(trainer.state)
    with pytest.raises(ValueError):
        if issue in ('size','bank','trainable','nonfinite'):trainer.update_timed_input(target,values)
        else:trainer.step(x[None],[0])
    assert trainer.state==before and trainer.neuron_state is None and trainer.clock_tick==0


@pytest.mark.parametrize('method',['euler','rk4'])
def test_timed_input_snapshot_preserves_absolute_time(method):
    net,g,syn,drive,x,_=model(method=method)
    net.run(.4*b.ms,namespace={})
    inp=next(o for o in net.objects if o.name=='timed_input')
    hidden=next(o for o in net.objects if o.name=='timed_hidden')
    bundle=lower_brian_dynamic_training(net,input_group=inp,layers=[hidden,g])
    actual=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights).evaluate(x[None,2:],[0])
    net.run(1.2*b.ms,namespace={})
    np.testing.assert_allclose(actual['final_membrane'][0][1:],g.v[:],atol=2e-13)
    slots=bundle.provenance['dynamic_state_layout'][syn.name]['z']
    np.testing.assert_allclose(np.array(actual['final_state'][0])[slots],syn.z[:],atol=2e-13)


def test_timed_input_si_units_and_negative_last_bin():
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='cython'
    drive=b.TimedArray(np.array([[.4,.7],[1.2,.6],[.8,1.3]])*b.mV,dt=.3*b.ms)
    inp=b.SpikeGeneratorGroup(1,[],[]*b.ms,dt=.2*b.ms)
    hidden=b.NeuronGroup(1,'dv/dt=-v/ms:1',threshold='v>1',reset='v=0',dt=.2*b.ms,method='euler')
    g=b.NeuronGroup(2,'dv/dt=(-v+drive(t-.2*ms,i))/ms:volt\ntheta:volt (constant)',threshold='v>theta',reset='v-=.8*mV',dt=.2*b.ms,method='rk2',namespace={'drive':drive})
    g.v=[.7,1.2]*b.mV;g.theta=1*b.mV;net=b.Network(inp,hidden,g)
    bundle=lower_brian_dynamic_training(net,input_group=inp,layers=[hidden,g])
    actual=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights).evaluate(np.zeros((1,8,1)),[0])
    net.run(1.6*b.ms,namespace={})
    np.testing.assert_allclose(actual['final_membrane'][0][1:],np.asarray(g.v[:]),atol=2e-15)



def test_timed_input_gpu_descriptor_budget_and_float32_update(backend):
    *_,x,bundle=model(backend=backend)
    trainer=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights)
    result=trainer.evaluate(x[None],[0]);source=bundle.provenance['timed_inputs'][0]['bank']
    before=copy.deepcopy(trainer.state)
    with pytest.raises(ValueError,match='float32'):trainer.update_timed_input(source,[1e100]*len(bundle.weights[source]))
    assert trainer.state==before
    limited=copy.deepcopy(bundle.plan);limited['max_tape_bytes']=result['tape_bytes']-1
    trainer=NativeLIFTrainer(limited,runner=RUNNER,weights=bundle.weights);before=copy.deepcopy(trainer.state)
    with pytest.raises(ValueError,match='budget'):trainer.step(x[None],[0])
    assert trainer.state==before and trainer.neuron_state is None


@pytest.mark.parametrize('issue',['nonfinite_table','empty_table','bad_dt','input_budget'])
def test_timed_input_conversion_rejects_invalid_sources(issue):
    net,g,syn,drive,x,_=model()
    if issue=='nonfinite_table':drive.values[0,0]=float('nan')
    elif issue=='empty_table':drive.values=np.empty((0,2))
    elif issue=='bad_dt':drive.dt=0.
    else:drive.values=np.ones((1024,2))
    inp=next(o for o in net.objects if o.name=='timed_input')
    hidden=next(o for o in net.objects if o.name=='timed_hidden')
    before=np.array(g.v[:]);options=dict(max_tape_bytes=32768) if issue=='input_budget' else {}
    with pytest.raises(ValueError,match='input|TimedArray'):
        lower_brian_dynamic_training(net,input_group=inp,layers=[hidden,g],**options)
    np.testing.assert_array_equal(g.v[:],before);assert float(net.t)==0


@pytest.mark.parametrize('kind',['path','continuous','summed'])
def test_synapse_only_timed_input_allocation_matches_brian(kind):
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='cython'
    drive=b.TimedArray([[.2,.5],[.7,.3],[1.1,.8]],dt=.4*b.ms)
    inp=b.SpikeGeneratorGroup(1,[0,0,0],[0,.4,.8]*b.ms,dt=.2*b.ms)
    hidden=b.NeuronGroup(1,'dv/dt=-v/ms:1',threshold='v>1',reset='v=0',dt=.2*b.ms,method='euler')
    g=b.NeuronGroup(2,'dv/dt=(-v+current)/ms:1\ncurrent:1',threshold='v>1',reset='v-=.8',dt=.2*b.ms,method='euler')
    g.v=[.8,1.1]
    declaration='w:1'
    if kind=='continuous':declaration+='\ndz/dt=(-z+drive(t,j))/ms:1 (clock-driven)'
    if kind=='summed':declaration+='\ncurrent_post=w*drive(t,j):1 (summed)'
    code='v_post+=w*drive(t,j)' if kind=='path' else 'v_post+=w+z' if kind=='continuous' else 'v_post+=w'
    syn=b.Synapses(inp,g,declaration,on_pre=code,dt=.2*b.ms,method='euler',namespace={'drive':drive})
    syn.connect();syn.w=[.3,.4];net=b.Network(inp,hidden,g,syn)
    bundle=lower_brian_dynamic_training(net,input_group=inp,layers=[hidden,g])
    source=bundle.provenance['timed_inputs'][0]
    assert len(bundle.provenance['timed_inputs'])==1 and source['aliases']==[dict(object=syn.name,variable='drive')]
    result=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights).evaluate([[[1],[0],[1],[0],[1],[0]]],[0])
    net.run(1.2*b.ms,namespace={})
    np.testing.assert_allclose(result['final_membrane'][0][1:],g.v[:],atol=2e-13)
    if kind=='continuous':
        slots=bundle.provenance['dynamic_state_layout'][syn.name]['z']
        np.testing.assert_allclose(np.array(result['final_state'][0])[slots],syn.z[:],atol=2e-13)
