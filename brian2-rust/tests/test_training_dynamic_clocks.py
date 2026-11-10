"""Native Brian multi-clock sampling, event decay, carry and pathwise VJP."""
import copy
import os
import tempfile
import numpy as np
import pytest
import brian2 as b
from brian2_rust import NativeLIFTrainer, lower_brian_dynamic_training, TrainingConversionError
from brian2_rust.training_dynamic import compile_dynamic_transform
from brian2_rust.training_equations import ClockTime, compile_training_equation
from test_native_training import RUNNER
from test_training_summed import model as summed_model


@pytest.fixture(scope='module',autouse=True)
def cython_cache():
    old=b.prefs.codegen.runtime.cython.cache_dir
    with tempfile.TemporaryDirectory(prefix='b2-clock-cython-') as directory:
        b.prefs.codegen.runtime.cython.cache_dir=directory
        try:yield
        finally:b.prefs.codegen.runtime.cython.cache_dir=old


def model(clock_dt=.3,warmup=0.,event_driven=True,third=False):
    net,inp,groups,static,old,x,dt=summed_model();net.remove(old)
    eq='dz/dt=-z/(.9*ms):1 (event-driven)' if event_driven else 'z:1'
    syn=b.Synapses(*groups,'w:1\nseen:second\n'+eq+'\ng_post=.15*w*(1+t/ms)'+('' if event_driven else '+.1*z')+':1 (summed)',
        on_pre='z+=.02\nv_post+=w*(.6+.1*t/ms+.03*t_pre/ms+.02*t_post/ms+dt/ms+.07*dt_pre/ms+.09*dt_post/ms)+.05*z\nw*=.999\nseen=t',
        on_post='z+=.01\nv_pre+=.03*w*(1+t/ms)\nseen=t',dt=clock_dt*b.ms,name='clock_plastic')
    syn.connect(i=[1,0,1,0],j=[0,1,1,0]);syn.w=[.4,.3,.2,.5];syn.z=[.2,.3,.1,.4];net.add(syn)
    from brian2.codegen.runtime.cython_rt import CythonCodeObject
    for path in syn._pathways:path.codeobj_class=CythonCodeObject
    if third:net.add(b.StateMonitor(groups[0],'v',record=True,dt=.49998*b.ms,name='third_clock'))
    if warmup:net.run(warmup*b.ms,namespace={})
    bundle=lower_brian_dynamic_training(net,input_group=inp,layers=groups,learning_rate=1e-7)
    start=round(bundle.plan['clock']['origin']/float(dt))
    return net,inp,groups,syn,x[start:],dt,bundle


@pytest.mark.parametrize('clock_dt',[.1,.15,.3,.200015,.99999])
@pytest.mark.parametrize('warmup',[0.,.3,.65])
@pytest.mark.parametrize('event_driven',[False,True])
def test_async_event_and_summed_time_match_brian(clock_dt,warmup,event_driven):
    net,inp,groups,syn,x,dt,bundle=model(clock_dt,warmup,event_driven,third=clock_dt==.99999)
    trainer=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights);result=trainer.gradients(x[None],[0])
    monitors=[b.SpikeMonitor(g) for g in groups];net.add(*monitors)
    origin=bundle.plan['clock']['origin'];net.run((origin+len(x)*float(dt)-float(net.t))*b.second,namespace={})
    expected=np.zeros((len(x),4))
    for layer,(group,monitor) in enumerate(zip(groups,monitors)):
        ticks=np.rint((np.asarray(monitor.t/b.second)-origin)/float(dt)).astype(int)
        expected[ticks,2*layer+np.asarray(monitor.i)]=1
        for k,name in enumerate(bundle.provenance['state_names'][layer]):
            np.testing.assert_allclose(result['final_state'][0][4*layer+2*k:4*layer+2*k+2],group.variables[name].get_value(),rtol=5e-13,atol=5e-14)
    np.testing.assert_array_equal(result['spikes'][0],expected)
    for name,indices in bundle.provenance['dynamic_state_layout'][syn.name].items():
        np.testing.assert_allclose(np.asarray(result['final_state'])[0,indices],syn.variables[name].get_value(),rtol=5e-13,atol=5e-14)


@pytest.mark.parametrize('clock_dt',[.3,.99999])
@pytest.mark.parametrize('ranks',[1,2,8])
def test_async_clock_split_checkpoint_and_mpi(clock_dt,ranks,tmp_path):
    if ranks>1 and os.environ.get('B2_TEST_MPI')!='1':pytest.skip('local MPI required')
    *_,x,dt,bundle=model(clock_dt,.3,third=True)
    plan=copy.deepcopy(bundle.plan);plan['trainable']=[False]*len(bundle.weights)
    serial=NativeLIFTrainer(plan,runner=RUNNER,weights=bundle.weights);whole=serial.gradients(x[None],[0])
    if ranks>1:plan['mpi_ranks']=ranks
    trainer=NativeLIFTrainer(plan,runner=RUNNER,weights=bundle.weights)
    parallel=trainer.gradients(x[None],[0])
    for key in ('final_state','spikes','initial_state_gradients'):
        np.testing.assert_allclose(parallel[key],whole[key],rtol=3e-13,atol=3e-13)
    for a,c in zip(parallel['gradients'],whole['gradients']):np.testing.assert_allclose(a,c,rtol=3e-13,atol=3e-13)
    trainer.execute(x[None,:3],[0]);path=tmp_path/'clocks.json';trainer.store(path)
    restored=NativeLIFTrainer(plan,runner=RUNNER,weights=bundle.weights);restored.restore(path)
    tail=restored.execute(x[None,3:],[0],initial='carry')
    np.testing.assert_array_equal(tail['spikes'],np.asarray(whole['spikes'])[:,3:])
    np.testing.assert_allclose(tail['final_state'],whole['final_state'],rtol=1e-14,atol=1e-14)
    assert restored.clock_tick==len(x)


def test_third_clock_can_advance_synaptic_time_before_main_tick():
    net,inp,groups,static,old,x,dt=summed_model();net.remove(old)
    syn=b.Synapses(*groups,'w:1\ng_post=t/ms:1 (summed)',dt=.99999*b.ms);syn.connect();syn.w=.1;net.add(syn)
    net.add(b.StateMonitor(groups[0],'v',record=True,dt=.49998*b.ms))
    bundle=lower_brian_dynamic_training(net,input_group=inp,layers=groups)
    result=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights).evaluate(x[None,:6],[0])
    # At main t=1ms, the 0.99996ms third-clock event has already coalesced
    # the 0.99999ms synaptic event, but not the main event (smaller tolerance).
    # Merely ceil(t/dt_syn), even with Clock._calc_timestep, returns 0.99999ms.
    np.testing.assert_allclose(result['final_state'][0][6:8],[3.99996]*2,rtol=0,atol=1e-14)
    net.run(6*dt,namespace={});np.testing.assert_allclose(result['final_state'][0][6:8],groups[1].g[:],rtol=0,atol=1e-14)


@pytest.mark.parametrize('value',[-1,256,True,1.5])
def test_clock_ssa_rejects_invalid_index(value):
    with pytest.raises(ValueError,match='clock'):
        compile_dynamic_transform('x=t',states={'x':0},parameters={'t':ClockTime(value)})


def test_clock_ssa_requires_dynamic_contract():
    with pytest.raises(ValueError,match='clock'):
        compile_training_equation('v+t',states=['v'],parameters={'t':ClockTime(0)})


@pytest.mark.parametrize('issue',['index','missing','epsilon','start','origin','dt','precision','work'])
def test_native_clock_schedule_validation_is_transactional(issue):
    *_,x,dt,bundle=model();plan=copy.deepcopy(bundle.plan)
    if issue=='index':
        node=next(node for ps in plan['dynamic']['program_sets'] for p in ps for node in p if node['op']=='clock_time');node['index']=256
    elif issue=='missing':plan['dynamic'].pop('clocks')
    elif issue=='epsilon':plan['dynamic']['clocks']['epsilon']=.5
    elif issue=='start':plan['dynamic']['clocks']['start']=-1
    elif issue=='origin':plan['clock']['origin']=.01
    elif issue=='dt':plan['dynamic']['clocks']['dts'][1]=0
    elif issue=='precision':plan['dynamic']['clocks']['start']=1e30
    elif issue=='work':plan['dynamic']['clocks']['dts'][1]=1e-10
    trainer=NativeLIFTrainer(plan,runner=RUNNER,weights=bundle.weights);before=copy.deepcopy(trainer.state)
    with pytest.raises(ValueError,match='clock|SSA'):trainer.execute(x[None],[0])
    assert trainer.state==before and trainer.clock_tick==0


def brian_clock_samples(dts,count):
    # Use Brian's real scheduler as the independent source of all timestamps.
    b.set_device('runtime');b.start_scope();clocks=[b.Clock(dt=dt*b.second) for dt in dts];seen=[]
    objects=[b.NetworkOperation(lambda:seen.append(float(clocks[1].variables['t'].get_value()[0])),clock=clocks[0],when='end')]
    objects.extend(b.NetworkOperation(lambda:None,clock=clock) for clock in clocks[1:])
    b.Network(*objects).run(count*dts[0]*b.second)
    assert len(seen)==count
    return np.array(seen)


@pytest.mark.parametrize('clock_dt',[.0003,.00099999])
@pytest.mark.parametrize('detach,window',[(True,None),(False,None),(True,3),(False,3)])
@pytest.mark.parametrize('explicit_initial',[False,True])
def test_async_event_decay_independent_vjp(clock_dt,detach,window,explicit_initial):
    from test_training_dynamic import model as raw_model,oracle
    plan,weights,x=raw_model(detach,window,event_driven=True)
    plan['dynamic']['clocks']=dict(start=0.,dts=[.0002,clock_dt,.00049998],epsilon=float(b.Clock.epsilon_dt))
    for programs in plan['dynamic']['program_sets']:
        for program in programs:
            for node in program:
                if node['op']=='time':node.update(op='clock_time',index=1)
    times=brian_clock_samples(plan['dynamic']['clocks']['dts'],len(x))
    initial=np.array(plan['dynamic']['initial']) if explicit_initial else None
    actual=NativeLIFTrainer(plan,runner=RUNNER,weights=weights).gradients(x[None],[0],initial=None if initial is None else initial[None])
    loss,spikes,state,anchors=oracle(plan,weights,x,initial,clock_times=times)
    assert actual['loss']==pytest.approx(loss,abs=2e-14)
    np.testing.assert_array_equal(actual['spikes'][0],spikes)
    np.testing.assert_allclose(actual['final_state'][0],state,rtol=5e-14,atol=5e-14)
    for bank,row in enumerate(weights):
        for j,value in enumerate(row):
            eps=1e-6*max(abs(value),1e-3);a=copy.deepcopy(weights);c=copy.deepcopy(weights);a[bank][j]+=eps;c[bank][j]-=eps
            fd=(oracle(plan,a,x,initial,anchors,clock_times=times)[0]-oracle(plan,c,x,initial,anchors,clock_times=times)[0])/(2*eps)
            assert actual['gradients'][bank][j]==pytest.approx(fd,abs=3e-6,rel=3e-4)
    if explicit_initial:
        for j in range(len(initial)):
            if plan['dynamic']['detached'][j]:
                assert actual['initial_state_gradients'][0][j]==0;continue
            a=initial.copy();c=initial.copy();a[j]+=1e-6;c[j]-=1e-6
            fd=(oracle(plan,weights,x,a,anchors,clock_times=times)[0]-oracle(plan,weights,x,c,anchors,clock_times=times)[0])/2e-6
            assert actual['initial_state_gradients'][0][j]==pytest.approx(fd,abs=2e-7,rel=3e-4)


def test_reused_clock_with_stale_network_time_is_rejected():
    net,inp,groups,syn,*_=model(warmup=.3)
    # The group state is after warmup; a new Network would reset its clocks to
    # zero at before_run. It is not a faithful continuation snapshot.
    net.t_=0
    with pytest.raises(TrainingConversionError,match='snapshot clock'):
        lower_brian_dynamic_training(net,input_group=inp,layers=groups)


def test_large_start_tick_replay_is_bounded_and_transactional():
    *_,x,dt,bundle=model();trainer=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights)
    before=copy.deepcopy(trainer.state)
    with pytest.raises(ValueError,match='work budget'):
        trainer.execute(x[None],[0],start_tick=10_000_000)
    assert trainer.state==before and trainer.clock_tick==0
