"""Complete native clock visits against the actual Brian Network scheduler."""
import json
import subprocess
from pathlib import Path

import brian2 as b
import numpy as np
import pytest


@pytest.fixture(scope='module')
def probe(tmp_path_factory):
    out=tmp_path_factory.mktemp('clock-itinerary')/'probe'
    subprocess.run(['rustc','--edition=2021','-O',str(Path(__file__).with_name('clock_itinerary_probe.rs')),'-o',str(out)],check=True,capture_output=True,text=True)
    return out


def native(probe,dts,start,end,order,state=None,restart=None):
    command=[str(probe),str(start),str(end),','.join(map(str,dts)),','.join(map(str,order))]
    if state is not None:command.append(':'.join([str(state['visits']),*(','.join(map(str,state[k])) for k in ('ticks','calls','initial_calls')),str(state['start'])]))
    if restart is not None:command.append(str(restart))
    result=subprocess.run(command,capture_output=True,text=True)
    if result.returncode:raise ValueError(result.stderr.strip())
    return json.loads(result.stdout)


def reference(dts,start,end,boundaries=()):
    b.set_device('runtime');b.start_scope();clocks=[b.Clock(dt=dt*b.second,name=f'visit_clock_{i}') for i,dt in enumerate(dts)]
    trace=[];net=b.Network()
    def callback(i):
        def record():trace.append([float(net.t/b.second),i,[int(c.variables['timestep'].get_value()[0]) for c in clocks],
                                 [float(c.variables['t'].get_value()[0]) for c in clocks]])
        return record
    for i,c in enumerate(clocks):net.add(b.NetworkOperation(callback(i),clock=c,when='start',name=f'visit_op_{i:03d}'))
    if start:net.run(start*b.second,namespace={});trace.clear()
    order=[clocks.index(c) for c in {obj.clock for obj in net.sorted_objects}]
    for stop in (*boundaries,end):net.run((stop-float(net.t))*b.second,namespace={})
    return dict(trace=trace,ticks=[int(c.variables['timestep'].get_value()[0]) for c in clocks],
                times=[float(c.variables['t'].get_value()[0]) for c in clocks]),order


DTS=[(.0002,),(.0002,.0002),(.0002,.0001,.0003),(.0002,.00015,.00033),
     (.0002,.00099999,.00049998),(.0002,.0004,.00040003),(.0002,.000200015,.00007),(.00040003,.0004,.0002),(.0002,.00099995,.00049996)]
@pytest.mark.parametrize('dts',DTS)
@pytest.mark.parametrize('start',[0.,.0003,.00065])
def test_all_clock_visits_match_real_brian(probe,dts,start):
    expected,order=reference(dts,start,.0021)
    actual=native(probe,dts,start,.0021,order)
    assert actual['trace']==expected['trace']
    assert actual['ticks']==expected['ticks'] and actual['times']==expected['times']
    # The comparison includes visits with no active main clock, not merely
    # the values sampled at the primary neuron's timestep.
    if min(dts)<dts[0]:assert any(row[1]!=0 and row[0]/dts[0]%1>.01 for row in actual['trace'])


@pytest.mark.parametrize('dts',DTS)
def test_cursor_checkpoint_across_processes(probe,dts):
    _,order=reference(dts,0.,.0021)
    whole=native(probe,dts,0.,.0021,order)
    head=native(probe,dts,0.,.0007,order)
    tail=native(probe,dts,0.,.0021,order,head)
    assert head['trace']+tail['trace']==whole['trace']
    assert tail['ticks']==whole['ticks'] and tail['visits']==whole['visits']
    bad=dict(head,ticks=[head['ticks'][0]+1,*head['ticks'][1:]])
    with pytest.raises(ValueError,match='checkpoint'):native(probe,dts,0.,.0021,order,bad)


@pytest.mark.parametrize('dts',DTS)
@pytest.mark.parametrize('split',[.0007,.001])
def test_brian_run_boundaries_realign_ticks_keep_distinct_draws(probe,dts,split):
    if dts==(.0002,.00099995,.00049996) and split==.001:
        # The actual repository's Clock.advance rejects coalescing a clock
        # whose own end index was reached, even when the earliest clock runs.
        with pytest.raises(StopIteration):reference(dts,0.,.0021,boundaries=(split,))
        for order in ([0,1,2],[1,0,2],[2,0,1]):
            with pytest.raises(ValueError,match='interval end'):native(probe,dts,0.,split,order)
        return
    expected,order=reference(dts,0.,.0021,boundaries=(split,))
    head=native(probe,dts,0.,split,order)
    tail=native(probe,dts,0.,.0021,order,head,restart=split)
    assert head['trace']+tail['trace']==expected['trace']
    assert tail['ticks']==expected['ticks'] and tail['times']==expected['times']
    draws=head['draws']+tail['draws']
    for i in range(len(dts)):
        calls=[r[2] for r in draws if r[0]==i]
        assert calls==list(range(len(calls))) and tail['calls'][i]==len(calls)
    # A new process can restore the re-aligned cursor, not just the initial one.
    continuation=native(probe,dts,0.,.0025,order,tail)
    assert all(a<=c for a,c in zip(tail['calls'],continuation['calls']))


def test_minimum_tie_order_is_semantically_significant(probe):
    dts=[.0002,.0004,.00040003]
    a=native(probe,dts,.0004,.00045,[0,1,2])
    c=native(probe,dts,.0004,.00045,[1,0,2])
    assert a['trace']!=c['trace']
    assert a['trace'][-1][0]==dts[2] and c['trace'][-1][0]==dts[1]


@pytest.mark.parametrize('order',[[0,0],[1],[0,2],[-1,0]])
def test_invalid_tie_orders_fail(probe,order):
    with pytest.raises(ValueError):native(probe,[.0002,.0003],0.,.001,order)


from brian2_rust import NativeLIFTrainer,lower_brian_dynamic_training
from test_native_training import RUNNER
from test_training_integer_ir import engine


def training_model(warm=0.,backend='cpu',dts=(.00040003,.0004,.0002)):
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='numpy'
    main=b.Clock(dt=dts[0]*b.second,name='clock_main');other=b.Clock(dt=dts[1]*b.second,name='clock_other');third=b.Clock(dt=dts[2]*b.second,name='clock_third')
    source=b.SpikeGeneratorGroup(1,[],[]*b.second,clock=main,name='clock_input')
    groups=[b.NeuronGroup(1,'dv/dt=g/ms:1\ng:1',threshold='v>100',reset='v=0',method='euler',clock=main,name=f'clock_group_{i}') for i in range(2)]
    syn=b.Synapses(*groups,'w:1\ng_post=t/ms:1 (summed)',clock=other,name='clock_syn');syn.connect();syn.w=.1
    monitor=b.StateMonitor(groups[0],'v',record=True,clock=third,name='clock_monitor');net=b.Network(source,*groups,syn,monitor)
    if warm:net.run(warm*b.second,namespace={})
    bundle=lower_brian_dynamic_training(net,input_group=source,layers=groups,backend=backend)
    return net,groups,bundle


@pytest.mark.parametrize('warm',[0.,.00065,.00080006001])
def test_frontend_clock_priority_matches_brian(engine,warm):
    net,groups,bundle=training_model(warm,engine);clock=bundle.plan['dynamic']['clocks']
    assert clock['order']==[clock['dts'].index(float(c.dt_)) for c in {obj.clock for obj in net.sorted_objects}]
    actual=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).evaluate(np.zeros((1,4,1)),[0])
    end=bundle.plan['clock']['origin']+4*bundle.plan['clock']['dt'];net.run((end-float(net.t))*b.second,namespace={})
    for g in groups:
        for name,indices in bundle.provenance['neuron_state_layout'][g.name].items():
            np.testing.assert_allclose(np.array(actual['final_state'])[0,indices],g.variables[name].get_value(),rtol=4e-5 if engine!='cpu' else 2e-13,atol=2e-7 if engine!='cpu' else 2e-14)
    if engine!='cpu':assert actual['gpu_dispatches']>0


@pytest.mark.parametrize('earliest,expected',[(.0004,.4),(.0002,.8)])
def test_clock_tie_priority_changes_native_sampling(engine,earliest,expected):
    _,groups,bundle=training_model(backend=engine);clock=bundle.plan['dynamic']['clocks']
    first=clock['dts'].index(earliest);clock['order']=[first,*[i for i in range(3) if i!=first]]
    actual=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).evaluate(np.zeros((1,2,1)),[0])
    index=bundle.provenance['neuron_state_layout'][groups[1].name]['g'][0]
    np.testing.assert_allclose(actual['final_state'][0][index],expected,rtol=4e-6 if engine!='cpu' else 2e-14)
    if engine!='cpu':assert actual['gpu_dispatches']>0


@pytest.mark.parametrize('order',[[0,0,1],[0,1],[0,1,3],[0,1,True]])
def test_invalid_native_clock_priority_is_transactional(order):
    import copy
    _,_,bundle=training_model();bundle.plan['dynamic']['clocks']['order']=order
    trainer=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER);before=copy.deepcopy(trainer.state)
    with pytest.raises(ValueError):trainer.step(np.zeros((1,2,1)),[0])
    assert trainer.state==before and trainer.clock_tick==0


def test_interval_overrun_matches_brian_and_is_transactional(engine):
    import copy
    net,_,bundle=training_model(backend=engine,dts=(.0002,.00099995,.00049996))
    trainer=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER);before=copy.deepcopy(trainer.state)
    with pytest.raises(ValueError,match='clock.*interval'):trainer.step(np.zeros((1,5,1)),[0])
    assert trainer.state==before and trainer.clock_tick==0 and trainer.neuron_state is None
    with pytest.raises(StopIteration):net.run(.001*b.second,namespace={})
