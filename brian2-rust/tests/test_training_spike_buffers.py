"""Persistent threshold buffers and loss-bearing emissions on every visit."""
import copy
import os
import numpy as np
import pytest
import brian2 as b
from brian2_rust import NativeLIFTrainer
from brian2_rust.training import lif_training_plan
from brian2_rust.training_equations import neuron_parameter_bank
from brian2_rust.training_dynamic import compile_dynamic_transform,dynamic_action
from test_native_training import RUNNER
from test_training_integer_ir import engine
from test_training_linked import cython_cache


def model(dts=(.1,.3,.2),start=0.,window=None,backend='cpu',ranks=None,slow_read=False):
    origin=int(np.ceil(start/.2-1e-10))*.0002
    identity=[[[dict(op='state',index=0)]]]
    p=lif_training_plan([1,1,2],projections=[neuron_parameter_bank(5)],state_equations=identity*2,state_resets=identity*2,
        threshold=1.,detach_reset=False,clock=dict(origin=origin,dt=.0002),tbptt_window=window,backend=backend,mpi_ranks=ranks)
    clocks=[.0002]
    for dt in dts:
        if dt/1000 not in clocks:clocks.append(dt/1000)
    ids=[clocks.index(dt/1000) for dt in dts];programs=[];actions=[]
    def add(code,reads,owner,clock,params=None,gate=None):
        states={'v':0}
        if gate is not None:reads=[*reads,gate];states['buffer']=len(reads)-1
        compiled=compile_dynamic_transform(code,states=states,parameters=params)
        index=len(programs);programs.append(compiled['programs'])
        a=dynamic_action(compiled,reads,owner=owner,program_set=index,
            trigger=None if gate is None else dict(external=False,state=True,index=gate),detach_trigger=False)
        a['clock']=clock;actions.append(a)
    for j in range(3):add('v+=(drive-v)*h',[j],j,ids[j],dict(drive=(0,j),h=dts[j]))
    for j in range(3):actions.append(dict(owner=j,reads=[j],writes=[],program_set=None,threshold=j,trigger=None,clock=ids[j]))
    add('v+=w',[1],1,ids[0],dict(w=(0,3)),3)
    add('v+=w',[2],2,ids[0] if slow_read else ids[1],dict(w=(0,4)),4)
    for j in range(3):add('v-=.7',[j],j,ids[j],gate=3+j)
    p.update(schema='b2-dynamic-training-plan-v5',dynamic=dict(clocks=dict(start=start/1000,dts=clocks,epsilon=.0001),
        initial=[.8,1.1,.9,0.,1.,0.],initial_parameters=[None]*6,detached=[False]*6,
        voltage=[0,1,2],binary_states=[3,4,5],spike_buffers=[3,4,5],program_sets=programs,actions=actions))
    return p,[[8.2,1.7,1.3,.45,.35]],np.zeros((1,6,1)),ids


def oracle(p,w,dts,start=0.,initial=None,anchors=None,steps=6,slow_read=False):
    z=np.array(p['dynamic']['initial'] if initial is None else initial,float).copy()
    ticks=np.array([int(np.ceil(start/d-1e-10))*round(d*1000) for d in dts]+[int(np.ceil(start/.2-1e-10))*200])
    delta=np.array([round(d*1000) for d in dts]+[200]);end=ticks[3]+steps*200
    before=[];margins=[];spikes=[];frames=[];times=[];frame=0;main=0
    while ticks.min()<end:
        at=ticks.min();active=ticks==at
        if active[3]:frame=main;main+=1
        if anchors is not None and active[3] and p.get('tbptt_window') and frame and frame%p['tbptt_window']==0:z=anchors['before'][len(before)].copy()
        before.append(z.copy());s=np.zeros(3);margin=np.zeros(3)
        for j in range(3):
            if active[j]:z[j]+=(w[0][j]-z[j])*dts[j]
        for j in range(3):
            if active[j]:
                margin[j]=z[j]-1;s[j]=float(margin[j]>0)
                if anchors is not None:
                    a=anchors['margins'][len(before)-1][j];s[j]=float(a>0)+p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(a))**2*(margin[j]-a)
                if j==1 and len(z)>6:s[j]*=z[6]
                z[3+j]=s[j]
        if active[0]:z[1]+=w[0][3]*z[3]
        if active[0 if slow_read else 1]:z[2]+=w[0][4]*z[4]
        for j in range(3):
            if active[j]:z[j]-=.7*z[3+j]
        spikes.append(s);margins.append(margin);frames.append(frame);times.append(at);ticks[active]+=delta[active]
    counts=np.zeros((steps,3))
    for f,s in zip(frames,spikes):counts[f]+=s
    logits=counts[:,1:].sum(0)*p['logit_scale']/steps;m=logits.max();loss=m+np.log(np.exp(logits-m).sum())-logits[0]
    return loss,z,counts,np.array(spikes),dict(before=before,margins=margins,frames=frames,times=times)


@pytest.mark.parametrize('dts',[(.1,.3,.2),(.15,.37,.2)])
@pytest.mark.parametrize('window',[None,2])
@pytest.mark.parametrize('start',[0.,.05])
def test_buffered_all_visits_and_independent_vjp(engine,dts,window,start):
    p,w,x,ids=model(dts,start,window,engine,slow_read=True);result=NativeLIFTrainer(p,weights=w,runner=RUNNER).gradients(x,[0])
    loss,z,counts,events,anchors=oracle(p,w,dts,start,slow_read=True);tol=3e-3 if engine!='cpu' else 3e-6
    np.testing.assert_array_equal(result['spikes'][0],counts)
    np.testing.assert_array_equal(result['event_visits']['spikes'][0],events)
    assert result['event_visits']['frame_indices']==anchors['frames'];assert result['event_visits']['neuron_clocks']==ids
    np.testing.assert_allclose(result['final_state'][0],z,rtol=tol*.1,atol=2e-6)
    assert result['loss']==pytest.approx(loss,rel=tol)
    for j in range(5):
        hi=copy.deepcopy(w);lo=copy.deepcopy(w);hi[0][j]+=1e-6;lo[0][j]-=1e-6
        fd=(oracle(p,hi,dts,start,anchors=anchors,slow_read=True)[0]-oracle(p,lo,dts,start,anchors=anchors,slow_read=True)[0])/2e-6
        assert result['gradients'][0][j]==pytest.approx(fd,rel=tol,abs=tol*.01)
    for j in range(6):
        hi=np.array(p['dynamic']['initial']);lo=hi.copy();hi[j]+=1e-6;lo[j]-=1e-6
        fd=(oracle(p,w,dts,start,initial=hi,anchors=anchors,slow_read=True)[0]-oracle(p,w,dts,start,initial=lo,anchors=anchors,slow_read=True)[0])/2e-6
        assert result['initial_state_gradients'][0][j]==pytest.approx(fd,rel=tol,abs=tol*.01)
    if dts[0]==.1:assert counts.max()>1
    if start and window is None:assert abs(result['initial_state_gradients'][0][4])>1e-7
    if engine!='cpu':assert result['gpu_dispatches']>0


@pytest.mark.parametrize('start',[0.,.05])
def test_buffered_emissions_match_actual_brian(engine,start):
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='cython'
    dts=(.1,.3,.2);p,w,x,_=model(dts,start,backend=engine)
    inp=b.SpikeGeneratorGroup(1,[],[]*b.ms,dt=.2*b.ms)
    groups=[b.NeuronGroup(1,'dv/dt=(drive-v)/ms:1\ndrive:1',threshold='v>1',reset='v-=.7',method='euler',dt=dt*b.ms,name=f'buffer_n{j}') for j,dt in enumerate(dts)]
    for j,g in enumerate(groups):g.v=p['dynamic']['initial'][j];g.drive=w[0][j]
    syn=[]
    for j in range(2):
        s=b.Synapses(groups[j],groups[j+1],on_pre=f'v_post+={w[0][3+j]}',name=f'buffer_s{j}');s.connect();syn.append(s)
    net=b.Network(inp,*groups,*syn)
    if start:net.run(start*b.ms)
    p['dynamic']['initial']=[float(g.v[0]) for g in groups]+[float(g.variables['_spikespace'].get_value()[-1]>0) for g in groups]
    monitors=[b.SpikeMonitor(g) for g in groups];net.add(*monitors)
    result=NativeLIFTrainer(p,weights=w,runner=RUNNER).evaluate(x,[0])
    net.run((p['clock']['origin']+len(x[0])*.0002-float(net.t))*b.second)
    visits=result['event_visits'];all_events=np.array(visits['spikes'][0]);times=np.array(visits['clock_times'])
    for j,m in enumerate(monitors):
        np.testing.assert_allclose(times[all_events[:,j]>0,visits['neuron_clocks'][j]],np.asarray(m.t/b.second),rtol=0,atol=2e-15)
        assert result['final_state'][0][j]==pytest.approx(float(groups[j].v[0]),rel=2e-5,abs=2e-6)


@pytest.mark.parametrize('ranks',[2,8])
@pytest.mark.parametrize('window',[None,2])
def test_buffered_mpi_carry_and_checkpoint(engine,ranks,window,tmp_path):
    if os.environ.get('B2_TEST_MPI')!='1':pytest.skip('local MPI required')
    p,w,x,_=model(window=window,backend=engine,ranks=ranks,slow_read=True);p['trainable']=[False];x=np.repeat(x,2,axis=0)
    t=NativeLIFTrainer(p,weights=w,runner=RUNNER);whole=t.gradients(x,[0,1])
    cp=copy.deepcopy(p);cp.update(backend='cpu',mpi_ranks=None)
    ref=NativeLIFTrainer(cp,weights=w,runner=RUNNER).gradients(x,[0,1])
    for key in ('spikes','final_state','initial_state_gradients'):np.testing.assert_allclose(whole[key],ref[key],rtol=.003,atol=3e-6)
    np.testing.assert_allclose(whole['gradients'],ref['gradients'],rtol=.003,atol=3e-6)
    t.step(x[:,:2],[0,1]);path=tmp_path/'buffer.json';t.store(path)
    new=NativeLIFTrainer(p,runner=RUNNER);new.restore(path);tail=new.gradients(x[:,2:],[0,1],initial='carry')
    np.testing.assert_allclose(tail['final_state'],whole['final_state'],rtol=3e-5,atol=3e-6)
    np.testing.assert_array_equal(tail['spikes'],np.array(whole['spikes'])[:,2:])


@pytest.mark.parametrize('bad',['alias','detached','write','direct','missing'])
def test_buffer_admission(bad):
    p,w,x,_=model();d=p['dynamic']
    if bad=='alias':d['spike_buffers'][1]=d['spike_buffers'][0]
    elif bad=='detached':d['detached'][3]=True
    elif bad=='write':d['actions'][-1]['writes']=[5];d['actions'][-1]['reads'][0]=5
    elif bad=='direct':d['actions'][-1]['trigger']=dict(external=False,index=2)
    else:d.pop('spike_buffers')
    with pytest.raises((ValueError,RuntimeError)):NativeLIFTrainer(p,weights=w,runner=RUNNER).gradients(x,[0])


def test_inactive_threshold_overwrites_buffer_and_its_adjoint(engine):
    dts=(.1,.3,.2);start=.05
    p,w,x,_=model(dts,start,backend=engine,slow_read=True);d=p['dynamic']
    d['initial'].append(0.);d['initial_parameters'].append(None);d['detached'].append(True);d['binary_states'].append(6)
    next(a for a in d['actions'] if a.get('threshold')==1)['reads'].append(6)
    result=NativeLIFTrainer(p,weights=w,runner=RUNNER).gradients(x,[0])
    loss,z,counts,events,anchors=oracle(p,w,dts,start,slow_read=True)
    np.testing.assert_array_equal(result['event_visits']['spikes'][0],events)
    np.testing.assert_array_equal(result['spikes'][0],counts)
    assert events[:,1].sum()==0
    assert result['initial_state_gradients'][0][1]==0
    assert result['initial_state_gradients'][0][6]==0
    assert abs(result['initial_state_gradients'][0][4])>1e-7
    for j in range(6):
        hi=np.array(d['initial']);lo=hi.copy();hi[j]+=1e-6;lo[j]-=1e-6
        fd=(oracle(p,w,dts,start,initial=hi,anchors=anchors,slow_read=True)[0]-oracle(p,w,dts,start,initial=lo,anchors=anchors,slow_read=True)[0])/2e-6
        assert result['initial_state_gradients'][0][j]==pytest.approx(fd,rel=.003,abs=3e-6)


@pytest.mark.parametrize('same_clock',[False,True])
def test_buffered_margin_clock_admission(engine,same_clock):
    p,w,x,ids=model(backend=engine);reference=NativeLIFTrainer(p,weights=w,runner=RUNNER).gradients(x,[0]);d=p['dynamic']
    d['initial'].append(0.);d['initial_parameters'].append(None);d['detached'].append(False)
    index=len(d['program_sets']);d['program_sets'].append([[dict(op='state',index=0),dict(op='constant',value=1.),dict(op='sub',left=0,right=1)]])
    a=next(a for a in d['actions'] if a.get('threshold')==0);where=d['actions'].index(a)
    a['reads']=[6];a['threshold_margin']=True
    d['actions'].insert(where,dict(owner=0,clock=ids[0] if same_clock else 0,reads=[0,6],writes=[6],program_set=index,threshold=None,trigger=None))
    if not same_clock:
        with pytest.raises((ValueError,RuntimeError)):NativeLIFTrainer(p,weights=w,runner=RUNNER).gradients(x,[0])
    else:
        actual=NativeLIFTrainer(p,weights=w,runner=RUNNER).gradients(x,[0])
        np.testing.assert_allclose(actual['gradients'],reference['gradients'],rtol=3e-5,atol=3e-7)
        np.testing.assert_array_equal(actual['event_visits']['spikes'],reference['event_visits']['spikes'])
