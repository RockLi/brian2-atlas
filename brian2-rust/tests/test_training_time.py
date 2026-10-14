"""Time-dependent v4 training: independent Brian, analytic and VJP checks."""
import copy
import json
import os
import subprocess
import sys

import brian2 as b
import numpy as np
import pytest

from brian2_rust import NativeLIFTrainer, lower_brian_training
from brian2_rust.training_brian import TrainingConversionError
from test_native_training import RUNNER
from test_native_training_gpu_mpi import compare


def model(method='euler', refractory=False, reset_only=False, units=False):
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='numpy'
    dt=.2*b.ms;dim='volt' if units else '1';scale=b.mV if units else 1.
    x=np.array([[1,0],[0,1],[1,1],[0,0],[1,0],[1,1],[0,1],[1,0],[0,0],[1,1],[1,0],[1,1]],float)
    ticks,ids=np.nonzero(x);source=b.SpikeGeneratorGroup(2,ids,ticks*dt,dt=dt,name='time_input')
    groups=[];synapses=[]
    for l in range(2):
        forcing='drive' if reset_only else 'drive*sin(t/tau)'
        eq=f'''dv/dt=(-v+.3*a+{forcing})/tau : {dim}{' (unless refractory)' if refractory else ''}
        da/dt=(gain*v-.4*a)/tau : {dim}
        tau : second (constant,shared)
        theta : {dim} (constant,shared)
        kick : {dim} (constant,shared)
        gain : 1 (constant,shared)
        drive : {dim} (constant,shared)'''
        g=b.NeuronGroup(2,eq,threshold='v>theta',reset='a+=kick*t/tau+.1*v\nv-=theta',
                       method=method,dt=dt,refractory=3*dt if refractory else False,name=f'time_layer_{l}')
        g.tau=(1+l*.1)*b.ms;g.theta=1.0625*scale;g.kick=.08*scale;g.gain=.15;g.drive=.7*scale
        g.v=np.array([.2,1.8])*scale;g.a=np.array([.1,.3])*scale;groups.append(g)
    for q,(src,dst) in enumerate([(source,groups[0]),(groups[0],groups[1]),(groups[1],groups[0])]):
        syn=b.Synapses(src,dst,'w:'+dim,on_pre='v_post+=w',name=f'time_syn_{q}')
        syn.connect();syn.w=(.4+.3*((np.asarray(syn.i)+2*np.asarray(syn.j)+q)%4))*scale;synapses.append(syn)
    net=b.Network(source,*groups,*synapses)
    requested={g.name:['tau','theta','kick','gain','drive'] for g in groups}
    return net,source,groups,x,dt,requested


def lower(net,source,groups,requested,**options):
    return lower_brian_training(net,input_group=source,layers=groups,trainable_neuron_parameters=requested,
                                learning_rate=1e-9,**options)


@pytest.mark.parametrize('method',['euler','rk2','rk4'])
@pytest.mark.parametrize('refractory',[False,True])
@pytest.mark.parametrize('reset_only',[False,True])
@pytest.mark.parametrize('warmup,units',[(0,False),(3,True)])
def test_time_forward_matches_brian(method,refractory,reset_only,warmup,units):
    net,source,groups,x,dt,requested=model(method,refractory,reset_only,units)
    if warmup:net.run(warmup*dt,namespace={})
    bundle=lower(net,source,groups,requested)
    assert bundle.plan['clock']==dict(origin=float(warmup*dt),dt=float(dt))
    result=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights).evaluate(x[None,warmup:],[0],initial=[bundle.initial_state])
    monitors=[b.SpikeMonitor(g) for g in groups];net.add(*monitors);net.run((len(x)-warmup)*dt,namespace={})
    spikes=np.zeros((len(x)-warmup,4));expected=[]
    for l,(g,m) in enumerate(zip(groups,monitors)):
        ticks=np.rint(np.asarray(m.t/b.second)/float(dt)).astype(int)-warmup
        spikes[ticks,l*2+np.asarray(m.i)]=1
        expected.extend(g.variables['v'].get_value());expected.extend(g.variables['a'].get_value())
        if refractory:
            from brian2.core.functions import timestep
            elapsed=timestep(float(g.clock.variables['t'].get_value()[0])-g.variables['lastspike'].get_value(),float(dt))
            expected.extend(np.maximum(3-elapsed,0))
    np.testing.assert_array_equal(result['spikes'][0],spikes)
    np.testing.assert_allclose(result['final_state'][0],expected,atol=3e-15,rtol=3e-12)
    assert result['final_tick']==len(x)-warmup


def oracle(bundle,w,x,initial,anchors=None,start_tick=0):
    """NumPy RK stages with time, plus a frozen local surrogate for FD VJPs."""
    p=bundle.plan;dt=p['clock']['dt'];live=initial.copy();old=[];pres=[];spikes=[];params=[]
    for name in bundle.provenance['layer_names']:
        row={}
        for binding in bundle.provenance['bindings']:
            if binding['object']==name:row.update(zip(binding['variables'],w[binding['bank']]))
        params.append(row)
    theta=np.repeat([q['theta'] for q in params],2);volt=np.array([0,1,4,5])
    for tick in range(len(x)):
        t=p['clock']['origin']+(start_tick+tick)*dt
        if anchors is not None and p['tbptt_window'] and tick and tick%p['tbptt_window']==0:live=anchors[2][tick].copy()
        old.append(live.copy());u=live.copy()
        for l,q in enumerate(params):
            z=live[4*l:4*l+4].reshape(2,2)
            def f(z,t):return np.array([(-z[0]+.3*z[1]+q['drive']*np.sin(t/q['tau']))/q['tau'],(q['gain']*z[0]-.4*z[1])/q['tau']])
            k1=f(z,t);method=bundle.provenance['integrators'][l]
            if method=='euler':next_z=z+dt*k1
            elif method=='rk2':next_z=z+dt*f(z+dt*k1/2,t+dt/2)
            else:
                k2=f(z+dt*k1/2,t+dt/2);k3=f(z+dt*k2/2,t+dt/2);k4=f(z+dt*k3,t+dt)
                next_z=z+dt*(k1+2*k2+2*k3+k4)/6
            u[4*l:4*l+4]=next_z.ravel()
        pre=u[volt].copy();s=(pre>theta).astype(float);gate=s.copy()
        if anchors is not None:
            phi=p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(anchors[0][tick]-anchors[3]))**2
            s=anchors[1][tick]+phi*(pre-theta-anchors[0][tick]+anchors[3])
            gate=anchors[1][tick] if p['detach_reset'] else s
        for proj,row in zip(p['projections'],w):
            src=proj['source_layer'];dst=proj['target_layer']-1
            inputs=x[tick] if src==0 else s[2*(src-1):2*src]
            for i,j,k in zip(proj['sources'],proj['targets'],proj['parameter_ids']):u[4*dst+j]+=inputs[i]*row[k]
        for l,q in enumerate(params):
            z=u[4*l:4*l+4].copy();reset=z.copy();reset[2:]+=q['kick']*t/q['tau']+.1*z[:2];reset[:2]-=q['theta']
            u[4*l:4*l+4]+=np.tile(gate[2*l:2*l+2],2)*(reset-z)
        live=u;pres.append(pre);spikes.append(s)
    spikes=np.array(spikes);logits=spikes[:,2:].mean(axis=0)*p['logit_scale']
    loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
    return loss,spikes,live,(np.array(pres),spikes,np.array(old),theta)


@pytest.mark.parametrize('method',['euler','rk2','rk4'])
@pytest.mark.parametrize('detach,window',[(True,None),(False,None),(True,3),(False,3)])
def test_time_independent_vjp(method,detach,window):
    net,source,groups,x,dt,requested=model(method);net.run(2*dt,namespace={})
    bundle=lower(net,source,groups,requested,detach_reset=detach,tbptt_window=window)
    w=bundle.weights;initial=np.array(bundle.initial_state);x=x[2:]
    result=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=w).gradients(x[None],[0],initial=initial[None],start_tick=2)
    loss,spikes,live,anchors=oracle(bundle,w,x,initial,start_tick=2)
    assert result['loss']==pytest.approx(loss,abs=2e-14)
    np.testing.assert_array_equal(result['spikes'][0],spikes)
    np.testing.assert_allclose(result['final_state'][0],live,atol=3e-14,rtol=3e-14)
    for bank,row in enumerate(w):
        for i,value in enumerate(row):
            eps=1e-6*max(abs(value),1e-3);plus=copy.deepcopy(w);minus=copy.deepcopy(w)
            plus[bank][i]+=eps;minus[bank][i]-=eps
            fd=(oracle(bundle,plus,x,initial,anchors,start_tick=2)[0]-oracle(bundle,minus,x,initial,anchors,start_tick=2)[0])/(2*eps)
            assert result['gradients'][bank][i]==pytest.approx(fd,rel=2e-4,abs=4e-6)
    for i in range(len(initial)):
        plus=initial.copy();minus=initial.copy();plus[i]+=1e-6;minus[i]-=1e-6
        fd=(oracle(bundle,w,x,plus,anchors,start_tick=2)[0]-oracle(bundle,w,x,minus,anchors,start_tick=2)[0])/2e-6
        assert result['initial_state_gradients'][0][i]==pytest.approx(fd,rel=2e-4,abs=8e-8)


@pytest.mark.parametrize('method',['euler','rk2','rk4'])
@pytest.mark.parametrize('backend,ranks',[('cpu',None),('cpu',2),('cpu',8),('metal',None),('metal',2),('metal',8),('cuda',None),('cuda',2),('cuda',8)])
def test_time_backends_carry_and_restore(method,backend,ranks,tmp_path):
    if ranks and os.environ.get('B2_TEST_MPI')!='1':pytest.skip('MPI required')
    if backend=='metal' and os.environ.get('B2_TEST_GPU')!='1':pytest.skip('Metal required')
    if backend=='cuda' and os.environ.get('B2_TEST_CUDA_TRAIN')!='1':pytest.skip('CUDA required')
    net,source,groups,x,dt,requested=model(method,refractory=True);net.run(2*dt,namespace={});x=x[2:]
    bundle=lower(net,source,groups,requested,detach_reset=False);p=copy.deepcopy(bundle.plan);p['backend']=backend
    if ranks:p['mpi_ranks']=ranks
    cpu=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights);target=NativeLIFTrainer(p,runner=RUNNER,weights=bundle.weights)
    for start,end in [(0,3),(3,6)]:
        initial=[bundle.initial_state] if start==0 else 'carry'
        a=cpu.step(x[None,start:end],[0],initial=initial);c=target.step(x[None,start:end],[0],initial=initial);compare(a,c)
        for key in ['final_state','initial_state_gradients']:np.testing.assert_allclose(a[key],c[key],atol=6e-6,rtol=4e-4)
        assert target.clock_tick==c['final_tick']==end
    checkpoint=tmp_path/'checkpoint';target.store(checkpoint)
    request=tmp_path/'request';result=tmp_path/'result';request.write_text(json.dumps(dict(plan=p,x=x[None,6:].tolist())))
    code='''import json,sys
from brian2_rust import NativeLIFTrainer
r=json.load(open(sys.argv[1]));t=NativeLIFTrainer(r['plan'],runner=sys.argv[4]);t.restore(sys.argv[2])
json.dump(t.step(r['x'],[0],initial='carry'),open(sys.argv[3],'w'))
'''
    subprocess.run([sys.executable,'-c',code,str(request),str(checkpoint),str(result),str(RUNNER)],check=True,timeout=120)
    assert json.loads(result.read_text())==target.step(x[None,6:],[0],initial='carry')
    # Frozen weights isolate continuity of state and absolute time across calls.
    p['trainable']=[False]*len(p['masks'])
    whole=NativeLIFTrainer(p,runner=RUNNER,weights=bundle.weights)
    parts=NativeLIFTrainer(p,runner=RUNNER,weights=bundle.weights)
    full=whole.step(x[None],[0],initial=[bundle.initial_state])
    first=parts.step(x[None,:4],[0],initial=[bundle.initial_state]);last=parts.step(x[None,4:],[0],initial='carry')
    assert first['spikes'][0]+last['spikes'][0]==full['spikes'][0]
    assert last['final_state']==full['final_state']
    assert parts.clock_tick==len(x)


def test_clock_fresh_manual_readonly_failure_and_topology(tmp_path):
    args=model();bundle=lower(args[0],args[1],args[2],args[5]);x=args[3][None]
    bundle.plan['trainable']=[False]*len(bundle.weights)
    trainer=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights)
    first=trainer.step(x[:,:3],[0],initial=[bundle.initial_state]);assert trainer.clock_tick==3
    before=copy.deepcopy((trainer.state,trainer.neuron_state,trainer.clock_tick,trainer.elapsed_ticks))
    for operation in ['evaluate','gradients']:
        a=trainer.execute(x[:,3:],[0],operation=operation,initial='carry')
        c=trainer.execute(x[:,3:],[0],operation=operation,initial=first['final_state'],start_tick=3)
        assert a==c and a['final_tick']==12
        assert (trainer.state,trainer.neuron_state,trainer.clock_tick,trainer.elapsed_ticks)==before
    for kwargs in [dict(start_tick=-1),dict(start_tick=True),dict(start_tick=2**53),dict(start_tick=1.5),dict(start_tick=1,initial='carry')]:
        with pytest.raises(ValueError):trainer.step(x,[0],**kwargs)
        assert (trainer.state,trainer.neuron_state,trainer.clock_tick,trainer.elapsed_ticks)==before
    assert trainer.step(x[:,:3],[0],initial=[bundle.initial_state])['final_state']==first['final_state']
    assert trainer.clock_tick==3 and trainer.elapsed_ticks==6
    trainer.plan['clock']['dt']*=2
    with pytest.raises(ValueError,match='topology'):trainer.step(x,[0])


@pytest.mark.parametrize('change',['missing_clock','negative_dt','infinite_dt','negative_origin','unknown_field','clock_overflow','clock_precision','start_without_clock'])
def test_clock_native_validation_is_atomic(change):
    args=model();bundle=lower(args[0],args[1],args[2],args[5]);p=copy.deepcopy(bundle.plan);tick=0
    if change=='missing_clock':del p['clock']
    elif change=='negative_dt':p['clock']['dt']=-1
    elif change=='infinite_dt':p['clock']['dt']=float('inf')
    elif change=='negative_origin':p['clock']['origin']=-1
    elif change=='unknown_field':p['clock']['extra']=1
    elif change=='clock_overflow':p['clock']['dt']=1e308
    elif change=='clock_precision':p['clock']['origin']=1e20
    elif change=='start_without_clock':
        del p['clock'];tick=1
        for layer in p['state_equations']+p['state_resets']:
            for program in layer:
                for node in program:
                    if node['op']=='time':node.update(op='constant',value=0.)
    if change=='infinite_dt':
        with pytest.raises(ValueError):NativeLIFTrainer(p,runner=RUNNER,weights=bundle.weights)
        return
    trainer=NativeLIFTrainer(p,runner=RUNNER,weights=bundle.weights);before=copy.deepcopy(trainer.state)
    with pytest.raises(ValueError):trainer.step(args[3][None],[0],initial=[bundle.initial_state],start_tick=tick)
    assert trainer.state==before and trainer.neuron_state is None and trainer.clock_tick==0


@pytest.mark.parametrize('backend',['metal','cuda'])
def test_gpu_clock_precision_failure_is_atomic(backend):
    if os.environ.get('B2_TEST_GPU' if backend=='metal' else 'B2_TEST_CUDA_TRAIN')!='1':pytest.skip('GPU required')
    args=model();bundle=lower(args[0],args[1],args[2],args[5]);bundle.plan['backend']=backend
    bundle.plan['clock']['origin']=1e5
    trainer=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights);before=copy.deepcopy(trainer.state)
    with pytest.raises(ValueError,match='clock'):trainer.step(args[3][None],[0],initial=[bundle.initial_state])
    assert trainer.state==before and trainer.clock_tick==0


def test_temporal_frontend_rejects_mismatched_clock_and_derived_override():
    args=model();net,source,groups,x,dt,requested=args
    groups[0].clock._set_t_update_dt(target_t=dt)
    with pytest.raises(TrainingConversionError,match='snapshot time'):lower(net,source,groups,requested)
    args=model()
    with pytest.raises(TrainingConversionError,match='derived model fields'):lower(args[0],args[1],args[2],args[5],clock=dict(origin=0,dt=1))


@pytest.mark.parametrize('method',['euler','rk2','rk4'])
@pytest.mark.parametrize('reset_only',[False,True])
def test_single_state_time_forces_v4_and_matches_closed_form(method,reset_only):
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='numpy'
    dt=.1*b.ms;source=b.SpikeGeneratorGroup(1,[],[]*b.ms,dt=dt);groups=[]
    for _ in range(2):
        g=b.NeuronGroup(1,'dv/dt=0/second:1' if reset_only else 'dv/dt=t/(ms*ms):1',
                        threshold='v>1' if reset_only else 'v>100',
                        reset='v=.3+t/ms' if reset_only else 'v=0',method=method,dt=dt)
        g.v=2 if reset_only else 0;groups.append(g)
    syn=b.Synapses(source,groups[0],'w:1',on_pre='v_post+=w');syn.connect();syn.w=0
    bundle=lower_brian_training(b.Network(source,*groups,syn),input_group=source,layers=groups)
    assert bundle.plan['schema']=='b2-state-training-plan-v4'
    trainer=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights)
    out=trainer.evaluate(np.zeros((1,3,1)),[0],initial=[bundle.initial_state],start_tick=4)
    expected=.7 if reset_only else (.15 if method=='euler' else .165)
    np.testing.assert_allclose(out['final_state'],[[expected,expected]],atol=2e-16,rtol=1e-14)


def test_checkpoint_invalid_clock_does_not_commit(tmp_path):
    import hashlib
    from brian2_rust.protocol import canonical_bytes
    args=model();bundle=lower(args[0],args[1],args[2],args[5]);trainer=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights)
    trainer.step(args[3][None,:3],[0],initial=[bundle.initial_state])
    path=tmp_path/'state';trainer.store(path);stored=json.loads(path.read_text())
    before=copy.deepcopy((trainer.state,trainer.neuron_state,trainer.clock_tick,trainer.elapsed_ticks))
    for tick in [None,-1,2**53+1,True,1.5]:
        payload=json.loads(stored['payload']);payload['clock_tick']=tick;raw=canonical_bytes(payload)
        envelope=dict(stored,payload=raw.decode(),sha256=hashlib.sha256(raw).hexdigest());path.write_bytes(canonical_bytes(envelope))
        with pytest.raises(ValueError,match='clock tick'):trainer.restore(path)
        assert (trainer.state,trainer.neuron_state,trainer.clock_tick,trainer.elapsed_ticks)==before
