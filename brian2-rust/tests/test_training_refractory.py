"""Fixed-duration refractory: independent Brian and local-surrogate oracles."""
import copy
import json
import os
import subprocess
import sys

import brian2 as b
from brian2.core.functions import timestep
import numpy as np
import pytest

from brian2_rust import NativeLIFTrainer, lower_brian_training
from brian2_rust.training_brian import TrainingConversionError
from test_native_training import RUNNER
from test_native_training_gpu_mpi import compare


def model(clamp=('v',), duration=3., units=False, method='euler'):
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='numpy'
    dt=.2*b.ms;scale=b.mV if units else 1.;dim='volt' if units else '1'
    x=np.array([[1,0],[0,1],[1,1],[0,0],[1,0],[1,1],[0,1],[1,0],[0,0],[1,1],[1,0],[1,1]],float)
    ticks,ids=np.nonzero(x);source=b.SpikeGeneratorGroup(2,ids,ticks*dt,dt=dt,name='ref_input')
    groups=[]
    for l,name in enumerate(['ref_hidden','ref_output']):
        vflag=' (unless refractory)' if 'v' in clamp else ''
        aflag=' (unless refractory)' if 'a' in clamp else ''
        eq=f'''dv/dt=(-v+.3*a)/tau : {dim}{vflag}
        da/dt=(.15*v-.4*a)/tau : {dim}{aflag}
        tau : second (constant,shared)
        theta : {dim} (constant,shared)
        kick : {dim} (constant,shared)'''
        g=b.NeuronGroup(2,eq,threshold='v>theta',reset='a+=kick+.1*v\nv-=theta',
                        refractory=duration*dt,method=method[l] if isinstance(method,tuple) else method,dt=dt,name=name)
        g.tau=(1+l*.1)*b.ms;g.theta=1.0625*scale;g.kick=.07*scale
        g.v=np.array([.2,1.8])*scale;g.a=np.array([.1,.3])*scale;groups.append(g)
    synapses=[]
    for q,(src,dst) in enumerate([(source,groups[0]),(groups[0],groups[1]),(groups[1],groups[0])]):
        syn=b.Synapses(src,dst,'w:'+dim,on_pre='v_post+=w',name=f'ref_syn_{q}')
        syn.connect(i=[1,0,1,0],j=[0,0,1,1]);syn.w=np.array([1.2,.4,.3,1.1])*scale;synapses.append(syn)
    return b.Network(source,*groups,*synapses),source,groups,x,dt


def lower(net,source,groups,**options):
    return lower_brian_training(net,input_group=source,layers=groups,
        trainable_neuron_parameters={g.name:['tau','theta','kick'] for g in groups},learning_rate=1e-8,**options)


@pytest.mark.parametrize('clamp',[(),('v',),('a',),('v','a')])
@pytest.mark.parametrize('duration',[0.,1.,3.,3.5])
@pytest.mark.parametrize('warmup,units',[(0,False),(3,True)])
def test_refractory_forward_matches_brian(clamp,duration,warmup,units,method='euler'):
    net,source,groups,x,dt=model(clamp,duration,units,method)
    if warmup:net.run(warmup*dt,namespace={})
    bundle=lower(net,source,groups)
    assert bundle.provenance['state_names']==[['v','a','__refractory_ticks']]*2
    result=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights).evaluate(x[None,warmup:],[0],initial=[bundle.initial_state])
    monitors=[b.SpikeMonitor(g) for g in groups];net.add(*monitors);net.run((len(x)-warmup)*dt,namespace={})
    spikes=np.zeros((len(x)-warmup,4));expected=[]
    for l,(g,m) in enumerate(zip(groups,monitors)):
        ticks=np.rint(np.asarray(m.t/b.second)/float(dt/b.second)).astype(int)-warmup
        spikes[ticks,2*l+np.asarray(m.i)]=1
        elapsed=timestep(float(g.clock.variables['t'].get_value()[0])-g.variables['lastspike'].get_value(),float(dt))
        expected.extend([*g.variables['v'].get_value(),*g.variables['a'].get_value(),*np.maximum(int(timestep(float(duration*dt),float(dt)))-elapsed,0)])
    np.testing.assert_array_equal(result['spikes'][0],spikes)
    np.testing.assert_allclose(result['final_state'][0],expected,rtol=5e-13,atol=2e-15)


def oracle(bundle,w,x,initial,anchors=None):
    p=bundle.plan;dt=bundle.provenance['dt_seconds'];live=initial.copy()
    banks=[binding['bank'] for binding in bundle.provenance['bindings'] if binding['kind']=='neuron']
    kick,tau,theta=np.asarray([w[i] for i in banks]).T;theta=np.repeat(theta,2)
    starts=[];pres=[];spikes=[];active_rows=[]
    for t in range(len(x)):
        if anchors is not None and p['tbptt_window'] and t and t%p['tbptt_window']==0:live=anchors[2][t].copy()
        starts.append(live.copy());active=live[[4,5,10,11]]==0
        if anchors is not None:active=anchors[4][t]
        active_rows.append(active.copy());u=live.copy()
        for l,offset in enumerate([0,6]):
            z=live[offset:offset+4].reshape(2,2)
            def f(z):
                dz=np.array([(-z[0]+.3*z[1])/tau[l],(.15*z[0]-.4*z[1])/tau[l]])
                for s in p['refractory'][l]['clamp']:dz[s]*=active[2*l:2*l+2]
                return dz
            method=bundle.provenance['integrators'][l];k1=f(z)
            if method=='euler':next_z=z+dt*k1
            elif method=='rk2':next_z=z+dt*f(z+dt*k1/2)
            else:
                assert method=='rk4'
                k2=f(z+dt*k1/2);k3=f(z+dt*k2/2);k4=f(z+dt*k3)
                next_z=z+dt*(k1+2*k2+2*k3+k4)/6
            u[offset:offset+4]=next_z.ravel();u[offset+4:offset+6]=np.maximum(live[offset+4:offset+6]-1,0)
        pre=u[[0,1,6,7]].copy();hard=((pre>theta)&active).astype(float);s=hard.copy();gate=hard.copy()
        if anchors is not None:
            phi=p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(anchors[0][t]-anchors[3]))**2
            s=anchors[1][t]+active*phi*(pre-theta-anchors[0][t]+anchors[3])
            gate=anchors[1][t] if p['detach_reset'] else s;hard=anchors[1][t]
        for projection,wrow in zip(p['projections'][:3],w[:3]):
            src=projection['source_layer'];dst=projection['target_layer']-1;offset=6*dst
            inputs=x[t] if src==0 else s[2*(src-1):2*src]
            for i,j,k in zip(projection['sources'],projection['targets'],projection['parameter_ids']):
                if 0 not in p['refractory'][dst]['clamp'] or (active[2*dst+j] and hard[2*dst+j]==0):u[offset+j]+=inputs[i]*wrow[k]
        for l,offset in enumerate([0,6]):
            reset=u[offset:offset+4].copy();reset[2:]+=kick[l]+.1*reset[:2];reset[:2]-=theta[2*l:2*l+2]
            u[offset:offset+4]+=np.tile(gate[2*l:2*l+2],2)*(reset-u[offset:offset+4])
            u[offset+4:offset+6]=np.where(hard[2*l:2*l+2]!=0,max(p['refractory'][l]['steps']-1,0),u[offset+4:offset+6])
        live=u;pres.append(pre);spikes.append(s)
    spikes=np.array(spikes);logits=spikes[:,2:].mean(axis=0)*p['logit_scale']
    loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
    return loss,spikes,live,(np.array(pres),spikes,np.array(starts),theta,np.array(active_rows))


@pytest.mark.parametrize('clamp',[(),('v',),('a',),('v','a')])
@pytest.mark.parametrize('detach,window',[(True,None),(False,None),(True,3),(False,3)])
def test_refractory_independent_vjp(clamp,detach,window,method='euler'):
    net,source,groups,x,dt=model(clamp,method=method);bundle=lower(net,source,groups,detach_reset=detach,tbptt_window=window)
    w=bundle.weights;initial=np.array(bundle.initial_state)
    result=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=w).gradients(x[None],[0],initial=initial[None])
    loss,spikes,live,anchors=oracle(bundle,w,x,initial)
    assert result['loss']==pytest.approx(loss,abs=2e-15)
    np.testing.assert_array_equal(result['spikes'][0],spikes)
    np.testing.assert_allclose(result['final_state'][0],live,rtol=2e-14,atol=2e-14)
    for bank,row in enumerate(w):
        for i,value in enumerate(row):
            eps=1e-6*max(abs(value),1e-3);plus=copy.deepcopy(w);minus=copy.deepcopy(w)
            plus[bank][i]+=eps;minus[bank][i]-=eps
            fd=(oracle(bundle,plus,x,initial,anchors)[0]-oracle(bundle,minus,x,initial,anchors)[0])/(2*eps)
            assert result['gradients'][bank][i]==pytest.approx(fd,abs=3e-6,rel=1e-4)
    for i in range(len(initial)):
        if i in [4,5,10,11]:assert result['initial_state_gradients'][0][i]==0;continue
        plus=initial.copy();minus=initial.copy();plus[i]+=1e-6;minus[i]-=1e-6
        fd=(oracle(bundle,w,x,plus,anchors)[0]-oracle(bundle,w,x,minus,anchors)[0])/2e-6
        assert result['initial_state_gradients'][0][i]==pytest.approx(fd,abs=5e-8,rel=1e-4)


@pytest.mark.parametrize('backend,ranks',[('cpu',2),('cpu',8),('metal',None),('metal',2),('metal',8),('cuda',None),('cuda',2),('cuda',8)])
@pytest.mark.parametrize('clamp',[(),('v','a')])
def test_refractory_backends_carry_and_fresh_restore(backend,ranks,clamp,tmp_path,method='euler'):
    if ranks and os.environ.get('B2_TEST_MPI')!='1':pytest.skip('MPI required')
    if backend=='metal' and os.environ.get('B2_TEST_GPU')!='1':pytest.skip('Metal required')
    if backend=='cuda' and os.environ.get('B2_TEST_CUDA_TRAIN')!='1':pytest.skip('CUDA required')
    net,source,groups,x,dt=model(clamp,method=method);bundle=lower(net,source,groups,detach_reset=False)
    p=copy.deepcopy(bundle.plan);p['backend']=backend
    if ranks:p['mpi_ranks']=ranks
    cpu=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights)
    target=NativeLIFTrainer(p,runner=RUNNER,weights=bundle.weights)
    for start,end in [(0,3),(3,7)]:
        initial=[bundle.initial_state] if start==0 else 'carry'
        a=cpu.step(x[None,start:end],[0],initial=initial);c=target.step(x[None,start:end],[0],initial=initial)
        compare(a,c)
        for key in ['final_state','initial_state_gradients']:np.testing.assert_allclose(a[key],c[key],atol=6e-6,rtol=4e-4)
        assert np.count_nonzero(np.array(c['final_state'])[:,[4,5,10,11]])>0
    checkpoint=tmp_path/'checkpoint';target.store(checkpoint)
    request=tmp_path/'request.json';result=tmp_path/'result.json'
    request.write_text(json.dumps(dict(plan=p,x=x[None,7:].tolist())))
    code='''import json,sys
from brian2_rust import NativeLIFTrainer
r=json.load(open(sys.argv[1]));t=NativeLIFTrainer(r['plan'],runner=sys.argv[4]);t.restore(sys.argv[2])
json.dump(t.step(r['x'],[0],initial='carry'),open(sys.argv[3],'w'))
'''
    subprocess.run([sys.executable,'-c',code,str(request),str(checkpoint),str(result),str(RUNNER)],check=True,timeout=120)
    assert json.loads(result.read_text())==target.step(x[None,7:],[0],initial='carry')


@pytest.mark.parametrize('change',['negative','conditional','legacy','counter','program'])
def test_refractory_rejections(change):
    net,source,groups,x,dt=model()
    if change=='negative':groups[0]._refractory=-dt
    elif change=='conditional':groups[0]._refractory='v>theta'
    if change=='legacy':b.prefs.legacy.refractory_timing=True
    try:
        if change in ('negative','conditional','legacy'):
            with pytest.raises(TrainingConversionError,match='refractory'):lower(net,source,groups)
        else:
            bundle=lower(net,source,groups);initial=np.array([bundle.initial_state])
            if change=='counter':initial[0,4]=.5
            else:bundle.plan['state_equations'][0][0]=[dict(op='state',index=2)]
            trainer=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights)
            before=copy.deepcopy(trainer.state)
            with pytest.raises(ValueError,match='refractory'):trainer.step(x[None],[0],initial=initial)
            assert trainer.state==before and trainer.neuron_state is None
    finally:b.prefs.legacy.refractory_timing=False
