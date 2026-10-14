"""Native SDE training: common-noise Brian reference and pathwise VJP."""
import ast
import copy
import os
from unittest.mock import patch

import brian2 as b
import numpy as np
import pytest

from brian2_rust import NativeLIFTrainer, lower_brian_training
from brian2_rust.training_brian import TrainingConversionError
from brian2_rust.training import lif_training_plan
from brian2_rust.training_equations import neuron_parameter_bank
from test_native_training import RUNNER
from test_native_training_gpu_mpi import compare

MASK=(1<<64)-1

def normal(seed,sequence,batch,layer,neuron,tick,stream):
    """Independent integer reference for the published counter-address format."""
    def mix(x):
        x=((x^(x>>30))*0xbf58476d1ce4e5b9)&MASK;x=((x^(x>>27))*0x94d049bb133111eb)&MASK
        return x^(x>>31)
    key=mix(seed^0x42325344454e3031)
    for x in [sequence,batch,layer,neuron,tick,stream]:key=mix(key^mix((x+0x9e3779b97f4a7c15)&MASK))
    u=((mix(key^0xa0761d6478bd642f)>>11)+.5)/2**53
    v=((mix(key^0xe7037ed1a0b428db)>>11)+.5)/2**53
    return np.sqrt(-2*np.log(u))*np.cos(2*np.pi*v)


def model(method='euler',shared=False,refractory=False,units=False):
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='numpy'
    dt=.2*b.ms;dim='volt' if units else '1';scale=b.mV if units else 1.
    x=np.array([[1,0],[0,1],[1,1],[0,0],[1,0],[1,1],[0,1],[1,0],[0,0],[1,1],[1,0],[1,1]],float)
    ticks,ids=np.nonzero(x);source=b.SpikeGeneratorGroup(2,ids,ticks*dt,dt=dt,name='sde_input')
    groups=[];synapses=[];noise_v='xi_common' if shared else 'xi_v';noise_a='xi_common' if shared else 'xi_a'
    for l in range(2):
        gain_v='1' if method=='euler' else '(v/unit+1)';gain_a='1' if method=='euler' else '(a/unit+1)'
        extra_v='+.2*sigma*(a/unit+1)*xi_extra/sqrt(tau)' if shared=='mixed' else ''
        extra_a='+.1*sigma*(v/unit+1)*xi_extra/sqrt(tau)' if shared=='mixed' else ''
        eq=f'''dv/dt=(-v+.3*a+drive*sin(t/tau))/tau+sigma*{gain_v}*{noise_v}/sqrt(tau){extra_v} : {dim}{' (unless refractory)' if refractory else ''}
        da/dt=(gain*v-.4*a)/tau+sigma*{gain_a}*{noise_a}/sqrt(tau){extra_a} : {dim}
        tau : second (constant,shared)
        theta : {dim} (constant,shared)
        kick : {dim} (constant,shared)
        gain : 1 (constant,shared)
        drive : {dim} (constant,shared)
        sigma : {dim} (constant,shared)
        unit : {dim} (constant,shared)'''
        g=b.NeuronGroup(2,eq,threshold='v>theta',reset='a+=kick+.1*v\nv-=theta',method=method,dt=dt,
                       refractory=3*dt if refractory else False,name=f'sde_layer_{l}')
        g.tau=(1+l*.1)*b.ms;g.theta=1.0625*scale;g.kick=.08*scale;g.gain=.15;g.drive=.7*scale;g.sigma=.13*scale;g.unit=scale
        g.v=np.array([.2,1.8])*scale;g.a=np.array([.1,.3])*scale;groups.append(g)
    for q,(src,dst) in enumerate([(source,groups[0]),(groups[0],groups[1]),(groups[1],groups[0])]):
        syn=b.Synapses(src,dst,'w:'+dim,on_pre='v_post+=w',name=f'sde_syn_{q}')
        syn.connect();syn.w=(.4+.3*((np.asarray(syn.i)+2*np.asarray(syn.j)+q)%4))*scale;synapses.append(syn)
    return b.Network(source,*groups,*synapses),source,groups,x,dt


def lower(net,source,groups,**options):
    return lower_brian_training(net,input_group=source,layers=groups,seed=7123,learning_rate=1e-9,
                               trainable_neuron_parameters={g.name:['tau','theta','kick','gain','drive','sigma'] for g in groups},**options)


@pytest.mark.parametrize('method,shared',[('euler',False),('euler',True),('heun',False),('heun',True),('heun','mixed'),('milstein',False)])
@pytest.mark.parametrize('refractory',[False,True])
@pytest.mark.parametrize('warmup,units',[(0,False),(2,True)])
def test_sde_forward_matches_brian_common_noise(method,shared,refractory,warmup,units):
    net,source,groups,x,dt=model(method,shared,refractory,units)
    if warmup:net.run(warmup*dt,namespace={})
    bundle=lower(net,source,groups);sequence=17
    result=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights).evaluate(x[None,warmup:],[0],initial=[bundle.initial_state],noise_sequence=sequence)
    # Brian's own generated updater determines draw order. Replace only the
    # external normal sampler; retain all Brian integration/event execution.
    net.run(0*dt,namespace={});order=[]
    for l,g in enumerate(groups):
        code=ast.parse(g.state_updater.abstract_code)
        for statement in code.body:
            if isinstance(statement,ast.Assign) and any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='randn' for n in ast.walk(statement.value)):
                order.append((l,bundle.provenance['noise_names'][l].index(statement.targets[0].id)))
    draws=iter([np.array([normal(7123,sequence,0,l,j,t,s) for j in range(2)]) for t in range(len(x)-warmup) for l,s in order])
    def randn(n):
        values=next(draws);assert len(values)==n;return values
    monitors=[b.SpikeMonitor(g) for g in groups];net.add(*monitors)
    with patch('numpy.random.randn',randn):net.run((len(x)-warmup)*dt,namespace={})
    with pytest.raises(StopIteration):next(draws)
    expected=[];spikes=np.zeros((len(x)-warmup,4))
    for l,(g,m) in enumerate(zip(groups,monitors)):
        ticks=np.rint(np.asarray(m.t/b.second)/float(dt)).astype(int)-warmup;spikes[ticks,2*l+np.asarray(m.i)]=1
        expected.extend(g.variables['v'].get_value());expected.extend(g.variables['a'].get_value())
        if refractory:
            from brian2.core.functions import timestep
            elapsed=timestep(float(g.clock.variables['t'].get_value()[0])-g.variables['lastspike'].get_value(),float(dt));expected.extend(np.maximum(3-elapsed,0))
    np.testing.assert_array_equal(result['spikes'][0],spikes)
    np.testing.assert_allclose(result['final_state'][0],expected,atol=4e-15,rtol=5e-12)
    assert result['noise_sequence']==sequence


def oracle(bundle,w,x,initial,anchors=None,sequence=0,start_tick=0):
    p=bundle.plan;dt=p['clock']['dt'];live=initial.copy();old=[];pres=[];spikes=[];params=[]
    for name in bundle.provenance['layer_names']:
        row={}
        for bind in bundle.provenance['bindings']:
            if bind['object']==name:row.update(zip(bind['variables'],w[bind['bank']]))
        params.append(row)
    theta=np.repeat([q['theta'] for q in params],2);volt=np.array([0,1,4,5])
    for tick in range(len(x)):
        t=p['clock']['origin']+(start_tick+tick)*dt
        if anchors is not None and p['tbptt_window'] and tick and tick%p['tbptt_window']==0:live=anchors[2][tick].copy()
        old.append(live.copy());u=live.copy()
        for l,q in enumerate(params):
            z=live[4*l:4*l+4].reshape(2,2);method=bundle.provenance['integrators'][l];streams=bundle.provenance['noise_names'][l]
            drift=np.array([(-z[0]+.3*z[1]+q['drive']*np.sin(t/q['tau']))/q['tau'],(q['gain']*z[0]-.4*z[1])/q['tau']])
            next_z=z+dt*drift
            for s,name in enumerate(streams):
                dw=np.sqrt(dt)*np.array([normal(p['seed'],sequence,0,l,j,start_tick+tick,s) for j in range(2)])
                mask=np.array([name in ('xi_v','xi_common'),name in ('xi_a','xi_common')])[:,None]
                def g(z):
                    if name=='xi_extra':return q['sigma']/np.sqrt(q['tau'])*np.array([.2*(z[1]+1),.1*(z[0]+1)])
                    return mask*q['sigma']/np.sqrt(q['tau'])*(np.ones_like(z) if method=='euler' else z+1)
                base=g(z)
                if method=='euler':next_z=next_z+base*dw
                elif method=='heun':next_z=next_z+.5*dw*(base+g(z+base*dw))
                else:next_z=next_z+base*dw+(g(z+dt*drift+np.sqrt(dt)*base)-base)*dw**2/(2*np.sqrt(dt))
            u[4*l:4*l+4]=next_z.ravel()
        pre=u[volt].copy();s=(pre>theta).astype(float);gate=s.copy()
        if anchors is not None:
            phi=p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(anchors[0][tick]-anchors[3]))**2
            s=anchors[1][tick]+phi*(pre-theta-anchors[0][tick]+anchors[3]);gate=anchors[1][tick] if p['detach_reset'] else s
        for proj,row in zip(p['projections'],w):
            src=proj['source_layer'];dst=proj['target_layer']-1;inputs=x[tick] if src==0 else s[2*(src-1):2*src]
            for i,j,k in zip(proj['sources'],proj['targets'],proj['parameter_ids']):u[4*dst+j]+=inputs[i]*row[k]
        for l,q in enumerate(params):
            z=u[4*l:4*l+4].copy();reset=z.copy();reset[2:]+=q['kick']+.1*z[:2];reset[:2]-=q['theta']
            u[4*l:4*l+4]+=np.tile(gate[2*l:2*l+2],2)*(reset-z)
        live=u;pres.append(pre);spikes.append(s)
    spikes=np.array(spikes);logits=spikes[:,2:].mean(axis=0)*p['logit_scale']
    loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
    return loss,spikes,live,(np.array(pres),spikes,np.array(old),theta)


@pytest.mark.parametrize('method,shared',[('euler',True),('heun',True),('heun','mixed'),('milstein',False)])
@pytest.mark.parametrize('detach,window',[(True,None),(False,None),(True,3),(False,3)])
def test_sde_independent_pathwise_vjp(method,shared,detach,window):
    net,source,groups,x,dt=model(method,shared);bundle=lower(net,source,groups,detach_reset=detach,tbptt_window=window)
    w=bundle.weights;initial=np.array(bundle.initial_state);sequence=9;start_tick=4
    result=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=w).gradients(x[None],[0],initial=initial[None],noise_sequence=sequence,start_tick=start_tick)
    def run(w,z,anchors=None):return oracle(bundle,w,x,z,anchors,sequence,start_tick)
    loss,spikes,live,anchors=run(w,initial)
    assert result['loss']==pytest.approx(loss,abs=2e-14);np.testing.assert_array_equal(result['spikes'][0],spikes)
    np.testing.assert_allclose(result['final_state'][0],live,atol=4e-14,rtol=4e-14)
    for bank,row in enumerate(w):
        for i,value in enumerate(row):
            eps=1e-6*max(abs(value),1e-3);plus=copy.deepcopy(w);minus=copy.deepcopy(w);plus[bank][i]+=eps;minus[bank][i]-=eps
            fd=(run(plus,initial,anchors)[0]-run(minus,initial,anchors)[0])/(2*eps)
            assert result['gradients'][bank][i]==pytest.approx(fd,rel=3e-4,abs=5e-6)
    for i in range(len(initial)):
        plus=initial.copy();minus=initial.copy();plus[i]+=1e-6;minus[i]-=1e-6
        fd=(run(w,plus,anchors)[0]-run(w,minus,anchors)[0])/2e-6
        assert result['initial_state_gradients'][0][i]==pytest.approx(fd,rel=3e-4,abs=1e-7)


def raw_noise_plan(backend='cpu',ranks=None):
    programs=[[[dict(op='noise',stream=0)],[dict(op='noise',stream=1)]]] * 2
    identity=[[[dict(op='state',index=0)],[dict(op='state',index=1)]]] * 2
    return lif_training_plan([1,2,2],projections=[neuron_parameter_bank(1)],state_equations=programs,state_resets=identity,
                             clock=dict(origin=0.,dt=.001),noise_streams=[2,2],threshold=10000.,backend=backend,mpi_ranks=ranks,
                             seed=918273,trainable=[False])


def test_native_noise_address_distribution_and_independence():
    trainer=NativeLIFTrainer(raw_noise_plan(),runner=RUNNER,weights=[[0.]])
    observed=np.array(trainer.evaluate(np.zeros((4096,1,1)),np.zeros(4096,int),noise_sequence=7,start_tick=13)['final_state'])
    for batch in [0,1,23,4095]:
        expected=[normal(918273,7,batch,l,j,13,s) for l in range(2) for s in range(2) for j in range(2)]
        np.testing.assert_allclose(observed[batch],expected,rtol=1e-14,atol=2e-15)
    assert abs(observed.mean())<.02 and abs(observed.var()-1)<.035
    assert np.max(np.abs(np.corrcoef(observed.T)-np.eye(8)))<.055
    assert len(np.unique(observed))==observed.size
    for sequence,tick in [(8,13),(7,14)]:
        other=trainer.evaluate(np.zeros((4096,1,1)),np.zeros(4096,int),noise_sequence=sequence,start_tick=tick)['final_state']
        assert abs(np.corrcoef(observed.ravel(),np.array(other).ravel())[0,1])<.025


@pytest.mark.parametrize('method',['euler','heun','milstein'])
@pytest.mark.parametrize('backend,ranks',[('cpu',None),('cpu',2),('cpu',8),('metal',None),('metal',2),('metal',8),('cuda',None),('cuda',2),('cuda',8)])
def test_sde_backends_replay_and_restore(method,backend,ranks,tmp_path):
    import json,subprocess,sys
    if ranks and os.environ.get('B2_TEST_MPI')!='1':pytest.skip('MPI required')
    if backend=='metal' and os.environ.get('B2_TEST_GPU')!='1':pytest.skip('Metal required')
    if backend=='cuda' and os.environ.get('B2_TEST_CUDA_TRAIN')!='1':pytest.skip('CUDA required')
    net,source,groups,x,dt=model(method,refractory=True);bundle=lower(net,source,groups,detach_reset=False)
    plan=copy.deepcopy(bundle.plan);plan['backend']=backend
    if ranks:plan['mpi_ranks']=ranks
    target=NativeLIFTrainer(plan,runner=RUNNER,weights=bundle.weights)
    cpu=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights)
    batch=np.stack([x,x[::-1]]);initial=[bundle.initial_state]*2;labels=[0,1]
    for start,end in [(0,3),(3,7)]:
        a=cpu.step(batch[:,start:end],labels,initial=initial if start==0 else 'carry')
        c=target.step(batch[:,start:end],labels,initial=initial if start==0 else 'carry');compare(a,c)
        for key in ['final_state','initial_state_gradients']:np.testing.assert_allclose(a[key],c[key],rtol=5e-4,atol=1e-5)
        assert target.noise_sequence==c['noise_sequence']==0 and target.next_noise_sequence==1
    path=tmp_path/'checkpoint';target.store(path);req=tmp_path/'request';out=tmp_path/'result'
    req.write_text(json.dumps(dict(plan=plan,x=batch[:,7:].tolist(),labels=labels)))
    code='''import json,sys
from brian2_rust import NativeLIFTrainer
r=json.load(open(sys.argv[1]));t=NativeLIFTrainer(r['plan'],runner=sys.argv[4]);t.restore(sys.argv[2])
a=t.step(r['x'],r['labels'],initial='carry');b=t.step(r['x'],r['labels'])
json.dump([a,b,t.noise_sequence,t.next_noise_sequence],open(sys.argv[3],'w'))
'''
    subprocess.run([sys.executable,'-c',code,str(req),str(path),str(out),str(RUNNER)],check=True,timeout=120)
    a=target.step(batch[:,7:],labels,initial='carry');c=target.step(batch[:,7:],labels)
    assert json.loads(out.read_text())==[a,c,target.noise_sequence,target.next_noise_sequence]
    assert c['noise_sequence']==1 and target.next_noise_sequence==2
    plan['trainable']=[False]*len(plan['masks'])
    whole=NativeLIFTrainer(plan,runner=RUNNER,weights=bundle.weights);parts=NativeLIFTrainer(plan,runner=RUNNER,weights=bundle.weights)
    full=whole.step(batch,labels,initial=initial,noise_sequence=17)
    first=parts.step(batch[:,:4],labels,initial=initial,noise_sequence=17);last=parts.step(batch[:,4:],labels,initial='carry')
    assert last['final_state']==full['final_state'] and parts.noise_sequence==17
    for b in range(2):assert first['spikes'][b]+last['spikes'][b]==full['spikes'][b]


def test_noise_sequence_readonly_failure_and_manual_continuation(tmp_path):
    plan=raw_noise_plan();trainer=NativeLIFTrainer(plan,runner=RUNNER,weights=[[0.]])
    x=np.zeros((2,3,1));labels=[0,1]
    def snapshot():return copy.deepcopy((trainer.state,trainer.neuron_state,trainer.clock_tick,trainer.noise_sequence,trainer.next_noise_sequence))
    a=trainer.step(x,labels);assert trainer.noise_sequence==0 and trainer.next_noise_sequence==1
    before=snapshot()
    for operation in ['evaluate','gradients']:
        carry=trainer.execute(x,labels,operation=operation,initial='carry')
        manual=trainer.execute(x,labels,operation=operation,initial=a['final_state'],start_tick=3,noise_sequence=0)
        assert carry==manual and before==snapshot()
        fresh=trainer.execute(x,labels,operation=operation)
        assert fresh['noise_sequence']==1 and fresh['final_state']!=carry['final_state'] and before==snapshot()
    for sequence in [True,-1,2**64-1,1.5]:
        with pytest.raises(ValueError):trainer.step(x,labels,noise_sequence=sequence)
        assert before==snapshot()
    with pytest.raises(ValueError):trainer.step(x,labels,initial='carry',noise_sequence=0)
    with pytest.raises(ValueError):trainer.step(x,[2,3])
    assert before==snapshot()
    path=tmp_path/'checkpoint';trainer.store(path);clone=NativeLIFTrainer(plan,runner=RUNNER,weights=[[0.]]);clone.restore(path)
    assert clone.step(x,labels,initial='carry')==trainer.step(x,labels,initial='carry')


@pytest.mark.parametrize('change',['missing_streams','bad_count','negative_count','no_clock','stream_range','missing_layer','unknown_key'])
def test_noise_native_validation(change):
    p=raw_noise_plan()
    if change=='missing_streams':del p['noise_streams']
    elif change=='bad_count':p['noise_streams']=[17,2]
    elif change=='negative_count':p['noise_streams']=[-1,2]
    elif change=='no_clock':del p['clock']
    elif change=='stream_range':p['noise_streams']=[1,2]
    elif change=='missing_layer':p['noise_streams']=[2]
    else:p['state_equations'][0][0][0]['extra']=1
    t=NativeLIFTrainer(p,runner=RUNNER,weights=[[0.]]);before=copy.deepcopy(t.state)
    with pytest.raises(ValueError):t.step([[[0.]]],[0])
    assert t.state==before and t.neuron_state is None and t.next_noise_sequence==0


def test_noise_methods_preserve_brian_restrictions():
    net,source,groups,x,dt=model('milstein',shared=True)
    with pytest.raises(TrainingConversionError,match='diagonal'):lower(net,source,groups)
    net,source,groups,x,dt=model('heun')
    for g in groups:g.state_updater.method_choice='euler'
    with pytest.raises(TrainingConversionError,match='multiplicative'):lower(net,source,groups)
    for g in groups:g.state_updater.method_choice='rk4'
    with pytest.raises(TrainingConversionError,match='stochastic'):lower(net,source,groups)


def test_noise_checkpoint_rejects_invalid_sequence_atomically(tmp_path):
    import json,hashlib
    from brian2_rust.protocol import canonical_bytes
    t=NativeLIFTrainer(raw_noise_plan(),runner=RUNNER,weights=[[0.]]);t.step([[[0.]]],[0])
    path=tmp_path/'checkpoint';t.store(path);original=json.loads(path.read_text())
    before=copy.deepcopy((t.state,t.neuron_state,t.clock_tick,t.noise_sequence,t.next_noise_sequence))
    for sequence,next_sequence in [(None,1),(0,0),(True,1),(1,1),(0,1.5),(0,2**64),(0,None)]:
        payload=json.loads(original['payload']);payload.update(noise_sequence=sequence,next_noise_sequence=next_sequence)
        raw=canonical_bytes(payload);path.write_bytes(canonical_bytes(dict(original,payload=raw.decode(),sha256=hashlib.sha256(raw).hexdigest())))
        with pytest.raises(ValueError,match='noise sequence'):t.restore(path)
        assert (t.state,t.neuron_state,t.clock_tick,t.noise_sequence,t.next_noise_sequence)==before
