"""Uniform native streams: independent counters, VJPs, GPU and MPI replay."""
import copy
import os
from unittest.mock import patch
import brian2 as b
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer,lower_brian_dynamic_training
from brian2_rust.training import lif_training_plan
from brian2_rust.training_equations import (compile_training_equation,neuron_parameter_bank,
    UniformNoise,NormalNoise,NeuronParameter)
from test_native_training import RUNNER
from test_training_integer_ir import engine
from test_training_linked import cython_cache
from test_training_stochastic import normal
from test_training_regular import model as regular_model


def uniform(seed,sequence,batch,domain,entity,tick,stream):
    mask=(1<<64)-1
    def mix(x):
        x=((x^(x>>30))*0xbf58476d1ce4e5b9)&mask
        x=((x^(x>>27))*0x94d049bb133111eb)&mask
        return x^(x>>31)
    key=mix(seed^0x4232554e49463031)
    for x in (sequence,batch,domain,entity,tick,stream):key=mix(key^mix((x+0x9e3779b97f4a7c15)&mask))
    return (mix(key^0xa0761d6478bd642f)>>11)/2**53


def plan(dynamic=False,mixed=True,window=None,backend='cpu',ranks=None):
    updates=[[compile_training_equation('.8*v+w*u'+('+.03*(v+1)*n' if mixed else ''),states=['v'],
        parameters={'w':NeuronParameter(k,0),'u':UniformNoise(0),'n':NormalNoise(1)})] for k in range(2)]
    resets=[[compile_training_equation('v-.3-.02*u' if mixed else 'v-.3',states=['v'],parameters={'u':UniformNoise(0)})]]*2
    p=lif_training_plan([1,2,2],projections=[neuron_parameter_bank(2)]*2,threshold=.6,detach_reset=False,
        state_equations=updates,state_resets=resets,clock={'origin':0.,'dt':.001},noise_streams=[2 if mixed else 1]*2,
        tbptt_window=window,backend=backend,mpi_ranks=ranks,seed=731)
    initial=[.7,.3,.6,.4];weights=[[.3,.4],[.5,.2]]
    if dynamic:
        programs=updates+resets;actions=[]
        for j in range(4):actions.append(dict(owner=j,reads=[j],writes=[j],program_set=j//2,threshold=None,trigger=None,
            parameter_index=j%2,noise_domain=j//2,noise_entity=j%2,noise_streams=2 if mixed else 1))
        for j in range(4):actions.append(dict(owner=j,reads=[j],writes=[],program_set=None,threshold=j,trigger=None))
        for j in range(4):actions.append(dict(owner=j,reads=[j],writes=[j],program_set=2+j//2,threshold=None,
            trigger=dict(external=False,index=j),noise_domain=j//2,noise_entity=j%2,noise_streams=2 if mixed else 1))
        p.update(schema='b2-dynamic-training-plan-v5',dynamic=dict(initial=initial,initial_parameters=[None]*4,
            detached=[False]*4,voltage=list(range(4)),program_sets=programs,actions=actions))
    return p,weights,np.asarray(initial,float)


def oracle(p,w,z,mixed,anchors=None,sequence=9,start=0,length=6):
    z=z.copy();spikes=[];before=[];margins=[]
    for tick in range(length):
        if anchors is not None and p['tbptt_window'] and tick and tick%p['tbptt_window']==0:z=anchors['before'][tick].copy()
        before.append(z.copy())
        u=np.array([uniform(p['seed'],sequence,0,j//2,j%2,start+tick,0) for j in range(4)])
        n=np.array([normal(p['seed'],sequence,0,j//2,j%2,start+tick,1) for j in range(4)])
        z=.8*z+np.asarray(w).ravel()*u+(.03*(z+1)*n if mixed else 0)
        margin=z-.6;event=(margin>0).astype(float)
        if anchors is not None:
            old=anchors['margins'][tick];event=(old>0).astype(float)+p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(old))**2*(margin-old)
        margins.append(margin.copy());spikes.append(event.copy());z-=(.3+.02*u if mixed else .3)*event
    logits=np.asarray(spikes)[:,2:].mean(0)*p['logit_scale']
    loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
    return loss,z,spikes,dict(before=before,margins=margins)


@pytest.mark.parametrize('dynamic',[False,True])
@pytest.mark.parametrize('mixed',[False,True])
@pytest.mark.parametrize('window',[None,2])
def test_uniform_all_derivatives(engine,dynamic,mixed,window):
    p,w,z=plan(dynamic,mixed,window,engine);x=np.zeros((1,6,1))
    actual=NativeLIFTrainer(p,weights=w,runner=RUNNER).gradients(x,[0],initial=z[None],noise_sequence=9)
    loss,final,spikes,anchors=oracle(p,w,z,mixed);tol=2e-3 if engine!='cpu' else 5e-7
    np.testing.assert_allclose(actual['loss'],loss,rtol=tol)
    np.testing.assert_allclose(actual['final_state'][0],final,rtol=tol,atol=tol*1e-2)
    np.testing.assert_array_equal(actual['spikes'][0],spikes)
    eps=1e-6
    for bank in range(2):
        for j in range(2):
            a=copy.deepcopy(w);c=copy.deepcopy(w);a[bank][j]+=eps;c[bank][j]-=eps
            expected=(oracle(p,a,z,mixed,anchors)[0]-oracle(p,c,z,mixed,anchors)[0])/(2*eps)
            np.testing.assert_allclose(actual['gradients'][bank][j],expected,rtol=tol,atol=tol*1e-2)
    for j in range(4):
        a=z.copy();c=z.copy();a[j]+=eps;c[j]-=eps
        expected=(oracle(p,w,a,mixed,anchors)[0]-oracle(p,w,c,mixed,anchors)[0])/(2*eps)
        np.testing.assert_allclose(actual['initial_state_gradients'][0][j],expected,rtol=tol,atol=tol*1e-2)
    if engine!='cpu':assert actual['gpu_dispatches']>0


@pytest.mark.parametrize('dynamic',[False,True])
@pytest.mark.parametrize('ranks',[2,8])
def test_uniform_mpi_and_restore(engine,dynamic,ranks,tmp_path):
    if os.environ.get('B2_TEST_MPI')!='1':pytest.skip('local MPI opt-in')
    p,w,z=plan(dynamic,backend=engine,ranks=ranks);x=np.zeros((1,6,1))
    reference=copy.deepcopy(p);reference['backend']='cpu';reference['mpi_ranks']=None
    expected_gradient=NativeLIFTrainer(reference,weights=w,runner=RUNNER).gradients(x,[0],initial=z[None],noise_sequence=9)
    actual_gradient=NativeLIFTrainer(p,weights=w,runner=RUNNER).gradients(x,[0],initial=z[None],noise_sequence=9)
    for key in ('loss','spikes','gradients','initial_state_gradients','final_state'):
        np.testing.assert_allclose(actual_gradient[key],expected_gradient[key],rtol=2e-3 if engine!='cpu' else 3e-12,atol=2e-6 if engine!='cpu' else 3e-13)
    p['trainable']=[False,False];trainer=NativeLIFTrainer(p,weights=w,runner=RUNNER)
    trainer.step(x[:,:2],[0],initial=z[None],noise_sequence=9)
    path=tmp_path/'uniform.json';trainer.store(path);restored=NativeLIFTrainer(p,runner=RUNNER);restored.restore(path)
    result=restored.step(x[:,2:],[0],initial='carry');expected=oracle(p,w,z,True)[1]
    np.testing.assert_allclose(result['final_state'][0],expected,rtol=4e-5 if engine!='cpu' else 3e-12)
    assert result['noise_sequence']==9 and result['final_tick']==6


@pytest.mark.parametrize('dynamic',[False,True])
def test_uniform_distribution_and_batch_addresses(engine,dynamic):
    p,w,z=plan(dynamic,mixed=False,backend=engine)
    count=128;batch=80;x=np.zeros((batch,1,1));p['sizes']=[1,count,count]
    p['threshold']=[2.,2.];code=[[dict(op='uniform_noise',stream=0)]]
    p['state_equations']=[code,code]
    if dynamic:
        d=p['dynamic'];d.update(initial=[0.]*(2*count),initial_parameters=[None]*(2*count),
            detached=[False]*(2*count),voltage=list(range(2*count)))
        d['program_sets']=[code,code];d['actions']=[]
        for j in range(2*count):d['actions'].append(dict(owner=j,reads=[j],writes=[j],program_set=j//count,
            threshold=None,trigger=None,noise_domain=j//count,noise_entity=j%count,noise_streams=1))
        for j in range(2*count):d['actions'].append(dict(owner=j,reads=[j],writes=[],program_set=None,threshold=j,trigger=None))
    p['seed']=2**64-1
    result=NativeLIFTrainer(p,weights=w,runner=RUNNER).evaluate(x,[0]*batch,noise_sequence=2**64-2)
    expected=np.array([[uniform(p['seed'],2**64-2,i,j//count,j%count,0,0) for j in range(2*count)] for i in range(batch)])
    values=np.asarray(result['final_state']);assert np.all((values>=0)&(values<1))
    np.testing.assert_allclose(values,expected,rtol=1e-7 if engine!='cpu' else 0,atol=0)
    assert abs(values.mean()-.5)<.01 and abs(values.var()-1/12)<.003
    assert len(np.unique(values))>values.size*.995


@pytest.mark.parametrize('dynamic',[False,True])
def test_uniform_conflicting_stream_rejected(dynamic):
    p,w,z=plan(dynamic)
    code=[dict(op='noise',stream=0),dict(op='uniform_noise',stream=0),dict(op='add',left=0,right=1)]
    if dynamic:p['dynamic']['program_sets'][0]=[code]
    else:p['state_equations'][0]=[code]
    with pytest.raises(ValueError,match='mixes'):
        NativeLIFTrainer(p,weights=w,runner=RUNNER).gradients(np.zeros((1,2,1)),[0])


@pytest.mark.parametrize('warm',[0,2])
def test_uniform_regular_compiled_brian(engine,warm):
    net,inp,groups,synapses,x,_=regular_model('shared','end',warm=warm)
    runners={obj.name:obj for obj in net.sorted_objects if obj.name in ('regular_neuron','regular_synapse')}
    runners['regular_neuron'].abstract_code='g+=.01*rand()\nu=rand()\nn=randn()\nv+=.03*(v+1)*u+.002*n'
    runners['regular_synapse'].abstract_code='h+=.01*randn()\nu=rand()\nw+=.02*w*u+.001*h'
    bundle=lower_brian_dynamic_training(net,input_group=inp,layers=groups,backend=engine);bundle.plan['trainable']=[False]*len(bundle.weights)
    trainer=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER);net.run(0*b.ms,namespace={})
    ordered=[o for o in net.sorted_objects if o.name in runners]
    for obj in ordered:assert obj.codeobj.compiled_code['run'] is not None
    cursor=0
    for part in (x[:2],x[2:]):
        result=trainer.step(part[None],[0],**({'initial':'carry'} if cursor else {'noise_sequence':9}))
        draws={'rand':[],'randn':[]}
        for tick in range(cursor,cursor+len(part)):
            for obj in ordered:
                domain=bundle.provenance['regular_runner_layout'][obj.name]['noise_domain']
                if obj.name=='regular_neuron':
                    draws['rand'].append(uniform(bundle.plan['seed'],9,0,domain,0,tick,0))
                    for j in range(3):
                        draws['rand'].append(uniform(bundle.plan['seed'],9,0,domain,j,tick,1))
                        draws['randn'].append(normal(bundle.plan['seed'],9,0,domain,j,tick,2))
                else:
                    draws['randn'].append(normal(bundle.plan['seed'],9,0,domain,0,tick,0))
                    draws['rand'].extend(uniform(bundle.plan['seed'],9,0,domain,j,tick,1) for j in range(4))
        calls={name:0 for name in draws};device=b.get_device()
        device.rand_buffer_index[:]=0;device.randn_buffer_index[:]=0
        def sampler(name):
            def refill(n):
                assert n==20000 and calls[name]==0;calls[name]+=1
                out=np.zeros(n);out[:len(draws[name])]=draws[name];return out
            return refill
        with patch('numpy.random.rand',sampler('rand')),patch('numpy.random.randn',sampler('randn')):net.run(len(part)*.2*b.ms,namespace={})
        for name in draws:assert calls[name]==1 and getattr(device,name+'_buffer_index')[0]==len(draws[name])
        device.rand_buffer_index[:]=0;device.randn_buffer_index[:]=0
        state=np.asarray(result['final_state'])[0];tol=4e-5 if engine!='cpu' else 3e-12
        for g in groups:
            np.testing.assert_allclose(state[bundle.provenance['neuron_state_layout'][g.name]['v']],g.v[:],rtol=tol,atol=tol*1e-2)
        for name,slots in bundle.provenance['dynamic_state_layout']['regular_s'].items():np.testing.assert_allclose(state[slots],synapses[0].variables[name].get_value(),rtol=tol,atol=tol*1e-2)
        cursor+=len(part)


def test_uniform_regular_mask_migration(engine,tmp_path):
    net,inp,groups,synapses,x,_=regular_model('shared','end')
    runner=next(o for o in net.sorted_objects if o.name=='regular_synapse')
    runner.abstract_code='h+=.01\nu=rand()\nw+=.03*w*u\nv_post+=.01*w'
    bundle=lower_brian_dynamic_training(net,input_group=inp,layers=groups,backend=engine)
    bank=next(e['bank'] for e in bundle.provenance['bindings'] if e['object']=='regular_s' and e['variables']==['w'])
    scratch=bundle.provenance['regular_runner_layout'][runner.name]['temporary']['u']
    entry=next(e for e in bundle.plan['dynamic']['migration']['cells'] if e['index']==scratch)
    assert entry['owners']==[[bank,k] for k in range(4)] and entry['restart']=={'kind':'initial'}
    cpu_plan=copy.deepcopy(bundle.plan);cpu_plan['backend']='cpu'
    trainers=[NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER) for p in (bundle.plan,cpu_plan)]
    for start,stop in ((0,2),(2,4),(4,6)):
        results=[trainer.step(x[None,start:stop],[0],**({'initial':'carry'} if start else {'noise_sequence':9})) for trainer in trainers]
        for key in ('loss','gradients','final_state','initial_state_gradients'):
            # Banks have heterogeneous widths.
            if key=='gradients':
                for a,c in zip(results[0][key],results[1][key]):np.testing.assert_allclose(a,c,rtol=2e-3,atol=2e-6)
            else:np.testing.assert_allclose(results[0][key],results[1][key],rtol=2e-3,atol=2e-6)
        if stop==6:break
        for k,trainer in enumerate(trainers):
            masks=copy.deepcopy(trainer.plan['masks']);masks[bank]=[0. if stop==2 else 1.]*4
            before=(trainer.clock_tick,trainer.noise_sequence,trainer.next_noise_sequence,trainer.state['step'])
            trainer.update_mask(masks,growth_weight=.12)
            assert before==(trainer.clock_tick,trainer.noise_sequence,trainer.next_noise_sequence,trainer.state['step'])
            assert trainer.neuron_state[0][scratch]==0
            path=tmp_path/f'migration-{k}.json';trainer.store(path)
            restored=NativeLIFTrainer(trainer.plan,runner=RUNNER);restored.restore(path);trainers[k]=restored
