"""Physical parent storage and relative endpoint coordinates against real Brian."""
import ast
import re
from unittest.mock import patch
import numpy as np
import pytest
import brian2 as b
from brian2_rust import NativeLIFTrainer,lower_brian_dynamic_training,lower_brian_training,TrainingConversionError
from test_native_training import RUNNER
from test_training_linked import cython_cache
from test_training_integer_ir import engine
from test_training_stochastic import normal


def model(layout='both',noisy=False,delayed=False,warm=0,refractory=False,shared=False,**options):
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='cython';dt=.2*b.ms
    x=np.array([[1,1,0,1,1],[1,0,1,0,1],[0,1,1,1,0],[1,1,0,0,1],
                [0,0,1,1,1],[1,1,1,0,0],[0,1,0,1,1],[1,0,1,1,0]],float)
    inp=b.NeuronGroup(5,'v:1',threshold='input_events(t,i)>0',reset='',dt=dt,
        namespace={'input_events':b.TimedArray(x,dt=dt)},name='view_input')
    groups=[]
    for k in range(2):
        g=b.NeuronGroup(6,'dv/dt=(.7-v+.1*q)/ms'+('+.03*(v+1)*xi/sqrt(ms)' if noisy else '')+':1'+(' (unless refractory)' if refractory else '')+'\nq:1\ngain:1 ('+('shared,' if shared else '')+'constant)',
            threshold='v>.6',reset='v-=.4',refractory=.4*b.ms if refractory else False,method='heun' if noisy else 'euler',dt=dt,name=f'view_g{k}')
        g.v=np.array([.8,.3,.7,.4,.9,.2])+.02*k;g.q=np.arange(6)*.13+.1;g.gain=.3+.03*k if shared else np.arange(6)*.07+.2+.03*k;groups.append(g)
    a,c=groups;views=[];synapses=[]
    for k,(source,target) in enumerate(((inp,a),(a,c))):
        if layout in ('source','both'):source=source[1:4] if k==0 else source[1:5];views.append(source)
        if layout in ('target','both'):target=target[2:5] if k==0 else target[1:4];views.append(target)
        pre='v_post+=w+.01*z+.02*gain_post+.001*i+.002*j+.003*N_pre+.004*N_post\na+=.1\nw+=.01*a'
        if k:pre+='\nv_pre+=.01*gain_pre'
        if delayed:pre+='\ndelay=.6*delay+.1*ms*w'
        post='a+=.05\nw-=.002*gain_post'+('\ndelay=.7*delay+.04*ms*w' if delayed else '')
        s=b.Synapses(source,target,'w:1\nda/dt=-a/(2*ms):1 (event-driven)\n'
            'dz/dt=(-z+.1*w+.01*i)/ms'+('+.02*(z+1)*xi/sqrt(ms)' if noisy else '')+':1 (clock-driven)\n'
            'q_post=w+.1*z+.01*gain_post+.002*i+.003*j:1 (summed)',
            on_pre=pre,on_post=post,method='heun' if noisy else 'euler',dt=dt,name=f'view_s{k}')
        s.connect(i=[2,0,1,2] if k==0 else [3,1,0,2],j=[1,2,0,2]);s.w=[.12,.15,.18,.21];s.a=[.1,.2,.3,.4];s.z=[.2,.3,.4,.5]
        if delayed:s.pre.delay=[.04,.44,.24,.04]*b.ms;s.post.delay=[.24,.04,.44,.04]*b.ms
        synapses.append(s)
    net=b.Network(inp,*groups,*synapses,*views)
    if warm:b.seed(99);net.run(warm*dt,namespace={});x=x[warm:]
    bundle=lower_brian_dynamic_training(net,input_group=inp,layers=groups,
        trainable_neuron_parameters={g.name:['gain'] for g in groups},**options)
    return net,groups,synapses,x,bundle


def common_noise(net,groups,synapses,bundle,length,start_tick):
    net.run(0*b.ms,namespace={});schedules=[]
    for updater in net.sorted_objects:
        matched=[g for g in [*groups,*synapses] if updater is g.state_updater]
        if not matched:continue
        g=matched[0];names=sorted(g.equations.stochastic_variables);order=[]
        for line in updater.codeobj.code.run.splitlines():
            match=re.match(r'\s*(\w+)\s*=.*\b_randn\(',line)
            if match:order.append(names.index(match[1]))
        abstract=[]
        for statement in ast.parse(updater.abstract_code).body:
            if isinstance(statement,ast.Assign) and any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='randn' for n in ast.walk(statement.value)):
                abstract.append(names.index(statement.targets[0].id))
        assert order and sorted(order)==sorted(abstract)
        domain=groups.index(g) if g in groups else bundle.provenance['synaptic_noise_domains'][g.name]
        schedules.append((domain,len(g),order))
    assert len(schedules)==4 and len({d for d,_,_ in schedules})==4
    draws=[normal(bundle.plan['seed'],11,0,domain,j,t,stream) for t in range(start_tick,start_tick+length) for domain,count,order in schedules for j in range(count) for stream in order]
    assert len(draws)==length*20;calls=[]
    def refill(n):
        assert n==20000 and not calls;calls.append(n);values=np.zeros(n);values[:len(draws)]=draws;return values
    device=b.get_device();device.randn_buffer_index[:]=0
    with patch('numpy.random.randn',refill):net.run(length*.2*b.ms,namespace={})
    assert calls==[20000] and device.randn_buffer_index[0]==len(draws);device.randn_buffer_index[:]=0


@pytest.mark.parametrize('layout,shared',[('source',False),('target',False),('both',False),('both',True)])
@pytest.mark.parametrize('noisy',[False,True])
@pytest.mark.parametrize('delayed',[False,True])
@pytest.mark.parametrize('warm,refractory',[(0,False),(2,True)])
def test_subgroup_dynamic_matches_compiled_brian(engine,layout,shared,noisy,delayed,warm,refractory,tmp_path):
    net,groups,synapses,x,bundle=model(layout,noisy,delayed,warm,refractory,shared,backend=engine)
    bundle.plan['trainable']=[False]*len(bundle.weights)
    trainer=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights)
    monitors=[b.SpikeMonitor(g) for g in groups];net.add(monitors);spikes=[];cursor=0
    for length in (3,len(x)-3):
        result=trainer.step(x[None,cursor:cursor+length],[0],**(dict(initial='carry') if cursor else dict(noise_sequence=11) if noisy else {}))
        if noisy:common_noise(net,groups,synapses,bundle,length,cursor)
        else:net.run(length*.2*b.ms,namespace={})
        assert all(r.codeobj.compiled_code['run'] is not None for s in synapses for r in (s.state_updater,s.pre,s.post,*s.summed_updaters.values()))
        assert all(r.codeobj.compiled_code['run'] is not None for g in groups for r in (g.state_updater,g.thresholder['spike'],g.resetter['spike']))
        cursor+=length;z=np.asarray(result['final_state'])[0];tol=3e-12 if engine=='cpu' else 4e-5
        for g in groups:
            for name,slots in bundle.provenance['neuron_state_layout'][g.name].items():
                if not name.startswith('__'):np.testing.assert_allclose(z[slots],g.variables[name].get_value(),rtol=tol,atol=tol*1e-3,err_msg=g.name+'.'+name)
            if refractory:np.testing.assert_array_equal(z[bundle.provenance['refractory_activity_layout'][g.name]],g.not_refractory[:])
        for s in synapses:
            for name,slots in bundle.provenance['dynamic_state_layout'][s.name].items():np.testing.assert_allclose(z[slots],s.variables[name].get_value(),rtol=tol,atol=tol*1e-3,err_msg=s.name+'.'+name)
            for path in s._pathways:
                for name,slots in bundle.provenance['pathway_state_layout'].get(path.name,{}).items():np.testing.assert_allclose(z[slots],np.asarray(getattr(path,name)[:]),rtol=tol,atol=tol*1e-3)
        if engine!='cpu':assert result['gpu_dispatches']>0
        spikes.extend(np.asarray(result['spikes'])[0]);saved=tmp_path/'subgroup.json';trainer.store(saved)
        restored=NativeLIFTrainer(trainer.plan,runner=RUNNER);restored.restore(saved);trainer=restored
    expected=np.zeros((len(x),12))
    for k,mon in enumerate(monitors):expected[np.rint((np.asarray(mon.t/b.second)-bundle.plan['clock']['origin'])/.0002).astype(int),np.asarray(mon.i)+6*k]=1
    np.testing.assert_array_equal(spikes,expected)
