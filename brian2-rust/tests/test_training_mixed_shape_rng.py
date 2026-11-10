"""Shared Normal and physical-row Poisson samples in scheduled callbacks."""
import copy
import brian2 as b
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer,lower_brian_dynamic_training
from test_training_stochastic import normal
from test_training_regular_callback_rng import count
from test_training_integer_ir import engine
from test_training_poisson_zero_vjp import mpi
from test_native_training import RUNNER


def mixed_plain(x,mean,spread):
    draw=np.random.normal(0.,spread)
    return x+.1*draw+.1*np.random.poisson(mean)


def mixed_chain(x,mean,spread):
    draw=np.random.normal(0.,spread)
    return x+.1*draw+.1*np.random.poisson(mean+.2*draw)


def mixed_capture(array):
    def f(x,mean,spread):
        captured=array
        captured*=.8
        x*=.7
        draw=np.random.normal(0.,spread)
        return x+.1*draw+.1*np.random.poisson(mean+.2*draw)
    return f


def callback(chain,capture,array):
    return mixed_capture(array) if capture else mixed_chain if chain else mixed_plain


def model(backend,phase,chain,capture,method='euler',window=None):
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='numpy';dt=.2*b.ms
    source=b.SpikeGeneratorGroup(1,[],np.array([])*dt,dt=dt)
    hidden=b.NeuronGroup(2 if phase=='synaptic' else 1,'dv/dt=0/second:1',threshold='v>100',reset='v=0',method='euler',dt=dt)
    if phase=='synaptic':
        g=b.NeuronGroup(2,'dv/dt=(q+.01)/ms:1\nq:1',threshold='v>.5',reset='v-=.5',method='euler',dt=dt);g.v=[.383,.713]
        obj=b.Synapses(hidden,g,'dh/dt=.1*gain*f(a,mean,spread)/ms:1 (clock-driven)\na:1\nq_post=h:1 (summed)\nmean:1 (constant)\nspread:1 (constant,shared)\ngain:1 (constant)',method=method,dt=dt)
        obj.connect(i=[1,0],j=[0,1]);obj.h=[.173,.219];net=b.Network(source,hidden,g,obj)
    else:
        update='gain+.1*f(a,mean,spread)' if phase=='update' else 'gain'
        threshold='f(v,mean,spread)>.5' if phase=='threshold' else 'v>.5'
        reset='v=.2*f(a,mean,spread)+v-.5' if phase=='reset' else 'v-=.5'
        g=b.NeuronGroup(2,'dv/dt=('+update+')/ms:1\na:1\nmean:1 (constant)\nspread:1 (constant,shared)\ngain:1 (constant)',method=method,threshold=threshold,reset=reset,dt=dt);obj=g;g.v=[.473,.719];net=b.Network(source,hidden,g)
    obj.a=[.317,.611];obj.mean=[1.3,.7];obj.spread=.13;obj.gain=[.217,.319]
    obj.namespace['f']=b.Function(callback(chain,capture,obj.variables['a'].get_value()),arg_units=[1,1,1],return_unit=1,stateless=False)
    bundle=lower_brian_dynamic_training(net,input_group=source,layers=[hidden,g],backend=backend,mpi_ranks=2,tbptt_window=window,detach_reset=False,
        **({'trainable_synapse_parameters':{obj.name:['gain','mean','spread','a','h']}} if phase=='synaptic' else {'trainable_neuron_parameters':{g.name:['gain','mean','spread']}}))
    return net,hidden,g,obj,dt,bundle


PROFILES=[('threshold',False,False,'euler'),('threshold',True,True,'euler'),
 ('reset',False,False,'euler'),('reset',True,True,'euler'),
 ('update',False,False,'euler'),('update',True,True,'rk2'),
 ('synaptic',False,False,'euler'),('synaptic',True,True,'rk2')]


@pytest.mark.parametrize('phase,chain,capture,method',PROFILES)
def test_mixed_shape_original_restore(engine,phase,chain,capture,method,monkeypatch,tmp_path):
    mpi(2);net,hidden,g,obj,dt,bundle=model(engine,phase,chain,capture,method)
    monitor=b.SpikeMonitor(g);net.add(monitor);domain=2 if phase=='synaptic' else 1
    records=bundle.provenance['synaptic_integrator_draws'][obj.name] if phase=='synaptic' else bundle.provenance['neuron_draws'][1][phase]
    assert [(r['kind'],r.get('scalar',False)) for r in records]==[('randn',True),('poisson',False)]*(2 if method=='rk2' else 1)
    p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights);trainer=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER)
    saved_count=0
    for tick in range(4):
        serial=0;previous_draw=0.
        selected=np.flatnonzero(np.asarray(g.v[:])+.2*np.asarray(g.gain[:])>.5) if phase=='reset' else np.arange(2)
        def scalar(loc,spread):
            nonlocal serial,previous_draw
            assert serial%2==0 and np.ndim(loc)==np.ndim(spread)==0 and loc==0.
            previous_draw=float(spread*normal(p['seed'],9,0,domain,0,tick,serial));serial+=1;return previous_draw
        def vector(rate):
            nonlocal serial,saved_count
            assert serial%2==1
            np.testing.assert_allclose(rate,np.asarray(obj.mean[:])[selected]+(.2*previous_draw if chain else 0.))
            result=np.array([count(p['seed'],9,domain,int(j),tick,float(rate[k]),serial) for k,j in enumerate(selected)]);serial+=1;saved_count+=len(selected);return result
        monkeypatch.setattr(np.random,'normal',scalar);monkeypatch.setattr(np.random,'poisson',vector)
        out=trainer.step(np.zeros((1,1,1)),[0],**({'initial':'carry'} if tick else {'noise_sequence':9}));net.run(dt,namespace={})
        assert serial==len(records)
        for group in (hidden,g):
            for name in ('v','a','q'):
                if name not in bundle.provenance['neuron_state_layout'][group.name]:continue
                cells=bundle.provenance['neuron_state_layout'][group.name][name]
                np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],group.variables[name].get_value(),rtol=8e-5,atol=8e-6)
        if phase=='synaptic':
            for name in ('a','h'):
                if name not in bundle.provenance['dynamic_state_layout'][obj.name]:continue
                cells=bundle.provenance['dynamic_state_layout'][obj.name][name]
                np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],obj.variables[name].get_value(),rtol=8e-5,atol=8e-6)
        recorded=np.asarray(monitor.i)[np.isclose(np.asarray(monitor.t/b.second),tick*float(dt),rtol=0.,atol=1e-12)]
        np.testing.assert_array_equal(np.asarray(out['spikes'])[0,-1,len(hidden):],np.isin(np.arange(2),recorded).astype(float))
        path=tmp_path/str(tick);trainer.store(path);trainer=NativeLIFTrainer(p,runner=RUNNER);trainer.restore(path)
    assert len(trainer.poisson_state['entries'])==saved_count
    assert (out['gpu_dispatches']>0)==(engine!='cpu')


def parameter(data,weights,name):
    obj=data[3];row=next(row for row in data[5].provenance['bindings'] if row['object']==obj.name and name in row['variables'])
    return weights[row['bank']][row['variables'].index(name)] if obj.variables[name].scalar else np.asarray(weights[row['bank']])


def reference(data,phase,chain,capture,method,weights,initial=None,anchors=None,override=None):
    from unittest.mock import patch
    from brian2_rust.training_brian import _INTEGRATORS
    _,hidden,g,obj,_,bundle=data;p=bundle.plan;domain=2 if phase=='synaptic' else 1
    z=np.array(bundle.initial_state if initial is None else initial,dtype=float)
    if initial is None:
        for cell,ref in enumerate(p['dynamic']['initial_parameters']):
            if ref is not None:z[cell]=weights[ref[0]][ref[1]]
    layout=bundle.provenance['neuron_state_layout'][g.name];v=layout['v']
    obj_layout=bundle.provenance['dynamic_state_layout'][obj.name] if phase=='synaptic' else layout
    a=obj_layout.get('a');target=obj_layout['h'] if phase=='synaptic' else v;name='h' if phase=='synaptic' else 'v'
    stage=compile(_INTEGRATORS[method](obj.equations,variables={**obj.variables,'f':obj.namespace['f']}),'<independent mixed integrator>','exec') if phase in ('update','synaptic') else None
    before=[];margins=[];hard=[];events=[];samples=[]
    for tick in range(4):
        if anchors is not None and p['tbptt_window'] and tick and tick%p['tbptt_window']==0:z=anchors['before'][tick].copy()
        before.append(z.copy())
        if phase=='synaptic':z[layout['q']]=z[target];z[v]+=.2*(z[layout['q']]+.01)
        elif phase!='update':z[v]+=.2*parameter(data,weights,'gain')
        physical=z[a].copy() if a is not None else parameter(data,weights,'a').copy();serial=0
        active=(z[v]>.5) if phase=='reset' else np.ones(2,dtype=bool)
        def scalar(loc,spread):
            nonlocal serial
            stream=serial;serial+=1
            return float(loc+spread*normal(p['seed'],9,0,domain,0,tick,stream))
        def vector(rate):
            nonlocal serial
            stream=serial;serial+=1;rate=np.asarray(rate);values=[]
            for j,r in enumerate(rate):
                value=next(c for t,s,k,rr,c,act in anchors['samples'] if (t,s,k)==(tick,stream,j)) if anchors is not None else count(p['seed'],9,domain,j,tick,float(r),stream)
                if override==(tick,stream,j):value+=1
                samples.append((tick,stream,j,float(r),value,bool(active[j])));values.append(value)
            return np.array(values)
        with patch.object(np.random,'normal',scalar),patch.object(np.random,'poisson',vector):
            f=callback(chain,capture,physical)
            if stage is not None:
                env={name:z[target].copy(),'a':physical,'mean':parameter(data,weights,'mean'),'spread':parameter(data,weights,'spread'),'gain':parameter(data,weights,'gain'),'dt':.0002,'ms':.001,'f':f}
                exec(stage,env);z[target]=env[name]
                margin=z[v]-.5
            elif phase=='threshold':
                value=z[v].copy();result=f(value,parameter(data,weights,'mean'),parameter(data,weights,'spread'));z[v]=value;margin=result-.5
            else:
                margin=z[v]-.5
                candidate=.2*f(physical.copy(),parameter(data,weights,'mean'),parameter(data,weights,'spread'))+z[v]-.5
        if a is not None:z[a]=physical
        event=(margin>0).astype(float);margins.append(margin.copy());hard.append(event.copy())
        if anchors is not None:event=anchors['hard'][tick]+p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(anchors['margins'][tick]))**2*(margin-anchors['margins'][tick])
        events.append(event.copy())
        if phase=='reset':z[v]+=event*(candidate-z[v])
        else:z[v]-=.5*event
    logits=np.array(events).mean(0)*p['logit_scale'];loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
    return loss,z,dict(before=before,margins=margins,hard=hard,events=events,samples=samples)


@pytest.mark.parametrize('phase,chain,capture,method,zero',[(phase,chain,capture,method,False) for phase,chain,capture,method in PROFILES]+[(phase,False,False,'euler',True) for phase in ('threshold','reset','update','synaptic')])
@pytest.mark.parametrize('window',[None,2])
def test_mixed_shape_all_bank_initial_vjps(engine,phase,chain,capture,method,zero,window):
    mpi(2);data=model(engine,phase,chain,capture,method,window);bundle=data[5];x=np.zeros((1,4,1))
    row=next(row for row in bundle.provenance['bindings'] if row['object']==data[3].name and 'mean' in row['variables']);mean_bank=row['bank']
    if zero:
        bundle.weights[mean_bank][0]=0.;cell=bundle.provenance['neuron_state_layout'][data[2].name]['v'][0]
        voltage_range=np.arange(.65,.85,.0005) if phase=='reset' else np.arange(.25,.7,.0001 if phase=='synaptic' else .0005)
        for voltage in voltage_range:
            bundle.initial_state[cell]=float(voltage);bundle.plan['dynamic']['initial'][cell]=float(voltage)
            loss,_,anchors=reference(data,phase,chain,capture,method,bundle.weights)
            weak=sum(reference(data,phase,chain,capture,method,bundle.weights,override=(t,s,j))[0]-loss for t,s,j,r,c,act in anchors['samples'] if r==0. and act)
            if abs(weak)>1e-5 and min(abs(m[0]) for m in anchors['margins'])>1e-5:break
        else:raise AssertionError('mixed shape zero-rate replay must be nonzero')
    loss,z,anchors=reference(data,phase,chain,capture,method,bundle.weights)
    out=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).gradients(x,[0],noise_sequence=9)
    assert out['loss']==pytest.approx(loss,abs=8e-6)
    cells=list(bundle.provenance['neuron_state_layout'][data[2].name]['v'])
    for name in ('a','q'):
        cells+=bundle.provenance['neuron_state_layout'][data[2].name].get(name,[])
    if phase=='synaptic':
        for indices in bundle.provenance['dynamic_state_layout'][data[3].name].values():cells+=indices
    np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],z[cells],rtol=8e-5,atol=8e-6)
    np.testing.assert_array_equal(np.asarray(out['spikes'])[0,:,len(data[1]):],anchors['events'])
    for bank,values in enumerate(bundle.weights):
        for index in range(len(values)):
            hi=copy.deepcopy(bundle.weights);lo=copy.deepcopy(bundle.weights);hi[bank][index]+=1e-6;lo[bank][index]-=1e-6
            high=reference(data,phase,chain,capture,method,hi,anchors=anchors);low=reference(data,phase,chain,capture,method,lo,anchors=anchors)
            fd=(high[0]-low[0])/2e-6;score=0.
            for base,up,down in zip(anchors['samples'],high[2]['samples'],low[2]['samples']):
                tick,stream,j,rate,value,active=base
                if not active:continue
                derivative=(up[3]-down[3])/2e-6
                signal=loss*(value/rate-1.) if rate else reference(data,phase,chain,capture,method,bundle.weights,override=(tick,stream,j))[0]-loss
                score+=signal*derivative
            assert out['gradients'][bank][index]==pytest.approx(fd+score,rel=3e-3,abs=2e-5),(phase,bank,index)
    initial=np.array(bundle.initial_state,dtype=float);explicit=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).gradients(x,[0],initial=initial[None],noise_sequence=9)
    _,_,anchors=reference(data,phase,chain,capture,method,bundle.weights,initial=initial)
    for index in range(len(initial)):
        if bundle.plan['dynamic']['detached'][index] or index in bundle.plan['dynamic']['integer_states']:
            assert explicit['initial_state_gradients'][0][index]==0.;continue
        hi=initial.copy();lo=initial.copy();hi[index]+=1e-6;lo[index]-=1e-6
        fd=(reference(data,phase,chain,capture,method,bundle.weights,initial=hi,anchors=anchors)[0]-reference(data,phase,chain,capture,method,bundle.weights,initial=lo,anchors=anchors)[0])/2e-6
        assert explicit['initial_state_gradients'][0][index]==pytest.approx(fd,rel=3e-3,abs=2e-5),(phase,index)


@pytest.mark.parametrize('phase',['threshold','reset','update','synaptic'])
@pytest.mark.parametrize('invalid',['mean','spread'])
def test_mixed_shape_invalid_restore_atomic(engine,phase,invalid,tmp_path):
    mpi(2);data=model(engine,phase,False,False);bundle=data[5];x=np.zeros((1,1,1));p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights)
    trainer=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER);trainer.step(x,[0],noise_sequence=9)
    path=tmp_path/'prefix';trainer.store(path);trainer=NativeLIFTrainer(p,runner=RUNNER);trainer.restore(path)
    # Force the bad physical row active in reset, where inactive array rates
    # are deliberately skipped. Shared spread is validated for every block.
    cell=bundle.provenance['neuron_state_layout'][data[2].name]['v'][1];trainer.neuron_state[0][cell]=.8
    row=next(row for row in bundle.provenance['bindings'] if row['object']==data[3].name and invalid in row['variables'])
    trainer.state['weights'][row['bank']][1 if invalid=='mean' else row['variables'].index(invalid)]=-100. if invalid=='mean' else -1.
    before=copy.deepcopy((trainer.state,trainer.neuron_state,trainer.clock_state,trainer.poisson_state,trainer.elapsed_ticks,trainer.next_noise_sequence))
    for _ in range(2):
        with pytest.raises(ValueError):trainer.step(x,[0],initial='carry')
        assert (trainer.state,trainer.neuron_state,trainer.clock_state,trainer.poisson_state,trainer.elapsed_ticks,trainer.next_noise_sequence)==before
