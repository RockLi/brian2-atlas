"""Update and threshold whole callbacks reserve independent streams."""
import copy
import brian2 as b
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer, lower_brian_dynamic_training
from test_training_neuron_whole_rng import normal_factory, uniform_factory, poisson_factory, derived_factory
from test_training_integer_ir import engine
from test_training_poisson_zero_vjp import mpi
from test_training_stochastic import normal
from test_training_uniform import uniform
from test_training_regular_callback_rng import count
from test_native_training import RUNNER


def model(backend, kind, method, mode, window=None, threshold_parameters=False):
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='numpy';dt=.2*b.ms
    source=b.SpikeGeneratorGroup(1,[],np.array([])*dt,dt=dt)
    storage=b.NeuronGroup(4,'dv/dt=0/second:1\na:1',threshold='v>100',reset='v=0',method='euler',dt=dt);storage.a=[.1,.2,.3,.4]
    sde=method in ('euler','heun','milstein')
    equation='dv/dt=(gain+.1*f(v,k,mean,spread))/ms'+('+noise*xi/ms**.5' if sde else '')+(':1 (unless refractory)' if mode!='none' else ':1')+'\nk:integer\nmean:1 (constant,shared)\nspread:1 (constant,shared)\ngain:1 (constant)\nnoise:1 (constant,shared)\nref_duration:second (constant,shared)'
    if threshold_parameters:equation+='\nthreshold_k:integer\nthreshold_mean:1 (constant,shared)\nthreshold_spread:1 (constant,shared)'
    refractory={'none':False,'fixed':.4*b.ms,'duration':'ref_duration','boolean':'v>.2'}[mode]
    threshold='h(v,threshold_k,threshold_mean,threshold_spread)>.5' if threshold_parameters else 'f(v,k,mean,spread)>.5'
    g=b.NeuronGroup(2,equation,threshold=threshold,reset='v-=.5',refractory=refractory,method=method,dt=dt)
    g.v=[.473,.719];g.k=[3,0];g.mean=1.3 if kind in ('poisson','derived') else .043;g.spread=.13;g.gain=[.217,.319];g.noise=.011;g.ref_duration=.4*b.ms
    g.namespace['f']=b.Function(globals()[kind+'_factory'](storage.variables['a'].get_value()),arg_units=[1,1,1,1],arg_types=['float','integer','float','float'],return_unit=1,stateless=False)
    if threshold_parameters:
        g.threshold_k=[3,0];g.threshold_mean=g.mean[:];g.threshold_spread=g.spread[:]
        g.namespace['h']=b.Function(globals()[kind+'_factory'](storage.variables['a'].get_value()),arg_units=[1,1,1,1],arg_types=['float','integer','float','float'],return_unit=1,stateless=False)
    net=b.Network(source,storage,g)
    bundle=lower_brian_dynamic_training(net,input_group=source,layers=[storage,g],backend=backend,mpi_ranks=2,tbptt_window=window,detach_reset=False,trainable_neuron_parameters={g.name:['gain','mean','spread',*(['noise'] if sde else []),*(['threshold_mean','threshold_spread'] if threshold_parameters else [])]})
    return net,storage,g,dt,bundle


@pytest.mark.parametrize('kind',['normal','uniform','poisson','derived'])
@pytest.mark.parametrize('method',['euler','rk2','rk4','heun','milstein'])
@pytest.mark.parametrize('mode',['none','fixed','duration','boolean'])
def test_whole_phase_original_restore(engine,kind,method,mode,monkeypatch,tmp_path):
    mpi(2);net,storage,g,dt,bundle=model(engine,kind,method,mode)
    monitor=b.SpikeMonitor(g);net.add(monitor)
    p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights)
    trainer=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER)
    records=[*bundle.provenance['whole_neuron_update_layout'][g.name]['draws'],*bundle.provenance['whole_neuron_threshold_layout'][g.name]['draws']]
    callbacks=[row for row in records if not row.get('named',False)]
    named=[row for row in records if row.get('named',False)]
    assert callbacks and all(row['length']==4 for row in callbacks)
    assert len({row['stream'] for row in records})==len(records)
    assert max(row['stream'] for row in bundle.provenance['whole_neuron_update_layout'][g.name]['draws']) < min(row['stream'] for row in bundle.provenance['whole_neuron_threshold_layout'][g.name]['draws'])
    flags=[];calls=[]
    for tick in range(6):
        streams=iter(row['stream'] for row in callbacks)
        def sample(*args,**kwargs):
            stream=next(streams);calls.append((tick,stream));assert kwargs['size']==4
            mean=float(g.mean[:]);spread=float(g.spread[:]);rate=mean+np.asarray(storage.a[:]) if kind=='derived' else mean
            np.testing.assert_allclose(args,(rate,) if kind in ('poisson','derived') else (mean,spread if kind=='normal' else mean+spread))
            if kind in ('poisson','derived'):return np.array([count(p['seed'],9,1,j,tick,float(np.broadcast_to(rate,(4,))[j]),stream) for j in range(4)])
            return mean+spread*np.array([(normal if kind=='normal' else uniform)(p['seed'],9,0,1,j,tick,stream) for j in range(4)])
        def standard(n):
            assert n==2 and len(named)==1
            return np.array([normal(p['seed'],9,0,1,j,tick,named[0]['stream']) for j in range(2)])
        monkeypatch.setattr(np.random,'poisson' if kind=='derived' else kind,sample);monkeypatch.setattr(np.random,'randn',standard)
        out=trainer.step(np.zeros((1,1,1)),[0],**({'initial':'carry'} if tick else {'noise_sequence':9}))
        net.run(dt,namespace={})
        assert next(streams,None) is None
        for obj,names in [(storage,('v','a')),(g,('v','k'))]:
            for name in names:
                cells=bundle.provenance['neuron_state_layout'][obj.name][name]
                np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],obj.variables[name].get_value(),rtol=8e-5,atol=8e-6)
        recorded=np.asarray(monitor.i)[np.isclose(np.asarray(monitor.t/b.second),tick*float(dt),rtol=0.,atol=1e-12)]
        np.testing.assert_array_equal(np.asarray(out['spikes'])[0,-1,4:],np.isin(np.arange(2),recorded).astype(float))
        flags.append(np.asarray(g.not_refractory[:]).copy() if mode!='none' else np.ones(2,dtype=bool))
        path=tmp_path/str(tick);trainer.store(path);trainer=NativeLIFTrainer(p,runner=RUNNER);trainer.restore(path)
    assert len(calls)==6*len(callbacks)
    if mode!='none':assert any(not np.all(flag) for flag in flags), 'must exercise inactive rows'
    np.testing.assert_allclose(storage.a[:],np.array([.1,.2,.3,.4])*.8**len(calls),rtol=1e-12)
    if kind in ('poisson','derived'):assert len(trainer.poisson_state['entries'])==24*len(callbacks)
    assert (out['gpu_dispatches']>0)==(engine!='cpu')


def reference(data,kind,weights,initial=None,anchors=None,override=None):
    """Run Brian's real NumPy code with independent physical storage.

    Finite differences freeze hard events, refractory gates, cached counts and
    TBPTT boundaries. Positive/zero Poisson rates use score/full replay terms.
    """
    from unittest.mock import patch
    from brian2.devices.device import get_device
    net,storage,g,dt,bundle=data;p=bundle.plan;provenance=bundle.provenance
    if g.state_updater.codeobj is None:net.run(0*b.second,namespace={})
    code=str(g.state_updater.codeobj.code.run)
    if anchors is not None:
        lines=code.splitlines()
        lines=[line.split('=')[0]+'= _b2_anchor_flags.copy()' if line.startswith('not_refractory =') else line for line in lines]
        code='\n'.join(lines)
    executable=compile(code,'<independent Brian NumPy refractory>','exec')
    z=np.array(bundle.initial_state if initial is None else initial,dtype=float)
    if initial is None:
        for cell,ref in enumerate(p['dynamic']['initial_parameters']):
            if ref is not None:z[cell]=weights[ref[0]][ref[1]]
    layout=provenance['neuron_state_layout'];a=layout[storage.name]['a'];v=layout[g.name]['v'];k=layout[g.name]['k']
    flag_cells=provenance['refractory_activity_layout'].get(g.name,[]);counter=layout[g.name].get('__refractory_ticks',[])
    last=np.asarray(g.lastspike[:]/b.second).copy() if flag_cells else np.full(2,-1e4);before=[];last_before=[];gates=[];hard=[];margins=[];samples=[];events=[]
    records=[*provenance['whole_neuron_update_layout'][g.name]['draws'],*provenance['whole_neuron_threshold_layout'][g.name]['draws']];named=[row for row in records if row.get('named')]
    callbacks=[row for row in records if not row.get('named')]
    device=get_device()
    def array_name(name):return device.get_array_name(g.variables[name])
    for tick in range(6):
        if anchors is not None and p['tbptt_window'] and tick and tick%p['tbptt_window']==0:
            z=anchors['before'][tick].copy();last=anchors['last_before'][tick].copy()
        before.append(z.copy());last_before.append(last.copy())
        physical_a=z[a].copy();physical_v=z[v].copy();streams=iter(row['stream'] for row in callbacks)
        env=dict(g.state_updater.codeobj.namespace)
        env[array_name('v')]=physical_v;env[array_name('k')]=z[k].astype(np.int32)
        if flag_cells:env[array_name('lastspike')]=last.copy();env[array_name('not_refractory')]=z[flag_cells].astype(bool)
        env[array_name('t')]=np.array([tick*float(dt)])
        for binding in provenance['bindings']:
            if binding['object']!=g.name:continue
            for index,name in enumerate(binding['variables']):
                values=weights[binding['bank']] if binding['kind']=='neuron_array' else [weights[binding['bank']][index]]
                env[array_name(name)]=np.asarray(values,dtype=g.variables[name].dtype)
        env['f']=globals()[kind+'_factory'](physical_a)
        if named:
            env['randn']=lambda indices:np.array([normal(p['seed'],9,0,1,j,tick,named[0]['stream']) for j in range(2)])
        if anchors is not None:env['_b2_anchor_flags']=anchors['gates'][tick]
        def sample(*args,**kwargs):
            stream=next(streams);assert kwargs['size']==4
            if kind in ('poisson','derived'):
                rates=np.broadcast_to(args[0],(4,));draws=[]
                for j,rate in enumerate(rates):
                    value=count(p['seed'],9,1,j,tick,float(rate),stream)
                    if anchors is not None:value=next(row[4] for row in anchors['samples'] if row[:3]==(tick,stream,j))
                    if override==(tick,stream,j):value+=1
                    samples.append((tick,stream,j,float(rate),value));draws.append(value)
                return np.array(draws)
            return args[0]+(args[1] if kind=='normal' else args[1]-args[0])*np.array([(normal if kind=='normal' else uniform)(p['seed'],9,0,1,j,tick,stream) for j in range(4)])
        with patch.object(np.random,'poisson' if kind=='derived' else kind,sample):
            exec(executable,env)
            # The threshold's input arrays are the actual physical writeback,
            # and its closure sees the updater's preceding capture mutation.
            margin=globals()[kind+'_factory'](physical_a)(physical_v,z[k].astype(np.int32),env[array_name('mean')][0],env[array_name('spread')][0])-.5
        assert next(streams,None) is None
        z[a]=physical_a;z[v]=physical_v;gate=env[array_name('not_refractory')].astype(bool) if flag_cells else np.ones(2,dtype=bool);gates.append(gate.copy())
        event=((margin>0)&gate).astype(float);margins.append(margin.copy());hard.append(event.copy())
        if anchors is not None:event=anchors['hard'][tick]+gate*p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(anchors['margins'][tick]))**2*(margin-anchors['margins'][tick])
        events.append(event.copy());z[v]-=.5*event
        control=hard[-1] if anchors is None else anchors['hard'][tick]
        last=np.where(control.astype(bool),tick*float(dt),last)
        if flag_cells:z[flag_cells]=gate&~control.astype(bool)
        if flag_cells:z[counter]=np.where(control.astype(bool),max(p['refractory'][1]['steps']-1,0),np.maximum(z[counter]-1,0))
    logits=np.array(events).mean(0)*p['logit_scale'];loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
    return loss,z,dict(before=before,last_before=last_before,gates=gates,margins=margins,hard=hard,samples=samples,events=events)


@pytest.mark.parametrize('kind,method,mode',[('normal','euler','none'),('normal','euler','fixed'),('uniform','rk2','duration'),('derived','rk4','boolean'),('poisson','heun','fixed'),('normal','milstein','duration')])
@pytest.mark.parametrize('window',[None,2])
def test_whole_phase_all_bank_initial_vjps(engine,kind,method,mode,window):
    mpi(2);data=model(engine,kind,method,mode,window);bundle=data[4];x=np.zeros((1,6,1))
    loss,z,anchors=reference(data,kind,bundle.weights)
    out=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).gradients(x,[0],noise_sequence=9)
    assert out['loss']==pytest.approx(loss,abs=8e-6)
    cells=[*bundle.provenance['neuron_state_layout'][data[1].name]['a'],*bundle.provenance['neuron_state_layout'][data[2].name]['v']]
    np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],z[cells],rtol=8e-5,atol=8e-6)
    np.testing.assert_array_equal(np.asarray(out['spikes'])[0,:,4:],anchors['events'])
    def expected(high,low,baseline,base_loss):
        derivative=(high[0]-low[0])/2e-6;score=0.
        for base,up,down in zip(baseline['samples'],high[2]['samples'],low[2]['samples']):
            tick,stream,j,rate,value=base
            signal=base_loss*(value/rate-1.) if rate else reference(data,kind,bundle.weights,override=(tick,stream,j))[0]-base_loss
            score+=signal*(up[3]-down[3])/2e-6
        return derivative+score
    for bank,values in enumerate(bundle.weights):
        for index in range(len(values)):
            hi=copy.deepcopy(bundle.weights);lo=copy.deepcopy(bundle.weights);hi[bank][index]+=1e-6;lo[bank][index]-=1e-6
            high=reference(data,kind,hi,anchors=anchors);low=reference(data,kind,lo,anchors=anchors)
            assert out['gradients'][bank][index]==pytest.approx(expected(high,low,anchors,loss),rel=3e-3,abs=2e-5),(bank,index)
    initial=np.array(bundle.initial_state,dtype=float)
    explicit=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).gradients(x,[0],initial=initial[None],noise_sequence=9)
    loss,_,anchors=reference(data,kind,bundle.weights,initial=initial)
    for index in range(len(initial)):
        if bundle.plan['dynamic']['detached'][index] or index in bundle.plan['dynamic']['integer_states']:
            assert explicit['initial_state_gradients'][0][index]==0.;continue
        hi=initial.copy();lo=initial.copy();hi[index]+=1e-6;lo[index]-=1e-6
        high=reference(data,kind,bundle.weights,initial=hi,anchors=anchors);low=reference(data,kind,bundle.weights,initial=lo,anchors=anchors)
        assert explicit['initial_state_gradients'][0][index]==pytest.approx(expected(high,low,anchors,loss),rel=3e-3,abs=2e-5),index


@pytest.mark.parametrize('method',['euler','heun'])
@pytest.mark.parametrize('mode',['none','fixed'])
@pytest.mark.parametrize('window',[None,2])
def test_whole_phase_zero_state_score(engine,method,mode,window,monkeypatch):
    data=model(engine,'derived',method,mode,window);bundle=data[4]
    row=next(row for row in bundle.provenance['bindings'] if row['object']==data[2].name and 'mean' in row['variables'])
    bundle.weights[row['bank']][row['variables'].index('mean')]=0.
    cell=bundle.provenance['neuron_state_layout'][data[1].name]['a'][3]
    bundle.initial_state[cell]=0.;bundle.plan['dynamic']['initial'][cell]=0.
    voltage=bundle.provenance['neuron_state_layout'][data[2].name]['v'][0]
    stream=bundle.provenance['whole_neuron_threshold_layout'][data[2].name]['draws'][0]['stream']
    for value in np.arange(.15,.35,.0005):
        bundle.initial_state[voltage]=float(value);bundle.plan['dynamic']['initial'][voltage]=float(value)
        loss,_,anchors=reference(data,'derived',bundle.weights)
        weak=reference(data,'derived',bundle.weights,override=(5,stream,3))[0]-loss
        if abs(weak)>1e-5 and min(abs(m[0]) for m in anchors['margins'])>1e-5:break
    else:raise AssertionError('zero-rate threshold count must change complete phase replay')
    monkeypatch.setattr(__import__(__name__),'model',lambda *args,**kwargs:data)
    test_whole_phase_all_bank_initial_vjps(engine,'derived',method,mode,window)


@pytest.mark.parametrize('kind',['normal','uniform','poisson','derived'])
@pytest.mark.parametrize('mode',['none','fixed','duration','boolean'])
def test_whole_phase_threshold_failure_atomic(engine,kind,mode,tmp_path):
    mpi(2);data=model(engine,kind,'heun',mode,threshold_parameters=True);bundle=data[4]
    p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights)
    trainer=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER);x=np.zeros((1,1,1))
    trainer.step(x,[0],noise_sequence=9);path=tmp_path/'prefix';trainer.store(path)
    trainer=NativeLIFTrainer(p,runner=RUNNER);trainer.restore(path)
    # The updater remains valid and executes first; failure is in the later
    # threshold's independently bound domain, after all preceding state work.
    name='threshold_mean' if kind in ('poisson','derived') else 'threshold_spread'
    row=next(row for row in bundle.provenance['bindings'] if row['object']==data[2].name and name in row['variables'])
    trainer.state['weights'][row['bank']][row['variables'].index(name)]=float('inf') if kind=='uniform' else -2.
    before=copy.deepcopy((trainer.state,trainer.neuron_state,trainer.clock_state,trainer.poisson_state,trainer.elapsed_ticks,trainer.next_noise_sequence))
    for _ in range(2):
        with pytest.raises(ValueError):trainer.step(x,[0],initial='carry')
        assert (trainer.state,trainer.neuron_state,trainer.clock_state,trainer.poisson_state,trainer.elapsed_ticks,trainer.next_noise_sequence)==before


@pytest.mark.parametrize('kind',['normal','derived'])
@pytest.mark.parametrize('mode',['none','fixed'])
def test_whole_phase_threshold_index_atomic(engine,kind,mode,tmp_path):
    mpi(2);data=model(engine,kind,'milstein',mode,threshold_parameters=True);bundle=data[4]
    p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights)
    trainer=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER);x=np.zeros((1,1,1))
    trainer.step(x,[0],noise_sequence=9);path=tmp_path/'prefix';trainer.store(path)
    trainer=NativeLIFTrainer(p,runner=RUNNER);trainer.restore(path)
    cell=bundle.provenance['neuron_state_layout'][data[2].name]['threshold_k'][0]
    trainer.neuron_state[0][cell]=8.
    before=copy.deepcopy((trainer.state,trainer.neuron_state,trainer.clock_state,trainer.poisson_state,trainer.elapsed_ticks,trainer.next_noise_sequence))
    for _ in range(2):
        with pytest.raises(ValueError):trainer.step(x,[0],initial='carry')
        assert (trainer.state,trainer.neuron_state,trainer.clock_state,trainer.poisson_state,trainer.elapsed_ticks,trainer.next_noise_sequence)==before
