"""Scalar NumPy synapse loops keep row RNG and whole neuron captures."""
import copy
import brian2 as b
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer,lower_brian_dynamic_training
import test_training_scalar_event_rng_functions as factories
from test_training_regular_callback_rng import count
from test_training_stochastic import normal
from test_training_uniform import uniform
from test_training_integer_ir import engine
from test_training_poisson_zero_vjp import mpi
from test_native_training import RUNNER


def model(backend,kind,capture,when='start',window=None,masked=False):
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='numpy';dt=.2*b.ms
    source=b.SpikeGeneratorGroup(1,[],np.array([])*dt,dt=dt)
    a=b.NeuronGroup(2,'dv/dt=0/second:1',threshold='v>100',reset='v=0',method='euler',dt=dt)
    c=b.NeuronGroup(2,'dv/dt=(gain*q+.01)/ms:1\nq:1\ngain:1 (constant,shared)',threshold='v>.5',reset='v-=.5',method='euler',dt=dt)
    c.v=[.59,.38];c.gain=.2
    syn=b.Synapses(a,c,'h:1\nmean:1 (constant,shared)\nspread:1 (constant,shared)'+('\nw:1 (constant)\nq_post=w*h:1 (summed)' if masked else '\nq_post=h:1 (summed)'),dt=dt)
    syn.connect(i=[0,1,0],j=[0,0,1]);syn.h=[.173,.219,.26]
    if masked:syn.w=[.4,.3,.2]
    syn.mean=1.3 if kind in ('poisson','mixed') else .04;syn.spread=.13
    function=getattr(factories,kind)(c.variables['v'].get_value()) if capture else getattr(factories,'pure_'+kind)
    syn.namespace['f']=b.Function(function,arg_units=[1,1,1],return_unit=1,stateless=False)
    runner=syn.run_regularly('v_post=.05*f(h,mean,spread)+.9*v_post;h+=.01*randn()',when=when,dt=dt)
    net=b.Network(source,a,c,syn)
    bundle=lower_brian_dynamic_training(net,input_group=source,layers=[a,c],backend=backend,mpi_ranks=2,tbptt_window=window,detach_reset=False,
        trainable_neuron_parameters={c.name:['gain']},trainable_synapse_parameters={syn.name:['mean','spread','h']+(['w'] if masked else [])})
    assert bundle.provenance['regular_runner_layout'][runner.name]['numpy_mode']=='scalar'
    return net,(a,c),syn,runner,dt,bundle,np.zeros((1,4,1))


@pytest.mark.parametrize('kind,capture',[('normal',False),('normal',True),('uniform',False),('poisson',True),('mixed',True)])
@pytest.mark.parametrize('when',['start','after_groups'])
def test_synaptic_scalar_fallback_original_restore(engine,kind,capture,when,monkeypatch,tmp_path):
    mpi(2);net,groups,syn,runner,dt,bundle,x=model(engine,kind,capture,when)
    domain=bundle.provenance['regular_runner_layout'][runner.name]['noise_domain'];p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights)
    trainer=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER)
    order=(['normal','uniform','poisson'] if kind=='mixed' else [kind])+['randn'];calls=[]
    for tick in range(4):
        serial=0
        def sample(which,*args,**kwargs):
            nonlocal serial
            entity,stream=divmod(serial,len(order));assert which==order[stream] and entity<3;serial+=1;calls.append((tick,entity,stream,which))
            if which=='randn':return normal(p['seed'],9,0,domain,entity,tick,stream)
            mean=float(syn.mean[:]);spread=float(syn.spread[:]);assert all(np.ndim(a)==0 for a in args)
            np.testing.assert_allclose(args,(mean,) if which=='poisson' else (mean,spread if which=='normal' else mean+spread))
            if which=='poisson':return int(count(p['seed'],9,domain,entity,tick,mean,stream))
            return mean+spread*(normal if which=='normal' else uniform)(p['seed'],9,0,domain,entity,tick,stream)
        for which in set(order):monkeypatch.setattr(np.random,which,lambda *args,_which=which,**kwargs:sample(_which,*args,**kwargs))
        out=trainer.step(x[:,tick:tick+1],[0],**({'initial':'carry'} if tick else {'noise_sequence':9}));net.run(dt,namespace={})
        for group in groups:
            for name in ('v','q'):
                if name not in bundle.provenance['neuron_state_layout'][group.name]:continue
                cells=bundle.provenance['neuron_state_layout'][group.name][name]
                np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],group.variables[name].get_value(),rtol=8e-5,atol=8e-6,err_msg=name)
        cells=bundle.provenance['dynamic_state_layout'][syn.name]['h']
        np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],syn.h[:],rtol=8e-5,atol=8e-6)
        path=tmp_path/str(tick);trainer.store(path);trainer=NativeLIFTrainer(p,runner=RUNNER);trainer.restore(path)
    assert len(calls)==12*len(order)
    if kind in ('poisson','mixed'):assert len(trainer.poisson_state['entries'])==12


def parameter(data,weights,name):
    obj=data[2].name if name in ('mean','spread') else data[1][1].name
    row=next(row for row in data[5].provenance['bindings'] if row['object']==obj and name in row['variables'])
    return weights[row['bank']][row['variables'].index(name)]


def reference(data,kind,capture,weights,initial=None,anchors=None,override=None):
    _,groups,syn,runner,_,bundle,_=data;p=bundle.plan
    z=np.array(bundle.initial_state if initial is None else initial,dtype=float)
    if initial is None:
        for cell,ref in enumerate(p['dynamic']['initial_parameters']):
            if ref is not None:z[cell]=weights[ref[0]][ref[1]]
    ns=bundle.provenance['neuron_state_layout'];ss=bundle.provenance['dynamic_state_layout'][syn.name]
    v=ns[groups[1].name]['v'];q=ns[groups[1].name]['q'];h=ss['h'];domain=bundle.provenance['regular_runner_layout'][runner.name]['noise_domain']
    kinds=['normal','uniform','poisson'] if kind=='mixed' else [kind]
    mean=parameter(data,weights,'mean');spread=parameter(data,weights,'spread')
    before=[];margins=[];hard=[];events=[];samples=[]
    for tick in range(4):
        if anchors is not None and p['tbptt_window'] and tick and tick%p['tbptt_window']==0:z=anchors['before'][tick].copy()
        before.append(z.copy())
        def update():
            for entity,target in enumerate((0,0,1)):
                # NumPy scalar fallback loads a row before calling Python.
                peer=z[v[target]];operand=z[h[entity]]
                if capture:z[v]*=.8;operand*=.7
                total=0.
                for stream,which in enumerate(kinds):
                    if which=='poisson':
                        value=count(p['seed'],9,domain,entity,tick,float(mean),stream)
                        if anchors is not None:value=next(c for t,j,s,r,c in anchors['samples'] if (t,j,s)==(tick,entity,stream))
                        if override==(tick,entity,stream):value+=1
                        samples.append((tick,entity,stream,float(mean),value))
                    else:value=mean+spread*(normal if which=='normal' else uniform)(p['seed'],9,0,domain,entity,tick,stream)
                    total+=value
                z[v[target]]=.05*(operand+.1*total)+.9*peer
                z[h[entity]]+=.01*normal(p['seed'],9,0,domain,entity,tick,len(kinds))
        if runner.when=='start':update()
        z[q]=[z[h[0]]+z[h[1]],z[h[2]]]
        z[v]+=.2*(parameter(data,weights,'gain')*z[q]+.01)
        if runner.when=='after_groups':update()
        margin=z[v]-.5;event=(margin>0).astype(float);margins.append(margin.copy());hard.append(event.copy())
        if anchors is not None:event=anchors['hard'][tick]+p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(anchors['margins'][tick]))**2*(margin-anchors['margins'][tick])
        events.append(event.copy());z[v]-=.5*event
    logits=np.array(events).mean(0)*p['logit_scale'];loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
    return loss,z,dict(before=before,margins=margins,hard=hard,samples=samples)


@pytest.mark.parametrize('kind,capture,zero',[('normal',True,False),('mixed',True,False),('poisson',True,True)])
@pytest.mark.parametrize('when',['start','after_groups'])
@pytest.mark.parametrize('window',[None,2])
def test_synaptic_scalar_fallback_all_bank_initial_vjps(engine,kind,capture,zero,when,window):
    mpi(2);data=model(engine,kind,capture,when,window);bundle=data[5]
    row=next(row for row in bundle.provenance['bindings'] if row['object']==data[2].name and 'mean' in row['variables']);mean=(row['bank'],row['variables'].index('mean'))
    if zero:
        bundle.weights[mean[0]][mean[1]]=0.;cell=bundle.provenance['neuron_state_layout'][data[1][1].name]['v'][0]
        for voltage in np.arange(.35,1.25,.001):
            bundle.initial_state[cell]=float(voltage);bundle.plan['dynamic']['initial'][cell]=float(voltage)
            loss,_,a=reference(data,kind,capture,bundle.weights)
            weak=sum(reference(data,kind,capture,bundle.weights,override=(t,j,s))[0]-loss for t,j,s,r,c in a['samples'])
            if abs(weak)>1e-5 and min(abs(m[0]) for m in a['margins'])>1e-5:break
        else:raise AssertionError('zero fixture requires a nonzero counterfactual')
    loss,z,anchors=reference(data,kind,capture,bundle.weights)
    out=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).gradients(data[6],[0],noise_sequence=9)
    assert out['loss']==pytest.approx(loss,abs=8e-6)
    ns=bundle.provenance['neuron_state_layout'];cells=sorted(set(ns[data[1][0].name]['v']+ns[data[1][1].name]['v']+ns[data[1][1].name]['q']+bundle.provenance['dynamic_state_layout'][data[2].name]['h']))
    np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],z[cells],rtol=8e-5,atol=8e-6)
    np.testing.assert_array_equal(np.asarray(out['spikes'])[0,:,2:],anchors['hard'])
    weak=sum(loss*(c/r-1.) if r else reference(data,kind,capture,bundle.weights,override=(t,j,s))[0]-loss for t,j,s,r,c in anchors['samples'])
    if zero:assert abs(weak)>1e-5
    for bank,values in enumerate(bundle.weights):
        for index in range(len(values)):
            hi=copy.deepcopy(bundle.weights);lo=copy.deepcopy(bundle.weights);hi[bank][index]+=1e-6;lo[bank][index]-=1e-6
            fd=0. if zero and (bank,index)==mean else (reference(data,kind,capture,hi,anchors=anchors)[0]-reference(data,kind,capture,lo,anchors=anchors)[0])/2e-6
            assert out['gradients'][bank][index]==pytest.approx(fd+(weak if (bank,index)==mean else 0.),rel=3e-3,abs=2e-5),(bank,index)
    initial=np.array(bundle.initial_state,dtype=float)
    explicit=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).gradients(data[6],[0],initial=initial[None],noise_sequence=9)
    _,_,anchors=reference(data,kind,capture,bundle.weights,initial=initial)
    for index in range(len(initial)):
        if bundle.plan['dynamic']['detached'][index] or index in bundle.plan['dynamic']['integer_states']:
            assert explicit['initial_state_gradients'][0][index]==0.;continue
        hi=initial.copy();lo=initial.copy();hi[index]+=1e-6;lo[index]-=1e-6
        fd=(reference(data,kind,capture,bundle.weights,initial=hi,anchors=anchors)[0]-reference(data,kind,capture,bundle.weights,initial=lo,anchors=anchors)[0])/2e-6
        assert explicit['initial_state_gradients'][0][index]==pytest.approx(fd,rel=3e-3,abs=2e-5),index


@pytest.mark.parametrize('kind',['normal','poisson'])
@pytest.mark.parametrize('when',['start','after_groups'])
def test_synaptic_scalar_fallback_invalid_restore_atomic(engine,kind,when,tmp_path):
    mpi(2);data=model(engine,kind,True,when);bundle=data[5];x=data[6]
    p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights)
    trainer=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER);trainer.step(x[:,:1],[0],noise_sequence=9)
    path=tmp_path/'prefix';trainer.store(path);trainer=NativeLIFTrainer(p,runner=RUNNER);trainer.restore(path)
    name='mean' if kind=='poisson' else 'spread'
    row=next(row for row in bundle.provenance['bindings'] if row['object']==data[2].name and name in row['variables'])
    trainer.state['weights'][row['bank']][row['variables'].index(name)]=-1.
    before=copy.deepcopy((trainer.state,trainer.neuron_state,trainer.clock_state,trainer.poisson_state,trainer.elapsed_ticks,trainer.next_noise_sequence))
    with pytest.raises(ValueError):trainer.step(x[:,1:2],[0],initial='carry')
    assert (trainer.state,trainer.neuron_state,trainer.clock_state,trainer.poisson_state,trainer.elapsed_ticks,trainer.next_noise_sequence)==before


def test_synaptic_scalar_fallback_prune_regrow_rng_ownership(engine,tmp_path):
    mpi(2);data=model(engine,'poisson',True,masked=True);bundle=data[5];syn=data[2];c=data[1][1];x=data[6]
    p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights)
    w=next(row['bank'] for row in bundle.provenance['bindings'] if row['object']==syn.name and row['variables']==['w'])
    mean=next(row['bank'] for row in bundle.provenance['bindings'] if row['object']==syn.name and row['variables']==['mean'])
    hbank=next(row['bank'] for row in bundle.provenance['bindings'] if row['object']==syn.name and row['variables']==['h'])
    h=bundle.provenance['dynamic_state_layout'][syn.name]['h'];v=bundle.provenance['neuron_state_layout'][c.name]['v'];q=bundle.provenance['neuron_state_layout'][c.name]['q']
    assert set(map(tuple,p['dynamic']['migration']['controlled_masks']))=={(w,j) for j in range(3)}
    trainer=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER);trainer.step(x[:,:1],[0],noise_sequence=9)
    live=np.asarray(trainer.neuron_state)[0].copy();clock=copy.deepcopy((trainer.clock_state,trainer.elapsed_ticks,trainer.next_noise_sequence,trainer.state['step']))
    masks=copy.deepcopy(trainer.plan['masks']);masks[w]=[0.,0.,0.];trainer.update_mask(masks)
    assert clock==(trainer.clock_state,trainer.elapsed_ticks,trainer.next_noise_sequence,trainer.state['step'])
    np.testing.assert_allclose(np.asarray(trainer.neuron_state)[0,v],live[v],atol=0)
    np.testing.assert_array_equal(np.asarray(trainer.neuron_state)[0,h],0.)
    trainer.state['weights'][mean][0]=-1.
    entries=copy.deepcopy(trainer.poisson_state['entries'])
    trainer.step(x[:,1:2],[0],initial='carry')
    assert trainer.poisson_state['entries']==entries,'inactive rows must not draw or validate their rate'
    np.testing.assert_array_equal(np.asarray(trainer.neuron_state)[0,h+q],0.)
    trainer.state['weights'][mean][0]=1.3
    path=tmp_path/'pruned';trainer.store(path);trainer=NativeLIFTrainer(trainer.plan,runner=RUNNER);trainer.restore(path)
    masks=copy.deepcopy(trainer.plan['masks']);masks[w]=[0.,1.,0.];before_clock=copy.deepcopy((trainer.clock_state,trainer.elapsed_ticks,trainer.next_noise_sequence,trainer.state['step']))
    trainer.update_mask(masks,growth_weight=.4)
    assert before_clock==(trainer.clock_state,trainer.elapsed_ticks,trainer.next_noise_sequence,trainer.state['step'])
    np.testing.assert_array_equal(np.asarray(trainer.neuron_state)[0,h],[0.,trainer.state['weights'][hbank][1],0.])
    before=np.asarray(trainer.neuron_state)[0].copy();expected=before.copy();domain=bundle.provenance['regular_runner_layout'][data[3].name]['noise_domain']
    peer=before[v[0]];operand=before[h[1]]
    sampled=count(p['seed'],9,domain,1,2,1.3,0)
    expected[v]*=.8;expected[v[0]]=.9*peer+.05*(.7*operand+.1*sampled)
    expected[h[1]]+=.01*normal(p['seed'],9,0,domain,1,2,1)
    expected[q]=[.4*expected[h[1]],0.]
    expected[v]+=.2*(parameter(data,trainer.state['weights'],'gain')*expected[q]+.01)
    expected[v]-=.5*(expected[v]>.5)
    out=trainer.step(x[:,2:3],[0],initial='carry');cells=v+q+h
    np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],expected[cells],rtol=8e-5,atol=8e-6)
    old={str(entry['identity']):entry for entry in entries}
    current={str(entry['identity']):entry for entry in trainer.poisson_state['entries']}
    assert all(current[key]==entry for key,entry in old.items())
    added=[entry for key,entry in current.items() if key not in old]
    assert len(added)==1,'only the regrown row owns a new count'
    assert added[0]['identity']['site']['entity']==1 and added[0]['identity']['instant']==2
