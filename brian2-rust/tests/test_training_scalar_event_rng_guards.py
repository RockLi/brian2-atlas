"""A scalar refractory Python if must guard the lifted body and sampler."""
import copy
import brian2 as b
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer,lower_brian_dynamic_training
from test_training_guarded_scalar_captures import model as guarded_model
import test_training_scalar_event_rng_functions as factories
from test_training_integer_ir import engine
from test_training_poisson_zero_vjp import mpi
from test_training_event_noise import draw
from test_training_batch_poisson_captures import observation_stream
from test_native_training import RUNNER


def model(backend,ranks,kind,mixed=False,zero=False):
    net,g,old,dt,_,x=guarded_model(backend,ranks)
    net.remove(old);code=('h=f(h,mean,spread);' if mixed else '')+'v_post+=gain*f(v_post,mean,spread)'
    syn=b.Synapses(old.source,g,'h:1\ngain:1 (constant)\nmean:1 (constant,shared)\nspread:1 (constant,shared)',on_pre=code,dt=dt)
    syn.connect(i=[0,0],j=[0,0]);syn.h=[.173,.113];syn.gain=[.19,.11];syn.mean=1.3 if kind=='poisson' else .04;syn.spread=.13
    if zero:syn.mean=0.
    syn.namespace['f']=b.Function(getattr(factories,kind)(g.variables['u'].get_value()),arg_units=[1,1,1],return_unit=1,stateless=False);net.add(syn)
    hidden=next(o for o in net.objects if isinstance(o,b.NeuronGroup) and o is not g)
    bundle=lower_brian_dynamic_training(net,input_group=syn.source,layers=[hidden,g],backend=backend,mpi_ranks=ranks,detach_reset=False,trainable_synapse_parameters={syn.name:['h','gain','mean','spread']})
    assert bundle.provenance['event_callback_modes'][syn.pre.name]['mode']=='scalar'
    return net,g,syn,dt,bundle,x


def compare(out,bundle,g,syn):
    for obj,key,names in [(g,'neuron_state_layout',['v','u']),(syn,'dynamic_state_layout',['h'])]:
        for name in names:
            if name not in bundle.provenance[key][obj.name]:continue
            cells=bundle.provenance[key][obj.name][name]
            np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],obj.variables[name].get_value(),rtol=8e-5,atol=8e-6)


@pytest.mark.parametrize('kind',['normal','uniform','poisson'])
@pytest.mark.parametrize('mixed',[False,True])
@pytest.mark.parametrize('ranks',[None,2])
def test_scalar_guard_rng_original_restore(engine,kind,mixed,ranks,tmp_path,monkeypatch):
    mpi(ranks);net,g,syn,dt,bundle,x=model(engine,ranks,kind,mixed)
    p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights);trainer=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER)
    domain=bundle.provenance['scheduled_noise_domains'][syn.pre.name];calls=[]
    for tick in range(4):
        serial=0
        def sample(*args,**kwargs):
            nonlocal serial
            free=bool(g.not_refractory[0]);assert free or mixed
            per_edge=2 if mixed and free else 1;edge=serial//per_edge;stream=serial%per_edge if mixed else 0
            assert edge<2;calls.append((tick,edge,stream,free));serial+=1
            mean=float(syn.mean[:]);spread=float(syn.spread[:])
            assert all(np.ndim(value)==0 for value in args)
            np.testing.assert_allclose(args,(mean,) if kind=='poisson' else (mean,spread if kind=='normal' else mean+spread))
            assert kwargs==({'size':None} if kind=='uniform' else {})
            if kind=='poisson':return int(observation_stream(p['seed'],9,domain,edge,tick,mean,stream))
            return float(mean+spread*draw('randn' if kind=='normal' else 'rand',p['seed'],9,0,domain,edge,tick,None,stream))
        monkeypatch.setattr(np.random,kind,sample)
        out=trainer.step(x[:,tick:tick+1],[0],**({'initial':'carry'} if tick else {'noise_sequence':9}));net.run(dt,namespace={});compare(out,bundle,g,syn)
        path=tmp_path/str(tick);trainer.store(path);trainer=NativeLIFTrainer(p,runner=RUNNER);trainer.restore(path)
    assert calls and (mixed or calls[0][0]>0)
    if kind=='poisson':assert len(trainer.poisson_state['entries'])==len(calls)


@pytest.mark.parametrize('kind',['normal','poisson'])
@pytest.mark.parametrize('ranks',[None,2])
def test_scalar_guard_rng_invalid_only_when_active(engine,kind,ranks,tmp_path):
    mpi(ranks);net,g,syn,dt,bundle,x=model(engine,ranks,kind)
    p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights);trainer=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER)
    trainer.step(x[:,:1],[0],noise_sequence=9);net.run(dt,namespace={})
    path=tmp_path/'prefix';trainer.store(path);trainer=NativeLIFTrainer(p,runner=RUNNER);trainer.restore(path)
    name='mean' if kind=='poisson' else 'spread';bank=next(row['bank'] for row in bundle.provenance['bindings'] if row['variables']==[name])
    trainer.state['weights'][bank][0]=-1.;setattr(syn,name,-1.)
    out=trainer.step(x[:,1:2],[0],initial='carry');net.run(dt,namespace={});compare(out,bundle,g,syn)
    failed=False
    for tick in (2,3):
        before=copy.deepcopy((trainer.state,trainer.neuron_state,trainer.clock_state,trainer.poisson_state,trainer.elapsed_ticks,trainer.next_noise_sequence))
        try:trainer.step(x[:,tick:tick+1],[0],initial='carry')
        except ValueError:
            assert (trainer.state,trainer.neuron_state,trainer.clock_state,trainer.poisson_state,trainer.elapsed_ticks,trainer.next_noise_sequence)==before
            with pytest.raises(b.BrianObjectException):net.run(dt,namespace={})
            failed=True;break
        net.run(dt,namespace={})
    assert failed


def reference(data,kind,mixed,weights,window,initial=None,anchors=None,override=None):
    _,g,syn,_,bundle,_=data;p=bundle.plan;z=np.array(bundle.initial_state if initial is None else initial,float)
    if initial is None:
        for cell,ref in enumerate(p['dynamic']['initial_parameters']):
            if ref is not None:z[cell]=weights[ref[0]][ref[1]]
    layout=bundle.provenance['neuron_state_layout'][g.name];v=layout['v'];u=layout['u'];counter=layout['__refractory_ticks'];activity=bundle.provenance['refractory_activity_layout'][g.name]
    h=bundle.provenance['dynamic_state_layout'][syn.name].get('h')
    banks={row['variables'][0]:row['bank'] for row in bundle.provenance['bindings'] if row['object']==syn.name};domain=bundle.provenance['scheduled_noise_domains'][syn.pre.name]
    before=[];margins=[];hard_rows=[];free_rows=[];spikes=[];samples=[]
    for tick in range(4):
        if anchors is not None and window and tick and tick%window==0:z=anchors['before'][tick].copy()
        before.append(z.copy());free=z[counter]<.5;z[counter]=np.maximum(z[counter]-1,0.)
        if anchors is not None:free=anchors['free'][tick]
        margin=z[v]-.5;hard=(margin>0)&free;event=hard.astype(float)
        if anchors is not None:event=anchors['hard'][tick].astype(float)+free*p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(anchors['margins'][tick]))**2*(margin-anchors['margins'][tick]);hard=anchors['hard'][tick]
        margins.append(margin.copy());hard_rows.append(hard.copy());free_rows.append(free.copy());spikes.append(event.copy());flag=free&~hard;z[activity]=flag
        def response(value,edge,stream):
            mean=weights[banks['mean']][0];spread=weights[banks['spread']][0]
            if kind=='poisson':
                count=observation_stream(p['seed'],9,domain,edge,tick,float(mean),stream)
                if anchors is not None:count=next(c for t,e,s,r,c in anchors['samples'] if (t,e,s)==(tick,edge,stream))
                if override==(tick,edge,stream):count+=1
                samples.append((tick,edge,stream,float(mean),count));noise=count
            else:noise=mean+spread*draw('randn' if kind=='normal' else 'rand',p['seed'],9,0,domain,edge,tick,None,stream)
            z[u]*=.8
            return .7*value+.1*noise
        for edge in range(2):
            local_v=z[v[0]]
            if mixed:local_h=response(z[h[edge]],edge,0)
            if flag[0]:local_v+=weights[banks['gain']][edge]*response(local_v,edge,int(mixed))
            z[v[0]]=local_v
            if mixed:z[h[edge]]=local_h
        z[v]-=.5*event;z[counter]=np.where(hard,1.,z[counter])
    logits=np.array(spikes).mean(0)*p['logit_scale'];loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
    return loss,z,dict(before=before,margins=margins,hard=hard_rows,free=free_rows,samples=samples)


@pytest.mark.parametrize('kind,zero',[('normal',False),('uniform',False),('poisson',False),('poisson',True)])
@pytest.mark.parametrize('mixed',[False,True])
@pytest.mark.parametrize('window',[None,2])
@pytest.mark.parametrize('ranks',[None,2])
def test_scalar_guard_rng_all_vjps(engine,kind,zero,mixed,window,ranks):
    mpi(ranks);data=model(engine,ranks,kind,mixed,zero);_,g,syn,_,bundle,x=data;p=copy.deepcopy(bundle.plan);p['tbptt_window']=window
    out=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER).gradients(x,[0],noise_sequence=9);args=(data,kind,mixed,bundle.weights,window);loss,z,anchors=reference(*args)
    cells=[*bundle.provenance['neuron_state_layout'][g.name]['v'],*bundle.provenance['neuron_state_layout'][g.name]['u'],*bundle.provenance['dynamic_state_layout'][syn.name].get('h',[])]
    np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],z[cells],rtol=8e-5,atol=8e-6);assert out['loss']==pytest.approx(loss,abs=8e-6)
    mean=next(row['bank'] for row in bundle.provenance['bindings'] if row['variables']==['mean'])
    weak=sum(loss*(count/rate-1.) if rate else reference(*args,override=(tick,edge,stream))[0]-loss for tick,edge,stream,rate,count in anchors['samples']) if kind=='poisson' else 0.
    if kind=='poisson':assert anchors['samples'] and abs(weak)>1e-5
    for bank,row in enumerate(bundle.weights):
        for index in range(len(row)):
            hi=copy.deepcopy(bundle.weights);lo=copy.deepcopy(bundle.weights);hi[bank][index]+=1e-6;lo[bank][index]-=1e-6
            fd=(reference(data,kind,mixed,hi,window,anchors=anchors)[0]-reference(data,kind,mixed,lo,window,anchors=anchors)[0])/2e-6
            assert out['gradients'][bank][index]==pytest.approx(fd+(weak if bank==mean else 0.),rel=1e-3,abs=1e-5),(bank,index)
    for index in range(len(z)):
        if p['dynamic']['detached'][index] or index in p['dynamic']['integer_states']:assert out['initial_state_gradients'][0][index]==0;continue
        hi=np.array(bundle.initial_state);lo=hi.copy();hi[index]+=1e-6;lo[index]-=1e-6
        fd=(reference(*args,initial=hi,anchors=anchors)[0]-reference(*args,initial=lo,anchors=anchors)[0])/2e-6
        assert out['initial_state_gradients'][0][index]==pytest.approx(fd,rel=1e-3,abs=1e-5),index
