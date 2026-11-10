"""Real scalar fallback calls sample separately for each FIFO arrival."""
import copy
import brian2 as b
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer,lower_brian_dynamic_training
from test_training_event_captures import model as capture_model
import test_training_scalar_event_rng_functions as factories
from test_training_integer_ir import engine
from test_training_poisson_zero_vjp import mpi
from test_training_event_noise import draw
from test_training_batch_poisson_captures import observation_stream
from test_native_training import RUNNER


def model(backend,ranks,delay,kind,capture=True,window=None,zero=False):
    net,g,old,dt,_,x=capture_model('voltage',ranks,window,delay,backend)
    net.remove(old)
    syn=b.Synapses(old.source,g,'h:1\ngain:1 (constant)\nmean:1 (constant,shared)\nspread:1 (constant,shared)',on_pre='h=f(v_post,mean,spread);v_post+=gain*h',dt=dt)
    syn.connect(i=[0,0],j=[0,0]);syn.h=[.113,.173];syn.gain=[.11,.19];syn.mean=1.3 if kind in ('poisson','mixed') else .04;syn.spread=.13;syn.delay=delay*dt
    if zero:
        syn.mean=0.
        g.v=[(.4 if capture else .375) if delay else (.334 if capture else .28),.379]
    function=getattr(factories,kind)(g.variables['v'].get_value()) if capture else getattr(factories,'pure_'+kind)
    syn.namespace['f']=b.Function(function,arg_units=[1,1,1],return_unit=1,stateless=False);net.add(syn)
    hidden=next(o for o in net.objects if isinstance(o,b.NeuronGroup) and o is not g)
    bundle=lower_brian_dynamic_training(net,input_group=syn.source,layers=[hidden,g],backend=backend,mpi_ranks=ranks,tbptt_window=window,detach_reset=False,trainable_synapse_parameters={syn.name:['h','gain','mean','spread']})
    assert bundle.provenance['event_callback_modes'][syn.pre.name]['mode']=='scalar'
    return net,g,syn,dt,bundle,x


@pytest.mark.parametrize('kind,capture',[(kind,capture) for kind in ('normal','uniform','poisson','mixed') for capture in (False,True) if capture or kind!='mixed'])
@pytest.mark.parametrize('delay',[0,1])
@pytest.mark.parametrize('ranks',[None,2])
def test_scalar_fifo_rng_original_restore(engine,kind,capture,delay,ranks,tmp_path,monkeypatch):
    mpi(ranks);net,g,syn,dt,bundle,x=model(engine,ranks,delay,kind,capture)
    p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights);trainer=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER)
    domain=bundle.provenance['scheduled_noise_domains'][syn.pre.name];calls=[];kinds=['normal','uniform','poisson'] if kind=='mixed' else [kind]
    for tick in range(4):
        serial=0
        def sample(which,*args,**kwargs):
            nonlocal serial
            assert all(np.ndim(arg)==0 for arg in args)
            edge=serial//len(kinds);stream=serial%len(kinds);assert which==kinds[stream] and edge<2
            mean=float(syn.mean[:]);spread=float(syn.spread[:])
            np.testing.assert_allclose(args,(mean,) if which=='poisson' else (mean,spread if which=='normal' else mean+spread))
            expected_kwargs={'size':None} if which=='uniform' and kind!='mixed' else {};assert kwargs==expected_kwargs
            emitted=tick-delay;assert emitted in (0,2);calls.append((tick,edge,stream,which));serial+=1
            if which=='poisson':return int(observation_stream(p['seed'],9,domain,edge,emitted,mean,stream))
            return float(mean+spread*draw('randn' if which=='normal' else 'rand',p['seed'],9,0,domain,edge,emitted,None,stream))
        for which in kinds:monkeypatch.setattr(np.random,which,lambda *args,_which=which,**kwargs:sample(_which,*args,**kwargs))
        out=trainer.step(x[:,tick:tick+1],[0],**({'initial':'carry'} if tick else {'noise_sequence':9}));net.run(dt,namespace={})
        for obj,key,names in [(g,'neuron_state_layout',['v']),(syn,'dynamic_state_layout',['h'])]:
            for name in names:
                cells=bundle.provenance[key][obj.name][name]
                np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],obj.variables[name].get_value(),rtol=8e-5,atol=8e-6)
        path=tmp_path/str(tick);trainer.store(path);trainer=NativeLIFTrainer(p,runner=RUNNER);trainer.restore(path)
    assert len(calls)==4*len(kinds)
    if 'poisson' in kinds:assert len(trainer.poisson_state['entries'])==4


def reference(data,kind,capture,delay,weights,initial=None,anchors=None,override=None):
    _,g,syn,_,bundle,x=data;p=bundle.plan;z=np.array(bundle.initial_state if initial is None else initial,float)
    if initial is None:
        for cell,ref in enumerate(p['dynamic']['initial_parameters']):
            if ref is not None:z[cell]=weights[ref[0]][ref[1]]
    v=bundle.provenance['neuron_state_layout'][g.name]['v'];h=bundle.provenance['dynamic_state_layout'][syn.name]['h']
    banks={row['variables'][0]:row['bank'] for row in bundle.provenance['bindings'] if row['object']==syn.name}
    routes=bundle.provenance['delay_queues'][syn.pre.name]['new'];domain=bundle.provenance['scheduled_noise_domains'][syn.pre.name]
    before=[];margins=[];hard_rows=[];spikes=[];samples=[]
    for tick in range(4):
        if anchors is not None and p['tbptt_window'] and tick and tick%p['tbptt_window']==0:z=anchors['before'][tick].copy()
        before.append(z.copy());margin=z[v]-.5;event=(margin>0).astype(float);margins.append(margin.copy());hard_rows.append(event.copy())
        if anchors is not None:event=anchors['hard'][tick]+p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(anchors['margins'][tick]))**2*(margin-anchors['margins'][tick])
        spikes.append(event.copy())
        for edge in range(2):
            gate=z[routes[edge]['states'][0]] if delay else x[0,tick,0]
            old=z[v].copy();mean=weights[banks['mean']][0];spread=weights[banks['spread']][0];emitted=tick-delay
            if kind=='poisson':
                noise=observation_stream(p['seed'],9,domain,edge,emitted,float(mean),0)
                if anchors is not None:
                    noise=next((count for t,e,rate,count in anchors['samples'] if (t,e)==(tick,edge)),noise)
                if override==(tick,edge):noise+=1
                if gate>.5:samples.append((tick,edge,float(mean),noise))
            else:noise=mean+spread*draw('randn' if kind=='normal' else 'rand',p['seed'],9,0,domain,edge,emitted,None,0)
            new_h=(.7 if capture else 1.)*old[0]+.1*noise
            if capture:z[v]+=gate*(-.2*z[v])
            z[h[edge]]+=gate*(new_h-z[h[edge]])
            z[v[0]]=old[0]+gate*weights[banks['gain']][edge]*new_h
        for row in routes:
            cells=row['states']
            if cells:z[cells[:-1]]=z[cells[1:]];z[cells[-1]]=x[0,tick,0]
        z[v]-=.5*event
    logits=np.array(spikes).mean(0)*p['logit_scale'];loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
    return loss,z,dict(before=before,margins=margins,hard=hard_rows,samples=samples)


@pytest.mark.parametrize('kind,zero',[('normal',False),('uniform',False),('poisson',False),('poisson',True)])
@pytest.mark.parametrize('capture',[False,True])
@pytest.mark.parametrize('delay',[0,1])
@pytest.mark.parametrize('window',[None,2])
@pytest.mark.parametrize('ranks',[None,2])
def test_scalar_fifo_rng_all_vjps(engine,kind,zero,capture,delay,window,ranks):
    mpi(ranks);data=model(engine,ranks,delay,kind,capture,window,zero);_,g,syn,_,bundle,x=data
    out=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).gradients(x,[0],noise_sequence=9)
    args=(data,kind,capture,delay,bundle.weights);loss,z,anchors=reference(*args)
    np.testing.assert_allclose(np.asarray(out['final_state'])[0],z,rtol=8e-5,atol=8e-6);assert out['loss']==pytest.approx(loss,abs=8e-6)
    mean=next(row['bank'] for row in bundle.provenance['bindings'] if row['variables']==['mean'])
    weak=sum(loss*(count/rate-1.) if rate else reference(*args,override=(tick,edge))[0]-loss for tick,edge,rate,count in anchors['samples']) if kind=='poisson' else 0.
    if kind=='poisson':assert abs(weak)>1e-5 and len(anchors['samples'])==4
    for bank,row in enumerate(bundle.weights):
        for index in range(len(row)):
            hi=copy.deepcopy(bundle.weights);lo=copy.deepcopy(bundle.weights);hi[bank][index]+=1e-6;lo[bank][index]-=1e-6
            fd=(reference(data,kind,capture,delay,hi,anchors=anchors)[0]-reference(data,kind,capture,delay,lo,anchors=anchors)[0])/2e-6
            assert out['gradients'][bank][index]==pytest.approx(fd+(weak if bank==mean else 0.),rel=1e-3,abs=1e-5),(bank,index)
    for index in range(len(z)):
        if bundle.plan['dynamic']['detached'][index] or index in bundle.plan['dynamic']['integer_states']:assert out['initial_state_gradients'][0][index]==0;continue
        hi=np.array(bundle.initial_state);lo=hi.copy();hi[index]+=1e-6;lo[index]-=1e-6
        fd=(reference(*args,initial=hi,anchors=anchors)[0]-reference(*args,initial=lo,anchors=anchors)[0])/2e-6
        assert out['initial_state_gradients'][0][index]==pytest.approx(fd,rel=1e-3,abs=1e-5),index
