"""Real NumPy scalar sampling, shared arrival counts and persistent records."""
import copy
import brian2 as b
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer,lower_brian_dynamic_training
from test_training_callback_scalar_noise import model as scalar_model
import test_training_callback_scalar_poisson_functions as factories
from test_training_integer_ir import engine
from test_training_poisson_zero_vjp import mpi
from test_training_batch_poisson_captures import observation_stream
from test_native_training import RUNNER


def model(backend,ranks,mode,variant='direct',zero=False):
    net,g,syn,dt,_,x=scalar_model(backend,ranks,mode,'parameter_normal')
    syn.mean=0. if zero else 1.3
    syn.namespace['f']=b.Function(getattr(factories,variant+'_poisson'),arg_units=[1,1,1],return_unit=1,stateless=False)
    hidden=next(o for o in net.objects if isinstance(o,b.NeuronGroup) and o is not g)
    bundle=lower_brian_dynamic_training(net,input_group=syn.source,layers=[hidden,g],backend=backend,mpi_ranks=ranks,detach_reset=False,trainable_synapse_parameters={syn.name:['h','mean','scale','gain']})
    return net,g,syn,dt,bundle,x


@pytest.mark.parametrize('mode',['array','vectorised'])
@pytest.mark.parametrize('variant',['direct','stored','response','twice','default','derived'])
@pytest.mark.parametrize('ranks',[None,2])
def test_scalar_poisson_original_restore(engine,mode,variant,ranks,tmp_path,monkeypatch):
    mpi(ranks);net,g,syn,dt,bundle,x=model(engine,ranks,mode,variant)
    p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights)
    t=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER)
    domain=bundle.provenance['scheduled_noise_domains'][syn.pre.name];calls=[];streams=2 if variant=='twice' else 1
    for tick in range(5):
        site=0
        def sampler(*args,**kwargs):
            nonlocal site
            rows=[edge for edge in range(2) if x[0,tick-1,edge]>0] if tick else []
            assert rows
            rate=kwargs['lam'] if variant=='direct' else 1. if variant=='default' else args[0]
            assert np.ndim(rate)==0 and float(rate)==pytest.approx(1. if variant=='default' else 2.6 if variant=='derived' else 1.3)
            assert kwargs==({'lam':rate,'size':None} if variant=='direct' else {})
            calls.append((tick,site,len(rows)))
            result=observation_stream(p['seed'],9,domain,rows[0],tick-1,float(rate),site);site+=1
            return int(result)
        monkeypatch.setattr(np.random,'poisson',sampler)
        out=t.step(x[:,tick:tick+1],[0],**({'initial':'carry'} if tick else {'noise_sequence':9}))
        net.run(dt,namespace={})
        for name in ('v','u'):
            cells=bundle.provenance['neuron_state_layout'][g.name][name]
            np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],g.variables[name].get_value(),rtol=8e-5,atol=8e-6)
        path=tmp_path/str(tick);t.store(path);t=NativeLIFTrainer(p,runner=RUNNER);t.restore(path)
    assert calls==[(tick,site,size) for tick,size in ((1,2),(3,1)) for site in range(streams)]
    assert len(t.poisson_state['entries'])==2*streams


@pytest.mark.parametrize('mode',['array','vectorised'])
@pytest.mark.parametrize('ranks',[None,2])
def test_scalar_poisson_negative_restore_atomic(engine,mode,ranks,tmp_path):
    mpi(ranks);net,g,syn,dt,bundle,x=model(engine,ranks,mode)
    p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights)
    trainer=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER)
    trainer.step(x[:,:3],[0],noise_sequence=9);net.run(3*dt,namespace={})
    path=tmp_path/'prefix';trainer.store(path);trainer=NativeLIFTrainer(p,runner=RUNNER);trainer.restore(path)
    bank=next(row['bank'] for row in bundle.provenance['bindings'] if row['variables']==['mean'])
    trainer.state['weights'][bank][0]=-1.;syn.mean=-1.
    before=copy.deepcopy((trainer.state,trainer.neuron_state,trainer.clock_state,trainer.poisson_state,trainer.elapsed_ticks,trainer.next_noise_sequence))
    with pytest.raises(ValueError):trainer.step(x[:,3:4],[0],initial='carry')
    assert (trainer.state,trainer.neuron_state,trainer.clock_state,trainer.poisson_state,trainer.elapsed_ticks,trainer.next_noise_sequence)==before
    with pytest.raises(b.BrianObjectException):net.run(dt,namespace={})


@pytest.mark.parametrize('mode',['array','vectorised'])
@pytest.mark.parametrize('empty',[False,True])
@pytest.mark.parametrize('ranks',[None,2])
def test_scalar_poisson_masked_original_restore(engine,mode,empty,ranks,tmp_path,monkeypatch):
    from test_training_poisson_masked_whole import model as masked_model
    mpi(ranks);net,g,syn,dt,_,x=masked_model(engine,ranks,mode,empty=empty,lower=False)
    syn.pre.code='v_post+=.1*f(h,mean,0.)+.01*h';syn.namespace['mean']=1.3
    syn.namespace['f']=b.Function(factories.direct_poisson,arg_units=[1,1,1],return_unit=1,stateless=False)
    hidden=next(o for o in net.objects if isinstance(o,b.NeuronGroup) and o is not g)
    bundle=lower_brian_dynamic_training(net,input_group=syn.source,layers=[hidden,g],backend=engine,mpi_ranks=ranks,detach_reset=False,trainable_synapse_parameters={syn.name:['h','rate']})
    p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights);trainer=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER)
    domain=bundle.provenance['scheduled_noise_domains'][syn.pre.name];calls=[]
    for tick in range(5):
        def sample(*args,**kwargs):
            raw=[edge for edge in range(2) if x[0,tick-1,edge]>0] if tick else []
            selected=[edge for edge in raw if g.not_refractory[int(syn.j[edge])]]
            assert raw and not args and kwargs=={'lam':1.3,'size':None};calls.append((tick,len(raw),len(selected)))
            return int(observation_stream(p['seed'],9,domain,raw[0],tick-1,1.3,0))
        monkeypatch.setattr(np.random,'poisson',sample)
        out=trainer.step(x[:,tick:tick+1],[0],**({'initial':'carry'} if tick else {'noise_sequence':9}));net.run(dt,namespace={})
        for name in ('v','u'):
            cells=bundle.provenance['neuron_state_layout'][g.name][name]
            np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],g.variables[name].get_value(),rtol=8e-5,atol=8e-6)
        path=tmp_path/str(tick);trainer.store(path);trainer=NativeLIFTrainer(p,runner=RUNNER);trainer.restore(path)
    assert calls[0]==(1,2,0 if empty else 1) and len(calls)==2 and len(trainer.poisson_state['entries'])==2


def reference(data,weights,mode,window=None,initial=None,anchors=None,override=None,variant='direct'):
    _,g,syn,_,bundle,x=data;p=bundle.plan;z=np.array(bundle.initial_state if initial is None else initial,float)
    if initial is None:
        for cell,ref in enumerate(p['dynamic']['initial_parameters']):
            if ref is not None:z[cell]=weights[ref[0]][ref[1]]
    v=bundle.provenance['neuron_state_layout'][g.name]['v'];u=bundle.provenance['neuron_state_layout'][g.name]['u'];posts=np.asarray(syn.j[:],int)
    banks={row['variables'][0]:row['bank'] for row in bundle.provenance['bindings'] if row['object']==syn.name}
    local=bundle.provenance['event_callback_caller_locals'][syn.pre.name];trace=local['trace'];paths=bundle.provenance['event_callback_stage_groups'][syn.pre.name][1:]
    fields={row['key'][1]:row['fields'] for row in local['rows'] if row['key'][0]=='new'}
    derived=variant=='derived';sample_stage=5+int(derived);value_stage=sample_stage+1;final_stage=value_stage+1
    assert len(trace['stages'])==8+int(derived) and len(paths)==8+int(derived)+(mode=='array')
    first=trace['stages'][0]['inputs'];h_token=first['h'];gain_token=first['gain'];v_token=first['v_post']
    count_token=trace['stages'][sample_stage]['outputs']['_b2_scalar_poisson_cache_0'];value_token=next(iter(trace['stages'][value_stage]['whole_local_outputs'].values()))
    value_cell=bundle.provenance['event_callback_whole_locals'][syn.pre.name][str(value_token)]['cells'][0]
    domain=bundle.provenance['scheduled_noise_domains'][syn.pre.name];before=[];margins=[];hard_rows=[];spikes=[];selections=[];samples=[]
    for tick in range(5):
        if anchors is not None and window and tick and tick%window==0:z=anchors['before'][tick].copy()
        before.append(z.copy());z[v]+=.2*z[u];margin=z[v]-.5;hard=(margin>0).astype(float);event=hard.copy();margins.append(margin.copy());hard_rows.append(hard.copy())
        if anchors is not None:event=anchors['hard'][tick]+p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(anchors['margins'][tick]))**2*(margin-anchors['margins'][tick])
        spikes.append(event.copy());selections.append([])
        for stage,path in enumerate(paths):
            for row in bundle.provenance['event_callback_snapshots'].get(path,[]):z[row['cache']]=z[row['source']]
            rows=bundle.provenance['delay_queues'][path]['new'];gates=[z[row['states'][0]] for row in rows];selected=[row['edge'] for row,gate in zip(rows,gates) if gate>.5]
            if anchors is not None:selected=anchors['selections'][tick][stage]
            selections[-1].append(selected)
            if stage==0:
                for edge,field in fields.items():z[field[h_token]]=weights[banks['h']][edge];z[field[gain_token]]=weights[banks['gain']][edge];z[field[v_token]]=z[v[posts[edge]]]
            if stage in (2,3):
                name='mean' if stage==2 else 'scale';token=trace['stages'][stage]['inputs'][name]
                for edge,field in fields.items():z[field[token]]=weights[banks[name]][0]
            if selected and derived and stage==5:
                token=next(iter(trace['stages'][5]['whole_local_outputs'].values()))
                cell=bundle.provenance['event_callback_whole_locals'][syn.pre.name][str(token)]['cells'][0]
                z[cell]=2.*weights[banks['mean']][0]
            if selected and stage==sample_stage:
                rate=(2. if derived else 1.)*weights[banks['mean']][0];edge=selected[0]
                count=observation_stream(p['seed'],9,domain,edge,tick-1,float(rate),0)
                if anchors is not None:count=next(c for t,e,r,c in anchors['samples'] if (t,e)==(tick,edge))
                if override==(tick,edge):count+=1
                samples.append((tick,edge,float(rate),count))
                for row in selected:z[fields[row][count_token]]=count if row==edge else 0
            if selected and stage==value_stage:z[value_cell]=z[fields[selected[0]][count_token]]
            for row,gate in zip(rows,gates):
                edge=row['edge'];field=fields[edge]
                if stage==final_stage:
                    value=z[field[gain_token]]*z[value_cell]+.01*z[field[h_token]]
                    if mode=='vectorised':z[v[posts[edge]]]+=gate*value
                    else:z[field[v_token]]+=gate*value
                elif stage==final_stage+1:z[v[posts[edge]]]+=gate*(z[field[v_token]]-z[v[posts[edge]]])
        for queue in bundle.provenance['delay_queues'].values():
            for row in queue['new']:
                cells=row['states'];z[cells[:-1]]=z[cells[1:]];z[cells[-1]]=x[0,tick,row['edge']]
        z[v]-=.5*event
    logits=np.array(spikes).mean(0)*p['logit_scale'];loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
    return loss,z,dict(before=before,margins=margins,hard=hard_rows,selections=selections,samples=samples)


@pytest.mark.parametrize('mode',['array','vectorised'])
@pytest.mark.parametrize('window',[None,2])
@pytest.mark.parametrize('zero',[False,True])
@pytest.mark.parametrize('variant',['direct','derived'])
@pytest.mark.parametrize('ranks',[None,2])
def test_scalar_poisson_all_vjps(engine,mode,window,zero,variant,ranks):
    mpi(ranks);data=model(engine,ranks,mode,variant=variant,zero=zero);_,g,syn,_,bundle,x=data;p=copy.deepcopy(bundle.plan);p['tbptt_window']=window
    out=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER).gradients(x,[0],noise_sequence=9)
    def physical(data,weights,mode,**kwargs):return reference(data,weights,mode,variant=variant,**kwargs)
    args=(data,bundle.weights,mode);loss,z,anchors=physical(*args,window=window)
    cells=[*bundle.provenance['neuron_state_layout'][g.name]['v'],*bundle.provenance['neuron_state_layout'][g.name]['u']]
    np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],z[cells],rtol=8e-5,atol=8e-6);assert out['loss']==pytest.approx(loss,abs=8e-6)
    mean=next(row['bank'] for row in bundle.provenance['bindings'] if row['variables']==['mean'])
    weak=(2. if variant=='derived' else 1.)*sum(loss*(count/rate-1.) if rate else physical(*args,override=(tick,edge))[0]-loss for tick,edge,rate,count in anchors['samples'])
    assert abs(weak)>1e-5 and len(anchors['samples'])==2
    for bank,row in enumerate(bundle.weights):
        for index in range(len(row)):
            hi=copy.deepcopy(bundle.weights);lo=copy.deepcopy(bundle.weights);hi[bank][index]+=1e-6;lo[bank][index]-=1e-6
            fd=(physical(data,hi,mode,window=window,anchors=anchors)[0]-physical(data,lo,mode,window=window,anchors=anchors)[0])/2e-6
            assert out['gradients'][bank][index]==pytest.approx(fd+(weak if bank==mean else 0.),rel=1e-3,abs=1e-5),(bank,index)
    for index in range(len(z)):
        if p['dynamic']['detached'][index] or index in p['dynamic']['integer_states']:assert out['initial_state_gradients'][0][index]==0;continue
        hi=np.array(bundle.initial_state);lo=hi.copy();hi[index]+=1e-6;lo[index]-=1e-6
        fd=(physical(*args,window=window,initial=hi,anchors=anchors)[0]-physical(*args,window=window,initial=lo,anchors=anchors)[0])/2e-6
        assert out['initial_state_gradients'][0][index]==pytest.approx(fd,rel=1e-3,abs=1e-5),index
