"""No-capture callbacks preserve one scalar draw and actual argument effects."""
import copy
import brian2 as b
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer,lower_brian_dynamic_training
from test_training_callback_scalar_noise import model as scalar_model
import test_training_callback_uncaptured_functions as factories
from test_training_integer_ir import engine
from test_training_poisson_zero_vjp import mpi
from test_training_event_noise import draw
from test_native_training import RUNNER


def model(backend,ranks,mode,kind,variant):
    net,g,syn,dt,_,x=scalar_model(backend,ranks,mode,'parameter_'+kind)
    syn.namespace['f']=b.Function(getattr(factories,variant+'_'+kind),arg_units=[1,1,1],return_unit=1,stateless=False)
    hidden=next(o for o in net.objects if isinstance(o,b.NeuronGroup) and o is not g)
    bundle=lower_brian_dynamic_training(net,input_group=syn.source,layers=[hidden,g],backend=backend,mpi_ranks=ranks,detach_reset=False,trainable_synapse_parameters={syn.name:['h','mean','scale','gain']})
    return net,g,syn,dt,bundle,x


@pytest.mark.parametrize('mode',['array','vectorised'])
@pytest.mark.parametrize('kind',['normal','uniform'])
@pytest.mark.parametrize('variant',['stored','direct','pure','scalar'])
@pytest.mark.parametrize('ranks',[None,2])
def test_uncaptured_original_restore(engine,mode,kind,variant,ranks,tmp_path,monkeypatch):
    mpi(ranks);net,g,syn,dt,bundle,x=model(engine,ranks,mode,kind,variant);p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights)
    t=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER);domain=bundle.provenance['scheduled_noise_domains'][syn.pre.name];calls=[]
    for tick in range(5):
        def sampler(*args,**kwargs):
            rows=[edge for edge in range(2) if x[0,tick-1,edge]>0] if tick else [];assert rows;calls.append((tick,len(rows)))
            np.testing.assert_allclose(args,(.04,.13 if kind=='normal' else .17));assert kwargs==({} if kind=='normal' else {'size':None})
            return float(.04+.13*draw('randn' if kind=='normal' else 'rand',p['seed'],9,0,domain,rows[0],tick-1,None,0))
        monkeypatch.setattr(np.random,kind,sampler)
        out=t.step(x[:,tick:tick+1],[0],**({'initial':'carry'} if tick else {'noise_sequence':9}));net.run(dt,namespace={})
        for name in ('v','u'):
            cells=bundle.provenance['neuron_state_layout'][g.name][name];np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],g.variables[name].get_value(),rtol=8e-5,atol=8e-6)
        np.testing.assert_allclose(g.u[:],[.017,.031])
        path=tmp_path/str(tick);t.store(path);t=NativeLIFTrainer(p,runner=RUNNER);t.restore(path)
    assert calls==[(1,2),(3,1)]


def reference(data,weights,mode,kind,variant,window=None,initial=None,anchors=None):
    _,g,syn,_,bundle,x=data;p=bundle.plan;z=np.array(bundle.initial_state if initial is None else initial,float)
    if initial is None:
        for cell,ref in enumerate(p['dynamic']['initial_parameters']):
            if ref is not None:z[cell]=weights[ref[0]][ref[1]]
    v=bundle.provenance['neuron_state_layout'][g.name]['v'];u=bundle.provenance['neuron_state_layout'][g.name]['u'];posts=np.asarray(syn.j[:],int)
    banks={row['variables'][0]:row['bank'] for row in bundle.provenance['bindings'] if row['object']==syn.name}
    local=bundle.provenance['event_callback_caller_locals'][syn.pre.name];trace=local['trace'];paths=bundle.provenance['event_callback_stage_groups'][syn.pre.name][1:]
    fields={row['key'][1]:row['fields'] for row in local['rows'] if row['key'][0]=='new'};whole=bundle.provenance['event_callback_whole_locals'][syn.pre.name]
    mutate=variant in ('stored','direct');noise_stage=6+int(mutate);value_stage=noise_stage+1;final_stage=noise_stage+2
    assert len(trace['stages'])==9+int(mutate) and len(paths)==9+int(mutate)+(mode=='array')
    first=trace['stages'][0]['inputs'];h_token=first['h'];gain_token=first['gain'];v_token=first['v_post'];noise_token=next(iter(trace['stages'][noise_stage]['outputs'].values()))
    value_token=next(iter(trace['stages'][value_stage]['whole_local_outputs'].values()));value_cell=whole[str(value_token)]['cells'][0]
    domain=bundle.provenance['scheduled_noise_domains'][syn.pre.name];before=[];margins=[];hard_rows=[];spikes=[];selections=[]
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
            if selected:
                if kind=='uniform' and stage==noise_stage-1:
                    token=next(iter(trace['stages'][stage]['whole_local_outputs'].values()));z[whole[str(token)]['cells'][0]]=weights[banks['mean']][0]+weights[banks['scale']][0]
                if stage==noise_stage:
                    for edge in selected:z[fields[edge][noise_token]]=draw('randn' if kind=='normal' else 'rand',p['seed'],9,0,domain,edge,tick-1,None,0)
                if stage==value_stage:z[value_cell]=weights[banks['mean']][0]+weights[banks['scale']][0]*z[fields[selected[0]][noise_token]]
            for row,gate in zip(rows,gates):
                edge=row['edge'];field=fields[edge]
                if mutate and stage==4:z[field[h_token]]*=1.-.3*gate
                elif stage==final_stage:
                    response=z[value_cell] if variant=='scalar' else z[field[h_token]]+.1*z[value_cell]
                    value=z[field[gain_token]]*response+.01*z[field[h_token]]
                    if mode=='vectorised':z[v[posts[edge]]]+=gate*value
                    else:z[field[v_token]]+=gate*value
                elif stage==final_stage+1:z[v[posts[edge]]]+=gate*(z[field[v_token]]-z[v[posts[edge]]])
        for queue in bundle.provenance['delay_queues'].values():
            for row in queue['new']:
                cells=row['states'];z[cells[:-1]]=z[cells[1:]];z[cells[-1]]=x[0,tick,row['edge']]
        z[v]-=.5*event
    logits=np.array(spikes).mean(0)*p['logit_scale'];loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
    return loss,z,dict(before=before,margins=margins,hard=hard_rows,selections=selections)


@pytest.mark.parametrize('mode',['array','vectorised'])
@pytest.mark.parametrize('kind',['normal','uniform'])
@pytest.mark.parametrize('variant',['stored','direct','pure','scalar'])
@pytest.mark.parametrize('window',[None,2])
@pytest.mark.parametrize('ranks',[None,2])
def test_uncaptured_all_vjps(engine,mode,kind,variant,window,ranks):
    mpi(ranks);data=model(engine,ranks,mode,kind,variant);_,g,syn,_,bundle,x=data;p=copy.deepcopy(bundle.plan);p['tbptt_window']=window
    out=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER).gradients(x,[0],noise_sequence=9);args=(data,bundle.weights,mode,kind,variant);loss,z,anchors=reference(*args,window=window)
    cells=[*bundle.provenance['neuron_state_layout'][g.name]['v'],*bundle.provenance['neuron_state_layout'][g.name]['u']]
    np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],z[cells],rtol=8e-5,atol=8e-6);assert out['loss']==pytest.approx(loss,abs=8e-6)
    for bindex,row in enumerate(bundle.weights):
        for index in range(len(row)):
            hi=copy.deepcopy(bundle.weights);lo=copy.deepcopy(bundle.weights);hi[bindex][index]+=1e-6;lo[bindex][index]-=1e-6
            fd=(reference(data,hi,mode,kind,variant,window=window,anchors=anchors)[0]-reference(data,lo,mode,kind,variant,window=window,anchors=anchors)[0])/2e-6
            assert out['gradients'][bindex][index]==pytest.approx(fd,rel=1e-3,abs=1e-5),(bindex,index)
    for index in range(len(z)):
        if p['dynamic']['detached'][index] or index in p['dynamic']['integer_states']:assert out['initial_state_gradients'][0][index]==0;continue
        hi=np.array(bundle.initial_state);lo=hi.copy();hi[index]+=1e-6;lo[index]-=1e-6
        fd=(reference(*args,window=window,initial=hi,anchors=anchors)[0]-reference(*args,window=window,initial=lo,anchors=anchors)[0])/2e-6
        assert out['initial_state_gradients'][0][index]==pytest.approx(fd,rel=1e-3,abs=1e-5),index


@pytest.mark.parametrize('mode',['array','vectorised'])
@pytest.mark.parametrize('variant',['stored','direct','pure','scalar'])
@pytest.mark.parametrize('ranks',[None,2])
def test_uncaptured_negative_restore_atomic(engine,mode,variant,ranks,tmp_path):
    mpi(ranks);net,g,syn,dt,bundle,x=model(engine,ranks,mode,'normal',variant);p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights)
    t=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER);t.step(x[:,:3],[0],noise_sequence=9);net.run(3*dt,namespace={});path=tmp_path/'prefix';t.store(path);t=NativeLIFTrainer(p,runner=RUNNER);t.restore(path)
    bank=next(row['bank'] for row in bundle.provenance['bindings'] if row['variables']==['scale']);t.state['weights'][bank][0]=-1.;syn.scale=-1.
    before=copy.deepcopy((t.state,t.neuron_state,t.clock_state,t.poisson_state,t.elapsed_ticks,t.next_noise_sequence))
    with pytest.raises(ValueError):t.step(x[:,3:4],[0],initial='carry')
    assert (t.state,t.neuron_state,t.clock_state,t.poisson_state,t.elapsed_ticks,t.next_noise_sequence)==before
    with pytest.raises(b.BrianObjectException):net.run(dt,namespace={})


@pytest.mark.parametrize('mode',['array','vectorised'])
@pytest.mark.parametrize('kind',['normal','uniform'])
@pytest.mark.parametrize('variant',['direct','scalar'])
@pytest.mark.parametrize('ranks',[None,2])
def test_uncaptured_zero_width_all_vjps(engine,mode,kind,variant,ranks):
    mpi(ranks);data=model(engine,ranks,mode,kind,variant);_,g,syn,_,bundle,x=data;weights=copy.deepcopy(bundle.weights)
    bank=next(row['bank'] for row in bundle.provenance['bindings'] if row['variables']==['scale']);weights[bank]=[0.]
    out=NativeLIFTrainer(bundle.plan,weights=weights,runner=RUNNER).gradients(x,[0],noise_sequence=9);args=(data,weights,mode,kind,variant);loss,z,anchors=reference(*args)
    for bindex,row in enumerate(weights):
        for index in range(len(row)):
            hi=copy.deepcopy(weights);hi[bindex][index]+=1e-6
            fd=(reference(data,hi,mode,kind,variant,anchors=anchors)[0]-reference(*args,anchors=anchors)[0])/1e-6
            assert out['gradients'][bindex][index]==pytest.approx(fd,rel=1e-3,abs=1e-5),(bindex,index)
    for index in range(len(z)):
        if bundle.plan['dynamic']['detached'][index] or index in bundle.plan['dynamic']['integer_states']:assert out['initial_state_gradients'][0][index]==0;continue
        hi=np.array(bundle.initial_state);hi[index]+=1e-6
        fd=(reference(*args,initial=hi,anchors=anchors)[0]-reference(*args,initial=bundle.initial_state,anchors=anchors)[0])/1e-6
        assert out['initial_state_gradients'][0][index]==pytest.approx(fd,rel=1e-3,abs=1e-5),index


@pytest.mark.parametrize('mode',['array','vectorised'])
@pytest.mark.parametrize('kind',['normal','uniform'])
@pytest.mark.parametrize('variant',['pure','scalar'])
@pytest.mark.parametrize('empty',[False,True])
@pytest.mark.parametrize('ranks',[None,2])
def test_uncaptured_masked_original_restore(engine,mode,kind,variant,empty,ranks,tmp_path,monkeypatch):
    from test_training_poisson_masked_whole import model as masked_model
    mpi(ranks);net,g,syn,dt,_,x=masked_model(engine,ranks,mode,empty=empty,lower=False)
    syn.pre.code='v_post+=.1*f(h,mean,spread)+.01*h';syn.namespace.update(mean=.04,spread=.13,f=b.Function(getattr(factories,variant+'_'+kind),arg_units=[1,1,1],return_unit=1,stateless=False))
    hidden=next(o for o in net.objects if isinstance(o,b.NeuronGroup) and o is not g)
    bundle=lower_brian_dynamic_training(net,input_group=syn.source,layers=[hidden,g],backend=engine,mpi_ranks=ranks,detach_reset=False,trainable_synapse_parameters={syn.name:['h','rate']})
    p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights);t=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER);domain=bundle.provenance['scheduled_noise_domains'][syn.pre.name];calls=[]
    for tick in range(5):
        def sampler(*args,**kwargs):
            raw=[edge for edge in range(2) if x[0,tick-1,edge]>0] if tick else [];selected=[edge for edge in raw if g.not_refractory[int(syn.j[edge])]];calls.append((tick,len(raw),len(selected)));assert raw
            np.testing.assert_allclose(args,(.04,.13 if kind=='normal' else .17));assert kwargs==({} if kind=='normal' else {'size':None})
            return float(.04+.13*draw('randn' if kind=='normal' else 'rand',p['seed'],9,0,domain,raw[0],tick-1,None,0))
        monkeypatch.setattr(np.random,kind,sampler)
        out=t.step(x[:,tick:tick+1],[0],**({'initial':'carry'} if tick else {'noise_sequence':9}));net.run(dt,namespace={})
        for name in ('v','u'):
            cells=bundle.provenance['neuron_state_layout'][g.name][name];np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],g.variables[name].get_value(),rtol=8e-5,atol=8e-6)
        np.testing.assert_allclose(g.u[:],[.017,.031]);path=tmp_path/str(tick);t.store(path);t=NativeLIFTrainer(p,runner=RUNNER);t.restore(path)
    assert [entry[:2] for entry in calls]==[(1,2),(3,1)] and calls[0][2]==(0 if empty else 1)
