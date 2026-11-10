"""One scalar sample per actual NumPy callback, shared by every selected row."""
import copy
import brian2 as b
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer,lower_brian_dynamic_training
from test_training_callback_affine_noise import model as affine_model
from test_training_callback_noise_functions import scalar_normal_selected,scalar_uniform_selected,scalar_randn_selected,scalar_rand_selected,scalar_parameter_normal_selected,scalar_parameter_uniform_selected,scalar_alias_normal_selected
from test_training_integer_ir import engine
from test_training_poisson_zero_vjp import mpi
from test_training_event_noise import draw
from test_native_training import RUNNER


def model(backend,ranks,mode,kind):
    normal=kind in ('normal','randn','parameter_normal','alias_normal')
    net,g,syn,dt,_,x=affine_model(backend,ranks,mode,'normal' if normal else 'uniform')
    parameterised=kind.startswith('parameter_')
    if parameterised:
        old=syn;net.remove(old)
        syn=b.Synapses(old.source,g,'h:1\ngain:1 (constant)\nmean:1 (constant, shared)\nscale:1 (constant, shared)',on_pre='v_post+=gain*f(h,mean,scale)+.01*h',dt=dt)
        syn.connect(i=[0,1],j=[0,0] if mode=='vectorised' else [0,1]);syn.h=[1.1,1.9];syn.gain=[.11,.19];syn.mean=.04;syn.scale=.13;syn.delay=dt
        if mode=='array':syn.variables['_postsynaptic_idx'].unique=True
        net.add(syn)
    factory=dict(normal=scalar_normal_selected,uniform=scalar_uniform_selected,randn=scalar_randn_selected,rand=scalar_rand_selected,parameter_normal=scalar_parameter_normal_selected,parameter_uniform=scalar_parameter_uniform_selected,alias_normal=scalar_alias_normal_selected)[kind]
    syn.namespace['f']=b.Function(factory(g.variables['u'].get_value()),arg_units=[1]*(3 if parameterised else 2),return_unit=1,stateless=False)
    hidden=next(o for o in net.objects if isinstance(o,b.NeuronGroup) and o is not g)
    bundle=lower_brian_dynamic_training(net,input_group=syn.source,layers=[hidden,g],backend=backend,mpi_ranks=ranks,detach_reset=False,trainable_synapse_parameters={syn.name:['h','mean','scale','gain'] if parameterised else ['h','spread','gain']})
    return net,g,syn,dt,bundle,x


@pytest.mark.parametrize('mode',['array','vectorised'])
@pytest.mark.parametrize('kind',['normal','uniform','randn','rand','parameter_normal','parameter_uniform','alias_normal'])
@pytest.mark.parametrize('ranks',[None,2])
def test_scalar_original_restore(engine,mode,kind,ranks,tmp_path,monkeypatch):
    mpi(ranks);net,g,syn,dt,bundle,x=model(engine,ranks,mode,kind);p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights)
    t=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER);domain=bundle.provenance['scheduled_noise_domains'][syn.pre.name];calls=[]
    for tick in range(5):
        def sampler(*args,**kwargs):
            rows=[edge for edge in range(2) if x[0,tick-1,edge]>0] if tick else [];assert rows;calls.append((tick,tuple(rows)))
            if kind in ('rand','randn'):assert not args and not kwargs;first=0.;width=1.
            else:
                np.testing.assert_allclose(args,(.04,.13 if kind in ('normal','parameter_normal','alias_normal') else .17));assert kwargs==({} if kind in ('normal','parameter_normal','alias_normal') else {'size':None});first=.04;width=.13
            noise=draw('randn' if kind in ('normal','randn','parameter_normal','alias_normal') else 'rand',p['seed'],9,0,domain,rows[0],tick-1,None,0)
            return float(first+width*noise)
        monkeypatch.setattr(np.random,kind.removeprefix('parameter_').removeprefix('alias_'),sampler)
        out=t.step(x[:,tick:tick+1],[0],**({'initial':'carry'} if tick else {'noise_sequence':9}));net.run(dt,namespace={})
        for name in ('v','u'):
            cells=bundle.provenance['neuron_state_layout'][g.name][name]
            np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],g.variables[name].get_value(),rtol=8e-5,atol=8e-6)
        path=tmp_path/str(tick);t.store(path);t=NativeLIFTrainer(p,runner=RUNNER);t.restore(path)
    assert calls==[(1,(0,1)),(3,(0,))]


@pytest.mark.parametrize('mode',['array','vectorised'])
@pytest.mark.parametrize('kind',['normal','rand'])
@pytest.mark.parametrize('empty',[False,True])
@pytest.mark.parametrize('ranks',[None,2])
def test_scalar_masked_original_restore(engine,mode,kind,empty,ranks,tmp_path,monkeypatch):
    from test_training_poisson_masked_whole import model as masked_model
    mpi(ranks);net,g,syn,dt,_,x=masked_model(engine,ranks,mode,empty=empty,lower=False)
    syn.pre.code='v_post+=.1*f(h,rate)+.01*h';factory=scalar_normal_selected if kind=='normal' else scalar_rand_selected
    syn.namespace['f']=b.Function(factory(g.variables['u'].get_value()),arg_units=[1,1],return_unit=1,stateless=False)
    hidden=next(o for o in net.objects if isinstance(o,b.NeuronGroup) and o is not g)
    bundle=lower_brian_dynamic_training(net,input_group=syn.source,layers=[hidden,g],backend=engine,mpi_ranks=ranks,detach_reset=False,trainable_synapse_parameters={syn.name:['h','rate']})
    p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights);t=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER);domain=bundle.provenance['scheduled_noise_domains'][syn.pre.name];calls=[]
    for tick in range(5):
        def sampler(*args,**kwargs):
            raw=[edge for edge in range(2) if x[0,tick-1,edge]>0] if tick else [];selected=[edge for edge in raw if g.not_refractory[int(syn.j[edge])]];calls.append((tick,len(raw),len(selected)))
            assert raw and args==((.04,.13) if kind=='normal' else ()) and not kwargs
            noise=draw('randn' if kind=='normal' else 'rand',p['seed'],9,0,domain,raw[0],tick-1,None,0)
            return float(.04+.13*noise if kind=='normal' else noise)
        monkeypatch.setattr(np.random,kind,sampler)
        out=t.step(x[:,tick:tick+1],[0],**({'initial':'carry'} if tick else {'noise_sequence':9}));net.run(dt,namespace={})
        for name in ('v','u'):
            cells=bundle.provenance['neuron_state_layout'][g.name][name];np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],g.variables[name].get_value(),rtol=8e-5,atol=8e-6)
        path=tmp_path/str(tick);t.store(path);t=NativeLIFTrainer(p,runner=RUNNER);t.restore(path)
    assert [entry[:2] for entry in calls]==[(1,2),(3,1)]
    assert calls[0][2]==(0 if empty else 1)


def reference(data,weights,mode,kind,window=None,initial=None,anchors=None):
    import ast
    _,g,syn,_,bundle,x=data;p=bundle.plan;z=np.array(bundle.initial_state if initial is None else initial,float)
    if initial is None:
        for cell,ref in enumerate(p['dynamic']['initial_parameters']):
            if ref is not None:z[cell]=weights[ref[0]][ref[1]]
    v=bundle.provenance['neuron_state_layout'][g.name]['v'];u=bundle.provenance['neuron_state_layout'][g.name]['u'];posts=np.asarray(syn.j[:],int)
    banks={row['variables'][0]:row['bank'] for row in bundle.provenance['bindings'] if row['object']==syn.name}
    local=bundle.provenance['event_callback_caller_locals'][syn.pre.name];trace=local['trace'];paths=bundle.provenance['event_callback_stage_groups'][syn.pre.name][1:]
    fields={row['key'][1]:row['fields'] for row in local['rows'] if row['key'][0]=='new'};whole=bundle.provenance['event_callback_whole_locals'][syn.pre.name]
    parameterised=kind.startswith('parameter_');offset=int(parameterised);noise_stage=7+offset;value_stage=8+offset
    assert len(trace['stages'])==12+offset and len(paths)==12+offset+(mode=='array')
    h_token=trace['stages'][0]['inputs']['h'];gain_token=trace['stages'][0]['inputs']['gain'];v_token=trace['stages'][0]['inputs']['v_post'];noise_token=next(iter(trace['stages'][noise_stage]['outputs'].values()))
    value_token=next(iter(trace['stages'][value_stage]['whole_local_outputs'].values()));value_cell=whole[str(value_token)]['cells'][0]
    helper=ast.parse(trace['stages'][value_stage]['code']).body[-1].value
    def token_of(arg):
        name=arg.args[0].id if isinstance(arg,ast.Call) else arg.id
        return trace['stages'][value_stage]['whole_local_inputs'][name]['token'] if name in trace['stages'][value_stage]['whole_local_inputs'] else trace['stages'][value_stage]['inputs'][name]
    first_token,second_token=map(token_of,helper.args[:2])
    def read(token,edge):return z[whole[str(token)]['cells'][0]] if str(token) in whole else z[fields[edge][token]]
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
                for edge,field in fields.items():
                    z[field[h_token]]=weights[banks['h']][edge];z[field[gain_token]]=weights[banks['gain']][edge];z[field[v_token]]=z[v[posts[edge]]]
                    if not parameterised:z[field[trace['stages'][0]['inputs']['spread']]]=weights[banks['spread']][edge]
            if parameterised and stage in (2,3):
                name='mean' if stage==2 else 'scale';token=trace['stages'][stage]['inputs'][name]
                for edge,field in fields.items():z[field[token]]=weights[banks[name]][0]
            if selected:
                if not parameterised and stage in (5,6):
                    token=first_token if stage==5 else second_token;constant=(0. if kind in ('rand','randn') else .04) if stage==5 else (1. if kind in ('rand','randn') else .13 if kind=='normal' else .17)
                    z[whole[str(token)]['cells'][0]]=constant
                if parameterised and stage==7 and kind=='parameter_uniform':z[whole[str(second_token)]['cells'][0]]=weights[banks['mean']][0]+weights[banks['scale']][0]
                if stage==noise_stage:
                    for edge in selected:z[fields[edge][noise_token]]=draw('randn' if kind in ('normal','randn','parameter_normal') else 'rand',p['seed'],9,0,domain,edge,tick-1,None,0)
                if stage==value_stage:
                    first=read(first_token,selected[0]);second=read(second_token,selected[0]);noise=z[fields[selected[0]][noise_token]]
                    z[value_cell]=first+(second if kind in ('normal','randn','parameter_normal') else second-first)*noise
                if stage==10+offset:z[u]+=.03*z[value_cell]
            for row,gate in zip(rows,gates):
                edge=row['edge'];field=fields[edge]
                if stage==4+offset:z[field[h_token]]*=1.-.3*gate
                elif stage==11+offset:
                    value=z[field[gain_token]]*(z[field[h_token]]+.1*z[value_cell])+.01*z[field[h_token]]
                    if mode=='vectorised':z[v[posts[edge]]]+=gate*value
                    else:z[field[v_token]]+=gate*value
                elif stage==12+offset:z[v[posts[edge]]]+=gate*(z[field[v_token]]-z[v[posts[edge]]])
        for queue in bundle.provenance['delay_queues'].values():
            for row in queue['new']:
                cells=row['states'];z[cells[:-1]]=z[cells[1:]];z[cells[-1]]=x[0,tick,row['edge']]
        z[v]-=.5*event
    logits=np.array(spikes).mean(0)*p['logit_scale'];loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
    return loss,z,dict(before=before,margins=margins,hard=hard_rows,selections=selections)


@pytest.mark.parametrize('mode',['array','vectorised'])
@pytest.mark.parametrize('kind',['normal','uniform','randn','rand','parameter_normal','parameter_uniform'])
@pytest.mark.parametrize('window',[None,2])
@pytest.mark.parametrize('ranks',[None,2])
def test_scalar_all_vjps(engine,mode,kind,window,ranks):
    mpi(ranks);data=model(engine,ranks,mode,kind);_,g,syn,_,bundle,x=data;p=copy.deepcopy(bundle.plan);p['tbptt_window']=window
    out=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER).gradients(x,[0],noise_sequence=9);args=(data,bundle.weights,mode,kind);loss,z,anchors=reference(*args,window=window)
    cells=[*bundle.provenance['neuron_state_layout'][g.name]['v'],*bundle.provenance['neuron_state_layout'][g.name]['u']]
    np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],z[cells],rtol=8e-5,atol=8e-6);assert out['loss']==pytest.approx(loss,abs=8e-6)
    for bindex,row in enumerate(bundle.weights):
        for index in range(len(row)):
            hi=copy.deepcopy(bundle.weights);lo=copy.deepcopy(bundle.weights);hi[bindex][index]+=1e-6;lo[bindex][index]-=1e-6
            fd=(reference(data,hi,mode,kind,window=window,anchors=anchors)[0]-reference(data,lo,mode,kind,window=window,anchors=anchors)[0])/2e-6
            assert out['gradients'][bindex][index]==pytest.approx(fd,rel=1e-3,abs=1e-5),(bindex,index)
    for index in range(len(z)):
        if p['dynamic']['detached'][index] or index in p['dynamic']['integer_states']:assert out['initial_state_gradients'][0][index]==0;continue
        hi=np.array(bundle.initial_state);lo=hi.copy();hi[index]+=1e-6;lo[index]-=1e-6
        fd=(reference(*args,window=window,initial=hi,anchors=anchors)[0]-reference(*args,window=window,initial=lo,anchors=anchors)[0])/2e-6
        assert out['initial_state_gradients'][0][index]==pytest.approx(fd,rel=1e-3,abs=1e-5),index


@pytest.mark.parametrize('mode',['array','vectorised'])
@pytest.mark.parametrize('ranks',[None,2])
def test_scalar_negative_restore_atomic(engine,mode,ranks,tmp_path):
    mpi(ranks);net,g,syn,dt,bundle,x=model(engine,ranks,mode,'parameter_normal');p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights)
    t=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER);t.step(x[:,:3],[0],noise_sequence=9);net.run(3*dt,namespace={});path=tmp_path/'prefix';t.store(path);t=NativeLIFTrainer(p,runner=RUNNER);t.restore(path)
    bank=next(row['bank'] for row in bundle.provenance['bindings'] if row['variables']==['scale']);t.state['weights'][bank][0]=-1.;syn.scale=-1.
    before=copy.deepcopy((t.state,t.neuron_state,t.clock_state,t.poisson_state,t.elapsed_ticks,t.next_noise_sequence))
    with pytest.raises(ValueError):t.step(x[:,3:4],[0],initial='carry')
    assert (t.state,t.neuron_state,t.clock_state,t.poisson_state,t.elapsed_ticks,t.next_noise_sequence)==before
    with pytest.raises(b.BrianObjectException):net.run(dt,namespace={})


@pytest.mark.parametrize('mode',['array','vectorised'])
@pytest.mark.parametrize('kind',['parameter_normal','parameter_uniform'])
@pytest.mark.parametrize('ranks',[None,2])
def test_scalar_zero_width_all_vjps(engine,mode,kind,ranks):
    mpi(ranks);data=model(engine,ranks,mode,kind);_,g,syn,_,bundle,x=data;weights=copy.deepcopy(bundle.weights)
    bank=next(row['bank'] for row in bundle.provenance['bindings'] if row['variables']==['scale']);weights[bank]=[0.]
    out=NativeLIFTrainer(bundle.plan,weights=weights,runner=RUNNER).gradients(x,[0],noise_sequence=9);args=(data,weights,mode,kind);loss,z,anchors=reference(*args)
    for bindex,row in enumerate(weights):
        for index in range(len(row)):
            hi=copy.deepcopy(weights);hi[bindex][index]+=1e-6
            fd=(reference(data,hi,mode,kind,anchors=anchors)[0]-reference(*args,anchors=anchors)[0])/1e-6
            assert out['gradients'][bindex][index]==pytest.approx(fd,rel=1e-3,abs=1e-5),(bindex,index)
    for index in range(len(z)):
        if bundle.plan['dynamic']['detached'][index] or index in bundle.plan['dynamic']['integer_states']:assert out['initial_state_gradients'][0][index]==0;continue
        hi=np.array(bundle.initial_state);hi[index]+=1e-6
        fd=(reference(*args,initial=hi,anchors=anchors)[0]-reference(*args,initial=bundle.initial_state,anchors=anchors)[0])/1e-6
        assert out['initial_state_gradients'][0][index]==pytest.approx(fd,rel=1e-3,abs=1e-5),index


@pytest.mark.parametrize('mode',['array','vectorised'])
@pytest.mark.parametrize('kind',['normal','rand'])
@pytest.mark.parametrize('ranks',[None,2])
def test_scalar_repeated_stream_restore(engine,mode,kind,ranks,tmp_path,monkeypatch):
    mpi(ranks);net,g,syn,dt,_,x=model(engine,ranks,mode,kind);syn.pre.code='v_post+=gain*(f(h,spread)+f(h,spread))+.01*h'
    hidden=next(o for o in net.objects if isinstance(o,b.NeuronGroup) and o is not g);bundle=lower_brian_dynamic_training(net,input_group=syn.source,layers=[hidden,g],backend=engine,mpi_ranks=ranks,detach_reset=False,trainable_synapse_parameters={syn.name:['h','spread','gain']})
    p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights);t=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER);domain=bundle.provenance['scheduled_noise_domains'][syn.pre.name];calls=[]
    for tick in range(5):
        streams=[]
        def sampler(*args,**kwargs):
            raw=[edge for edge in range(2) if x[0,tick-1,edge]>0] if tick else [];stream=len(streams);streams.append(stream);calls.append((tick,stream));assert raw and stream<2
            assert args==((.04,.13) if kind=='normal' else ()) and not kwargs
            noise=draw('randn' if kind=='normal' else 'rand',p['seed'],9,0,domain,raw[0],tick-1,None,stream)
            return float(.04+.13*noise if kind=='normal' else noise)
        monkeypatch.setattr(np.random,kind,sampler)
        out=t.step(x[:,tick:tick+1],[0],**({'initial':'carry'} if tick else {'noise_sequence':9}));net.run(dt,namespace={})
        for name in ('v','u'):
            cells=bundle.provenance['neuron_state_layout'][g.name][name];np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],g.variables[name].get_value(),rtol=8e-5,atol=8e-6)
        path=tmp_path/str(tick);t.store(path);t=NativeLIFTrainer(p,runner=RUNNER);t.restore(path)
    assert calls==[(1,0),(1,1),(3,0),(3,1)]
