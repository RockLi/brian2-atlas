"""Affine callback distributions keep argument order and parameter gradients."""
import copy
import brian2 as b
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer,lower_brian_dynamic_training
from test_training_callback_noise_functions import affine_normal_selected,affine_uniform_selected,keyword_normal_selected,keyword_uniform_selected,default_normal_selected,default_uniform_selected
from test_training_integer_ir import engine
from test_training_poisson_zero_vjp import mpi
from test_training_event_noise import draw
from test_native_training import RUNNER


def model(backend,ranks,mode,kind,signature='positional'):
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='numpy';dt=.2*b.ms
    source=b.SpikeGeneratorGroup(2,[0,1,0],np.array([0,0,2])*dt,dt=dt)
    hidden=b.NeuronGroup(1,'dv/dt=0/second:1',threshold='v>100',reset='v=0',method='euler',dt=dt)
    g=b.NeuronGroup(2,'dv/dt=u/ms:1\nu:1',threshold='v>.5',reset='v-=.5',method='euler',dt=dt)
    g.v=[.41,.31];g.u=[.017,.031]
    syn=b.Synapses(source,g,'h:1\nspread:1 (constant)\ngain:1 (constant)',on_pre='v_post+=gain*f(h,spread)+.01*h',dt=dt)
    syn.connect(i=[0,1],j=[0,0] if mode=='vectorised' else [0,1]);syn.h=[1.1,1.9];syn.spread=[.09,.13];syn.gain=[.11,.19];syn.delay=dt
    if mode=='array':syn.variables['_postsynaptic_idx'].unique=True
    factory=affine_normal_selected if kind=='normal' else affine_uniform_selected
    if signature=='keyword':factory=keyword_normal_selected if kind=='normal' else keyword_uniform_selected
    if signature=='default':factory=default_normal_selected if kind=='normal' else default_uniform_selected
    syn.namespace['f']=b.Function(factory(g.variables['u'].get_value()),arg_units=[1,1],arg_names=['x','spread'],return_unit=1,stateless=False)
    net=b.Network(source,hidden,g,syn);x=np.zeros((1,5,2));x[0,0,:]=1;x[0,2,0]=1
    bundle=lower_brian_dynamic_training(net,input_group=source,layers=[hidden,g],backend=backend,mpi_ranks=ranks,detach_reset=False,trainable_synapse_parameters={syn.name:['h','spread','gain']})
    return net,g,syn,dt,bundle,x


@pytest.mark.parametrize('mode',['array','vectorised'])
@pytest.mark.parametrize('kind',['normal','uniform'])
@pytest.mark.parametrize('ranks',[None,2])
def test_callback_affine_original_restore(engine,mode,kind,ranks,tmp_path,monkeypatch):
    mpi(ranks);net,g,syn,dt,bundle,x=model(engine,ranks,mode,kind);p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights)
    t=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER);domain=bundle.provenance['scheduled_noise_domains'][syn.pre.name];calls=[]
    for tick in range(5):
        def sampler(first,second,*args,**kwargs):
            rows=[edge for edge in range(2) if x[0,tick-1,edge]>0] if tick else [];base=.7*np.asarray(syn.h[:])[rows];spread=np.asarray(syn.spread[:])[rows]
            np.testing.assert_allclose(first,base);np.testing.assert_allclose(second,spread if kind=='normal' else base+spread)
            assert not args and kwargs=={'size':len(rows)};calls.append((tick,len(rows)))
            values=np.array([draw('randn' if kind=='normal' else 'rand',p['seed'],9,0,domain,edge,tick-1,None,0) for edge in rows])
            return base+spread*values
        monkeypatch.setattr(np.random,kind,sampler)
        out=t.step(x[:,tick:tick+1],[0],**({'initial':'carry'} if tick else {'noise_sequence':9}));net.run(dt,namespace={})
        for name in ('v','u'):
            cells=bundle.provenance['neuron_state_layout'][g.name][name]
            np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],g.variables[name].get_value(),rtol=8e-5,atol=8e-6)
        path=tmp_path/str(tick);t.store(path);t=NativeLIFTrainer(p,runner=RUNNER);t.restore(path)
    assert calls==[(1,2),(3,1)] and t.next_noise_sequence==10


@pytest.mark.parametrize('mode',['array','vectorised'])
@pytest.mark.parametrize('kind',['normal','uniform'])
@pytest.mark.parametrize('ranks',[None,2])
def test_callback_affine_invalid_restore_atomic(engine,mode,kind,ranks,tmp_path):
    mpi(ranks);net,g,syn,dt,bundle,x=model(engine,ranks,mode,kind);p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights)
    t=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER);t.step(x[:,:3],[0],noise_sequence=9);net.run(3*dt,namespace={})
    path=tmp_path/'prefix';t.store(path);t=NativeLIFTrainer(p,runner=RUNNER);t.restore(path)
    bank=next(row['bank'] for row in bundle.provenance['bindings'] if row['variables']==['spread']);t.state['weights'][bank][0]=-1. if kind=='normal' else 1.7e308;syn.spread[0]=t.state['weights'][bank][0]
    if kind=='uniform':
        hbank=next(row['bank'] for row in bundle.provenance['bindings'] if row['variables']==['h']);t.state['weights'][hbank][0]=1e308;syn.h[0]=1e308
    before=copy.deepcopy((t.state,t.neuron_state,t.clock_state,t.poisson_state,t.elapsed_ticks,t.next_noise_sequence))
    with pytest.raises(ValueError):t.step(x[:,3:4],[0],initial='carry')
    assert (t.state,t.neuron_state,t.clock_state,t.poisson_state,t.elapsed_ticks,t.next_noise_sequence)==before
    with pytest.raises(b.BrianObjectException):net.run(dt,namespace={})


def reference(data,weights,mode,kind,window=None,initial=None,anchors=None):
    import ast
    _,g,syn,_,bundle,x=data;p=bundle.plan;z=np.array(bundle.initial_state if initial is None else initial,float)
    if initial is None:
        for cell,ref in enumerate(p['dynamic']['initial_parameters']):
            if ref is not None:z[cell]=weights[ref[0]][ref[1]]
    v=bundle.provenance['neuron_state_layout'][g.name]['v'];u=bundle.provenance['neuron_state_layout'][g.name]['u']
    banks={row['variables'][0]:row['bank'] for row in bundle.provenance['bindings'] if row['object']==syn.name}
    local=bundle.provenance['event_callback_caller_locals'][syn.pre.name];trace=local['trace'];paths=bundle.provenance['event_callback_stage_groups'][syn.pre.name][1:]
    fields={row['key'][1]:row['fields'] for row in local['rows'] if row['key'][0]=='new'}
    assert len(trace['stages'])==13 and len(paths)==(13 if mode=='vectorised' else 14)
    inputs=trace['stages'][0]['inputs'];h_token=inputs['h'];v_token=inputs['v_post'];gain_token=inputs['gain'];spread_token=inputs['spread'];noise_token=next(iter(trace['stages'][8]['outputs'].values()));value_token=next(iter(trace['stages'][9]['outputs'].values()))
    args=ast.parse(trace['stages'][9]['code']).body[0].value.args;first_token=trace['stages'][9]['inputs'][args[0].id];second_token=trace['stages'][9]['inputs'][args[1].id]
    posts=np.asarray(syn.j[:],int);domain=bundle.provenance['scheduled_noise_domains'][syn.pre.name];before=[];margins=[];hard_rows=[];spikes=[];selections=[]
    for tick in range(5):
        if anchors is not None and window and tick and tick%window==0:z=anchors['before'][tick].copy()
        before.append(z.copy());z[v]+=.2*z[u];margin=z[v]-.5;hard=(margin>0).astype(float);event=hard.copy()
        margins.append(margin.copy());hard_rows.append(hard.copy())
        if anchors is not None:event=anchors['hard'][tick]+p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(anchors['margins'][tick]))**2*(margin-anchors['margins'][tick])
        spikes.append(event.copy());selections.append([])
        for stage,path in enumerate(paths):
            for row in bundle.provenance['event_callback_snapshots'].get(path,[]):z[row['cache']]=z[row['source']]
            rows=bundle.provenance['delay_queues'][path]['new'];gates=[z[row['states'][0]] for row in rows];selected=[row['edge'] for row,gate in zip(rows,gates) if gate>.5]
            if anchors is not None:selected=anchors['selections'][tick][stage]
            selections[-1].append(selected)
            if stage==0:
                for edge,field in fields.items():
                    z[field[h_token]]=weights[banks['h']][edge];z[field[gain_token]]=weights[banks['gain']][edge];z[field[spread_token]]=weights[banks['spread']][edge];z[field[v_token]]=z[v[posts[edge]]]
            if stage==8:
                for edge in selected:z[fields[edge][noise_token]]=draw('randn' if kind=='normal' else 'rand',p['seed'],9,0,domain,edge,tick-1,None,0)
            if stage==11 and selected:
                incoming=np.array([.03*z[fields[edge][value_token]] for edge in selected]);z[u]+=incoming[0] if len(incoming)==1 else incoming
            for row,gate in zip(rows,gates):
                edge=row['edge'];field=fields[edge]
                if stage==4:z[field[h_token]]*=1.-.3*gate
                elif stage==6 and kind=='uniform':z[field[second_token]]+=gate*(z[field[h_token]]+z[field[spread_token]]-z[field[second_token]])
                elif stage==9:
                    first=z[field[first_token]];second=z[field[second_token]];value=first+(second if kind=='normal' else second-first)*z[field[noise_token]]
                    z[field[value_token]]+=gate*(value-z[field[value_token]])
                elif stage==12:
                    value=z[field[gain_token]]*(z[field[h_token]]+.1*z[field[value_token]])+.01*z[field[h_token]]
                    if mode=='vectorised':z[v[posts[edge]]]+=gate*value
                    else:z[field[v_token]]+=gate*value
                elif stage==13:z[v[posts[edge]]]+=gate*(z[field[v_token]]-z[v[posts[edge]]])
        for queue in bundle.provenance['delay_queues'].values():
            for row in queue['new']:
                cells=row['states'];z[cells[:-1]]=z[cells[1:]];z[cells[-1]]=x[0,tick,row['edge']]
        z[v]-=.5*event
    logits=np.array(spikes).mean(0)*p['logit_scale'];loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
    return loss,z,dict(before=before,margins=margins,hard=hard_rows,selections=selections)


@pytest.mark.parametrize('mode',['array','vectorised'])
@pytest.mark.parametrize('kind',['normal','uniform'])
@pytest.mark.parametrize('window',[None,2])
@pytest.mark.parametrize('ranks',[None,2])
def test_callback_affine_all_vjps(engine,mode,kind,window,ranks):
    mpi(ranks);data=model(engine,ranks,mode,kind);_,g,syn,_,bundle,x=data;p=copy.deepcopy(bundle.plan);p['tbptt_window']=window
    out=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER).gradients(x,[0],noise_sequence=9);args=(data,bundle.weights,mode,kind)
    loss,z,anchors=reference(*args,window=window);cells=[*bundle.provenance['neuron_state_layout'][g.name]['v'],*bundle.provenance['neuron_state_layout'][g.name]['u']]
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
@pytest.mark.parametrize('kind',['normal','uniform'])
@pytest.mark.parametrize('ranks',[None,2])
def test_callback_affine_zero_width_all_vjps(engine,mode,kind,ranks):
    mpi(ranks);data=model(engine,ranks,mode,kind);_,g,syn,_,bundle,x=data
    bank=next(row['bank'] for row in bundle.provenance['bindings'] if row['variables']==['spread']);weights=copy.deepcopy(bundle.weights);weights[bank]=[0.,0.]
    out=NativeLIFTrainer(bundle.plan,weights=weights,runner=RUNNER).gradients(x,[0],noise_sequence=9)
    args=(data,weights,mode,kind);loss,z,anchors=reference(*args)
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
@pytest.mark.parametrize('ranks',[None,2])
def test_callback_uniform_reverse_restore(engine,mode,ranks,tmp_path,monkeypatch):
    mpi(ranks);net,g,syn,dt,bundle,x=model(engine,ranks,mode,'uniform');p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights)
    bank=next(row['bank'] for row in bundle.provenance['bindings'] if row['variables']==['spread']);weights=copy.deepcopy(bundle.weights);weights[bank]=[-.09,-.13];syn.spread=[-.09,-.13]
    t=NativeLIFTrainer(p,weights=weights,runner=RUNNER);domain=bundle.provenance['scheduled_noise_domains'][syn.pre.name]
    for tick in range(5):
        def sampler(low,high,*args,**kwargs):
            rows=[edge for edge in range(2) if x[0,tick-1,edge]>0] if tick else [];base=.7*np.asarray(syn.h[:])[rows];width=np.asarray(syn.spread[:])[rows]
            np.testing.assert_allclose(low,base);np.testing.assert_allclose(high,base+width);assert not args and kwargs=={'size':len(rows)}
            return base+width*np.array([draw('rand',p['seed'],9,0,domain,edge,tick-1,None,0) for edge in rows])
        monkeypatch.setattr(np.random,'uniform',sampler)
        out=t.step(x[:,tick:tick+1],[0],**({'initial':'carry'} if tick else {'noise_sequence':9}));net.run(dt,namespace={})
        for name in ('v','u'):
            cells=bundle.provenance['neuron_state_layout'][g.name][name]
            np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],g.variables[name].get_value(),rtol=8e-5,atol=8e-6)
        path=tmp_path/str(tick);t.store(path);t=NativeLIFTrainer(p,runner=RUNNER);t.restore(path)


@pytest.mark.parametrize('mode',['array','vectorised'])
@pytest.mark.parametrize('kind',['normal','uniform'])
@pytest.mark.parametrize('signature',['keyword','default'])
@pytest.mark.parametrize('ranks',[None,2])
def test_callback_affine_signature_restore(engine,mode,kind,signature,ranks,tmp_path,monkeypatch):
    mpi(ranks);net,g,syn,dt,bundle,x=model(engine,ranks,mode,kind,signature);p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights)
    t=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER);domain=bundle.provenance['scheduled_noise_domains'][syn.pre.name];calls=[]
    for tick in range(5):
        def sampler(*args,**kwargs):
            rows=[edge for edge in range(2) if x[0,tick-1,edge]>0] if tick else [];assert not args;calls.append((tick,len(rows)))
            if signature=='default':assert kwargs=={'size':len(rows)};base=np.zeros(len(rows));spread=np.ones(len(rows))
            else:
                base=.7*np.asarray(syn.h[:])[rows];spread=np.asarray(syn.spread[:])[rows];first='loc' if kind=='normal' else 'low';second='scale' if kind=='normal' else 'high'
                np.testing.assert_allclose(kwargs[first],base);np.testing.assert_allclose(kwargs[second],spread if kind=='normal' else base+spread);assert kwargs['size']==len(rows)
                assert list(kwargs)==(['scale','size','loc'] if kind=='normal' else ['high','size','low'])
            return base+spread*np.array([draw('randn' if kind=='normal' else 'rand',p['seed'],9,0,domain,edge,tick-1,None,0) for edge in rows])
        monkeypatch.setattr(np.random,kind,sampler)
        out=t.step(x[:,tick:tick+1],[0],**({'initial':'carry'} if tick else {'noise_sequence':9}));net.run(dt,namespace={})
        for name in ('v','u'):
            cells=bundle.provenance['neuron_state_layout'][g.name][name]
            np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],g.variables[name].get_value(),rtol=8e-5,atol=8e-6)
        path=tmp_path/str(tick);t.store(path);t=NativeLIFTrainer(p,runner=RUNNER);t.restore(path)
    assert calls==[(1,2),(3,1)]
