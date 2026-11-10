"""Callback-internal selected Normal/Uniform arrays keep event identities."""
import copy
import brian2 as b
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer,lower_brian_dynamic_training
from test_training_callback_noise_functions import normal_selected,uniform_selected,mixed_selected,normal_domain_selected,uniform_domain_selected
from test_training_integer_ir import engine
from test_training_poisson_zero_vjp import mpi
from test_training_event_noise import draw
from test_training_batch_poisson_captures import observation_stream
from test_native_training import RUNNER


def model(backend,ranks,mode,kind,domain=False):
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='numpy';dt=.2*b.ms
    source=b.SpikeGeneratorGroup(2,[0,1,0],np.array([0,0,2])*dt,dt=dt)
    hidden=b.NeuronGroup(1,'dv/dt=0/second:1',threshold='v>100',reset='v=0',method='euler',dt=dt)
    g=b.NeuronGroup(2,'dv/dt=u/ms:1\nu:1',threshold='v>.5',reset='v-=.5',method='euler',dt=dt)
    g.v=[.41,.31];g.u=[.017,.031]
    syn=b.Synapses(source,g,'h:1\ngain:1 (constant)',on_pre='v_post+=gain*f(h)+.01*h',dt=dt)
    syn.connect(i=[0,1],j=[0,0] if mode=='vectorised' else [0,1]);syn.h=[1.1,1.9];syn.gain=[.11,.19];syn.delay=dt
    if mode=='array':syn.variables['_postsynaptic_idx'].unique=True
    factory=mixed_selected if kind=='mixed' else normal_selected if kind=='randn' else uniform_selected
    if domain:factory=normal_domain_selected if kind=='randn' else uniform_domain_selected
    syn.namespace['f']=b.Function(factory(g.variables['u'].get_value()),arg_units=[1],arg_names=['x'],return_unit=1,stateless=False)
    net=b.Network(source,hidden,g,syn);x=np.zeros((1,5,2));x[0,0,:]=1;x[0,2,0]=1
    bundle=lower_brian_dynamic_training(net,input_group=source,layers=[hidden,g],backend=backend,mpi_ranks=ranks,detach_reset=False,trainable_synapse_parameters={syn.name:['h','gain']})
    return net,g,syn,dt,bundle,x


@pytest.mark.parametrize('mode',['array','vectorised'])
@pytest.mark.parametrize('kind',['randn','rand'])
@pytest.mark.parametrize('ranks',[None,2])
def test_callback_body_noise_original_restore(engine,mode,kind,ranks,tmp_path,monkeypatch):
    mpi(ranks);net,g,syn,dt,bundle,x=model(engine,ranks,mode,kind);p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights)
    t=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER);domain=bundle.provenance['scheduled_noise_domains'][syn.pre.name];calls=[]
    for tick in range(5):
        def sampler(*shape):
            rows=[edge for edge in range(2) if x[0,tick-1,edge]>0] if tick else [];assert shape==(len(rows),);calls.append((tick,len(rows)))
            return np.array([draw(kind,p['seed'],9,0,domain,edge,tick-1,None,0) for edge in rows])
        monkeypatch.setattr(np.random,kind,sampler)
        out=t.step(x[:,tick:tick+1],[0],**({'initial':'carry'} if tick else {'noise_sequence':9}));net.run(dt,namespace={})
        for name in ('v','u'):
            cells=bundle.provenance['neuron_state_layout'][g.name][name]
            np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],g.variables[name].get_value(),rtol=8e-5,atol=8e-6)
        path=tmp_path/str(tick);t.store(path);t=NativeLIFTrainer(p,runner=RUNNER);t.restore(path)
    assert calls==[(1,2),(3,1)] and t.next_noise_sequence==10


def reference(data,weights,mode,kind,window=None,initial=None,anchors=None):
    _,g,syn,_,bundle,x=data;p=bundle.plan;z=np.array(bundle.initial_state if initial is None else initial,float)
    if initial is None:
        for cell,ref in enumerate(p['dynamic']['initial_parameters']):
            if ref is not None:z[cell]=weights[ref[0]][ref[1]]
    v=bundle.provenance['neuron_state_layout'][g.name]['v'];u=bundle.provenance['neuron_state_layout'][g.name]['u']
    banks={row['variables'][0]:row['bank'] for row in bundle.provenance['bindings'] if row['object']==syn.name}
    local=bundle.provenance['event_callback_caller_locals'][syn.pre.name];trace=local['trace'];paths=bundle.provenance['event_callback_stage_groups'][syn.pre.name][1:]
    fields={row['key'][1]:row['fields'] for row in local['rows'] if row['key'][0]=='new'}
    assert len(trace['stages'])==8 and len(paths)==(8 if mode=='vectorised' else 9)
    inputs=trace['stages'][0]['inputs'];h_token=inputs['h'];v_token=inputs['v_post'];gain_token=inputs['gain'];noise_token=next(iter(trace['stages'][4]['outputs'].values()))
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
                for edge,field in fields.items():z[field[h_token]]=weights[banks['h']][edge];z[field[gain_token]]=weights[banks['gain']][edge];z[field[v_token]]=z[v[posts[edge]]]
            if stage==4:
                for edge in selected:z[fields[edge][noise_token]]=draw(kind,p['seed'],9,0,domain,edge,tick-1,None,0)
            if stage==6 and selected:
                incoming=np.array([.03*z[fields[edge][noise_token]] for edge in selected]);z[u]+=incoming[0] if len(incoming)==1 else incoming
            for row,gate in zip(rows,gates):
                edge=row['edge'];field=fields[edge]
                if stage==3:z[field[h_token]]*=1.-.3*gate
                elif stage==7:
                    value=z[field[gain_token]]*(z[field[h_token]]+.1*z[field[noise_token]])+.01*z[field[h_token]]
                    if mode=='vectorised':z[v[posts[edge]]]+=gate*value
                    else:z[field[v_token]]+=gate*value
                elif stage==8:z[v[posts[edge]]]+=gate*(z[field[v_token]]-z[v[posts[edge]]])
        for queue in bundle.provenance['delay_queues'].values():
            for row in queue['new']:
                cells=row['states'];z[cells[:-1]]=z[cells[1:]];z[cells[-1]]=x[0,tick,row['edge']]
        z[v]-=.5*event
    logits=np.array(spikes).mean(0)*p['logit_scale'];loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
    return loss,z,dict(before=before,margins=margins,hard=hard_rows,selections=selections)


@pytest.mark.parametrize('mode',['array','vectorised'])
@pytest.mark.parametrize('kind',['randn','rand'])
@pytest.mark.parametrize('window',[None,2])
@pytest.mark.parametrize('ranks',[None,2])
def test_callback_body_noise_all_vjps(engine,mode,kind,window,ranks):
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
@pytest.mark.parametrize('ranks',[None,2])
def test_callback_body_three_streams_restore(engine,mode,ranks,tmp_path,monkeypatch):
    mpi(ranks);net,g,syn,dt,bundle,x=model(engine,ranks,mode,'mixed');p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights)
    t=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER);domain=bundle.provenance['scheduled_noise_domains'][syn.pre.name];calls=[]
    for tick in range(5):
        site=[0]
        def rows():return [edge for edge in range(2) if x[0,tick-1,edge]>0] if tick else []
        def normal(*shape):
            assert shape==(len(rows()),) and site[0]==0;site[0]+=1;calls.append((tick,0))
            return np.array([draw('randn',p['seed'],9,0,domain,edge,tick-1,None,0) for edge in rows()])
        def uniform(*shape):
            assert shape==(len(rows()),) and site[0]==1;site[0]+=1;calls.append((tick,1))
            return np.array([draw('rand',p['seed'],9,0,domain,edge,tick-1,None,1) for edge in rows()])
        def poisson(rate,*args,**kwargs):
            assert not args and not kwargs and site[0]==2;site[0]+=1;calls.append((tick,2))
            expected=.7*np.asarray(syn.h[:])[rows()]+.02*np.array([draw('randn',p['seed'],9,0,domain,edge,tick-1,None,0) for edge in rows()]);np.testing.assert_allclose(rate,expected)
            return np.array([observation_stream(p['seed'],9,domain,edge,tick-1,float(value),2) for edge,value in zip(rows(),expected)],dtype=np.int64)
        monkeypatch.setattr(np.random,'randn',normal);monkeypatch.setattr(np.random,'rand',uniform);monkeypatch.setattr(np.random,'poisson',poisson)
        out=t.step(x[:,tick:tick+1],[0],**({'initial':'carry'} if tick else {'noise_sequence':9}));net.run(dt,namespace={})
        for name in ('v','u'):
            cells=bundle.provenance['neuron_state_layout'][g.name][name]
            np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],g.variables[name].get_value(),rtol=8e-5,atol=8e-6)
        path=tmp_path/str(tick);t.store(path);t=NativeLIFTrainer(p,runner=RUNNER);t.restore(path)
    assert calls==[(1,0),(1,1),(1,2),(3,0),(3,1),(3,2)]
    assert len(t.poisson_state['entries'])==3 and {row['identity']['site']['stream'] for row in t.poisson_state['entries']}=={2}


@pytest.mark.parametrize('mode',['array','vectorised'])
@pytest.mark.parametrize('kind',['randn','rand'])
@pytest.mark.parametrize('ranks',[None,2])
def test_callback_body_noise_domain_restore_atomic(engine,mode,kind,ranks,tmp_path):
    mpi(ranks);net,g,syn,dt,bundle,x=model(engine,ranks,mode,kind,domain=True);p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights)
    t=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER);t.step(x[:,:3],[0],noise_sequence=9);net.run(3*dt,namespace={})
    path=tmp_path/'prefix';t.store(path);t=NativeLIFTrainer(p,runner=RUNNER);t.restore(path)
    bank=next(row['bank'] for row in bundle.provenance['bindings'] if row['variables']==['h']);t.state['weights'][bank][0]=-1.;syn.h[0]=-1.
    before=copy.deepcopy((t.state,t.neuron_state,t.clock_state,t.poisson_state,t.elapsed_ticks,t.next_noise_sequence))
    with pytest.raises(ValueError):t.step(x[:,3:4],[0],initial='carry')
    assert (t.state,t.neuron_state,t.clock_state,t.poisson_state,t.elapsed_ticks,t.next_noise_sequence)==before
    with np.errstate(invalid='raise'),pytest.raises(b.BrianObjectException):net.run(dt,namespace={})
