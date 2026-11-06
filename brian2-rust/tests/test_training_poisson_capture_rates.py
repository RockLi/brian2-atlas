"""Returned whole closure rates are projected at actual Poisson arrival sites."""
import copy
import brian2 as b
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer,lower_brian_dynamic_training
from test_training_capture_callbacks import add_selected
from test_training_returned_captures import return_capture
from test_training_integer_ir import engine
from test_training_poisson_zero_vjp import mpi
from test_training_batch_poisson_captures import observation_stream
from test_native_training import RUNNER


def model(backend,ranks,mode,singleton,derived=False,zero=False,with_h=False):
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='numpy';dt=.2*b.ms
    indices=[0,1,0] if singleton else [0,1,0,1]
    times=np.array([0,0,2] if singleton else [0,0,2,2])*dt
    source=b.SpikeGeneratorGroup(2,indices,times,dt=dt)
    hidden=b.NeuronGroup(1,'dv/dt=0/second:1',threshold='v>100',reset='v=0',method='euler',dt=dt)
    g=b.NeuronGroup(2,'dv/dt=u/ms:1\nu:1',threshold='v>.5',reset='v-=.5',method='euler',dt=dt)
    g.v=[.41,.31];g.u=[.017,.031]
    rate='q(h)+.1' if derived else 'q(h)'
    code='v_post+=.1*f(.2*poisson('+rate+'))'+('+.01*h' if with_h else '')
    syn=b.Synapses(source,g,'h:1\nrate:1 (constant)',on_pre=code,dt=dt)
    syn.connect(i=[0,1],j=[0,0] if mode=='vectorised' else [0,1]);syn.h=[.113,.173];syn.rate=[-.1 if derived else 0.]*2 if zero else [1.1,1.9];syn.delay=dt
    if mode=='array':syn.variables['_postsynaptic_idx'].unique=True
    rates=syn.variables['rate'].get_value()[:1] if singleton else syn.variables['rate'].get_value()[::-1]
    syn.namespace['q']=b.Function(return_capture(rates),arg_units=[1],arg_names=['x'],return_unit=1,stateless=False)
    syn.namespace['f']=b.Function(add_selected(g.variables['u'].get_value()),arg_units=[1],arg_names=['x'],return_unit=1,stateless=False)
    net=b.Network(source,hidden,g,syn);x=np.zeros((1,5,2));x[0,0,:]=1;x[0,2,0]=1
    if not singleton:x[0,2,1]=1
    bundle=lower_brian_dynamic_training(net,input_group=source,layers=[hidden,g],backend=backend,mpi_ranks=ranks,detach_reset=False,trainable_synapse_parameters={syn.name:['h','rate']})
    return net,g,syn,dt,bundle,x


@pytest.mark.parametrize('mode',['array','vectorised'])
@pytest.mark.parametrize('singleton',[False,True])
@pytest.mark.parametrize('derived',[False,True])
@pytest.mark.parametrize('ranks',[None,2])
def test_whole_rate_original_restore(engine,mode,singleton,derived,ranks,tmp_path,monkeypatch):
    mpi(ranks);net,g,syn,dt,bundle,x=model(engine,ranks,mode,singleton,derived)
    p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights)
    t=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER)
    domain=bundle.provenance['scheduled_noise_domains'][syn.pre.name];calls=[]
    for tick in range(5):
        def sampler(rate,*args,**kwargs):
            rows=[edge for edge in range(2) if x[0,tick-1,edge]>0] if tick else []
            expected=np.array(syn.rate[:])[:1] if singleton else np.array(syn.rate[:])[::-1]
            expected=expected+(.1 if derived else 0.)
            np.testing.assert_allclose(rate,expected);assert not args and kwargs=={'size':len(rows)}
            calls.append((tick,len(rows)))
            return np.array([observation_stream(p['seed'],9,domain,edge,tick-1,float(expected[0 if singleton else index]),0) for index,edge in enumerate(rows)],dtype=np.int32)
        monkeypatch.setattr(np.random,'poisson',sampler)
        out=t.step(x[:,tick:tick+1],[0],**({'initial':'carry'} if tick else {'noise_sequence':9}));net.run(dt,namespace={})
        for name in ('v','u'):
            cells=bundle.provenance['neuron_state_layout'][g.name][name]
            np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],g.variables[name].get_value(),rtol=8e-5,atol=8e-6)
        path=tmp_path/str(tick);t.store(path);t=NativeLIFTrainer(p,runner=RUNNER);t.restore(path)
    assert calls==[(1,2),(3,1 if singleton else 2)] and t.next_noise_sequence==10
    assert len(t.poisson_state['entries'])==(3 if singleton else 4)



def reference(data,weights,mode,singleton,derived,window=None,initial=None,anchors=None,override=None):
    _,g,syn,_,bundle,x=data;p=bundle.plan
    z=np.array(bundle.initial_state if initial is None else initial,float)
    if initial is None:
        for cell,ref in enumerate(p['dynamic']['initial_parameters']):
            if ref is not None:z[cell]=weights[ref[0]][ref[1]]
    v=bundle.provenance['neuron_state_layout'][g.name]['v'];u=bundle.provenance['neuron_state_layout'][g.name]['u']
    bank=next(row['bank'] for row in bundle.provenance['bindings'] if row['object']==syn.name and row['variables']==['rate'])
    rates=np.array(weights[bank])[:1] if singleton else np.array(weights[bank])[::-1]
    rates=rates+(.1 if derived else 0.)
    paths=bundle.provenance['event_callback_stage_groups'][syn.pre.name][1:]
    assert len(paths)==(5 if mode=='vectorised' else 6)
    locals_=bundle.provenance['event_callback_caller_locals'][syn.pre.name]
    fields={row['key'][1]:row['fields'] for row in locals_['rows'] if row['key'][0]=='new'}
    output=[next(iter(stage['outputs'].values())) for stage in locals_['trace']['stages']]
    source=locals_['trace']['initial'];assert len(output)==5
    domain=bundle.provenance['scheduled_noise_domains'][syn.pre.name]
    posts=np.array(syn.j[:],int);samples=[];before=[];margins=[];hard=[];spikes=[];selections=[]
    for tick in range(5):
        if anchors is not None and window and tick and tick%window==0:z=anchors['before'][tick].copy()
        before.append(z.copy());z[v]+=.2*z[u];margin=z[v]-.5;event=(margin>0).astype(float)
        margins.append(margin.copy());hard.append(event.copy())
        if anchors is not None:
            old=anchors['margins'][tick];event=anchors['hard'][tick]+p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(old))**2*(margin-old)
        spikes.append(event.copy());selections.append([])
        for stage,path in enumerate(paths):
            for row in bundle.provenance['event_callback_snapshots'].get(path,[]):z[row['cache']]=z[row['source']]
            rows=bundle.provenance['delay_queues'][path]['new'];gates=[z[row['states'][0]] for row in rows]
            selected=[row['edge'] for row,gate in zip(rows,gates) if gate>.5]
            if anchors is not None:selected=anchors['selections'][tick][stage]
            selections[-1].append(selected)
            if stage==0:
                for edge,field in fields.items():z[field[source['v_post']]]=z[v[posts[edge]]]
            if stage==4 and selected:
                incoming=np.array([z[fields[edge][output[1]]]*z[fields[edge][output[3]]] for edge in selected])
                z[u]+=incoming[0] if len(incoming)==1 else incoming
            for row,gate in zip(rows,gates):
                edge=row['edge'];field=fields[edge]
                if stage==0:z[field[output[0]]]+=gate*(.1-z[field[output[0]]])
                elif stage==1:z[field[output[1]]]+=gate*(.2-z[field[output[1]]])
                elif stage==3 and edge in selected:
                    ordinal=selected.index(edge);index=0 if singleton else ordinal
                    count=observation_stream(p['seed'],9,domain,edge,tick-1,float(rates[index]),0)
                    samples.append((tick,edge,index,count))
                    if override==(tick,edge):count+=1
                    z[field[output[3]]]=count
                elif stage==4:
                    value=.7*z[field[output[0]]]*z[field[output[1]]]*z[field[output[3]]]
                    if mode=='vectorised':z[v[posts[edge]]]+=gate*value
                    else:z[field[output[4]]]+=gate*value
                elif stage==5:z[v[posts[edge]]]+=gate*(z[field[output[4]]]-z[v[posts[edge]]])
        for queue in bundle.provenance['delay_queues'].values():
            for row in queue['new']:
                cells=row['states'];z[cells[:-1]]=z[cells[1:]];z[cells[-1]]=x[0,tick,row['edge']]
        z[v]-=.5*event
    logits=np.array(spikes).mean(0)*p['logit_scale'];loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
    return loss,z,dict(before=before,margins=margins,hard=hard,samples=samples,selections=selections)


@pytest.mark.parametrize('mode',['array','vectorised'])
@pytest.mark.parametrize('singleton',[False,True])
@pytest.mark.parametrize('derived',[False,True])
@pytest.mark.parametrize('zero',[False,True])
@pytest.mark.parametrize('window',[None,2])
@pytest.mark.parametrize('ranks',[None,2])
def test_whole_rate_all_vjps(engine,mode,singleton,derived,zero,window,ranks):
    mpi(ranks);data=model(engine,ranks,mode,singleton,derived,zero);_,g,syn,_,bundle,x=data
    p=copy.deepcopy(bundle.plan);p['tbptt_window']=window
    out=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER).gradients(x,[0],noise_sequence=9)
    args=(data,bundle.weights,mode,singleton,derived)
    loss,z,anchors=reference(*args,window=window)
    cells=[*bundle.provenance['neuron_state_layout'][g.name]['v'],*bundle.provenance['neuron_state_layout'][g.name]['u']]
    np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],z[cells],rtol=8e-5,atol=8e-6);assert out['loss']==pytest.approx(loss,abs=8e-6)
    bank=next(row['bank'] for row in bundle.provenance['bindings'] if row['object']==syn.name and row['variables']==['rate'])
    rate_values=np.array(bundle.weights[bank])[:1] if singleton else np.array(bundle.weights[bank])[::-1]
    rate_values+=.1 if derived else 0.
    expected=np.zeros(2)
    for tick,edge,index,count in anchors['samples']:
        rate=rate_values[index];parameter=0 if singleton else 1-index
        expected[parameter]+=reference(*args,override=(tick,edge))[0]-loss if not rate else loss*(count/rate-1.)
    assert np.any(abs(expected)>1e-5)
    for bindex,row in enumerate(bundle.weights):
        for index in range(len(row)):
            assert out['gradients'][bindex][index]==pytest.approx(expected[index] if bindex==bank else 0.,rel=1e-3,abs=1e-5),(bindex,index)
    for index in range(len(z)):
        if p['dynamic']['detached'][index] or index in p['dynamic']['integer_states']:
            assert out['initial_state_gradients'][0][index]==0;continue
        hi=np.array(bundle.initial_state);lo=hi.copy();hi[index]+=1e-6;lo[index]-=1e-6
        fd=(reference(*args,window=window,initial=hi,anchors=anchors)[0]-reference(*args,window=window,initial=lo,anchors=anchors)[0])/2e-6
        assert out['initial_state_gradients'][0][index]==pytest.approx(fd,rel=1e-3,abs=1e-5),index


@pytest.mark.parametrize('derived',[False,True])
@pytest.mark.parametrize('kind',['shape','negative'])
@pytest.mark.parametrize('ranks',[None,2])
def test_whole_rate_restore_failure_atomic(engine,derived,kind,ranks,tmp_path):
    mpi(ranks);net,g,syn,dt,bundle,x=model(engine,ranks,'vectorised',kind=='negative',derived)
    if kind=='shape':
        source=next(obj for obj in net.objects if isinstance(obj,b.SpikeGeneratorGroup))
        source.set_spikes([0,1,0],np.array([0,0,2])*dt);x[0,2,1]=0
    p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights)
    t=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER);t.step(x[:,:3],[0],noise_sequence=9)
    path=tmp_path/'before-rate-error';t.store(path);t=NativeLIFTrainer(p,runner=RUNNER);t.restore(path)
    net.run(3*dt,namespace={})
    if kind=='negative':
        bank=next(row['bank'] for row in bundle.provenance['bindings'] if row['object']==syn.name and row['variables']==['rate'])
        t.state['weights'][bank][0]=-1.;syn.rate[0]=-1.
    before=copy.deepcopy((t.state,t.neuron_state,t.clock_state,t.poisson_state,t.elapsed_ticks,t.next_noise_sequence))
    with pytest.raises(ValueError):t.step(x[:,3:4],[0],initial='carry')
    assert (t.state,t.neuron_state,t.clock_state,t.poisson_state,t.elapsed_ticks,t.next_noise_sequence)==before
    with pytest.raises(b.BrianObjectException):net.run(dt,namespace={})


@pytest.mark.parametrize('derived',[False,True])
@pytest.mark.parametrize('ranks',[None,2])
def test_whole_rate_bank_replacement_restore(engine,derived,ranks,tmp_path,monkeypatch):
    mpi(ranks);net,g,syn,dt,bundle,x=model(engine,ranks,'vectorised',True,derived)
    p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights)
    t=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER)
    bank=next(row['bank'] for row in bundle.provenance['bindings'] if row['object']==syn.name and row['variables']==['rate'])
    domain=bundle.provenance['scheduled_noise_domains'][syn.pre.name]
    for tick in range(5):
        if tick==2:t.state['weights'][bank][0]=1.7;syn.rate[0]=1.7
        def sampler(rate,*args,**kwargs):
            rows=[edge for edge in range(2) if x[0,tick-1,edge]>0] if tick else []
            expected=np.array([float(syn.rate[0])+(.1 if derived else 0.)])
            np.testing.assert_allclose(rate,expected)
            return np.array([observation_stream(p['seed'],9,domain,edge,tick-1,float(expected[0]),0) for edge in rows],dtype=np.int32)
        monkeypatch.setattr(np.random,'poisson',sampler)
        out=t.step(x[:,tick:tick+1],[0],**({'initial':'carry'} if tick else {'noise_sequence':9}));net.run(dt,namespace={})
        for name in ('v','u'):
            cells=bundle.provenance['neuron_state_layout'][g.name][name]
            np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],g.variables[name].get_value(),rtol=8e-5,atol=8e-6)
        path=tmp_path/str(tick);t.store(path);t=NativeLIFTrainer(p,runner=RUNNER);t.restore(path)
    np.testing.assert_allclose(sorted({row['rate'] for row in t.poisson_state['entries']}),[1.2,1.8] if derived else [1.1,1.7],rtol=2e-6,atol=1e-7)


@pytest.mark.parametrize('mode',['array','vectorised'])
@pytest.mark.parametrize('ranks',[None,2])
def test_whole_rate_argument_borrow_reaches_later_operand(engine,mode,ranks,tmp_path,monkeypatch):
    mpi(ranks);net,g,syn,dt,bundle,x=model(engine,ranks,mode,True,with_h=True)
    p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights)
    t=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER)
    domain=bundle.provenance['scheduled_noise_domains'][syn.pre.name]
    for tick in range(5):
        def sampler(rate,*args,**kwargs):
            rows=[edge for edge in range(2) if x[0,tick-1,edge]>0] if tick else []
            return np.array([observation_stream(p['seed'],9,domain,edge,tick-1,float(syn.rate[0]),0) for edge in rows],dtype=np.int32)
        monkeypatch.setattr(np.random,'poisson',sampler)
        out=t.step(x[:,tick:tick+1],[0],**({'initial':'carry'} if tick else {'noise_sequence':9}));net.run(dt,namespace={})
        for name in ('v','u'):
            cells=bundle.provenance['neuron_state_layout'][g.name][name]
            np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],g.variables[name].get_value(),rtol=8e-5,atol=8e-6)
        path=tmp_path/str(tick);t.store(path);t=NativeLIFTrainer(p,runner=RUNNER);t.restore(path)
    np.testing.assert_array_equal(syn.h[:],[.113,.173])
