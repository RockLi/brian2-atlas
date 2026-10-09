"""Multi-level selected Brian references after dependency-ordered code generation."""
import copy
import numpy as np
import pytest
import brian2 as b
from brian2_rust import lower_brian_dynamic_training,NativeLIFTrainer
from test_native_training import RUNNER
from test_training_linked import cython_cache
from test_training_integer_ir import engine


def model(method='euler',noisy=False,routing='nested',**options):
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='cython';dt=.2*b.ms
    x=np.array([[1,1],[1,0],[0,1],[1,1],[1,0],[0,1]],float);ticks,ids=np.nonzero(x)
    inp=b.SpikeGeneratorGroup(2,ids,ticks*dt,dt=dt,name='bank_input')
    a=b.NeuronGroup(3,'dv/dt=-v/ms:1\ngain:1 (constant)'+('\nroute:integer'+(' (constant)' if routing=='fixed_map' else '')),threshold='v>10',reset='v=0',dt=dt,method=method,name='bank_a');a.gain=[.2,.35,.5];a.v=[.1,.2,.3]
    a.route=[2,0,1]
    c=b.NeuronGroup(2,'dv/dt=(-v+.2*peer)/ms'+('+.04*peer*xi/sqrt(ms)' if noisy else '')+':1\npeer:1 (linked)\npick:integer'+'\nroute:integer (linked)',
        threshold='v>peer',reset='pick=(pick+1)%3\nv-=.3*peer',dt=dt,method=method,name='bank_c')
    c.route=b.linked_var(a,'route',index='pick')
    c.variables.add_reference('peer',a,'gain',index='route')
    c.pick=[0,1];c.v=[.8,.55]
    syn=b.Synapses(inp,c,'edge_pick:integer\nedge_peer:1 (linked)',on_pre='edge_pick=(edge_pick+1)%3\nv_post+=.2*edge_peer',dt=dt,name='bank_syn');syn.connect(j='i');syn.edge_pick=[1,2];syn.edge_peer=b.linked_var(a,'gain',index='edge_pick')
    net=b.Network(inp,a,c,syn);bundle=lower_brian_dynamic_training(net,input_group=inp,layers=[a,c],trainable_neuron_parameters={a.name:['gain']},**options)
    assert '_deferred_' not in str(bundle.plan)
    return net,inp,a,c,syn,x,bundle


@pytest.mark.parametrize('method',['euler','rk4'])
@pytest.mark.parametrize('routing',['fixed_map','nested'])
def test_parameter_links_match_compiled_brian(engine,method,routing):
    net,inp,a,c,syn,x,bundle=model(method=method,routing=routing,backend=engine)
    result=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights).evaluate(x[None],[0]);mon=b.SpikeMonitor(c);net.add(mon);net.run(0*b.ms,namespace={})
    assert all(runner.codeobj.compiled_code['run'] is not None for runner in
               (syn.pre,c.state_updater,c.resetter['spike'],c.thresholder['spike']))
    net.run(len(x)*.2*b.ms,namespace={})
    z=np.asarray(result['final_state'])[0];layout=bundle.provenance['neuron_state_layout'];tol=1e-13 if engine=='cpu' else 5e-6
    for g in (a,c):np.testing.assert_allclose(z[layout[g.name]['v']],g.v[:],rtol=tol,atol=tol)
    np.testing.assert_array_equal(z[layout[c.name]['pick']],c.pick[:]);np.testing.assert_array_equal(z[bundle.provenance['dynamic_state_layout'][syn.name]['edge_pick']],syn.edge_pick[:])
    expected=np.zeros((len(x),2));expected[np.rint(np.asarray(mon.t/b.second)/.0002).astype(int),np.asarray(mon.i)]=1
    np.testing.assert_array_equal(np.asarray(result['spikes'])[0,:,3:],expected)
    if engine!='cpu':assert result['gpu_dispatches']>0


def oracle(bundle,x,w=None,initial=None,anchors=None):
    from test_training_stochastic import normal
    p=bundle.plan;w=bundle.weights if w is None else w;z=np.asarray(bundle.initial_state if initial is None else initial,float).copy()
    layout=bundle.provenance['neuron_state_layout'];av=layout['bank_a']['v'];cv=layout['bank_c']['v'];cp=layout['bank_c']['pick'];sp=bundle.provenance['dynamic_state_layout']['bank_syn']['edge_pick']
    bank=next(e['bank'] for e in bundle.provenance['bindings'] if e['object']=='bank_a' and e['variables']==['gain']);gain=np.asarray(w[bank])
    route_bank=next((e['bank'] for e in bundle.provenance['bindings'] if e['object']=='bank_a' and e['variables']==['route']),None)
    before=[];margins=[];hard=[];spikes=[]
    for t,inp in enumerate(x):
        if anchors is not None and p['tbptt_window'] and t and t%p['tbptt_window']==0:z=anchors['before'][t].copy()
        before.append(z.copy());z[av]*=.8;pick=z[cp].astype(int)
        if 'route' in layout['bank_a']:pick=z[layout['bank_a']['route']][pick].astype(int)
        elif route_bank is not None:pick=np.asarray(w[route_bank],int)[pick]
        peer=gain[pick]
        z[cv]=.8*z[cv]+.04*peer
        if p.get('noise_streams'):z[cv]+=.04*peer*np.sqrt(.2)*np.array([normal(p['seed'],0,0,1,j,t,0) for j in range(2)])
        margin=z[cv]-peer;z[bundle.provenance['threshold_margin_layout']['bank_c']]=margin
        h=(margin>0).astype(float);s=h.copy()
        if anchors is not None:h=anchors['hard'][t];s=h+p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(anchors['margin'][t]))**2*(margin-anchors['margin'][t])
        margins.append(margin.copy());hard.append(h);spikes.append(s)
        for j in range(2):
            if inp[j]:old=int(z[sp[j]]);z[sp[j]]=(old+1)%3;z[cv[j]]+=.2*gain[old]
        reset=h if p['detach_reset'] else s;z[cv]-=.3*peer*reset
        for j in range(2):
            if h[j]:z[cp[j]]=(int(z[cp[j]])+1)%3
    logits=np.asarray(spikes).mean(axis=0)*p['logit_scale'];loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
    return loss,z,np.asarray(spikes),dict(before=before,margin=np.asarray(margins),hard=np.asarray(hard))


@pytest.mark.parametrize('noisy',[False,True])
@pytest.mark.parametrize('detach,window',[(True,None),(False,2)])
@pytest.mark.parametrize('routing',['fixed_map','nested'])
def test_parameter_links_all_float_vjps(engine,noisy,detach,window,routing):
    *_,x,bundle=model(noisy=noisy,routing=routing,backend=engine,detach_reset=detach,tbptt_window=window)
    result=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights).gradients(x[None],[0]);loss,z,spikes,anchors=oracle(bundle,x)
    tol=5e-6 if engine=='cpu' else 9e-4;absolute=5e-8 if engine=='cpu' else 6e-6;eps=1e-6
    assert result['loss']==pytest.approx(loss,abs=absolute);np.testing.assert_allclose(result['final_state'][0],z,rtol=tol,atol=absolute);np.testing.assert_array_equal(np.asarray(result['spikes'])[0,:,3:],spikes)
    for bank,row in enumerate(bundle.weights):
        if any(ref[0]==bank for ref in bundle.plan['dynamic']['integer_parameters']):
            assert result['gradients'][bank]==[0.]*len(row);continue
        for k in range(len(row)):
            hi=copy.deepcopy(bundle.weights);lo=copy.deepcopy(bundle.weights);hi[bank][k]+=eps;lo[bank][k]-=eps
            fd=(oracle(bundle,x,w=hi,anchors=anchors)[0]-oracle(bundle,x,w=lo,anchors=anchors)[0])/(2*eps)
            assert result['gradients'][bank][k]==pytest.approx(fd,rel=tol,abs=absolute)
    for k,detached in enumerate(bundle.plan['dynamic']['detached']):
        if detached:assert result['initial_state_gradients'][0][k]==0.;continue
        hi=np.asarray(bundle.initial_state).copy();lo=hi.copy();hi[k]+=eps;lo[k]-=eps
        fd=(oracle(bundle,x,initial=hi,anchors=anchors)[0]-oracle(bundle,x,initial=lo,anchors=anchors)[0])/(2*eps)
        assert result['initial_state_gradients'][0][k]==pytest.approx(fd,rel=tol,abs=absolute)
