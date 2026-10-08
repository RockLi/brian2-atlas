"""Seeded mixed-feature networks: exact arithmetic exposes scheduling errors."""
import json
from pathlib import Path
import brian2 as b
import numpy as np
import pytest
from brian2_rust.export import lower_network
from brian2_rust.metal import build_metal_plan
from test_metal_delays import device,ROOT
from test_gpu_spike_generator import BACKENDS,execute
from test_gpu_expression_contract import oracle
from test_gpu_custom_events import compare
from test_gpu_workgroup import save

DT=b.second/1024


def setup(path,engine='reference',**kwargs):
    b.set_device('rust_standalone',engine=engine,directory=path,runner=ROOT/'target/release/b2-runner',
        **(dict(numeric_mode='float32',event_delivery='sparse') if engine in {'metal','cuda'} else {}),**kwargs)


def network(seed,*,edges=1028,traced=False):
    rng=np.random.default_rng(seed)
    pre=b.NeuronGroup(257,'v:1\nu:1\ncounter:integer',dtype={'counter':np.int64},
        threshold='v>=1',reset='v-=1',events={'burst':'v>=0.75'},dt=DT,name='pre')
    post=b.NeuronGroup(263,'v:1\nu:1\ncounter:integer\nremote:integer (linked)\nmapping:integer (constant)',
        dtype={'counter':np.int64,'remote':np.int64,'mapping':np.int64},threshold='v>=1',reset='v-=1',dt=1.5*DT,name='post')
    pre.v=rng.integers(0,32,len(pre))/32;post.v=rng.integers(0,32,len(post))/32
    pre.counter=np.arange(len(pre),dtype=np.int64)+2**54
    post.mapping=rng.integers(0,len(pre),len(post),dtype=np.int64)
    post.remote=b.linked_var(pre,'counter',index='mapping')
    pre.run_regularly('v+=0.125; counter+=1',when='groups',name='pre_update')
    post.run_regularly('v+=0.0625; counter=remote',dt=2*DT,when='before_thresholds',name='post_update')
    equations='w:1\nhits:integer\ndx/dt=1/second:1 (clock-driven)\nu_post=w:1 (summed)\nu_pre=x:1 (summed)'
    if traced:equations+='\ndeligibility/dt=-64*Hz*eligibility:1 (event-driven)'
    syn=b.Synapses(pre,post,equations,
        dtype={'hits':np.int64},on_pre='v_post+=w; w=clip(w+1.0/4096,0,1.0/16); hits+=1',
        on_post='w=clip(w-1.0/8192,0,1.0/16)',on_event={'pre':'burst','post':'spike'},
        clock=pre.clock,method='euler',name='plastic')
    syn.connect(i=rng.integers(0,len(pre),edges),j=rng.integers(0,len(post),edges))
    syn.w=rng.integers(1,32,edges)/1024;syn.hits=np.arange(edges,dtype=np.int64)+2**54
    if traced:syn.eligibility=rng.integers(1,32,edges)/32
    syn.pre.delay=rng.integers(0,9,edges)*DT;syn.post.delay=rng.integers(0,7,edges)*1.5*DT
    feedback=b.Synapses(post,pre,'w:1 (constant)',on_pre='v_post-=w',clock=post.clock,name='feedback')
    feedback.connect(i=rng.integers(0,len(post),789),j=rng.integers(0,len(pre),789))
    feedback.w=rng.integers(1,8,789)/1024;feedback.delay=rng.integers(0,5,789)*1.5*DT
    monitors=[b.StateMonitor(p,['v','u','counter'],record=[0,len(p)//2,len(p)-1],name='monitor_'+p.name) for p in (pre,post)]
    spikes=[b.SpikeMonitor(p,name='spikes_'+p.name) for p in (pre,post)]
    events=b.EventMonitor(pre,'burst',variables=['v'],name='bursts')
    net=b.Network(pre,post,syn,feedback,*monitors,*spikes,events)
    if seed%2:net.schedule=['start','groups','synapses','thresholds','resets','end']
    return net,pre,post,syn,monitors,spikes,events


def snapshot(pre,post,syn,monitors,spikes,events):
    d={}
    for p in (pre,post):
        for name in ('v','u','counter'):d[p.name+'/'+name]=np.asarray(getattr(p,name)[:]).copy()
    d['post/remote']=np.asarray(post.remote[:]).copy()
    for name in ('w','x','hits',*(('eligibility','lastupdate') if 'eligibility' in syn.variables else ())):
        d['syn/'+name]=np.asarray(getattr(syn,name)[:]).copy()
    for i,m in enumerate(monitors):
        for name in ('v','u','counter'):d[f'monitor/{i}/{name}']=np.asarray(getattr(m,name)[:]).copy()
        d[f'monitor/{i}/t']=np.asarray(m.t[:]).copy()
    for i,s in enumerate(spikes):
        for name in ('i','t','count'):d[f'spikes/{i}/{name}']=np.asarray(getattr(s,name)[:]).copy()
    for name in ('v','i','t','count'):d['burst/'+name]=np.asarray(getattr(events,name)[:]).copy()
    return d


@pytest.mark.parametrize('seed',[7,42,2026,65537])
@pytest.mark.parametrize('route',['scan','sparse'])
@pytest.mark.parametrize('backend',BACKENDS)
def test_seeded_mixed_clocks_links_delays_plasticity(device,tmp_path,seed,route,backend):
    setup(tmp_path/'ref');net,*_=network(seed);model=lower_network(net,24*DT)
    (tmp_path/'model.json').write_text(json.dumps(model,sort_keys=True)+'\n')
    plan=build_metal_plan(model,numeric_mode='float32',event_delivery=route)
    assert len(plan.logical.clocks)==3
    expected=oracle(model,tmp_path/'oracle')
    assert sum(len(p['indices']) for p in expected['populations'])>0
    assert any(len(p.get('event_streams',{}).get('burst',{}).get('indices',[])) for p in expected['populations'])
    actual=execute(model,tmp_path/backend,backend,route)
    compare(actual,expected)
    save(tmp_path/'composed-results.npz',actual,expected)


@pytest.mark.parametrize('seed',[7,42])
@pytest.mark.parametrize('queued',[False,True])
@pytest.mark.parametrize('backend',BACKENDS[1:])
def test_seeded_segment_pending_restore_and_mutation(device,tmp_path,seed,queued,backend):
    records=[]
    for engine in ('reference',backend):
        device.reinit();setup(tmp_path/engine,engine,build_on_run=not queued)
        net,pre,post,syn,monitors,spikes,events=network(seed)
        net.run(6*DT)
        if not queued:net.store('pending')
        net.run(6*DT)
        if not queued:
            before=snapshot(pre,post,syn,monitors,spikes,events)
            net.restore('pending');net.run(6*DT)
            restored=snapshot(pre,post,syn,monitors,spikes,events)
            for key in before:np.testing.assert_array_equal(restored[key],before[key],err_msg=key)
            syn.w=np.asarray(syn.w[:])+1/4096
        net.run(12*DT)
        if queued:device.build()
        records.append(snapshot(pre,post,syn,monitors,spikes,events))
    assert records[0].keys()==records[1].keys()
    for key in records[0]:np.testing.assert_array_equal(records[1][key],records[0][key],err_msg=key)
    np.savez_compressed(tmp_path/'composed-device.npz',**{label+'/'+k:v for label,d in zip(('reference','actual'),records) for k,v in d.items()})
