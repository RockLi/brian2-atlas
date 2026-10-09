"""Delayed recurrent STDP conformance adapters; timings are diagnostic until gated.

Uses the full target-pathway workload (heterogeneous pre, delayed post, recurrent
voltage transmission, mutable weights and event-driven traces). No f64 gate is
relaxed on behalf of an adapter. GeNN delay grouping is explicit in each report.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time
import traceback
import numpy as np

DT=1/1024
FIELDS=('w','Apre','Apost','lastupdate')
BACKENDS=('rust','cpu-f32','metal','cuda','brian2cuda','brian2genn','genn')
MAX_NEURONS=16384


def workload_options(drive=.125,delay_span=4,post_delay=2,topology_kind="ring",topology_seed=0):
    if isinstance(drive,bool) or not isinstance(drive,(float,int)) or not np.isfinite(drive) or not 0<=drive<=.5:
        raise ValueError('drive must be finite within 0..0.5 per tick')
    if type(delay_span) is not int or not 1<=delay_span<=16:raise ValueError('delay_span must be 1..16')
    if type(post_delay) is not int or not 0<=post_delay<=16:raise ValueError('post_delay must be 0..16')
    if topology_kind not in {'ring','random-fixed-outdegree'}:raise ValueError('Unknown topology kind')
    if type(topology_seed) is not int or not 0<=topology_seed<2**32:raise ValueError('topology_seed must be a uint32 integer')
    if topology_kind=='ring' and topology_seed!=0:raise ValueError('Ring topology requires seed zero')
    result=dict(drive=float(drive),delay_span=delay_span,post_delay=post_delay)
    if topology_kind!='ring':result.update(topology_kind=topology_kind,topology_seed=topology_seed)
    return result


def topology(neurons,degree,*,delay_span=4,topology_kind="ring",topology_seed=0):
    if not 2<=neurons<=MAX_NEURONS or not 1<=degree<neurons or degree>128:
        raise ValueError('Require 2..16384 neurons and 1..min(N-1,128) degree')
    workload_options(delay_span=delay_span,topology_kind=topology_kind,topology_seed=topology_seed)
    if topology_kind=='random-fixed-outdegree' and neurons>4096:
        raise ValueError('Rank-based random topology is bounded to 4096 neurons; use ring for larger comparisons')
    sources=np.repeat(np.arange(neurons,dtype=np.int64),degree)
    if topology_kind=='ring':
        targets=(sources+np.tile(np.arange(degree),neurons)+1)%neurons
    else:
        # SplitMix64 scores define version-independent pseudorandom target
        # ranks. Select without replacement; exclude self and sort each row
        # to make creation order explicit. No global RNG state is consumed.
        targets=np.empty(len(sources),np.int64)
        candidates=np.arange(neurons,dtype=np.uint64)
        for source in range(neurons):
            keys=candidates+np.uint64(source*neurons+topology_seed)
            scores=keys+np.uint64(0x9e3779b97f4a7c15)
            scores=(scores^(scores>>np.uint64(30)))*np.uint64(0xbf58476d1ce4e5b9)
            scores=(scores^(scores>>np.uint64(27)))*np.uint64(0x94d049bb133111eb)
            scores=scores^(scores>>np.uint64(31))
            eligible=np.concatenate((np.arange(source),np.arange(source+1,neurons)))
            chosen=eligible[np.argsort(scores[eligible],kind='stable')[:degree]]
            targets[source*degree:(source+1)*degree]=np.sort(chosen)
    return sources,targets,np.arange(len(sources),dtype=np.int64)%delay_span


def configuration(neurons,degree,steps,*,drive=.125,delay_span=4,post_delay=2,topology_kind="ring",topology_seed=0):
    opts=workload_options(drive,delay_span,post_delay,topology_kind,topology_seed)
    sources,targets,delay=topology(neurons,degree,delay_span=delay_span,topology_kind=topology_kind,topology_seed=topology_seed)
    result=dict(case='recurrent-delayed-stdp-v0',neurons=neurons,degree=degree,steps=steps,dt_seconds=DT,
        initial_v='(i%16)/16',initial_w=.25,neuron='start: v+=.125; groups: v+=(-v/32); threshold: v>1; reset: v=0',
        taupre_ticks=16,taupost_ticks=32,
        pre='decay both traces; v_post+=w/16; Apre+=.0078125; w=clip(w+Apost,0,.5); lastupdate=t',
        post='decay both traces; Apost-=.00390625; w=clip(w+Apre,0,.5); lastupdate=t',
        pre_delay='edge_index%4 ticks',post_delay_ticks=2,pre_before_post=True,
        topology_sha256=hashlib.sha256(sources.tobytes()+targets.tobytes()+delay.tobytes()).hexdigest(),
        recording='all spikes; final v,w,Apre,Apost,lastupdate; start v after drive of neurons 0,N-1 every tick',
        acceptance=dict(float_rtol=2e-5,float_atol=2e-6,spikes='exact ticks and neuron indices',lastupdate='exact'))
    if opts!=workload_options():
        result.update(case='recurrent-delayed-stdp-v1',workload=opts,
            neuron=f'start: v+={float(drive)!r}; groups: v+=(-v/32); threshold: v>1; reset: v=0',
            pre_delay=f'edge_index%{delay_span} ticks',post_delay_ticks=post_delay)
    if topology_kind!='ring':
        counts=np.bincount(targets,minlength=neurons)
        result.update(case='recurrent-delayed-stdp-v2',
            topology=dict(kind=topology_kind,seed=topology_seed,
                algorithm='splitmix64-key-source-times-N-plus-target-plus-seed-v1',
                self_edges=False,duplicate_edges=False,creation_order='source, then sorted target',
                indegree_min=int(counts.min()),indegree_max=int(counts.max()),
                indegree_mean=float(counts.mean()),indegree_std=float(counts.std())))
    return result


def oracle(neurons,degree,steps,dtype=np.float64,*,drive=.125,delay_span=4,post_delay=2,topology_kind="ring",topology_seed=0):
    """Independent ordered event recurrence in seconds, including reset ordering."""
    workload_options(drive,delay_span,post_delay,topology_kind,topology_seed)
    source,target,delay=topology(neurons,degree,delay_span=delay_span,topology_kind=topology_kind,topology_seed=topology_seed);edges=len(source)
    v=(np.arange(neurons)%16/16).astype(dtype);w=np.full(edges,.25,dtype)
    ap=np.zeros(edges,dtype);ao=ap.copy();last=np.zeros(edges,np.float64)
    history=np.zeros((steps,neurons),bool);trace=[];ticks=[];indices=[]
    # Brian queues older emissions first, then source index and creation order.
    order=np.lexsort((np.arange(edges),source,-delay))
    for tick in range(steps):
        v+=dtype(drive);trace.append(v[[0,neurons-1]].copy());v+=dtype(-1/32)*v
        fired=v>1;history[tick]=fired;sp=np.flatnonzero(fired)
        ticks.extend([tick]*len(sp));indices.extend(sp)
        pre=order[(tick>=delay[order]) & history[np.maximum(tick-delay[order],0),source[order]]]
        post=np.flatnonzero(history[tick-post_delay,target]) if tick>=post_delay else np.empty(0,np.int64)
        for events,is_pre in ((pre,True),(post,False)):
            # Events for a given edge are serial; target stores preserve global order.
            elapsed=tick*DT-last[events]
            ap[events]*=np.exp(-elapsed/(16*DT)).astype(dtype)
            ao[events]*=np.exp(-elapsed/(32*DT)).astype(dtype)
            if is_pre:
                np.add.at(v,target[events],w[events]/dtype(16));ap[events]+=dtype(.0078125)
                w[events]=np.clip(w[events]+ao[events],0,.5)
            else:
                ao[events]-=dtype(.00390625);w[events]=np.clip(w[events]+ap[events],0,.5)
            last[events]=tick*DT
        v[fired]=0
    return dict(v=v,w=w,Apre=ap,Apost=ao,lastupdate=last,trace=np.asarray(trace),
        ticks=np.asarray(ticks,np.int64),indices=np.asarray(indices,np.int64))



def activity(neurons,degree,steps,result,*,drive=.125,delay_span=4,post_delay=2,topology_kind="ring",topology_seed=0):
    workload_options(drive,delay_span,post_delay,topology_kind,topology_seed)
    source,target,delay=topology(neurons,degree,delay_span=delay_span,topology_kind=topology_kind,topology_seed=topology_seed)
    history=np.zeros((steps+1,neurons),np.int64)
    np.add.at(history,(result['ticks']+1,result['indices']),1)
    history=np.cumsum(history,axis=0)
    return dict(spikes=len(result['ticks']),mean_firing_hz=len(result['ticks'])/(neurons*steps*DT),
        delivered_pre_events=int(history[np.maximum(steps-delay,0),source].sum()),
        delivered_post_events=int(history[max(steps-post_delay,0),target].sum()))


def checks(actual,expected):
    rows={}
    for key,value in expected.items():
        a=actual.get(key);shape=a is not None and a.shape==value.shape
        exact=shape and np.array_equal(a,value)
        ok=exact if key in {'ticks','indices','lastupdate'} else shape and np.allclose(a,value,rtol=2e-5,atol=2e-6)
        rows[key]=dict(passed=bool(ok),exact=bool(exact),shape=list(a.shape) if a is not None else None,
            max_abs=float(np.max(np.abs(a.astype(float)-value.astype(float)),initial=0)) if shape else None)
    return dict(passed=all(r['passed'] for r in rows.values()),fields=rows)


def build_brian(neurons,degree,split=False,*,drive=.125,delay_span=4,post_delay=2,topology_kind="ring",topology_seed=0):
    workload_options(drive,delay_span,post_delay,topology_kind,topology_seed)
    import brian2 as b
    dt=DT*b.second;b.defaultclock.dt=dt
    pop=b.NeuronGroup(neurons,'dv/dt=-v/tau:1',threshold='v>1',reset='v=0',method='euler',
        namespace={'tau':32*dt},clock=b.defaultclock,name='population')
    pop.v=np.arange(neurons)%16/16;pop.run_regularly(f'v+={float(drive)!r}',name='population_run_regularly')
    source,target,delays=topology(neurons,degree,delay_span=delay_span,topology_kind=topology_kind,topology_seed=topology_seed);groups=[]
    partitions=[np.flatnonzero(delays==d) for d in reversed(range(delay_span))] if split else [np.arange(len(source))]
    for q,edge in enumerate(partitions):
        if not len(edge):continue
        options={'delay':{'pre':int(delays[edge[0]])*dt,'post':post_delay*dt}} if split else {}
        syn=b.Synapses(pop,pop,'dApre/dt=-Apre/taupre:1 (event-driven)\ndApost/dt=-Apost/taupost:1 (event-driven)\nw:1',
            on_pre='v_post+=w/16; Apre+=0.0078125; w=clip(w+Apost,0,0.5)',
            on_post='Apost-=0.00390625; w=clip(w+Apre,0,0.5)',
            namespace={'taupre':16*dt,'taupost':32*dt},clock=pop.clock,name='plastic'+str(q),**options)
        syn.connect(i=source[edge],j=target[edge]);syn.w=.25
        if not split:syn.pre.delay=delays*dt;syn.post.delay=post_delay*dt
        groups.append((syn,edge))
    spikes=b.SpikeMonitor(pop,name='spikes');trace=b.StateMonitor(pop,'v',record=[0,neurons-1],when='start',name='voltage')
    return b.Network(pop,*(s for s,e in groups),spikes,trace),pop,groups,spikes,trace


def brian_run(backend,neurons,degree,steps,output,split=False,*,prepared=None,cpp_threads=1,drive=.125,delay_span=4,post_delay=2,topology_kind="ring",topology_seed=0):
    opts=workload_options(drive,delay_span,post_delay,topology_kind,topology_seed)
    import brian2 as b
    b.get_device().reinit();b.prefs.core.default_float_dtype=np.float64 if backend in {'rust','cpp-f64'} else np.float32
    if backend in {'rust','cpu-f32','metal','cuda'}:
        import brian2_atlas
        from brian2.devices.device import all_devices
        all_devices['atlas'].reinit()
        options=dict(engine='aot' if backend=='rust' else 'reference',directory=output/'project',
            runner=Path(os.environ.get('B2_RUNNER',Path(__file__).resolve().parents[1]/'target/release/b2-runner')))
        if backend in {'metal','cuda'}:options.update(engine=backend,numeric_mode='float32',event_delivery='sparse')
        b.set_device('atlas',**options)
    elif backend=='cpp-f64':
        from brian2.devices.device import all_devices
        all_devices['cpp_standalone'].reinit()
        if cpp_threads not in (1,2,4):raise ValueError('CPP OpenMP threads must be 1, 2 or 4')
        import sys
        if sys.platform=='darwin':
            from threaded_benchmark import prepare_cxx
            compiler,_=prepare_cxx(output,'apple-clang-libomp');os.environ['CXX']=compiler
        b.set_device('cpp_standalone',directory=str(output/'project'))
        b.prefs.devices.cpp_standalone.openmp_threads=cpp_threads
        b.prefs.devices.cpp_standalone.extra_make_args_unix=['-j2']
        b.prefs.codegen.cpp.extra_compile_args_gcc=['-O3','-std=c++17','-fno-fast-math','-ffp-contract=off']
        b.get_device().insert_code('before_end',
            '#pragma omp parallel\n{\n#pragma omp single\n{\n'
            'std::ofstream proof(brian::results_dir + "openmp_threads.txt"); '
            'proof << omp_get_num_threads() << "\\n";\n}\n}\n')
    elif backend=='brian2cuda':
        import brian2cuda
        b.set_device('cuda_standalone',directory=str(output/'project'))
        b.prefs.devices.cpp_standalone.extra_make_args_unix=['-j2']
    else:
        import brian2genn
        b.set_device('genn',directory=str(output/'project'),use_GPU=True)
        b.prefs.devices.cpp_standalone.extra_make_args_unix=['-j2']
    net,pop,groups,spikes,trace=build_brian(neurons,degree,split,**opts)
    if prepared is not None:prepared.update(network=net,population=pop,groups=groups,monitor=spikes,trace=trace,device=b.get_device())
    if backend=='cpu-f32':
        from gpu_baseline import cpu_f32_control
        control_context=prepared if prepared is not None else {}
        raw=cpu_f32_control(net,steps*DT*b.second,output,event_delivery='sparse',prepared=control_context)
        group_index={syn['name']:q for q,syn in enumerate(control_context['control'].model['definition']['synapses'])}
        p=raw['populations'][0]
        result=dict(v=p['states']['v'],ticks=p['spike_ticks'],indices=p['indices'],trace=p['trace']['v'])
        for key in FIELDS:
            result[key]=np.empty(neurons*degree,np.float64 if key=='lastupdate' else np.float32)
            for syn,edges in groups:result[key][edges]=raw['synapses'][group_index[syn.name]]['states'][key]
        return result,dict(simulation_seconds=raw['run_seconds'])
    start=time.perf_counter();net.run(steps*DT*b.second);elapsed=time.perf_counter()-start
    ticks=np.rint(np.asarray(spikes.t[:]/b.second)/DT).astype(np.int64);indices=np.asarray(spikes.i[:],np.int64)
    order=np.lexsort((indices,ticks))
    result=dict(v=np.asarray(pop.v[:]).copy(),ticks=ticks[order],indices=indices[order],trace=np.asarray(trace.v).T.copy())
    for key in FIELDS:
        result[key]=np.empty(neurons*degree,np.float64 if key=='lastupdate' else result['v'].dtype)
        for syn,edges in groups:
            values=getattr(syn,key)[:]
            result[key][edges]=np.asarray(values/b.second if key=='lastupdate' else values)
    return result,dict(build_run_seconds=elapsed)


def genn_run(neurons,degree,steps,output,*,prepared=None,trace_mode="host",drive=.125,delay_span=4,post_delay=2,topology_kind="ring",topology_seed=0):
    """Four axonal-delay groups; an explicit final synapse flush aligns readback.

    GeNN processes previous spikes before neuron updates. The extra last step
    runs pending synapses without another neuron update or spike. Input received
    in the same Brian tick as a reset must be discarded. These mappings remain
    subject to full trajectory/weight gates (especially coincident pre/post).
    """
    workload_options(drive,delay_span,post_delay,topology_kind,topology_seed)
    if trace_mode not in {'host','device'}:raise ValueError('Unknown GeNN trace mode')
    from pygenn import GeNNModel,create_neuron_model,create_weight_update_model,init_weight_update,init_postsynaptic
    import pygenn.genn_model as gm
    dt_ms=DT*1000
    model=GeNNModel('float','b2_delayed_stdp',backend='cuda');model.dt=dt_ms
    record=f'if(id==0) trace[2*(unsigned int)(t/dt)]=v_start; if(id=={neurons-1}) trace[2*(unsigned int)(t/dt)+1]=v_start;' if trace_mode=='device' else ''
    neuron=create_neuron_model('DelayedSTDPNeuron',params=['stop'],vars=[('v','scalar'),('previous_spike','int'),('v_start','scalar')],
        sim_code='if(!previous_spike) v += Isyn; previous_spike=0; if(t<stop) { v+='+repr(float(drive))+'f; v_start=v; '+record+' v+=(-0.03125f)*v; }',
        extra_global_params=[('trace','scalar*')] if trace_mode=='device' else None,
        threshold_condition_code='t<stop && v>1.0f',reset_code='v=0.0f; previous_spike=1;')
    pop=model.add_neuron_population('population',neurons,neuron,{'stop':steps*dt_ms},
        dict(v=(np.arange(neurons)%16/16).astype(np.float32),previous_spike=0,v_start=0))
    pop.spike_recording_enabled=True
    if trace_mode=='device':pop.extra_global_params['trace'].set_init_values(np.zeros(2*steps,np.float32))
    decay='const scalar event_time=(t/dt-1.0f)*0.0009765625f; const scalar elapsed=event_time-lastupdate; Apre*=exp(-elapsed/0.015625f); Apost*=exp(-elapsed/0.03125f); '
    update=create_weight_update_model('DelayedPairSTDP',vars=[(k,'scalar') for k in FIELDS],
        pre_spike_syn_code=decay+'addToPost(w/16.0f); Apre+=0.0078125f; w=fmin(0.5f,fmax(0.0f,w+Apost)); lastupdate=event_time;',
        post_spike_syn_code=decay+'Apost-=0.00390625f; w=fmin(0.5f,fmax(0.0f,w+Apre)); lastupdate=event_time;')
    source,target,delays=topology(neurons,degree,delay_span=delay_span,topology_kind=topology_kind,topology_seed=topology_seed);groups=[]
    for d in reversed(range(delay_span)):
        edges=np.flatnonzero(delays==d)
        if not len(edges):continue
        syn=model.add_synapse_population('delay'+str(d),'SPARSE',pop,pop,
            init_weight_update(update,{},dict(w=.25,Apre=0,Apost=0,lastupdate=0)),init_postsynaptic('DeltaCurr'))
        syn.axonal_delay_steps=d;syn.back_prop_delay_steps=post_delay;syn.set_sparse_connections(source[edges],target[edges]);groups.append((syn,edges))
    (output/'project').mkdir(parents=True,exist_ok=True)
    old=gm.cpu_count;start=time.perf_counter()
    try:gm.cpu_count=lambda logical=True:2;model.build(path_to_model=str(output/'project'))
    finally:gm.cpu_count=old
    compile_seconds=time.perf_counter()-start
    context=dict(model=model,population=pop,groups=groups,neurons=neurons,degree=degree,trace_mode=trace_mode)
    if prepared is not None:prepared.update(context)
    start=time.perf_counter();result=genn_replay(context,steps);elapsed=time.perf_counter()-start
    return result,dict(compile_seconds=compile_seconds,initialize_simulate_readback_seconds=elapsed,
        adapter=f'{len(groups)} homogeneous axonal groups, {post_delay}-tick backprop delay; final synapse flush; previous-spike reset discard',
        trace_mode=trace_mode,trace_readback='per-step diagnostic pulls' if trace_mode=='host' else 'GPU trace buffer, one final pull')


def genn_replay(context,steps):
    model=context['model'];pop=context['population'];groups=context['groups'];neurons=context['neurons']
    start=time.perf_counter();model.load(num_recording_timesteps=steps+1)
    loaded=time.perf_counter();timings=dict(load_seconds=loaded-start)
    try:
        traces=[]
        for tick in range(steps+1):
            model.step_time()
            if context['trace_mode']=='host':
                pop.vars['v_start'].pull_from_device()
                if tick<steps:traces.append(pop.vars['v_start'].current_values.reshape(-1)[[0,neurons-1]].copy())
        if context['trace_mode']=='device':
            pop.extra_global_params['trace'].pull_from_device()
            trace=pop.extra_global_params['trace'].values.reshape(steps,2).copy()
        else:trace=np.asarray(traces)
        model.pull_recording_buffers_from_device()
        synced=time.perf_counter();timings['steps_trace_and_spike_sync_seconds']=synced-loaded
        pop.vars['v'].pull_from_device()
        times,indices=pop.spike_recording_data[0];ticks=np.rint(times/(DT*1000)).astype(np.int64);indices=np.asarray(indices,np.int64)
        order=np.lexsort((indices,ticks));result=dict(v=pop.vars['v'].current_values.copy().reshape(-1),ticks=ticks[order],indices=indices[order],trace=trace)
        for key in FIELDS:
            result[key]=np.empty(neurons*context['degree'],np.float64 if key=='lastupdate' else np.float32)
            for syn,edges in groups:
                syn.vars[key].pull_from_device()
                result[key][edges[syn.synapse_order]]=syn.vars[key].values.reshape(-1)
        timings['remaining_state_pull_and_decode_seconds']=time.perf_counter()-synced
        return result
    finally:
        start_unload=time.perf_counter();model.unload();timings['unload_seconds']=time.perf_counter()-start_unload
        context['last_replay_timings']=timings


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--backend',choices=BACKENDS,required=True)
    p.add_argument('--neurons',type=int,default=256);p.add_argument('--degree',type=int,default=128);p.add_argument('--steps',type=int,default=128)
    p.add_argument('--genn-trace',choices=('host','device'),default='host');p.add_argument('--split-delays',action='store_true');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    config=configuration(a.neurons,a.degree,a.steps)
    if not 1<=a.steps<=512:p.error('steps must be 1..512')
    a.output.mkdir(parents=True,exist_ok=False)
    report=dict(schema='b2-stdp-comparison-v0',backend=a.backend,configuration=config,
        model_sha256=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest(),
        split_delays=a.split_delays or a.backend=='genn',timing_eligible=False,
        timing_scope='diagnostic fresh adapter through full arrays, includes imports/build/initialization/run/readback; excludes oracle/checks/serialization',
        versions={},compiler_flags={k:os.environ.get(k,'') for k in ('NVCC_PREPEND_FLAGS','NVCC_APPEND_FLAGS')})
    for name in ('numpy','brian2','brian2cuda','brian2genn','pygenn'):
        try:report['versions'][name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:pass
    start=time.perf_counter()
    try:
        if a.backend=='genn':result,timing=genn_run(a.neurons,a.degree,a.steps,a.output,trace_mode=a.genn_trace)
        else:result,timing=brian_run(a.backend,a.neurons,a.degree,a.steps,a.output,a.split_delays)
        report.update(adapter_wall_seconds=time.perf_counter()-start,timing=timing)
        expected=oracle(a.neurons,a.degree,a.steps)
        report['reference_f64']=checks(result,expected)
        report['result_arrays']={k:dict(dtype=v.dtype.str,shape=list(v.shape),sha256=hashlib.sha256(v.tobytes()).hexdigest()) for k,v in result.items()}
        np.savez_compressed(a.output/'result.npz',**result)
        report['status']='passed' if report['reference_f64']['passed'] else 'correctness_failed'
    except Exception as error:
        report.update(status='execution_error',error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),adapter_wall_seconds=time.perf_counter()-start)
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    if report['status']!='passed':raise SystemExit(1)


if __name__=='__main__':main()
