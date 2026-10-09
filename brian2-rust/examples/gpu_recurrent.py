"""Shared recurrent current-based LIF workload for isolated GPU comparisons."""
import time
from pathlib import Path

import numpy as np

DT_MS=.1
TAU_MS=20.0
SYN_TAU_MS=5.0
REF_TICKS=20
DELAY_TICKS=20


def validate_size(neurons,degree):
    ne=int(.8*neurons);ki=max(1,degree//5);ke=degree-ki
    if not 1<=ke<=ne or not 1<=ki<=neurons-ne:
        raise ValueError('degree must supply both E/I inputs within available source populations')
    if neurons*degree>1000000:
        raise ValueError('comparison workload is bounded to 1,000,000 total edges')


def configuration(neurons,steps,degree):
    return dict(case='recurrent-cuba-v0',neurons=neurons,steps=steps,degree=degree,
                dt_ms=DT_MS,tau_ms=TAU_MS,syn_tau_ms=SYN_TAU_MS,refractory_ticks=REF_TICKS,
                delay_ticks=DELAY_TICKS,excitatory_fraction=.8,weights=[.05,-.2],
                drive='1.15 + (i % 7)*.02',threshold='v>1',reset='v=0',method='euler',
                recording='all spike ticks/indices and final v; raw synaptic current diagnostic',
                state_dtype='float32',seed=1729,
                acceptance=dict(v_rtol=1e-4,v_atol=5e-6,spikes='exact ticks and indices'))


def arrays(neurons,degree):
    validate_size(neurons,degree)
    ne=int(.8*neurons);ki=max(1,degree//5);ke=degree-ki
    index=np.arange(neurons)
    initial=((index*37%1024)/1024).astype(np.float32)
    drive=(1.15+(index%7)*.02).astype(np.float32)
    projections=[];mask=(1<<64)-1
    for kind,(start,size,count,weight) in enumerate([(0,ne,ke,.05),(ne,neurons-ne,ki,-.2)]):
        sources=[];targets=[]
        for target in range(neurons):
            chosen=set();state=1729+target*2+kind
            while len(chosen)<count:
                state=(state+0x9e3779b97f4a7c15)&mask
                x=((state^(state>>30))*0xbf58476d1ce4e5b9)&mask
                x=((x^(x>>27))*0x94d049bb133111eb)&mask
                source=start+((x^(x>>31))%size)
                if source in chosen:continue
                chosen.add(source);sources.append(source);targets.append(target)
        projections.append((np.asarray(sources,np.uint32),np.asarray(targets,np.uint32),np.float32(weight)))
    return initial,drive,projections


def oracle(neurons,steps,degree):
    """Independent f64 Euler/event reference; initial arrays retain their f32 values."""
    v,drive,projections=arrays(neurons,degree);v=v.astype(np.float64);drive=drive.astype(np.float64)
    current=np.zeros(neurons);until=np.zeros(neurons,np.int64);events={};ticks=[];indices=[]
    adjacency=[]
    for sources,targets,weight in projections:
        order=np.argsort(sources,kind='stable')
        offsets=np.concatenate(([0],np.cumsum(np.bincount(sources,minlength=neurons))))
        adjacency.append((offsets,targets[order],float(weight)))
    for tick in range(steps):
        available=tick>=until
        v[available]+=DT_MS*(drive[available]-v[available]+current[available])/TAU_MS
        current-=DT_MS*current/SYN_TAU_MS
        fired=np.flatnonzero(available&(v>1))
        ticks.extend([tick]*len(fired));indices.extend(fired)
        events[tick+DELAY_TICKS]=fired
        arriving=events.pop(tick,np.empty(0,np.int64))
        for offsets,targets,weight in adjacency:
            for source in arriving:np.add.at(current,targets[offsets[source]:offsets[source+1]],weight)
        v[fired]=0;until[fired]=tick+REF_TICKS
    return dict(v=v,ticks=np.asarray(ticks,np.int64),indices=np.asarray(indices,np.int64),synaptic_current=current)


def brian_run(backend,neurons,steps,degree,output,profile,*,prepared=None,state_dtype=None):
    import brian2 as b
    from gpu_baseline import canonical_result
    if backend in {'rust','cuda','metal','cpu-f32'}:
        import brian2_atlas
        opts=dict(engine='aot' if backend in {'rust','cpu-f32'} else backend,directory=output/'project',
                  runner=Path(__file__).resolve().parents[1]/'target/release/b2-runner')
        if backend in {'cuda','metal'}:opts.update(numeric_mode='float32',event_delivery='sparse')
        b.set_device('atlas',**opts)
    elif backend=='brian2cuda':
        import brian2cuda
        b.set_device('cuda_standalone',directory=str(output/'project'))
    else:
        import brian2genn
        b.set_device('genn',directory=str(output/'project'),use_GPU=True)
    b.prefs.core.default_float_dtype=np.float32 if state_dtype is None else state_dtype;b.defaultclock.dt=DT_MS*b.ms
    start=time.perf_counter();v,drive,projections=arrays(neurons,degree)
    pop=b.NeuronGroup(neurons,'dv/dt=(drive-v+I_syn)/(20*ms):1 (unless refractory)\ndI_syn/dt=-I_syn/(5*ms):1\ndrive:1 (constant)',
                      threshold='v>1',reset='v=0',refractory=REF_TICKS*DT_MS*b.ms,method='euler',name='population')
    pop.v=v;pop.drive=drive
    synapses=[]
    for label,(sources,targets,weight) in zip(('exc','inh'),projections):
        syn=b.Synapses(pop,pop,'w:1 (constant)',on_pre='I_syn_post+=w',delay=DELAY_TICKS*DT_MS*b.ms,name=label)
        syn.connect(i=sources.astype(int),j=targets.astype(int));syn.w=float(weight);synapses.append(syn)
    monitor=b.SpikeMonitor(pop);net=b.Network(pop,*synapses,monitor)
    construction=time.perf_counter()-start
    if prepared is not None:prepared.update(network=net,population=pop,monitor=monitor,device=b.get_device())
    start=time.perf_counter()
    if backend=='cpu-f32':
        from gpu_baseline import cpu_f32_control
        raw=cpu_f32_control(net,steps*DT_MS*b.ms,output,event_delivery='sparse',prepared=prepared);p=raw['populations'][0]
        return dict(v=p['states']['v'],ticks=p['spike_ticks'],indices=p['indices'],synaptic_current=p['states']['I_syn']),dict(
            model_construction_seconds=construction,build_run_seconds=time.perf_counter()-start,
            simulation_seconds=raw['run_seconds'],scope='one-worker compiled C++ GPU-f32 expression control')
    net.run(steps*DT_MS*b.ms,profile=profile);build_run=time.perf_counter()-start
    start=time.perf_counter();result=canonical_result(pop.v[:],monitor.t[:]/b.ms,monitor.i[:],DT_MS)
    result['synaptic_current']=np.asarray(pop.I_syn[:]).copy()
    timing=dict(model_construction_seconds=construction,build_run_seconds=build_run,
                final_array_read_seconds=time.perf_counter()-start,
                backend_reported_last_run_seconds=float(b.get_device()._last_run_time),
                last_run_scope='backend-specific; not directly comparable across implementations')
    if backend in {'rust','cuda','metal'}:
        import json
        timing['native_metadata']=json.loads((b.get_device().last_run_directory/'rust/summary.json').read_text())
    return result,timing


GENN_POLICIES=('legacy','brian-euler','projection-inputs','brian-euler-projection-inputs')


def genn_program(policy):
    """Explicit finite-precision adapter variants; the real-valued ODE is unchanged."""
    if policy not in GENN_POLICIES:raise ValueError('Unknown GeNN recurrent arithmetic policy')
    split=policy in {'projection-inputs','brian-euler-projection-inputs'}
    brian=policy in {'brian-euler','brian-euler-projection-inputs'}
    inputs='I_syn += exc_input; I_syn += inh_input;' if split else 'I_syn += Isyn;'
    if brian:
        # Brian's float32 Euler lowering computes coefficients in SI units and
        # evaluates (I_syn + drive) - v before multiplying. Keep every rounding
        # point explicit, instead of an algebraically equivalent ms expression.
        update='\nconst scalar dt_seconds = 0.0001f;\nconst scalar ms_seconds = 0.001f;\nconst scalar current_coefficient = (0.2f * dt_seconds) / ms_seconds;\nconst scalar voltage_coefficient = (0.05f * dt_seconds) / ms_seconds;\nif(ref==0) v = (voltage_coefficient * ((I_syn + drive) - v)) + v;\nI_syn = (current_coefficient * (-I_syn)) + I_syn;\n'
    else:update='if(ref==0) v += dt*(drive-v+I_syn)/20.0f; I_syn -= dt*I_syn/5.0f;'
    return dict(policy=policy,split_projection_inputs=split,
                sim_code=inputs+' if(ref>0) --ref; '+update,
                input_targets=['exc_input','inh_input'] if split else ['Isyn','Isyn'])


def genn_run(neurons,steps,degree,output,profile,policy='legacy',*,prepared=None):
    from pygenn import GeNNModel,create_neuron_model,init_weight_update,init_postsynaptic
    from gpu_baseline import canonical_result
    start=time.perf_counter();v,drive,projections=arrays(neurons,degree)
    program=genn_program(policy)
    model=GeNNModel('float','b2_recurrent_cuba',backend='cuda');model.dt=DT_MS;model.timing_enabled=profile
    neuron=create_neuron_model('CubaLIF',vars=[('v','scalar'),('I_syn','scalar'),('drive','scalar'),('ref','int')],
        sim_code=program['sim_code'],
        additional_input_vars=[('exc_input','scalar',0.0),('inh_input','scalar',0.0)] if program['split_projection_inputs'] else None,
        threshold_condition_code='ref==0 && v>1.0f',reset_code='v=0.0f; ref=20;')
    pop=model.add_neuron_population('population',neurons,neuron,{},dict(v=v,I_syn=0,drive=drive,ref=0))
    pop.spike_recording_enabled=True
    for label,(sources,targets,weight) in zip(('exc','inh'),projections):
        syn=model.add_synapse_population(label,'SPARSE',pop,pop,
            init_weight_update('StaticPulseConstantWeight',{'g':float(weight)}),init_postsynaptic('DeltaCurr'))
        if program['split_projection_inputs']:syn.post_target_var=label+'_input'
        syn.axonal_delay_steps=DELAY_TICKS
        syn.set_sparse_connections(sources,targets)
    construction=time.perf_counter()-start
    (output/'project').mkdir(parents=True,exist_ok=True)
    import pygenn.genn_model as gm
    old=gm.cpu_count;start=time.perf_counter()
    try:
        gm.cpu_count=lambda logical=True:2;model.build(path_to_model=str(output/'project'))
    finally:gm.cpu_count=old
    compile_seconds=time.perf_counter()-start
    start=time.perf_counter();model.load(num_recording_timesteps=steps);initialization=time.perf_counter()-start
    start=time.perf_counter()
    for _ in range(steps):model.step_time()
    model.pull_recording_buffers_from_device();pop.vars['v'].pull_from_device();pop.vars['I_syn'].pull_from_device()
    elapsed=time.perf_counter()-start
    times,indices=pop.spike_recording_data[0]
    result=canonical_result(pop.vars['v'].current_values,times,indices,DT_MS)
    result['synaptic_current']=pop.vars['I_syn'].current_values.copy().reshape(-1)
    timing=dict(model_construction_seconds=construction,compile_seconds=compile_seconds,
                initialization_seconds=initialization,simulation_and_readback_seconds=elapsed)
    if profile:timing.update(neuron_kernel_seconds=model.neuron_update_time,presynaptic_kernel_seconds=model.presynaptic_update_time)
    import hashlib,json
    timing['adapter_program']=program
    timing['adapter_program_sha256']=hashlib.sha256(json.dumps(program,sort_keys=True).encode()).hexdigest()
    if prepared is not None:prepared.update(model=model,population=pop,neuron_model=neuron,program=program)
    model.unload();return result,timing
