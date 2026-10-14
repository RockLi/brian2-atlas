"""HH ionic dynamics, independent f64 oracle and matched native GPU adapters.

Rates follow the repository's COBAHH example, with deterministic nonnegative
initial conductances and a constant injected current. There is no recurrent
connectivity. All six states and start-of-tick traces are compared in SI units.
"""
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

DT=.0001
FIELDS=('v','m','n','h','ge','gi')
CM=2e-10
GL=1e-8
GNA=2e-5
GK=6e-6
EL=-.060
ENA=.050
EK=-.090
VT=-.063
EE=0.
EI=-.080
REF_TICKS=30

EQUATIONS='''
dv/dt=(gl*(El-v)+ge*(Ee-v)+gi*(Ei-v)-g_na*m**3*h*(v-ENa)-g_kd*n**4*(v-EK)+I_ext)/Cm : volt
dm/dt=alpha_m*(1-m)-beta_m*m : 1
dn/dt=alpha_n*(1-n)-beta_n*n : 1
dh/dt=alpha_h*(1-h)-beta_h*h : 1
dge/dt=-ge/(5*ms) : siemens
dgi/dt=-gi/(10*ms) : siemens
I_ext : amp (constant)
alpha_m=1.28/exprel((13*mV-v+VT)/(4*mV))/ms : Hz
beta_m=1.4/exprel((v-VT-40*mV)/(5*mV))/ms : Hz
alpha_h=0.128*exp((17*mV-v+VT)/(18*mV))/ms : Hz
beta_h=4/(1+exp((40*mV-v+VT)/(5*mV)))/ms : Hz
alpha_n=0.16/exprel((15*mV-v+VT)/(5*mV))/ms : Hz
beta_n=0.5*exp((10*mV-v+VT)/(40*mV))/ms : Hz
'''


def configuration(neurons,steps):
    return dict(case='hh-ionic-v0',neurons=neurons,steps=steps,dt_seconds=DT,
        equations=EQUATIONS,constants=dict(Cm=CM,gl=GL,g_na=GNA,g_kd=GK,El=EL,ENa=ENA,EK=EK,VT=VT,Ee=EE,Ei=EI),
        method='exponential_euler',threshold='v>-20*mV',refractory_ticks=REF_TICKS,
        refractory_scope='spike suppression only; all ODEs continue',reset='none',
        initialization='deterministic f32 SI arrays',gpu_dtype='float32',reference_dtype='float64',
        recording=dict(final=list(FIELDS),trace=list(FIELDS),indices=list(range(min(16,neurons))),when='start',spikes='all ticks and indices'),
        acceptance=dict(state_rtol=3e-4,atol=dict(v=5e-6,m=2e-5,n=2e-5,h=2e-5,ge=1e-12,gi=1e-12),spikes='exact ticks and indices'))


def arrays(neurons):
    i=np.arange(neurons)
    return dict(v=np.asarray((-.065+(i%11-5)*.0005),np.float32),
        m=np.zeros(neurons,np.float32),n=np.zeros(neurons,np.float32),h=np.zeros(neurons,np.float32),
        ge=np.asarray((i%7+1)*8e-9,np.float32),gi=np.asarray((i%5)*20e-9,np.float32),
        I_ext=np.asarray((.2+(i%9)*.05)*1e-9,np.float32))


def exprel(x):
    x=np.asarray(x,np.float64)
    return np.divide(np.expm1(x),x,out=np.ones_like(x),where=x!=0)


def rates(v):
    return dict(m=(1280/exprel((.013-v+VT)/.004),1400/exprel((v-VT-.040)/.005)),
        h=(128*np.exp((.017-v+VT)/.018),4000/(1+np.exp((.040-v+VT)/.005))),
        n=(160/exprel((.015-v+VT)/.005),500*np.exp((.010-v+VT)/.040)))


def oracle(neurons,steps):
    """Independent analytic exponential Euler; every update reads old state."""
    state={k:v.astype(np.float64) for k,v in arrays(neurons).items()}
    traces={k:[] for k in FIELDS};ticks=[];indices=[];until=np.zeros(neurons,np.int64)
    for tick in range(steps):
        for name in FIELDS:traces[name].append(state[name][:min(16,neurons)].copy())
        v,m,n,h,ge,gi=(state[k] for k in FIELDS)
        conductance_na=GNA*m**3*h;conductance_k=GK*n**4
        conductance=GL+ge+gi+conductance_na+conductance_k
        equilibrium=(GL*EL+ge*EE+gi*EI+conductance_na*ENA+conductance_k*EK+state['I_ext'])/conductance
        next_v=v+(equilibrium-v)*(-np.expm1(-DT*conductance/CM))
        for name,(alpha,beta) in rates(v).items():
            old=state[name];state[name]=old+(alpha/(alpha+beta)-old)*(-np.expm1(-DT*(alpha+beta)))
        state.update(v=next_v,ge=ge*np.exp(-DT/.005),gi=gi*np.exp(-DT/.010))
        fired=np.flatnonzero((tick>=until)&(next_v>-.020))
        ticks.extend([tick]*len(fired));indices.extend(fired);until[fired]=tick+REF_TICKS
    return {**{k:state[k] for k in FIELDS},**{'trace_'+k:np.asarray(v) for k,v in traces.items()},
            'ticks':np.asarray(ticks,np.int64),'indices':np.asarray(indices,np.int64)}


def checks(result,expected,config):
    check={};details={}
    for name,wanted in expected.items():
        actual=result[name]
        if name in ('ticks','indices'):check[name]=bool(np.array_equal(actual,wanted));continue
        field=name.removeprefix('trace_');atol=config['acceptance']['atol'][field];rtol=config['acceptance']['state_rtol']
        check[name]=bool(actual.shape==wanted.shape and np.allclose(actual,wanted,atol=atol,rtol=rtol))
        details[name]=dict(atol=atol,rtol=rtol,passed=check[name])
    return check,details


def brian_run(backend,neurons,steps,output,profile):
    import brian2 as b
    from gpu_baseline import canonical_result
    adjustments=[]
    if backend in {'rust','cuda','metal','cpu-f32'}:
        import brian2_rust
        options=dict(engine='aot' if backend in {'rust','cpu-f32'} else backend,directory=output/'project',
                     runner=Path(__file__).resolve().parents[1]/'target/release/b2-runner')
        if backend in {'metal','cuda'}:options['numeric_mode']='float32'
        b.set_device('atlas',**options)
    elif backend=='brian2cuda':
        import brian2cuda
        b.set_device('cuda_standalone',directory=str(output/'project'))
    else:
        import brian2genn
        from brian2.core.functions import DEFAULT_FUNCTIONS
        from brian2.codegen.generators.cpp_generator import CPPCodeGenerator
        from brian2genn.genn_generator import GeNNCodeGenerator
        # Brian2GeNN 1.7.0 omits exprel. Register Brian's own C++ helper via
        # the public API, preserving its double arithmetic and singularity
        # guards; only add GeNN's host/device annotation. Report this bridge.
        try:DEFAULT_FUNCTIONS['exprel'].implementations[GeNNCodeGenerator]
        except KeyError:
            import hashlib
            implementation=DEFAULT_FUNCTIONS['exprel'].implementations[CPPCodeGenerator]
            code=implementation.get_code(None)
            if not isinstance(code,str) or 'static inline double' not in code:
                raise RuntimeError('Unexpected pinned Brian exprel implementation')
            code=code.replace('static inline double','SUPPORT_CODE_FUNC double',1)
            DEFAULT_FUNCTIONS['exprel'].implementations.add_implementation(GeNNCodeGenerator,code=code,name=implementation.name)
            adjustments.append(dict(function='exprel',bridge='Brian C++ helper with GeNN host/device annotation',
                arithmetic='double helper; float32 state storage',source=code,sha256=hashlib.sha256(code.encode()).hexdigest()))
        # GeNN 4 emits exp(integer constants), selecting libstdc++'s host-
        # only integer overload under CUDA 12. Preserve native float/double
        # overloads and explicitly promote only integer arguments to double.
        import hashlib
        exp_code = """
SUPPORT_CODE_FUNC float b2_hh_exp(float x) { return exp(x); }
SUPPORT_CODE_FUNC double b2_hh_exp(double x) { return exp(x); }
SUPPORT_CODE_FUNC double b2_hh_exp(int x) { return exp(double(x)); }
"""
        DEFAULT_FUNCTIONS['exp'].implementations.add_implementation(GeNNCodeGenerator,code=exp_code,name='b2_hh_exp')
        adjustments.append(dict(function='exp',bridge='device-callable integer overload preserving native float/double overloads',
            arithmetic='int promoted to double; float/double overload precision unchanged',source=exp_code,sha256=hashlib.sha256(exp_code.encode()).hexdigest()))
        b.set_device('genn',directory=str(output/'project'),use_GPU=True)
    b.prefs.core.default_float_dtype=np.float64 if backend=='rust' else np.float32
    b.defaultclock.dt=DT*b.second
    namespace=dict(Cm=CM*b.farad,gl=GL*b.siemens,g_na=GNA*b.siemens,g_kd=GK*b.siemens,
                   El=EL*b.volt,ENa=ENA*b.volt,EK=EK*b.volt,VT=VT*b.volt,Ee=EE*b.volt,Ei=EI*b.volt)
    start=time.perf_counter()
    pop=b.NeuronGroup(neurons,EQUATIONS,threshold='v>-20*mV',refractory=REF_TICKS*DT*b.second,
                      method='exponential_euler',namespace=namespace,name='population')
    for name,value in arrays(neurons).items():pop.variables[name].set_value(value)
    monitor=b.StateMonitor(pop,list(FIELDS),record=list(range(min(16,neurons))),when='start')
    spikes=b.SpikeMonitor(pop);net=b.Network(pop,monitor,spikes)
    construction=time.perf_counter()-start;start=time.perf_counter()
    if backend=='cpu-f32':
        from brian2_rust.export import lower_network
        from brian2_rust.cuda import CudaExecutor,build_cuda_plan
        model=lower_network(net,steps*DT*b.second);control=SimpleNamespace(model=model,directory=output/'control',plan=build_cuda_plan(model,numeric_mode='float32'))
        control.directory.mkdir()
        raw=CudaExecutor._cpu_control(control,512*1024**2,1);p=raw['populations'][0]
        result={**{k:p['states'][k] for k in FIELDS},**{'trace_'+k:p['trace'][k] for k in FIELDS},'ticks':p['spike_ticks'],'indices':p['indices']}
        return result,dict(model_construction_seconds=construction,build_run_seconds=time.perf_counter()-start,simulation_seconds=raw['run_seconds'],scope='one-worker compiled C++ GPU-f32 expression control')
    net.run(steps*DT*b.second,profile=profile);elapsed=time.perf_counter()-start;start=time.perf_counter()
    result=canonical_result(pop.variables['v'].get_value(),spikes.t[:]/b.ms,spikes.i[:],DT*1000)
    for name in FIELDS:
        result[name]=np.asarray(pop.variables[name].get_value()).copy()
        result['trace_'+name]=np.asarray(getattr(monitor,name+'_')).T.copy()
    timing=dict(model_construction_seconds=construction,build_run_seconds=elapsed,final_array_read_seconds=time.perf_counter()-start,
                backend_reported_last_run_seconds=float(b.get_device()._last_run_time),last_run_scope='backend-specific; not directly comparable across implementations')
    if backend in {'rust','cuda','metal'}:
        import json
        timing['native_metadata']=json.loads((b.get_device().last_run_directory/'rust/summary.json').read_text())
    timing['adapter_adjustments']=adjustments
    return result,timing


def genn_run(neurons,steps,output,profile):
    from pygenn import GeNNModel,create_neuron_model
    from gpu_baseline import canonical_result
    # GeNN time is labelled ms. Keep its clock in ms and convert dt in the
    # equations; voltages, conductances and currents remain stored in SI.
    model=GeNNModel('float','b2_hh_ionic',backend='cuda');model.dt=DT*1000;model.timing_enabled=profile
    count=min(16,neurons)
    trace_code=''.join(f'trace_{k}[record_step*{count}+id]={k};' for k in FIELDS)
    sim=f'if(id<{count}) {{ {trace_code} }} ++record_step; if(ref>0) --ref;'+'''
        const scalar ds=dt*0.001;
        const scalar xm=(0.013-v-0.063)/0.004;
        const scalar xb=(v+0.063-0.040)/0.005;
        const scalar xn=(0.015-v-0.063)/0.005;
        const scalar rm=fabs(xm)<1e-6 ? 1+xm*0.5+xm*xm/6 : expm1(xm)/xm;
        const scalar rb=fabs(xb)<1e-6 ? 1+xb*0.5+xb*xb/6 : expm1(xb)/xb;
        const scalar rn=fabs(xn)<1e-6 ? 1+xn*0.5+xn*xn/6 : expm1(xn)/xn;
        const scalar am=1280/rm; const scalar bm=1400/rb;
        const scalar an=160/rn; const scalar bn=500*exp((0.010-v-0.063)/0.040);
        const scalar ah=128*exp((0.017-v-0.063)/0.018);
        const scalar bh=4000/(1+exp((0.040-v-0.063)/0.005));
        const scalar gna=0.00002*m*m*m*h; const scalar gk=0.000006*n*n*n*n;
        const scalar total=0.00000001+ge+gi+gna+gk;
        const scalar veq=(-0.0000000006-0.080*gi+0.050*gna-0.090*gk+I_ext)/total;
        const scalar next_v=v+(veq-v)*(-expm1(-ds*total/0.0000000002));
        m+=(am/(am+bm)-m)*(-expm1(-ds*(am+bm)));
        n+=(an/(an+bn)-n)*(-expm1(-ds*(an+bn)));
        h+=(ah/(ah+bh)-h)*(-expm1(-ds*(ah+bh)));
        v=next_v; ge*=exp(-ds/0.005); gi*=exp(-ds/0.010);
    '''
    start=time.perf_counter()
    neuron=create_neuron_model('HhIonic',vars=[(k,'scalar') for k in (*FIELDS,'I_ext')]+[('ref','int'),('record_step','unsigned int')],
        extra_global_params=[('trace_'+k,'scalar*') for k in FIELDS],sim_code=sim,
        threshold_condition_code='ref==0 && v>-0.020',reset_code='ref=30;')
    pop=model.add_neuron_population('population',neurons,neuron,{},dict(**arrays(neurons),ref=0,record_step=0))
    pop.spike_recording_enabled=True
    for name in FIELDS:pop.extra_global_params['trace_'+name].set_init_values(np.zeros(steps*count,np.float32))
    construction=time.perf_counter()-start;(output/'project').mkdir(parents=True,exist_ok=True)
    import pygenn.genn_model as gm
    previous=gm.cpu_count;start=time.perf_counter()
    try:
        gm.cpu_count=lambda logical=True:2;model.build(path_to_model=str(output/'project'))
    finally:gm.cpu_count=previous
    compile_seconds=time.perf_counter()-start;start=time.perf_counter();model.load(num_recording_timesteps=steps);initialization=time.perf_counter()-start
    start=time.perf_counter()
    for _ in range(steps):model.step_time()
    model.pull_recording_buffers_from_device()
    for k in FIELDS:pop.vars[k].pull_from_device();pop.extra_global_params['trace_'+k].pull_from_device()
    elapsed=time.perf_counter()-start
    times,indices=pop.spike_recording_data[0]
    result=canonical_result(pop.vars['v'].current_values,times,indices,DT*1000)
    for k in FIELDS:
        result[k]=pop.vars[k].current_values.copy().reshape(-1)
        result['trace_'+k]=pop.extra_global_params['trace_'+k].view.copy().reshape(steps,count)
    timing=dict(model_construction_seconds=construction,compile_seconds=compile_seconds,initialization_seconds=initialization,
                simulation_and_readback_seconds=elapsed,trace_scope='device writes at neuron-entry; one final transfer')
    if profile:timing['neuron_kernel_seconds']=model.neuron_update_time
    model.unload();return result,timing
