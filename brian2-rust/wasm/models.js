import {equationExamples} from './equation-library.js';
// Model-specific controls and display metadata. Built-in exports and browser-authored equations share the validated B2IR executor.
const shared = {neurons:160,duration_ms:600,dt_ms:0.1,seed:42,synchronize:false};
const field=(key,label,min,max,step=1,unit='')=>({key,label,min,max,step,unit});
const lif={...shared,model:'adaptive_lif',drive:1.65,spread:.55,tau_ms:20,refractory_ms:2,adaptation:.06,tau_adapt_ms:120};
const izh={...shared,model:'izhikevich',neurons:80,duration_ms:400,drive:10,spread:2,a:.02,b:.2,c:-65,d:8};
const hh={...shared,model:'hodgkin_huxley',neurons:48,duration_ms:100,dt_ms:.025,drive:10,spread:2,g_na:120,g_k:36,g_l:.3};
const fly={...shared,model:'flywire',neurons:240,duration_ms:300,seed:783,recurrent_weight_mv:.275,inhibitory_gain:4,background_rate_hz:300,background_weight_mv:3.5,sensory_rate_hz:80,sensory_weight_mv:40,stimulus_start_ms:80,stimulus_end_ms:220,sensory_output:1};
export const models={
  adaptive_lif:{gpuScales:[160, 1024, 4096, 16384, 65536, 98304],gpuMaxTicks:600000000,scales:[160, 1024, 4096, 16384, 32768],maxTicks:200000000,name:'Adaptive LIF',tag:'Threshold + reset',template:'editor-template.json',defaults:lif,
    description:'A compact baseline: integration, threshold crossing and spike-frequency adaptation.',
    equations:'τ · dv/dt = drive − v − w\nτw · dw/dt = −w',reset:'v > 1 → v = 0, w += adaptation',
    source:null,
    sourceLabel:'Brian2 neuron models',
    note:'Euler integration. Voltage and adaptation are normalized. Spikes use a hard threshold and reset.',
    voltageUnit:'normalized voltage',threshold:1,aux:['w'],auxUnit:'adaptation / normalized',
    traceNote:'v: normalized membrane voltage. w: adaptation. Dashed line: firing threshold at v = 1.',
    fields:[field('drive','Input drive',0,4,.05,'/ threshold'),field('spread','Drive heterogeneity',0,1.5,.05),field('adaptation','Adaptation strength',0,1,.01),field('tau_ms','Membrane time constant',5,100,1,'ms'),field('tau_adapt_ms','Adaptation time constant',20,500,1,'ms'),field('refractory_ms','Refractory period',0,20,.1,'ms')],dt:[.05,1],
    presets:{asynchronous:{label:'Asynchronous',hint:'Diverse spike timing',config:lif},synchronous:{label:'Synchronous',hint:'Shared initial state',config:{...lif,spread:0,adaptation:0,synchronize:true}},adapting:{label:'Adaptation',hint:'Rate slows over time',config:{...lif,drive:2.2,spread:.4,adaptation:.3,tau_adapt_ms:180}},silent:{label:'Subthreshold',hint:'Below firing threshold',config:{...lif,drive:.65,spread:.2,adaptation:0}}}},
  izhikevich:{gpuScales:[80, 512, 1024, 4096, 16384, 32768],gpuMaxTicks:150000000,scales:[80, 512, 1024, 4096, 8192],maxTicks:40000000,name:'Izhikevich',tag:'Spiking + bursting',template:'izhikevich-template.json',defaults:izh,
    description:'Two state variables reproduce distinct firing patterns, from regular spikes to clusters of bursts.',
    equations:'dv/dt = 0.04v² + 5v + 140 − u + I\ndu/dt = a(bv − u)',reset:'v ≥ 30 → v = c, u += d · time in ms',
    source:'https://brian2.readthedocs.io/en/stable/examples/frompapers.Izhikevich_2003.html',sourceLabel:'Izhikevich (2003) · Brian2 example',
    note:'Euler integration. I and u use model units; v uses the standard mV coordinate. The spike peak is terminated by reset.',
    voltageUnit:'voltage / mV',threshold:30,aux:['u'],auxUnit:'recovery u / model units',
    traceNote:'v: membrane voltage (mV). u: recovery variable (model units). Dashed line: spike cutoff at 30 mV.',
    fields:[field('drive','Input current',0,30,.5,'model units'),field('spread','Current heterogeneity',0,10,.5),field('a','Recovery rate · a',.01,.1,.005),field('b','Voltage coupling · b',.1,.3,.01),field('c','Reset voltage · c',-80,-45,1,'mV'),field('d','Recovery jump · d',0,10,.5)],dt:[.025,.25],
    presets:{regular:{label:'Regular spiking',hint:'Adapting spike train',config:izh},bursting:{label:'Bursting',hint:'Clusters of spikes',config:{...izh,c:-50,d:2}},fast:{label:'Fast spiking',hint:'Rapid recovery',config:{...izh,a:.1,d:2}},silent:{label:'Resting',hint:'No injected current',config:{...izh,drive:0,spread:0}}}},
  hodgkin_huxley:{gpuScales:[48, 256, 1024, 2048, 8192, 16384],gpuMaxTicks:70000000,scales:[48, 256, 1024, 2048, 4096],maxTicks:20000000,name:'Hodgkin–Huxley',tag:'Ion-channel dynamics',template:'hodgkin_huxley-template.json',defaults:hh,
    description:'Sodium, potassium and leak currents generate continuous action potentials without a voltage reset.',
    equations:'C · dV/dt = I − INa − IK − IL\nINa = gNa · m³h · (V − ENa)\nIK = gK · n⁴ · (V − EK)\ndx/dt = αx(V)(1 − x) − βx(V)x',reset:'x ∈ {m, h, n} · C = 1 µF/cm² · no voltage reset',
    source:'https://brian2.readthedocs.io/en/stable/examples/compartmental.hodgkin_huxley_1952.html',sourceLabel:'Hodgkin–Huxley (1952) · Brian2 equations',
    note:'Single-compartment squid-axon kinetics at 6.3 °C, exponential Euler. ENa = 50, EK = −77, EL = −54.387 mV. Spike detection rearms below −40 mV.',
    voltageUnit:'voltage / mV',threshold:0,aux:['m','h','n'],auxUnit:'channel gate / fraction',
    traceNote:'V: voltage (mV). m: Na activation; h: Na inactivation; n: K activation. Dashed line: 0 mV spike detection, with no reset.',
    fields:[field('drive','Injected current density',0,30,.5,'µA/cm²'),field('spread','Current heterogeneity',0,10,.5,'µA/cm²'),field('g_na','Sodium conductance',60,180,5,'mS/cm²'),field('g_k','Potassium conductance',18,60,1,'mS/cm²'),field('g_l','Leak conductance',.1,1,.05,'mS/cm²')],dt:[.005,.05],
    presets:{tonic:{label:'Tonic firing',hint:'Repeated action potentials',config:hh},strong:{label:'Strong drive',hint:'Higher injected current',config:{...hh,drive:20}},synchronous:{label:'Synchronous',hint:'Identical cells and input',config:{...hh,spread:0,synchronize:true}},silent:{label:'Resting',hint:'No injected current',config:{...hh,drive:0,spread:0}}}},
  flywire:{scales:[240, 1024, 4096],maxTicks:40000000,name:'FlyWire · DM1',tag:'Real connected circuit',template:'flywire-template.json',defaults:fly,discreteScale:true,
    description:'A real olfactory subgraph with 240, 1,024 or 4,096 neurons: stimulate DM1 sensory cells and follow activity through signed, delayed synapses.',
    equations:'τ · dV/dt = −(V + 52) − geV − gi(V + 70)\nτs · dge/dt = −ge; τs · dgi/dt = −gi',reset:'V > −45 → V = −52 mV · τ = 20 ms · τs = 5 ms',
    source:'https://zenodo.org/records/10676866',sourceLabel:'FlyWire v783 · CC BY 4.0 · Data source',
    note:'Real induced connectivity, reduced conductance LIF. 2.2 ms refractory; 1.8 ms synaptic delay. Reference input amplitudes convert to conductance by dividing by 52. Omitted full-brain inputs are absent; stimulation is synthetic.',
    voltageUnit:'voltage / mV',threshold:-45,aux:['ge','gi'],auxUnit:'conductance / relative to leak',
    traceNote:'V: membrane voltage. ge / gi: excitatory / inhibitory conductance, relative to leak. Each group has two recorded probes. The shaded interval marks the configured stimulus window; baseline presets apply no stimulus.',
    fields:[field('sensory_rate_hz','DM1 stimulus rate',0,160,5,'Hz'),field('sensory_weight_mv','Stimulus amplitude',0,80,1,'ref. mV'),field('recurrent_weight_mv','Synaptic contact strength',0,.6,.025,'ref. mV'),field('inhibitory_gain','Inhibitory multiplier',0,8,.25,'×'),field('background_rate_hz','Background input rate',0,500,10,'Hz'),field('sensory_output','Sensory output · 0 off / 1 on',0,1,1),{...field('background_weight_mv','Background amplitude',0,8,.25,'ref. mV'),advanced:true},{...field('stimulus_start_ms','Stimulus starts',0,1500,10,'ms'),advanced:true},{...field('stimulus_end_ms','Stimulus ends',0,2000,10,'ms'),advanced:true}],dt:[.1,.1],
    presets:{odor:{label:'Stimulate DM1',hint:'Input + intact circuit',config:fly},rest:{label:'Background only',hint:'Matched unstimulated run',config:{...fly,sensory_weight_mv:0}},cut:{label:'Block sensory output',hint:'Input + transmission cut',config:{...fly,sensory_output:0}},cut_rest:{label:'Cut baseline',hint:'No input + transmission cut',config:{...fly,sensory_weight_mv:0,sensory_output:0}}}},

};
export const executionLimits=(meta,backend='wasm')=>({scales:backend==='webgpu'&&meta.gpuScales?meta.gpuScales:meta.scales,maxTicks:backend==='webgpu'&&meta.gpuMaxTicks?meta.gpuMaxTicks:meta.maxTicks});
export const modelFor=config=>{const key=config.model??'adaptive_lif';return Object.hasOwn(models,key)?models[key]:undefined;};

const editableDefaults=(model,example)=>({...shared,neurons:160,model,...Object.fromEntries(Object.entries(example).filter(([k])=>k!=='label'))});
const equationMeta={template:'editor-template.json',editable:true,scales:[160,1024,4096,8192],maxTicks:100000000,gpuScales:[160,1024,4096,8192,32768],gpuMaxTicks:400000000,dt:[.01,1],fields:[field('drive','Input drive',-10,40,.01,'model units'),field('spread','Drive heterogeneity',0,10,.01)],aux:['w'],auxUnit:'state / model units',voltageUnit:'v / model units',voltageRange:[-1,1],threshold:null,
  tag:'Editable equations',description:'Edit equations, parameters, initial values, threshold and reset. Compile and simulate entirely in your browser.',
  equations:'Your equations → validated B2IR → WASM or WebGPU',reset:'Explicit Euler · 1–4 states · independent cells',
  note:'Brian2-style equation subset, not Python. States use dimensionless numerical coordinates; ms and second carry time units. Parameters and initial values are numeric. Input drive varies across cells by the configured spread. Only explicitly marked states freeze during the fixed refractory period.',
  traceNote:'All declared states are recorded for up to 12 probes. Spikes are threshold events; choose a reset to avoid repeated events above threshold.',source:null};
const adex=editableDefaults('adex',equationExamples.adex),qif=editableDefaults('quadratic_if',equationExamples.qif),custom=editableDefaults('custom',equationExamples.lif);
models.adex={...equationMeta,name:'AdEx',defaults:adex,voltageRange:[-70,-25],threshold:-30,source:'https://brian2.readthedocs.io/en/stable/examples/frompapers.Naud_et_al_2008_adex_firing_patterns.html',sourceLabel:'Naud et al. (2008) · Brian2 example',description:'Adaptive exponential integrate-and-fire. Numerical voltage uses mV; current and adaptation are divided by leak conductance. All equations are editable.',presets:{tonic:{label:'Tonic firing',hint:'Exponential spike onset',config:adex},adapting:{label:'Strong adaptation',hint:'Larger recovery jump',config:{...adex,custom:{...adex.custom,parameters:{...adex.custom.parameters,jump:5}}}},silent:{label:'Subthreshold',hint:'No injected drive',config:{...adex,drive:0,spread:0}}}};
models.quadratic_if={...equationMeta,name:'Quadratic IF',source:'https://neuronaldynamics.epfl.ch/online/Ch5.S3.html',sourceLabel:'Neuronal Dynamics · Quadratic integrate-and-fire',defaults:qif,aux:[],voltageRange:[-2.2,2.2],threshold:2,description:'A nonlinear integrate-and-fire model with quadratic voltage dynamics. Finite cutoff and reset at ±2 in normalized coordinates. All equations are editable.',presets:{tonic:{label:'Tonic firing',hint:'Quadratic integration',config:qif},strong:{label:'Strong drive',hint:'Faster repeated firing',config:{...qif,drive:3}},silent:{label:'Resting',hint:'Stable negative equilibrium',config:{...qif,drive:-1,spread:0}}}};
models.custom={...equationMeta,name:'Equation Lab',defaults:custom,description:'Build your own model in the browser. Start with an example, then edit the actual dynamics and press Run.',presets:Object.fromEntries(Object.entries(equationExamples).map(([id,e])=>[id,{label:e.label,hint:'Editable starting point',config:editableDefaults('custom',e)}]))};
export function presentationFor(data){if(data.presentation)return data.presentation;const meta=modelFor(data);if(!data.config?.custom)return meta;const variables=Object.keys(data.trace??data.config.custom.initial);return {...meta,aux:variables.filter(v=>v!=='v'),threshold:null,voltageRange:[data.config.custom.initial.v-1,data.config.custom.initial.v+1]};}
