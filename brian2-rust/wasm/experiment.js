import {makeEquationDraft,parseEquations} from './equations.js';
import {models,modelFor,executionLimits} from './models.js';
export const defaults = models.adaptive_lif.defaults;
export const presets = Object.fromEntries(Object.entries(models.adaptive_lif.presets).map(([key,preset])=>[key,preset.config]));
export function validateConfig(value,{backend='wasm'}={}) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw Error('Configuration must be a JSON object.');
  const model=modelFor(value);
  if(!model)throw Error('Unknown model. Choose one of the available models.');
  for (const key of Object.keys(value)) if (!Object.hasOwn(model.defaults,key)) throw Error(`Unknown configuration field for ${model.name}: ${key}`);
  const limits=executionLimits(model,backend);
  const ranges = {neurons:[8,limits.scales.at(-1)],duration_ms:[50,2000],dt_ms:model.dt,seed:[0,4294967295],...Object.fromEntries(model.fields.map(f=>[f.key,[f.min,f.max]]))};
  for (const [key,[min,max]] of Object.entries(ranges)) {
    if (typeof value[key] !== 'number' || !Number.isFinite(value[key]) || value[key]<min || value[key]>max)
      throw Error(`${key} must be between ${min} and ${max}.`);
  }
  if (!Number.isInteger(value.neurons) || !Number.isInteger(value.seed)) throw Error('neurons and seed must be integers.');
  if (typeof value.synchronize !== 'boolean') throw Error('synchronize must be true or false.');
  for (const key of ['duration_ms',...(value.refractory_ms===undefined?[]:['refractory_ms'])]) {
    const ticks=value[key]/value.dt_ms;
    if (Math.abs(ticks-Math.round(ticks))>1e-7) throw Error(`${key} must be a multiple of dt_ms.`);
  }
  if(value.model==='flywire'){
    if(!model.scales.includes(value.neurons))throw Error('FlyWire supports real subgraphs of 240, 1024 or 4096 neurons.');
    if(value.synchronize)throw Error('FlyWire uses independent background input; synchronize must be false.');
    if(value.sensory_output!==0&&value.sensory_output!==1)throw Error('sensory_output must be 0 or 1.');
    if(value.stimulus_start_ms>=value.stimulus_end_ms||value.stimulus_end_ms>value.duration_ms)throw Error('Stimulus must start before it ends and fit within the simulation duration.');
  }
  if(model.editable)parseEquations(value.custom);
  const steps=Math.round(value.duration_ms/value.dt_ms),work=steps*value.neurons;
  if(work>limits.maxTicks || steps>40000)throw Error(`Reduce neuron count or duration: ${model.name} allows up to ${limits.maxTicks.toLocaleString()} neuron-ticks and 40,000 steps per run.`);
  return {...value,model:value.model??'adaptive_lif'};
}
export function bits(value) {
  const data=new DataView(new ArrayBuffer(8));data.setFloat64(0,value,false);
  return data.getBigUint64(0,false).toString(16).padStart(16,'0');
}
export function number(value) { const data=new DataView(new ArrayBuffer(8));data.setBigUint64(0,BigInt(`0x${value}`),false);return data.getFloat64(0,false); }
export function makeDraft(template, input, options={}) {
  const c=validateConfig(input,options);if(c.custom)return makeEquationDraft(template,c);if(c.model==='flywire')return makeFlywireDraft(template,c);
  const m=structuredClone(template),p=m.definition.populations[0],i=m.instance.populations[0];
  const n=c.neurons,steps=Math.round(c.duration_ms/c.dt_ms),dt=bits(c.dt_ms/1000);
  const record=Array.from({length:Math.min(12,n)},(_,k)=>Math.round(k*(n-1)/(Math.min(12,n)-1)));
  p.count=n;p.steps=steps;p.dt=dt;p.monitor.record=record;p.monitor.window_steps=steps;
  p.state_monitors[0].record=record;
  m.definition.clocks[0].dt=dt;m.instance.neuron_count=n;m.instance.rng_seed=c.seed;
  m.run.duration=bits(c.duration_ms/1000);m.run.clocks[0].steps=steps;
  let state=c.seed>>>0;
  const random=()=>{state=(Math.imul(state,1664525)+1013904223)>>>0;return state/4294967296;};
  const expected=c.model==='adaptive_lif'?['v','w']:c.model==='izhikevich'?['u','v']:['h','m','n','v'];
  if(Object.keys(i.initial_state).sort().join()!==expected.join())throw Error('Model template does not match the selected model.');
  const voltage=c.model==='adaptive_lif'?Array.from({length:n},()=>c.synchronize?0:random()*.95):Array.from({length:n},()=>-65+(c.synchronize?0:(random()-.5)*2));
  i.initial_state.v=voltage.map(bits);
  i.parameters.drive=Array.from({length:n},()=>bits(c.drive+(c.synchronize?0:(random()-.5)*c.spread)));
  if(c.model==='adaptive_lif') {
    i.initial_state.w=Array(n).fill(bits(0));
    i.parameters.tau=[bits(c.tau_ms/1000)];i.parameters.tau_adapt=[bits(c.tau_adapt_ms/1000)];i.parameters.adaptation=[bits(c.adaptation)];
  } else if(c.model==='izhikevich') {
    i.initial_state.u=voltage.map(v=>bits(c.b*v));
    for(const key of ['a','b','c','d'])i.parameters[key]=[bits(c[key])];
  } else {
    const exprel=x=>Math.abs(x)<1e-8?1+x/2:Math.expm1(x)/x;
    const rates={m:v=>[1/exprel(-(v+40)/10),4*Math.exp(-(v+65)/18)],h:v=>[.07*Math.exp(-(v+65)/20),1/(1+Math.exp(-(v+35)/10))],n:v=>[.1/exprel(-(v+55)/10),.125*Math.exp(-(v+65)/80)]};
    for(const [key,rate] of Object.entries(rates))i.initial_state[key]=voltage.map(v=>{const [a,b]=rate(v);return bits(a/(a+b));});
    for(const key of ['g_na','g_k','g_l'])i.parameters[key]=[bits(c[key])];
  }
  if(i.refractory) i.refractory={period:bits((c.refractory_ms??0)/1000),period_ticks:Math.round((c.refractory_ms??0)/c.dt_ms),
    initial_lastspike:Array(n).fill(bits(-10000)),initial_not_refractory:Array(n).fill(true)};
  // Hashes in this editable draft are intentionally stale. Only the explicit
  // WASM authoring API may replace them after validating the resulting model.
  return m;
}

function makeFlywireDraft(template,c){
  const m=structuredClone(template),p=m.definition.populations[0],i=m.instance.populations[0];
  if(p.count!==c.neurons||!i.initial_state.ge||!i.initial_state.gi||m.instance.synapses.length!==1)throw Error('Invalid FlyWire template.');
  const steps=Math.round(c.duration_ms/c.dt_ms);p.steps=steps;p.monitor.window_steps=steps;
  m.run.duration=bits(c.duration_ms/1000);m.run.clocks[0].steps=steps;m.instance.rng_seed=c.seed;
  let seed=c.seed>>>0;const random=()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed/4294967296;};
  i.initial_state.v=Array.from({length:c.neurons},()=>bits(-52+(random()-.5)*1.6));
  i.initial_state.ge=Array(c.neurons).fill(bits(c.background_rate_hz*.005*c.background_weight_mv/52));
  i.initial_state.gi=Array(c.neurons).fill(bits(0));
  const mapping={background_rate:c.background_rate_hz,background_gain:c.background_weight_mv/52,sensory_rate:c.sensory_rate_hz,sensory_gain:c.sensory_weight_mv/52,stim_start:c.stimulus_start_ms/1000,stim_end:c.stimulus_end_ms/1000};
  for(const [key,value] of Object.entries(mapping))i.parameters[key]=[bits(value)];
  i.initial_state.transmission=i.parameters.is_sensory.map(value=>bits(number(value)===1?c.sensory_output:1));
  const syn=m.instance.synapses[0];syn.parameters.recurrent_weight=[bits(c.recurrent_weight_mv/52)];syn.parameters.inhibitory_gain=[bits(c.inhibitory_gain)];
  return m;
}

// Typed decoding for the experiment's single-population, f64 result format.
// Reject incompatible input instead of silently interpreting another layout.
export function decodeExperiment(bytes, model) {
  const v=new DataView(bytes.buffer,bytes.byteOffset,bytes.byteLength);let at=0;
  const take=n=>{if (!Number.isSafeInteger(n)||n<0||at+n>v.byteLength) throw Error('Result data is truncated.');const p=at;at+=n;return p;};
  const u32=()=>v.getUint32(take(4),true);
  const u64=()=>{const n=v.getBigUint64(take(8),true);if(n>BigInt(Number.MAX_SAFE_INTEGER))throw Error('Result index exceeds the safe integer range.');return Number(n);};
  const f64=()=>v.getFloat64(take(8),true);
  const magic=text=>{const got=new TextDecoder().decode(bytes.subarray(at,at+8));take(8);if(got!==text)throw Error('Invalid result format.');};
  magic('B2DMP001');if(u32()!==3||u32()!==0x01020304)throw Error('Unsupported result version.');
  if(u64()!==1)throw Error('Playground charts require a single-population model.');
  const n=u64();if(u64()!==bytes.length)throw Error('Result length mismatch.');
  const p=model.definition.populations[0],steps=p.steps,record=p.monitor.record;
  const fields=Array.from({length:8},u64);
  if(fields.slice(0,5).join()!==[n,steps,record.length,p.monitor.variables.length,p.states.length].join()||n!==p.count)throw Error('Result dimensions do not match.');
  if(p.states.some(s=>s.dtype!=='f64'))throw Error('Playground traces require f64 states.');
  const [,,,,,spikes,last,flags]=fields,trace={};
  for(const name of p.monitor.variables) { const a=new Float64Array(steps*record.length);for(let j=0;j<a.length;j++){a[j]=f64();if(!Number.isFinite(a[j]))throw Error('Simulation became unstable. Reduce the timestep or input drive.');}trace[name]=a; }
  const ticks=new Float64Array(spikes),indices=new Uint32Array(spikes),counts=new Uint32Array(n);
  for(let j=0;j<spikes;j++) { ticks[j]=u64();indices[j]=u64();if(ticks[j]>=steps||indices[j]>=n)throw Error('Spike is out of range.');counts[indices[j]]++; }
  for(let j=0;j<n;j++)if(u64()!==counts[j])throw Error('Spike counts do not match.');
  take(last*8);const finalState={};for(const state of p.states){const a=new Float64Array(n);for(let j=0;j<n;j++){a[j]=f64();if(!Number.isFinite(a[j]))throw Error('Simulation produced non-finite final state. Reduce timestep or input drive.');}finalState[state.name]=a;}
  if(flags===1){take(n*8);take(n);}else if(flags!==0)throw Error('Invalid refractory data.');
  const synapseCount=u64();if(synapseCount!==model.definition.synapses.length)throw Error('Synapse result count mismatch.');
  let synapticEvents=0;
  for(let k=0;k<synapseCount;k++){
    const definition=model.definition.synapses[k],instance=model.instance.synapses[k],stateCount=u64(),edges=u64();
    if(stateCount!==definition.states.length||edges!==instance.source?.length||definition.states.some(s=>s.dtype!=='f64'))throw Error('Unsupported synapse result layout.');
    take(stateCount*edges*8);synapticEvents+=u64();
  }
  const duration=f64();magic('B2END001');if(at!==bytes.length)throw Error('Unexpected trailing result data.');
  return {n,steps,record,trace,ticks,indices,counts,dtMs:number(p.dt)*1000,durationMs:duration*1000,
    finalState,synapticEvents,meanHz:spikes/n/duration,active:counts.filter(x=>x>0).length,spikes};
}
const rateCache=new WeakMap();
export function firingRate(data,binMs) {
  if(!Number.isFinite(binMs)||binMs<=0)throw Error('Invalid bin width.');
  let cached=rateCache.get(data);if(!cached){cached=new Map();rateCache.set(data,cached);}if(cached.has(binMs))return cached.get(binMs);
  const count=Math.ceil(data.durationMs/binMs),rate=new Float64Array(count);
  for(const tick of data.ticks) {const bin=Math.floor(tick*data.dtMs/binMs);if(bin<count)rate[bin]++;}
  for(let k=0;k<count;k++)rate[k]/=data.n*(Math.min(binMs,data.durationMs-k*binMs)/1000);
  cached.set(binMs,rate);return rate;
}
