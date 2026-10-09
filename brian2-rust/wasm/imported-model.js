// Read metadata without ever reserializing the original model/plan JSON strings.
import {number} from './experiment.js';
export const MAX_IMPORT_BYTES=32*1024*1024;
const widths={bool:1,f32:4,f64:8,i32:4,u32:4,i64:8,u64:8};
const check=(condition,message)=>{if(!condition)throw Error(message);};
const bounded=(v,max,label,min=0)=>{check(Number.isSafeInteger(v)&&v>=min&&v<=max,`${label} must be an integer in [${min}, ${max}].`);return v;};
function symbols(p){return new Map([...p.states,...p.parameters,...(p.linked_variables??[])].map(s=>[s.name,s]));}
function edges(instance){const kind=instance.topology?.kind??'explicit';check(kind!=='binary_csr','Browser imports cannot read binary CSR files. Export inline or procedural connectivity.');return bounded(kind==='explicit'?instance.source?.length:instance.topology?.edge_count,1000000,'Synapse edge count');}
export function inspectBundle(bundle){
 check(bundle?.schema==='b2-wasm-bundle-v0','Choose a Brian2 WASM model bundle (.browser.json), not a raw .wasm file or a design JSON.');
 check(typeof bundle.model_json==='string'&&typeof bundle.plan_json==='string'&&/^[0-9a-f]{64}$/.test(bundle.plan_sha256),'Bundle is missing its model, execution plan or SHA-256.');
 check(new TextEncoder().encode(bundle.model_json).length<=24*1024*1024&&new TextEncoder().encode(bundle.plan_json).length<=4*1024*1024,'Model or plan exceeds the import size budget.');
 let model;try{model=JSON.parse(bundle.model_json);JSON.parse(bundle.plan_json);}catch{throw Error('Model or execution plan contains invalid JSON.');}
 try{
  const d=model.definition,instance=model.instance,populations=d.populations;
  bounded(populations.length,16,'Population count',1);bounded(d.clocks.length,16,'Clock count',1);
  check(instance.populations.length===populations.length&&d.synapses.length===instance.synapses.length,'Definition and instance dimensions differ.');
  bounded(d.synapses.length,64,'Synapse object count');check(d.numeric_profile==='reference-f64','The import runner requires reference-f64 plans.');
  check(d.functions.every(f=>f.body!=null),'Native-only functions cannot execute in a browser. Export portable function bodies.');
  const runStart=number(model.run.start),runDuration=number(model.run.duration);check(Number.isFinite(runStart)&&runStart>=0&&Number.isFinite(runDuration)&&runDuration>0&&runDuration<=60,'Run duration must be positive and at most 60 seconds.');
  let neurons=0,ticks=0,traceValues=0,stateBytes=0,edgeCount=0;
  const info=populations.map((p,k)=>{
   const count=bounded(p.count,100000,'Population neuron count',1),steps=bounded(p.steps,40000,'Population steps',1),dt=number(p.dt);
   check(Number.isFinite(dt)&&dt>0&&steps*dt<=60,'Each population must have a positive timestep and at most 60 seconds of simulated duration.');check(Number.isSafeInteger(Math.round(runStart/dt)+steps),'Clock ticks exceed the exact browser plotting range.');neurons+=count;ticks+=count*steps;
   const monitor=p.monitor,window=bounded(monitor.window_steps,steps,'Recording window',1);bounded(monitor.record.length,64,'Recorded probes');bounded(monitor.variables.length,32,'Recorded variables');bounded(p.states.length,64,'State variables');
   check(!p.event_monitors.length,'Custom EventMonitor imports are not supported by this viewer yet. Remove them before exporting; SpikeMonitor is supported.');
   const names=symbols(p);check(new Set(monitor.record).size===monitor.record.length,'Duplicate recorded probe.');for(const i of monitor.record)bounded(i,count-1,'Recorded index');
   for(const name of monitor.variables){const s=names.get(name);check(s&&Object.hasOwn(widths,s.dtype),'Unknown trace type.');traceValues+=window*monitor.record.length;stateBytes+=window*monitor.record.length*widths[s.dtype];}
   for(const s of p.states){check(Object.hasOwn(widths,s.dtype),'Unknown state type.');stateBytes+=count*widths[s.dtype];}
   stateBytes+=count*25;
   return {name:p.name,index:k,count,steps,dtMs:dt*1000,windowSteps:window,record:monitor.record,variables:monitor.variables,spikesRecorded:p.spike_monitor!=null};
  });
  bounded(neurons,100000,'Total neuron count',1);check(neurons===instance.neuron_count,'Total neuron count differs.');check(ticks<=200000000,'Import exceeds 200 million neuron-ticks. Reduce population or duration.');
  check(traceValues<=2000000,'Import exceeds 2 million recorded state values. Record fewer probes or a shorter window.');
  for(let k=0;k<d.synapses.length;k++){const count=edges(instance.synapses[k]);edgeCount+=count;for(const s of d.synapses[k].states){check(Object.hasOwn(widths,s.dtype),'Unknown synapse state type.');stateBytes+=count*widths[s.dtype];}}
  check(edgeCount<=1000000,'Import exceeds one million synaptic edges.');check(stateBytes<=64*1024*1024,'Recorded state exceeds the 64 MiB import budget.');
  return {model,info,neurons,edgeCount,runStartMs:runStart*1000,runDurationMs:runDuration*1000};
 }catch(error){if(error instanceof TypeError)throw Error('Incomplete or unsupported model metadata. Export with the current Brian2 Rust tools.');throw error;}
}
export function parseBundleText(text){check(typeof text==='string'&&new TextEncoder().encode(text).length<=MAX_IMPORT_BYTES,'Model bundles must be at most 32 MiB.');let bundle;try{bundle=JSON.parse(text);}catch{throw Error('Invalid bundle JSON. Choose a .browser.json export.');}return {bundle,...inspectBundle(bundle)};}

// Decode each population on its own clock and recording window. Integer values
// outside the exact JavaScript plotting range are omitted, never rounded silently.
export function decodeImported(bytes,model){
 const view=new DataView(bytes.buffer,bytes.byteOffset,bytes.byteLength);let at=0;
 const take=n=>{check(Number.isSafeInteger(n)&&n>=0&&at+n<=bytes.byteLength,'Truncated result data.');const pos=at;at+=n;return pos;};
 const u32=()=>view.getUint32(take(4),true),u64=()=>{const n=view.getBigUint64(take(8),true);check(n<=BigInt(Number.MAX_SAFE_INTEGER),'Result index exceeds the plotting range.');return Number(n);};
 const f64=()=>view.getFloat64(take(8),true);
 const magic=value=>{check(new TextDecoder().decode(bytes.subarray(at,at+value.length))===value,'Invalid result marker.');take(value.length);};
 function values(dtype,count,keep=true){const width=widths[dtype];check(width,'Unsupported result dtype.');check(Number.isSafeInteger(count)&&count>=0&&count*width<=bytes.length-at,'Result array exceeds its buffer.');const out=keep?new Float64Array(count):null;let exact=true;
  for(let i=0;i<count;i++){const pos=take(width);let value;
   if(dtype==='i64'||dtype==='u64'){const integer=dtype==='i64'?view.getBigInt64(pos,true):view.getBigUint64(pos,true);if(integer>BigInt(Number.MAX_SAFE_INTEGER)||integer<BigInt(Number.MIN_SAFE_INTEGER))exact=false;value=Number(integer);}
   else if(dtype==='f64')value=view.getFloat64(pos,true);else if(dtype==='f32')value=view.getFloat32(pos,true);else if(dtype==='i32')value=view.getInt32(pos,true);else if(dtype==='u32')value=view.getUint32(pos,true);else{value=view.getUint8(pos);check(value<=1,'Invalid boolean state.');}
   check(Number.isFinite(value),'Simulation produced non-finite state.');if(out)out[i]=value;
  }return exact?out:null;
 }
 magic('B2DMP001');check(u32()===3&&u32()===0x01020304,'Unsupported result version.');
 const pops=model.definition.populations;check(u64()===pops.length&&u64()===model.instance.neuron_count&&u64()===bytes.length,'Result header differs from the model.');
 const populations=[];let totalSpikes=0;
 for(let k=0;k<pops.length;k++){
  const p=pops[k],n=p.count,monitor=p.monitor,steps=monitor.window_steps,dtMs=number(p.dt)*1000,startTick=Math.round(number(model.run.start)/number(p.dt))+p.steps-steps;
  const fields=Array.from({length:8},u64);check(fields.slice(0,5).join()===[n,p.steps,monitor.record.length,monitor.variables.length,p.states.length].join(),'Population result dimensions differ.');
  const [,,,,,spikes,last,flags]=fields;totalSpikes+=spikes;check(totalSpikes<=4000000,'Recorded spikes exceed the four-million-event budget.');
  const names=symbols(p),trace={},omitted=[];for(const name of monitor.variables){const result=values(names.get(name).dtype,steps*monitor.record.length);if(result)trace[name]=result;else omitted.push(name);}
  const ticks=new Float64Array(spikes),indices=new Uint32Array(spikes),counts=new Uint32Array(n);let previousTick=-1,previousIndex=-1;
  for(let j=0;j<spikes;j++){const tick=u64()-startTick,index=u64();check(tick>=0&&tick<steps&&index<n,'Spike outside its recording window.');check(tick>previousTick||(tick===previousTick&&index>previousIndex),'Spikes are not strictly ordered.');ticks[j]=tick;indices[j]=index;counts[index]++;previousTick=tick;previousIndex=index;}
  for(let j=0;j<n;j++)check(u64()===counts[j],'Spike counts differ from events.');check(last<=n,'Invalid last-spike count.');for(let j=0;j<last;j++)check(u64()<n,'Invalid last-spike index.');
  for(const state of p.states)values(state.dtype,n,false);
  check(flags===(model.instance.populations[k].refractory?1:0),'Refractory flag differs.');if(flags){values('f64',n,false);values('bool',n,false);}
  const durationMs=steps*dtMs;
  populations.push({name:p.name,population:k,n,steps,record:monitor.record,trace,omitted,ticks,indices,counts,dtMs,durationMs,startMs:startTick*dtMs,spikes,meanHz:spikes/n/(durationMs/1000),active:counts.filter(c=>c>0).length,spikesRecorded:p.spike_monitor!=null,symbols:Object.fromEntries(names),synapticEvents:0});
 }
 check(u64()===model.definition.synapses.length,'Synapse result count differs.');let synapticEvents=0;
 for(let k=0;k<model.definition.synapses.length;k++){const s=model.definition.synapses[k],count=edges(model.instance.synapses[k]);check(u64()===s.states.length&&u64()===count,'Synapse result dimensions differ.');for(const state of s.states)values(state.dtype,count,false);synapticEvents+=u64();}
 const finalTime=f64();check(finalTime===number(model.run.start)+number(model.run.duration),'Result final time differs.');magic('B2END001');check(at===bytes.length,'Unexpected trailing result data.');
 return {populations,synapticEvents,totalSpikes};
}
