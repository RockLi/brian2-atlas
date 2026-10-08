// Bounded WGSL lowering of the lab's exported Brian2 single-cell programs.
// Each lane owns one cell, its spike bitmap and its recorded trace locations.
import {number,validateConfig} from './experiment.js';
export const WEBGPU_PROFILE='webgpu-f32-independent-v1';
const supported=new Set(['adaptive_lif','izhikevich','hodgkin_huxley','adex','quadratic_if','custom']);
export function supportsWebGPU(model){return supported.has(model);}
export async function webgpuInfo(){
  if(!globalThis.isSecureContext||!navigator.gpu)return {available:false,reason:'WebGPU requires a supported browser and HTTPS or localhost.'};
  const adapter=await navigator.gpu.requestAdapter();
  if(!adapter)return {available:false,reason:'No WebGPU adapter is available. Select WASM.'};
  return {available:true,adapter:{vendor:adapter.info?.vendor,architecture:adapter.info?.architecture,description:adapter.info?.description},limits:Object.fromEntries(['maxBufferSize','maxStorageBufferBindingSize','maxComputeWorkgroupsPerDimension','maxComputeInvocationsPerWorkgroup','maxStorageBuffersPerShaderStage'].map(k=>[k,adapter.limits[k]]))};
}
const literal=x=>{if(!Number.isFinite(x)||!Number.isFinite(Math.fround(x)))throw Error('Value cannot be represented as f32.');return `${Math.fround(x).toExponential(9)}f`;};
export function compileWebGPU(model,input){
  const c=validateConfig(input,{backend:'webgpu'});
  if(!supportsWebGPU(c.model))throw Error('WebGPU supports independent built-in and Equation Lab models. Use WASM for FlyWire.');
  const d=model.definition,p=d.populations[0],instance=model.instance.populations[0];
  if(d.populations.length!==1||d.synapses.length||d.clocks.length!==1||d.functions.length||p.linked_variables.length||p.count!==c.neurons||number(model.run.start)!==0||p.events.join()!=='spike'||p.event_monitors.length)throw Error('WebGPU requires a single independent population, one clock and a fresh run.');
  if(p.states.some(s=>s.dtype!=='f64')||p.parameters.some(s=>s.dtype!=='f64'))throw Error('WebGPU lab lowering expects f64 authoring states and explicitly converts them to f32.');
  const n=p.count,steps=p.steps,record=p.monitor.record,variables=p.monitor.variables,states=p.states.map(s=>s.name),words=Math.ceil(steps/32);
  if(steps!==Math.round(c.duration_ms/c.dt_ms)||number(p.dt)!==c.dt_ms/1000||record.length>12||new Set(record).size!==record.length||record.some(i=>i<0||i>=n))throw Error('WebGPU dimensions or clock mismatch.');
  const symbols=new Map(),types=new Map();let serial=0;const bind=(key,type)=>{const value=`x${serial++}`;symbols.set(key,value);types.set(key,type);return value;};
  let init='',finish='',parameters=[],offset=0;
  for(const [k,name] of states.entries()){const v=bind(name,'f32');init+=`var ${v}: f32 = state[${k*n}u+i];\n`;finish+=`state[${k*n}u+i]=${v};\n`;}
  for(const parameter of p.parameters){const values=instance.parameters[parameter.name];if(values.length!==(parameter.index_domain==='scalar'?1:n))throw Error('Invalid parameter shape.');const v=bind(parameter.name,'f32');init+=`let ${v}=parameters[${offset}u${values.length===1?'':'+i'}];\n`;for(const value of values)parameters.push(number(value));offset+=values.length;}
  symbols.set('dt',literal(number(p.dt)));types.set('dt','f32');symbols.set('i','i32(i)');types.set('i','i32');symbols.set('N',`${n}`);types.set('N','i32');symbols.set('t',`(f32(tick)*${literal(number(p.dt))})`);types.set('t','f32');
  const refractory=bind('not_refractory','bool');init+=`var ${refractory}=state[${(states.length+1)*n}u+i]>0.0f;\nvar last=i32(state[${states.length*n}u+i]);\nvar failed=0.0f;\n`;
  function expression(e){
    if(e.op==='load'){if(!symbols.has(e.name))throw Error(`Unsupported WebGPU symbol: ${e.name}`);return {s:symbols.get(e.name),t:types.get(e.name)};}
    if(e.op==='literal')return {s:literal(number(e.bits)),t:'f32'};
    if(e.op==='integer'){const v=Number(e.value);if(!Number.isSafeInteger(v)||v<-2147483648||v>2147483647)throw Error('WebGPU integer exceeds i32.');return {s:String(v),t:'i32'};}
    if(e.op==='cast'){const a=expression(e.arg),t=e.dtype==='bool'?'bool':e.dtype==='f64'||e.dtype==='f32'?'f32':'i32';return {s:a.t===t?a.s:a.t==='bool'?`select(${t}(0),${t}(1),${a.s})`:t==='bool'?`(${a.s}!=${a.t}(0))`:`${t}(${a.s})`,t};}
    if(e.op==='neg'||e.op==='not'){const a=expression(e.arg);return {s:`(${e.op==='neg'?'-':'!'}${a.s})`,t:a.t};}
    if(e.op==='exp'||e.op==='exprel'){const a=expression(e.arg);return {s:`${e.op==='exprel'?'b2_exprel':'exp'}(${a.s})`,t:'f32'};}
    const operators={add:'+',sub:'-',mul:'*',div:'/',gt:'>',ge:'>=',lt:'<',le:'<=',eq:'==',ne:'!=',and:'&&',or:'||'};
    if(e.op==='pow'){
      let power=e.right;while(power.op==='cast')power=power.arg;const exponent=power.op==='integer'?Number(power.value):power.op==='literal'?number(power.bits):NaN;
      if(!Number.isInteger(exponent)||exponent<0||exponent>8)throw Error('WebGPU lab supports only small nonnegative integer powers.');
      const a=expression(e.left);return {s:exponent?`(${Array(exponent).fill(a.s).join('*')})`:'1.0f',t:'f32'};
    }
    if(operators[e.op]){const a=expression(e.left),b=expression(e.right);if(a.t!==b.t)throw Error('WebGPU expression operand types differ.');return {s:`(${a.s} ${operators[e.op]} ${b.s})`,t:['gt','ge','lt','le','eq','ne','and','or'].includes(e.op)?'bool':a.t};}
    throw Error(`Unsupported WebGPU expression: ${e.op}`);
  }
  function code(object){
    const savedSymbols=new Map(symbols),savedTypes=new Map(types);let text='{\n';
    for(const assignment of [...object.scalar,...object.vector]){
      const a=expression(assignment.value);let name=symbols.get(assignment.target),declaration='';
      if(!name){name=bind(assignment.target,a.t);declaration=`var `;}
      const line=`${declaration}${name}${declaration?': '+a.t:''}=${a.s};\n`;
      if(assignment.condition){const condition=symbols.get(assignment.condition);if(!condition||declaration)throw Error('Unsupported conditional local.');text+=`if(${condition}){${line}}\n`;}else text+=line;
    }
    if(object.kind==='threshold'){const condition=symbols.get('_cond');if(!condition)throw Error('Missing spike condition.');text+=`spiked=${condition}${p.refractory?' && '+refractory:''};\nif(spiked){last=i32(tick);${p.refractory?refractory+'=false;':''}}\n`;}
    text+='}\n';symbols.clear();types.clear();for(const [k,v]of savedSymbols)symbols.set(k,v);for(const [k,v]of savedTypes)types.set(k,v);return text;
  }
  let body='var spiked=false;\n';let monitors=0,thresholds=0;
  if(p.refractory?.mode==='fixed')body+=`${refractory}=i32(tick)-last>=${instance.refractory.period_ticks};\n`;
  for(const node of d.schedule.nodes){
    if(node.owner_kind!=='population'||node.owner_index!==0||node.clock!==0)throw Error('Unsupported WebGPU schedule.');
    if(node.operation==='state_monitor'){
      const monitor=p.state_monitors[node.item_index];if(monitors++||monitor.record.join()!==record.join()||monitor.variables.join()!==variables.join())throw Error('Unsupported WebGPU state monitor.');
      body+=`if(probe>=0){\n${variables.map((v,k)=>`traces[${k*steps*record.length}u+tick*${record.length}u+u32(probe)]=${symbols.get(v)};`).join('\n')}\n}\n`;
    }else if(node.operation==='spike_monitor')body+=`if(spiked){spikes[i*${words}u+tick/32u] |= 1u<<(tick%32u);}\n`;
    else if(node.operation==='code_object'){
      const object=p.code_objects[node.item_index];if(!['state_update','threshold','reset'].includes(object.kind))throw Error('WebGPU lab does not support random input or custom events.');
      if(object.kind==='threshold')thresholds++;
      const emitted=code(object);body+=object.kind==='reset'?`if(spiked)${emitted}`:emitted;
    }else throw Error('Unsupported WebGPU schedule operation.');
  }
  if(monitors!==1||thresholds!==1)throw Error('WebGPU requires one state monitor and one threshold.');
  body+=`if(!(${states.map(s=>`abs(${symbols.get(s)})<=3.402823e38f`).join(' && ')})){failed=1.0f;break;}\n`;
  finish+=`state[${states.length*n}u+i]=f32(last);state[${(states.length+1)*n}u+i]=select(0.0f,1.0f,${refractory});state[${(states.length+2)*n}u+i]=failed;`;
  const source=`// ${WEBGPU_PROFILE}: Brian2 B2IR expressions lowered to WGSL f32.
@group(0) @binding(0) var<storage,read_write> state: array<f32>;
@group(0) @binding(1) var<storage,read> parameters: array<f32>;
@group(0) @binding(2) var<storage,read_write> spikes: array<u32>;
@group(0) @binding(3) var<storage,read_write> traces: array<f32>;
struct Chunk {start:u32,end:u32,pad0:u32,pad1:u32};
@group(0) @binding(4) var<uniform> chunk:Chunk;
fn b2_exprel(x:f32)->f32 {
  // Cancellation-safe series, as required by the native GPU exprel contract.
  if(abs(x)<0.125f){return 1.0f+x*(0.5f+x*(0.1666666667f+x*(0.0416666667f+x*(0.0083333333f+x*0.0013888889f))));}
  if(x>=88.722839f){let half_exp=exp(0.5f*x);return (half_exp/x)*half_exp;}
  return (exp(x)-1.0f)/x;
}
@compute @workgroup_size(64) fn simulate(@builtin(global_invocation_id) gid:vec3<u32>){
 let i=gid.x;if(i>=${n}u){return;}
 ${init}
 var probe:i32=-1;${record.map((v,k)=>`if(i==${v}u){probe=${k};}`).join('\n')}
 for(var tick=chunk.start;tick<chunk.end;tick++){${body}}
 ${finish}
}`;
  const initial=new Float32Array((states.length+3)*n);states.forEach((name,k)=>initial.set(instance.initial_state[name].map(number),k*n));
  initial.fill(-1000000,states.length*n,(states.length+1)*n);initial.fill(1,(states.length+1)*n,(states.length+2)*n);
  if(initial.some(v=>!Number.isFinite(v))||parameters.some(v=>!Number.isFinite(Math.fround(v))))throw Error('WebGPU input exceeds finite f32 range.');
  const sizes=[initial.byteLength,Math.max(4,parameters.length*4),n*words*4,steps*record.length*variables.length*4];
  return {source,initial,parameters:new Float32Array(parameters),sizes,n,steps,record,variables,states,words,dtMs:c.dt_ms,durationMs:c.duration_ms};
}
export function checkWebGPULimits(program,limits){
  if(limits.maxComputeInvocationsPerWorkgroup<64||limits.maxStorageBuffersPerShaderStage<4||Math.ceil(program.n/64)>limits.maxComputeWorkgroupsPerDimension)throw Error('WebGPU adapter compute limits are too small.');
  const max=Math.min(limits.maxBufferSize,limits.maxStorageBufferBindingSize);
  if(program.sizes.some(size=>size>max))throw Error(`WebGPU buffer exceeds the device limit (${Math.floor(max/1048576)} MiB). Reduce population or duration.`);
  // Output, mapped staging and host copies coexist. This is an application budget,
  // not a promise about the browser's available GPU memory.
  if(program.sizes.reduce((a,b)=>a+b,0)*3>256*1048576)throw Error('WebGPU working set exceeds the 256 MiB lab budget. Reduce population or duration.');
}
export async function executeWebGPU(program,{onProgress=()=>{},chunkTicks=256}={}){
  if(!Number.isInteger(chunkTicks)||chunkTicks<1||chunkTicks>256)throw Error('WebGPU chunks must contain 1–256 ticks.');
  if(!navigator.gpu)throw Error('WebGPU is unavailable. Select WASM.');
  const adapter=await navigator.gpu.requestAdapter();if(!adapter)throw Error('No WebGPU adapter is available. Select WASM.');
  const device=await adapter.requestDevice(),buffers=[];let lost;
  device.lost.then(info=>{lost=info.message||'WebGPU device lost';});
  device.addEventListener('uncapturederror',e=>{lost=e.error.message;});
  try{
    checkWebGPULimits(program,device.limits);device.pushErrorScope('validation');device.pushErrorScope('out-of-memory');
    const module=device.createShaderModule({code:program.source}),messages=await module.getCompilationInfo();
    const errors=messages.messages.filter(m=>m.type==='error');if(errors.length)throw Error(errors.map(m=>`${m.lineNum}: ${m.message}`).join('\n'));
    const pipeline=await device.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint:'simulate'}});
    const create=(size,usage)=>{const buffer=device.createBuffer({size,usage});buffers.push(buffer);return buffer;};
    const storage=program.sizes.map(size=>create(size,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST));
    const uniform=create(16,GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST);
    const bind=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[...storage,uniform].map((buffer,binding)=>({binding,resource:{buffer}}))});
    device.queue.writeBuffer(storage[0],0,program.initial);device.queue.writeBuffer(storage[1],0,program.parameters);
    for(let start=0;start<program.steps;start+=chunkTicks){
      device.queue.writeBuffer(uniform,0,new Uint32Array([start,Math.min(start+chunkTicks,program.steps),0,0]));
      const encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,bind);pass.dispatchWorkgroups(Math.ceil(program.n/64));pass.end();device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();if(lost)throw Error(lost);onProgress({fraction:Math.min(1,(start+chunkTicks)/program.steps),phase:'compute'});
    }
    const readbacks=[0,2,3].map(i=>create(program.sizes[i],GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ)),encoder=device.createCommandEncoder();
    [0,2,3].forEach((i,j)=>encoder.copyBufferToBuffer(storage[i],0,readbacks[j],0,program.sizes[i]));device.queue.submit([encoder.finish()]);
    await Promise.all(readbacks.map(b=>b.mapAsync(GPUMapMode.READ)));
    const output=readbacks.map(b=>{const copy=b.getMappedRange().slice(0);b.unmap();return copy;});
    const oom=await device.popErrorScope(),validation=await device.popErrorScope();if(oom||validation||lost)throw Error(oom?.message||validation?.message||lost);
    return {state:new Float32Array(output[0]),bitmap:new Uint32Array(output[1]),traces:new Float32Array(output[2]),device:{vendor:adapter.info?.vendor,architecture:adapter.info?.architecture,description:adapter.info?.description},limits:Object.fromEntries(['maxBufferSize','maxStorageBufferBindingSize','maxComputeWorkgroupsPerDimension'].map(k=>[k,device.limits[k]]))};
  }finally{for(const buffer of buffers)buffer.destroy();device.destroy();}
}
export function decodeWebGPU(program,output){
  const {n,steps,record,variables,states,words,dtMs,durationMs}=program,{state,bitmap,traces}=output;
  if(state.some(v=>!Number.isFinite(v))||traces.some(v=>!Number.isFinite(v))||state.subarray((states.length+2)*n).some(v=>v!==0))throw Error('WebGPU produced non-finite state. Reduce timestep or input drive.');
  const counts=new Uint32Array(n),perTick=new Uint32Array(steps);let spikes=0;
  const each=callback=>{for(let i=0;i<n;i++)for(let w=0;w<words;w++){let bits=bitmap[i*words+w]>>>0;while(bits){const b=31-Math.clz32((bits&-bits)>>>0),tick=w*32+b;if(tick>=steps)throw Error('WebGPU spike out of range.');callback(i,tick);bits=(bits&(bits-1))>>>0;}}};
  each((i,t)=>{counts[i]++;perTick[t]++;spikes++;if(spikes>4000000)throw Error('WebGPU spike output exceeds the 4 million event budget. Reduce neuron count, duration or drive.');});
  const offsets=new Uint32Array(steps);let offset=0;for(let t=0;t<steps;t++){offsets[t]=offset;offset+=perTick[t];}
  const ticks=new Float64Array(spikes),indices=new Uint32Array(spikes);each((i,t)=>{const k=offsets[t]++;ticks[k]=t;indices[k]=i;});
  const trace=Object.fromEntries(variables.map((v,k)=>[v,new Float64Array(traces.subarray(k*steps*record.length,(k+1)*steps*record.length))]));
  return {n,steps,record,trace,ticks,indices,counts,dtMs,durationMs,synapticEvents:0,meanHz:spikes/n/(durationMs/1000),active:counts.filter(x=>x>0).length,spikes,finalState:Object.fromEntries(states.map((s,k)=>[s,state.slice(k*n,(k+1)*n)]))};
}
