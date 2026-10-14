import {models} from './models.js';
import {makeDraft,decodeExperiment,bits} from './experiment.js';
import {compileWebGPU,executeWebGPU,decodeWebGPU,webgpuInfo,checkWebGPULimits} from './webgpu.js';
import {runDraft} from './runtime.js';
const $=id=>document.getElementById(id),assert=(ok,message)=>{if(!ok)throw Error(message);};
function compare(a,b){
 let countMismatch=0,maxCountDelta=0;for(let i=0;i<a.n;i++){if(a.counts[i]!==b.counts[i])countMismatch++;maxCountDelta=Math.max(maxCountDelta,Math.abs(a.counts[i]-b.counts[i]));}
 const byCell=x=>{const cells=Array.from({length:x.n},()=>[]);x.indices.forEach((i,k)=>cells[i].push(x.ticks[k]));return cells;},aa=byCell(a),bb=byCell(b);let maxTickShift=0;
 aa.forEach((times,i)=>times.slice(0,bb[i].length).forEach((t,k)=>maxTickShift=Math.max(maxTickShift,Math.abs(t-bb[i][k]))));
 const trace=Object.fromEntries(Object.entries(a.trace).map(([key,values])=>{let maxAbs=0,squared=0;values.forEach((v,i)=>{const delta=v-b.trace[key][i];maxAbs=Math.max(maxAbs,Math.abs(delta));squared+=delta*delta;});return [key,{maxAbs,rmse:Math.sqrt(squared/values.length)}];}));
 return {countMismatch,maxCountDelta,maxTickShift,meanRateDelta:a.meanHz-b.meanHz,trace};
}
$('run').onclick=async()=>{
 $('run').disabled=true;document.body.dataset.done='false';const report={date:new Date().toISOString(),userAgent:navigator.userAgent,cases:[],checks:[]};
 try{
  report.hardware=await webgpuInfo();assert(report.hardware.available,report.hardware.reason);const templates={};
  for(const [id,meta] of Object.entries(models)){if(id==='flywire')continue;templates[id]=await (await fetch(meta.template)).json();}
  const cases=[];
  for(const [id,meta] of Object.entries(models)){if(id==='flywire')continue;for(const [name,preset]of Object.entries(meta.presets))cases.push([`${id}/${name}`,preset.config]);cases.push([`${id}/maximum`,{...meta.defaults,neurons:meta.scales.at(-1)}]);}
  cases.push(['adaptive_lif/exact-binary-clock',{...models.adaptive_lif.defaults,neurons:8,duration_ms:64,dt_ms:1,tau_ms:16,drive:2,spread:0,adaptation:0,synchronize:true}]);
  cases.push(['hodgkin_huxley/exprel-poles',{...models.hodgkin_huxley.defaults,neurons:8}]);
  for(const [name,c]of cases){
   $('status').textContent=`Running ${name}`;const draft=makeDraft(templates[c.model],c);
   if(name.endsWith('exprel-poles'))draft.instance.populations[0].initial_state.v=[-40,-55,-40.00001,-39.99999,-55.00001,-54.99999,-65,-65].map(bits);
   const program=compileWebGPU(draft,c),started=performance.now(),output=await executeWebGPU(program),gpu=decodeWebGPU(program,output),gpuMs=performance.now()-started;
   const cpuStarted=performance.now(),reference=await runDraft(draft),wasm=decodeExperiment(reference.results,draft),wasmMs=performance.now()-cpuStarted,comparison=compare(gpu,wasm);
   assert(gpu.counts.reduce((a,b)=>a+b,0)===gpu.spikes,`${name}: full spike counts`);
   if(name.endsWith('/silent'))assert(gpu.spikes===0,`${name}: silent preset`);else assert(gpu.spikes>0,`${name}: active preset`);
   if(c.model==='hodgkin_huxley')for(const key of ['m','h','n'])assert(gpu.trace[key].every(v=>v>=0&&v<=1),`${name}: gate bounds`);
   if(name.endsWith('exact-binary-clock')){assert(comparison.countMismatch===0&&comparison.maxTickShift===0,'Exact fixture spike coordinates');assert(comparison.trace.v.maxAbs<1e-6,'Exact fixture voltage');
    const chunked=decodeWebGPU(program,await executeWebGPU(program,{chunkTicks:17}));assert(JSON.stringify(compare(chunked,gpu))===JSON.stringify(compare(gpu,gpu)),'Chunk-independent GPU output');report.checks.push('Exact binary-clock fixture and chunk sizes 17/256');}
   const oracleResponse=await fetch(`oracle-${name.replace('/','-')}.json`);assert(oracleResponse.ok,'Generate CPU-f32 controls with tools/prepare_webgpu_checks.py');
   const oracle=await oracleResponse.json();oracle.meanHz=oracle.indices.length/oracle.n/(c.duration_ms/1000);
   const f32Comparison=compare(gpu,oracle);
   report.cases.push({f32Comparison,name,n:c.neurons,steps:gpu.steps,gpuMs,wasmMs,gpuSpikes:gpu.spikes,wasmSpikes:wasm.spikes,comparison,estimatedGpuAndStagingBytes:program.sizes.reduce((a,b)=>a+b,0)*3});
   $('report').textContent=JSON.stringify(report,null,2);
  }
  const c=models.adaptive_lif.defaults,p=compileWebGPU(makeDraft(templates.adaptive_lif,c),c);let rejected=0;
  for(const limits of [{maxBufferSize:1,maxStorageBufferBindingSize:1,maxComputeInvocationsPerWorkgroup:64,maxStorageBuffersPerShaderStage:8,maxComputeWorkgroupsPerDimension:65535},{maxBufferSize:1e9,maxStorageBufferBindingSize:1e9,maxComputeInvocationsPerWorkgroup:32,maxStorageBuffersPerShaderStage:8,maxComputeWorkgroupsPerDimension:65535}])try{checkWebGPULimits(p,limits);}catch{rejected++;}
  assert(rejected===2,'Limits rejected before allocation');report.checks.push('Buffer and compute limits rejected');
  let flyRejected=false;try{compileWebGPU({},models.flywire.defaults);}catch{flyRejected=true;}assert(flyRejected,'FlyWire must fail closed');report.checks.push('FlyWire WebGPU rejected');
  const abort=new AbortController(),job=runDraft(makeDraft(templates.adaptive_lif,{...c,neurons:16384}),{backend:'webgpu',config:{...c,neurons:16384},signal:abort.signal});abort.abort();let stopped=false;try{await job;}catch(e){stopped=e.name==='AbortError';}assert(stopped,'Worker cancellation');report.checks.push('Worker cancellation');
  // Paired end-to-end timing: both paths create a Worker, validate/compile,
  // execute and transfer results. One warmup followed by five alternating replays.
  report.paired=[];
  for(const id of ['adaptive_lif','izhikevich','hodgkin_huxley']){
   const c={...models[id].defaults,neurons:models[id].scales.at(-1)},draft=makeDraft(templates[id],c),timings={wasm:[],webgpu:[]};
   $('status').textContent=`Paired replay ${id}`;
   for(let round=-1;round<5;round++)for(const backend of round%2===0?['wasm','webgpu']:['webgpu','wasm']){
    const started=performance.now();const result=await runDraft(draft,{backend,config:c});
    if(backend==='wasm')decodeExperiment(result.results,draft);
    if(round>=0)timings[backend].push(performance.now()-started);
   }
   report.paired.push({model:id,n:c.neurons,timings});
  }
  report.gpuScale=[];
  for(const id of ['adaptive_lif','izhikevich','hodgkin_huxley']){
   const c={...models[id].defaults,neurons:models[id].gpuScales.at(-1)},draft=makeDraft(templates[id],c,{backend:'webgpu'}),started=performance.now();
   $('status').textContent=`WebGPU scale ${id} / ${c.neurons}`;
   const result=await runDraft(draft,{backend:'webgpu',config:c});assert(result.data.n===c.neurons&&result.data.spikes>0,'Larger GPU population runs');
   report.gpuScale.push({model:id,n:c.neurons,ms:performance.now()-started,spikes:result.data.spikes,meanHz:result.data.meanHz,limits:result.bundle.limits});
  }
  report.passed=true;$('status').textContent='Checks passed';
 }catch(error){report.passed=false;report.error=String(error.stack||error);$('status').textContent='Checks failed';}
 finally{$('report').textContent=JSON.stringify(report,null,2);document.body.dataset.done='true';document.body.dataset.passed=String(report.passed);$('run').disabled=false;}
};
