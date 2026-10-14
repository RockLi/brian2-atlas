import {models} from './models.js';
import {makeDraft,decodeExperiment} from './experiment.js';
import {runDraft} from './runtime.js';
const $=id=>document.getElementById(id),assert=(value,message)=>{if(!value)throw Error(message);};
$('run').onclick=async()=>{
 $('run').disabled=true;const report={agent:navigator.userAgent,cases:[],scales:[]};
 try{
 const template=await (await fetch('./editor-template.json')).json();
 for(const id of ['adex','quadratic_if','custom'])for(const [name,preset] of Object.entries(models[id].presets)){
  const c=preset.config,draft=makeDraft(template,c),start=performance.now();$('status').textContent=`Checking ${id} / ${name}`;
  const wasm=await runDraft(draft,{backend:'wasm',config:c}),reference=decodeExperiment(wasm.results,draft),gpu=await runDraft(draft,{backend:'webgpu',config:c}),actual=gpu.data;
  const countDifferences=actual.counts.reduce((sum,count,i)=>sum+(count!==reference.counts[i]),0),maxCountDelta=actual.counts.reduce((max,count,i)=>Math.max(max,Math.abs(count-reference.counts[i])),0);
  assert(actual.n===c.neurons&&actual.record.length===12,'dimensions');assert(maxCountDelta<=1,`${id}/${name}: per-cell spike count delta exceeds one`);if(name==='silent')assert(actual.spikes===0,'silent preset');else assert(actual.spikes>0,'active preset');
  report.cases.push({id,name,wasmSpikes:reference.spikes,gpuSpikes:actual.spikes,countDifferences,maxCountDelta,ms:Math.round(performance.now()-start)});$('report').textContent=JSON.stringify(report,null,2);
 }
 // An exactly representable coupled system exposes accidental sequential Euler.
 const exact={...models.custom.defaults,neurons:8,duration_ms:62.5,dt_ms:0.9765625,custom:{equations:'dv/dt = 1024*w / second : 1\ndw/dt = -1024*v / second : 1\ndz/dt = -512*z / second : 1\ndA/dt = 0 / second : 1',parameters:{},initial:{v:1,w:0,z:1,A:3},threshold:'v > 1e12 and not w > 1e12',reset:'',refractory_ms:0}},draft=makeDraft(template,exact);
 const a=decodeExperiment((await runDraft(draft,{backend:'wasm',config:exact})).results,draft),b=(await runDraft(draft,{backend:'webgpu',config:exact})).data;
 for(const key of Object.keys(a.trace))assert(a.trace[key].every((v,i)=>v===b.trace[key][i]),'Exact coupled Euler differs: '+key);report.exactFourStateEuler='passed';
 for(const [backend,ids] of [['webgpu',['adaptive_lif','izhikevich','hodgkin_huxley','adex','quadratic_if','custom']],['wasm',['adaptive_lif','izhikevich','hodgkin_huxley','adex','quadratic_if','custom']]])for(const id of ids){
  const meta=models[id],c={...meta.defaults,neurons:(backend==='webgpu'?meta.gpuScales:meta.scales).at(-1)},t=await (await fetch('./'+meta.template)).json();$('status').textContent=`Scale check: ${backend} / ${id} / ${c.neurons}`;
  const draft=makeDraft(t,c,{backend}),start=performance.now(),r=await runDraft(draft,{backend,config:c}),d=r.data??decodeExperiment(r.results,draft);
  assert(d.spikes>0&&d.counts.reduce((a,b)=>a+b,0)===d.spikes,'full statistics');report.scales.push({backend,id,n:c.neurons,spikes:d.spikes,ms:Math.round(performance.now()-start)});$('report').textContent=JSON.stringify(report,null,2);
 }
 report.status='passed';$('status').textContent='Passed';
 }catch(error){report.status='failed';report.error=String(error);$('status').textContent='Failed';}finally{$('report').textContent=JSON.stringify(report,null,2);$('run').disabled=false;}
};
