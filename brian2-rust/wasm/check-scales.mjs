import {readFileSync,writeFileSync,mkdirSync} from 'node:fs';
import {pathToFileURL} from 'node:url';
import assert from 'node:assert/strict';
import {models} from './models.js';
import {validateConfig,makeDraft,decodeExperiment} from './experiment.js';
import {validateCircuit} from './network.js';
const [root,output]=process.argv.slice(2);
const {initSync,compile_model,BrowserExecutor}=await import(pathToFileURL(`${root}/pkg/b2_runner.js`));
initSync({module:readFileSync(`${root}/pkg/b2_runner_bg.wasm`)});
const metrics=[];
for(const [id,n] of [['adaptive_lif',32768],['izhikevich',8192],['hodgkin_huxley',4096],['flywire',1024],['flywire',4096]]){
 const config={...models[id].defaults,neurons:n},start=performance.now();
 const filename=id==='flywire'?`flywire-${n}-template.json`:models[id].template;
 const template=JSON.parse(readFileSync(`${root}/${filename}`));
 if(id==='flywire')validateCircuit(JSON.parse(readFileSync(`${root}/flywire-circuit-${n}.json`)),template);
 const draft=makeDraft(template,config),bundle=JSON.parse(compile_model(JSON.stringify(draft))),executor=new BrowserExecutor(bundle.model_json,bundle.plan_json);
 try{
  while(!executor.finished)executor.step(128);
  const bytes=executor.results(),data=decodeExperiment(bytes,draft);
  assert.equal(data.n,n);assert.equal(data.record.length,12);assert.ok(data.spikes>0);for(const values of Object.values(data.trace))assert.ok(values.every(Number.isFinite));
  assert.equal(data.counts.reduce((a,b)=>a+b,0),data.spikes);
  const entry={model:id,neurons:n,spikes:data.spikes,meanHz:data.meanHz,synapticEvents:data.synapticEvents,ms:Math.round(performance.now()-start),modelBytes:bundle.model_json.length,resultBytes:bytes.length};metrics.push(entry);console.log(JSON.stringify(entry));
  const dir=`${output}/${id}-${n}`;mkdirSync(dir,{recursive:true});writeFileSync(`${dir}/model.json`,bundle.model_json);writeFileSync(`${dir}/results.bin`,bytes);writeFileSync(`${dir}/events.bin`,executor.events());writeFileSync(`${dir}/summary.json`,executor.summary());
 }finally{executor.free();}
}
assert.throws(()=>validateConfig({...models.adaptive_lif.defaults,neurons:32769}));
assert.throws(()=>validateConfig({...models.adaptive_lif.defaults,neurons:16384,duration_ms:2000}));
assert.throws(()=>validateConfig({...models.flywire.defaults,neurons:512}));
assert.throws(()=>makeDraft(JSON.parse(readFileSync(`${root}/flywire-template.json`)),{...models.flywire.defaults,neurons:1024}));
writeFileSync(`${output}/metrics.json`,JSON.stringify(metrics,null,2));
