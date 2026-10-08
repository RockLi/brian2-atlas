import {readFileSync,writeFileSync,mkdirSync} from 'node:fs';
import {pathToFileURL} from 'node:url';
import assert from 'node:assert/strict';
import {defaults,presets,validateConfig,makeDraft,decodeExperiment,firingRate} from './experiment.js';
const [pkg,templatePath,output]=process.argv.slice(2);
const {initSync,compile_model,BrowserExecutor}=await import(pathToFileURL(`${pkg}/b2_runner.js`));
initSync({module:readFileSync(`${pkg}/b2_runner_bg.wasm`)});
const template=JSON.parse(readFileSync(templatePath,'utf8'));
mkdirSync(output,{recursive:true});const results={};
for(const [name,config] of Object.entries(presets)) {
  const draft=makeDraft(template,config),bundle=JSON.parse(compile_model(JSON.stringify(draft))),plan=JSON.parse(bundle.plan_json);
  assert.equal(plan.logical.clocks[0].steps,config.duration_ms/config.dt_ms);
  const executor=new BrowserExecutor(bundle.model_json,bundle.plan_json);
  try{
    while(!executor.finished)executor.step(128);
    const bytes=executor.results(),data=decodeExperiment(bytes,draft),summary=JSON.parse(executor.summary());
    assert.equal(data.spikes,summary.spike_count);
    assert.equal(data.meanHz,data.spikes/data.n/(config.duration_ms/1000));
    const rates=firingRate(data,17);let spikes=0;
    for(let k=0;k<rates.length;k++)spikes+=rates[k]*data.n*Math.min(17,data.durationMs-k*17)/1000;
    assert.ok(Math.abs(spikes-data.spikes)<1e-7,'partial final rate bin must use its actual duration');
    const path=`${output}/${name}`;mkdirSync(path,{recursive:true});writeFileSync(`${path}/model.json`,bundle.model_json);writeFileSync(`${path}/results.bin`,bytes);writeFileSync(`${path}/events.bin`,executor.events());writeFileSync(`${path}/summary.json`,executor.summary());
    if(name==='silent')assert.equal(data.spikes,0);
    if(name==='synchronous') {assert.ok(data.spikes>0);assert.ok(data.counts.every(x=>x===data.counts[0]));for(let k=0;k<data.spikes;k+=data.n)assert.ok(data.ticks.slice(k,k+data.n).every(t=>t===data.ticks[k]));}
    if(name==='adapting'){const bin=firingRate(data,100);assert.ok(bin[0]>bin.at(-1),'adaptation should reduce late firing rate');}
    assert.throws(()=>decodeExperiment(bytes.slice(0,-1),draft));
    results[name]={spikes:data.spikes,meanHz:data.meanHz,plan:bundle.plan_sha256};
  }finally{executor.free();}
}
assert.notEqual(results.asynchronous.plan,results.synchronous.plan);
const changed=makeDraft(template,{...defaults,drive:2.5});
assert.throws(()=>new BrowserExecutor(JSON.stringify(changed),JSON.stringify({})),/hash/,'strict loading must not repair edited wire hashes');
changed.definition.populations[0].count=0;
assert.throws(()=>compile_model(JSON.stringify(changed)),/population|count/,'authoring must reject invalid semantics');
const effects=makeDraft(template,defaults);effects.definition.schedule.nodes[0].effects.reads.push('forged');assert.throws(()=>compile_model(JSON.stringify(effects)),/effects/);
assert.throws(()=>validateConfig({...defaults,dt_ms:0}));assert.throws(()=>validateConfig({...defaults,neurons:9.1}));assert.throws(()=>validateConfig({...defaults,unknown:1}));assert.throws(()=>validateConfig({...defaults,refractory_ms:2.03}));
console.log(JSON.stringify(results,null,2));
