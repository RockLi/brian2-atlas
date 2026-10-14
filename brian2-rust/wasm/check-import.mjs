import assert from 'node:assert/strict';
import {readFileSync,writeFileSync} from 'node:fs';
import {pathToFileURL} from 'node:url';
import {parseBundleText,inspectBundle,decodeImported} from './imported-model.js';
const [root,input,output]=process.argv.slice(2),text=readFileSync(input,'utf8'),loaded=parseBundleText(text),original=loaded.bundle.model_json;
const {initSync,BrowserExecutor}=await import(pathToFileURL(`${root}/pkg/b2_runner.js`));initSync({module:readFileSync(`${root}/pkg/b2_runner_bg.wasm`)});
function run(bundle){const e=new BrowserExecutor(bundle.model_json,bundle.plan_json);try{assert.equal(e.plan_sha256,bundle.plan_sha256);while(!e.finished)e.step(128);return {results:e.results(),events:e.events(),summary:JSON.parse(e.summary())};}finally{e.free();}}
const result=run(loaded.bundle),decoded=decodeImported(result.results,loaded.model);assert.equal(loaded.bundle.model_json,original,'Original wire JSON must remain untouched');
const bad={...loaded.bundle,plan_sha256:'0'.repeat(64)};assert.throws(()=>run(bad));
const corruptPlan=JSON.parse(loaded.bundle.plan_json);corruptPlan.strategy='forged';assert.throws(()=>run({...loaded.bundle,plan_json:JSON.stringify(corruptPlan)}));
const edited=JSON.parse(original);edited.instance.rng_seed=3;assert.throws(()=>run({...loaded.bundle,model_json:JSON.stringify(edited)}));
assert.throws(()=>parseBundleText('not json'));assert.throws(()=>parseBundleText('{"schema":"neural-lab-design-v1"}'));
for(const modify of [m=>m.definition.populations[0].count=100001,m=>m.definition.populations[0].monitor.record=Array(65).fill(0),m=>m.definition.populations[0].event_monitors=[{}],m=>m.definition.functions=[{body:null}],m=>m.definition.populations[0].steps=40001]){const m=JSON.parse(original);modify(m);assert.throws(()=>inspectBundle({...loaded.bundle,model_json:JSON.stringify(m)}));}
assert.throws(()=>decodeImported(result.results.subarray(0,result.results.length-1),loaded.model));
writeFileSync(`${output}/results.bin`,result.results);writeFileSync(`${output}/events.bin`,result.events);writeFileSync(`${output}/summary.json`,JSON.stringify(result.summary));
writeFileSync(`${output}/decoded.json`,JSON.stringify({...decoded,populations:decoded.populations.map(p=>({...p,ticks:Array.from(p.ticks),indices:Array.from(p.indices),counts:Array.from(p.counts),trace:Object.fromEntries(Object.entries(p.trace).map(([key,v])=>[key,Array.from(v)]))}))}));
console.log(JSON.stringify({populations:decoded.populations.map(p=>({name:p.name,n:p.n,spikes:p.spikes,variables:Object.keys(p.trace),omitted:p.omitted,windowMs:p.durationMs,startMs:p.startMs})),totalSpikes:decoded.totalSpikes,synapticEvents:decoded.synapticEvents,guards:'passed'}));
