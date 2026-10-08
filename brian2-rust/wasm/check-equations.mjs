import assert from 'node:assert/strict';
import {readFileSync,writeFileSync,mkdirSync} from 'node:fs';
import {pathToFileURL} from 'node:url';
import {models} from './models.js';
import {makeDraft,decodeExperiment,validateConfig} from './experiment.js';
import {parseEquations,expression} from './equations.js';
import {compileWebGPU} from './webgpu.js';
const [root,output]=process.argv.slice(2),template=JSON.parse(readFileSync(`${root}/editor-template.json`));
const {initSync,compile_model,BrowserExecutor}=await import(pathToFileURL(`${root}/pkg/b2_runner.js`));
initSync({module:readFileSync(`${root}/pkg/b2_runner_bg.wasm`)});
function run(c){const draft=makeDraft(template,c),bundle=JSON.parse(compile_model(JSON.stringify(draft))),executor=new BrowserExecutor(bundle.model_json,bundle.plan_json);try{while(!executor.finished)executor.step(128);return {draft,data:decodeExperiment(executor.results(),draft)};}finally{executor.free();}}
const cases=Object.fromEntries(['adex','quadratic_if','custom'].flatMap(id=>Object.entries(models[id].presets).map(([name,p])=>[`${id}-${name}`,{...p.config,neurons:8}])));
const simultaneous={...models.custom.defaults,neurons:8,duration_ms:50,dt_ms:1,spread:0,custom:{equations:'dv/dt = w / ms : 1\ndw/dt = -v / ms : 1',parameters:{},initial:{v:1,w:0},threshold:'v > 1e12',reset:'',refractory_ms:0}};
cases.simultaneous=simultaneous;
for(const [id,c] of Object.entries(cases)){
 const {data,draft}=run(c);assert.deepEqual(data.ticks,run(c).data.ticks);compileWebGPU(draft,c);
 if(id==='simultaneous'){assert.equal(data.trace.v[8],1);assert.equal(data.trace.w[8],-1);assert.equal(data.trace.v[16],0);assert.equal(data.trace.w[16],-2);}
 const dir=`${output}/${id}`;mkdirSync(dir,{recursive:true});writeFileSync(`${dir}/config.json`,JSON.stringify(c));writeFileSync(`${dir}/model.json`,JSON.stringify(draft));writeFileSync(`${dir}/observed.json`,JSON.stringify({ticks:Array.from(data.ticks),indices:Array.from(data.indices),trace:Object.fromEntries(Object.entries(data.trace).map(([k,v])=>[k,Array.from(v)]))}));
}
const allowed=new Set(['v']);for(const invalid of ['globalThis.alert(1)','import os','v.constructor','rand()','v**-1','v**0.5','v**9','exp(v);v','((((v**8)**8)**8)**8)','1e999','('.repeat(40)+'v'+')'.repeat(40)])assert.throws(()=>expression(invalid,allowed),invalid);
for(const change of [{initial:{v:0}},{parameters:{ms:1}},{equations:'dv/dt = -v : volt'},{threshold:'missing > 0'},{reset:'drive = 0'},{refractory_ms:-1}])assert.throws(()=>parseEquations({...models.custom.defaults.custom,...change}));
assert.throws(()=>run({...simultaneous,custom:{...simultaneous.custom,equations:'dv/dt = w : 1\ndw/dt = -v / ms : 1'}}),/dimension/);
assert.throws(()=>run({...simultaneous,custom:{...simultaneous.custom,threshold:'v + w'}}));
assert.throws(()=>run({...simultaneous,custom:{...simultaneous.custom,equations:'dv/dt = exp(1000) / ms : 1\ndw/dt = 0 / ms : 1'}}),/finite|unstable/);
// New maximum reference sizes run with the same default durations and steps.
const scales=[];
for(const id of ['adaptive_lif','izhikevich','hodgkin_huxley']){const meta=models[id],c={...meta.defaults,neurons:meta.scales.at(-1)},t=JSON.parse(readFileSync(`${root}/${meta.template}`)),draft=makeDraft(t,c),bundle=JSON.parse(compile_model(JSON.stringify(draft))),e=new BrowserExecutor(bundle.model_json,bundle.plan_json),start=performance.now();try{while(!e.finished)e.step(128);const d=decodeExperiment(e.results(),draft);assert.ok(d.spikes>0);scales.push({id,n:d.n,spikes:d.spikes,ms:performance.now()-start});}finally{e.free();}assert.throws(()=>validateConfig({...c,neurons:c.neurons+1}));}
console.log(JSON.stringify({cases:Object.keys(cases),invalidSyntax:'rejected',unitErrors:'rejected',simultaneousEuler:'passed',scales},null,2));
