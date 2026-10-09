import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {pathToFileURL} from 'node:url';
import {models} from './models.js';
import {makeDraft,decodeExperiment} from './experiment.js';
const [root]=process.argv.slice(2);
const template=JSON.parse(readFileSync(`${root}/editor-template.json`));
const {initSync,compile_model,BrowserExecutor}=await import(pathToFileURL(`${root}/pkg/b2_runner.js`));
initSync({module:readFileSync(`${root}/pkg/b2_runner_bg.wasm`)});
// Replace the template's v,w states with one to four states, including new names.
for(const names of [['v'],['v','u'],['v','u','z'],['v','u','z','a']]){
  const config={...models.custom.defaults,neurons:8,duration_ms:50,dt_ms:1,spread:0,
    custom:{equations:names.map((name,i)=>`d${name}/dt = ${i+1} / ms : 1`).join('\n'),
      parameters:{},initial:Object.fromEntries(names.map((name,i)=>[name,i])),
      threshold:'v > 1e9',reset:'',refractory_ms:0}};
  const draft=makeDraft(template,config),bundle=JSON.parse(compile_model(JSON.stringify(draft)));
  const executor=new BrowserExecutor(bundle.model_json,bundle.plan_json);
  try{
    while(!executor.finished)executor.step(7);
    const observed=decodeExperiment(executor.results(),draft);
    assert.deepEqual(Object.keys(observed.trace),names);
    for(const [i,name] of names.entries())for(let tick=0;tick<50;tick++)for(let neuron=0;neuron<8;neuron++)
      assert.equal(observed.trace[name][tick*8+neuron],i+tick*(i+1));
  }finally{executor.free();}
  const invalid=structuredClone(draft);
  invalid.definition.populations[0].state_monitors[0].output_variables=['missing_state'];
  assert.throws(()=>compile_model(JSON.stringify(invalid)));
}
console.log('One- through four-state monitor outputs and invalid output rejection passed.');
