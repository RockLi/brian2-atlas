import {readFileSync,writeFileSync,mkdirSync} from 'node:fs';
import {pathToFileURL} from 'node:url';
import assert from 'node:assert/strict';
import {models} from './models.js';
import {makeDraft,decodeExperiment,validateConfig,bits,number} from './experiment.js';
const [root,output]=process.argv.slice(2),circuit=JSON.parse(readFileSync(new URL('./flywire-circuit.json',import.meta.url),'utf8'));
const template=JSON.parse(readFileSync(`${root}/flywire-template.json`,'utf8'));
const {initSync,compile_model,BrowserExecutor}=await import(pathToFileURL(`${root}/pkg/b2_runner.js`));
initSync({module:readFileSync(`${root}/pkg/b2_runner_bg.wasm`)});
const runs={},metrics={};
for(const [name,preset] of Object.entries(models.flywire.presets)){
  const config=preset.config,draft=makeDraft(template,config),bundle=JSON.parse(compile_model(JSON.stringify(draft)));
  const executor=new BrowserExecutor(bundle.model_json,bundle.plan_json);
  try{
    while(!executor.finished)executor.step(128);
    const bytes=executor.results(),data=decodeExperiment(bytes,draft);runs[name]=data;
    const groups=Object.fromEntries(circuit.groups.map(g=>[g.id,0]));
    for(let i=0;i<data.n;i++)groups[circuit.nodes[i].group]+=data.counts[i];
    metrics[name]={spikes:data.spikes,meanHz:data.meanHz,synapticEvents:data.synapticEvents,groups};
    assert.ok(data.synapticEvents>0,'Real recurrent events must be delivered');
    for(const key of ['v','ge','gi'])assert.ok(data.trace[key].every(Number.isFinite));
    assert.ok(data.trace.ge.every(x=>x>=0)&&data.trace.gi.every(x=>x>=0));
    const dir=`${output}/${name}`;mkdirSync(dir,{recursive:true});
    writeFileSync(`${dir}/model.json`,bundle.model_json);writeFileSync(`${dir}/config.json`,JSON.stringify(config));
    writeFileSync(`${dir}/results.bin`,bytes);writeFileSync(`${dir}/events.bin`,executor.events());writeFileSync(`${dir}/summary.json`,executor.summary());
    // Tick chunking cannot change any stochastic draws or delayed deliveries.
    if(name==='odor'){
      const chunked=new BrowserExecutor(bundle.model_json,bundle.plan_json);
      try{while(!chunked.finished)chunked.step(17);assert.deepEqual(chunked.results(),bytes);}finally{chunked.free();}
    }
  }finally{executor.free();}
}
const nonsensory=data=>Array.from(data.ticks,(tick,k)=>[tick,data.indices[k]]).filter(([,i])=>circuit.nodes[i].group!=='sensory');
assert.deepEqual(nonsensory(runs.cut),nonsensory(runs.cut_rest),'Cut isolates downstream activity from the sensory stimulus');
assert.ok(metrics.odor.groups.sensory>metrics.rest.groups.sensory,'Stimulus activates sensory cells');
assert.ok(metrics.odor.groups.projection>metrics.rest.groups.projection,'Intact stimulus increases projection-neuron response');
assert.ok(metrics.odor.groups.projection>metrics.cut.groups.projection,'Blocking sensory output reduces projection response');
assert.throws(()=>validateConfig({...models.flywire.defaults,neurons:128}));
assert.throws(()=>validateConfig({...models.flywire.defaults,stimulus_start_ms:250,stimulus_end_ms:200}));
assert.throws(()=>validateConfig({...models.flywire.defaults,dt_ms:.2}));
// A deterministic conductance pulse enables independent Brian NumPy comparison.
{
  const config={...models.flywire.defaults,background_rate_hz:0,sensory_rate_hz:0},draft=makeDraft(template,config);
  const instance=draft.instance.populations[0];instance.initial_state.ge=instance.parameters.is_sensory.map(x=>bits(number(x)?2:0));
  const bundle=JSON.parse(compile_model(JSON.stringify(draft))),executor=new BrowserExecutor(bundle.model_json,bundle.plan_json);
  try{
    while(!executor.finished)executor.step(128);
    const data=decodeExperiment(executor.results(),draft);assert.ok(data.spikes>68&&data.synapticEvents>0,'A deterministic pulse must propagate through recurrent edges');
    const dir=`${output}/pulse`;mkdirSync(dir,{recursive:true});writeFileSync(`${dir}/model.json`,bundle.model_json);writeFileSync(`${dir}/config.json`,JSON.stringify(config));writeFileSync(`${dir}/results.bin`,executor.results());writeFileSync(`${dir}/events.bin`,executor.events());writeFileSync(`${dir}/summary.json`,executor.summary());
    metrics.pulse={spikes:data.spikes,synapticEvents:data.synapticEvents};
  }finally{executor.free();}
}
console.log(JSON.stringify(metrics,null,2));
