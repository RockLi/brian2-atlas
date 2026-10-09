import {readFileSync,writeFileSync,mkdirSync} from 'node:fs';
import {pathToFileURL} from 'node:url';
import assert from 'node:assert/strict';
import {models} from './models.js';
import {makeDraft,decodeExperiment,validateConfig} from './experiment.js';
const [root,output]=process.argv.slice(2);
const {initSync,compile_model,BrowserExecutor}=await import(pathToFileURL(`${root}/pkg/b2_runner.js`));
initSync({module:readFileSync(`${root}/pkg/b2_runner_bg.wasm`)});
const metrics={};
for(const [id,meta] of Object.entries(models)){
  if(id==='flywire')continue;
  const template=JSON.parse(readFileSync(`${root}/${meta.template}`,'utf8'));metrics[id]={};
  for(const [name,preset] of Object.entries(meta.presets)){
    const config=preset.config,draft=makeDraft(template,config),bundle=JSON.parse(compile_model(JSON.stringify(draft)));
    const executor=new BrowserExecutor(bundle.model_json,bundle.plan_json);
    try{
      while(!executor.finished)executor.step(128);
      const bytes=executor.results(),data=decodeExperiment(bytes,draft);
      const spikes=Array.from(data.ticks).filter((_,k)=>data.indices[k]===0).map(t=>t*data.dtMs);
      const isi=spikes.slice(1).map((t,k)=>t-spikes[k]);
      const bounds=Object.fromEntries(Object.entries(data.trace).map(([key,values])=>{let min=Infinity,max=-Infinity;for(const v of values){assert.ok(Number.isFinite(v));min=Math.min(min,v);max=Math.max(max,v);}return [key,[min,max]];}));
      metrics[id][name]={spikes:data.spikes,meanHz:data.meanHz,isiMin:isi.length?Math.min(...isi):null,isiMax:isi.length?Math.max(...isi):null,bounds};
      if(name==='silent')assert.equal(data.spikes,0,`${id}: resting preset must stay silent`);else assert.ok(data.spikes>0);
      if(id==='hodgkin_huxley'){
        for(const key of ['m','h','n']){assert.ok(bounds[key][0]>=0&&bounds[key][1]<=1,'gates remain probabilities');}
        if(name!=='silent'){assert.ok(bounds.v[1]>20&&bounds.v[0]<-60,'continuous HH action potential');assert.ok(Math.min(...isi)>1,'one event per action potential');}
      }
      const dir=`${output}/${id}-${name}`;mkdirSync(dir,{recursive:true});
      writeFileSync(`${dir}/model.json`,bundle.model_json);writeFileSync(`${dir}/config.json`,JSON.stringify(config));
      writeFileSync(`${dir}/results.bin`,bytes);writeFileSync(`${dir}/events.bin`,executor.events());writeFileSync(`${dir}/summary.json`,executor.summary());
    }finally{executor.free();}
  }
  assert.throws(()=>validateConfig({...meta.defaults,dt_ms:meta.dt[1]*2}));
}
assert.ok(metrics.izhikevich.fast.meanHz>metrics.izhikevich.regular.meanHz,'fast preset fires faster');
assert.ok(metrics.izhikevich.bursting.isiMax>metrics.izhikevich.bursting.isiMin*3,'burst preset separates intra-burst and inter-burst times');
assert.ok(metrics.hodgkin_huxley.strong.meanHz>metrics.hodgkin_huxley.tonic.meanHz,'higher current increases HH rate in the demonstrated range');
console.log(JSON.stringify(metrics,null,2));
