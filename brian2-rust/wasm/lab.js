import {runDraft} from './runtime.js';
import {webgpuInfo,supportsWebGPU} from './webgpu.js';
import {defaults,validateConfig,makeDraft,decodeExperiment} from './experiment.js';
import {models,modelFor,executionLimits,presentationFor} from './models.js';
import {validateCircuit,renderNetwork,networkCellAt,groupRates,groupColors} from './network.js';
import {renderCharts,emptyCharts} from './charts.js';
const $=id=>document.getElementById(id);
let template=true,selectedNetworkNode,displayedModel,config={...defaults},data,previous,bundle,lastRunConfig,controller,playing=false,frameId,running=false;
let panelMode='controls',activePanelModel;
const storageKey='brian2-neural-lab-v1';
const singleCellScope=$('scope-footer').textContent;
function showError(error){$('error').textContent=String(error.message??error);$('error').hidden=false;}
function status(text){$('status').textContent=text;}
function renderModel(){
  const meta=modelFor(config),limits=executionLimits(meta,$('backend').value),displayKey=config.model+$('backend').value;if(displayedModel===displayKey)return;displayedModel=displayKey;
  document.querySelectorAll('[data-model]').forEach(button=>button.setAttribute('aria-pressed',String(button.dataset.model===config.model)));
  $('tab-equations').hidden=!meta.editable;document.querySelector('.equations').hidden=!!meta.editable;if(activePanelModel!==config.model){selectPanel(meta.editable?'equations':'controls');activePanelModel=config.model;}
  $('model-tag').textContent=meta.name;$('model-description').textContent=meta.description;
  $('model-equations').textContent=meta.equations;$('model-reset').textContent=meta.reset;$('model-note').textContent=meta.note;
  $('model-source').hidden=!meta.source;if(meta.source){$('model-source').href=meta.source;$('model-source').textContent=meta.sourceLabel+' ↗';}
  const presets=document.querySelector('.preset-grid');presets.replaceChildren();
  for(const [key,preset] of Object.entries(meta.presets)) {const button=document.createElement('button'),small=document.createElement('small');button.dataset.preset=key;button.append(preset.label);small.textContent=preset.hint;button.append(small);presets.append(button);}
  $('model-fields').replaceChildren();const advanced=document.createElement('details');advanced.className='advanced-fields';const summary=document.createElement('summary');summary.textContent='Advanced input parameters';advanced.append(summary);
  for(const f of meta.fields){
    const field=document.createElement('div');field.className='slider-field';
    const label=document.createElement('label');label.htmlFor=f.key;const name=document.createElement('span');name.textContent=f.label;const value=document.createElement('span'),output=document.createElement('output');output.id=f.key+'-value';value.append(output,' '+f.unit);label.append(name,value);
    const input=document.createElement('input');Object.assign(input,{id:f.key,type:'range',min:f.min,max:f.max,step:f.step});input.dataset.key=f.key;field.append(label,input);(f.advanced?advanced:$('model-fields')).append(field);
  }
  if(advanced.children.length>1)$('model-fields').append(advanced);
  $('dt').min=meta.dt[0];$('dt').max=meta.dt[1];$('dt').disabled=meta.dt[0]===meta.dt[1];$('neurons').max=limits.scales.at(-1);$('neurons').disabled=!!meta.discreteScale;$('neurons').title=meta.discreteScale?'Choose a real subgraph with Population scale':'';$('synchronize').parentElement.hidden=!!meta.discreteScale;
  $('scale').replaceChildren(...limits.scales.map(n=>{const o=document.createElement('option');o.value=n;o.textContent=`${n.toLocaleString()} neurons`;return o;}));if(!meta.discreteScale){const o=document.createElement('option');o.value='custom';o.textContent='Custom count';$('scale').append(o);}
  $('scale-note').textContent=`Up to ${limits.scales.at(-1).toLocaleString()} neurons · ${(limits.maxTicks/1e6).toLocaleString()} million neuron-ticks per run. ${meta.discreteScale?'Larger circuits load on Run.':'Independent cells.'} All cells contribute to statistics; up to 12 voltage probes.`;
}
function syncControls(){renderModel();$('scale').value=executionLimits(modelFor(config),$('backend').value).scales.includes(config.neurons)?String(config.neurons):'custom';document.querySelectorAll('[data-key]').forEach(input=>{const key=input.dataset.key;if(input.type==='checkbox')input.checked=config[key];else input.value=config[key];});for(const f of modelFor(config).fields)$(f.key+'-value').textContent=Number(config[f.key].toFixed(3));}
function writeConfig(){syncControls();$('config').value=JSON.stringify(config,null,2);syncEquationEditor();updateResultContext();}
function syncEquationEditor(){
  $('equation-editor').hidden=!config.custom||panelMode!=='equations';if(!config.custom)return;
  for(const [id,key] of [['source','equations'],['parameters','parameters'],['initial','initial'],['threshold','threshold'],['reset','reset'],['refractory','refractory_ms']])$('equation-'+id).value=['parameters','initial'].includes(key)?JSON.stringify(config.custom[key],null,2):config.custom[key];
  $('model-equations').textContent=config.custom.equations;$('model-reset').textContent=config.custom.threshold+' → '+config.custom.reset.replaceAll('\n','; ');
}
function readEquationEditor(current){
  if(!current.custom)return current;
  return {...current,custom:{equations:$('equation-source').value,parameters:JSON.parse($('equation-parameters').value),initial:JSON.parse($('equation-initial').value),threshold:$('equation-threshold').value,reset:$('equation-reset').value,refractory_ms:Number($('equation-refractory').value)}};
}
// The draft and the last completed run have separate identities. Old charts are
// only visible when the complete draft (including custom equations) matches.
const normalized=value=>JSON.stringify(value,(_key,v)=>v&&typeof v==='object'&&!Array.isArray(v)?Object.fromEntries(Object.keys(v).sort().map(k=>[k,v[k]])):v);
function updateResultContext(){
  let matches=false;
  if(data){try{matches=data.backend===$('backend').value&&normalized(readEquationEditor(readConfig()))===normalized(data.config);}catch{/* incomplete edits have no matching result */}}
  const visible=matches&&!running,selected=modelFor(config).name;
  $('results-content').hidden=!visible;
  $('result-context').classList.toggle('pending',!visible);
  $('restore-result').hidden=!data||matches;$('restore-result').disabled=running;
  if(visible){
    $('result-title').textContent=`Results · ${modelFor(data).name} · ${data.backend==='webgpu'?'WebGPU / f32':'WASM / f64'}`;
    $('result-message').textContent=`${data.n.toLocaleString()} neurons · ${data.durationMs} ms · ${data.dtMs} ms timestep. These charts match the current settings.`;
    $('dirty').textContent='Results match the current settings';
  }else{
    $('result-title').textContent=running?'Simulation in progress':`${selected} · Run required`;
    $('result-message').textContent=(running?'Results will appear when this run finishes and matches the selected settings.':`Press Run to generate results for ${selected} with the current settings.`)+(data?` Last completed run: ${modelFor(data).name} · ${data.backend==='webgpu'?'WebGPU / f32':'WASM / f64'} (hidden).`:'');
    $('dirty').textContent=running?'Running · Results hidden until complete':'Current settings have no displayed results · Press Run';
  }
  $('download').disabled=!visible;$('save-chart').disabled=!visible;
  if(!visible){stopReplay();$('tooltip').hidden=true;}
  else draw();
  return visible;
}
function markEdited(){const visible=updateResultContext();document.querySelectorAll('[data-preset]').forEach(b=>b.classList.remove('selected'));if(!running)status(visible?'Current settings match the last completed run':'Selection changed · Run required');}

function readConfig(){return validateConfig(JSON.parse($('config').value),{backend:$('backend').value});}
function save(){try{localStorage.setItem(storageKey,JSON.stringify(config));}catch{}}
function draw(){if($('results-content').hidden)return;if(data){const options={cutoff:Number($('scrub').value),binMs:Number($('bin').value),probe:Number($('probe').value),previous,selected:selectedNetworkNode};renderCharts(data,options);if(data.network)renderNetwork($('network'),data,options);}else emptyCharts();}
function configureNetworkResult(){
  $('network-panel').hidden=!data.network;const timeline=document.querySelector('.timeline');
  (data.network?$('network-timeline'):document.querySelector('.raster-panel')).append(timeline);
  $('scope-footer').textContent=data.network?`FlyWire v783 · ${data.n.toLocaleString()}-neuron induced subgraph · Real connections, synthetic stimulation and reduced conductance LIF.`:singleCellScope;
  selectedNetworkNode=undefined;if(!data.network)return;
  const circuit=data.network;$('network-size').textContent=`${data.n.toLocaleString()} REAL NEURONS`;
  $('network-facts').textContent=`${data.n} neurons · ${circuit.weighted_edges.toLocaleString()} weighted edges · ${circuit.biological_contacts.toLocaleString()} biological contacts`;
  $('network-scope').textContent=`${data.synapticEvents.toLocaleString()} delivered edge events (including zero-amplitude events). Source: FlyWire Consortium / Dorkenwald et al. (2024), v783, CC BY 4.0. This induced subset omits the rest of the brain. Group positions are schematic, not anatomical.`;
  $('network-selection').textContent='Click a cell to inspect its root ID and select a recorded probe from its group. Use Replay to follow activity; pan horizontally on small screens.';
  $('group-rates').replaceChildren(...groupRates(data).map(group=>{const card=document.createElement('article'),label=document.createElement('span'),value=document.createElement('strong'),note=document.createElement('small');label.textContent=group.label;value.textContent=group.hz.toFixed(1);note.textContent=`${group.spikes.toLocaleString()} spikes`;card.style.color=groupColors[group.id];card.append(label,value,note);return card;}));
}
function stopReplay(){playing=false;cancelAnimationFrame(frameId);$('replay').textContent='▷ Replay';}
function setTime(t){$('scrub').value=t;$('time-value').textContent=`${Math.round(t)} ms`;draw();}
function setBusy(busy){running=busy;$('run').disabled=busy||!template;$('cancel').hidden=!busy;$('run').querySelector('span').textContent=busy?'Running…':'Run experiment';updateResultContext();}
async function run(){
  if(running||!template)return;
  $('error').hidden=true;
  let model,current,circuit,completed=false;const backend=$('backend').value;
  try{current=validateConfig(readEquationEditor(readConfig()),{backend});if(backend==='webgpu'&&!supportsWebGPU(current.model))throw Error('FlyWire requires WASM. WebGPU supports independent built-in and Equation Lab models.');}catch(error){showError(error);return;}
  stopReplay();config=current;writeConfig();save();setBusy(true);controller=new AbortController();$('progress').style.width='0';status('Compiling model and execution plan…');
  const start=performance.now();
  try{
    status('Loading model and selected circuit…');
    const assets=await loadAssets(current,controller.signal);circuit=assets.circuit;
    controller.signal.throwIfAborted();model=makeDraft(assets.template,current,{backend});
    const result=await runDraft(model,{signal:controller.signal,backend,config:current,batchTicks:128,onProgress:({batches,phase,fraction})=>{
      if(phase==='compile'){status('Validating model and building execution plan…');return;}
      const ratio=fraction??Math.min(1,batches*128/model.run.clocks[0].steps);$('progress').style.width=`${ratio*100}%`;status(`Running ${backend==='webgpu'?'WebGPU':'WASM'} · ${Math.round(ratio*100)}%`);
    }});
    const next={...(result.data??decodeExperiment(result.results,model)),backend,model:current.model,config:structuredClone(current),network:current.model==='flywire'?circuit:undefined},elapsed=performance.now()-start;
    previous=data?{model:data.model,backend:data.backend,durationMs:data.durationMs,meanHz:data.meanHz,n:data.n,dtMs:data.dtMs,ticks:data.ticks}:undefined;data=next;bundle=result.bundle;lastRunConfig={...current};
    const meta=presentationFor(data);$('trace-model').textContent=meta.name;$('trace-note').textContent=meta.traceNote+` ${data.record.length} probes recorded.`;
    $('aux-trace').hidden=meta.aux.length===0;$('trace-legend').replaceChildren();for(const [k,name] of ['v',...meta.aux].entries()){const dot=document.createElement('i');dot.style.background=['#7570df','#28a99f','#dc9b45','#ce6c92'][k];$('trace-legend').append(dot,' '+name+' ');}

    $('mean-rate').textContent=data.meanHz.toFixed(1);$('spike-count').textContent=data.spikes.toLocaleString();$('active-count').textContent=(data.active/data.n*100).toFixed(0);$('elapsed').textContent=Math.round(elapsed).toLocaleString();
    $('raster-note').textContent=data.spikes>60000?`Showing a uniform sample of up to 60,000 spikes. Rates and counts use all ${data.spikes.toLocaleString()} spikes.`:'Every dot is a spike. Click to select the nearest recorded probe.';
    $('population-info').textContent=`${data.n} neurons · ${data.durationMs.toFixed(0)} ms`;
    $('rate-delta').textContent=previous?`Since ${modelFor(previous).name} / ${previous.backend==='webgpu'?'f32':'f64'}: ${data.meanHz>=previous.meanHz?'+':''}${(data.meanHz-previous.meanHz).toFixed(1)} Hz${previous.durationMs===data.durationMs?' · Previous curve in gray':''}`:'spikes ÷ neurons ÷ simulated seconds';
    $('scrub').max=data.durationMs;$('scrub').disabled=false;$('probe').replaceChildren(...data.record.map(n=>{const option=document.createElement('option');option.value=n;option.textContent=`Neuron ${n}`;return option;}));
    $('replay').disabled=false;$('download').disabled=false;$('save-chart').disabled=false;$('progress').style.width='100%';
    $('identity').textContent=`${result.summary.plan_sha256.slice(0,12)} · ${data.steps.toLocaleString()} ticks · ${result.summary.numeric_profile??'reference-f64'}`;
    configureNetworkResult();setTime(data.durationMs);status(`✓ ${meta.name} · ${backend==='webgpu'?'WebGPU / f32':'WASM / f64'} · Computed locally`);
    updateResultContext();
    // These attributes are observable results for browser acceptance checks.
    document.body.dataset.backend=backend;document.body.dataset.spikes=data.spikes;document.body.dataset.meanHz=data.meanHz;document.body.dataset.plan=result.summary.plan_sha256;completed=true;
  }catch(error){status(error.name==='AbortError'?'Stopped · Previous results preserved':'Experiment could not complete');if(error.name!=='AbortError')showError(error);$('progress').style.width='0';}
  finally{setBusy(false);if(completed&&$('results-content').hidden)status('Run completed · Selected settings still require Run');}
}
$('controls').addEventListener('input',event=>{const input=event.target;if(!input.dataset.key)return;if(config.custom){try{config=readEquationEditor(config);}catch(error){showError(error);syncControls();return;}}try{config=readConfig();}catch{/* use last valid config while repairing JSON */}config[input.dataset.key]=input.type==='checkbox'?input.checked:Number(input.value);writeConfig();markEdited();});
$('backend').onchange=()=>{syncControls();markEdited();$('backend-note').textContent=$('backend').value==='webgpu'?'Experimental f32 · independent built-in and custom models. Spike times and counts can differ from WASM/f64. Failures require an explicit backend change.':'WASM · f64 reference. Supports all built-in and custom models, including FlyWire.';};
$('scale').onchange=()=>{if(config.custom){try{config=readEquationEditor(config);}catch(error){showError(error);syncControls();return;}}if($('scale').value==='custom'){$('neurons').focus();return;}config.neurons=Number($('scale').value);writeConfig();markEdited();};
$('config').addEventListener('input',()=>{try{config=readConfig();syncControls();syncEquationEditor();$('error').hidden=true;}catch{/* allow incomplete JSON */}markEdited();});
function selectPanel(mode){panelMode=mode;for(const item of ['controls','equations','code'])$(`tab-${item}`).setAttribute('aria-selected',String(mode===item));$('controls').hidden=mode!=='controls';$('code-panel').hidden=mode!=='code';$('equation-editor').hidden=mode!=='equations';}
for(const mode of ['controls','equations','code'])$(`tab-${mode}`).onclick=()=>selectPanel(mode);
document.querySelector('.preset-grid').onclick=event=>{const button=event.target.closest('[data-preset]');if(!button)return;config={...modelFor(config).presets[button.dataset.preset].config,neurons:config.neurons};writeConfig();markEdited();button.classList.add('selected');$('error').hidden=true;};
document.querySelector('.model-picker').onclick=event=>{const button=event.target.closest('[data-model]');if(!button||button.dataset.model===config.model)return;config={...models[button.dataset.model].defaults};writeConfig();markEdited();$('error').hidden=true;};
$('reset').onclick=()=>{config={...modelFor(config).defaults};writeConfig();markEdited();document.querySelector('[data-preset]').classList.add('selected');};
$('equation-editor').addEventListener('input',()=>{try{config=readEquationEditor(config);$('config').value=JSON.stringify(config,null,2);$('error').hidden=true;}catch{/* preserve incomplete JSON until Run */}markEdited();});
$('save-design').onclick=()=>{try{const c=validateConfig(readEquationEditor(readConfig()),{backend:$('backend').value});downloadFile(new Blob([JSON.stringify({schema:'neural-lab-design-v1',config:c},null,2)],{type:'application/json'}),'neural-lab-design.json');}catch(e){showError(e);}};
$('load-design').onclick=()=>$('design-file').click();
$('design-file').onchange=async()=>{const file=$('design-file').files[0];if(!file)return;try{if(file.size>131072)throw Error('Design files must be at most 128 KiB.');const value=JSON.parse(await file.text());if(value.schema!=='neural-lab-design-v1')throw Error('Choose a design saved with Save design.');config=validateConfig(value.config,{backend:$('backend').value});writeConfig();markEdited();$('error').hidden=true;}catch(e){showError(e);}finally{$('design-file').value='';}};
$('restore-result').onclick=()=>{if(!data||running)return;config=structuredClone(data.config);$('backend').value=data.backend;writeConfig();$('backend').onchange();$('error').hidden=true;status(`Restored ${modelFor(data).name} · Last completed run`);};
$('run').onclick=run;$('cancel').onclick=()=>controller?.abort();
window.addEventListener('keydown',e=>{if((e.metaKey||e.ctrlKey)&&e.key==='Enter'){e.preventDefault();run();}});
$('bin').onchange=draw;$('probe').onchange=draw;
$('scrub').oninput=()=>{stopReplay();setTime(Number($('scrub').value));};$('show-all').onclick=()=>{if(data){stopReplay();setTime(data.durationMs);}};
$('replay').onclick=()=>{if(!data)return;if(playing){stopReplay();return;}playing=true;$('replay').textContent='Ⅱ Pause';let start=performance.now();const from=Number($('scrub').value)>=data.durationMs?0:Number($('scrub').value);const animate=now=>{if(!playing)return;const t=Math.min(data.durationMs,from+(now-start)*data.durationMs/8000);setTime(t);if(t>=data.durationMs)stopReplay();else frameId=requestAnimationFrame(animate);};frameId=requestAnimationFrame(animate);};
const raster=$('raster');raster.addEventListener('pointermove',e=>{if(!data)return;const box=raster.getBoundingClientRect(),x=e.clientX-box.left,y=e.clientY-box.top;if(x<49||x>box.width-20||y<20||y>box.height-32){$('tooltip').hidden=true;return;}const t=(x-49)/(box.width-69)*data.durationMs,n=Math.round((1-(y-20)/(box.height-52))*(data.n-1));$('tooltip').textContent=`Neuron ${n} · ${t.toFixed(1)} ms · ${(data.counts[n]/(data.durationMs/1000)).toFixed(1)} Hz`;$('tooltip').style.left=`${Math.min(x+10,box.width-235)}px`;$('tooltip').style.top=`${Math.max(0,y-40)}px`;$('tooltip').hidden=false;});raster.onpointerleave=()=>$('tooltip').hidden=true;
raster.onclick=e=>{if(!data)return;const box=raster.getBoundingClientRect(),n=Math.round((1-(e.clientY-box.top-20)/(box.height-52))*(data.n-1));const nearest=data.record.reduce((a,b)=>Math.abs(a-n)<Math.abs(b-n)?a:b);$('probe').value=nearest;draw();};
function downloadFile(blob,name){const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
$('download').onclick=()=>{if(bundle)downloadFile(new Blob([JSON.stringify({...bundle,experiment_config:lastRunConfig,...(data.network?{network:data.network}:{})},null,2)],{type:'application/json'}),'neural-lab.browser.json');};
$('save-chart').onclick=()=>raster.toBlob(blob=>{if(blob)downloadFile(blob,'neural-lab-raster.png');});
$('network').onclick=event=>{if(!data?.network)return;const id=networkCellAt($('network'),data.network,event);if(id<0)return;selectedNetworkNode=id;const node=data.network.nodes[id],probe=data.record.find(i=>i===id)??data.record.find(i=>data.network.nodes[i].group===node.group);if(probe!==undefined)$('probe').value=probe;$('network-selection').textContent=`${node.cell_type||node.cell_class} · Root ID ${node.root_id} · ${data.counts[id]} spikes · ${node.sign>0?'Excitatory':node.sign<0?'Inhibitory':'Unmodelled fast action'} · Recorded group probe: ${data.network.nodes[probe].root_id}`;draw();};
new ResizeObserver(()=>draw()).observe(document.querySelector('.results-column'));
try{const saved=localStorage.getItem(storageKey);if(saved)config=validateConfig(JSON.parse(saved));}catch{}
const requestedModel=new URLSearchParams(location.search).get('model');if(Object.hasOwn(models,requestedModel)&&config.model!==requestedModel)config={...models[requestedModel].defaults};
writeConfig();emptyCharts();setBusy(false);
// Keep only the most recently loaded assets; cancelled fetches never enter the cache.
let assetCache;
async function loadAssets(current,signal){
  const key=current.model==='flywire'?`flywire-${current.neurons}`:current.model;
  if(assetCache?.key===key)return assetCache;
  const suffix=current.neurons===240?'':`-${current.neurons}`;
  const filenames=current.model==='flywire'?[`flywire${suffix}-template.json`,`flywire-circuit${suffix}.json`]:[models[current.model].template];
  const values=await Promise.all(filenames.map(async name=>{const response=await fetch('./'+name,{signal});if(!response.ok)throw Error(`Could not load ${name}. Please retry.`);return response.json();}));
  signal.throwIfAborted();const circuit=current.model==='flywire'?validateCircuit(values[1],values[0]):undefined;
  assetCache={key,template:values[0],circuit};return assetCache;
}
webgpuInfo().then(info=>{$('backend').querySelector('[value=webgpu]').disabled=!info.available;if(!info.available)$('backend-note').textContent=info.reason;}).catch(()=>{$('backend-note').textContent='Could not inspect WebGPU. WASM is available.';});
await run();
