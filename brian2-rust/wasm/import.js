import {runBundle} from './runtime.js';
import {MAX_IMPORT_BYTES,parseBundleText,decodeImported} from './imported-model.js';
import {renderCharts,emptyCharts} from './charts.js';
const $=id=>document.getElementById(id);
let loaded,completed,data,controller,running=false,loading=false,loadGeneration=0,playing=false,frame;
const status=text=>{$('status').textContent=text;},error=e=>{$('error').textContent=String(e.message??e);$('error').hidden=false;};
function busy(){ $('run').disabled=running||loading||!loaded;$('cancel').hidden=!running;$('example').disabled=running||loading;$('bundle-file').disabled=running;$('run').querySelector('span').textContent=running?'Running…':'Run imported model'; }
function stopReplay(){playing=false;cancelAnimationFrame(frame);$('replay').textContent='▷ Replay';}
export async function loadFile(file){if(running){error(Error('Stop the current run before loading another model.'));return;}if(file.size>MAX_IMPORT_BYTES){error(Error('Model bundles must be at most 32 MiB.'));return;}await loadText(()=>file.text(),file.name);}
async function loadText(read,name){
 const generation=++loadGeneration;loading=true;busy();$('error').hidden=true;status('Reading model metadata…');
 try{const text=await read();if(generation!==loadGeneration)return;const candidate=parseBundleText(text);loaded={...candidate,name};
  $('loaded-file').textContent=name;$('model-info').textContent=`${candidate.info.length} populations · ${candidate.neurons.toLocaleString()} neurons\n${candidate.edgeCount.toLocaleString()} edges · ${candidate.runDurationMs.toLocaleString()} ms run\n${candidate.info.map(p=>p.name+': '+p.count+' cells, '+p.dtMs+' ms step').join('\n')}`;
  $('dirty').textContent='Loaded · Press Run to verify and execute';status('Model loaded · Execution plan not yet verified');document.querySelector('.results-column').classList.toggle('stale',!!completed);
 }catch(e){error(e);status('Could not load this model · Previous selection preserved');}finally{if(generation===loadGeneration){loading=false;busy();}}
}
$('bundle-file').onchange=()=>{const file=$('bundle-file').files[0];if(file)loadFile(file);$('bundle-file').value='';};
for(const event of ['dragover','dragleave','drop'])$('drop-zone').addEventListener(event,e=>{e.preventDefault();$('drop-zone').classList.toggle('dragging',event==='dragover');if(event==='drop'){if(e.dataTransfer.files.length===1)loadFile(e.dataTransfer.files[0]);else error(Error('Drop one .browser.json file.'));}});
$('example').onclick=()=>loadText(async()=>{const r=await fetch('./import-example.browser.json');if(!r.ok)throw Error('Example bundle is unavailable. Rebuild the browser assets.');return r.text();},'Two connected Brian2 populations');
export async function run(){
 if(running||loading||!loaded)return;const current=loaded;running=true;busy();stopReplay();controller=new AbortController();$('error').hidden=true;$('progress').style.width='0';status('Verifying model and execution plan…');const start=performance.now();
 try{
  const result=await runBundle(current.bundle,{signal:controller.signal,imported:true,batchTicks:128,onProgress:({batches,ticksPerBatch=128})=>{status(`Running WASM · batch ${batches}`);$('progress').style.width=`${Math.min(95,batches*ticksPerBatch/current.info.reduce((sum,p)=>sum+p.steps,0)*100)}%`;}});
  const decoded=decodeImported(result.results,current.model);completed={...decoded,result,loaded:current,elapsed:performance.now()-start};
  $('population').replaceChildren(...decoded.populations.map((p,k)=>{const option=document.createElement('option');option.value=k;option.textContent=`${p.name} · ${p.n.toLocaleString()} cells`;return option;}));$('population').disabled=false;
  for(const id of ['download-model','download-results','download-events','download-summary','save-chart'])$(id).disabled=false;
  $('identity').textContent=`${result.summary.plan_sha256.slice(0,12)} · verified WASM plan · reference-f64`;$('progress').style.width='100%';$('dirty').textContent='Verified · Results match the loaded model';document.querySelector('.results-column').classList.remove('stale');status(`✓ ${current.name} · Computed locally`);selectPopulation();
  document.body.dataset.plan=result.summary.plan_sha256;document.body.dataset.spikes=decoded.totalSpikes;document.body.dataset.populations=decoded.populations.length;
 }catch(e){status(e.name==='AbortError'?'Stopped · Previous results preserved':'Execution failed · Previous results preserved');if(e.name!=='AbortError')error(e);$('progress').style.width='0';}
 finally{running=false;busy();}
}
function selectPopulation(){
 stopReplay();if(!completed)return;const p=completed.populations[Number($('population').value)],names=Object.keys(p.trace);data={...p,backend:'wasm',model:'imported'};
 $('variable').replaceChildren(...names.map(name=>{const o=document.createElement('option');o.value=name;o.textContent=name;return o;}));if(names.includes('v'))$('variable').value='v';$('variable').disabled=!names.length;
 $('probe').replaceChildren(...p.record.map(i=>{const o=document.createElement('option');o.value=i;o.textContent=`Neuron ${i}`;return o;}));$('probe').disabled=!p.record.length;
 $('mean-rate').textContent=p.spikesRecorded?p.meanHz.toFixed(1):'—';$('spike-count').textContent=p.spikesRecorded?p.spikes.toLocaleString():'—';$('active-count').textContent=p.spikesRecorded?(100*p.active/p.n).toFixed(0):'—';$('elapsed').textContent=Math.round(completed.elapsed).toLocaleString();
 $('population-info').textContent=`${p.n.toLocaleString()} neurons · recorded window ${p.durationMs.toLocaleString()} ms`;$('rate-delta').textContent=p.spikesRecorded?'Selected population · recorded window':'No SpikeMonitor exported for this population';
 $('raster-note').textContent=p.spikesRecorded?(p.spikes>60000?`Drawing up to 60,000 sampled dots; statistics include all ${p.spikes.toLocaleString()} recorded spikes.`:'Every dot is a recorded spike.'): 'No SpikeMonitor was exported. Zero recorded events does not establish that this population was silent.';
 $('window-note').textContent=`Viewing ${p.name} · time axis relative to recorded window [${p.startMs.toLocaleString()}, ${(p.startMs+p.durationMs).toLocaleString()}) ms · ${p.dtMs} ms timestep. ${completed.synapticEvents.toLocaleString()} synaptic deliveries across the whole network. ${p.omitted.length?'Not plotted (integers outside the exact JavaScript range): '+p.omitted.join(', ')+'. Full values remain in results.bin.':''}`;
 $('trace-model').textContent=p.name;$('aux-trace').hidden=true;$('trace-note').textContent=p.record.length&&names.length?'Choose a trace variable and recorded neuron. Values use exported SI units or dimensionless model coordinates.':'No plottable StateMonitor data was exported for this population.';
 document.querySelector('.trace-panel').hidden=!(p.record.length&&names.length);$('spike-charts').hidden=!p.spikesRecorded;
 $('scrub').step=p.dtMs;$('scrub').max=p.durationMs;$('scrub').value=p.durationMs;$('scrub').disabled=false;$('replay').disabled=false;setTime(p.durationMs);
}
function unit(dim){const common={'2,1,-3,-1,0,0,0':'V','0,0,0,1,0,0,0':'A','0,0,1,0,0,0,0':'s','-2,-1,3,2,0,0,0':'S','-2,-1,4,2,0,0,0':'F'};if(dim&&common[dim.join()])return common[dim.join()];const labels=['m','kg','s','A','K','mol','cd'];if(!dim||dim.every(x=>x===0))return 'model units';return dim.flatMap((x,i)=>x===0?[]:[labels[i]+(x===1?'':`^${x}`)]).join(' · ');}
function draw(){if(!data){emptyCharts();return;}const primary=$('variable').value;data.presentation={name:data.name,primary,aux:[],threshold:null,voltageRange:[0,0],voltageUnit:`${primary||'state'} / ${unit(data.symbols[primary]?.dimensions)}`,auxUnit:''};$('trace-legend').textContent=primary||'No recorded trace';renderCharts(data,{cutoff:Number($('scrub').value),binMs:Number($('bin').value),probe:Number($('probe').value)});}
function setTime(t){$('scrub').value=t;$('time-value').textContent=`${Number(t.toFixed(3))} ms`;draw();}
$('population').onchange=selectPopulation;$('variable').onchange=draw;$('probe').onchange=draw;$('bin').onchange=draw;
$('run').onclick=run;$('cancel').onclick=()=>controller?.abort();window.addEventListener('keydown',e=>{if((e.metaKey||e.ctrlKey)&&e.key==='Enter'){e.preventDefault();run();}});
$('scrub').oninput=()=>{stopReplay();setTime(Number($('scrub').value));};$('show-all').onclick=()=>{if(data){stopReplay();setTime(data.durationMs);}};
$('replay').onclick=()=>{if(!data)return;if(playing){stopReplay();return;}playing=true;$('replay').textContent='Ⅱ Pause';const start=performance.now(),from=Number($('scrub').value)>=data.durationMs?0:Number($('scrub').value);const next=now=>{if(!playing)return;const t=Math.min(data.durationMs,from+(now-start)*data.durationMs/8000);setTime(t);if(t===data.durationMs)stopReplay();else frame=requestAnimationFrame(next);};frame=requestAnimationFrame(next);};
$('raster').onclick=e=>{if(!data?.record.length)return;const rect=$('raster').getBoundingClientRect(),n=Math.round((1-(e.clientY-rect.top-20)/(rect.height-52))*(data.n-1));$('probe').value=data.record.reduce((a,b)=>Math.abs(a-n)<Math.abs(b-n)?a:b);draw();};
function download(content,name,type){const url=URL.createObjectURL(new Blob([content],{type})),a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
$('download-model').onclick=()=>{if(completed)download(JSON.stringify(completed.loaded.bundle),'network.browser.json','application/json');};
$('download-results').onclick=()=>{if(completed)download(completed.result.results,'results.bin','application/octet-stream');};
$('download-events').onclick=()=>{if(completed)download(completed.result.events,'events.bin','application/octet-stream');};
$('download-summary').onclick=()=>{if(completed)download(JSON.stringify(completed.result.summary,null,2),'summary.json','application/json');};
$('save-chart').onclick=()=>$('raster').toBlob(blob=>{if(blob)download(blob,'imported-raster.png','image/png');});
new ResizeObserver(draw).observe(document.querySelector('.results-column'));emptyCharts();busy();status('Choose a Brian2 model bundle');
