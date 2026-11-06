import {loadFile,run} from './import.js';
import {runBundle} from './runtime.js';
const $=id=>document.getElementById(id),assert=(condition,message)=>{if(!condition)throw Error(message);};
$('acceptance').onclick=async()=>{
 $('acceptance').disabled=true;const report={checks:[]};
 try{
  const text=await (await fetch('./import-example.browser.json')).text(),file=new File([text],'exported-by-brian2.browser.json',{type:'application/json'});
  await loadFile(file);assert(!$('run').disabled,'Run enabled after loading');assert(!document.body.dataset.plan,'Loading must not execute the model');report.checks.push('File object loaded without execution');
  await run();assert($('error').hidden,'Example run failed: '+$('error').textContent);assert(document.body.dataset.populations==='2','Both populations decoded');const plan=document.body.dataset.plan,spikes=document.body.dataset.spikes;report.spikes=Number(spikes);report.plan=plan;
  $('population').value='1';$('population').dispatchEvent(new Event('change'));assert($('trace-legend').textContent==='vm','Non-v SI variable must be selectable');assert($('mean-rate').textContent!=='—','SpikeMonitor statistics missing');report.checks.push('Two populations, SI trace and spike statistics');
  const bad=JSON.parse(text);bad.plan_sha256='0'.repeat(64);await loadFile(new File([JSON.stringify(bad)],'corrupted.browser.json'));await run();assert(!$('error').hidden,'Corrupt plan was accepted');assert(document.body.dataset.plan===plan&&document.body.dataset.spikes===spikes,'Failure replaced previous results');report.checks.push('Tampered plan rejected; prior results preserved');
  await loadFile(new File(['not JSON'],'invalid.browser.json'));assert(!$('error').hidden,'Invalid JSON accepted');report.checks.push('Invalid JSON rejected');
  const controller=new AbortController();let progressed=false;try{await runBundle(JSON.parse(text),{imported:true,signal:controller.signal,onProgress:()=>{progressed=true;controller.abort();}});throw Error('Cancellation did not stop');}catch(e){assert(e.name==='AbortError','Unexpected cancellation result');}assert(progressed,'Cancellation must follow actual progress');report.checks.push('Worker cancellation after progress');
  await loadFile(file);await run();assert($('error').hidden,'Recovery run failed');report.checks.push('Successful reload and replay after failures');report.status='passed';
 }catch(e){report.status='failed';report.error=String(e);}finally{$('acceptance-report').textContent=JSON.stringify(report,null,2);$('acceptance').disabled=false;}
};
