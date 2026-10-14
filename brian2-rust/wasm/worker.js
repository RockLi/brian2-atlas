import {inspectBundle} from './imported-model.js';
import init, { BrowserExecutor, compile_model } from './pkg/b2_runner.js';
import {compileWebGPU,executeWebGPU,decodeWebGPU,WEBGPU_PROFILE} from './webgpu.js';

self.onmessage = async ({ data }) => {
  let executor;
  try {
    let { bundle, draft, batchTicks = 128 } = data;
    await init();
    if (draft !== undefined) {
      self.postMessage({type: 'progress', phase:'compile', batches:0});
      bundle=JSON.parse(compile_model(draft));
    }
    if (bundle?.schema !== 'b2-wasm-bundle-v0' || typeof bundle.model_json !== 'string' || typeof bundle.plan_json !== 'string') throw new Error('Unsupported WASM bundle');
    if (!Number.isInteger(batchTicks) || batchTicks < 1 || batchTicks > 1000000)
      throw new Error('batchTicks must be an integer in [1, 1000000]');
    if(data.backend==='webgpu'){
      const model=JSON.parse(bundle.model_json),program=compileWebGPU(model,data.config);
      const output=await executeWebGPU(program,{onProgress:progress=>self.postMessage({type:'progress',...progress})});
      const result=decodeWebGPU(program,output);
      const identity=JSON.stringify({profile:WEBGPU_PROFILE,model:bundle.model_json,wgsl:program.source});
      const hash=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',new TextEncoder().encode(identity))),b=>b.toString(16).padStart(2,'0')).join('');
      const exported={schema:'b2-webgpu-lab-v1',numeric_profile:WEBGPU_PROFILE,execution_sha256:hash,reference_bundle:bundle,wgsl:program.source,device:output.device,limits:output.limits};
      self.postMessage({type:'result',data:result,bundle:exported,summary:{plan_sha256:hash,numeric_profile:WEBGPU_PROFILE,backend:'webgpu'}});
      return;
    }
    if(data.backend&&data.backend!=='wasm')throw Error('Unknown browser backend');
    if(data.imported){const info=inspectBundle(bundle);batchTicks=Math.min(batchTicks,Math.max(1,Math.floor(100000/info.neurons)));}
    executor = new BrowserExecutor(bundle.model_json, bundle.plan_json);
    if (executor.plan_sha256 !== bundle.plan_sha256) throw new Error('Bundle plan hash mismatch');
    let batches = 0;
    while (!executor.finished) {
      executor.step(batchTicks);
      if(executor.spike_count>4000000)throw Error('WASM spike output exceeds the 4 million event budget. Reduce neuron count, duration or drive.');
      self.postMessage({ type: 'progress', batches: ++batches, ticksPerBatch: batchTicks });
      await new Promise(resolve => setTimeout(resolve, 0));
    }
    const results = executor.results();
    const events = executor.events();
    const summary = JSON.parse(executor.summary());
    self.postMessage({ type: 'result', results, events, summary, bundle }, [results.buffer, events.buffer]);
  } catch (error) {
    self.postMessage({ type: 'error', message: String(error) });
  } finally {
    executor?.free();
  }
};
