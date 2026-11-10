import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {models} from './models.js';
import {makeDraft,validateConfig,bits} from './experiment.js';
import {compileWebGPU,checkWebGPULimits,decodeWebGPU,executeWebGPU} from './webgpu.js';
const root=process.argv[2],templates=Object.fromEntries(['adaptive_lif','izhikevich','hodgkin_huxley'].map(id=>[id,JSON.parse(readFileSync(`${root}/${models[id].template}`))]));
for(const id of Object.keys(templates)){
 const c={...models[id].defaults,neurons:models[id].gpuScales.at(-1)};
 assert.throws(()=>validateConfig(c),'Large GPU populations must not bypass WASM limits');
 const model=makeDraft(templates[id],c,{backend:'webgpu'}),program=compileWebGPU(model,c);
 assert.equal(program.n,c.neurons);assert.equal(program.record.length,12);
 checkWebGPULimits(program,{maxBufferSize:268435456,maxStorageBufferBindingSize:134217728,maxComputeInvocationsPerWorkgroup:256,maxStorageBuffersPerShaderStage:8,maxComputeWorkgroupsPerDimension:65535});
 assert.throws(()=>checkWebGPULimits(program,{maxBufferSize:4,maxStorageBufferBindingSize:4,maxComputeInvocationsPerWorkgroup:64,maxStorageBuffersPerShaderStage:8,maxComputeWorkgroupsPerDimension:65535}));
 assert.throws(()=>validateConfig({...c,neurons:c.neurons+1},{backend:'webgpu'}));
}
const c={...models.adaptive_lif.defaults,neurons:8,duration_ms:64,dt_ms:1},draft=makeDraft(templates.adaptive_lif,c),p=compileWebGPU(draft,c);
assert.throws(()=>compileWebGPU({},models.flywire.defaults),/FlyWire/);
for(const mutate of [m=>m.definition.populations[0].linked_variables.push({}),m=>m.definition.synapses.push({}),m=>m.definition.schedule.nodes[0].operation='event_source',m=>m.definition.populations[0].code_objects[0].vector[0].value={op:'rand'},m=>m.instance.populations[0].initial_state.v[0]=bits(1e300)]){
 const invalid=structuredClone(draft);mutate(invalid);assert.throws(()=>compileWebGPU(invalid,c));
}
await assert.rejects(executeWebGPU(p,{chunkTicks:0}),/1–256/);
// Bitmaps must preserve boundary bits (31/32/63), full counts and time/cell order.
const bitmap=new Uint32Array(8*2);bitmap[0]=0x80000001;bitmap[1]=0x80000001;bitmap[2]=1;
const output={state:new Float32Array(p.initial.length),bitmap,traces:new Float32Array(p.sizes[3]/4)},decoded=decodeWebGPU(p,output);
assert.deepEqual(Array.from(decoded.ticks),[0,0,31,32,63]);assert.deepEqual(Array.from(decoded.indices),[0,1,0,0,0]);assert.equal(decoded.counts[0],4);assert.equal(decoded.spikes,5);
output.state[0]=Infinity;assert.throws(()=>decodeWebGPU(p,output),/non-finite/);
console.log('WebGPU CPU-side checks passed: bounds, eligibility, f32 overflow, invalid chunk size, bitmap boundaries, ordering and full counts.');
