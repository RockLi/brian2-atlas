// Node runs the exact browser WASM module. The browser Worker is tested separately.
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { pathToFileURL } from 'node:url';
import assert from 'node:assert/strict';
const [pkg, input, output, budgetText = '1'] = process.argv.slice(2);
const { initSync, BrowserExecutor } = await import(pathToFileURL(`${pkg}/b2_runner.js`));
const bytes = readFileSync(`${pkg}/b2_runner_bg.wasm`);
const module = new WebAssembly.Module(bytes);
assert.ok(!WebAssembly.Module.imports(module).some(i => i.module.startsWith('wasi')));
initSync({ module });
const bundle = JSON.parse(readFileSync(input, 'utf8'));
const plan = JSON.parse(bundle.plan_json);
for (const mutate of [p => p.logical.nodes.reverse(), p => p.logical.nodes.pop(),
    p => p.logical.nodes[0].reads.push('forged'), p => p.logical.clocks[0].steps++,
    p => p.instance_sha256 = '0'.repeat(64), p => p.strategy = 'forged',
    p => p.logical.nodes[0].dependencies.push('forged'), p => p.extra = true]) {
  const bad = structuredClone(plan); mutate(bad);
  if (JSON.stringify(bad) === JSON.stringify(plan)) continue;
  assert.throws(() => new BrowserExecutor(bundle.model_json, JSON.stringify(bad)), /plan/);
}
assert.throws(() => new BrowserExecutor('{', bundle.plan_json));
const executor = new BrowserExecutor(bundle.model_json, bundle.plan_json);
try {
  assert.equal(executor.plan_sha256, bundle.plan_sha256);
  assert.throws(() => executor.step(0), /positive/);
  if (!executor.finished) assert.throws(() => executor.results(), /completed/);
  let calls = 0;
  while (!executor.finished) {
    executor.step(Number(budgetText));
    assert.ok(++calls < 100000, 'executor did not advance');
  }
  assert.equal(executor.step(1), true);
  const results = executor.results(), events = executor.events();
  assert.deepEqual(executor.results(), results);
  mkdirSync(output, { recursive: true });
  writeFileSync(`${output}/results.bin`, results);
  if (events.length) writeFileSync(`${output}/events.bin`, events);
  writeFileSync(`${output}/summary.json`, executor.summary());
  console.log(JSON.stringify({ calls, summary: JSON.parse(executor.summary()) }));
} finally { executor.free(); }
