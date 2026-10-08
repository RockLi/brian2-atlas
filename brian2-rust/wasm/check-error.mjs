import { readFileSync } from 'node:fs';
import { pathToFileURL } from 'node:url';
import assert from 'node:assert/strict';
const [pkg, input] = process.argv.slice(2);
const { initSync, BrowserExecutor } = await import(pathToFileURL(`${pkg}/b2_runner.js`));
initSync({ module: readFileSync(`${pkg}/b2_runner_bg.wasm`) });
const bundle = JSON.parse(readFileSync(input, 'utf8'));
const executor = new BrowserExecutor(bundle.model_json, bundle.plan_json);
try {
  assert.throws(() => executor.step(1), /linked variable index/);
  assert.throws(() => executor.step(1), /execution failed/);
  assert.throws(() => executor.results(), /completed execution/);
} finally { executor.free(); }
