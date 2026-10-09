/** Execute a verified bundle in a dedicated worker. Abort terminates the worker,
 * including model compilation; each run owns its memory and cannot affect another.
 */
export function runBundle(bundle, { signal, onProgress, batchTicks = 128, draft, backend = "wasm", config, imported = false } = {}) {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) { reject(new DOMException('Aborted', 'AbortError')); return; }
    const worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
    const cleanup = () => { worker.terminate(); signal?.removeEventListener('abort', abort); };
    const abort = () => { cleanup(); reject(new DOMException('Aborted', 'AbortError')); };
    signal?.addEventListener('abort', abort, { once: true });
    worker.onerror = event => { cleanup(); reject(new Error(event.message)); };
    worker.onmessage = ({ data }) => {
      if (data.type === 'progress') {
        try { onProgress?.(data); } catch (error) { cleanup(); reject(error); }
        return;
      }
      cleanup();
      if (data.type === 'error') reject(new Error(data.message));
      else resolve(data);
    };
    try { worker.postMessage({ bundle, batchTicks, draft, backend, config, imported }); }
    catch (error) { cleanup(); reject(error); }
  });
}

export function runDraft(draft, options = {}) {
  return runBundle(undefined, {...options, draft: typeof draft === "string" ? draft : JSON.stringify(draft)});
}
