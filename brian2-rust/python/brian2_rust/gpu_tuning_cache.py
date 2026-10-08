"""Opt-in bounded decisions for exact-input replays; never cache model results."""
from collections import OrderedDict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

from .gpu_autotune import (DEFAULT_MAX_BUFFER_BYTES, POLICIES,
                           observable_fingerprint, run_with_buffer_budget, tune)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def file_digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def input_key(model, runner, options, *, source_root=None):
    """Rehash complete inputs and implementation, without retaining their arrays.

    Opaque native functions can depend on files outside the declared model and
    generated include set. They may still autotune, but cannot cache decisions.
    Binary CSR content is independently rechecked by the Device before this call.
    """
    if any(f['body'] is None for f in model['definition']['functions']):
        return None
    root = Path(source_root) if source_root is not None else Path(__file__).parent
    sources = {str(p.relative_to(root)): file_digest(p) for p in sorted(root.rglob('*'))
               if p.is_file() and not p.name.startswith('._')
               and p.suffix in {'.py', '.m', '.h'}}
    return digest(dict(model=model, runner=file_digest(runner), options=options,
                       sources=sources, python=sys.version, numpy=np.__version__,
                       platform=platform.platform(), machine=platform.machine()))


def executor_context(executor):
    """Bind a choice to the actual compiler/runtime context of its fresh executor."""
    context = dict(device=executor.device_name, dag_execution=executor.dag_execution)
    if hasattr(executor, 'cp'):
        context.update(backend='cuda', device_ordinal=executor.device.id,
                       architecture=executor.architecture, nvcc=executor.nvcc_version,
                       cupy=executor.cp.__version__,
                       driver=executor.cp.cuda.runtime.driverGetVersion(),
                       compiler=executor.compilation_report['context_sha256'])
    else:
        context.update(backend='metal', bridge=executor._bridge_identity)
    # Normalize tuples and avoid references to executor-owned mutable metadata.
    return json.loads(json.dumps(context))


class TuningCache:
    """Eight LRU metadata records per Device; no arrays, executors or binaries."""
    capacity = 8

    def __init__(self):
        self._entries = OrderedDict()

    def __len__(self):
        return len(self._entries)

    def clear(self):
        self._entries.clear()

    def discard(self, key):
        self._entries.pop(key, None)

    def get(self, key):
        # Only successful publication changes LRU order.
        value = self._entries.get(key)
        return deepcopy(value) if value is not None else None

    def publish(self, key, value):
        if key is None:
            return
        self._entries[key] = deepcopy(value)
        self._entries.move_to_end(key)
        while len(self._entries) > self.capacity:
            self._entries.popitem(last=False)


def resolve(factories, directory, cache, key, *, context_for=executor_context,
            started=None, max_buffer_bytes=DEFAULT_MAX_BUFFER_BYTES):
    """Return executor/result/report and an uncommitted cache record.

    factories(subdirectory) returns fresh construction/planning callbacks for
    each attempt. A rejected cached executor is closed before full calibration.
    The caller publishes the record only after all frontend result loading has
    succeeded. Exceptions/interrupts never transfer executor ownership.
    """
    started = time.perf_counter() if started is None else started
    directory.mkdir(parents=True, exist_ok=True)
    record = cache.get(key) if key is not None else None
    report = dict(schema='b2-gpu-tuning-cache-v1', status='running',
                  cache=dict(status='miss', input_sha256=key,
                             reason='input not cached' if key is not None else 'opaque native function'),
                  replay=None)
    executor = None

    def save():
        report['total_seconds'] = time.perf_counter() - started
        (directory / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')

    try:
        if record is not None:
            try:
                name = record['calibration']['selected']
                policy = next(p for p in POLICIES if p[0] == name)
                make, plan_for = factories(directory / 'cached')
                plan = plan_for(*policy)
                if digest(plan.to_dict()) != record['plan']:
                    raise ValueError('cached physical plan changed')
                executor = make(*policy)
                if digest(executor.plan.to_dict()) != record['plan']:
                    raise ValueError('cached executor does not match prepared plan')
                if context_for(executor) != record['context']:
                    raise ValueError('cached compiler/device context changed')
                start = time.perf_counter()
                result = run_with_buffer_budget(executor,max_buffer_bytes)
                elapsed = time.perf_counter() - start
                if not np.isfinite(elapsed) or elapsed <= 0:
                    raise RuntimeError("Invalid cached replay duration")
                fingerprint = observable_fingerprint(result)
                report['replay'] = dict(seconds=elapsed, observable_sha256=fingerprint,
                                        plan_sha256=executor.plan.sha256,
                                        compilation=deepcopy(executor.compilation_report))
                if fingerprint != record['calibration']['reference_observable_sha256']:
                    raise ValueError('cached complete result differs from calibrated baseline')
                report['cache'].update(status='hit', reason='exact input, plan, context and complete result verified')
            except Exception as error:
                cache.discard(key)
                report['cache'].update(status='invalidated', reason=type(error).__name__ + ': ' + str(error))
                record = None
                if executor is not None:
                    rejected, executor = executor, None
                    rejected.close()
        if record is None:
            make, plan_for = factories(directory / 'calibration')
            executor, result, calibration = tune(
                make, directory / 'calibration', plan_for=plan_for,
                max_buffer_bytes=max_buffer_bytes)
            record = dict(calibration=deepcopy(calibration), plan=digest(executor.plan.to_dict()),
                          context=context_for(executor) if key is not None else None)
        report.update(status='passed', selected=record['calibration']['selected'],
                      selected_plan_sha256=executor.plan.sha256,
                      reference_observable_sha256=record['calibration']['reference_observable_sha256'],
                      calibration=deepcopy(record['calibration']))
        save()
        winner, executor = executor, None
        return winner, result, report, record
    except BaseException as error:
        report.update(status='failed', error=type(error).__name__ + ': ' + str(error))
        save()
        raise
    finally:
        if executor is not None:
            executor.close()
