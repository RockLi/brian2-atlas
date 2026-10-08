"""Build and test the Atlas backend from this checkout, recording its identity."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

CORE = ('b2ir_v1', 'compact_validation', 'probe', 'device', 'native', 'resource_limits')
CPU = ('test_artifact.py', 'test_b2ir_v1.py', 'test_binary_topology.py', 'test_clock_tick_continuation.py', 'test_coba_hh.py', 'test_codegen_sums.py', 'test_compact_validation.py', 'test_device.py', 'test_encoded_array.py', 'test_execution_plan.py', 'test_instance_encoding.py', 'test_monitor_observables.py', 'test_native.py', 'test_nmda_deterministic_core.py', 'test_population.py', 'test_presynaptic_write_summed_only.py', 'test_probe.py', 'test_rate_monitor.py', 'test_refractory.py', 'test_resource_limits.py', 'test_results_bounded.py', 'test_results_times.py', 'test_rng_poisson.py', 'test_string_connection.py', 'test_summed_cache.py', 'test_synapses.py', 'test_weighted_binomial.py')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=('core', 'cpu', 'all'), default='core')
    parser.add_argument('--test', action='append', help='Test filename, optionally followed by ::nodeid')
    parser.add_argument('--runner', type=Path, help='Use this existing runner instead of building one')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--timeout', type=int, default=300)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    backend = root / 'brian2-rust'
    output = (args.output or root / 'build/atlas-checks/cpu').resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.environ['RUSTUP_TOOLCHAIN'] = '1.98.1'
    if args.runner:
        runner = args.runner.expanduser().resolve()
    else:
        target = Path(os.environ.get('CARGO_TARGET_DIR', backend / 'target')).expanduser()
        if not target.is_absolute():
            target = backend / target
        subprocess.run(['cargo', 'build', '--release', '--locked', '--bin', 'b2-runner',
                        '--manifest-path', str(backend / 'Cargo.toml'),
                        '--target-dir', str(target)], cwd=backend, check=True)
        runner = target / 'release' / ('b2-runner.exe' if os.name == 'nt' else 'b2-runner')
    if not runner.is_file():
        raise FileNotFoundError(runner)
    # Frozen tests also address Cargo's conventional checkout path directly.
    conventional = backend / 'target/release' / runner.name
    if conventional.resolve() != runner.resolve():
        if conventional.exists():
            if conventional.read_bytes() != runner.read_bytes():
                raise RuntimeError(f'Existing test runner differs from the selected runner: {conventional}')
        else:
            conventional.parent.mkdir(parents=True, exist_ok=True)
            conventional.symlink_to(runner)
    os.environ['B2_RUNNER'] = str(runner)
    source_paths = [str(root), str(backend / 'python')]
    # Fresh Python checkpoint workers must import the same tested source tree.
    os.environ['PYTHONPATH'] = os.pathsep.join(source_paths)
    sys.path[:0] = source_paths
    os.chdir(root)
    import brian2
    import brian2_rust
    import pytest
    from brian2.tests import PreferencePlugin
    assert Path(brian2.__file__).resolve().is_relative_to(root / 'brian2')
    assert Path(brian2_rust.__file__).resolve().is_relative_to(backend / 'python')
    brian2.prefs.codegen.target = 'numpy'
    brian2.prefs.codegen.runtime.cython.cache_dir = str(output / 'cython')
    if args.test:
        selections = args.test
    elif args.suite == 'cpu':
        selections = list(CPU)
    elif args.suite == 'core':
        selections = ['test_' + name + '.py' for name in CORE]
    else:
        selections = sorted(path.name for path in (backend / 'tests').glob('test_*.py'))
    for selection in selections:
        filename = selection.split('::')[0]
        if Path(filename).name != filename or not (backend / 'tests' / filename).is_file():
            parser.error(f'Unknown backend test: {selection}')
    files = [str(backend / 'tests' / name) for name in selections]
    command = ['-q', '--maxfail=8', '--timeout=' + str(args.timeout),
               '--junitxml=' + str(output / 'results.xml'),
               '--basetemp=' + str(output / 'tmp'),
               '-o', 'cache_dir=' + str(output / 'pytest-cache'), *files]
    meta = {'schema': 'atlas-backend-gate-v1', 'started_unix': time.time(),
            'source_root': str(root), 'python': sys.version,
            'gate_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'runner': str(runner), 'runner_sha256': hashlib.sha256(runner.read_bytes()).hexdigest(),
            'rustc': subprocess.check_output(['rustc', '--version'], text=True).strip(),
            'tests': selections, 'pytest_args': command}
    try:
        meta['git_head'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
        meta['git_diff_sha256'] = hashlib.sha256(subprocess.check_output(['git', 'diff', 'HEAD'])).hexdigest()
    except subprocess.CalledProcessError:
        meta['git_head'] = None
    code = pytest.main(command, plugins=[PreferencePlugin(dict(brian2.prefs))])
    meta.update(exit_code=int(code), ended_unix=time.time())
    (output / 'validation.json').write_text(json.dumps(meta, indent=2) + '\n')
    return int(code)


if __name__ == '__main__':
    raise SystemExit(main())
