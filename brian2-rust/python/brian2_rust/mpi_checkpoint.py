"""Strict boundary-snapshot compatibility for MPI Device checkpoints.

The frontend snapshot is committed only after the launcher has successfully
collected every owner and finalized all ranks. There is no live rank checkpoint.
"""
import hashlib
import platform
from pathlib import Path
import subprocess

from .protocol import canonical_bytes


def contract(model, options, runner):
    import brian2
    import numpy
    from .mpi_gpu import backend_policy
    ranks = options.get('ranks', 2)
    backends = backend_policy(ranks, options.get('rank_backends'),
                              options.get('numeric_mode', 'reference-f64'))
    # Duration and recording window are activation data, not network identity.
    def definition(value):
        if isinstance(value, dict):
            return {key: definition(item) for key, item in value.items()
                    if key not in {'steps', 'window_steps'}}
        if isinstance(value, list):
            return [definition(item) for item in value]
        return value
    topology = [{key: syn[key] for key in ('topology', 'source', 'target') if key in syn}
                | {'pathways': [{key: path[key] for key in
                                ('name', 'kind', 'event', 'delay_ticks')}
                               for path in syn['pathways']]}
                for syn in model['instance']['synapses']]
    package = Path(__file__).parent
    digest = hashlib.sha256()
    for path in sorted([*package.glob('*.py'), *package.glob('mpi_runtime/*')]):
        if path.is_file():
            digest.update(path.relative_to(package).as_posix().encode())
            digest.update(path.read_bytes())
    return {
        'schema': 'b2-mpi-boundary-checkpoint-v1',
        'network': hashlib.sha256(canonical_bytes(definition(model['definition']))).hexdigest(),
        'topology': hashlib.sha256(canonical_bytes(topology)).hexdigest(),
        'ranks': ranks,
        'partition': 'contiguous-per-population-target-owned',
        'rank_backends': list(backends) or ['cpu'] * ranks,
        'numeric_mode': options.get('numeric_mode', 'reference-f64'),
        'runtime_source': digest.hexdigest(),
        'validator_binary': hashlib.sha256(Path(runner).read_bytes()).hexdigest(),
        'compiler': subprocess.check_output(['rustc', '--version'], text=True).strip(),
        'host_abi': [platform.system(), platform.machine()],
        'brian2': brian2.__version__, 'numpy': numpy.__version__,
    }


def verify(saved, current):
    from .export import require
    require(isinstance(saved, dict), 'MPI checkpoint lacks compatibility metadata')
    differing = sorted(key for key in set(saved) | set(current)
                       if saved.get(key) != current.get(key))
    require(not differing, 'MPI checkpoint configuration mismatch: ' + ', '.join(differing))
