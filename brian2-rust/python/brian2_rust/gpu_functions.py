"""Native scalar Function adapters for the explicit GPU float32 profile.

Portable bodies keep their established lowering. A native-only Function uses
the selected backend's self-contained source verbatim inside a private namespace.
It is trusted code; neither source hashing nor a signature proves purity.
"""
import hashlib
import re

from . import gpu_types as gt
from .plan import PlanValidationError


def namespace(function):
    return 'b2_native_' + hashlib.sha256(function['name'].encode()).hexdigest()


def validate_functions(model, backend):
    if backend not in {'metal', 'cuda'}:
        raise PlanValidationError('Unsupported native GPU Function backend')
    for function in model['definition']['functions']:
        effects = function['effects']
        if (effects['stateful'] or effects['rng'] or not effects['deterministic']
                or not effects['thread_safe']):
            raise PlanValidationError('GPU requires pure deterministic thread-safe Functions')
        if function['body'] is None:
            if backend not in function['backend_implementations']:
                raise PlanValidationError(f"Function {function['name']} requires a {backend} implementation or portable body")
            dtypes = [a['dtype'] for a in function['arguments']] + [function['return_dtype']]
            if any(t not in {'f64', 'i64', 'bool'} for t in dtypes):
                raise PlanValidationError('Native GPU Function requires scalar storage dtypes')


def require_cpu_mirror(model):
    if any(f['body'] is None for f in model['definition']['functions']):
        raise PlanValidationError('CPU f32 mirror requires portable Function bodies; native GPU source has no CPU mirror')


def source_block(model, backend):
    """Return the exact private native declarations, without rewriting source."""
    parts = []
    for function in model['definition']['functions']:
        if function['body'] is not None:
            continue
        native = function['backend_implementations'][backend]
        # The native text is never processed by Metal-to-CUDA or CPU rewriting.
        types = [gt.CTYPES[a['dtype']] for a in function['arguments']]
        result = gt.CTYPES[function['return_dtype']]
        arguments = ', '.join(f'{t} a{i}' for i, t in enumerate(types))
        values = ', '.join(f'a{i}' for i in range(len(types)))
        qualifier = '__device__ inline' if backend == 'cuda' else 'inline'
        parts.append(f'''namespace {namespace(function)} {{
{native['source']}
template<class A, class B> struct b2_signature {{ enum {{ value=0 }}; }};
template<class A> struct b2_signature<A,A> {{ enum {{ value=1 }}; }};
static_assert(b2_signature<decltype({native['symbol']}), {result}({', '.join(types)})>::value,
              "Native GPU Function signature must match the float32 profile");
{qualifier} {result} b2_invoke({arguments}) {{ return {native['symbol']}({values}); }}
}}
''')
    return '\n'.join(parts)


def inject_sources(source, model, backend):
    block = source_block(model, backend)
    if not block:
        return source
    marker = 'kernel void ' if backend == 'metal' else 'extern "C" __global__ void '
    at = source.index(marker)
    # Fusion lowers earlier kernel entries into inline stage helpers. Native
    # namespaces must precede those callers as well as the final kernel. Match
    # the whole declaration line so CUDA's __device__ qualifier stays attached.
    helper = re.search(r'(?m)^(?:__device__ )?inline void ', source)
    if helper is not None:
        at = min(at, helper.start())
    return source[:at] + block + source[at:]
