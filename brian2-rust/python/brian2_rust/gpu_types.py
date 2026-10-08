"""Exact integer/bool storage over the shared GPU 32-bit word buffer ABI.

Floating states retain the explicit f32 arithmetic profile. Integer words are
bit-cast, never numerically converted to float; 64-bit values use low/high SoA
planes. This also keeps existing all-floating buffer offsets unchanged.
"""
import struct
import numpy as np

from .plan import PlanValidationError

CTYPES = dict(f32='float', f64='float', bool='bool', i32='int', i64='long',
              u32='uint', u64='ulong', index='uint', tick='long')
DTYPES = dict(f32='<f4', f64='<f4', bool='?', i32='<i4', i64='<i8', u32='<u4', u64='<u8')
INTEGERS = frozenset(('i32','i64','u32','u64'))


def width(dtype):
    return 2 if dtype in {'i64','u64'} else 1


def layout(symbols, count):
    result = {}; offset = 0
    for symbol in symbols:
        size = 1 if symbol.get('index_domain') == 'scalar' else count
        result[symbol['name']] = (symbol['dtype'], offset, size)
        offset += size*width(symbol['dtype'])
    return result, offset


def read(buffer, field, index='i'):
    dtype, offset, count = field
    at = f'{offset}+({index})'
    raw = f'{buffer}[{at}]'
    if dtype in {'f32','f64'}: return raw
    if dtype == 'bool': return f'(as_type<uint>({raw}) != 0u)'
    if dtype in {'i32','u32'}: return f'as_type<{CTYPES[dtype]}>({raw})'
    raw = f'(ulong(as_type<uint>({raw})) | (ulong(as_type<uint>({buffer}[{offset+count}+({index})])) << 32))'
    return f'as_type<long>({raw})' if dtype == 'i64' else raw


def write(buffer, field, index, value):
    dtype, offset, count = field
    at = f'{offset}+({index})'
    if dtype in {'f32','f64'}: return f'{buffer}[{at}] = {value};'
    if dtype == 'bool': value = f'uint(bool({value}))'
    if dtype in {'bool','i32','u32'}:
        return f'{buffer}[{at}] = as_type<float>({value});'
    return (f'{{ {CTYPES[dtype]} b2_store = {value}; '
            f'{buffer}[{at}] = as_type<float>(uint(ulong(b2_store))); '
            f'{buffer}[{offset+count}+({index})] = as_type<float>(uint(ulong(b2_store) >> 32)); }}')


def pack(values, dtype):
    if dtype in {'f32','f64'}:
        # Validated wire arrays have one fixed-width big-endian hexadecimal
        # representation. Decode the whole byte stream without making one
        # Python float per element. Mixed representations retain the scalar path.
        if (isinstance(values,(list,tuple)) and values and
                isinstance(values[0],str) and len(values[0]) in {8,16} and
                all(isinstance(v,str) and len(v)==len(values[0]) for v in values)):
            size=len(values[0])
            wire=bytes.fromhex(''.join(values))
            if len(wire)!=len(values)*(size//2):
                # Preserve rejection of noncanonical padded element strings;
                # joining must not turn two malformed elements into a value.
                raise ValueError('GPU hexadecimal float has an invalid element width')
            values=np.frombuffer(wire,dtype='>f4' if size==8 else '>f8')
        elif not (isinstance(values,np.ndarray) and values.ndim==1 and values.dtype.kind in 'biuf'):
            values = [struct.unpack('>f' if len(v)==8 else '>d', bytes.fromhex(v))[0]
                      if isinstance(v,str) else v for v in values]
        # Always own the result, including already-f32 initializer arrays. An
        # executor must never mutate the caller or a sibling candidate's input.
        with np.errstate(over='ignore'): result = np.array(values,dtype=np.float32,copy=True)
        if not np.isfinite(result).all():
            raise PlanValidationError('GPU initial value cannot be represented as finite float32')
        return result
    raw = np.asarray([int(v,16) if isinstance(v,str) else int(v) for v in values],
                     dtype=np.uint64 if width(dtype)==2 else np.uint32)
    if dtype=='bool': raw=(raw!=0).astype(np.uint32)
    if width(dtype)==2:
        return np.concatenate((raw.astype(np.uint32),(raw >> np.uint64(32)).astype(np.uint32))).view(np.float32)
    return raw.view(np.float32)


def unpack(buffer, field):
    dtype, offset, count = field
    raw = buffer[offset:offset+count]
    if dtype in {'f32','f64'}: return raw.copy()
    raw=raw.view(np.uint32)
    if dtype=='bool': return (raw != 0)
    if width(dtype)==2:
        raw=raw.astype(np.uint64) | (buffer[offset+count:offset+2*count].view(np.uint32).astype(np.uint64) << np.uint64(32))
    return raw.view(DTYPES[dtype]).copy()


def finite(values):
    return all(a.dtype.kind not in 'fc' or np.isfinite(a).all() for a in values)


def integer_literal(value, dtype):
    bits = 32 if dtype.endswith('32') else 64
    raw = int(value) % (1 << bits)
    value = f'0x{raw:x}{"u" if bits==32 else "ul"}'
    return f'as_type<{CTYPES[dtype]}>({value})' if dtype.startswith('i') else value


# B2IR integer casts truncate/wrap between widths. Signed arithmetic uses the
# corresponding unsigned word so C++ signed overflow cannot change semantics.
def wrap(value, dtype):
    unsigned = 'uint' if dtype.endswith('32') else 'ulong'
    raw = f'{unsigned}({value})'
    return f'as_type<{CTYPES[dtype]}>({raw})' if dtype.startswith('i') else raw


def _source():
    out=[]
    for dtype in sorted(INTEGERS):
        ctype=CTYPES[dtype]; unsigned='uint' if dtype.endswith('32') else 'ulong'
        bits=32 if dtype.endswith('32') else 64; signed=dtype.startswith('i')
        minimum=integer_literal(-(1 << (bits-1)),dtype) if signed else '0u'
        maximum=integer_literal((1 << (bits-int(signed)))-1,dtype)
        lower=f'-0x1p{bits-1}f' if signed else '0.0f'
        upper=f'0x1p{bits-int(signed)}f'
        out.append(f'''inline {ctype} b2_cast_{dtype}(float x) {{
    if (isnan(x)) return {ctype}(0);
    if (x <= {lower}) return {minimum};
    if (x >= {upper}) return {maximum};
    return {ctype}(x);
}}''')
        exception=f'if (a=={minimum} && b=={ctype}(-1)) return modulo ? {ctype}(0) : a;' if signed else ''
        adjust=(f'if (r != 0 && (r < 0) != (b < 0)) {{ q={wrap(f"{unsigned}(q)-1",dtype)}; r={wrap(f"{unsigned}(r)+{unsigned}(b)",dtype)}; }}' if signed else '')
        out.append(f'''inline {ctype} b2_div_{dtype}({ctype} a, {ctype} b, bool modulo, thread bool *fault) {{
    if (b==0) {{ *fault=true; return {ctype}(0); }}
    {exception}
    {ctype} q=a/b, r=a%b;
    {adjust}
    return modulo ? r : q;
}}''')
    return '\n'.join(out)+'\n'


SOURCE = _source()
SOURCE += '''inline float b2_float_div(float a, float b, bool modulo, thread bool *fault) {
    if (!isfinite(a) || !isfinite(b) || b==0.0f) { *fault=true; return 0.0f; }
    float q=floor(a/b);
    float result=modulo ? a-q*b : q;
    if (!isfinite(result)) { *fault=true; return 0.0f; }
    return result;
}
'''
