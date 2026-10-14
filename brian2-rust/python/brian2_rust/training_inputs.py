"""Conversion-time registry for native Brian TimedArray reads."""
import math
import numpy as np
from brian2.input.timedarray import TimedArray, _find_K
from brian2.core.names import Nameable
from brian2.units.fundamentalunits import check_units, get_dimensions
from brian2 import second


from .training_equations import TimedInput

class BatchTimedArray(Nameable):
    """Explicit sample-major external table, with Brian units and time grid.

    Values have shape (sample, time, physical element), or (sample, time)
    for scalar physical fields. This descriptor is only for external_state_inputs;
    signal generators remain outside the native training graph.
    """
    @check_units(dt=second)
    def __init__(self, values, dt, name='batch_timed_array*'):
        dimensions=get_dimensions(values);array=np.asarray(values,dtype=float)
        if array.ndim not in (2,3) or any(n==0 for n in array.shape) or not 1<=array.shape[0]<=4096:
            raise ValueError('BatchTimedArray requires sample x time [x physical element] values')
        if not np.all(np.isfinite(array)) or not math.isfinite(float(dt)) or float(dt)<=0:
            raise ValueError('BatchTimedArray requires finite values and a positive time grid')
        Nameable.__init__(self,name)
        self.values=array.copy();self.values.flags.writeable=False
        self.dim=dimensions;self.dt=float(dt)


def sample_major_to_timed_columns(values):
    """Pack (B,T[,N]) as (T,B*N), keeping each sample's physical columns."""
    array=np.asarray(values)
    if array.ndim not in (2,3):raise ValueError('invalid per-sample external table shape')
    return array.reshape(array.shape[0],array.shape[1],-1).transpose(1,0,2).reshape(array.shape[1],-1)


def external_state_input_vjp(source, bank_gradient):
    """Return a native frozen-bank VJP in the caller's original table shape."""
    shape=source['shape'];dtype=source['dtype'];values=np.asarray(bank_gradient,float)
    encoded_size=int(np.prod(shape))*(2 if dtype=='integer' else 1)
    if values.size!=encoded_size:raise ValueError('external input VJP bank shape mismatch')
    if dtype in ('integer','boolean'):
        if np.any(values!=0):raise ValueError('discrete external fields must have zero VJP')
        return np.zeros(shape)
    if dtype!='float':raise ValueError('invalid external input VJP dtype')
    if 'samples' not in source:return values.reshape(shape)
    b,t=shape[:2];n=1 if len(shape)==2 else shape[2]
    return values.reshape(t,b,n).transpose(1,0,2).reshape(shape)

def encode_external_state_values(values, dtype):
    """Encode discrete fields without passing an int32 value through float32.

    Signed high and unsigned low 16-bit limbs are adjacent columns in one
    frozen timed bank. Both limbs are exactly representable on every backend;
    replacing that single bank is an atomic native boundary operation.
    """
    values=np.asarray(values)
    if values.ndim not in (1,2) or not values.size or values.dtype.kind not in 'fibu':
        raise ValueError('external state values require a nonempty numeric 1D/2D table')
    if not np.all(np.isfinite(values)):
        raise ValueError('external state values must be finite')
    if dtype=='boolean':
        if not np.all((values==0)|(values==1)):
            raise ValueError('external boolean values must be zero or one')
        return values.astype(float)
    if dtype!='integer':raise ValueError('unsupported external discrete dtype')
    if not np.all((values==np.trunc(values)) & (values>=-2**31) & (values<2**31)):
        raise ValueError('external integer values must be exact int32')
    integers=values.astype(np.int64).reshape(values.shape[0],-1)
    encoded=np.empty((integers.shape[0],integers.shape[1]*2),float)
    encoded[:,0::2]=integers>>16
    encoded[:,1::2]=integers&65535
    return encoded

class TimedInputRegistry:
    def __init__(self):
        self.sources={}
        self.entries=[]

    def resolve(self, value, owner, name, allocate, *, max_values):
        if type(value) is not TimedArray:
            raise ValueError('native timed inputs require an unmodified Brian TimedArray type')
        values=np.asarray(value.values)
        if (values.ndim not in (1,2) or not 0<values.size<=max_values or any(n==0 for n in values.shape)
                or not np.all(np.isfinite(values))):
            raise ValueError('TimedArray requires finite nonempty 1D/2D values within the input budget')
        # Brian Cython/C++ emits dt as an 18-place decimal literal.
        dt=float(value.dt);emitted_dt=float(f'{dt:.18f}')
        if not math.isfinite(dt) or emitted_dt<=0:raise ValueError('invalid TimedArray dt')
        ratio=8/float(owner.clock.dt_)*dt
        if not math.isfinite(ratio) or ratio>2**53:raise ValueError("TimedArray time resolution is not representable")
        # CodeObject.owner is the group, even when a pathway runs on another clock.
        k=_find_K(float(owner.clock.dt_),dt);epsilon=emitted_dt/k
        if k>2**53 or not math.isfinite(epsilon) or epsilon<=0:
            raise ValueError('TimedArray time resolution is not representable')
        identity=id(value)
        if identity not in self.sources:
            bank=allocate(value.name,values.reshape(-1))
            self.sources[identity]=bank
            self.entries.append(dict(name=value.name,bank=bank,shape=list(values.shape),dt_seconds=dt,aliases=[]))
        bank=self.sources[identity];entry=next(e for e in self.entries if e['bank']==bank)
        alias=dict(object=owner.name,variable=name)
        if alias not in entry['aliases']:entry['aliases'].append(alias)
        return TimedInput(bank,values.shape[0],1 if values.ndim==1 else values.shape[1],epsilon,k,values.ndim)
