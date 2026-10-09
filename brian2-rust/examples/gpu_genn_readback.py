"""Version-pinned host gather for the static sparse GeNN comparison only.

The original getter is retained as a bitwise bootstrap cross-check. No device
pointer, host allocation or mutable value survives a load/unload boundary.
"""
from contextlib import contextmanager
import hashlib
from pathlib import Path
import numpy as np

EXPECTED_SOURCE_SHA256='38639fe0219efc076b87d9470473c34295492865ca5b5b55bdc97da88f83ab43'


class SparseRows:
    def __init__(self,group):
        self.group=group
        if not group.connections_set or group._any_ccu_references:
            raise ValueError('Gather requires static manually supplied connectivity')
        self.rows=np.asarray(group.row_lengths,dtype=np.int64).copy()
        self.stride=int(group.max_connections);self.size=int(group.weight_update_var_size)
        if self.rows.ndim!=1 or self.stride<=0 or self.size!=len(self.rows)*self.stride or np.any(self.rows<0) or np.any(self.rows>self.stride):
            raise ValueError('Unsupported GeNN sparse row layout')
        starts=np.cumsum(self.rows)-self.rows
        self.indices=np.repeat(np.arange(len(self.rows))*self.stride-starts,self.rows)+np.arange(int(self.rows.sum()))
        self.indices.setflags(write=False)

    def validate(self):
        g=self.group
        if not g.connections_set or g._any_ccu_references or int(g.max_connections)!=self.stride or int(g.weight_update_var_size)!=self.size or not np.array_equal(g.row_lengths,self.rows):
            raise ValueError('Static sparse layout changed after gather preparation')

    def gather(self,variable):
        # This private host view is inspected in the exact pinned GeNN source;
        # reject unknown packages and layouts rather than guessing its ABI.
        view=variable._view
        if not isinstance(view,np.ndarray) or view.ndim not in (1,2) or view.size!=self.size or view.shape[-1]!=self.size or view.dtype!=np.dtype(np.float32):
            raise ValueError('Expected one loaded float32 sparse batch')
        return np.take(view,self.indices,axis=-1)


def prepare(context):
    import pygenn.model_preprocessor as module
    digest=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    if digest!=EXPECTED_SOURCE_SHA256:
        raise ValueError('Unrecognized GeNN sparse variable implementation')
    from gpu_stdp_compare import FIELDS
    plans=[];variables={}
    for syn,edges in context['groups']:
        plan=SparseRows(syn)
        if len(plan.indices)!=len(edges):raise ValueError('Sparse edge count does not match adapter topology')
        plans.append(plan)
        for key in FIELDS:
            variable=syn.vars[key]
            if type(variable) is not module.SynapseVariable:
                raise ValueError('Unsupported GeNN synapse variable type')
            variables[id(variable)]=(variable,plan)
    context['host_gather']=dict(plans=plans,variables=variables,variable_class=module.SynapseVariable)
    return dict(label='Version-pinned static sparse host gather; corrected GeNN GPU program unchanged',
        source_sha256=digest,groups=len(plans),variables=len(variables),
        cached_index_bytes=sum(p.indices.nbytes+p.rows.nbytes for p in plans),
        row_slices_per_public_readback=sum(len(p.rows) for p in plans)*len(FIELDS))


@contextmanager
def gather_values(state,verify=False):
    cls=state['variable_class'];original=cls.values;calls={}
    for plan in state['plans']:plan.validate()
    def values(variable):
        entry=state['variables'].get(id(variable))
        if entry is None:return original.fget(variable)
        expected,plan=entry
        if variable is not expected:raise RuntimeError('Synapse variable identity changed')
        result=plan.gather(variable)
        if verify:
            reference=original.fget(variable)
            if result.shape!=reference.shape or result.dtype!=reference.dtype or result.tobytes()!=reference.tobytes():
                raise RuntimeError('Host gather differs from GeNN public sparse values')
        calls[id(variable)]=calls.get(id(variable),0)+1
        return result
    cls.values=property(values,original.fset,original.fdel,original.__doc__)
    try:yield calls
    finally:cls.values=original


def replay(context,steps,verify=False):
    from gpu_stdp_compare import genn_replay
    state=context['host_gather']
    with gather_values(state,verify) as calls:result=genn_replay(context,steps)
    if set(calls)!=set(state['variables']) or any(n!=1 for n in calls.values()):
        raise RuntimeError('Gather did not cover each expected synapse state exactly once')
    context['last_gather']=dict(calls=sum(calls.values()),public_getter_cross_checked=verify,
        view_lifetime='fresh loaded host view per access; only row indices persist')
    return result
