"""Native CPU Xylo integer forward, surrogate BPTT and quantization-aware weights."""
import hashlib
from .protocol import canonical_bytes
from .training import NativeLIFTrainer


def xylo_training_plan(sizes, *, thresholds=(32,32), weight_scale=32., learning_rate=.003,
                       logit_scale=8., seed=20261014, max_tape_bytes=256*1024**2):
    return dict(schema='atlas-xylo-training-plan-v1',backend='cpu',sizes=list(sizes),
        threshold=list(thresholds),weight_scale=weight_scale,synaptic_shift=1,membrane_shift=1,
        detach_reset=True,surrogate=dict(kind='fast_sigmoid',slope=3.,scale=1.),
        optimizer=dict(kind='adam',learning_rate=learning_rate,beta1=.9,beta2=.999,epsilon=1e-8),
        logit_scale=logit_scale,max_tape_bytes=max_tape_bytes,seed=seed)


def xylo_recurrent_training_plan(sizes, *, readout_start=0, **kwargs):
    """CPU recurrent v2; fresh BPTT and explicit chunked evaluation state."""
    plan=xylo_training_plan(sizes,**kwargs)
    plan.update(schema='atlas-xylo-training-plan-v2',recurrent=True,readout_start=readout_start)
    return plan


class NativeXyloTrainer(NativeLIFTrainer):
    """Transport to b2-train; fresh training and explicit v2 evaluation state.

    Latent f64 weights and Adam moments stay in the checkpoint. Forward uses
    nearest-even int8 weights and the exact admitted Xylo integer transition.
    Backward uses fast-sigmoid count surrogates, detached reset and a declared
    straight-through estimator for weight/state rounding. CPU only in v1.
    """
    def __init__(self, plan, *, runner=None, weights=None, request_timeout=120):
        if plan.get('schema') not in ('atlas-xylo-training-plan-v1','atlas-xylo-training-plan-v2') or plan.get('backend')!='cpu':
            raise ValueError('NativeXyloTrainer requires its CPU v1/v2 plan')
        super().__init__(plan,runner=runner,weights=weights,request_timeout=request_timeout)

    def _topology_identity(self, plan=None):
        return hashlib.sha256(canonical_bytes(self.plan if plan is None else plan)).hexdigest()

    def execute(self, inputs, labels, *, operation='train', initial=None, **kwargs):
        if any(v is not None for v in kwargs.values()):
            raise ValueError('Xylo state supplies its own clock; other continuation options are unsupported')
        if initial is not None and (self.plan['schema']!='atlas-xylo-training-plan-v2'
                or operation!='evaluate' or not isinstance(initial,dict)):
            raise ValueError('Xylo training requires fresh sequences; explicit v2 state is evaluation-only')
        return super().execute(inputs,labels,operation=operation,initial=initial)

    def evaluate_chunks(self, inputs, labels, *, chunk_steps):
        """Evaluate one <=128-tick sequence in chunks without committing state.

        The returned logits/counts include the full declared readout window.
        This is inference state carry, not truncated training or optimizer carry.
        """
        import numpy as np
        if self.plan['schema']!='atlas-xylo-training-plan-v2':
            raise ValueError('chunked evaluation requires the recurrent v2 plan')
        x=np.asarray(inputs)
        if type(chunk_steps) is not int or chunk_steps<1 or x.ndim!=3 or not 1<=x.shape[1]<=128:
            raise ValueError('requires positive chunk_steps and a batch of 1..128 ticks')
        traces={k:[] for k in ('spikes','currents','membranes')};initial=None
        for start in range(0,x.shape[1],chunk_steps):
            result=self.evaluate(x[:,start:start+chunk_steps],labels,initial=initial)
            initial=result['continuation']
            for key in traces:traces[key].append(np.asarray(result[key]))
        result.update({key:np.concatenate(parts,axis=1).tolist() for key,parts in traces.items()})
        result['chunk_steps']=chunk_steps
        return result
