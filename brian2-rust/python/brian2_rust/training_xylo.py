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


class NativeXyloTrainer(NativeLIFTrainer):
    """Transport to b2-train; every sequence starts from zero integer state.

    Latent f64 weights and Adam moments stay in the checkpoint. Forward uses
    nearest-even int8 weights and the exact admitted Xylo integer transition.
    Backward uses fast-sigmoid count surrogates, detached reset and a declared
    straight-through estimator for weight/state rounding. CPU only in v1.
    """
    def __init__(self, plan, *, runner=None, weights=None, request_timeout=120):
        if plan.get('schema')!='atlas-xylo-training-plan-v1' or plan.get('backend')!='cpu':
            raise ValueError('NativeXyloTrainer requires its CPU v1 plan')
        super().__init__(plan,runner=runner,weights=weights,request_timeout=request_timeout)

    def _topology_identity(self, plan=None):
        return hashlib.sha256(canonical_bytes(self.plan if plan is None else plan)).hexdigest()

    def execute(self, inputs, labels, *, operation='train', initial=None, **kwargs):
        if initial is not None or any(v is not None for v in kwargs.values()):
            raise ValueError('NativeXyloTrainer v1 supports fresh sequences only')
        return super().execute(inputs,labels,operation=operation)
