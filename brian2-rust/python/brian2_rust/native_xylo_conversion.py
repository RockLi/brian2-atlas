"""Exact fresh-inference export of native integer Xylo training artifacts."""
import copy
import hashlib
from pathlib import Path
import numpy as np
from .protocol import canonical_bytes

SCHEMA='atlas.native-xylo-frozen.v1'
RECURRENT_SCHEMA='atlas.native-xylo-frozen.v2'

def digest(value):return hashlib.sha256(canonical_bytes(value)).hexdigest()

def require(ok,message):
    if not ok:raise ValueError('Native Xylo export: '+message)


def freeze_native_xylo(trainer):
    """Freeze the same integer weights already used in every training forward.

    No new rounding approximation, clipping or dynamics adaptation is added.
    The source's latent weights are rounded once by the declared native rule.
    """
    from .training_xylo import NativeXyloTrainer
    require(isinstance(trainer,NativeXyloTrainer),'requires NativeXyloTrainer')
    require(trainer.state is not None and trainer.state['step']>0,'requires trained weights')
    # The native evaluator validates the entire plan/state, without committing.
    trainer.evaluate(np.zeros((1,1,trainer.plan['sizes'][0]),dtype=int),[0])
    f=dict(schema=RECURRENT_SCHEMA if trainer.plan['schema']=='atlas-xylo-training-plan-v2' else SCHEMA,plan=copy.deepcopy(trainer.plan),
        target_weights=[np.rint(np.asarray(w)*trainer.plan['weight_scale']).astype(int).tolist()
                        for w in trainer.state['weights']],
        initial_state='zero',quantization='nearest-even int8 already used during training',
        provenance=dict(training_state_sha256=digest(trainer.state),optimizer_step=trainer.state['step'],
            training_runner_sha256=hashlib.sha256(trainer.runner.read_bytes()).hexdigest()))
    f['sha256']=digest(f);validate_frozen(f);return f


def validate_frozen(f):
    require(f.get('schema') in (SCHEMA,RECURRENT_SCHEMA) and f.get('sha256')==digest({k:v for k,v in f.items() if k!='sha256'}),
            'frozen model integrity mismatch')
    from .training_xylo import xylo_training_plan
    p=f['plan'];sizes=p['sizes'];recurrent=f['schema']==RECURRENT_SCHEMA
    expected=set(xylo_training_plan([1,1,1]))|({'recurrent','readout_start'} if recurrent else set())
    require(set(p)==expected and p['schema']==('atlas-xylo-training-plan-v2' if recurrent else 'atlas-xylo-training-plan-v1')
        and p['backend']=='cpu' and p['synaptic_shift']==p['membrane_shift']==1 and p['detach_reset'] is True,
        'unsupported training dynamics')
    require(len(sizes)==3 and all(type(n) is int and 1<=n<=limit for n,limit in zip(sizes,(16,64,8))),
            'sizes must fit 16/64/8')
    require(f['initial_state']=='zero','sequence must start from zero')
    if recurrent:
        require(p['recurrent'] is True and type(p['readout_start']) is int and 0<=p['readout_start']<128,
                'invalid recurrent readout window')
    require(len(p['threshold'])==2 and all(type(t) is int and 1<=t<=32767 for t in p['threshold']),
            'thresholds must be integer 1..32767')
    require(np.isfinite(p['logit_scale']) and 0<p['logit_scale']<=100,'invalid logit scale')
    shapes=list(zip(sizes,sizes[1:]))+([(sizes[1],sizes[1])] if recurrent else [])
    require(len(f['target_weights'])==len(shapes),'invalid weight banks')
    for index,(bank,(n,m)) in enumerate(zip(f['target_weights'],shapes)):
        w=np.asarray(bank)
        require(w.shape==(n*m,) and np.isfinite(w).all() and (w==np.rint(w)).all()
                and ((w>=-128)&(w<=127)).all() and (index==2 or np.any(w)),'nonzero input/readout int8 projections required')
    if recurrent:
        require(np.all(np.count_nonzero(np.asarray(f['target_weights'][2]).reshape(sizes[1],sizes[1]),axis=1)<=63),
                'recurrent fanout must be at most 63 per hidden neuron')
    return sizes


def validate_inputs(f,inputs):
    sizes=validate_frozen(f);x=np.asarray(inputs)
    require(x.ndim==3 and len(x)>0 and 1<=x.shape[1]<=128 and x.shape[2]==sizes[0]
            and np.isfinite(x).all() and ((x==0)|(x==1)).all(),'requires binary batch x 1..128 ticks x input')
    return x.astype(np.uint8)


def build_native_xylo_network(f,raster):
    from .xylo_network import build_xylo_network
    x=validate_inputs(f,np.asarray(raster)[None])[0]
    return build_xylo_network(f['plan']['sizes'],f['target_weights'],f['plan']['threshold'],x)


def export_native_xylo(f,directory,*,steps,runner,target='xylosim'):
    """Export to stock XyloSim or the explicit custom SpiNNaker2 C host kernel."""
    import brian2 as b
    from .export import export_network
    from brian2_atlas.neuromorphic import export_ir
    sizes=validate_frozen(f)
    require(target in ('xylosim','spinnaker2'),'target must be xylosim or spinnaker2')
    profile='xylo-audio2-temporal-v1' if target=='xylosim' else 'spinnaker2-xylo-integer-v1'
    require(type(steps) is int and f['plan'].get('readout_start',0)<steps<=128,'duration must be 1..128 ticks and include readout')
    directory=Path(directory);require(not directory.exists(),'destination already exists')
    previous=b.prefs.codegen.loop_invariant_optimisations
    try:
        b.prefs.codegen.loop_invariant_optimisations=False
        network=build_native_xylo_network(f,np.zeros((steps,sizes[0]),dtype=np.uint8))
        directory.mkdir(parents=True)
        export_network(network,steps*b.ms,directory/'template.json')
        bundle=export_ir(directory/'template.json',directory/'converted',target=target,
                         profile=profile,runner=runner)
    finally:b.prefs.codegen.loop_invariant_optimisations=previous
    bundle['native_training_bridge']=dict(schema=f['schema'],frozen_sha256=f['sha256'],sizes=sizes,
        logit_scale=f['plan']['logit_scale'],fresh_state_per_sample=True,source_is_zero_input_template=True,
        dynamics_equivalent=True,quantization=f['quantization'])
    if f['schema']==RECURRENT_SCHEMA:
        bundle['native_training_bridge'].update(readout_start=f['plan']['readout_start'],stateful_chunks=True)
    (directory/'frozen.json').write_bytes(canonical_bytes(f))
    (directory/'converted/bundle.json').write_bytes(canonical_bytes(bundle))
    return bundle
