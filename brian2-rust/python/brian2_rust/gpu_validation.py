"""One fresh semantic validation per GPU activation, independent model snapshots."""
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path

from .gpu_initialization import prepare_model
from .plan import PlanValidationError, validate_model
from .protocol import canonical_bytes, CURRENT_SCHEMA


def runner_identity(runner):
    from ._runtime import executable_path
    path = executable_path("b2-runner", runner).resolve()
    with path.open('rb') as stream:
        return str(path),hashlib.file_digest(stream,'sha256').hexdigest()


@dataclass(frozen=True,slots=True,init=False)
class ValidatedActivation:
    """Private activation-local wire snapshot; never a persisted validation cache.

    A canonical immutable input snapshot is independently checked by Rust. Each
    planner/executor receives a fresh JSON decode, followed by the existing host
    initializer. No candidate shares a mutable model, array or runtime state.
    The supplied live input and validator binary must still match before use.
    """
    _input_sha256: str
    _validated_wire: bytes
    _runner: tuple[str,str]

    def __init__(self,model,*,runner=None):
        wire=canonical_bytes(model)
        identity=runner_identity(runner)
        # Current-schema validation already owns a deepcopy. Avoid decoding the
        # same large wire snapshot before that copy, and prove it checked exactly
        # these immutable bytes. Legacy migration uses its captured wire input.
        current_schema=model.get('schema')==CURRENT_SCHEMA
        current=validate_model(model if current_schema else json.loads(wire),runner=runner)
        validated_wire=canonical_bytes(current)
        if current_schema and validated_wire!=wire:
            raise PlanValidationError('GPU input changed during activation validation')
        if runner_identity(runner)!=identity:
            raise PlanValidationError('GPU validator changed during activation validation')
        object.__setattr__(self,'_runner',identity)
        object.__setattr__(self,'_input_sha256',hashlib.sha256(wire).hexdigest())
        object.__setattr__(self,'_validated_wire',wire if current_schema else validated_wire)

    def snapshot(self,model,*,runner=None):
        if (hashlib.sha256(canonical_bytes(model)).hexdigest()!=self._input_sha256 or
                runner_identity(runner)!=self._runner):
            raise PlanValidationError('GPU activation input or validator changed after validation')
        return prepare_model(json.loads(self._validated_wire),runner=runner)

    def plan(self,backend,*,runner=None,**options):
        # Planning describes this immutable validated activation, never a later
        # mutable caller input. Construction separately checks the live input.
        if runner_identity(runner)!=self._runner:
            raise PlanValidationError('GPU validator changed after validation')
        current=prepare_model(json.loads(self._validated_wire),runner=runner)
        if backend=='metal':
            from .metal import _derive_metal_plan as derive
        elif backend=='cuda':
            from .cuda import _derive_cuda_plan as derive
        else:
            raise ValueError('Unknown GPU activation backend')
        return derive(current,**options)


def executor_model(model,*,runner=None,activation=None):
    """Ordinary public construction still performs fresh independent validation."""
    if activation is None:
        return prepare_model(validate_model(model,runner=runner),runner=runner)
    if type(activation) is not ValidatedActivation:
        raise PlanValidationError('GPU validation requires an activation-local snapshot')
    return activation.snapshot(model,runner=runner)
