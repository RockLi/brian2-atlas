"""Native CPU/Metal/CUDA LIF training; Python transports plans and snapshots.

No Python tensor/autodiff runtime performs the forward, backward or optimizer.
The v1 dense and v2 projection plans are independent of the B2IR simulation ABI.
"""
import copy
from contextlib import contextmanager
import errno
import hashlib
import json
import math
import os
import signal
import sys
from pathlib import Path
import tempfile
import subprocess
import time

from .protocol import canonical_bytes


@contextmanager
def _request_directory():
    """Remove an owned request after late MPI error writers have settled."""
    temporary = tempfile.TemporaryDirectory(prefix='b2-train-')
    try:
        yield temporary.name
    finally:
        primary_error = sys.exc_info()[1]
        for attempt in range(7):
            try:
                temporary.cleanup()
                break
            except OSError as cleanup_error:
                # An aborted launcher can exit while a peer still atomically
                # writes its error. Retry cleanup only, never the native call.
                if cleanup_error.errno == errno.ENOTEMPTY and attempt < 6:
                    time.sleep(min(.01 * 2**attempt, .2))
                    continue
                if primary_error is not None:
                    if hasattr(primary_error, 'add_note'):
                        primary_error.add_note(
                            f'Native request cleanup failed; retained {temporary.name}: {cleanup_error}')
                    break
                raise


def lif_training_plan(sizes, *, backend='cpu', beta=0.9, threshold=1.0, reset='subtract',
                      detach_reset=True, surrogate_slope=5.0, surrogate_scale=1.0,
                      optimizer='adam', learning_rate=0.01, seed=1,
                      trainable=None, masks=None, logit_scale=5.0,
                      max_tape_bytes=64*1024**2, tbptt_window=None, projections=None, mpi_ranks=None, equations=None, threshold_parameters=None,
                      state_equations=None, state_resets=None, refractory=None, threshold_per_neuron=None, clock=None, noise_streams=None):
    """Source-major weights; all thresholds precede synapses and reset.

    inputs are batch × time × input; each row is a deterministic input spike
    amplitude. Output logits are mean output spikes times logit_scale; loss is
    batch-mean cross entropy. Refractory requires an explicit v4 specification.
    beta is the discrete Euler decay (1-dt/tau), per layer or shared.
    Passing projections selects v2: masks/trainable/weights are per projection
    parameter bank, with tied IDs sharing one weight and one optimizer slot.
    """
    sizes=list(sizes); layers=len(sizes)-1
    if not 3 <= len(sizes) <= 17 or any(type(n) is not int or not 0<n<=65536 for n in sizes):
        raise ValueError('training needs bounded integer input, hidden and output sizes')
    counts=[a*b for a,b in zip(sizes,sizes[1:])]
    topology_bytes=0
    if projections is not None:
        from .training_graph import validate_projections
        counts,topology_bytes=validate_projections(sizes,projections,allow_empty=equations is not None or state_equations is not None)
    if type(max_tape_bytes) is not int or max_tape_bytes<=0 or sum(counts)*48+topology_bytes>max_tape_bytes:
        raise ValueError('native training parameter budget exceeded before mask allocation')
    plan=dict(schema='b2-lif-training-plan-v1', backend=backend, sizes=sizes,
                beta=[beta]*layers if isinstance(beta,(int,float)) else list(beta),
                threshold=[threshold]*layers if isinstance(threshold,(int,float)) else list(threshold),
                reset=reset, detach_reset=detach_reset,
                surrogate=dict(kind='fast_sigmoid', slope=surrogate_slope, scale=surrogate_scale),
                optimizer=dict(kind=optimizer, learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8),
                trainable=[True]*len(counts) if trainable is None else list(trainable),
                masks=[[1.0]*n for n in counts] if masks is None else masks,
                seed=seed, logit_scale=logit_scale, max_tape_bytes=max_tape_bytes,
                tbptt_window=tbptt_window)
    if projections is not None:
        plan.update(schema='b2-lif-training-plan-v2',projections=copy.deepcopy(projections))
    if equations is not None:
        if projections is None:raise ValueError('equation training requires explicit projections')
        plan.update(schema='b2-equation-training-plan-v3',equations=copy.deepcopy(equations))
    if state_equations is not None or state_resets is not None:
        if equations is not None or projections is None or state_equations is None or state_resets is None:
            raise ValueError('v4 requires state equations, resets and projections without scalar equations')
        plan.update(schema='b2-state-training-plan-v4',state_equations=copy.deepcopy(state_equations),
                    state_resets=copy.deepcopy(state_resets))
    if threshold_parameters is not None:
        if equations is None and state_equations is None:raise ValueError('trainable thresholds require equations')
        plan['threshold_parameters']=copy.deepcopy(threshold_parameters)
    if refractory is not None:
        if state_equations is None:raise ValueError('refractory requires v4 state equations')
        plan['refractory']=copy.deepcopy(refractory)
    if threshold_per_neuron is not None:
        if state_equations is None or threshold_parameters is None:
            raise ValueError('per-neuron thresholds require v4 and parameter references')
        plan['threshold_per_neuron']=copy.deepcopy(threshold_per_neuron)
    if mpi_ranks is not None: plan['mpi_ranks']=mpi_ranks
    if clock is not None:
        if equations is None and state_equations is None:raise ValueError('clock requires equations')
        plan['clock']=copy.deepcopy(clock)
    if noise_streams is not None:
        if (equations is None and state_equations is None) or clock is None:raise ValueError('noise requires equations and a clock')
        plan['noise_streams']=copy.deepcopy(noise_streams)
    return plan


class NativeLIFTrainer:
    """Isolated native runner and optimizer checkpoint state.

    request_timeout bounds each native invocation in seconds (default 120).
    Larger weak-gradient replays may need an explicitly larger finite limit.
    The limit is a local transport setting, independent of saved model state.
    """
    def __init__(self, plan, *, runner=None, weights=None, request_timeout=120):
        if (type(request_timeout) not in (int,float) or
            not math.isfinite(request_timeout) or request_timeout<=0):
            raise ValueError('request_timeout must be finite positive seconds')
        self.request_timeout=request_timeout
        if plan.get('mpi_ranks') is not None:
            if (type(plan['mpi_ranks']) is not int or not 2<=plan['mpi_ranks']<=256
                or (plan.get('backend')!='cpu' and not (plan.get('backend') in ('metal','cuda') and plan.get('projections') is not None))):
                raise ValueError('target-owned MPI BPTT requires 2..256 CPU or GPU projection ranks')
        self.plan=copy.deepcopy(plan)
        self._topology_contract=self._topology_identity()
        from ._runtime import executable_path
        self.runner = executable_path("b2-train", runner).resolve()
        if not self.runner.is_file():
            raise FileNotFoundError('Build the isolated b2-train binary and pass runner= or B2_TRAIN_RUNNER')
        self.state=None
        self.neuron_state=None
        self.elapsed_ticks=0
        self.clock_tick=0
        self.clock_state=None
        self.poisson_state=None
        self.noise_sequence=None
        self.next_noise_sequence=0
        self.last_result=None
        self._metal_directory=None
        self._metal_library=None
        self._metal_hash=None
        if plan.get('backend') in ('metal','cuda'):
            if plan['backend']=='cuda':
                from .training_cuda import build
            else:
                from .training_metal import build
            self._metal_directory=tempfile.TemporaryDirectory(prefix='b2-train-metal-')
            self._metal_library=build(self._metal_directory.name)
            self._metal_hash=hashlib.sha256(self._metal_library.read_bytes()).hexdigest()
        self._mpi_directory=None
        self._mpi_library=None
        self._mpi_hash=None
        self._mpi_environment={}
        if plan.get('mpi_ranks') is not None:
            from .training_mpi import build
            self._mpi_directory=tempfile.TemporaryDirectory(prefix='b2-train-mpi-')
            self._mpi_library=build(self._mpi_directory.name)
            self._mpi_hash=hashlib.sha256(self._mpi_library.read_bytes()).hexdigest()
            import platform
            version=subprocess.run(['mpiexec','--version'],capture_output=True,text=True,check=True).stdout
            if platform.system()=='Darwin' and ('HYDRA' in version or 'MPICH' in version):
                # MPICH 5/macOS sockets provider can hang even in a minimal
                # MPI_Init/Finalize program. Scope the tested provider to this
                # launch and preserve an explicit caller override.
                self._mpi_environment['FI_PROVIDER']=os.environ.get('FI_PROVIDER','tcp')
        if weights is not None:
            values=[list(map(float,w)) for w in weights]
            self.state=dict(weights=values, first_moment=[[0.0]*len(w) for w in values],
                            second_moment=[[0.0]*len(w) for w in values],step=0,rng=plan['seed'])

    def execute(self, inputs, labels, *, operation='train', initial=None, start_tick=None, noise_sequence=None, clock_state=None, poisson_state=None):
        """Run a sequence; time starts at the plan's snapshot unless specified.

        ``initial='carry'`` uses both the last committed neuron state and its
        clock tick. Explicit states default to tick zero; pass ``start_tick``
        for manual continuation, together with ``poisson_state`` and
        ``clock_state`` from the prior
        result for exact multi-clock run boundaries. Only successful training
        commits the cursor.
        Stochastic sequences use an independent counter: fresh training chooses
        the next sequence, carry retains the old one. Pass noise_sequence for
        explicit replay (also required when manually continuing a noise path).
        """
        # The native engine rejects unknown plan keys/operations, validates
        # shapes, and checks the complete tape budget before allocating it.
        self._check_topology()
        def lists(value):
            return value.tolist() if hasattr(value,'tolist') else value
        if isinstance(initial,str):
            if initial!='carry' or self.neuron_state is None:
                raise ValueError('initial carry requires a previous native neuron state')
            if start_tick is not None:raise ValueError('carry already supplies its clock tick')
            start_tick=self.clock_tick
            if clock_state is not None:raise ValueError("carry already supplies its clock state")
            clock_state=self.clock_state
            if poisson_state is not None:raise ValueError("carry already supplies its Poisson draw state")
            poisson_state=self.poisson_state
            if noise_sequence is not None:raise ValueError('carry already supplies its noise sequence')
            noise_sequence=self.noise_sequence
            initial=self.neuron_state
        if start_tick is None:start_tick=0
        if type(start_tick) is not int or not 0<=start_tick<=2**53:
            raise ValueError('start_tick requires a bounded nonnegative integer')
        if noise_sequence is None:noise_sequence=self.next_noise_sequence if self.plan.get('noise_streams') is not None else 0
        if type(noise_sequence) is not int or not 0<=noise_sequence<2**64-1:
            raise ValueError('noise_sequence requires a bounded nonnegative integer')
        request=dict(plan=self.plan,state=self.state,operation=operation,
                     inputs=lists(inputs),labels=lists(labels),initial=lists(initial))
        if self.plan.get('clock') is not None or start_tick:
            request['start_tick']=start_tick
        if self.plan.get('noise_streams') is not None or noise_sequence:
            request['noise_sequence']=noise_sequence
        if clock_state is not None:request['clock_state']=clock_state
        if poisson_state is not None:request['poisson_state']=poisson_state
        result=self._run_request(request)
        candidate=None;contract=None
        if operation=='train' and 'updated_dynamic' in result:
            candidate=copy.deepcopy(self.plan);candidate['dynamic']=result['updated_dynamic']
            contract=self._topology_identity(candidate)
        self.state=result['state']; self.last_result=result
        if operation=='train':
            if candidate is not None:self.plan=candidate;self._topology_contract=contract
            self.neuron_state=result.get('final_state',result['final_membrane'])
            self.elapsed_ticks+=len(request['inputs'][0])
            self.clock_tick=result.get('final_tick',0)
            self.clock_state=copy.deepcopy(result.get('clock_state'))
            self.poisson_state=copy.deepcopy(result.get('poisson_state'))
            self.noise_sequence=result.get('noise_sequence')
            if self.noise_sequence is not None:
                self.next_noise_sequence=max(self.next_noise_sequence,self.noise_sequence+1)
        return result

    def _run_request(self, request):
        """Transport only; callers commit validated native results atomically."""
        with _request_directory() as directory:
            source=Path(directory)/'request.json'; output=Path(directory)/'result.json'
            source.write_bytes(canonical_bytes(request))
            env=os.environ.copy()
            env['B2_TRAIN_ERROR_RESULT']='1'
            if self._metal_library is not None:
                if hashlib.sha256(self._metal_library.read_bytes()).hexdigest()!=self._metal_hash:
                    raise ValueError('native Metal library integrity mismatch')
                env['B2_TRAIN_CUDA_LIB' if self.plan['backend']=='cuda' else 'B2_TRAIN_METAL_LIB']=str(self._metal_library)
            command=[str(self.runner),str(source),str(output)]
            if self._mpi_library is not None:
                if hashlib.sha256(self._mpi_library.read_bytes()).hexdigest()!=self._mpi_hash:
                    raise ValueError('native MPI library integrity mismatch')
                env['B2_TRAIN_MPI_LIB']=str(self._mpi_library)
                env.update(self._mpi_environment)
                command=['mpiexec','-n',str(self.plan['mpi_ranks']),*command]
            with subprocess.Popen(command, stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                                  text=True,env=env,stdin=subprocess.DEVNULL,start_new_session=True) as process:
                try:
                    stdout,stderr=process.communicate(timeout=self.request_timeout)
                except (subprocess.TimeoutExpired,KeyboardInterrupt):
                    try:os.killpg(process.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                    process.communicate()
                    raise
            if process.returncode:
                message=None
                try:
                    if output.stat().st_size<=65536:
                        error=json.loads(output.read_bytes())
                        if (isinstance(error,dict) and set(error)=={'schema','message'}
                            and error['schema']=='b2-native-training-error-v1'
                            and isinstance(error['message'],str) and error['message']
                            and len(error['message'].encode('utf-8'))<=4096):
                            message=error['message']
                except (OSError,ValueError,UnicodeError):pass
                raise ValueError(message or stderr.strip() or 'native training failed')
            result=json.loads(output.read_bytes())
        if result['schema']!='b2-lif-training-result-v1':
            raise ValueError('unsupported native training result')
        return result

    def step(self, inputs, labels, *, initial=None, start_tick=None, noise_sequence=None, clock_state=None, poisson_state=None):
        return self.execute(inputs,labels,initial=initial,start_tick=start_tick,noise_sequence=noise_sequence,clock_state=clock_state,poisson_state=poisson_state)

    def gradients(self, inputs, labels, *, initial=None, start_tick=None, noise_sequence=None, clock_state=None, poisson_state=None):
        return self.execute(inputs,labels,operation='gradients',initial=initial,start_tick=start_tick,noise_sequence=noise_sequence,clock_state=clock_state,poisson_state=poisson_state)

    def evaluate(self, inputs, labels, *, initial=None, start_tick=None, noise_sequence=None, clock_state=None, poisson_state=None):
        return self.execute(inputs,labels,operation='evaluate',initial=initial,start_tick=start_tick,noise_sequence=noise_sequence,clock_state=clock_state,poisson_state=poisson_state)

    def retire_poisson_history(self):
        """Discard provably unreachable draws at the committed run boundary.

        Keep pending/unknown identities that can still be read. No time, RNG,
        optimizer or physical state advances. The returned cache is bound to
        this exact continuation; restore a pre-retirement checkpoint to rewind
        or branch from a different physical initial state.
        """
        self._check_topology()
        if self.state is None or self.neuron_state is None or self.poisson_state is None:
            raise ValueError('Poisson retirement requires committed native draw/state history')
        result=self._run_request(dict(plan=self.plan,state=self.state,initial=self.neuron_state,
            inputs=[],labels=[],operation='retire_poisson_history',start_tick=self.clock_tick,
            noise_sequence=self.noise_sequence or 0,clock_state=self.clock_state,poisson_state=self.poisson_state))
        cache=copy.deepcopy(result['poisson_state'])
        self.poisson_state=cache
        return result

    def update_delays(self, pathways):
        """Set pathway delays in SI seconds at a committed run boundary.

        Keys are Brian pathway names from dynamic.delay_layout. Each value is
        a scalar or one value per original edge. Queued events keep their old
        arrivals; future emissions use the new half-up-quantized delays.
        This discrete boundary does not advance time, RNG or the optimizer.
        Subsequent initial='carry' uses the migrated state. Checkpoints include
        the updated plan; construct a fresh trainer from that plan to restore.
        """
        self._check_topology()
        if self.state is None or self.neuron_state is None:
            raise ValueError('delay update requires committed native state')
        if not isinstance(pathways,dict) or not pathways or any(not isinstance(n,str) for n in pathways):
            raise ValueError('delay update requires pathway names and values')
        from brian2 import second
        from brian2.units.fundamentalunits import Quantity, DIMENSIONLESS
        def seconds(value):
            if isinstance(value,Quantity) and value.dim not in (DIMENSIONLESS,second.dim):
                raise ValueError('delay quantities require time units')
            return float(value)
        try:
            values={name:[seconds(x) for x in value] if hasattr(value,'__iter__') and not isinstance(value,(str,bytes)) and getattr(value,'ndim',None)!=0
                    else [seconds(value)] for name,value in pathways.items()}
        except (TypeError,ValueError,OverflowError):raise ValueError('invalid delay values') from None
        if any(not math.isfinite(x) for row in values.values() for x in row):raise ValueError('delays must be finite')
        # NumPy staged event blocks share one user-visible emission delay.
        # Names encode the group so a checkpoint can restore this association
        # without a Python converter or separate provenance object.
        marker='::numpy-stage:'
        if any(marker in name for name in values):raise ValueError('update the original pathway to change staged event delays')
        layout=self.plan.get('dynamic',{}).get('delay_layout',{})
        for original,row in list(values.items()):
            prefix=original+marker
            for path in layout.get('paths',[]):
                if path['name'].startswith(prefix) and path['name'][len(prefix):].isdigit():values[path['name']]=list(row)
        request=dict(plan=self.plan,state=self.state,initial=self.neuron_state,inputs=[],labels=[],
                     operation='update_delays',delay_update=dict(pathways=values),
                     start_tick=self.clock_tick,noise_sequence=self.noise_sequence or 0,clock_state=self.clock_state,poisson_state=self.poisson_state)
        result=self._run_request(request)
        candidate=copy.deepcopy(self.plan);candidate['dynamic']=result['updated_dynamic']
        # Compute all replacement objects before committing the successful
        # native transaction. Model cells and optimizer banks retain indices.
        contract=self._topology_identity(candidate)
        clock_state=copy.deepcopy(result.get('clock_state'))
        poisson_state=copy.deepcopy(result.get('poisson_state'))
        self.plan=candidate;self.neuron_state=result['final_state'];self.state=result['state']
        self.clock_state=clock_state
        self.poisson_state=poisson_state
        self._topology_contract=contract
        return result

    def update_timed_input(self, bank, values):
        """Replace a frozen TimedArray bank between sequences, in native code.

        Use the bank from bundle.provenance['timed_inputs']; flatten 2D values
        in row-major order. Shape, sampling grid and topology stay fixed. No
        optimizer step, neuron state, simulation clock or RNG cursor advances.
        """
        self._check_topology()
        if type(bank) is not int or bank<0 or self.state is None:
            raise ValueError('input update requires a bank index and initialized native state')
        if hasattr(values,'reshape'):values=values.reshape(-1).tolist()
        request=dict(plan=self.plan,state=self.state,operation='update_timed_input',inputs=[],labels=[],
            initial=self.neuron_state,start_tick=self.clock_tick,noise_sequence=self.noise_sequence or 0,clock_state=self.clock_state,poisson_state=self.poisson_state,
            input_update=dict(bank=bank,values=list(values)))
        result=self._run_request(request)
        clock_state=copy.deepcopy(result.get('clock_state'))
        poisson_state=copy.deepcopy(result.get('poisson_state'))
        self.state=result['state']
        self.clock_state=clock_state
        self.poisson_state=poisson_state

    def update_external_state_input(self, source, values):
        """Atomically replace a field described by external_state_inputs.sources.

        Pass the field's provenance descriptor and values in its original
        physical table shape. Discrete values have zero VJP; int32 fields are
        encoded as exact 16-bit limbs in a single native timed bank.
        """
        import numpy as np
        from .training_inputs import encode_external_state_values, sample_major_to_timed_columns
        if not isinstance(source,dict) or set(source) not in ({'bank','dtype','shape','encoding'}, {'bank','dtype','shape','encoding','samples'}):
            raise ValueError('external input update requires its source descriptor')
        values=np.asarray(values)
        if list(values.shape)!=source['shape']:
            raise ValueError('external input shape mismatch')
        dtype=source['dtype']
        if dtype not in ('float','integer','boolean') or source['encoding']!=(
                'signed-high16-low16' if dtype=='integer' else 'plain'):
            raise ValueError('invalid external input encoding')
        if 'samples' in source:
            if type(source['samples']) is not int or values.ndim not in (2,3) or values.shape[0]!=source['samples']:
                raise ValueError('external input sample shape mismatch')
            values=sample_major_to_timed_columns(values)
        encoded=values if dtype=='float' else encode_external_state_values(values,dtype)
        self.update_timed_input(source['bank'],encoded)

    def update_mask(self, masks, *, growth_weight=0.0):
        """Atomically prune/regrow between sequences without advancing clocks.

        Dynamic plans require a committed runtime state and validated ownership.
        Pruned private cells are zeroed. Regrowth restores declared/learned
        initial values, clears event history and timestamps at the current clock.
        A shared cell survives while an owner remains active across the boundary.
        """
        self._check_topology()
        if self.state is None:
            raise ValueError('initialize the native state before structural updates')
        try:
            growth_weight=float(growth_weight)
            masks=[list(row) for row in masks]
        except (ValueError,TypeError,OverflowError):
            raise ValueError('invalid mask or growth weight') from None
        if not math.isfinite(growth_weight):raise ValueError('growth weight must be finite')
        if len(masks)!=len(self.plan['masks']) or any(len(a)!=len(b) for a,b in zip(masks,self.plan['masks'])):
            raise ValueError('mask shape mismatch; topology migration is unsupported')
        if any(v not in (0,1) for row in masks for v in row):
            raise ValueError('mask entries must be zero or one')
        masks=[[float(v) for v in row] for row in masks]
        if masks==self.plan['masks']:return
        if self.plan.get('dynamic') is not None:
            if self.plan['dynamic'].get('migration') is None:
                raise ValueError('dynamic mask migration requires an ownership layout')
            if self.neuron_state is None:
                raise ValueError('dynamic mask migration requires a committed runtime state')
            request=dict(plan=self.plan,state=self.state,operation='update_mask',inputs=[],labels=[],
                         initial=self.neuron_state,start_tick=self.clock_tick,noise_sequence=self.noise_sequence or 0,clock_state=self.clock_state,poisson_state=self.poisson_state,
                         mask_update=dict(masks=masks,growth_weight=growth_weight))
            result=self._run_request(request)
            state=result['state'];live=result['final_state'];clock_state=copy.deepcopy(result.get('clock_state'))
            poisson_state=copy.deepcopy(result.get('poisson_state'))
        else:
            state=copy.deepcopy(self.state);live=self.neuron_state;clock_state=self.clock_state
            poisson_state=self.poisson_state
            for l,(old,new) in enumerate(zip(self.plan['masks'],masks)):
                for e,(a,b) in enumerate(zip(old,new)):
                    if a!=b:
                        state['weights'][l][e]=growth_weight if b else 0.0
                        state['first_moment'][l][e]=state['second_moment'][l][e]=0.0
        self.state=state;self.neuron_state=live;self.plan['masks']=masks
        self.clock_state=clock_state
        self.poisson_state=poisson_state

    def store(self, filename):
        """Atomic checksummed snapshot; only load trusted local artifacts."""
        self._check_topology()
        payload=canonical_bytes(dict(plan=self.plan,state=self.state,
                    neuron_state=self.neuron_state,elapsed_ticks=self.elapsed_ticks,clock_tick=self.clock_tick,clock_state=self.clock_state,poisson_state=self.poisson_state,
                    noise_sequence=self.noise_sequence,next_noise_sequence=self.next_noise_sequence,
                    metal_source_sha256=self._metal_source_hash(),
                    mpi_runtime=self._mpi_runtime_identity(),
                    runtime_sha256=hashlib.sha256(self.runner.read_bytes()).hexdigest()))
        envelope=dict(schema='b2-native-training-checkpoint-v1',payload=payload.decode(),
                      sha256=hashlib.sha256(payload).hexdigest())
        path=Path(filename); temporary=None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent,prefix=path.name+'.',delete=False) as stream:
                temporary=Path(stream.name);stream.write(canonical_bytes(envelope));stream.flush();os.fsync(stream.fileno())
            os.replace(temporary,path)
        finally:
            if temporary is not None: temporary.unlink(missing_ok=True)

    def restore(self, filename):
        self._check_topology()
        stored=json.loads(Path(filename).read_bytes())
        if set(stored)!={'schema','payload','sha256'} or stored['schema']!='b2-native-training-checkpoint-v1' or hashlib.sha256(stored['payload'].encode()).hexdigest()!=stored['sha256']:
            raise ValueError('native training checkpoint integrity mismatch')
        payload=json.loads(stored['payload'])
        if payload['plan']!=self.plan or payload['runtime_sha256']!=hashlib.sha256(self.runner.read_bytes()).hexdigest():
            raise ValueError('native training checkpoint plan/runtime mismatch')
        if payload['metal_source_sha256']!=self._metal_source_hash():
            raise ValueError('native training checkpoint Metal runtime mismatch')
        if payload.get('mpi_runtime')!=self._mpi_runtime_identity():
            raise ValueError('native training checkpoint MPI runtime mismatch')
        tick=payload.get('clock_tick',0 if self.plan.get('clock') is None else None)
        if type(tick) is not int or not 0<=tick<=2**53:
            raise ValueError('invalid checkpoint clock tick')
        sequence=payload.get('noise_sequence')
        next_sequence=payload.get('next_noise_sequence',0 if self.plan.get('noise_streams') is None else None)
        if (type(next_sequence) is not int or not 0<=next_sequence<=2**64-1
            or sequence is not None and (type(sequence) is not int or not 0<=sequence<next_sequence)
            or self.plan.get('noise_streams') is not None and payload['neuron_state'] is not None and sequence is None):
            raise ValueError('invalid checkpoint noise sequence')
        clock_state=payload.get('clock_state')
        if (clock_state is None and payload['neuron_state'] is not None
            and (self.plan.get('dynamic') or {}).get('clocks') is not None):
            raise ValueError('checkpoint is missing its dynamic clock state')
        poisson_state=payload.get('poisson_state')
        programs=list((self.plan.get('dynamic') or {}).get('program_sets',[]))
        programs+=list(self.plan.get('state_equations') or [])+list(self.plan.get('state_resets') or [])
        if self.plan.get('equations') is not None:programs.append(self.plan['equations'])
        has_poisson=any(n.get('op')=='poisson' for group in programs for program in group for n in program)
        if has_poisson and payload['neuron_state'] is not None and poisson_state is None:
            raise ValueError('checkpoint is missing its Poisson draw state')
        if clock_state is not None or poisson_state is not None:
            # Validate with the same bounded native scheduler before committing
            # any Python optimizer, neuron, noise or clock state.
            self._run_request(dict(plan=self.plan,state=payload['state'],initial=payload['neuron_state'],
                operation='validate_clock_state' if clock_state is not None else 'validate_poisson_state',inputs=[],labels=[],start_tick=tick,
                noise_sequence=sequence or 0,clock_state=clock_state,poisson_state=poisson_state))
        self.state=payload['state']
        self.neuron_state=payload['neuron_state']
        self.elapsed_ticks=payload['elapsed_ticks']
        self.clock_tick=tick
        self.clock_state=clock_state
        self.poisson_state=poisson_state
        self.noise_sequence=sequence
        self.next_noise_sequence=next_sequence

    def _topology_identity(self, plan=None):
        plan=self.plan if plan is None else plan
        return hashlib.sha256(canonical_bytes({key:plan.get(key) for key in
                              ('schema','backend','sizes','projections','mpi_ranks','equations','threshold_parameters','threshold_per_neuron','state_equations','state_resets','refractory','clock','noise_streams','seed','dynamic')})).hexdigest()

    def _check_topology(self):
        if self._topology_identity()!=self._topology_contract:
            raise ValueError('native training topology/backend changed; create a new trainer')

    def _metal_source_hash(self):
        if self._metal_library is None: return None
        package=Path(__file__).parent
        if self.plan['backend']=='cuda':
            return hashlib.sha256(b''.join((package/name).read_bytes() for name in
                                  ('training_cuda.py','training_cuda.cu','training_metal.metal','training_metal_mpi.metal','training_state.metal','training_clock.metal','training_poisson.metal','training_dynamic.metal','training_dynamic_mpi.metal'))).hexdigest()
        return hashlib.sha256(b''.join((package/name).read_bytes() for name in
                              ('training_metal.m','training_metal.metal','training_metal_mpi.metal','training_state.metal','training_clock.metal','training_poisson.metal','training_dynamic.metal','training_dynamic_mpi.metal','training_metal.py'))).hexdigest()

    def _mpi_runtime_identity(self):
        if self._mpi_library is None:return None
        import platform
        package=Path(__file__).parent
        # Binary build UUIDs may vary; pin the shim sources, compiler/runtime,
        # rank count (in plan), and host architecture instead.
        version=subprocess.run(['mpiexec','--version'],capture_output=True,text=True,check=True).stdout
        compiler=subprocess.run(['mpicc','--version'],capture_output=True,text=True,check=True).stdout
        return dict(source_sha256=hashlib.sha256(b''.join((package/name).read_bytes() for name in
                    ('training_mpi.c','training_mpi.py'))).hexdigest(),runtime=version,
                    compiler=compiler,platform=platform.system(),machine=platform.machine(),
                    provider=self._mpi_environment.get('FI_PROVIDER',os.environ.get('FI_PROVIDER')))
