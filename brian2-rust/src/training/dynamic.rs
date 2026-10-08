//! Ordered dynamic state transforms and event-path reverse mode.
//! Contexts are taped before each action; targets are written simultaneously.
//! A pathway's sequential source statements must be composed before lowering.
use super::*;

#[derive(Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Trigger {
    pub external: bool,
    pub index: usize,
    #[serde(default)]
    pub state: bool,
}
#[derive(Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Action {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub clock: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub indirect: Option<indirect::Access>,
    #[serde(default)]
    pub threshold_predicate: bool,
    #[serde(default)]
    pub threshold_margin: bool,
    #[serde(default)]
    pub threshold_inclusive: bool,
    #[serde(default)]
    pub mask: Option<[usize; 2]>,
    pub owner: usize,
    pub reads: Vec<usize>,
    pub writes: Vec<usize>,
    pub program_set: Option<usize>,
    pub threshold: Option<usize>,
    pub trigger: Option<Trigger>,
    #[serde(default)]
    pub detach_trigger: bool,
    #[serde(default)]
    pub parameter_index: usize,
    #[serde(default)]
    pub noise_domain: u64,
    #[serde(default)]
    pub noise_entity: u64,
    #[serde(default)]
    pub noise_streams: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub event_noise: Option<noise::EventAddress>,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SampleIndex {
    pub slot: usize,
    pub samples: usize,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Spec {
    /// Native-owned sample identity for per-sample external tables.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sample_index: Option<SampleIndex>,
    #[serde(default, skip_serializing_if = "is_false")]
    pub clear_spike_buffers: bool,
    /// Persistent, differentiable binary cells overwritten only by thresholds.
    /// Consumers use ordinary state triggers, including on other clocks.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub spike_buffers: Vec<usize>,
    #[serde(default)]
    pub delay_layout: Option<delay::Layout>,
    #[serde(default)]
    pub integer_states: Vec<usize>,
    #[serde(default)]
    pub integer_parameters: Vec<[usize; 2]>,
    #[serde(default)]
    pub parameter_maps: Vec<Vec<usize>>,
    #[serde(default)]
    pub threshold_references: Option<Vec<Option<[usize; 2]>>>,
    #[serde(default)]
    pub migration: Option<migration::Layout>,
    #[serde(default)]
    pub binary_states: Vec<usize>,
    #[serde(default)]
    pub clocks: Option<ClockSchedule>,
    pub initial: Vec<f64>,
    pub initial_parameters: Vec<Option<[usize; 2]>>,
    pub detached: Vec<bool>,
    pub voltage: Vec<usize>,
    pub program_sets: Vec<Vec<equation::Program>>,
    pub actions: Vec<Action>,
}
fn is_false(value: &bool) -> bool { !value }
impl Spec {
    pub(super) fn validate(&self, plan: &Plan, neurons: usize) -> Result<()> {
        let width = self.initial.len();
        ensure(plan.clock.is_some() && width > 0 && width <= 1_000_000
            && self.initial.iter().all(|v| v.is_finite())
            && self.initial_parameters.len() == width && self.detached.len() == width,
            "invalid dynamic state layout or clock")?;
        if let Some(clocks) = &self.clocks { clocks.validate(plan.clock.as_ref().unwrap())?; }
        let map_entries = self.parameter_maps.iter().try_fold(0usize, |n, row| n.checked_add(row.len()))
            .ok_or("dynamic parameter map overflow")?;
        ensure(self.parameter_maps.len() <= 4096 && self.parameter_maps.iter().all(|r| !r.is_empty() && r.len() <= 65536)
            && map_entries.checked_mul(8).is_some_and(|b| b <= plan.max_tape_bytes), "dynamic parameter map budget exceeded")?;
        if let Some(references) = &self.threshold_references {
            ensure(references.len() == neurons && references.iter().flatten()
                .all(|&[b,i]| b < plan.masks.len() && i < plan.masks[b].len()), "invalid dynamic threshold reference mapping")?;
        }
        let binary: std::collections::HashSet<_> = self.binary_states.iter().copied().collect();
        ensure(binary.len() == self.binary_states.len() && binary.iter().all(|&k| k < width
            && (self.initial[k] == 0.0 || self.initial[k] == 1.0) && self.initial_parameters[k].is_none()),
            "invalid dynamic binary state layout or initial value")?;
        let integers: std::collections::HashSet<_> = self.integer_states.iter().copied().collect();
        ensure(integers.len() == self.integer_states.len() && integers.iter().all(|&k| k < width
            && self.detached[k] && !binary.contains(&k) && self.initial_parameters[k].is_none()
            && equation::int32(self.initial[k]).is_ok()), "invalid integer state layout/value")?;
        if let Some(sample) = &self.sample_index {
            ensure(sample.samples > 0 && sample.samples <= 4096 && integers.contains(&sample.slot)
                && self.initial[sample.slot] == 0.0, "invalid native sample index layout")?;
            ensure(self.migration.as_ref().is_none_or(|m| m.cells.iter().all(|c| c.index != sample.slot)),
                "native sample index cannot be structurally migrated")?;
        }
        ensure(self.integer_parameters.len() <= 1_000_000, "integer parameter budget exceeded")?;
        let integer_parameters: std::collections::HashSet<_> = self.integer_parameters.iter().copied().collect();
        ensure(integer_parameters.len() == self.integer_parameters.len() && integer_parameters.iter().all(|&[b,i]|
            b < plan.masks.len() && i < plan.masks[b].len() && plan.trainable.get(b)==Some(&false)), "integer parameters must be unique frozen slots")?;
        let integer_banks: std::collections::HashSet<_> = integer_parameters.iter().map(|r| r[0]).collect();
        let mut integer_bank_sizes = vec![0usize; plan.masks.len()];
        for &[bank, _] in &integer_parameters { integer_bank_sizes[bank] += 1; }
        ensure(self.voltage.len() == neurons && self.voltage.iter().all(|&k| k < width && !self.detached[k]),
            "invalid dynamic voltage mapping")?;
        let neuron_width = plan.state_equations.as_ref().unwrap().iter().zip(&plan.sizes[1..]).map(|(programs, size)| programs.len() * size).sum::<usize>();
        ensure(width >= neuron_width, "dynamic state omits declared neuron cells")?;
        ensure(!self.clear_spike_buffers || !self.spike_buffers.is_empty(), "buffer boundary clear requires spike buffers")?;
        let buffers:std::collections::HashSet<_>=self.spike_buffers.iter().copied().collect();
        ensure(self.spike_buffers.is_empty() || self.spike_buffers.len()==neurons
            && buffers.len()==neurons && buffers.iter().all(|&k|k>=neuron_width && k<width
                && binary.contains(&k) && !self.detached[k] && !self.voltage.contains(&k)),
            "invalid persistent spike buffer layout")?;
        let unique_voltage: std::collections::HashSet<_> = self.voltage.iter().collect();
        ensure(unique_voltage.len() == neurons, "dynamic voltages must be distinct")?;
        ensure(!self.actions.is_empty() && self.actions.len() <= 1_000_000
            && !self.program_sets.is_empty() && self.program_sets.len() <= 4096,
            "dynamic action/program budget exceeded")?;
        for (k, reference) in self.initial_parameters.iter().enumerate() {
            if let Some([bank, index]) = reference {
                ensure(!self.detached[k] && *bank < plan.masks.len() && *index < plan.masks[*bank].len(),
                    "invalid dynamic initial parameter binding")?;
            }
        }
        let mut prepared = vec![usize::MAX; width];
        let mut prepared_clock = vec![usize::MAX; width];
        let mut prepared_boolean = vec![false; width];
        let mut thresholds = vec![false; neurons];
        let mut used = vec![false; self.program_sets.len()];
        for action in &self.actions {
            if let Some(sample) = &self.sample_index {
                ensure(!action.writes.contains(&sample.slot) && action.indirect.as_ref().is_none_or(|a|
                    a.writes.values().all(|w| w.tables.last().is_none_or(|t| !t.contains(&sample.slot)))),
                    "native sample index is read-only")?;
            }
            let action_clock=action.clock.unwrap_or(0);
            ensure(action_clock < self.clocks.as_ref().map_or(1,|c|c.dts.len()), "dynamic action clock outside schedule")?;
            ensure(action_clock==0 || (!buffers.is_empty() || action.threshold.is_none() && action.event_noise.is_none())
                && action.trigger.as_ref().is_none_or(|g|g.state),
                "non-primary spike thresholds/pathways require asynchronous event buffers")?;
            ensure(buffers.is_empty() || action.trigger.as_ref().is_none_or(|g|g.state || g.external),
                "buffered spike consumers must use state triggers")?;
            if let Some(address) = &action.event_noise {
                ensure(action.trigger.is_some() && action.threshold.is_none() && action.noise_streams > 0
                    && address.delay <= 1_000_000 && address.pending.is_none_or(|id| id > 0 && address.delay == 0),
                    "invalid event noise address")?;
            }
            ensure((!action.threshold_margin && !action.threshold_inclusive && !action.threshold_predicate) || action.threshold.is_some(),
                "threshold comparison flags require a threshold action")?;
            ensure(!action.threshold_inclusive || action.threshold_margin, "inclusive comparison requires a margin")?;
            ensure(!action.threshold_predicate || action.threshold_margin && !action.threshold_inclusive,
                "Boolean threshold requires exclusive predicate mode")?;
            if let Some([bank, index]) = action.mask {
                ensure(bank < plan.masks.len() && index < plan.masks[bank].len() && action.threshold.is_none(), "invalid dynamic action mask")?;
            }
            ensure(action.owner < neurons && !action.reads.is_empty() && action.reads.len() <= 64
                && action.reads.iter().all(|&k| k < width) && action.writes.iter().all(|&k| k < width)
                && action.noise_streams <= 16,
                "invalid dynamic action context/owner")?;
            if let Some(neuron) = action.threshold {
                ensure(action.indirect.is_none(), "threshold actions cannot use indirect storage")?;
                ensure(action.program_set.is_none() && action.trigger.is_none() && action.writes.is_empty()
                    && neuron < neurons && !thresholds[neuron] && action.reads.len() <= 2
                    && action.noise_streams == 0
                    && if action.threshold_margin { action.reads[0] >= neuron_width && !self.detached[action.reads[0]]
                        && self.initial_parameters[action.reads[0]].is_none()
                        && prepared[action.reads[0]] == action.owner
                        && prepared_clock[action.reads[0]] == action_clock
                        && (!action.threshold_predicate || prepared_boolean[action.reads[0]]) }
                        else { action.reads[0] == self.voltage[neuron] },
                    "invalid or repeated dynamic threshold")?;
                thresholds[neuron] = true;
            } else {
                let id = action.program_set.ok_or("dynamic transform requires programs")?;
                ensure(id < self.program_sets.len(), "dynamic program set outside domain")?;
                used[id] = true;
                let programs = &self.program_sets[id];
                let unique: std::collections::HashSet<_> = action.writes.iter().collect();
                ensure(!programs.is_empty() && programs.len() <= 64 && programs.len() == action.writes.len()
                    // Indexed execution already tapes all destinations and
                    // resolves last-writer collisions. Preserve distinct Brian
                    // locals even when fixed aliases share one target: an
                    // overwritten integer output can still select another write.
                    && (unique.len() == action.writes.len() || action.indirect.is_some())
                    && action.writes.iter().all(|k| action.reads.contains(k)),
                    "invalid dynamic write/context shape or duplicate targets")?;
                if let Some(gate) = &action.trigger {
                    ensure(if gate.state { !gate.external && binary.contains(&gate.index) && action.reads.contains(&gate.index) }
                        else if gate.external { gate.index < plan.sizes[0] }
                        else { gate.index < neurons && thresholds[gate.index] },
                        "dynamic trigger must follow its threshold or read a binary state")?;
                }
                let (normal, uniform) = equation::noise_masks(programs.iter());
                ensure(normal & uniform == 0, "dynamic noise stream mixes distributions")?;
                poisson_ir::validate(programs, action, plan)?;
                for (program, &target) in programs.iter().zip(&action.writes) {
                    equation::validate_states(program, plan, action.reads.len())?;
                    ensure(equation::integer_node(program.last().unwrap()) == integers.contains(&target),
                        "integer output type does not match target state")?;
                    for node in program {
                        match node {
                            equation::Node::IntegerState { index } => ensure(integers.contains(&action.reads[*index]), "integer read requires integer storage")?,
                            equation::Node::State { index } | equation::Node::RefractoryActive { index } => ensure(!integers.contains(&action.reads[*index]), "integer storage requires typed read")?,
                            equation::Node::IntegerParameter { bank, index } => ensure(integer_parameters.contains(&[*bank,*index]), "integer parameter requires typed frozen storage")?,
                            equation::Node::IntegerNeuronParameter { bank, index } => ensure(index.checked_add(action.parameter_index)
                                .is_some_and(|k| integer_parameters.contains(&[*bank,k])), "integer indexed parameter outside typed storage")?,
                            equation::Node::IntegerMappedParameter { bank, mapping } => ensure(self.parameter_maps[*mapping].get(action.parameter_index)
                                .is_some_and(|&k| integer_parameters.contains(&[*bank,k])), "integer mapped parameter outside typed storage")?,
                            equation::Node::IntegerParameterGather { bank, .. } => ensure(integer_bank_sizes[*bank] == plan.masks[*bank].len(),
                                "integer parameter gather requires an entirely typed frozen bank")?,
                            equation::Node::ParameterGather { bank, .. } => ensure(!integer_banks.contains(bank),
                                "floating parameter gather cannot read integer storage")?,
                            equation::Node::Parameter { bank, index } => ensure(!integer_parameters.contains(&[*bank,*index]), "integer parameter requires typed read")?,
                            equation::Node::Voltage => ensure(!integers.contains(&action.reads[0]), "integer storage requires typed read")?,
                            equation::Node::TimedParameter { bank, .. } => ensure(!integer_banks.contains(bank), "TimedArray requires floating storage")?,
                            equation::Node::NeuronParameter { bank, index } => ensure(index.checked_add(action.parameter_index)
                                .is_some_and(|i| i < plan.masks[*bank].len() && !integer_parameters.contains(&[*bank,i])), "dynamic indexed parameter outside floating bank")?,
                            equation::Node::MappedParameter { bank, mapping } => ensure(self.parameter_maps[*mapping].get(action.parameter_index)
                                .is_some_and(|&i| !integer_parameters.contains(&[*bank,i])),
                                "dynamic parameter map index outside domain")?,
                            equation::Node::Noise { stream } | equation::Node::UniformNoise { stream } => ensure(*stream < action.noise_streams,
                                "dynamic noise stream outside action")?,
                            _ => (),
                        }
                    }
                }
                if let Some(access) = &action.indirect { access.validate(action, self, plan)?; }
                ensure(!action.possible_writes().any(|k|buffers.contains(&k)), "spike buffers are written only by thresholds")?;
                if action.indirect.as_ref().is_some_and(|a| !a.writes.is_empty()) {
                    // A runtime destination does not prove a particular scratch
                    // cell was prepared, even if its placeholder names that cell.
                    for index in action.possible_writes() { prepared[index] = usize::MAX; prepared_boolean[index] = false; }
                    continue;
                }
                for (&index, program) in action.writes.iter().zip(programs) {
                    prepared_boolean[index] = equation::boolean_node(program.last().unwrap());
                    // A stochastic unconditional write prepares the margin just as a
                    // deterministic one does; its draw is replayed for the VJP.
                    prepared_clock[index]=action_clock;
                    if action.mask.is_none() && action.trigger.is_none() { prepared[index] = action.owner; }
                    else { prepared[index] = usize::MAX; }
                }
            }
        }
        poisson_ir::validate_actions(self)?;
        ensure(thresholds.iter().all(|&v| v), "dynamic plan requires exactly one threshold per neuron")?;
        ensure(used.iter().all(|&v| v), "unused dynamic program set")?;
        if let Some(layout) = &self.migration { layout.validate(plan, self, neuron_width)?; }
        if let Some(layout) = &self.delay_layout { layout.validate(plan, self, neuron_width)?; }
        Ok(())
    }
    pub(super) fn memory_bytes(&self, batch: usize, time: usize, ranks: usize) -> Option<usize> {
        let reads = self.actions.iter().try_fold(0usize, |s, a| s.checked_add(a.reads.len()))?;
        let nodes = self.program_sets.iter().flatten().try_fold(0usize, |s, p| s.checked_add(p.len()))?;
        reads.checked_mul(batch)?.checked_mul(time)?.checked_mul(8)?
            .checked_add(self.voltage.len().checked_mul(batch)?.checked_mul(time)?.checked_mul(48)?)?
            .checked_add(reads.checked_mul(32)?)?
            .checked_add(self.actions.len().checked_mul(256)?)?
            .checked_add(self.actions.iter().try_fold(0usize, |n,a| n.checked_add(a.indirect.as_ref().map_or(0, |v| v.memory_bytes())))?)?
            .checked_add(self.actions.iter().filter(|a| a.indirect.is_some()).try_fold(0usize, |n,a| n.checked_add(128 + a.reads.len()*8 + a.writes.len()*16))?.checked_mul(batch)?.checked_mul(time)?)?
            .checked_add(nodes.checked_mul(128)?)?
            .checked_add(256)? // visited/activity arrays for one serial VJP
            .checked_add(self.initial.len().checked_mul(batch)?.checked_mul(64)?)?
            .checked_add(self.integer_parameters.len().checked_mul(48)?)?
            .checked_add(self.delay_layout.as_ref().map_or(0, |d| d.memory_bytes()))?
            .checked_add(self.migration.as_ref().map_or(0, |m| m.memory_bytes()))?
            .checked_add(self.parameter_maps.iter().try_fold(0usize, |n,r| n.checked_add(r.len().checked_mul(8)?.checked_add(24)?))?)?
            .checked_add(self.threshold_references.as_ref().map_or(0, Vec::len).checked_mul(24)?)?
            .checked_add(self.clocks.as_ref().map_or(0, |c| c.dts.len()).checked_mul(time.checked_mul(8)?.checked_add(256)?)?)?
            .checked_add(129usize.checked_mul(ranks + 2)?.checked_mul(8)?)
    }
}

/// Brian Clock._calc_timestep initializes each clock at Network.t. Subsequent
/// Network._nextclocks steps coalesce relative to the earliest clock, using the
/// smaller dt and a strict epsilon test. Replaying this schedule also handles
/// a third clock coalescing a synaptic clock before the next neuronal tick.
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClockSchedule {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub order: Vec<usize>,
    pub start: f64,
    pub dts: Vec<f64>,
    pub epsilon: f64,
}
impl ClockSchedule {
    pub(super) fn schedule(&self) -> clock::Schedule<'_> {
        clock::Schedule { start: self.start, dts: &self.dts, epsilon: self.epsilon, order: &self.order }
    }
    fn validate(&self, main: &Clock) -> Result<()> {
        let schedule = self.schedule(); schedule.validate()?;
        ensure(self.dts[0] == main.dt, "invalid dynamic clock schedule")?;
        ensure(schedule.tick_at(self.start, main.dt)? as f64 * main.dt == main.origin,
            "dynamic clock origin differs from snapshot schedule")
    }
    pub(super) fn boundary(&self, tick: u64) -> Result<Vec<f64>> {
        let n = self.dts.len();
        ensure(n > 0 && tick <= clock::MAX_WORK / n as u64, "dynamic clock replay exceeds work budget")?;
        let schedule = self.schedule(); let mut cursor = schedule.cursor()?;
        let end_main = cursor.visit().ticks[0].checked_add(tick).filter(|&t| t < (1u64 << 53)).ok_or("dynamic clock tick overflow")?;
        // Structural boundaries use the rounded main-clock time, which can
        // be slightly before Network.t at a tolerance-aligned warm snapshot.
        let limits = schedule.limits(end_main as f64 * self.dts[0])?;
        cursor.run_to_limits(&limits, |_| Ok(()))?;
        Ok(cursor.visit().times.to_vec())
    }
}

fn samples(plan: &Plan, uniform: u16, a: &Action, sequence: u64, batch: usize, tick: u64) -> [f64; 96] {
    let mut out = [0.0; 96];
    for (stream, value) in out[..a.noise_streams].iter_mut().enumerate() {
        *value = noise::event_sample(a.event_noise.as_ref(), uniform & (1 << stream) != 0, plan.seed, sequence, batch as u64, a.noise_domain, a.noise_entity, tick, stream as u64);
    }
    poisson_ir::encode_keys(&mut out, plan, a, sequence, batch, tick);
    out
}

pub(super) fn execute(plan: &Plan, state: State, inputs: &[Vec<Vec<f64>>], labels: &[usize],
    live: Vec<Vec<f64>>, bytes: usize, operation: &str, mpi: Option<&mpi::Context>,
    _start_tick: u64, noise_sequence: u64, default_initial: bool, itinerary: &itinerary::Tape, cache:Option<&poisson_cache::Checkpoint>) -> Result<Output> {
    execute_inner(plan,state,inputs,labels,live,bytes,operation,mpi,noise_sequence,default_initial,itinerary,None,0,cache)
}

// A replay always evaluates one complete sample and never differentiates or
// updates the optimizer. batch_origin retains its original random counter IDs.
fn execute_inner(plan: &Plan, mut state: State, inputs: &[Vec<Vec<f64>>], labels: &[usize],
    mut live: Vec<Vec<f64>>, bytes: usize, operation: &str, mpi: Option<&mpi::Context>,
    noise_sequence: u64, default_initial: bool, itinerary: &itinerary::Tape,
    replay: Option<(usize,usize,usize)>, batch_origin: usize, cache:Option<&poisson_cache::Checkpoint>) -> Result<Output> {
    let spec = plan.dynamic.as_ref().unwrap();
    let (offsets, n) = plan.validate()?;
    let batch = inputs.len(); let time = inputs[0].len(); let width = spec.initial.len();
    let visits=itinerary.len();
    let forced=replay.map(|(v,a,s)|poisson_cache::identity(&spec.actions[a],s,batch_origin,itinerary.tick(v,&spec.actions[a])));
    let _draw_scope=poisson_cache::install(plan,noise_sequence,batch,cache,forced)?;
    let boundary_possible = operation != "evaluate" && poisson_ir::boundary_possible(spec);
    let original = if boundary_possible {live.clone()} else {Vec::new()};
    ensure(inputs.iter().flatten().flatten().all(|&x| x == 0.0 || x == 1.0),
        "dynamic event inputs must be binary spikes")?;
    let mut binary = vec![false; width];
    let mut integers = vec![false; width];
    for &k in &spec.integer_states { integers[k] = true; }
    for &k in &spec.binary_states { binary[k] = true; }
    ensure(live.iter().all(|row| spec.binary_states.iter().all(|&k| row[k] == 0.0 || row[k] == 1.0)),
        "dynamic event history must be binary")?;
    let event_gate = |action: &Action, context: &[f64], b: usize, visit: usize, spikes: &[f64]| {
        action.trigger.as_ref().map_or(1.0, |g| if g.state {
            context[action.reads.iter().position(|&k| k == g.index).unwrap()]
        } else if g.external { inputs[b][itinerary.frames[visit].unwrap_or(0)][g.index] } else { spikes[(b * visits + visit) * n + g.index] })
    };
    let uniform_masks: Vec<u16> = spec.program_sets.iter().map(|p| equation::noise_masks(p.iter()).1).collect();
    let mut context_offsets = vec![0usize];
    for action in &spec.actions { context_offsets.push(context_offsets.last().unwrap() + action.reads.len()); }
    let context_width = *context_offsets.last().unwrap();
    let mut tape = vec![0.0; batch * visits * context_width];
    let mut address_offsets = Vec::with_capacity(spec.actions.len());
    let mut address_width = 0;
    for action in &spec.actions { address_offsets.push(address_width); address_width += usize::from(action.indirect.is_some()); }
    let mut address_tape: Vec<Option<indirect::Resolved>> = (0..batch*visits*address_width).map(|_| None).collect();
    let mut spikes = vec![0.0; batch * visits * n];
    let owns = |a: &Action| mpi.is_none_or(|m| m.owns(a.owner, n));
    let layer_for = |neuron: usize| (0..plan.sizes.len() - 1).find(|&l| neuron < offsets[l + 1]).unwrap();
    let theta = |neuron: usize| {
        let l = layer_for(neuron);
        plan.threshold_reference(l, neuron - offsets[l]).map_or(plan.threshold[l], |[bank, i]| state.weights[bank][i])
    };
    let thresholds: Vec<f64> = (0..n).map(theta).collect();
    for b in 0..batch {
        for visit in 0..visits {
            let nt = (b * visits + visit) * n;
            let clock_row = itinerary.row(visit);
            for (i, action) in spec.actions.iter().enumerate() {
                if !itinerary.enabled(visit,action) {continue;}
                let timestamp=clock_row[action.clock.unwrap_or(0)];
                let noise_tick=itinerary.tick(visit,action);
                let at = (b * visits + visit) * context_width + context_offsets[i];
                let context = &mut tape[at..at + action.reads.len()];
                for (v, &k) in context.iter_mut().zip(&action.reads) { *v = live[b][k]; }
                if let Some(neuron) = action.threshold {
                    let mut spike = [0.0];
                    if owns(action) {
                        ensure(context.len() == 1 || context[1] == 0.0 || context[1] == 1.0,
                            "dynamic threshold activity must be binary")?;
                        let threshold = if action.threshold_margin { 0.0 } else { thresholds[neuron] };
                        ensure(!action.threshold_predicate || context[0] == 0.0 || context[0] == 1.0, "threshold predicate must be binary")?;
                        let above = if action.threshold_inclusive { context[0] >= threshold } else { context[0] > threshold };
                        spike[0] = if (context.len() == 1 || context[1] == 1.0) && above { 1.0 } else { 0.0 };
                    }
                    if let Some(m) = mpi { m.sum(&mut spike)?; }
                    spikes[nt + neuron] = spike[0];
                    if !spec.spike_buffers.is_empty() { live[b][spec.spike_buffers[neuron]]=spike[0]; }
                } else {
                    if action.mask.is_some_and(|[bank, index]| plan.masks[bank][index] == 0.0) { continue; }
                    let gate = event_gate(action, context, b, visit, &spikes);
                    poisson_cache::begin(action,b+batch_origin,visit,i,noise_tick,true,gate!=0.&&owns(action));
                    if action.indirect.is_some() {
                        let mut noise = samples(plan, uniform_masks[action.program_set.unwrap()], action, noise_sequence, b + batch_origin, noise_tick);
                        if let Some((v,a,stream)) = replay {if (v,a) == (visit,i) {noise[80+stream] = 1.;}}
                        let eval = indirect::Evaluation { plan, spec, action, weights: &state.weights,
                            timestamp: timestamp,
                            noise: &noise, clocks: clock_row };
                        let record = eval.run(context, &mut live[b], gate, owns(action), mpi)?;
                        address_tape[(b*visits+visit)*address_width+address_offsets[i]] = Some(record);
                        poisson_cache::sync(mpi)?;
                        continue;
                    }
                    let mut delta = [0.0; 64];
                    if owns(action) && gate != 0.0 {
                        let mut noise = samples(plan, uniform_masks[action.program_set.unwrap()], action, noise_sequence, b + batch_origin, noise_tick);
                        if let Some((v,a,stream)) = replay {if (v,a) == (visit,i) {noise[80+stream] = 1.;}}
                        for (s, program) in spec.program_sets[action.program_set.unwrap()].iter().enumerate() {
                            let value = equation::forward_clocked(program, context, &state.weights, action.parameter_index,
                                timestamp, &noise, clock_row, &spec.parameter_maps)?;
                            delta[s] = value;
                        }
                    }
                    if let Some(m) = mpi { m.sum(&mut delta[..action.writes.len()])?; }
                    poisson_cache::sync(mpi)?;
                    if gate != 0.0 { for (&k, &value) in action.writes.iter().zip(&delta) {
                        ensure(!binary[k] || value == 0.0 || value == 1.0, "dynamic event history update must be binary")?;
                        ensure(!integers[k] || equation::int32(value).is_ok(), "integer state write is outside int32")?;
                        live[b][k] = value;
                    } }
                }
            }
        }
    }
    let classes = *plan.sizes.last().unwrap(); let output_start = n - classes;
    let mut logits = vec![vec![0.0; classes]; batch];
    let mut seeds = logits.clone(); let mut loss = 0.0;
    let mut sample_losses = vec![0.; batch];
    for b in 0..batch {
        for t in 0..visits { for j in 0..classes {
            logits[b][j] += spikes[(b * visits + t) * n + output_start + j] * plan.logit_scale / time as f64;
        }}
        let maximum = logits[b].iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let sum = logits[b].iter().map(|v| (v - maximum).exp()).sum::<f64>();
        sample_losses[b] = (maximum + sum.ln() - logits[b][labels[b]]) / batch as f64;
        loss += sample_losses[b];
        for j in 0..classes {
            seeds[b][j] = ((logits[b][j] - maximum).exp() / sum - if j == labels[b] { 1.0 } else { 0.0 })
                / batch as f64 * plan.logit_scale / time as f64;
        }
    }
    let mut gradients = state.weights.iter().map(|w| vec![0.0; w.len()]).collect::<Vec<_>>();
    let mut adjoints = vec![vec![0.0; width]; batch];
    if operation != "evaluate" {
        for b in 0..batch {
            for visit in (0..visits).rev() {
                let frame=itinerary.frames[visit];
                let t=frame.unwrap_or(0);
                let clock_row = itinerary.row(visit);
                let mut spike_adjoints = vec![0.0; n];
                if frame.is_some() || !spec.spike_buffers.is_empty() {spike_adjoints[output_start..].copy_from_slice(&seeds[b]);}
                for (i, action) in spec.actions.iter().enumerate().rev() {
                    if !itinerary.enabled(visit,action) {continue;}
                    let timestamp=clock_row[action.clock.unwrap_or(0)];
                    let noise_tick=itinerary.tick(visit,action);
                    let at = (b * visits + visit) * context_width + context_offsets[i];
                    let context = &tape[at..at + action.reads.len()];
                    if let Some(neuron) = action.threshold {
                        if !spec.spike_buffers.is_empty() {
                            let cell=spec.spike_buffers[neuron];
                            spike_adjoints[neuron]+=adjoints[b][cell];adjoints[b][cell]=0.0;
                        }
                        let mut delta = [0.0];
                        if owns(action) && (context.len() == 1 || context[1] == 1.0) {
                            delta[0] = spike_adjoints[neuron] * if action.threshold_predicate { 1.0 } else {
                                plan.surrogate.derivative(context[0] - if action.threshold_margin { 0.0 } else { thresholds[neuron] }) };
                            let l = layer_for(neuron);
                            if let Some([bank, index]) = if action.threshold_margin { None } else { plan.threshold_reference(l, neuron - offsets[l]) } {
                                gradients[bank][index] -= delta[0] * plan.masks[bank][index];
                            }
                        }
                        if let Some(m) = mpi { m.sum(&mut delta)?; }
                        adjoints[b][action.reads[0]] += delta[0];
                    } else {
                        if action.mask.is_some_and(|[bank, index]| plan.masks[bank][index] == 0.0) { continue; }
                        let gate = event_gate(action, context, b, visit, &spikes);
                        poisson_cache::begin(action,b+batch_origin,visit,i,noise_tick,false,false);
                        let mut boundary = [f64::NAN; 16];
                        if boundary_possible && gate != 0. {
                            let noise = samples(plan,uniform_masks[action.program_set.unwrap()],action,
                                noise_sequence,b + batch_origin,noise_tick);
                            let cells = if action.indirect.is_some() {
                                address_tape[(b*visits+visit)*address_width+address_offsets[i]].as_ref().unwrap().read_cells()
                            } else { &action.reads };
                            let (zero,sites) = equation::zero_poisson_sites(&spec.program_sets[action.program_set.unwrap()],
                                context,&state.weights,action.parameter_index,timestamp,&noise,clock_row,&spec.parameter_maps,
                                &plan.masks,cells,&spec.detached)?;
                            for stream in 0..16 {
                                if zero & (1 << stream) != 0 && sites & (1 << stream) == 0 {boundary[stream] = 0.;}
                            }
                            // Every rank visits the same sites and enters each replay's
                            // owner-compute collectives in the same order. Only the
                            // owner will subsequently inject the baseline rate VJP.
                            for stream in 0..16 {
                                if sites & (1 << stream) == 0 {continue;}
                                let alternate = execute_inner(plan,state.clone(),&inputs[b..b+1],&labels[b..b+1],
                                    vec![original[b].clone()],bytes,"evaluate",mpi,noise_sequence,false,itinerary,
                                    Some((visit,i,stream)),b + batch_origin,cache)?;
                                boundary[stream] = alternate.loss / batch as f64 - sample_losses[b];
                            }
                        }
                        if action.indirect.is_some() {
                            let gate = event_gate(action, context, b, visit, &spikes);
                            let differentiate_gate = !action.detach_trigger && action.trigger.as_ref().is_some_and(|g|
                                !g.external && (!g.state || !spec.detached[g.index]));
                            let noise = samples(plan, uniform_masks[action.program_set.unwrap()], action, noise_sequence, b + batch_origin, noise_tick);
                            let eval = indirect::Evaluation { plan, spec, action, weights: &state.weights,
                                timestamp: timestamp,
                                noise: &noise, clocks: clock_row };
                            let record = address_tape[(b*visits+visit)*address_width+address_offsets[i]].as_ref().unwrap();
                            let dg = eval.backward(record, context, &mut adjoints[b], &mut gradients,
                                gate, differentiate_gate, sample_losses[b], &boundary, owns(action), mpi)?;
                            if let Some(g) = &action.trigger {
                                if g.state { if !spec.detached[g.index] { adjoints[b][g.index] += dg; } }
                                else if !g.external { spike_adjoints[g.index] += dg; }
                            }
                            continue;
                        }
                        let c = action.reads.len(); let w = action.writes.len();
                        let mut delta = [0.0; 129];
                        if owns(action) {
                            let gate = event_gate(action, context, b, visit, &spikes);
                            let differentiate_gate = !action.detach_trigger && action.trigger.as_ref().is_some_and(|g|
                                !g.external && (!g.state || !spec.detached[g.index]));
                            let noise = samples(plan, uniform_masks[action.program_set.unwrap()], action, noise_sequence, b + batch_origin, noise_tick);
                            if gate != 0.0 {
                                equation::backward_poisson(&spec.program_sets[action.program_set.unwrap()], context, &state.weights,
                                    sample_losses[b], &boundary, &mut gradients, &plan.masks, &mut delta[..c], action.parameter_index,
                                    timestamp, &noise, clock_row, &spec.parameter_maps, &action.reads, &spec.detached)?;
                            }
                            for (s, program) in spec.program_sets[action.program_set.unwrap()].iter().enumerate() {
                                let target = action.writes[s];
                                if spec.detached[target] { continue; }
                                let g = adjoints[b][target]; if g == 0.0 { continue; }
                                delta[c + s] = g * (1.0 - gate);
                                if gate != 0.0 {
                                    equation::backward_clocked_addresses(program, context, &state.weights, g * gate, &mut gradients,
                                        &plan.masks, &mut delta[..c], action.parameter_index, timestamp, &noise, clock_row, &spec.parameter_maps,
                                        &action.reads, &spec.detached)?;
                                }
                                if differentiate_gate {
                                    let value = equation::forward_clocked(program, context, &state.weights, action.parameter_index,
                                        timestamp, &noise, clock_row, &spec.parameter_maps)?;
                                    let old = context[action.reads.iter().position(|&k| k == target).unwrap()];
                                    delta[c + w] += g * (value - old);
                                }
                            }
                        }
                        if let Some(m) = mpi { m.sum(&mut delta[..c + w + 1])?; }
                        for (j, &k) in action.writes.iter().enumerate() {
                            if !spec.detached[k] { adjoints[b][k] = delta[c + j]; }
                        }
                        for (j, &k) in action.reads.iter().enumerate() {
                            if !spec.detached[k] { adjoints[b][k] += delta[j]; }
                        }
                        if let Some(g) = &action.trigger {
                            if g.state { if !spec.detached[g.index] { adjoints[b][g.index] += delta[c + w]; } }
                            else if !g.external { spike_adjoints[g.index] += delta[c + w]; }
                        }
                    }
                }
                if frame.is_some() && plan.tbptt_window.is_some_and(|w| t > 0 && t % w == 0) { adjoints[b].fill(0.0); }
            }
        }
        if default_initial && mpi.is_none_or(|m| m.rank == 0) {
            for row in &adjoints { for (k, binding) in spec.initial_parameters.iter().enumerate() {
                if let Some([bank, index]) = binding { gradients[*bank][*index] += row[k] * plan.masks[*bank][*index]; }
            }}
        }
    }
    if let Some(m) = mpi { for row in &mut gradients { m.sum(row)?; } }
    ensure(loss.is_finite() && live.iter().flatten().chain(adjoints.iter().flatten()).chain(gradients.iter().flatten()).all(|v| v.is_finite()),
        "nonfinite dynamic training result")?;
    if operation == "train" { apply_optimizer_distributed(plan, &mut state, &gradients, mpi)?; }
    let voltage_rows = |rows: &[Vec<f64>]| rows.iter().map(|row| spec.voltage.iter().map(|&k| row[k]).collect::<Vec<_>>()).collect::<Vec<_>>();
    let (spikes,event_visits)=itinerary::outputs(spec,itinerary,&spikes,batch,time);
    Ok(Output { poisson_state: poisson_cache::checkpoint(), event_visits, clock_state: None, updated_dynamic: None,
        final_tick: None, noise_sequence: None, schema: "b2-lif-training-result-v1", state, loss,
        backend: "cpu", numeric_profile: if mpi.is_some() { "native-mpi-dynamic-actions-owner-f64" } else { "native-cpu-dynamic-actions-f64" },
        gpu_dispatches: 0, gradients, initial_gradients: voltage_rows(&adjoints), final_membrane: voltage_rows(&live),
        initial_state_gradients: Some(adjoints), final_state: Some(live), logits, tape_bytes: bytes,
        spikes,
        gradient_scope: if plan.tbptt_window.is_some_and(|w| w < time) { "tbptt-detach-boundaries" } else { "full-bptt" },
    })
}
