//! Versioned native LIF training plan. Thresholds are hard in the forward pass;
//! only the declared backward plan substitutes a derivative. The synchronous
//! schedule matches Brian groups/thresholds/synapses/resets for additive input
//! with additive input and reset after synaptic accumulation.
use serde::{Deserialize, Serialize};
pub mod equation;
mod math;
mod gpu;
#[cfg(unix)]
pub mod mpi;
#[cfg(not(unix))]
#[path = "training/mpi_unavailable.rs"]
pub mod mpi;
mod multistate;
mod scalar;
mod dynamic;
mod clock;
mod clock_state;
mod itinerary;
mod dynamic_gpu;
mod migration;
mod timed_input;
mod delay;
mod indirect;
mod noise;
mod poisson;
mod poisson_ir;
mod poisson_cache;
mod poisson_retirement;
mod poisson_gpu_cache;
mod poisson_gpu_validate;
mod static_poisson;
mod static_poisson_gpu;
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
/// A collective failure observed by a rank that did not own the error. The
/// CLI lets this peer finalize rather than abort the owner before it reports.
#[derive(Debug)]
pub struct PeerFailure;
impl std::fmt::Display for PeerFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("static state evaluation failed on another MPI rank")
    }
}
impl std::error::Error for PeerFailure {}
fn ensure(value: bool, reason: &str) -> Result<()> {
    if value {
        Ok(())
    } else {
        Err(reason.into())
    }
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Surrogate {
    pub kind: String,
    pub slope: f64,
    pub scale: f64,
}
impl Surrogate {
    pub fn derivative(&self, margin: f64) -> f64 {
        self.scale / (1.0 + self.slope * margin.abs()).powi(2)
    }
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Optimizer {
    pub kind: String,
    pub learning_rate: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
}
/// Ordered edges reference a projection-local parameter bank. Repeated
/// parameter IDs implement tied weights; source_layer 0 is external input.
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Projection {
    pub source_layer: usize,
    pub target_layer: usize,
    pub parameter_count: usize,
    pub sources: Vec<usize>,
    pub targets: Vec<usize>,
    pub parameter_ids: Vec<usize>,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Refractory {
    pub steps: usize,
    pub clamp: Vec<usize>,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Clock {
    pub origin: f64,
    pub dt: f64,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Plan {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dynamic: Option<dynamic::Spec>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub noise_streams: Option<Vec<usize>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub clock: Option<Clock>,
    #[serde(default = "cpu_backend")]
    pub backend: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mpi_ranks: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub equations: Option<Vec<equation::Program>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub state_equations: Option<Vec<Vec<equation::Program>>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub state_resets: Option<Vec<Vec<equation::Program>>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub refractory: Option<Vec<Option<Refractory>>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub threshold_parameters: Option<Vec<Option<[usize; 2]>>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub threshold_per_neuron: Option<Vec<bool>>,
    pub schema: String,
    pub sizes: Vec<usize>,
    pub beta: Vec<f64>,
    pub threshold: Vec<f64>,
    pub reset: String,
    pub detach_reset: bool,
    pub surrogate: Surrogate,
    pub optimizer: Optimizer,
    pub trainable: Vec<bool>,
    pub masks: Vec<Vec<f64>>,
    pub seed: u64,
    pub logit_scale: f64,
    pub max_tape_bytes: usize,
    pub tbptt_window: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub projections: Option<Vec<Projection>>,
}
fn cpu_backend() -> String {
    "cpu".into()
}
#[derive(Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct State {
    pub weights: Vec<Vec<f64>>,
    pub first_moment: Vec<Vec<f64>>,
    pub second_moment: Vec<Vec<f64>>,
    pub step: u64,
    pub rng: u64,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    #[serde(default)]
    pub poisson_state: Option<poisson_cache::Checkpoint>,
    #[serde(default)]
    pub clock_state: Option<clock_state::Checkpoint>,
    #[serde(default)]
    pub delay_update: Option<delay::Update>,
    #[serde(default)]
    pub input_update: Option<timed_input::Update>,
    #[serde(default)]
    pub mask_update: Option<migration::MaskUpdate>,
    #[serde(default)]
    pub noise_sequence: u64,
    #[serde(default)]
    pub start_tick: u64,
    pub plan: Plan,
    pub state: Option<State>,
    pub operation: String,
    pub inputs: Vec<Vec<Vec<f64>>>,
    pub labels: Vec<usize>,
    pub initial: Option<Vec<Vec<f64>>>,
}
#[derive(Serialize)]
pub struct Output {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub poisson_state: Option<poisson_cache::Checkpoint>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub event_visits: Option<itinerary::Events>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub clock_state: Option<clock_state::Checkpoint>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub updated_dynamic: Option<dynamic::Spec>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub noise_sequence: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub final_tick: Option<u64>,
    pub schema: &'static str,
    pub state: State,
    pub loss: f64,
    pub backend: &'static str,
    pub numeric_profile: &'static str,
    pub gpu_dispatches: usize,
    pub gradients: Vec<Vec<f64>>,
    pub initial_gradients: Vec<Vec<f64>>,
    pub final_membrane: Vec<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub final_state: Option<Vec<Vec<f64>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub initial_state_gradients: Option<Vec<Vec<f64>>>,
    pub spikes: Vec<Vec<Vec<f64>>>,
    pub logits: Vec<Vec<f64>>,
    pub tape_bytes: usize,
    pub gradient_scope: &'static str,
}

impl Plan {
    fn noise_at(&self, uniform: u16, sequence: u64, batch: usize, layer: usize, neuron: usize, tick: u64) -> [f64; 16] {
        let mut values = [0.0; 16];
        if let Some(counts) = &self.noise_streams {
            for (stream, value) in values[..counts[layer]].iter_mut().enumerate() {
                *value = noise::sample(uniform & (1 << stream) != 0,self.seed, sequence, batch as u64, layer as u64, neuron as u64, tick, stream as u64);
            }
        }
        values
    }
    fn time_at(&self, tick: u64) -> f64 {
        self.clock.as_ref().map_or(0.0, |c| c.origin + tick as f64 * c.dt)
    }
    fn validate(&self) -> Result<(Vec<usize>, usize)> {
        let vector = self.state_equations.is_some();
        ensure(self.dynamic.is_some() == (self.schema == "b2-dynamic-training-plan-v5"), "dynamic actions require v5")?;
        if let Some(c) = &self.clock {
            ensure((vector || self.equations.is_some()) && c.origin.is_finite() && c.origin >= 0.0 && c.dt.is_finite() && c.dt > 0.0,
                "clock requires equations, finite nonnegative origin and positive dt")?;
        }
        ensure(
            vector == self.state_resets.is_some(),
            "multi-state equations and resets are required together",
        )?;
        ensure(
            vector == matches!(self.schema.as_str(), "b2-state-training-plan-v4" | "b2-dynamic-training-plan-v5"),
            "state programs require v4",
        )?;
        ensure(
            (self.schema == "b2-lif-training-plan-v1"
                && self.projections.is_none()
                && self.equations.is_none())
                || (self.schema == "b2-lif-training-plan-v2"
                    && self.projections.is_some()
                    && self.equations.is_none())
                || (self.schema == "b2-equation-training-plan-v3"
                    && self.projections.is_some()
                    && self.equations.is_some())
                || (matches!(self.schema.as_str(), "b2-state-training-plan-v4" | "b2-dynamic-training-plan-v5")
                    && self.projections.is_some()
                    && self.equations.is_none()),
            "unsupported training schema or projection version",
        )?;
        ensure(
            matches!(self.backend.as_str(), "cpu" | "metal" | "cuda"),
            "unsupported native training backend",
        )?;
        ensure(
            (3..=17).contains(&self.sizes.len()) && self.sizes.iter().all(|&n| n > 0 && n <= 65536),
            "training needs input, hidden and output layers of bounded positive size",
        )?;
        ensure(
            self.mpi_ranks.is_none_or(|r| (2..=256).contains(&r)),
            "MPI training requires 2..256 ranks",
        )?;
        ensure(
            self.mpi_ranks.is_none()
                || self.backend == "cpu"
                || (matches!(self.backend.as_str(), "metal" | "cuda")
                    && self.projections.is_some()),
            "target-owned MPI BPTT requires CPU or GPU projection ranks",
        )?;
        let layers = self.sizes.len() - 1;
        if let Some(counts) = &self.noise_streams {
            ensure((vector || self.equations.is_some()) && self.clock.is_some() && counts.len() == layers
                && counts.iter().all(|&n| n <= 16) && (counts.iter().any(|&n| n > 0) || self.dynamic.is_some()),
                "noise streams require equations, a clock and 0..16 streams per layer")?;
        }
        if let Some(specs) = &self.refractory {
            ensure(vector && specs.len() == layers, "refractory requires v4 and one entry per layer")?;
        }
        if let Some(programs) = &self.state_equations {
            let resets = self.state_resets.as_ref().unwrap();
            ensure(
                programs.len() == layers && resets.len() == layers,
                "state program layer count mismatch",
            )?;
            for (l, (updates, reset)) in programs.iter().zip(resets).enumerate() {
                ensure(
                    (1..=16).contains(&updates.len()) && reset.len() == updates.len(),
                    "each layer needs 1..16 state updates and resets",
                )?;
                let (normal, uniform) = equation::noise_masks(updates.iter().chain(reset));
                ensure(normal & uniform == 0, "noise stream mixes normal and uniform distributions")?;
                for program in updates.iter().chain(reset) {
                    equation::validate_states(program, self, updates.len())?;
                    ensure(self.dynamic.is_some() || !equation::integer_node(program.last().unwrap()),
                        "v4 physical state outputs must be floating point")?;
                    for node in program {
                        if let equation::Node::Noise { stream } | equation::Node::UniformNoise { stream }
                            | equation::Node::Poisson { stream, .. } = node {
                            ensure(self.noise_streams.as_ref().is_some_and(|n| *stream < n[l]),
                                "noise stream exceeds layer declaration")?;
                        }
                        if let equation::Node::NeuronParameter { bank, index } = node {
                            ensure(index.checked_add(self.sizes[l + 1]).is_some_and(|end| end <= self.masks[*bank].len()),
                                "neuron parameter range exceeds bank")?;
                        }
                        if let equation::Node::MappedParameter { mapping, .. } = node {
                            ensure(self.dynamic.as_ref().is_some_and(|d| d.parameter_maps[*mapping].len() == self.sizes[l+1]),
                                "mapped parameter requires one index per layer neuron")?;
                        }
                    }
                }
                for (position, program) in updates.iter().chain(reset).enumerate() {
                    for node in program {
                        if let equation::Node::RefractoryActive { index } = node {
                            ensure(position < updates.len() - 1
                                && *index == updates.len() - 1
                                && self.refractory.as_ref().is_some_and(|r| r[l].is_some()),
                                "refractory activity requires the counter in a physical-state update")?;
                        }
                    }
                }
                if let Some(spec) = self.refractory.as_ref().and_then(|r| r[l].as_ref()) {
                    let counter = updates.len() - 1;
                    ensure(counter > 0 && spec.steps <= (1 << 24)
                        && spec.clamp.iter().all(|&s| s < counter), "invalid refractory steps or clamp")?;
                    for program in [&updates[counter], &reset[counter]] {
                        ensure(matches!(program.as_slice(), [equation::Node::State { index }] if *index == counter),
                            "refractory counter requires identity placeholder")?;
                    }
                    for program in updates[..counter].iter().chain(&reset[..counter]) {
                        ensure(!program.iter().any(|node| matches!(node, equation::Node::State { index } if *index == counter)),
                            "differentiable programs cannot read refractory counter")?;
                    }
                }
            }
        }
        if let Some(programs) = &self.equations {
            ensure(programs.len() == layers, "equation layer count mismatch")?;
            for (l, program) in programs.iter().enumerate() {
                equation::validate_states(program, self, 1)?;
                ensure(!equation::integer_node(program.last().unwrap()),
                    "scalar physical state output must be floating point")?;
                let (normal,uniform)=equation::noise_masks(std::iter::once(program));
                ensure(normal & uniform == 0,"noise stream mixes normal and uniform distributions")?;
                for node in program {
                    if let equation::Node::Noise{stream} | equation::Node::UniformNoise{stream}
                        | equation::Node::Poisson{stream,..}=node {
                        ensure(self.noise_streams.as_ref().is_some_and(|n|*stream<n[l]),
                            "noise stream exceeds layer declaration")?;
                    }
                    if let equation::Node::NeuronParameter{bank,index}=node {
                        ensure(index.checked_add(self.sizes[l+1]).is_some_and(|end|end<=self.masks[*bank].len()),
                            "neuron parameter range exceeds bank")?;
                    }
                    ensure(!matches!(node,equation::Node::RefractoryActive{..}),
                        "scalar refractory activity requires a vector state plan")?;
                }
            }
        }
        static_poisson::validate(self)?;
        if let Some(flags) = &self.threshold_per_neuron {
            ensure(vector && flags.len() == layers && self.threshold_parameters.is_some(),
                "per-neuron thresholds require v4 and one flag per layer")?;
        }
        if let Some(references) = &self.threshold_parameters {
            ensure(
                (self.equations.is_some() || vector) && references.len() == layers,
                "trainable thresholds require v3 and one reference per layer",
            )?;
            for [bank, index] in references.iter().flatten() {
                ensure(
                    *bank < self.masks.len() && *index < self.masks[*bank].len(),
                    "invalid threshold parameter reference",
                )?;
            }
            if let Some(flags) = &self.threshold_per_neuron {
                for (l, &indexed) in flags.iter().enumerate() {
                    if indexed {
                        let [bank, index] = references[l].ok_or("per-neuron threshold requires a parameter reference")?;
                        ensure(index.checked_add(self.sizes[l + 1]).is_some_and(|end| end <= self.masks[bank].len()),
                            "per-neuron threshold range exceeds bank")?;
                    }
                }
            }
        }
        let banks = self.projections.as_ref().map_or(layers, Vec::len);
        ensure(banks > 0 && banks <= 256, "invalid projection count")?;
        ensure(
            self.beta.len() == layers
                && self.threshold.len() == layers
                && self.trainable.len() == banks
                && self.masks.len() == banks,
            "layer configuration shape mismatch",
        )?;
        ensure(
            self.beta
                .iter()
                .all(|&v| v.is_finite() && v > 0.0 && v <= 1.0)
                && self.threshold.iter().all(|&v| v.is_finite() && v > 0.0),
            "invalid LIF beta/threshold",
        )?;
        ensure(
            matches!(self.reset.as_str(), "zero" | "subtract"),
            "unsupported reset",
        )?;
        ensure(
            self.surrogate.kind == "fast_sigmoid"
                && self.surrogate.slope.is_finite()
                && self.surrogate.slope > 0.0
                && self.surrogate.scale.is_finite()
                && self.surrogate.scale > 0.0,
            "unsupported surrogate or invalid parameters",
        )?;
        let opt = &self.optimizer;
        ensure(
            matches!(opt.kind.as_str(), "sgd" | "adam")
                && opt.learning_rate.is_finite()
                && opt.learning_rate > 0.0
                && opt.beta1.is_finite()
                && (0.0..1.0).contains(&opt.beta1)
                && opt.beta2.is_finite()
                && (0.0..1.0).contains(&opt.beta2)
                && opt.epsilon.is_finite()
                && opt.epsilon > 0.0,
            "invalid optimizer",
        )?;
        ensure(
            self.logit_scale.is_finite()
                && self.logit_scale > 0.0
                && self.max_tape_bytes > 0
                && self.max_tape_bytes <= 1024 * 1024 * 1024,
            "invalid training resource or loss configuration",
        )?;
        ensure(
            self.tbptt_window != Some(0),
            "TBPTT window must be positive",
        )?;
        for bank in 0..banks {
            let parameters = if let Some(projections) = &self.projections {
                let p = &projections[bank];
                ensure(
                    p.source_layer <= layers && p.target_layer > 0 && p.target_layer <= layers,
                    "invalid projection layer endpoint",
                )?;
                ensure(
                    (!p.sources.is_empty() || self.equations.is_some() || vector)
                        && p.sources.len() == p.targets.len()
                        && p.sources.len() == p.parameter_ids.len()
                        && p.parameter_count > 0,
                    "invalid projection edge/parameter shape",
                )?;
                ensure(
                    p.sources.iter().all(|&i| i < self.sizes[p.source_layer])
                        && p.targets.iter().all(|&j| j < self.sizes[p.target_layer])
                        && p.parameter_ids.iter().all(|&id| id < p.parameter_count),
                    "projection index outside declared domain",
                )?;
                p.parameter_count
            } else {
                self.sizes[bank]
                    .checked_mul(self.sizes[bank + 1])
                    .ok_or("edge count overflow")?
            };
            ensure(
                self.masks[bank].len() == parameters
                    && self.masks[bank].iter().all(|&v| v == 0.0 || v == 1.0),
                "invalid fixed connection mask",
            )?;
        }
        let mut offsets = vec![0];
        for layer in 0..layers {
            offsets.push(offsets.last().unwrap() + self.sizes[layer + 1]);
        }
        let n = *offsets.last().unwrap();
        if let Some(spec) = &self.dynamic { spec.validate(self, n)?; }
        Ok((offsets, n))
    }
    fn threshold_reference(&self, layer: usize, neuron: usize) -> Option<[usize; 2]> {
        if let Some(references) = self.dynamic.as_ref().and_then(|d| d.threshold_references.as_ref()) {
            return references[self.sizes[1..layer+1].iter().sum::<usize>() + neuron];
        }
        self.threshold_parameters.as_ref().and_then(|p| p[layer]).map(|[bank, index]| {
            [bank, index + if self.threshold_per_neuron.as_ref().is_some_and(|p| p[layer]) { neuron } else { 0 }]
        })
    }
    fn state_width(&self) -> usize {
        if let Some(spec) = &self.dynamic { return spec.initial.len(); }
        self.state_equations.as_ref().map_or_else(
            || self.sizes[1..].iter().sum(),
            |programs| {
                programs
                    .iter()
                    .zip(&self.sizes[1..])
                    .map(|(p, n)| p.len() * n)
                    .sum()
            },
        )
    }
    fn initialize(&self) -> State {
        let mut rng = self.seed;
        let weights: Vec<Vec<f64>> = self
            .masks
            .iter()
            .enumerate()
            .map(|(layer, mask)| {
                mask.iter()
                    .map(|&active| {
                        // Native counter stream, with its continuation state serialized.
                        rng = rng.wrapping_add(0x9e3779b97f4a7c15);
                        let mut z = rng;
                        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
                        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
                        z ^= z >> 31;
                        let draw = (z >> 11) as f64 / 9007199254740992.0;
                        let source = self
                            .projections
                            .as_ref()
                            .map_or(layer, |p| p[layer].source_layer);
                        active * (0.25 + draw) * (2.0 / self.sizes[source] as f64).sqrt()
                    })
                    .collect()
            })
            .collect();
        let zeros = weights
            .iter()
            .map(|w| vec![0.0; w.len()])
            .collect::<Vec<_>>();
        State {
            weights,
            first_moment: zeros.clone(),
            second_moment: zeros,
            step: 0,
            rng,
        }
    }
}

fn validate_state(plan: &Plan, state: &State) -> Result<Vec<f64>> {
    ensure(
        state.weights.len() == plan.masks.len()
            && state.first_moment.len() == plan.masks.len()
            && state.second_moment.len() == plan.masks.len(),
        "optimizer state shape mismatch",
    )?;
    for l in 0..plan.masks.len() {
        for values in [
            &state.weights[l],
            &state.first_moment[l],
            &state.second_moment[l],
        ] {
            ensure(
                values.len() == plan.masks[l].len() && values.iter().all(|v| v.is_finite()),
                "invalid training state",
            )?;
        }
        ensure(
            state.second_moment[l].iter().all(|&v| v >= 0.0),
            "invalid Adam second moment",
        )?;
        ensure(
            (0..state.weights[l].len()).all(|e| {
                plan.masks[l][e] != 0.0
                    || (state.weights[l][e] == 0.0
                        && state.first_moment[l][e] == 0.0
                        && state.second_moment[l][e] == 0.0)
            }),
            "masked edges must have zero weights and optimizer state",
        )?;
    }
    if let Some(spec) = &plan.dynamic {
        ensure(spec.integer_parameters.iter().all(|&[b,i]| equation::int32(state.weights[b][i]).is_ok()),
            "integer parameter value is outside int32")?;
    }
    let thresholds = (0..plan.sizes.len() - 1)
        .map(|l| {
            plan.threshold_parameters
                .as_ref()
                .and_then(|p| p[l])
                .map_or(plan.threshold[l], |[b, i]| state.weights[b][i])
        })
        .collect::<Vec<_>>();
    ensure(
        thresholds.iter().all(|v| v.is_finite() && *v > 0.0),
        "learned thresholds must remain finite and positive",
    )?;
    for l in 0..plan.sizes.len() - 1 {
        if plan.threshold_per_neuron.as_ref().is_some_and(|p| p[l]) || plan.dynamic.as_ref().is_some_and(|d| d.threshold_references.is_some()) {
            for j in 0..plan.sizes[l + 1] {
                if let Some([bank, index]) = plan.threshold_reference(l, j) {
                    ensure(state.weights[bank][index] > 0.0, "learned thresholds must remain finite and positive")?;
                }
            }
        }
    }
    Ok(thresholds)
}

fn validate_live(plan: &Plan, membrane: &[Vec<f64>]) -> Result<()> {
    ensure(
        !membrane.is_empty() && membrane.len() <= 4096
            && membrane
                .iter()
                .all(|row| row.len() == plan.state_width() && row.iter().all(|v| v.is_finite())),
        "invalid initial membrane",
    )?;
    if let Some(specs) = &plan.refractory {
        let mut offset = 0;
        for (l, updates) in plan.state_equations.as_ref().unwrap().iter().enumerate() {
            let size = plan.sizes[l + 1];
            if let Some(spec) = &specs[l] {
                let start = offset + (updates.len() - 1) * size;
                ensure(membrane.iter().all(|row| row[start..start + size].iter()
                    .all(|&x| x >= 0.0 && x <= spec.steps as f64 && x.fract() == 0.0)),
                    "invalid initial refractory counter")?;
            }
            offset += updates.len() * size;
        }
    }
    if let Some(spec) = &plan.dynamic {
        if let Some(sample) = &spec.sample_index {
            ensure(membrane.len() == sample.samples && membrane.iter().enumerate()
                .all(|(b,row)| row[sample.slot] == b as f64),
                "per-sample input batch or native sample identity mismatch")?;
        }
        ensure(membrane.iter().all(|row| spec.binary_states.iter().all(|&k| row[k] == 0.0 || row[k] == 1.0)),
            "dynamic event history must be binary")?;
        ensure(membrane.iter().all(|row| spec.integer_states.iter().all(|&k| equation::int32(row[k]).is_ok())),
            "integer live state is outside int32")?;
        if let Some(layout) = &spec.delay_layout { layout.validate_live(spec, membrane)?; }
    }
    Ok(())
}

pub fn execute(request: Request) -> Result<Output> {
    execute_distributed(request, None)
}

pub fn execute_distributed(request: Request, mpi: Option<&mpi::Context>) -> Result<Output> {
    let retirement_bytes = if request.operation == "retire_poisson_history" {
        request.plan.validate()?;
        let live=request.initial.as_deref().ok_or("Poisson retirement requires committed physical state")?;
        validate_live(&request.plan, live)?;
        Some(poisson_retirement::admission(&request.plan,
            request.poisson_state.as_ref().ok_or("Poisson retirement requires draw state")?, live)?)
    } else { None };
    let retired = request.poisson_state.as_ref().is_some_and(|s| s.continuation.is_some())
        || request.operation == "retire_poisson_history";
    if let Some(boundary) = request.poisson_state.as_ref().and_then(|s| s.continuation.as_ref()) {
        boundary.verify(&request.plan, request.initial.as_deref().ok_or("retired Poisson history requires its committed physical state")?,
            request.start_tick, request.clock_state.as_ref())?;
    }
    let plan = retired.then(|| request.plan.clone());
    let mut output = execute_inner(request, mpi)?;
    if let Some(bytes)=retirement_bytes { output.tape_bytes=bytes; }
    if let Some(mut plan) = plan {
        if let Some(spec) = &output.updated_dynamic { plan.dynamic = Some(spec.clone()); }
        if let Some(cache) = &mut output.poisson_state {
            let live = output.final_state.as_deref().unwrap_or(&output.final_membrane);
            cache.continuation = Some(poisson_retirement::Continuation::new(&plan, live,
                output.final_tick.ok_or("retired Poisson result requires a clock")?, output.clock_state.as_ref())?);
        }
    }
    Ok(output)
}

fn execute_inner(request: Request, mpi: Option<&mpi::Context>) -> Result<Output> {
    let Request {
        mut poisson_state,
        clock_state,
        delay_update,
        input_update,
        mask_update,
        noise_sequence,
        start_tick,
        mut plan,
        state,
        operation,
        inputs,
        labels,
        initial,
    } = request;
    let (_, n) = plan.validate()?;
    ensure(plan.noise_streams.is_some() || noise_sequence == 0, "noise_sequence requires noise streams")?;
    ensure(noise_sequence < u64::MAX, "noise sequence overflow")?;
    ensure(
        plan.mpi_ranks == mpi.map(|m| m.size),
        "MPI training plan/launch rank mismatch",
    )?;
    let validation_dispatches=if let Some(cache)=&poisson_state {
        ensure(plan.dynamic.is_some() || static_poisson::used(&plan),"Poisson draw state requires a Poisson execution plan")?;
        let batch=initial.as_ref().map_or(inputs.len(),Vec::len);
        let result=cache.validate(&plan,noise_sequence,batch);
        if let Some(m)=mpi {let mut failed=[f64::from(result.is_err())];m.sum(&mut failed)?;
            if failed[0]!=0. {return Err(result.err().unwrap_or_else(||"Poisson checkpoint validation failed on another rank".into()));}}
        result?
    }else{0};
    let boundary_times = clock_state::boundary(&plan, start_tick, clock_state.as_ref())?;
    if operation == "validate_clock_state" || operation == "validate_poisson_state" || operation == "retire_poisson_history" {
        ensure(inputs.is_empty() && labels.is_empty() && delay_update.is_none() && input_update.is_none() && mask_update.is_none() && (clock_state.is_some() || poisson_state.is_some()),
            "invalid clock validation request")?;
        let state = state.ok_or("clock validation requires optimizer state")?;
        validate_state(&plan, &state)?;
        if let Some(live) = &initial { validate_live(&plan, live)?; }
        if operation == "retire_poisson_history" {
            let live=initial.as_deref().ok_or("Poisson retirement requires committed physical state")?;
            poisson_retirement::retire(&plan, poisson_state.as_mut().ok_or("Poisson retirement requires draw state")?,
                live, start_tick, clock_state.as_ref())?;
        }
        return Ok(Output { poisson_state, event_visits: None, clock_state, updated_dynamic: None, final_tick: Some(start_tick), noise_sequence: Some(noise_sequence),
            schema: "b2-lif-training-result-v1", state, loss: 0.0, backend: if plan.backend=="cuda" {"cuda"}else if plan.backend=="metal" {"metal"}else{"cpu"}, numeric_profile: "native-clock-checkpoint-validation",
            gpu_dispatches: validation_dispatches, gradients: vec![], initial_gradients: vec![], final_membrane: vec![], final_state: initial,
            initial_state_gradients: None, spikes: vec![], logits: vec![], tape_bytes: 0, gradient_scope: "validation-no-gradient" });
    }
    if operation == "update_delays" {
        ensure(inputs.is_empty() && labels.is_empty() && input_update.is_none() && mask_update.is_none(), "delay update must not include training or other updates")?;
        let mut result = delay::execute(plan, state.ok_or("delay update requires optimizer state")?,
            initial.ok_or("delay update requires committed runtime state")?,
            delay_update.ok_or("missing delay update")?, start_tick, noise_sequence)?;
        result.clock_state = clock_state; result.poisson_state=poisson_state; result.gpu_dispatches+=validation_dispatches; return Ok(result);
    }
    ensure(delay_update.is_none(), "delay update requires a delay boundary")?;
    if operation == "update_timed_input" {
        ensure(inputs.is_empty() && labels.is_empty() && mask_update.is_none(), "input update must not include training or migration data")?;
        let mut result = timed_input::execute(plan, state.ok_or("input update requires optimizer state")?, initial,
            input_update.ok_or("missing timed input update")?, start_tick, noise_sequence,poisson_state.as_ref())?;
        result.clock_state = clock_state; result.poisson_state=poisson_state; result.gpu_dispatches+=validation_dispatches; return Ok(result);
    }
    ensure(input_update.is_none(), "input update is only valid at an input boundary")?;
    if operation == "update_mask" {
        ensure(inputs.is_empty() && labels.is_empty(), "mask migration must not include training data")?;
        let mut result = migration::execute(plan, state.ok_or("mask migration requires optimizer state")?,
            initial.ok_or("dynamic mask migration requires a committed runtime state")?,
            mask_update.ok_or("missing mask migration request")?, start_tick, noise_sequence, boundary_times)?;
        result.clock_state = clock_state; result.poisson_state=poisson_state; result.gpu_dispatches+=validation_dispatches; return Ok(result);
    }
    ensure(mask_update.is_none(), "mask update is only valid for structural migration")?;
    ensure(
        matches!(operation.as_str(), "train" | "evaluate" | "gradients"),
        "unsupported native training operation",
    )?;
    let layers = plan.sizes.len() - 1;
    let batch = inputs.len();
    let time = inputs.first().map_or(0, Vec::len);
    ensure(
        batch > 0 && batch <= 4096 && time > 0 && time <= 1_000_000 && labels.len() == batch,
        "invalid batch/time/labels",
    )?;
    let final_tick = start_tick.checked_add(time as u64)
        .filter(|&t| t <= (1u64 << 53)).ok_or("clock tick overflow")?;
    ensure(plan.clock.is_some() || start_tick == 0, "start_tick requires a clock")?;
    if plan.clock.is_some() {
        ensure(plan.time_at(final_tick).is_finite()
            && (start_tick..final_tick).all(|t| plan.time_at(t + 1) > plan.time_at(t)),
            "clock time overflow or insufficient precision")?;
    }
    ensure(
        inputs.iter().all(|sample| {
            sample.len() == time
                && sample
                    .iter()
                    .all(|row| row.len() == plan.sizes[0] && row.iter().all(|v| v.is_finite()))
        }) && labels.iter().all(|&label| label < plan.sizes[layers]),
        "invalid input shape/value or label",
    )?;
    let original_width = plan.state_width();
    let initial_bytes = original_width.checked_mul(batch).and_then(|n| n.checked_mul(64))
        .and_then(|n| plan.masks.iter().try_fold(n, |n,r| n.checked_add(r.len().checked_mul(48)?)))
        .ok_or("initial state memory overflow")?;
    ensure(initial_bytes <= plan.max_tape_bytes, "native initial state budget exceeded")?;
    let state = state.unwrap_or_else(|| plan.initialize());
    validate_state(&plan, &state)?;
    let default_initial = initial.is_none();
    let mut membrane = initial.unwrap_or_else(|| {
        if let Some(spec) = &plan.dynamic {
            let mut row = spec.initial.clone();
            for (i, reference) in spec.initial_parameters.iter().enumerate() {
                if let Some([bank, index]) = reference { row[i] = state.weights[*bank][*index]; }
            }
            (0..batch).map(|b| {
                let mut value=row.clone();
                if let Some(sample)=&spec.sample_index {value[sample.slot]=b as f64;}
                value
            }).collect()
        } else { vec![vec![0.0; original_width]; batch] }
    });
    ensure(membrane.len() == batch, "invalid initial membrane batch")?;
    validate_live(&plan, &membrane)?;
    let mut delay_mapping = None;
    let mut preparation_bytes = 0;
    if let Some(prepared) = delay::prepare(&plan, &membrane)? {
        let delay::Prepared { plan:next_plan, live, mapping, input_width, bytes } = prepared;
        plan = next_plan; membrane = live;
        delay_mapping = Some((mapping, input_width)); preparation_bytes = bytes;
    }
    let visits = if let Some(spec)=&plan.dynamic {
        itinerary::count(spec,start_tick,time,clock_state.as_ref())?
    } else {time};
    let cells = batch
        .checked_mul(time)
        .and_then(|v| v.checked_mul(n))
        .ok_or("tape size overflow")?;
    let edges = plan.masks.iter().map(Vec::len).sum::<usize>();
    let topology_bytes = plan.projections.as_ref().map_or(0, |ps| {
        ps.iter().map(|p| p.sources.len() * 24 + 128).sum::<usize>()
    });
    // Account for tape, returned spikes, gradients, optimizer and live state.
    let mpi_bytes = mpi.map_or(0, |m| {
        (n.max(plan.masks.iter().map(Vec::len).max().unwrap_or(0))) * 8 * (m.size + 1)
    });
    let width = plan.state_width();
    let bytes = cells
        .checked_mul(if plan.projections.is_some() { 32 } else { 24 })
        .and_then(|v| v.checked_add(batch * n * 32))
        .and_then(|v| v.checked_add(batch * plan.sizes[layers] * 16))
        .and_then(|v| v.checked_add(edges * 48))
        .and_then(|v| v.checked_add(topology_bytes))
        .and_then(|v| v.checked_add(mpi_bytes))
        .and_then(|v| {
            // Two full vector tapes, vector scratch/carry/output copies,
            // SSA metadata and rank-gather workspace, before any tape allocation.
            if let Some(programs) = &plan.state_equations {
                let extra = batch * time * width * 16
                    + batch * width * 64
                    + programs.iter().map(Vec::len).sum::<usize>() * 128 * 128
                    + width * 8 * mpi.map_or(0, |m| m.size + 1)
                    + 8192;
                v.checked_add(extra)
            } else {
                Some(v)
            }
        })
        .and_then(|v| {
            v.checked_add(if plan.equations.is_some() {
                cells * 8 + layers * 128 * 64 + 2048
            } else {
                0
            })
        })
        .and_then(|v| v.checked_add(batch * time * (plan.sizes[0] * 8 + 48) + batch * 8))
        .and_then(|v| if let Some(spec) = &plan.dynamic { v.checked_add(spec.memory_bytes(batch, visits, mpi.map_or(1, |m| m.size))?) } else { Some(v) })
        .and_then(|v| if plan.backend=="cpu" {if let Some(spec)=&plan.dynamic {v.checked_add(poisson_cache::memory_bytes(spec,batch,visits,poisson_state.as_ref())?)}else{Some(v)}}else{Some(v)})
        .and_then(|v| v.checked_add(static_poisson::memory_bytes(&plan,batch,time,poisson_state.as_ref())?))
        .and_then(|v| if let Some(spec)=&plan.dynamic {v.checked_add(itinerary::Tape::bytes(spec.clocks.as_ref().map_or(1,|c|c.dts.len()),visits)?)} else {Some(v)})
        .and_then(|v| v.checked_add(delay_mapping.as_ref().map_or(0, |(m,_)| m.len()*16)))
        .ok_or("training memory overflow")?;
    // One baseline tape plus one sequential counterfactual replay at most.
    // Reserve the full second execution footprint before either is allocated;
    // actual replay count affects runtime, not peak storage.
    let bytes = if plan.backend == "cpu" && operation != "evaluate"
        && (plan.dynamic.as_ref().is_some_and(poisson_ir::boundary_possible) || static_poisson::boundary_possible(&plan)) {
        bytes.checked_mul(2).ok_or("Poisson boundary replay memory overflow")?
    } else {bytes};
    ensure(
        bytes <= plan.max_tape_bytes,
        "native training tape budget exceeded",
    )?;
    if plan.dynamic.is_some() {
        let itinerary = itinerary::build(&plan,start_tick,time,clock_state.as_ref(),visits)?;
        let mut output = if plan.backend == "cpu" {
            dynamic::execute(&plan, state, &inputs, &labels, membrane, bytes, &operation, mpi, start_tick, noise_sequence, default_initial, &itinerary, poisson_state.as_ref())?
        } else {
            dynamic_gpu::execute(&plan, state, &inputs, &labels, &membrane, bytes, &operation, mpi, start_tick, noise_sequence, default_initial, &itinerary,poisson_state.as_ref())?
        };
        output.gpu_dispatches+=validation_dispatches;
        output.clock_state = itinerary.checkpoint;
        if let Some(spec)=plan.dynamic.as_ref().filter(|s|s.clear_spike_buffers) {
            if let Some(rows)=&mut output.final_state {for row in rows {for &k in &spec.spike_buffers {row[k]=0.0;}}}
        }
        output.final_tick = plan.clock.as_ref().map(|_| final_tick);
        output.noise_sequence = plan.noise_streams.as_ref().map(|_| noise_sequence);
        if plan.dynamic.as_ref().and_then(|s| s.delay_layout.as_ref()).is_some_and(delay::Layout::runtime) {
            output.gradient_scope = if plan.tbptt_window.is_some_and(|w| w < time) {
                "tbptt-detach-boundaries-and-delay-routing"
            } else { "full-bptt-detached-delay-routing" };
        }
        if plan.dynamic.as_ref().is_some_and(|s| s.actions.iter().any(|a| a.indirect.is_some())
            || s.program_sets.iter().flatten().flatten().any(|node|
                matches!(node, equation::Node::ParameterGather { .. } | equation::Node::IntegerParameterGather { .. }))) {
            let delayed = plan.dynamic.as_ref().and_then(|s| s.delay_layout.as_ref()).is_some_and(delay::Layout::runtime);
            output.gradient_scope = match (plan.tbptt_window.is_some_and(|w| w < time), delayed) {
                (false, false) => "full-bptt-detached-index-routing",
                (true, false) => "tbptt-detach-boundaries-and-index-routing",
                (false, true) => "full-bptt-detached-delay-and-index-routing",
                (true, true) => "tbptt-detach-boundaries-delay-and-index-routing",
            };
        }
        if let Some((mapping, input_width)) = delay_mapping {
            if let Some(rows) = &mut output.initial_state_gradients {
                for row in rows {
                    let mut original = vec![0.0; input_width];
                    for (&value, index) in row.iter().zip(&mapping) {
                        if let Some(index) = index { original[*index] += value; }
                    }
                    *row = original;
                }
            }
            output.tape_bytes = output.tape_bytes.max(preparation_bytes);
            output.updated_dynamic = plan.dynamic.take();
        }
        return Ok(output);
    }
    if plan.state_equations.is_some() && plan.backend == "cpu" {
        let mut output = multistate::execute(
            &plan, state, &inputs, &labels, membrane, bytes, &operation, mpi, start_tick, noise_sequence, poisson_state.as_ref(),
        )?;
        output.final_tick = plan.clock.as_ref().map(|_| final_tick);
        output.noise_sequence = plan.noise_streams.as_ref().map(|_| noise_sequence);
        return Ok(output);
    }
    if plan.backend == "metal" || plan.backend == "cuda" {
        let mut output = gpu::execute(
            &plan, state, &inputs, &labels, &membrane, bytes, &operation, mpi, start_tick, noise_sequence,poisson_state.as_ref(),
        )?;
        output.gpu_dispatches+=validation_dispatches;
        output.final_tick = plan.clock.as_ref().map(|_| final_tick);
        output.noise_sequence = plan.noise_streams.as_ref().map(|_| noise_sequence);
        return Ok(output);
    }
    scalar::execute(&plan,state,&inputs,&labels,membrane,bytes,&operation,mpi,
        start_tick,noise_sequence,poisson_state.as_ref())
}

fn apply_optimizer_distributed(
    plan: &Plan,
    state: &mut State,
    gradients: &[Vec<f64>],
    mpi: Option<&mpi::Context>,
) -> Result<()> {
    let count = state.weights.iter().map(Vec::len).sum::<usize>();
    let mut index = 0;
    state.step = state.step.checked_add(1).ok_or("optimizer step overflow")?;
    let opt = &plan.optimizer;
    for l in 0..state.weights.len() {
        for e in 0..state.weights[l].len() {
            let owned = mpi.is_none_or(|m| m.owns(index, count));
            index += 1;
            if !owned {
                state.weights[l][e] = 0.0;
                state.first_moment[l][e] = 0.0;
                state.second_moment[l][e] = 0.0;
                continue;
            }
            if !plan.trainable[l] || plan.masks[l][e] == 0.0 {
                continue;
            }
            let g = gradients[l][e];
            ensure(g.is_finite(), "nonfinite gradient")?;
            let update = if opt.kind == "sgd" {
                g
            } else {
                state.first_moment[l][e] =
                    opt.beta1 * state.first_moment[l][e] + (1.0 - opt.beta1) * g;
                state.second_moment[l][e] =
                    opt.beta2 * state.second_moment[l][e] + (1.0 - opt.beta2) * g * g;
                ensure(
                    state.first_moment[l][e].is_finite() && state.second_moment[l][e].is_finite(),
                    "nonfinite optimizer moment",
                )?;
                let m = state.first_moment[l][e] / (1.0 - opt.beta1.powf(state.step as f64));
                let v = state.second_moment[l][e] / (1.0 - opt.beta2.powf(state.step as f64));
                m / (v.sqrt() + opt.epsilon)
            };
            state.weights[l][e] -= opt.learning_rate * update;
            ensure(
                state.weights[l][e].is_finite(),
                "nonfinite optimizer weight",
            )?;
        }
    }
    if let Some(m) = mpi {
        for bank in 0..state.weights.len() {
            m.sum(&mut state.weights[bank])?;
            m.sum(&mut state.first_moment[bank])?;
            m.sum(&mut state.second_moment[bank])?;
        }
    }
    Ok(())
}
