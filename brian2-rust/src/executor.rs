//! Compile validated expressions once, then run bounded SoA tiles. String lookup
//! and AST traversal are confined to plan construction, outside the tick loop.
use super::*;
use crate::large_topology::{
    build_fixed_indegree_indices, build_fixed_total_indices, materialize_clipped_normal,
    materialize_uniform,
};
use std::sync::Arc;

const LANES: usize = 256;

pub(super) type MaterializedProjection = (
    Vec<usize>,
    Vec<usize>,
    BTreeMap<String, Vec<f64>>,
    Vec<Vec<usize>>,
);

pub(super) fn materialize_projection(
    d: &Definition,
    def: &SynapseDefinition,
    instance: &SynapseInstance,
) -> Result<MaterializedProjection> {
    let (source, target, generated_parameters, generated_delays) = match &instance.topology {
        TopologyInstance::Explicit => (
            instance.source.clone(),
            instance.target.clone(),
            BTreeMap::new(),
            instance
                .pathways
                .iter()
                .map(|pathway| pathway.delay_ticks.clone())
                .collect(),
        ),
        TopologyInstance::BinaryCsr {
            edge_count,
            path,
            column_count,
            initializers,
            sha256,
        } => {
            let csr = binary_topology::read_binary_csr(
                Path::new(path),
                def.source_count,
                def.target_count,
                *edge_count,
                *column_count,
                sha256,
                true,
            )?;
            let parameters = initializers
                .iter()
                .map(|(name, column)| (name.clone(), csr.columns[*column].clone()))
                .collect();
            let delays = instance
                .pathways
                .iter()
                .map(|p| p.delay_ticks.clone())
                .collect();
            (csr.source, csr.target, parameters, delays)
        }
        TopologyInstance::FixedTotal {
            edge_count,
            seed,
            initializers,
        } => {
            let (source, target) =
                build_fixed_total_indices(def.source_count, def.target_count, *edge_count, *seed)?;
            let mut parameters = BTreeMap::new();
            for (name, initializer) in initializers {
                parameters.insert(
                    name.clone(),
                    materialize_initializer(*edge_count, *seed, initializer)?,
                );
            }
            let mut delays = Vec::with_capacity(instance.pathways.len());
            for pathway in &instance.pathways {
                if let Some(initializer) = &pathway.delay_initializer {
                    let dt = if pathway.kind == "pre" {
                        decode_bits(&d.populations[def.source_population].dt)?
                    } else {
                        decode_bits(&d.populations[def.target_population].dt)?
                    };
                    delays.push(
                        materialize_delay_initializer(*edge_count, *seed, initializer, dt)?
                            .into_iter()
                            .map(|delay| (delay / dt + 0.5).floor() as usize)
                            .collect(),
                    );
                } else {
                    delays.push(pathway.delay_ticks.clone());
                }
            }
            (source, target, parameters, delays)
        }
        TopologyInstance::FixedIndegree {
            edge_count,
            indegree,
            seed,
            initializers,
        } => {
            let (source, target) =
                build_fixed_indegree_indices(def.source_count, def.target_count, *indegree, *seed)?;
            check(
                source.len() == *edge_count,
                "fixed-indegree edge count mismatch",
            )?;
            let mut parameters = BTreeMap::new();
            for (name, initializer) in initializers {
                parameters.insert(
                    name.clone(),
                    materialize_initializer(*edge_count, *seed, initializer)?,
                );
            }
            let mut delays = Vec::with_capacity(instance.pathways.len());
            for pathway in &instance.pathways {
                if let Some(initializer) = &pathway.delay_initializer {
                    let dt = if pathway.kind == "pre" {
                        decode_bits(&d.populations[def.source_population].dt)?
                    } else {
                        decode_bits(&d.populations[def.target_population].dt)?
                    };
                    delays.push(
                        materialize_delay_initializer(*edge_count, *seed, initializer, dt)?
                            .into_iter()
                            .map(|delay| (delay / dt + 0.5).floor() as usize)
                            .collect(),
                    );
                } else {
                    delays.push(pathway.delay_ticks.clone());
                }
            }
            (source, target, parameters, delays)
        }
    };
    Ok((source, target, generated_parameters, generated_delays))
}

fn materialize_initializer(
    edge_count: usize,
    seed: u64,
    initializer: &Initializer,
) -> Result<Vec<f64>> {
    match initializer.values()? {
        InitializerValues::ClippedNormal {
            mean,
            std,
            minimum,
            maximum,
            stream,
        } => materialize_clipped_normal(edge_count, seed, stream, mean, std, minimum, maximum),
        InitializerValues::Uniform {
            minimum,
            maximum,
            stream,
        } => materialize_uniform(edge_count, seed, stream, minimum, maximum),
    }
}

fn materialize_delay_initializer(
    edge_count: usize,
    seed: u64,
    initializer: &Initializer,
    dt: f64,
) -> Result<Vec<f64>> {
    match initializer.values()? {
        InitializerValues::ClippedNormal {
            mean,
            std,
            minimum,
            maximum,
            stream,
        } => materialize_clipped_normal(
            edge_count,
            seed,
            stream,
            mean,
            std,
            Some(minimum.unwrap_or(0.0).max(0.0)),
            Some(maximum.unwrap_or(dt * 1_000_000.0)),
        ),
        InitializerValues::Uniform {
            minimum,
            maximum,
            stream,
        } => materialize_uniform(edge_count, seed, stream, minimum, maximum),
    }
}

fn mix64(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn counter_uniform_draw(seed: u64, stream: u64, tick: u64, index: u64, draw: u64) -> f64 {
    let counter = seed
        ^ stream.wrapping_mul(0x9e37_79b9_7f4a_7c15)
        ^ tick.wrapping_mul(0xd1b5_4a32_d192_ed03)
        ^ index.wrapping_mul(0x94d0_49bb_1331_11eb)
        ^ draw.wrapping_mul(0x369d_ea0f_31a5_3f85);
    ((mix64(counter) >> 11) as f64) * (1.0 / 9_007_199_254_740_992.0)
}

fn counter_uniform(seed: u64, stream: u64, tick: u64, index: u64) -> f64 {
    counter_uniform_draw(seed, stream, tick, index, 0)
}

fn counter_normal(seed: u64, stream: u64, tick: u64, index: u64) -> f64 {
    let pair = index / 2;
    let mut draw = 0u64;
    loop {
        let x1 = 2.0 * counter_uniform_draw(seed, stream, tick, pair, draw) - 1.0;
        let x2 = 2.0 * counter_uniform_draw(seed, stream, tick, pair, draw + 1) - 1.0;
        let radius = x1 * x1 + x2 * x2;
        if radius < 1.0 && radius != 0.0 {
            let factor = (-2.0 * radius.ln() / radius).sqrt();
            return factor * if index & 1 == 0 { x1 } else { x2 };
        }
        draw = draw.wrapping_add(2);
    }
}

fn counter_binomial(
    seed: u64,
    stream: u64,
    tick: u64,
    index: u64,
    n: u64,
    p: f64,
    approximate: bool,
) -> f64 {
    if p <= 0.0 {
        return 0.0;
    }
    if p >= 1.0 {
        return n as f64;
    }
    let loc = n as f64 * p;
    let complement = n as f64 * (1.0 - p);
    if approximate && loc > 5.0 && complement > 5.0 {
        return counter_normal(seed, stream, tick, index) * (loc * (1.0 - p)).sqrt() + loc;
    }
    let reverse = p > 0.5;
    let probability = if reverse { 1.0 - p } else { p };
    let q = 1.0 - probability;
    let qn = (n as f64 * q.ln()).exp();
    let bound =
        (n as f64).min(n as f64 * probability + 10.0 * (n as f64 * probability * q + 1.0).sqrt());
    let mut draw = 0;
    'sample: loop {
        let mut x = 0u64;
        let mut px = qn;
        let mut u = counter_uniform_draw(seed, stream, tick, index, draw);
        loop {
            if u <= px {
                return if reverse { n - x } else { x } as f64;
            }
            x += 1;
            if x as f64 > bound {
                draw = draw.wrapping_add(1);
                continue 'sample;
            }
            u -= px;
            px = ((n - x + 1) as f64 * probability * px) / (x as f64 * q);
        }
    }
}

fn log_gamma_positive(value: f64) -> f64 {
    const COEFFICIENTS: [f64; 8] = [
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    let z = value - 1.0;
    let mut x = 0.999_999_999_999_809_9;
    for (index, coefficient) in COEFFICIENTS.iter().enumerate() {
        x += coefficient / (z + index as f64 + 1.0);
    }
    let t = z + 7.5;
    0.918_938_533_204_672_7 + (z + 0.5) * t.ln() - t + x.ln()
}

fn counter_poisson(seed: u64, stream: u64, tick: u64, index: u64, lambda: f64) -> f64 {
    if lambda == 0.0 {
        return 0.0;
    }
    if !lambda.is_finite() || !(0.0..=1.0e12).contains(&lambda) {
        return f64::NAN;
    }
    if lambda < 10.0 {
        let limit = (-lambda).exp();
        let mut product = 1.0;
        let mut sample = 0u64;
        loop {
            product *= counter_uniform_draw(seed, stream, tick, index, sample);
            if product <= limit {
                return sample as f64;
            }
            sample += 1;
        }
    }
    // Hörmann's transformed-rejection method (PTRS). It is exact up to the
    // floating-point evaluation of the acceptance test and avoids O(lambda)
    // work for large rates.
    let root = lambda.sqrt();
    let b = 0.931 + 2.53 * root;
    let a = -0.059 + 0.02483 * b;
    let inverse_alpha = 1.1239 + 1.1328 / (b - 3.4);
    let squeeze = 0.9277 - 3.6224 / (b - 2.0);
    let mut draw = 0u64;
    loop {
        let u = counter_uniform_draw(seed, stream, tick, index, draw) - 0.5;
        let v = counter_uniform_draw(seed, stream, tick, index, draw + 1);
        draw = draw.wrapping_add(2);
        let us = 0.5 - u.abs();
        let candidate = ((2.0 * a / us + b) * u + lambda + 0.43).floor();
        if us >= 0.07 && v <= squeeze {
            return candidate;
        }
        if candidate < 0.0 || (us < 0.013 && v > us) {
            continue;
        }
        let left = (v * inverse_alpha / (a / (us * us) + b)).ln();
        let right = -lambda + candidate * lambda.ln() - log_gamma_positive(candidate + 1.0);
        if left <= right {
            return candidate;
        }
    }
}

#[derive(Clone, Copy)]
enum Domain {
    Neuron,
    Pre,
    Post,
    Edge,
}

enum StateArray {
    Bool(Vec<u8>),
    F32(Vec<f32>),
    F64(Vec<f64>),
    I32(Vec<i32>),
    I64(Vec<i64>),
    U32(Vec<u32>),
    U64(Vec<u64>),
}

impl StateArray {
    fn empty(dtype: DType) -> Result<Self> {
        match dtype {
            DType::Bool => Ok(Self::Bool(Vec::new())),
            DType::F32 => Ok(Self::F32(Vec::new())),
            DType::F64 => Ok(Self::F64(Vec::new())),
            DType::I32 => Ok(Self::I32(Vec::new())),
            DType::I64 => Ok(Self::I64(Vec::new())),
            DType::U32 => Ok(Self::U32(Vec::new())),
            DType::U64 => Ok(Self::U64(Vec::new())),
            _ => Err("unsupported public array dtype".into()),
        }
    }

    fn from_encoded(symbol: &Symbol, values: &EncodedArray) -> Result<Self> {
        Ok(match symbol.dtype {
            DType::Bool => Self::Bool(
                values
                    .iter()
                    .map(|value| match value.as_str() {
                        "00" => Ok(0),
                        "01" => Ok(1),
                        _ => Err("bool must be encoded as 00 or 01".into()),
                    })
                    .collect::<Result<_>>()?,
            ),
            DType::F32 => Self::F32(
                values
                    .iter()
                    .map(|value| decode_f32_bits(value))
                    .collect::<Result<_>>()?,
            ),
            DType::F64 => Self::F64(
                values
                    .iter()
                    .map(|value| decode_bits(value))
                    .collect::<Result<_>>()?,
            ),
            DType::I32 => Self::I32(
                values
                    .iter()
                    .map(|value| Ok(u32::from_str_radix(value, 16)? as i32))
                    .collect::<Result<_>>()?,
            ),
            DType::I64 => Self::I64(
                values
                    .iter()
                    .map(|value| Ok(u64::from_str_radix(value, 16)? as i64))
                    .collect::<Result<_>>()?,
            ),
            DType::U32 => Self::U32(
                values
                    .iter()
                    .map(|value| Ok(u32::from_str_radix(value, 16)?))
                    .collect::<Result<_>>()?,
            ),
            DType::U64 => Self::U64(
                values
                    .iter()
                    .map(|value| Ok(u64::from_str_radix(value, 16)?))
                    .collect::<Result<_>>()?,
            ),
            _ => return Err("unsupported public state dtype".into()),
        })
    }

    fn get_register(&self, index: usize) -> f64 {
        match self {
            Self::Bool(values) => f64::from(values[index]),
            Self::F32(values) => values[index] as f64,
            Self::F64(values) => values[index],
            Self::I32(values) => f64::from_bits(values[index] as u32 as u64),
            Self::I64(values) => f64::from_bits(values[index] as u64),
            Self::U32(values) => f64::from_bits(values[index] as u64),
            Self::U64(values) => f64::from_bits(values[index]),
        }
    }

    fn get_index(&self, index: usize) -> Result<usize> {
        let value = match self {
            Self::I32(values) => usize::try_from(values[index])?,
            Self::I64(values) => usize::try_from(values[index])?,
            Self::U32(values) => usize::try_from(values[index])?,
            Self::U64(values) => usize::try_from(values[index])?,
            _ => return Err("linked mapping state must be integer".into()),
        };
        Ok(value)
    }

    fn len(&self) -> usize {
        match self {
            Self::Bool(values) => values.len(),
            Self::F32(values) => values.len(),
            Self::F64(values) => values.len(),
            Self::I32(values) => values.len(),
            Self::I64(values) => values.len(),
            Self::U32(values) => values.len(),
            Self::U64(values) => values.len(),
        }
    }

    fn set_register(&mut self, index: usize, value: f64) {
        match self {
            Self::Bool(values) => values[index] = u8::from(value != 0.0),
            Self::F32(values) => values[index] = value as f32,
            Self::F64(values) => values[index] = value,
            Self::I32(values) => values[index] = value.to_bits() as u32 as i32,
            Self::I64(values) => values[index] = value.to_bits() as i64,
            Self::U32(values) => values[index] = value.to_bits() as u32,
            Self::U64(values) => values[index] = value.to_bits(),
        }
    }

    fn copy_registers(&self, output: &mut [f64], start: usize) {
        match self {
            Self::Bool(values) => {
                for (output, &value) in output.iter_mut().zip(&values[start..]) {
                    *output = f64::from(value);
                }
            }
            Self::F32(values) => {
                for (output, &value) in output.iter_mut().zip(&values[start..]) {
                    *output = value as f64;
                }
            }
            Self::F64(values) => output.copy_from_slice(&values[start..start + output.len()]),
            Self::I32(values) => {
                for (output, &value) in output.iter_mut().zip(&values[start..]) {
                    *output = f64::from_bits(value as u32 as u64);
                }
            }
            Self::I64(values) => {
                for (output, &value) in output.iter_mut().zip(&values[start..]) {
                    *output = f64::from_bits(value as u64);
                }
            }
            Self::U32(values) => {
                for (output, &value) in output.iter_mut().zip(&values[start..]) {
                    *output = f64::from_bits(value as u64);
                }
            }
            Self::U64(values) => {
                for (output, &value) in output.iter_mut().zip(&values[start..]) {
                    *output = f64::from_bits(value);
                }
            }
        }
    }

    fn fill_f64(&mut self, range: std::ops::Range<usize>, value: f64) {
        match self {
            Self::Bool(values) => values[range].fill(u8::from(value != 0.0)),
            Self::F32(values) => values[range].fill(value as f32),
            Self::F64(values) => values[range].fill(value),
            Self::I32(values) => values[range].fill(value as i32),
            Self::I64(values) => values[range].fill(value as i64),
            Self::U32(values) => values[range].fill(value as u32),
            Self::U64(values) => values[range].fill(value as u64),
        }
    }

    fn add_f64(&mut self, index: usize, value: f64) {
        match self {
            Self::F32(values) => values[index] = (values[index] as f64 + value) as f32,
            Self::F64(values) => values[index] += value,
            _ => panic!("summed variables require floating-point state"),
        }
    }

    fn push_register(&mut self, value: f64) {
        match self {
            Self::Bool(values) => values.push(u8::from(value != 0.0)),
            Self::F32(values) => values.push(value as f32),
            Self::F64(values) => values.push(value),
            Self::I32(values) => values.push(value.to_bits() as u32 as i32),
            Self::I64(values) => values.push(value.to_bits() as i64),
            Self::U32(values) => values.push(value.to_bits() as u32),
            Self::U64(values) => values.push(value.to_bits()),
        }
    }

    fn try_reserve_exact(&mut self, additional: usize) -> Result<()> {
        match self {
            Self::Bool(values) => values.try_reserve_exact(additional)?,
            Self::F32(values) => values.try_reserve_exact(additional)?,
            Self::F64(values) => values.try_reserve_exact(additional)?,
            Self::I32(values) => values.try_reserve_exact(additional)?,
            Self::I64(values) => values.try_reserve_exact(additional)?,
            Self::U32(values) => values.try_reserve_exact(additional)?,
            Self::U64(values) => values.try_reserve_exact(additional)?,
        }
        Ok(())
    }

    fn byte_len(&self) -> usize {
        match self {
            Self::Bool(values) => values.len(),
            Self::F32(values) => values.len() * 4,
            Self::F64(values) => values.len() * 8,
            Self::I32(values) => values.len() * 4,
            Self::I64(values) => values.len() * 8,
            Self::U32(values) => values.len() * 4,
            Self::U64(values) => values.len() * 8,
        }
    }

    fn dump(&self, writer: &mut impl Write) -> Result<()> {
        match self {
            Self::Bool(values) => writer.write_all(values).map_err(Into::into),
            Self::F32(values) => dump_f32(writer, values),
            Self::F64(values) => dump_f64(writer, values),
            Self::I32(values) => dump_i32(writer, values),
            Self::I64(values) => dump_i64(writer, values),
            Self::U32(values) => dump_u32(writer, values),
            Self::U64(values) => dump_u64_values(writer, values),
        }
    }
}

struct PopulationRuntime {
    states: Vec<StateArray>,
    subexpression_update: Option<Program>,
    update: Option<Program>,
    adaptive_update: Option<AdaptiveRuntime>,
    spatial: Option<SpatialRuntime>,
    thresholds: Vec<Program>,
    spike_schedule: Vec<(usize, usize)>,
    spike_cursor: usize,
    poisson_inputs: Vec<Program>,
    regular: Vec<Program>,
    resets: Vec<Program>,
    refractory: Option<RefractoryRuntime>,
    tick: usize,
    end_tick: usize,
    fired: Vec<Vec<usize>>,
    event_history: Vec<Vec<(usize, usize)>>,
    counts: Vec<usize>,
    last_fired: Vec<usize>,
    samples: Vec<StateArray>,
    spikes: Vec<(usize, usize)>,
    monitor_values: Vec<MonitorValue>,
    event_monitors: Vec<EventMonitorRuntime>,
}

struct AdaptiveRuntime {
    program: Program,
    integrator: AdaptiveIntegrator,
    state_indices: Vec<usize>,
    derivative_registers: Vec<usize>,
    absolute_errors: Vec<f64>,
    adaptable_timestep: bool,
    max_steps: usize,
    last_timestep: Option<usize>,
    failed_steps: Option<usize>,
    step_count: Option<usize>,
    frozen: Vec<bool>,
}

#[derive(Clone, Copy)]
enum AdaptiveIntegrator {
    Rk2,
    Rk4,
    Rkf45,
    Rkck,
    Rk8pd,
}

impl AdaptiveRuntime {
    fn new(
        code: &CodeObjectSpec,
        inputs: BTreeMap<String, Input>,
        writable: &BTreeMap<String, WriteTarget>,
        state_indices: &BTreeMap<String, usize>,
        rng_seed: u64,
        functions: &[FunctionDefinition],
    ) -> Result<Option<Self>> {
        let Some(definition) = &code.adaptive else {
            return Ok(None);
        };
        let integrator = match definition.integrator.as_str() {
            "rk2" => AdaptiveIntegrator::Rk2,
            "rk4" => AdaptiveIntegrator::Rk4,
            "rkf45" => AdaptiveIntegrator::Rkf45,
            "rkck" => AdaptiveIntegrator::Rkck,
            "rk8pd" => AdaptiveIntegrator::Rk8pd,
            _ => return Err("unsupported adaptive integrator".into()),
        };
        let program = Program::compile(code, inputs, writable, rng_seed, functions)?;
        let derivative_registers = definition
            .derivatives
            .iter()
            .map(|name| {
                program
                    .symbols
                    .get(name)
                    .copied()
                    .ok_or_else(|| "adaptive derivative output is unavailable".into())
            })
            .collect::<Result<Vec<_>>>()?;
        let indices = definition
            .states
            .iter()
            .map(|name| {
                state_indices
                    .get(name)
                    .copied()
                    .ok_or_else(|| "adaptive state is unavailable".into())
            })
            .collect::<Result<Vec<_>>>()?;
        let frozen_names: BTreeSet<_> = definition.frozen_states.iter().collect();
        Ok(Some(Self {
            program,
            integrator,
            state_indices: indices,
            derivative_registers,
            absolute_errors: definition
                .absolute_errors
                .iter()
                .map(|value| decode_bits(value))
                .collect::<Result<_>>()?,
            adaptable_timestep: definition.adaptable_timestep,
            max_steps: definition.max_steps,
            last_timestep: definition
                .last_timestep
                .as_ref()
                .map(|name| state_indices[name]),
            failed_steps: definition
                .failed_steps
                .as_ref()
                .map(|name| state_indices[name]),
            step_count: definition
                .step_count
                .as_ref()
                .map(|name| state_indices[name]),
            frozen: definition
                .states
                .iter()
                .map(|name| frozen_names.contains(name))
                .collect(),
        }))
    }

    fn derivative(
        &mut self,
        states: &mut [StateArray],
        refractory: Option<&RefractoryRuntime>,
        neuron: usize,
        values: &[f64],
        time: f64,
        tick: usize,
    ) -> Result<Vec<f64>> {
        for (&state, &value) in self.state_indices.iter().zip(values) {
            states[state].set_register(neuron, value);
        }
        self.program.prepare(time, tick)?;
        self.program.run(
            states,
            states,
            states,
            &[],
            refractory,
            Batch::neurons(neuron, 1),
        )?;
        let unavailable = refractory.is_some_and(|runtime| !runtime.not_refractory[neuron]);
        Ok(self
            .derivative_registers
            .iter()
            .zip(&self.frozen)
            .map(|(&register, &frozen)| {
                if frozen && unavailable {
                    0.0
                } else {
                    self.program.registers[register][0]
                }
            })
            .collect())
    }

    fn stage(y: &[f64], h: f64, terms: &[(&[f64], f64)]) -> Vec<f64> {
        (0..y.len())
            .map(|index| {
                y[index]
                    + h * terms
                        .iter()
                        .map(|(values, coefficient)| coefficient * values[index])
                        .sum::<f64>()
            })
            .collect()
    }

    fn rkf45_step(
        &mut self,
        states: &mut [StateArray],
        refractory: Option<&RefractoryRuntime>,
        neuron: usize,
        y: &[f64],
        time: f64,
        h: f64,
        tick: usize,
    ) -> Result<(Vec<f64>, f64)> {
        let k1 = self.derivative(states, refractory, neuron, y, time, tick)?;
        let y2 = Self::stage(y, h, &[(&k1, 1.0 / 4.0)]);
        let k2 = self.derivative(states, refractory, neuron, &y2, time + h / 4.0, tick)?;
        let y3 = Self::stage(y, h, &[(&k1, 3.0 / 32.0), (&k2, 9.0 / 32.0)]);
        let k3 = self.derivative(states, refractory, neuron, &y3, time + 3.0 * h / 8.0, tick)?;
        let y4_stage = Self::stage(
            y,
            h,
            &[
                (&k1, 1932.0 / 2197.0),
                (&k2, -7200.0 / 2197.0),
                (&k3, 7296.0 / 2197.0),
            ],
        );
        let k4 = self.derivative(
            states,
            refractory,
            neuron,
            &y4_stage,
            time + 12.0 * h / 13.0,
            tick,
        )?;
        let y5_stage = Self::stage(
            y,
            h,
            &[
                (&k1, 439.0 / 216.0),
                (&k2, -8.0),
                (&k3, 3680.0 / 513.0),
                (&k4, -845.0 / 4104.0),
            ],
        );
        let k5 = self.derivative(states, refractory, neuron, &y5_stage, time + h, tick)?;
        let y6 = Self::stage(
            y,
            h,
            &[
                (&k1, -8.0 / 27.0),
                (&k2, 2.0),
                (&k3, -3544.0 / 2565.0),
                (&k4, 1859.0 / 4104.0),
                (&k5, -11.0 / 40.0),
            ],
        );
        let k6 = self.derivative(states, refractory, neuron, &y6, time + h / 2.0, tick)?;
        let fifth = Self::stage(
            y,
            h,
            &[
                (&k1, 16.0 / 135.0),
                (&k3, 6656.0 / 12825.0),
                (&k4, 28561.0 / 56430.0),
                (&k5, -9.0 / 50.0),
                (&k6, 2.0 / 55.0),
            ],
        );
        let error = (0..y.len())
            .map(|index| {
                let estimate = h
                    * (k1[index] / 360.0
                        - 128.0 * k3[index] / 4275.0
                        - 2197.0 * k4[index] / 75240.0
                        + k5[index] / 50.0
                        + 2.0 * k6[index] / 55.0);
                estimate.abs() / self.absolute_errors[index]
            })
            .fold(0.0f64, f64::max);
        Ok((fifth, error))
    }

    fn rk2_step(
        &mut self,
        states: &mut [StateArray],
        refractory: Option<&RefractoryRuntime>,
        neuron: usize,
        y: &[f64],
        time: f64,
        h: f64,
        tick: usize,
    ) -> Result<(Vec<f64>, f64)> {
        let k1 = self.derivative(states, refractory, neuron, y, time, tick)?;
        let midpoint: Vec<_> = y
            .iter()
            .zip(&k1)
            .map(|(&value, &derivative)| value + 0.5 * h * derivative)
            .collect();
        let k2 = self.derivative(states, refractory, neuron, &midpoint, time + 0.5 * h, tick)?;
        let endpoint: Vec<_> = y
            .iter()
            .zip(&k1)
            .zip(&k2)
            .map(|((&value, &first), &second)| value + h * (-first + 2.0 * second))
            .collect();
        let k3 = self.derivative(states, refractory, neuron, &endpoint, time + h, tick)?;
        let third_order: Vec<_> = k1
            .iter()
            .zip(&k2)
            .zip(&k3)
            .map(|((&first, &second), &third)| (first + 4.0 * second + third) / 6.0)
            .collect();
        let candidate: Vec<_> = y
            .iter()
            .zip(&third_order)
            .map(|(&value, &sum)| value + h * sum)
            .collect();
        let error = k2
            .iter()
            .zip(&third_order)
            .zip(&self.absolute_errors)
            .map(|((&second, &sum), &scale)| (h * (second - sum)).abs() / scale)
            .fold(0.0f64, f64::max);
        Ok((candidate, error))
    }

    fn rk4_advance(
        &mut self,
        states: &mut [StateArray],
        refractory: Option<&RefractoryRuntime>,
        neuron: usize,
        y: &[f64],
        time: f64,
        h: f64,
        tick: usize,
    ) -> Result<Vec<f64>> {
        let k1 = self.derivative(states, refractory, neuron, y, time, tick)?;
        let y2 = Self::stage(y, h, &[(&k1, 1.0 / 2.0)]);
        let k2 = self.derivative(states, refractory, neuron, &y2, time + h / 2.0, tick)?;
        let y3 = Self::stage(y, h, &[(&k2, 1.0 / 2.0)]);
        let k3 = self.derivative(states, refractory, neuron, &y3, time + h / 2.0, tick)?;
        let y4 = Self::stage(y, h, &[(&k3, 1.0)]);
        let k4 = self.derivative(states, refractory, neuron, &y4, time + h, tick)?;
        let mut candidate = y.to_vec();
        for index in 0..candidate.len() {
            candidate[index] += h / 6.0 * k1[index];
            candidate[index] += h / 3.0 * k2[index];
            candidate[index] += h / 3.0 * k3[index];
            candidate[index] += h / 6.0 * k4[index];
        }
        Ok(candidate)
    }

    fn rk4_step(
        &mut self,
        states: &mut [StateArray],
        refractory: Option<&RefractoryRuntime>,
        neuron: usize,
        y: &[f64],
        time: f64,
        h: f64,
        tick: usize,
    ) -> Result<(Vec<f64>, f64)> {
        let one_step = self.rk4_advance(states, refractory, neuron, y, time, h, tick)?;
        let first_half = self.rk4_advance(states, refractory, neuron, y, time, h / 2.0, tick)?;
        let candidate = self.rk4_advance(
            states,
            refractory,
            neuron,
            &first_half,
            time + h / 2.0,
            h / 2.0,
            tick,
        )?;
        let error = candidate
            .iter()
            .zip(&one_step)
            .zip(&self.absolute_errors)
            .map(|((&two_half, &one), &scale)| (4.0 * (two_half - one) / 15.0).abs() / scale)
            .fold(0.0f64, f64::max);
        Ok((candidate, error))
    }

    fn rkck_step(
        &mut self,
        states: &mut [StateArray],
        refractory: Option<&RefractoryRuntime>,
        neuron: usize,
        y: &[f64],
        time: f64,
        h: f64,
        tick: usize,
    ) -> Result<(Vec<f64>, f64)> {
        let k1 = self.derivative(states, refractory, neuron, y, time, tick)?;
        let y2 = Self::stage(y, h, &[(&k1, 1.0 / 5.0)]);
        let k2 = self.derivative(states, refractory, neuron, &y2, time + h / 5.0, tick)?;
        let y3 = Self::stage(y, h, &[(&k1, 3.0 / 40.0), (&k2, 9.0 / 40.0)]);
        let k3 = self.derivative(states, refractory, neuron, &y3, time + 3.0 * h / 10.0, tick)?;
        let y4 = Self::stage(
            y,
            h,
            &[(&k1, 3.0 / 10.0), (&k2, -9.0 / 10.0), (&k3, 6.0 / 5.0)],
        );
        let k4 = self.derivative(states, refractory, neuron, &y4, time + 3.0 * h / 5.0, tick)?;
        let y5 = Self::stage(
            y,
            h,
            &[
                (&k1, -11.0 / 54.0),
                (&k2, 5.0 / 2.0),
                (&k3, -70.0 / 27.0),
                (&k4, 35.0 / 27.0),
            ],
        );
        let k5 = self.derivative(states, refractory, neuron, &y5, time + h, tick)?;
        let y6 = Self::stage(
            y,
            h,
            &[
                (&k1, 1631.0 / 55296.0),
                (&k2, 175.0 / 512.0),
                (&k3, 575.0 / 13824.0),
                (&k4, 44275.0 / 110592.0),
                (&k5, 253.0 / 4096.0),
            ],
        );
        let k6 = self.derivative(states, refractory, neuron, &y6, time + 7.0 * h / 8.0, tick)?;
        let candidate = Self::stage(
            y,
            h,
            &[
                (&k1, 37.0 / 378.0),
                (&k3, 250.0 / 621.0),
                (&k4, 125.0 / 594.0),
                (&k6, 512.0 / 1771.0),
            ],
        );
        let error = (0..y.len())
            .map(|index| {
                let estimate = h
                    * ((37.0 / 378.0 - 2825.0 / 27648.0) * k1[index]
                        + (250.0 / 621.0 - 18575.0 / 48384.0) * k3[index]
                        + (125.0 / 594.0 - 13525.0 / 55296.0) * k4[index]
                        - 277.0 / 14336.0 * k5[index]
                        + (512.0 / 1771.0 - 1.0 / 4.0) * k6[index]);
                estimate.abs() / self.absolute_errors[index]
            })
            .fold(0.0f64, f64::max);
        Ok((candidate, error))
    }

    fn rk8pd_step(
        &mut self,
        states: &mut [StateArray],
        refractory: Option<&RefractoryRuntime>,
        neuron: usize,
        y: &[f64],
        time: f64,
        h: f64,
        tick: usize,
    ) -> Result<(Vec<f64>, f64)> {
        let k1 = self.derivative(states, refractory, neuron, y, time, tick)?;
        let y2 = Self::stage(y, h, &[(&k1, 1.0 / 18.0)]);
        let k2 = self.derivative(states, refractory, neuron, &y2, time + h / 18.0, tick)?;
        let y3 = Self::stage(y, h, &[(&k1, 1.0 / 48.0), (&k2, 1.0 / 16.0)]);
        let k3 = self.derivative(states, refractory, neuron, &y3, time + h / 12.0, tick)?;
        let y4 = Self::stage(y, h, &[(&k1, 1.0 / 32.0), (&k3, 3.0 / 32.0)]);
        let k4 = self.derivative(states, refractory, neuron, &y4, time + h / 8.0, tick)?;
        let y5 = Self::stage(
            y,
            h,
            &[(&k1, 5.0 / 16.0), (&k3, -75.0 / 64.0), (&k4, 75.0 / 64.0)],
        );
        let k5 = self.derivative(states, refractory, neuron, &y5, time + 5.0 * h / 16.0, tick)?;
        let y6 = Self::stage(
            y,
            h,
            &[(&k1, 3.0 / 80.0), (&k4, 3.0 / 16.0), (&k5, 3.0 / 20.0)],
        );
        let k6 = self.derivative(states, refractory, neuron, &y6, time + 3.0 * h / 8.0, tick)?;
        let y7 = Self::stage(
            y,
            h,
            &[
                (&k1, 29443841.0 / 614563906.0),
                (&k4, 77736538.0 / 692538347.0),
                (&k5, -28693883.0 / 1125000000.0),
                (&k6, 23124283.0 / 1800000000.0),
            ],
        );
        let k7 = self.derivative(
            states,
            refractory,
            neuron,
            &y7,
            time + 59.0 * h / 400.0,
            tick,
        )?;
        let y8 = Self::stage(
            y,
            h,
            &[
                (&k1, 16016141.0 / 946692911.0),
                (&k4, 61564180.0 / 158732637.0),
                (&k5, 22789713.0 / 633445777.0),
                (&k6, 545815736.0 / 2771057229.0),
                (&k7, -180193667.0 / 1043307555.0),
            ],
        );
        let k8 = self.derivative(
            states,
            refractory,
            neuron,
            &y8,
            time + 93.0 * h / 200.0,
            tick,
        )?;
        let y9 = Self::stage(
            y,
            h,
            &[
                (&k1, 39632708.0 / 573591083.0),
                (&k4, -433636366.0 / 683701615.0),
                (&k5, -421739975.0 / 2616292301.0),
                (&k6, 100302831.0 / 723423059.0),
                (&k7, 790204164.0 / 839813087.0),
                (&k8, 800635310.0 / 3783071287.0),
            ],
        );
        let k9 = self.derivative(
            states,
            refractory,
            neuron,
            &y9,
            time + 5490023248.0 * h / 9719169821.0,
            tick,
        )?;
        let y10 = Self::stage(
            y,
            h,
            &[
                (&k1, 246121993.0 / 1340847787.0),
                (&k4, -37695042795.0 / 15268766246.0),
                (&k5, -309121744.0 / 1061227803.0),
                (&k6, -12992083.0 / 490766935.0),
                (&k7, 6005943493.0 / 2108947869.0),
                (&k8, 393006217.0 / 1396673457.0),
                (&k9, 123872331.0 / 1001029789.0),
            ],
        );
        let k10 = self.derivative(
            states,
            refractory,
            neuron,
            &y10,
            time + 13.0 * h / 20.0,
            tick,
        )?;
        let y11 = Self::stage(
            y,
            h,
            &[
                (&k1, -1028468189.0 / 846180014.0),
                (&k4, 8478235783.0 / 508512852.0),
                (&k5, 1311729495.0 / 1432422823.0),
                (&k6, -10304129995.0 / 1701304382.0),
                (&k7, -48777925059.0 / 3047939560.0),
                (&k8, 15336726248.0 / 1032824649.0),
                (&k9, -45442868181.0 / 3398467696.0),
                (&k10, 3065993473.0 / 597172653.0),
            ],
        );
        let k11 = self.derivative(
            states,
            refractory,
            neuron,
            &y11,
            time + 1201146811.0 * h / 1299019798.0,
            tick,
        )?;
        let y12 = Self::stage(
            y,
            h,
            &[
                (&k1, 185892177.0 / 718116043.0),
                (&k4, -3185094517.0 / 667107341.0),
                (&k5, -477755414.0 / 1098053517.0),
                (&k6, -703635378.0 / 230739211.0),
                (&k7, 5731566787.0 / 1027545527.0),
                (&k8, 5232866602.0 / 850066563.0),
                (&k9, -4093664535.0 / 808688257.0),
                (&k10, 3962137247.0 / 1805957418.0),
                (&k11, 65686358.0 / 487910083.0),
            ],
        );
        let k12 = self.derivative(states, refractory, neuron, &y12, time + h, tick)?;
        let y13 = Self::stage(
            y,
            h,
            &[
                (&k1, 403863854.0 / 491063109.0),
                (&k4, -5068492393.0 / 434740067.0),
                (&k5, -411421997.0 / 543043805.0),
                (&k6, 652783627.0 / 914296604.0),
                (&k7, 11173962825.0 / 925320556.0),
                (&k8, -13158990841.0 / 6184727034.0),
                (&k9, 3936647629.0 / 1978049680.0),
                (&k10, -160528059.0 / 685178525.0),
                (&k11, 248638103.0 / 1413531060.0),
            ],
        );
        let k13 = self.derivative(states, refractory, neuron, &y13, time + h, tick)?;
        let high_terms = [
            (&k1[..], 14005451.0 / 335480064.0),
            (&k6[..], -59238493.0 / 1068277825.0),
            (&k7[..], 181606767.0 / 758867731.0),
            (&k8[..], 561292985.0 / 797845732.0),
            (&k9[..], -1041891430.0 / 1371343529.0),
            (&k10[..], 760417239.0 / 1151165299.0),
            (&k11[..], 118820643.0 / 751138087.0),
            (&k12[..], -528747749.0 / 2220607170.0),
            (&k13[..], 1.0 / 4.0),
        ];
        let candidate = Self::stage(y, h, &high_terms);
        let low_terms = [
            (&k1[..], 13451932.0 / 455176623.0),
            (&k6[..], -808719846.0 / 976000145.0),
            (&k7[..], 1757004468.0 / 5645159321.0),
            (&k8[..], 656045339.0 / 265891186.0),
            (&k9[..], -3867574721.0 / 1518517206.0),
            (&k10[..], 465885868.0 / 322736535.0),
            (&k11[..], 53011238.0 / 667516719.0),
            (&k12[..], 2.0 / 45.0),
        ];
        let error = (0..y.len())
            .map(|index| {
                let high = high_terms
                    .iter()
                    .map(|(values, coefficient)| coefficient * values[index])
                    .sum::<f64>();
                let low = low_terms
                    .iter()
                    .map(|(values, coefficient)| coefficient * values[index])
                    .sum::<f64>();
                (h * (low - high)).abs() / self.absolute_errors[index]
            })
            .fold(0.0f64, f64::max);
        Ok((candidate, error))
    }

    fn step(
        &mut self,
        states: &mut [StateArray],
        refractory: Option<&RefractoryRuntime>,
        neuron: usize,
        y: &[f64],
        time: f64,
        h: f64,
        tick: usize,
    ) -> Result<(Vec<f64>, f64, f64)> {
        match self.integrator {
            AdaptiveIntegrator::Rk2 => {
                let (candidate, error) =
                    self.rk2_step(states, refractory, neuron, y, time, h, tick)?;
                Ok((candidate, error, 2.0))
            }
            AdaptiveIntegrator::Rk4 => {
                let (candidate, error) =
                    self.rk4_step(states, refractory, neuron, y, time, h, tick)?;
                Ok((candidate, error, 4.0))
            }
            AdaptiveIntegrator::Rkf45 => {
                let (candidate, error) =
                    self.rkf45_step(states, refractory, neuron, y, time, h, tick)?;
                Ok((candidate, error, 5.0))
            }
            AdaptiveIntegrator::Rkck => {
                let (candidate, error) =
                    self.rkck_step(states, refractory, neuron, y, time, h, tick)?;
                Ok((candidate, error, 5.0))
            }
            AdaptiveIntegrator::Rk8pd => {
                let (candidate, error) =
                    self.rk8pd_step(states, refractory, neuron, y, time, h, tick)?;
                Ok((candidate, error, 8.0))
            }
        }
    }

    fn run(
        &mut self,
        states: &mut [StateArray],
        refractory: Option<&RefractoryRuntime>,
        count: usize,
        time: f64,
        dt: f64,
        tick: usize,
    ) -> Result<()> {
        for neuron in 0..count {
            let initial: Vec<_> = self
                .state_indices
                .iter()
                .map(|&state| states[state].get_register(neuron))
                .collect();
            let mut y = initial.clone();
            let end = time + dt;
            let mut current = time;
            let mut driver_h = self
                .last_timestep
                .map(|state| states[state].get_register(neuron))
                .unwrap_or(dt)
                .min(dt);
            if !(driver_h.is_finite() && driver_h > 0.0) {
                driver_h = dt;
            }
            let mut accepted = 0usize;
            let mut failed = 0usize;
            let outcome = (|| -> Result<()> {
                while current < end {
                    check(
                        accepted <= self.max_steps,
                        "adaptive GSL integrator exceeded max_steps",
                    )?;
                    let remaining = end - current;
                    let final_step = driver_h > remaining;
                    let h = driver_h.min(remaining);
                    check(
                        h > 0.0 && current + h > current,
                        "adaptive GSL timestep underflow",
                    )?;
                    let (candidate, error, order) =
                        self.step(states, refractory, neuron, &y, current, h, tick)?;
                    check(
                        error.is_finite() && candidate.iter().all(|v| v.is_finite()),
                        "adaptive GSL integration produced a non-finite value",
                    )?;
                    let decrease = error > 1.1;
                    let adjusted_h = if decrease {
                        h * (0.9 / error.powf(1.0 / order)).max(0.2)
                    } else if error < 0.5 {
                        let factor = if error <= f64::MIN_POSITIVE {
                            5.0
                        } else {
                            (0.9 / error.powf(1.0 / (order + 1.0))).clamp(1.0, 5.0)
                        };
                        h * factor
                    } else {
                        h
                    };
                    if !self.adaptable_timestep && decrease {
                        return Err("fixed-step GSL integration exceeded absolute_error".into());
                    }
                    if !self.adaptable_timestep || !decrease {
                        y = candidate;
                        current = if final_step { end } else { current + h };
                        accepted += 1;
                        if !self.adaptable_timestep {
                            break;
                        }
                        if !final_step {
                            driver_h = adjusted_h.min(dt);
                        }
                    } else {
                        failed += 1;
                        driver_h = adjusted_h;
                    }
                }
                Ok(())
            })();
            if let Err(error) = outcome {
                for (&state, &value) in self.state_indices.iter().zip(&initial) {
                    states[state].set_register(neuron, value);
                }
                return Err(error);
            }
            for (&state, &value) in self.state_indices.iter().zip(&y) {
                states[state].set_register(neuron, value);
            }
            if let Some(state) = self.last_timestep {
                states[state].set_register(neuron, driver_h);
            }
            if let Some(state) = self.failed_steps {
                states[state].set_register(neuron, f64::from_bits(failed as u32 as u64));
            }
            if let Some(state) = self.step_count {
                states[state].set_register(neuron, f64::from_bits(accepted as u32 as u64));
            }
        }
        Ok(())
    }
}

struct SpatialRuntime {
    program: Program,
    gtot_register: usize,
    i0_register: usize,
    voltage_state: usize,
    membrane_current_state: usize,
    capacitance: Vec<f64>,
    starts: Vec<usize>,
    ends: Vec<usize>,
    parents: Vec<usize>,
    child_slots: Vec<usize>,
    children_count: Vec<usize>,
    children: Vec<usize>,
    child_width: usize,
    ab0: Vec<f64>,
    ab1: Vec<f64>,
    ab2: Vec<f64>,
    b_plus: Vec<f64>,
    b_minus: Vec<f64>,
    invr0: Vec<f64>,
    invrn: Vec<f64>,
    gtot: Vec<f64>,
    i0: Vec<f64>,
    previous: Vec<f64>,
    c: Vec<f64>,
    v_star: Vec<f64>,
    u_plus: Vec<f64>,
    u_minus: Vec<f64>,
    p_diag: Vec<f64>,
    p_parent: Vec<f64>,
    p_children: Vec<f64>,
    rhs: Vec<f64>,
}

enum MonitorValue {
    State(usize),
    Scalar(f64),
    Parameter(Vec<f64>),
    Linked(Input),
}

struct EventMonitorRuntime {
    event_index: usize,
    values: Vec<MonitorValue>,
    events: Vec<(usize, usize)>,
    samples: Vec<StateArray>,
}

struct SynapseRuntime {
    source: Vec<usize>,
    target: Vec<usize>,
    states: Vec<StateArray>,
    subexpression_update: Option<Program>,
    summed: Vec<SummedRuntime>,
    update: Option<Program>,
    regular: Vec<Program>,
    pathways: Vec<PathwayRuntime>,
    monitors: Vec<SynapseMonitorRuntime>,
}

struct SynapseMonitorRuntime {
    values: Vec<SynapseMonitorValue>,
    samples: Vec<StateArray>,
}

enum SynapseMonitorValue {
    SynapseState(usize),
    PreState(usize),
    PostState(usize),
    Linked(Input),
}

struct PathwayRuntime {
    program: Program,
    topology: PathwayQueue,
    delay_ticks: Vec<usize>,
    is_pre: bool,
    event_index: usize,
    active_sources: Vec<usize>,
}

struct SummedRuntime {
    program: Program,
    result: usize,
    before_state: bool,
    target_population: usize,
    target_start: usize,
    target_count: usize,
    target_state: usize,
    target_is_pre: bool,
}

fn linked_input(
    population: usize,
    def: &PopulationDefinition,
    instance: &PopulationInstance,
    populations: &[PopulationDefinition],
    linked: &LinkedVariableDefinition,
) -> Result<Input> {
    let source_state = populations[linked.source_population]
        .states
        .iter()
        .position(|symbol| symbol.name == linked.source_state)
        .ok_or("linked source state is unavailable")?;
    let index = match &linked.index {
        LinkedIndexDefinition::Identity => LinkedInputIndex::Identity,
        LinkedIndexDefinition::Constant { values } => {
            LinkedInputIndex::Constant(Arc::from(values.clone()))
        }
        LinkedIndexDefinition::State { name } => LinkedInputIndex::State {
            population,
            state: def
                .states
                .iter()
                .position(|symbol| &symbol.name == name)
                .ok_or("linked index state is unavailable")?,
            pointer: 0,
        },
        LinkedIndexDefinition::Parameter { name } => {
            let position = def
                .parameters
                .iter()
                .position(|symbol| &symbol.name == name)
                .ok_or("linked index parameter is unavailable")?;
            let values =
                StateArray::from_encoded(&def.parameters[position], &instance.parameters[name])?;
            let indices = (0..values.len())
                .map(|index| values.get_index(index))
                .collect::<Result<Vec<_>>>()?;
            LinkedInputIndex::Constant(Arc::from(indices))
        }
    };
    Ok(Input::Linked {
        source_population: linked.source_population,
        source_state,
        source_pointer: 0,
        index,
        dtype: linked.dtype,
    })
}

fn synapse_linked_input(
    populations: &[PopulationDefinition],
    linked: &LinkedVariableDefinition,
) -> Result<Input> {
    let source_state = populations[linked.source_population]
        .states
        .iter()
        .position(|symbol| symbol.name == linked.source_state)
        .ok_or("linked source state is unavailable")?;
    let LinkedIndexDefinition::Constant { values } = &linked.index else {
        return Err("synapse linked variables require a constant index mapping".into());
    };
    Ok(Input::Linked {
        source_population: linked.source_population,
        source_state,
        source_pointer: 0,
        index: LinkedInputIndex::Constant(Arc::from(values.clone())),
        dtype: linked.dtype,
    })
}

struct PopulationBuildContext<'a> {
    populations: &'a [PopulationDefinition],
    run_clocks: &'a [RunClock],
    rng_seed: u64,
    functions: &'a [FunctionDefinition],
}

fn population_parameter_f64(
    def: &PopulationDefinition,
    instance: &PopulationInstance,
    name: &str,
) -> Result<Vec<f64>> {
    let symbol = def
        .parameters
        .iter()
        .find(|symbol| symbol.name == name)
        .ok_or("spatial parameter is unavailable")?;
    check(symbol.dtype == DType::F64, "spatial parameter must be f64")?;
    let values = instance.parameters[name]
        .iter()
        .map(|value| decode_bits(value))
        .collect::<Result<Vec<_>>>()?;
    if symbol.index_domain == IndexDomain::Scalar {
        Ok(vec![values[0]; def.count])
    } else {
        Ok(values)
    }
}

impl SpatialRuntime {
    fn new(
        def: &PopulationDefinition,
        instance: &PopulationInstance,
        inputs: BTreeMap<String, Input>,
        state_indices: &BTreeMap<String, usize>,
        rng_seed: u64,
        functions: &[FunctionDefinition],
    ) -> Result<Option<Self>> {
        let Some(spatial) = &def.spatial else {
            return Ok(None);
        };
        let code = def
            .code_objects
            .iter()
            .find(|code| code.kind == "spatial_state_update")
            .ok_or("missing spatial state-update code")?;
        let mut program = Program::compile(code, inputs, &BTreeMap::new(), rng_seed, functions)?;
        let gtot_register = *program
            .symbols
            .get("_gtot")
            .ok_or("spatial update must assign _gtot")?;
        let i0_register = *program
            .symbols
            .get("_I0")
            .ok_or("spatial update must assign _I0")?;
        let capacitance = population_parameter_f64(def, instance, &spatial.capacitance)?;
        let resistivity = population_parameter_f64(def, instance, &spatial.resistivity)?[0];
        let area = population_parameter_f64(def, instance, &spatial.area)?;
        let r_length_1 = population_parameter_f64(def, instance, &spatial.r_length_1)?;
        let r_length_2 = population_parameter_f64(def, instance, &spatial.r_length_2)?;
        let dt = decode_bits(&def.dt)?;
        let count = def.count;
        let sections = spatial.starts.len();
        let child_width = spatial.children.len() / (sections + 1);

        let mut invr = vec![0.0; count];
        for i in 1..count {
            invr[i] = 1.0 / (resistivity * (1.0 / r_length_2[i - 1] + 1.0 / r_length_1[i]));
        }
        for &start in &spatial.starts {
            invr[start] = 0.0;
        }
        let mut ab0 = vec![0.0; count];
        let mut ab1 = (0..count)
            .map(|i| -(capacitance[i] / dt) - invr[i] / area[i])
            .collect::<Vec<_>>();
        let mut ab2 = vec![0.0; count];
        for i in 1..count {
            ab0[i] = invr[i] / area[i - 1];
            ab2[i - 1] = invr[i] / area[i];
            ab1[i - 1] -= invr[i] / area[i - 1];
        }
        let mut b_plus = vec![0.0; count];
        let mut b_minus = vec![0.0; count];
        let mut invr0 = vec![0.0; sections];
        let mut invrn = vec![0.0; sections];
        for section in 0..sections {
            let first = spatial.starts[section];
            let last = spatial.ends[section] - 1;
            invr0[section] = r_length_1[first] / resistivity;
            invrn[section] = r_length_2[last] / resistivity;
            ab1[first] -= invr0[section] / area[first];
            ab1[last] -= invrn[section] / area[last];
            b_plus[last] = -invrn[section] / area[last];
            b_minus[first] = -invr0[section] / area[first];
        }
        check(
            resistivity.is_finite()
                && resistivity > 0.0
                && capacitance
                    .iter()
                    .all(|value| value.is_finite() && *value > 0.0)
                && area.iter().all(|value| value.is_finite() && *value > 0.0)
                && r_length_1
                    .iter()
                    .all(|value| value.is_finite() && *value > 0.0)
                && r_length_2
                    .iter()
                    .all(|value| value.is_finite() && *value > 0.0),
            "spatial geometry and cable parameters must be finite and positive",
        )?;
        // Drop any register contents populated while compiling; prepare/run
        // will fill them for the first scheduled timestep.
        program.tick = 0;
        Ok(Some(Self {
            program,
            gtot_register,
            i0_register,
            voltage_state: state_indices[&spatial.voltage],
            membrane_current_state: state_indices[&spatial.membrane_current],
            capacitance,
            starts: spatial.starts.clone(),
            ends: spatial.ends.clone(),
            parents: spatial.parents.clone(),
            child_slots: spatial.child_slots.clone(),
            children_count: spatial.children_count.clone(),
            children: spatial.children.clone(),
            child_width,
            ab0,
            ab1,
            ab2,
            b_plus,
            b_minus,
            invr0,
            invrn,
            gtot: vec![0.0; count],
            i0: vec![0.0; count],
            previous: vec![0.0; count],
            c: vec![0.0; count],
            v_star: vec![0.0; count],
            u_plus: vec![0.0; count],
            u_minus: vec![0.0; count],
            p_diag: vec![0.0; sections + 1],
            p_parent: vec![0.0; sections],
            p_children: vec![0.0; spatial.children.len()],
            rhs: vec![0.0; sections + 1],
        }))
    }

    fn run(
        &mut self,
        states: &mut [StateArray],
        refractory: Option<&RefractoryRuntime>,
        count: usize,
        dt: f64,
        time: f64,
        tick: usize,
    ) -> Result<()> {
        self.program.prepare(time, tick)?;
        for start in (0..count).step_by(LANES) {
            let len = LANES.min(count - start);
            self.program.run(
                states,
                states,
                states,
                &[],
                refractory,
                Batch::neurons(start, len),
            )?;
            self.gtot[start..start + len]
                .copy_from_slice(&self.program.registers[self.gtot_register][..len]);
            self.i0[start..start + len]
                .copy_from_slice(&self.program.registers[self.i0_register][..len]);
        }
        for (i, value) in self.previous.iter_mut().enumerate() {
            *value = states[self.voltage_state].get_register(i);
        }

        for section in 0..self.starts.len() {
            let start = self.starts[section];
            let end = self.ends[section];
            for j in start..end {
                self.v_star[j] = -(self.capacitance[j] / dt * self.previous[j]) - self.i0[j];
                self.u_plus[j] = self.b_plus[j];
                self.u_minus[j] = self.b_minus[j];
                let diagonal = self.ab1[j] - self.gtot[j];
                self.c[j] = if j < count - 1 { self.ab0[j + 1] } else { 0.0 };
                if j > 0 {
                    let lower = self.ab2[j - 1];
                    let scale = 1.0 / (diagonal - lower * self.c[j - 1]);
                    self.c[j] *= scale;
                    self.v_star[j] = (self.v_star[j] - lower * self.v_star[j - 1]) * scale;
                    self.u_plus[j] = (self.u_plus[j] - lower * self.u_plus[j - 1]) * scale;
                    self.u_minus[j] = (self.u_minus[j] - lower * self.u_minus[j - 1]) * scale;
                } else {
                    self.c[j] /= diagonal;
                    self.v_star[j] /= diagonal;
                    self.u_plus[j] /= diagonal;
                    self.u_minus[j] /= diagonal;
                }
            }
            for j in (start..end - 1).rev() {
                self.v_star[j] -= self.c[j] * self.v_star[j + 1];
                self.u_plus[j] -= self.c[j] * self.u_plus[j + 1];
                self.u_minus[j] -= self.c[j] * self.u_minus[j + 1];
            }
        }

        self.p_diag.fill(0.0);
        self.p_parent.fill(0.0);
        self.p_children.fill(0.0);
        self.rhs.fill(0.0);
        for section in 0..self.starts.len() {
            let parent = self.parents[section];
            let slot = self.child_slots[section];
            let first = self.starts[section];
            let last = self.ends[section] - 1;
            if section == 0 {
                self.p_diag[0] = self.u_minus[first] - 1.0;
                self.p_children[0] = self.u_plus[first];
                self.rhs[0] = -self.v_star[first];
            } else {
                self.p_diag[parent] += (1.0 - self.u_minus[first]) * self.invr0[section];
                self.p_children[parent * self.child_width + slot] =
                    -self.u_plus[first] * self.invr0[section];
                self.rhs[parent] += self.v_star[first] * self.invr0[section];
            }
            self.p_diag[section + 1] = (1.0 - self.u_plus[last]) * self.invrn[section];
            self.p_parent[section] = -self.u_minus[last] * self.invrn[section];
            self.rhs[section + 1] = self.v_star[last] * self.invrn[section];
        }
        for i in (0..self.p_diag.len()).rev() {
            for slot in 0..self.children_count[i] {
                let child = self.children[i * self.child_width + slot];
                let factor = self.p_children[i * self.child_width + slot] / self.p_diag[child];
                self.p_diag[i] -= factor * self.p_parent[child - 1];
                self.rhs[i] -= factor * self.rhs[child];
            }
        }
        self.rhs[0] /= self.p_diag[0];
        for i in 1..self.rhs.len() {
            let parent = self.parents[i - 1];
            self.rhs[i] = (self.rhs[i] - self.p_parent[i - 1] * self.rhs[parent]) / self.p_diag[i];
        }
        for section in 0..self.starts.len() {
            let parent = self.parents[section];
            for j in self.starts[section]..self.ends[section] {
                let voltage = self.v_star[j]
                    + self.rhs[parent] * self.u_minus[j]
                    + self.rhs[section + 1] * self.u_plus[j];
                states[self.voltage_state].set_register(j, voltage);
                states[self.membrane_current_state]
                    .set_register(j, self.capacitance[j] * (voltage - self.previous[j]) / dt);
            }
        }
        Ok(())
    }
}

fn population_runtime(
    population: usize,
    def: &PopulationDefinition,
    instance: &PopulationInstance,
    context: &PopulationBuildContext<'_>,
) -> Result<PopulationRuntime> {
    let state_indices: BTreeMap<_, _> = def
        .states
        .iter()
        .enumerate()
        .map(|(index, symbol)| (symbol.name.clone(), index))
        .collect();
    let states = def
        .states
        .iter()
        .map(|symbol| StateArray::from_encoded(symbol, &instance.initial_state[&symbol.name]))
        .collect::<Result<Vec<_>>>()?;
    let refractory = instance
        .refractory
        .as_ref()
        .map(RefractoryInstance::runtime)
        .transpose()?;
    let dt = decode_bits(&def.dt)?;
    let tick = context.run_clocks[def.clock].start_tick;
    let mut inputs = BTreeMap::from([
        ("dt".to_owned(), Input::Constant(dt, DType::F64)),
        ("t".to_owned(), Input::Time),
        (
            "N".to_owned(),
            Input::Constant(def.count as f64, DType::Index),
        ),
        ("i".to_owned(), Input::Index(Domain::Neuron)),
    ]);
    for (name, &index) in &state_indices {
        inputs.insert(
            name.clone(),
            Input::State(index, Domain::Neuron, def.states[index].dtype),
        );
    }
    for linked in &def.linked_variables {
        inputs.insert(
            linked.name.clone(),
            linked_input(population, def, instance, context.populations, linked)?,
        );
    }
    if refractory.is_some() {
        inputs.insert("lastspike".to_owned(), Input::LastSpike);
        inputs.insert(
            "not_refractory".to_owned(),
            Input::Available(Domain::Neuron),
        );
    }
    add_parameters(
        &mut inputs,
        &def.parameters,
        &instance.parameters,
        Domain::Neuron,
        &BTreeMap::new(),
    )?;
    let mut writable: BTreeMap<_, _> = state_indices
        .iter()
        .map(|(name, &index)| (name.clone(), WriteTarget::Neuron(index)))
        .collect();
    if refractory.is_some() {
        writable.insert("not_refractory".to_owned(), WriteTarget::Available);
    }
    let update_code = def.code_objects.iter().find(|c| c.kind == "state_update");
    let adaptive_update = update_code
        .map(|code| {
            AdaptiveRuntime::new(
                code,
                inputs.clone(),
                &writable,
                &state_indices,
                context.rng_seed,
                context.functions,
            )
        })
        .transpose()?
        .flatten();
    let update = update_code
        .filter(|code| code.adaptive.is_none())
        .map(|code| {
            Program::compile(
                code,
                inputs.clone(),
                &writable,
                context.rng_seed,
                context.functions,
            )
        })
        .transpose()?;
    let spatial = SpatialRuntime::new(
        def,
        instance,
        inputs.clone(),
        &state_indices,
        context.rng_seed,
        context.functions,
    )?;
    let subexpression_update = def
        .code_objects
        .iter()
        .find(|c| c.kind == "subexpression_update")
        .map(|code| {
            Program::compile(
                code,
                inputs.clone(),
                &writable,
                context.rng_seed,
                context.functions,
            )
        })
        .transpose()?;
    let thresholds = def
        .code_objects
        .iter()
        .filter(|c| c.kind == "threshold")
        .map(|code| {
            Program::compile(
                code,
                inputs.clone(),
                &writable,
                context.rng_seed,
                context.functions,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let spike_schedule = instance
        .spike_generator
        .as_ref()
        .map(|generator| {
            generator
                .spike_ticks
                .iter()
                .copied()
                .zip(generator.spike_indices.iter().copied())
                .collect()
        })
        .unwrap_or_default();
    let poisson_inputs = def
        .code_objects
        .iter()
        .filter(|code| code.kind == "poisson_input")
        .map(|code| {
            Program::compile(
                code,
                inputs.clone(),
                &writable,
                context.rng_seed,
                context.functions,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let regular = def
        .code_objects
        .iter()
        .filter(|code| code.kind == "run_regularly")
        .map(|code| {
            Program::compile(
                code,
                inputs.clone(),
                &writable,
                context.rng_seed,
                context.functions,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let resets = def
        .code_objects
        .iter()
        .filter(|c| c.kind == "reset")
        .map(|code| {
            Program::compile(
                code,
                inputs.clone(),
                &writable,
                context.rng_seed,
                context.functions,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let parameter_indices: BTreeMap<_, _> = def
        .parameters
        .iter()
        .enumerate()
        .map(|(index, symbol)| (symbol.name.clone(), index))
        .collect();
    let monitor_values = def
        .monitor
        .variables
        .iter()
        .map(|name| {
            if let Some(&index) = state_indices.get(name) {
                return Ok(MonitorValue::State(index));
            }
            if let Some(linked) = def
                .linked_variables
                .iter()
                .find(|linked| &linked.name == name)
            {
                return Ok(MonitorValue::Linked(linked_input(
                    population,
                    def,
                    instance,
                    context.populations,
                    linked,
                )?));
            }
            let index = parameter_indices[name];
            let symbol = &def.parameters[index];
            let values = instance.parameters[name]
                .iter()
                .map(|value| decode_typed_float(value, symbol.dtype))
                .collect::<Result<Vec<_>>>()?;
            Ok(if symbol.index_domain == IndexDomain::Scalar {
                MonitorValue::Scalar(values[0])
            } else {
                MonitorValue::Parameter(values)
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let samples = def
        .monitor
        .variables
        .iter()
        .map(|name| {
            def.states
                .iter()
                .chain(&def.parameters)
                .find(|symbol| &symbol.name == name)
                .map(|symbol| symbol.dtype)
                .or_else(|| {
                    def.linked_variables
                        .iter()
                        .find(|linked| &linked.name == name)
                        .map(|linked| linked.dtype)
                })
                .ok_or_else(|| "monitor symbol missing from population".into())
                .and_then(StateArray::empty)
        })
        .collect::<Result<Vec<_>>>()?;
    let event_monitors = def
        .event_monitors
        .iter()
        .map(|monitor| -> Result<EventMonitorRuntime> {
            let values = monitor
                .variables
                .iter()
                .map(|name| {
                    if let Some(&index) = state_indices.get(name) {
                        return Ok(MonitorValue::State(index));
                    }
                    if let Some(linked) = def
                        .linked_variables
                        .iter()
                        .find(|linked| &linked.name == name)
                    {
                        return Ok(MonitorValue::Linked(linked_input(
                            population,
                            def,
                            instance,
                            context.populations,
                            linked,
                        )?));
                    }
                    let index = parameter_indices[name];
                    let symbol = &def.parameters[index];
                    let values = instance.parameters[name]
                        .iter()
                        .map(|value| decode_typed_float(value, symbol.dtype))
                        .collect::<Result<Vec<_>>>()?;
                    Ok(if symbol.index_domain == IndexDomain::Scalar {
                        MonitorValue::Scalar(values[0])
                    } else {
                        MonitorValue::Parameter(values)
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            let samples = monitor
                .variables
                .iter()
                .map(|name| {
                    def.states
                        .iter()
                        .chain(&def.parameters)
                        .find(|symbol| &symbol.name == name)
                        .map(|symbol| symbol.dtype)
                        .or_else(|| {
                            def.linked_variables
                                .iter()
                                .find(|linked| &linked.name == name)
                                .map(|linked| linked.dtype)
                        })
                        .ok_or_else(|| "event monitor symbol missing from population".into())
                        .and_then(StateArray::empty)
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(EventMonitorRuntime {
                event_index: def
                    .events
                    .iter()
                    .position(|event| event == &monitor.event)
                    .ok_or("EventMonitor event missing from population")?,
                values,
                events: Vec::new(),
                samples,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    // ``window_steps`` is expressed on the population clock because it also
    // gates spike/event retention.  A StateMonitor can have a slower clock,
    // so reserve only the number of samples its own schedule can emit.
    let monitor_sample_steps = if def.state_monitors.is_empty() {
        0
    } else if def.monitor.window_steps < def.steps {
        def.monitor.window_steps
    } else {
        def.state_monitors
            .iter()
            .map(|monitor| monitor.clock)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .map(|clock| context.run_clocks[clock].steps)
            .sum()
    };
    let sample_count = monitor_sample_steps * def.monitor.record.len();
    let mut samples = samples;
    for sample in &mut samples {
        sample.try_reserve_exact(sample_count)?;
    }
    Ok(PopulationRuntime {
        states,
        subexpression_update,
        update,
        adaptive_update,
        spatial,
        thresholds,
        spike_schedule,
        spike_cursor: 0,
        poisson_inputs,
        regular,
        resets,
        refractory,
        tick,
        end_tick: tick + def.steps,
        fired: (0..def.events.len()).map(|_| Vec::new()).collect(),
        event_history: (0..def.events.len()).map(|_| Vec::new()).collect(),
        counts: vec![0; def.count],
        last_fired: Vec::new(),
        samples,
        spikes: Vec::new(),
        monitor_values,
        event_monitors,
    })
}

fn run_program_neurons(
    program: &mut Program,
    states: &mut [StateArray],
    synapse_states: &[StateArray],
    refractory: Option<&RefractoryRuntime>,
    count: usize,
    time: f64,
    tick: usize,
) -> Result<()> {
    program.prepare(time, tick)?;
    for start in (0..count).step_by(LANES) {
        let len = LANES.min(count - start);
        program.run(
            states,
            states,
            states,
            synapse_states,
            refractory,
            Batch::neurons(start, len),
        )?;
        program.commit_neurons(states, start, len);
    }
    Ok(())
}

fn run_population_update(
    program: &mut Program,
    states: &mut [StateArray],
    refractory: &mut Option<RefractoryRuntime>,
    count: usize,
    time: f64,
    tick: usize,
) -> Result<()> {
    program.prepare(time, tick)?;
    for start in (0..count).step_by(LANES) {
        let len = LANES.min(count - start);
        program.run(
            states,
            states,
            states,
            &[],
            refractory.as_ref(),
            Batch::neurons(start, len),
        )?;
        program.commit_neurons(states, start, len);
        if let Some(refractory) = refractory.as_mut() {
            program.commit_available(refractory, start, len);
        }
    }
    Ok(())
}

fn deliver_event(
    program: &mut Program,
    populations: &mut [PopulationRuntime],
    definition: &SynapseDefinition,
    synapse_states: &mut [StateArray],
    source: usize,
    target: usize,
    edge: usize,
) -> Result<()> {
    let batch = Batch {
        start: 0,
        len: 1,
        source,
        target,
        source_state: source + definition.source_start,
        target_state: target + definition.target_start,
        edge,
    };
    use std::cmp::Ordering;
    match definition
        .source_population
        .cmp(&definition.target_population)
    {
        Ordering::Equal => {
            let population = &mut populations[definition.source_population];
            program.run(
                &population.states,
                &population.states,
                &population.states,
                synapse_states,
                population.refractory.as_ref(),
                batch,
            )?;
            program.commit_event_same_population(
                &mut population.states,
                synapse_states,
                batch.source_state,
                batch.target_state,
                edge,
            );
        }
        Ordering::Less => {
            let (left, right) = populations.split_at_mut(definition.target_population);
            let source_pop = &mut left[definition.source_population];
            let target_pop = &mut right[0];
            program.run(
                &target_pop.states,
                &source_pop.states,
                &target_pop.states,
                synapse_states,
                target_pop.refractory.as_ref(),
                batch,
            )?;
            program.commit_event(
                &mut source_pop.states,
                &mut target_pop.states,
                synapse_states,
                batch.source_state,
                batch.target_state,
                edge,
            );
        }
        Ordering::Greater => {
            let (left, right) = populations.split_at_mut(definition.source_population);
            let target_pop = &mut left[definition.target_population];
            let source_pop = &mut right[0];
            program.run(
                &target_pop.states,
                &source_pop.states,
                &target_pop.states,
                synapse_states,
                target_pop.refractory.as_ref(),
                batch,
            )?;
            program.commit_event(
                &mut source_pop.states,
                &mut target_pop.states,
                synapse_states,
                batch.source_state,
                batch.target_state,
                edge,
            );
        }
    }
    Ok(())
}

fn deliver_event_to_synapse(
    program: &mut Program,
    populations: &mut [PopulationRuntime],
    definition: &SynapseDefinition,
    synapse_states: &mut [StateArray],
    target_states: &mut [StateArray],
    source: usize,
    target: usize,
    edge: usize,
) -> Result<()> {
    let batch = Batch {
        start: 0,
        len: 1,
        source,
        target,
        source_state: source + definition.source_start,
        target_state: target,
        edge,
    };
    let source_population = &mut populations[definition.source_population];
    program.run(
        target_states,
        &source_population.states,
        target_states,
        synapse_states,
        None,
        batch,
    )?;
    program.commit_event(
        &mut source_population.states,
        target_states,
        synapse_states,
        batch.source_state,
        batch.target_state,
        edge,
    );
    Ok(())
}

fn run_pathway(
    pathway: &mut PathwayRuntime,
    populations: &mut [PopulationRuntime],
    definition: &SynapseDefinition,
    source_indices: &[usize],
    target_indices: &[usize],
    synapse_states: &mut [StateArray],
    time: f64,
) -> Result<()> {
    let endpoint_population = if pathway.is_pre {
        definition.source_population
    } else {
        definition.target_population
    };
    let endpoint_start = if pathway.is_pre {
        definition.source_start
    } else {
        definition.target_start
    };
    let endpoint_count = if pathway.is_pre {
        definition.source_count
    } else {
        definition.target_count
    };
    let tick = populations[endpoint_population].tick;
    let end_tick = populations[endpoint_population].end_tick;
    pathway.active_sources.clear();
    pathway.active_sources.extend(
        populations[endpoint_population].fired[pathway.event_index]
            .iter()
            .copied()
            .filter(|&neuron| neuron >= endpoint_start && neuron < endpoint_start + endpoint_count)
            .map(|neuron| neuron - endpoint_start),
    );
    let queue_size = pathway.topology.queue.len();
    if let Some(delay) = pathway.topology.uniform_delay {
        let delivery = tick + delay;
        if delivery < end_tick {
            pathway.topology.queue[delivery % queue_size]
                .extend_from_slice(&pathway.active_sources);
        }
    } else {
        for &endpoint in &pathway.active_sources {
            for &edge in &pathway.topology.edges
                [pathway.topology.offsets[endpoint]..pathway.topology.offsets[endpoint + 1]]
            {
                let delivery = tick + pathway.delay_ticks[edge];
                if delivery < end_tick {
                    pathway.topology.queue[delivery % queue_size].push(edge);
                }
            }
        }
    }
    let mut pending = std::mem::take(&mut pathway.topology.queue[tick % queue_size]);
    pathway.program.prepare(time, tick)?;
    if pathway.topology.uniform_delay.is_some() {
        for &endpoint in &pending {
            let edges: Vec<_> = pathway.topology.edges
                [pathway.topology.offsets[endpoint]..pathway.topology.offsets[endpoint + 1]]
                .to_vec();
            for edge in edges {
                deliver_event(
                    &mut pathway.program,
                    populations,
                    definition,
                    synapse_states,
                    source_indices[edge],
                    target_indices[edge],
                    edge,
                )?;
                pathway.topology.delivered += 1;
            }
        }
    } else {
        for &edge in &pending {
            deliver_event(
                &mut pathway.program,
                populations,
                definition,
                synapse_states,
                source_indices[edge],
                target_indices[edge],
                edge,
            )?;
            pathway.topology.delivered += 1;
        }
    }
    pending.clear();
    pathway.topology.queue[tick % queue_size] = pending;
    Ok(())
}

fn run_pathway_to_synapse(
    pathway: &mut PathwayRuntime,
    populations: &mut [PopulationRuntime],
    definition: &SynapseDefinition,
    source_indices: &[usize],
    target_indices: &[usize],
    synapse_states: &mut [StateArray],
    target_states: &mut [StateArray],
    time: f64,
) -> Result<()> {
    check(pathway.is_pre, "a Synapses endpoint cannot generate events")?;
    let endpoint_population = definition.source_population;
    let endpoint_start = definition.source_start;
    let endpoint_count = definition.source_count;
    let tick = populations[endpoint_population].tick;
    let end_tick = populations[endpoint_population].end_tick;
    pathway.active_sources.clear();
    pathway.active_sources.extend(
        populations[endpoint_population].fired[pathway.event_index]
            .iter()
            .copied()
            .filter(|&neuron| neuron >= endpoint_start && neuron < endpoint_start + endpoint_count)
            .map(|neuron| neuron - endpoint_start),
    );
    let queue_size = pathway.topology.queue.len();
    if let Some(delay) = pathway.topology.uniform_delay {
        let delivery = tick + delay;
        if delivery < end_tick {
            pathway.topology.queue[delivery % queue_size]
                .extend_from_slice(&pathway.active_sources);
        }
    } else {
        for &endpoint in &pathway.active_sources {
            for &edge in &pathway.topology.edges
                [pathway.topology.offsets[endpoint]..pathway.topology.offsets[endpoint + 1]]
            {
                let delivery = tick + pathway.delay_ticks[edge];
                if delivery < end_tick {
                    pathway.topology.queue[delivery % queue_size].push(edge);
                }
            }
        }
    }
    let mut pending = std::mem::take(&mut pathway.topology.queue[tick % queue_size]);
    pathway.program.prepare(time, tick)?;
    if pathway.topology.uniform_delay.is_some() {
        for &endpoint in &pending {
            let edges: Vec<_> = pathway.topology.edges
                [pathway.topology.offsets[endpoint]..pathway.topology.offsets[endpoint + 1]]
                .to_vec();
            for edge in edges {
                deliver_event_to_synapse(
                    &mut pathway.program,
                    populations,
                    definition,
                    synapse_states,
                    target_states,
                    source_indices[edge],
                    target_indices[edge],
                    edge,
                )?;
                pathway.topology.delivered += 1;
            }
        }
    } else {
        for &edge in &pending {
            deliver_event_to_synapse(
                &mut pathway.program,
                populations,
                definition,
                synapse_states,
                target_states,
                source_indices[edge],
                target_indices[edge],
                edge,
            )?;
            pathway.topology.delivered += 1;
        }
    }
    pending.clear();
    pathway.topology.queue[tick % queue_size] = pending;
    Ok(())
}

fn run_edge_target_pathway(
    runtimes: &mut [SynapseRuntime],
    owner: usize,
    target_owner: usize,
    pathway_position: usize,
    populations: &mut [PopulationRuntime],
    definition: &SynapseDefinition,
    time: f64,
) -> Result<()> {
    check(
        owner != target_owner,
        "Synapses cannot target its own edge domain",
    )?;
    let (runtime, target_runtime) = if owner < target_owner {
        let (left, right) = runtimes.split_at_mut(target_owner);
        (&mut left[owner], &mut right[0])
    } else {
        let (left, right) = runtimes.split_at_mut(owner);
        (&mut right[0], &mut left[target_owner])
    };
    let SynapseRuntime {
        source,
        target,
        states,
        pathways,
        ..
    } = runtime;
    run_pathway_to_synapse(
        &mut pathways[pathway_position],
        populations,
        definition,
        source,
        target,
        states,
        &mut target_runtime.states,
        time,
    )
}

struct SynapseArrays<'a> {
    source: &'a [usize],
    target: &'a [usize],
    states: &'a mut [StateArray],
}

fn run_synapse_program(
    program: &mut Program,
    populations: &[PopulationRuntime],
    definition: &SynapseDefinition,
    arrays: SynapseArrays<'_>,
    time: f64,
    tick: usize,
) -> Result<()> {
    program.prepare(time, tick)?;
    for edge in 0..arrays.source.len() {
        let source_neuron = arrays.source[edge];
        let target_neuron = arrays.target[edge];
        let source_population = &populations[definition.source_population];
        let target_population = &populations[definition.target_population];
        program.run(
            &target_population.states,
            &source_population.states,
            &target_population.states,
            arrays.states,
            target_population.refractory.as_ref(),
            Batch {
                start: 0,
                len: 1,
                source: source_neuron,
                target: target_neuron,
                source_state: source_neuron + definition.source_start,
                target_state: target_neuron + definition.target_start,
                edge,
            },
        )?;
        program.commit_synapses(arrays.states, edge, 1);
    }
    Ok(())
}

fn run_summed_program(
    summed: &mut SummedRuntime,
    populations: &mut [PopulationRuntime],
    definition: &SynapseDefinition,
    source: &[usize],
    target: &[usize],
    synapse_states: &[StateArray],
    time: f64,
) -> Result<()> {
    populations[summed.target_population].states[summed.target_state].fill_f64(
        summed.target_start..summed.target_start + summed.target_count,
        0.0,
    );
    let tick = populations[summed.target_population].tick;
    summed.program.prepare(time, tick)?;
    for edge in 0..source.len() {
        let source_neuron = source[edge];
        let target_neuron = target[edge];
        let source_state = source_neuron + definition.source_start;
        let target_state = target_neuron + definition.target_start;
        {
            let source_population = &populations[definition.source_population];
            let target_population = &populations[definition.target_population];
            summed.program.run(
                &target_population.states,
                &source_population.states,
                &target_population.states,
                synapse_states,
                target_population.refractory.as_ref(),
                Batch {
                    start: 0,
                    len: 1,
                    source: source_neuron,
                    target: target_neuron,
                    source_state,
                    target_state,
                    edge,
                },
            )?;
        }
        let destination = if summed.target_is_pre {
            source_state
        } else {
            target_state
        };
        populations[summed.target_population].states[summed.target_state]
            .add_f64(destination, summed.program.registers[summed.result][0]);
    }
    Ok(())
}

fn run_summed_from_synapse(
    summed: &mut SummedRuntime,
    populations: &mut [PopulationRuntime],
    definition: &SynapseDefinition,
    source: &[usize],
    target: &[usize],
    synapse_states: &[StateArray],
    source_states: &[StateArray],
    time: f64,
) -> Result<()> {
    check(
        !summed.target_is_pre,
        "edge-domain summed input must target post",
    )?;
    populations[summed.target_population].states[summed.target_state].fill_f64(
        summed.target_start..summed.target_start + summed.target_count,
        0.0,
    );
    let tick = populations[summed.target_population].tick;
    summed.program.prepare(time, tick)?;
    for edge in 0..source.len() {
        let source_edge = source[edge];
        let target_neuron = target[edge];
        let target_state = target_neuron + definition.target_start;
        {
            let target_population = &populations[definition.target_population];
            summed.program.run(
                &target_population.states,
                source_states,
                &target_population.states,
                synapse_states,
                target_population.refractory.as_ref(),
                Batch {
                    start: 0,
                    len: 1,
                    source: source_edge,
                    target: target_neuron,
                    source_state: source_edge,
                    target_state,
                    edge,
                },
            )?;
        }
        populations[summed.target_population].states[summed.target_state]
            .add_f64(target_state, summed.program.registers[summed.result][0]);
    }
    Ok(())
}

fn run_summed_to_synapse(
    summed: &mut SummedRuntime,
    populations: &[PopulationRuntime],
    definition: &SynapseDefinition,
    source: &[usize],
    target: &[usize],
    synapse_states: &[StateArray],
    target_states: &mut [StateArray],
    time: f64,
) -> Result<()> {
    check(
        !summed.target_is_pre,
        "edge-domain summed input must target post",
    )?;
    target_states[summed.target_state].fill_f64(0..summed.target_count, 0.0);
    let tick = populations[definition.target_population].tick;
    summed.program.prepare(time, tick)?;
    for edge in 0..source.len() {
        let source_neuron = source[edge];
        let target_edge = target[edge];
        let source_state = source_neuron + definition.source_start;
        let source_population = &populations[definition.source_population];
        summed.program.run(
            target_states,
            &source_population.states,
            target_states,
            synapse_states,
            None,
            Batch {
                start: 0,
                len: 1,
                source: source_neuron,
                target: target_edge,
                source_state,
                target_state: target_edge,
                edge,
            },
        )?;
        target_states[summed.target_state]
            .add_f64(target_edge, summed.program.registers[summed.result][0]);
    }
    Ok(())
}

fn run_summed_between_synapses(
    summed: &mut SummedRuntime,
    populations: &[PopulationRuntime],
    definition: &SynapseDefinition,
    source: &[usize],
    target: &[usize],
    synapse_states: &[StateArray],
    source_states: &[StateArray],
    target_states: &mut [StateArray],
    time: f64,
) -> Result<()> {
    check(
        !summed.target_is_pre,
        "edge-domain summed input must target post",
    )?;
    target_states[summed.target_state].fill_f64(0..summed.target_count, 0.0);
    let tick = populations[definition.target_population].tick;
    summed.program.prepare(time, tick)?;
    for edge in 0..source.len() {
        let source_edge = source[edge];
        let target_edge = target[edge];
        summed.program.run(
            target_states,
            source_states,
            target_states,
            synapse_states,
            None,
            Batch {
                start: 0,
                len: 1,
                source: source_edge,
                target: target_edge,
                source_state: source_edge,
                target_state: target_edge,
                edge,
            },
        )?;
        target_states[summed.target_state]
            .add_f64(target_edge, summed.program.registers[summed.result][0]);
    }
    Ok(())
}

fn run_summed_within_synapse(
    summed: &mut SummedRuntime,
    populations: &[PopulationRuntime],
    definition: &SynapseDefinition,
    source: &[usize],
    target: &[usize],
    synapse_states: &[StateArray],
    endpoint_states: &mut [StateArray],
    time: f64,
) -> Result<()> {
    check(
        !summed.target_is_pre,
        "edge-domain summed input must target post",
    )?;
    endpoint_states[summed.target_state].fill_f64(0..summed.target_count, 0.0);
    let tick = populations[definition.target_population].tick;
    summed.program.prepare(time, tick)?;
    for edge in 0..source.len() {
        let source_edge = source[edge];
        let target_edge = target[edge];
        {
            let states = &*endpoint_states;
            summed.program.run(
                states,
                states,
                states,
                synapse_states,
                None,
                Batch {
                    start: 0,
                    len: 1,
                    source: source_edge,
                    target: target_edge,
                    source_state: source_edge,
                    target_state: target_edge,
                    edge,
                },
            )?;
        }
        endpoint_states[summed.target_state]
            .add_f64(target_edge, summed.program.registers[summed.result][0]);
    }
    Ok(())
}

fn three_distinct_mut<T>(
    values: &mut [T],
    first: usize,
    second: usize,
    third: usize,
) -> Result<(&mut T, &mut T, &mut T)> {
    check(
        first < values.len()
            && second < values.len()
            && third < values.len()
            && first != second
            && first != third
            && second != third,
        "edge-domain owners must be distinct",
    )?;
    if first < second {
        if second < third {
            let (before_third, from_third) = values.split_at_mut(third);
            let (before_second, from_second) = before_third.split_at_mut(second);
            Ok((
                &mut before_second[first],
                &mut from_second[0],
                &mut from_third[0],
            ))
        } else if first < third {
            let (before_second, from_second) = values.split_at_mut(second);
            let (before_third, from_third) = before_second.split_at_mut(third);
            Ok((
                &mut before_third[first],
                &mut from_second[0],
                &mut from_third[0],
            ))
        } else {
            let (before_second, from_second) = values.split_at_mut(second);
            let (before_first, from_first) = before_second.split_at_mut(first);
            Ok((
                &mut from_first[0],
                &mut from_second[0],
                &mut before_first[third],
            ))
        }
    } else if first < third {
        let (before_third, from_third) = values.split_at_mut(third);
        let (before_first, from_first) = before_third.split_at_mut(first);
        Ok((
            &mut from_first[0],
            &mut before_first[second],
            &mut from_third[0],
        ))
    } else if second < third {
        let (before_first, from_first) = values.split_at_mut(first);
        let (before_third, from_third) = before_first.split_at_mut(third);
        Ok((
            &mut from_first[0],
            &mut before_third[second],
            &mut from_third[0],
        ))
    } else {
        let (before_first, from_first) = values.split_at_mut(first);
        let (before_second, from_second) = before_first.split_at_mut(second);
        Ok((
            &mut from_first[0],
            &mut from_second[0],
            &mut before_second[third],
        ))
    }
}

fn run_edge_endpoint_summed(
    runtimes: &mut [SynapseRuntime],
    owner: usize,
    summed_position: usize,
    populations: &mut [PopulationRuntime],
    definition: &SynapseDefinition,
    time: f64,
) -> Result<()> {
    match (definition.source_synapse, definition.target_synapse) {
        (Some(source_owner), Some(target_owner)) if source_owner == target_owner => {
            check(
                owner != source_owner,
                "Synapses cannot target its own edge domain",
            )?;
            let (runtime, endpoint_runtime) = if owner < source_owner {
                let (left, right) = runtimes.split_at_mut(source_owner);
                (&mut left[owner], &mut right[0])
            } else {
                let (left, right) = runtimes.split_at_mut(owner);
                (&mut right[0], &mut left[source_owner])
            };
            let summed = runtime
                .summed
                .get_mut(summed_position)
                .ok_or("missing summed-variable runtime program")?;
            run_summed_within_synapse(
                summed,
                populations,
                definition,
                &runtime.source,
                &runtime.target,
                &runtime.states,
                &mut endpoint_runtime.states,
                time,
            )
        }
        (Some(source_owner), Some(target_owner)) => {
            let (runtime, source_runtime, target_runtime) =
                three_distinct_mut(runtimes, owner, source_owner, target_owner)?;
            let summed = runtime
                .summed
                .get_mut(summed_position)
                .ok_or("missing summed-variable runtime program")?;
            run_summed_between_synapses(
                summed,
                populations,
                definition,
                &runtime.source,
                &runtime.target,
                &runtime.states,
                &source_runtime.states,
                &mut target_runtime.states,
                time,
            )
        }
        (source_owner, target_owner) => {
            let endpoint_owner = source_owner
                .or(target_owner)
                .ok_or("missing edge endpoint")?;
            check(
                owner != endpoint_owner,
                "Synapses cannot target its own edge domain",
            )?;
            let (runtime, endpoint_runtime) = if owner < endpoint_owner {
                let (left, right) = runtimes.split_at_mut(endpoint_owner);
                (&mut left[owner], &mut right[0])
            } else {
                let (left, right) = runtimes.split_at_mut(owner);
                (&mut right[0], &mut left[endpoint_owner])
            };
            let summed = runtime
                .summed
                .get_mut(summed_position)
                .ok_or("missing summed-variable runtime program")?;
            if source_owner.is_some() {
                run_summed_from_synapse(
                    summed,
                    populations,
                    definition,
                    &runtime.source,
                    &runtime.target,
                    &runtime.states,
                    &endpoint_runtime.states,
                    time,
                )
            } else {
                run_summed_to_synapse(
                    summed,
                    populations,
                    definition,
                    &runtime.source,
                    &runtime.target,
                    &runtime.states,
                    &mut endpoint_runtime.states,
                    time,
                )
            }
        }
    }
}

fn emit_threshold(
    runtime: &mut PopulationRuntime,
    definition: &PopulationDefinition,
    threshold_position: usize,
    event_index: usize,
    is_spike: bool,
    time: f64,
) -> Result<()> {
    runtime.fired[event_index].clear();
    let threshold = runtime
        .thresholds
        .get_mut(threshold_position)
        .ok_or("scheduled threshold has no runtime program")?;
    threshold.prepare(time, runtime.tick)?;
    let condition = threshold.condition.ok_or("missing threshold condition")?;
    for start in (0..definition.count).step_by(LANES) {
        let len = LANES.min(definition.count - start);
        threshold.run(
            &runtime.states,
            &runtime.states,
            &runtime.states,
            &[],
            runtime.refractory.as_ref(),
            Batch::neurons(start, len),
        )?;
        for lane in 0..len {
            let neuron = start + lane;
            if threshold.registers[condition][lane] != 0.0
                && (!is_spike
                    || runtime
                        .refractory
                        .as_ref()
                        .is_none_or(|value| value.not_refractory[neuron]))
            {
                runtime.fired[event_index].push(neuron);
                if is_spike {
                    if let Some(refractory) = &mut runtime.refractory {
                        refractory.lastspike[neuron] = time;
                        refractory.not_refractory[neuron] = false;
                    }
                }
            }
        }
    }
    if runtime.tick >= runtime.end_tick - definition.monitor.window_steps {
        check(
            runtime.event_history[event_index].len() + runtime.fired[event_index].len()
                <= 10_000_000,
            "event stream recording budget exceeded",
        )?;
        runtime.event_history[event_index].extend(
            runtime.fired[event_index]
                .iter()
                .map(|&neuron| (runtime.tick, neuron)),
        );
    }
    Ok(())
}

fn emit_spike_generator(
    runtime: &mut PopulationRuntime,
    definition: &PopulationDefinition,
) -> Result<()> {
    runtime.fired[0].clear();
    while runtime.spike_cursor < runtime.spike_schedule.len()
        && runtime.spike_schedule[runtime.spike_cursor].0 < runtime.tick
    {
        runtime.spike_cursor += 1;
    }
    while runtime.spike_cursor < runtime.spike_schedule.len()
        && runtime.spike_schedule[runtime.spike_cursor].0 == runtime.tick
    {
        runtime.fired[0].push(runtime.spike_schedule[runtime.spike_cursor].1);
        runtime.spike_cursor += 1;
    }
    if runtime.tick >= runtime.end_tick - definition.monitor.window_steps {
        check(
            runtime.event_history[0].len() + runtime.fired[0].len() <= 10_000_000,
            "event stream recording budget exceeded",
        )?;
        runtime.event_history[0].extend(
            runtime.fired[0]
                .iter()
                .map(|&neuron| (runtime.tick, neuron)),
        );
    }
    Ok(())
}

fn record_spikes(runtime: &mut PopulationRuntime, definition: &PopulationDefinition) {
    if runtime.tick < runtime.end_tick - definition.monitor.window_steps {
        return;
    }
    let spike = definition
        .events
        .iter()
        .position(|event| event == "spike")
        .expect("validated SpikeMonitor event");
    for &neuron in &runtime.fired[spike] {
        runtime.counts[neuron] += 1;
        runtime.spikes.push((runtime.tick, neuron));
    }
}

fn record_events(runtime: &mut PopulationRuntime, monitor_index: usize) -> Result<()> {
    let event_index = runtime
        .event_monitors
        .get(monitor_index)
        .ok_or("scheduled EventMonitor has no runtime state")?
        .event_index;
    let fired = runtime.fired[event_index].clone();
    let monitor = &mut runtime.event_monitors[monitor_index];
    check(
        monitor.events.len() + fired.len() <= 10_000_000
            && monitor.samples.iter().map(StateArray::len).sum::<usize>()
                + fired.len() * monitor.values.len()
                <= 10_000_000,
        "EventMonitor recording budget exceeded",
    )?;
    for neuron in fired {
        monitor.events.push((runtime.tick, neuron));
        for (sample, value) in monitor.samples.iter_mut().zip(&monitor.values) {
            sample.push_register(match value {
                MonitorValue::State(state) => runtime.states[*state].get_register(neuron),
                MonitorValue::Scalar(value) => *value,
                MonitorValue::Parameter(values) => values[neuron],
                MonitorValue::Linked(input) => input.linked_value(neuron)?,
            });
        }
    }
    Ok(())
}

fn record_states(runtime: &mut PopulationRuntime, definition: &PopulationDefinition) -> Result<()> {
    if runtime.tick < runtime.end_tick - definition.monitor.window_steps {
        return Ok(());
    }
    for &neuron in &definition.monitor.record {
        for (sample, value) in runtime.samples.iter_mut().zip(&runtime.monitor_values) {
            sample.push_register(match value {
                MonitorValue::State(state) => runtime.states[*state].get_register(neuron),
                MonitorValue::Scalar(value) => *value,
                MonitorValue::Parameter(values) => values[neuron],
                MonitorValue::Linked(input) => input.linked_value(neuron)?,
            });
        }
    }
    Ok(())
}

fn record_synapse_states(
    runtime: &mut SynapseRuntime,
    definition: &SynapseDefinition,
    populations: &[PopulationRuntime],
    monitor_index: usize,
) -> Result<()> {
    let monitor_definition = definition
        .state_monitors
        .get(monitor_index)
        .ok_or("scheduled synapse StateMonitor has no definition")?;
    let SynapseRuntime {
        source,
        target,
        states,
        monitors,
        ..
    } = runtime;
    let monitor = monitors
        .get_mut(monitor_index)
        .ok_or("scheduled synapse StateMonitor has no runtime state")?;
    check(
        monitor.samples.iter().map(StateArray::len).sum::<usize>()
            + monitor_definition.record.len() * monitor.samples.len()
            <= 100_000_000,
        "synapse StateMonitor recording budget exceeded",
    )?;
    for &edge in &monitor_definition.record {
        for (sample, value) in monitor.samples.iter_mut().zip(&monitor.values) {
            sample.push_register(match value {
                SynapseMonitorValue::SynapseState(state) => states[*state].get_register(edge),
                SynapseMonitorValue::PreState(state) => populations[definition.source_population]
                    .states[*state]
                    .get_register(source[edge] + definition.source_start),
                SynapseMonitorValue::PostState(state) => populations[definition.target_population]
                    .states[*state]
                    .get_register(target[edge] + definition.target_start),
                SynapseMonitorValue::Linked(input) => input.linked_value(edge)?,
            });
        }
    }
    Ok(())
}

const DUMP_MAGIC: &[u8; 8] = b"B2DMP001";
const DUMP_END: &[u8; 8] = b"B2END001";
const EVENT_DUMP_MAGIC: &[u8; 8] = b"B2EVT001";
const EVENT_DUMP_END: &[u8; 8] = b"B2EEND01";

fn dump_u64(writer: &mut impl Write, value: usize) -> Result<()> {
    writer.write_all(&(value as u64).to_le_bytes())?;
    Ok(())
}

#[allow(unknown_lints, clippy::chunks_exact_to_as_chunks)]
fn dump_f64(writer: &mut impl Write, values: &[f64]) -> Result<()> {
    let mut bytes = [0u8; 8192];
    for chunk in values.chunks(1024) {
        for (slot, value) in bytes.chunks_exact_mut(8).zip(chunk) {
            slot.copy_from_slice(&value.to_le_bytes());
        }
        writer.write_all(&bytes[..chunk.len() * 8])?;
    }
    Ok(())
}

#[allow(unknown_lints, clippy::chunks_exact_to_as_chunks)]
fn dump_f32(writer: &mut impl Write, values: &[f32]) -> Result<()> {
    let mut bytes = [0u8; 8192];
    for chunk in values.chunks(2048) {
        for (slot, value) in bytes.chunks_exact_mut(4).zip(chunk) {
            slot.copy_from_slice(&value.to_le_bytes());
        }
        writer.write_all(&bytes[..chunk.len() * 4])?;
    }
    Ok(())
}

macro_rules! dump_integer {
    ($name:ident, $ty:ty, $width:expr, $chunk:expr) => {
        fn $name(writer: &mut impl Write, values: &[$ty]) -> Result<()> {
            let mut bytes = [0u8; 8192];
            for chunk in values.chunks($chunk) {
                for (slot, value) in bytes.chunks_exact_mut($width).zip(chunk) {
                    slot.copy_from_slice(&value.to_le_bytes());
                }
                writer.write_all(&bytes[..chunk.len() * $width])?;
            }
            Ok(())
        }
    };
}

dump_integer!(dump_i32, i32, 4, 2048);
dump_integer!(dump_i64, i64, 8, 1024);
dump_integer!(dump_u32, u32, 4, 2048);
dump_integer!(dump_u64_values, u64, 8, 1024);

#[allow(unknown_lints, clippy::chunks_exact_to_as_chunks)]
fn dump_indices(writer: &mut impl Write, values: &[usize]) -> Result<()> {
    let mut bytes = [0u8; 8192];
    for chunk in values.chunks(1024) {
        for (slot, &value) in bytes.chunks_exact_mut(8).zip(chunk) {
            slot.copy_from_slice(&(value as i64).to_le_bytes());
        }
        writer.write_all(&bytes[..chunk.len() * 8])?;
    }
    Ok(())
}

#[allow(unknown_lints, clippy::chunks_exact_to_as_chunks)]
fn dump_spikes(writer: &mut impl Write, values: &[(usize, usize)]) -> Result<()> {
    let mut bytes = [0u8; 8192];
    for chunk in values.chunks(512) {
        for (slot, &(tick, index)) in bytes.chunks_exact_mut(16).zip(chunk) {
            slot[..8].copy_from_slice(&(tick as i64).to_le_bytes());
            slot[8..].copy_from_slice(&(index as i64).to_le_bytes());
        }
        writer.write_all(&bytes[..chunk.len() * 16])?;
    }
    Ok(())
}

fn dump_population(
    writer: &mut impl Write,
    def: &PopulationDefinition,
    runtime: &PopulationRuntime,
) -> Result<()> {
    dump_u64(writer, def.count)?;
    dump_u64(writer, def.steps)?;
    dump_u64(writer, def.monitor.record.len())?;
    dump_u64(writer, def.monitor.variables.len())?;
    dump_u64(writer, def.states.len())?;
    dump_u64(writer, runtime.spikes.len())?;
    dump_u64(writer, runtime.last_fired.len())?;
    dump_u64(writer, usize::from(runtime.refractory.is_some()))?;
    for samples in &runtime.samples {
        samples.dump(writer)?;
    }
    dump_spikes(writer, &runtime.spikes)?;
    dump_indices(writer, &runtime.counts)?;
    dump_indices(writer, &runtime.last_fired)?;
    for state in &runtime.states {
        state.dump(writer)?;
    }
    if let Some(refractory) = &runtime.refractory {
        dump_f64(writer, &refractory.lastspike)?;
        for &available in &refractory.not_refractory {
            writer.write_all(&[u8::from(available)])?;
        }
    }
    Ok(())
}

fn dump_size(
    definition: &Definition,
    populations: &[PopulationRuntime],
    synapses: &[SynapseRuntime],
) -> usize {
    let mut size = 40usize;
    for (def, runtime) in definition.populations.iter().zip(populations) {
        size += 64
            + runtime
                .samples
                .iter()
                .map(StateArray::byte_len)
                .sum::<usize>()
            + runtime.spikes.len() * 16
            + runtime.counts.len() * 8
            + runtime.last_fired.len() * 8
            + runtime
                .states
                .iter()
                .map(StateArray::byte_len)
                .sum::<usize>();
        if let Some(refractory) = &runtime.refractory {
            size += refractory.lastspike.len() * 8 + refractory.not_refractory.len();
        }
        debug_assert_eq!(runtime.counts.len(), def.count);
    }
    size + 24
        + synapses
            .iter()
            .map(|synapse| {
                24 + synapse
                    .states
                    .iter()
                    .map(StateArray::byte_len)
                    .sum::<usize>()
                    + synapse
                        .monitors
                        .iter()
                        .flat_map(|monitor| &monitor.samples)
                        .map(StateArray::byte_len)
                        .sum::<usize>()
            })
            .sum::<usize>()
}

fn write_dump(
    model: &Model,
    populations: &[PopulationRuntime],
    synapses: &[SynapseRuntime],
    mut writer: impl Write,
) -> Result<usize> {
    let size = dump_size(&model.definition, populations, synapses);
    writer.write_all(DUMP_MAGIC)?;
    writer.write_all(&3u32.to_le_bytes())?;
    writer.write_all(&0x0102_0304u32.to_le_bytes())?;
    dump_u64(&mut writer, model.definition.populations.len())?;
    dump_u64(&mut writer, model.instance.neuron_count)?;
    dump_u64(&mut writer, size)?;
    for (def, runtime) in model.definition.populations.iter().zip(populations) {
        dump_population(&mut writer, def, runtime)?;
    }
    dump_u64(&mut writer, synapses.len())?;
    for (def, synapse) in model.definition.synapses.iter().zip(synapses) {
        dump_u64(&mut writer, def.states.len())?;
        dump_u64(&mut writer, synapse.source.len())?;
        for state in &synapse.states {
            state.dump(&mut writer)?;
        }
        for monitor in &synapse.monitors {
            for sample in &monitor.samples {
                sample.dump(&mut writer)?;
            }
        }
        dump_u64(
            &mut writer,
            synapse
                .pathways
                .iter()
                .map(|pathway| pathway.topology.delivered)
                .sum(),
        )?;
    }
    let final_time = decode_bits(&model.run.start)? + decode_bits(&model.run.duration)?;
    writer.write_all(&final_time.to_le_bytes())?;
    writer.write_all(DUMP_END)?;
    writer.flush()?;
    Ok(size)
}

fn write_event_monitor_dump(
    definition: &Definition,
    populations: &[PopulationRuntime],
    mut writer: impl Write,
) -> Result<usize> {
    let monitor_count: usize = definition
        .populations
        .iter()
        .map(|population| population.event_monitors.len())
        .sum();
    let stream_count: usize = definition
        .populations
        .iter()
        .map(|population| population.events.len())
        .sum();
    if stream_count == 0 && monitor_count == 0 {
        return Ok(0);
    }
    let size = 32usize
        + definition
            .populations
            .iter()
            .zip(populations)
            .flat_map(|(def, runtime)| def.events.iter().zip(&runtime.event_history))
            .map(|(_, events)| 8 + events.len() * 16)
            .sum::<usize>()
        + populations
            .iter()
            .flat_map(|population| &population.event_monitors)
            .map(|monitor| {
                16 + monitor.events.len() * 16
                    + monitor
                        .samples
                        .iter()
                        .map(StateArray::byte_len)
                        .sum::<usize>()
            })
            .sum::<usize>();
    writer.write_all(EVENT_DUMP_MAGIC)?;
    dump_u64(&mut writer, stream_count)?;
    for (_, events) in definition
        .populations
        .iter()
        .zip(populations)
        .flat_map(|(def, runtime)| def.events.iter().zip(&runtime.event_history))
    {
        dump_u64(&mut writer, events.len())?;
        dump_spikes(&mut writer, events)?;
    }
    dump_u64(&mut writer, monitor_count)?;
    for monitor in populations
        .iter()
        .flat_map(|population| &population.event_monitors)
    {
        dump_u64(&mut writer, monitor.events.len())?;
        dump_u64(&mut writer, monitor.values.len())?;
        dump_spikes(&mut writer, &monitor.events)?;
        for samples in &monitor.samples {
            samples.dump(&mut writer)?;
        }
    }
    writer.write_all(EVENT_DUMP_END)?;
    writer.flush()?;
    Ok(size)
}

impl PopulationRuntime {
    fn bind_links(&mut self, state_pointers: &[Vec<usize>]) {
        for program in self
            .subexpression_update
            .iter_mut()
            .chain(self.update.iter_mut())
            .chain(self.spatial.iter_mut().map(|spatial| &mut spatial.program))
            .chain(self.thresholds.iter_mut())
            .chain(self.poisson_inputs.iter_mut())
            .chain(self.regular.iter_mut())
            .chain(self.resets.iter_mut())
        {
            program.bind_links(state_pointers);
        }
        if let Some(adaptive) = &mut self.adaptive_update {
            adaptive.program.bind_links(state_pointers);
        }
        for value in &mut self.monitor_values {
            if let MonitorValue::Linked(input) = value {
                input.bind_link(state_pointers);
            }
        }
        for monitor in &mut self.event_monitors {
            for value in &mut monitor.values {
                if let MonitorValue::Linked(input) = value {
                    input.bind_link(state_pointers);
                }
            }
        }
    }
}

impl SynapseRuntime {
    fn bind_links(&mut self, state_pointers: &[Vec<usize>]) {
        for program in self
            .subexpression_update
            .iter_mut()
            .chain(self.update.iter_mut())
            .chain(self.regular.iter_mut())
            .chain(self.pathways.iter_mut().map(|pathway| &mut pathway.program))
            .chain(self.summed.iter_mut().map(|summed| &mut summed.program))
        {
            program.bind_links(state_pointers);
        }
        for monitor in &mut self.monitors {
            for value in &mut monitor.values {
                if let SynapseMonitorValue::Linked(input) = value {
                    input.bind_link(state_pointers);
                }
            }
        }
    }
}

pub(super) struct Runtime {
    model: Model,
    populations: Vec<PopulationRuntime>,
    synapse_runtimes: Vec<SynapseRuntime>,
    clock_ticks: Vec<usize>,
    clock_end_ticks: Vec<usize>,
    clock_epsilon: f64,
    failed: bool,
    dispatch: Vec<usize>,
}

impl Runtime {
    pub(super) fn new(model: Model, dispatch: Vec<usize>) -> Result<Self> {
        let d = &model.definition;
        let population_context = PopulationBuildContext {
            populations: &d.populations,
            run_clocks: &model.run.clocks,
            rng_seed: model.instance.rng_seed,
            functions: &d.functions,
        };
        let mut populations = d
            .populations
            .iter()
            .zip(&model.instance.populations)
            .enumerate()
            .map(|(population, (def, instance))| {
                population_runtime(population, def, instance, &population_context)
            })
            .collect::<Result<Vec<_>>>()?;
        let state_pointers: Vec<Vec<usize>> = populations
            .iter()
            .map(|population| {
                population
                    .states
                    .iter()
                    .map(|state| state as *const StateArray as usize)
                    .collect()
            })
            .collect();
        for population in &mut populations {
            population.bind_links(&state_pointers);
        }
        let mut synapse_runtimes = d
            .synapses
            .iter()
            .zip(&model.instance.synapses)
            .map(|(def, instance)| -> Result<SynapseRuntime> {
                let (source, target, generated_parameters, generated_delays) =
                    materialize_projection(d, def, instance)?;
                let source_def = &d.populations[def.source_population];
                let target_def = &d.populations[def.target_population];
                let source_state_symbols = def
                    .source_synapse
                    .map(|source| d.synapses[source].states.as_slice())
                    .unwrap_or(source_def.states.as_slice());
                let target_state_symbols = def
                    .target_synapse
                    .map(|target| d.synapses[target].states.as_slice())
                    .unwrap_or(target_def.states.as_slice());
                let source_indices: BTreeMap<_, _> = source_state_symbols
                    .iter()
                    .enumerate()
                    .map(|(i, s)| (s.name.clone(), i))
                    .collect();
                let target_indices: BTreeMap<_, _> = target_state_symbols
                    .iter()
                    .enumerate()
                    .map(|(i, s)| (s.name.clone(), i))
                    .collect();
                let synapse_indices: BTreeMap<_, _> = def
                    .states
                    .iter()
                    .enumerate()
                    .map(|(i, s)| (s.name.clone(), i))
                    .collect();
                let states = def
                    .states
                    .iter()
                    .map(|symbol| {
                        StateArray::from_encoded(symbol, &instance.initial_state[&symbol.name])
                    })
                    .collect::<Result<Vec<_>>>()?;
                let dt = decode_bits(&source_def.dt)?;
                let mut inputs = BTreeMap::from([
                    ("dt".to_owned(), Input::Constant(dt, DType::F64)),
                    ("t".to_owned(), Input::Time),
                    (
                        "N".to_owned(),
                        Input::Constant(source.len() as f64, DType::Index),
                    ),
                    (
                        "N_pre".to_owned(),
                        Input::Constant(def.source_count as f64, DType::Index),
                    ),
                    (
                        "N_post".to_owned(),
                        Input::Constant(def.target_count as f64, DType::Index),
                    ),
                    ("i".to_owned(), Input::Index(Domain::Pre)),
                    ("j".to_owned(), Input::Index(Domain::Post)),
                ]);
                for (name, &position) in &synapse_indices {
                    inputs.insert(
                        name.clone(),
                        Input::SynapseState(position, def.states[position].dtype),
                    );
                }
                for linked in &def.linked_variables {
                    inputs.insert(
                        linked.name.clone(),
                        synapse_linked_input(&d.populations, linked)?,
                    );
                }
                let mut writable = BTreeMap::new();
                for (alias, state) in &def.pre_state_aliases {
                    let position = source_indices[state];
                    inputs.insert(
                        alias.clone(),
                        Input::PreState(position, source_state_symbols[position].dtype),
                    );
                    writable.insert(alias.clone(), WriteTarget::PreNeuron(position));
                }
                for (alias, state) in &def.post_state_aliases {
                    let position = target_indices[state];
                    inputs.insert(
                        alias.clone(),
                        Input::PostState(position, target_state_symbols[position].dtype),
                    );
                    writable.insert(alias.clone(), WriteTarget::Neuron(target_indices[state]));
                }
                for (name, &position) in &synapse_indices {
                    writable.insert(name.clone(), WriteTarget::Synapse(position));
                }
                if def.target_synapse.is_none() && target_def.refractory.is_some() {
                    inputs.insert(
                        "not_refractory_post".to_owned(),
                        Input::Available(Domain::Post),
                    );
                }
                add_parameters(
                    &mut inputs,
                    &def.parameters,
                    &instance.parameters,
                    Domain::Edge,
                    &generated_parameters,
                )?;
                let clock_driven: BTreeMap<_, _> = def
                    .clock_driven_states
                    .iter()
                    .map(|name| (name.clone(), WriteTarget::Synapse(synapse_indices[name])))
                    .collect();
                let update = def
                    .code_objects
                    .iter()
                    .find(|code| code.kind == "synapse_state_update")
                    .map(|code| {
                        Program::compile(
                            code,
                            inputs.clone(),
                            &clock_driven,
                            model.instance.rng_seed,
                            &d.functions,
                        )
                    })
                    .transpose()?;
                let regular = def
                    .code_objects
                    .iter()
                    .filter(|code| code.kind == "synapse_run_regularly")
                    .map(|code| {
                        Program::compile(
                            code,
                            inputs.clone(),
                            &writable,
                            model.instance.rng_seed,
                            &d.functions,
                        )
                    })
                    .collect::<Result<Vec<_>>>()?;
                let subexpression_update = def
                    .code_objects
                    .iter()
                    .find(|code| code.kind == "synapse_subexpression_update")
                    .map(|code| {
                        Program::compile(
                            code,
                            inputs.clone(),
                            &writable,
                            model.instance.rng_seed,
                            &d.functions,
                        )
                    })
                    .transpose()?;
                let summed = def
                    .code_objects
                    .iter()
                    .filter(|code| code.kind == "summed_variable")
                    .map(|code| -> Result<SummedRuntime> {
                        let target_is_pre = code.summed_target.as_deref() == Some("pre");
                        let target_population = if target_is_pre {
                            def.source_population
                        } else {
                            def.target_population
                        };
                        let target_definition = &d.populations[target_population];
                        let target_synapse = if target_is_pre {
                            def.source_synapse
                        } else {
                            def.target_synapse
                        };
                        let target_state_symbols = target_synapse
                            .map(|target| d.synapses[target].states.as_slice())
                            .unwrap_or(target_definition.states.as_slice());
                        let target_indices: BTreeMap<_, _> = target_state_symbols
                            .iter()
                            .enumerate()
                            .map(|(index, symbol)| (symbol.name.clone(), index))
                            .collect();
                        let target_state_name = code
                            .summed_state
                            .as_ref()
                            .ok_or("missing summed target state")?;
                        let mut summed_inputs = inputs.clone();
                        summed_inputs.insert(
                            "dt".to_owned(),
                            Input::Constant(decode_bits(&target_definition.dt)?, DType::F64),
                        );
                        let program = Program::compile(
                            code,
                            summed_inputs,
                            &BTreeMap::new(),
                            model.instance.rng_seed,
                            &d.functions,
                        )?;
                        let result = *program
                            .symbols
                            .get("_synaptic_var")
                            .ok_or("missing summed expression result")?;
                        let target_code_objects = target_synapse
                            .map(|target| d.synapses[target].code_objects.as_slice())
                            .unwrap_or(target_definition.code_objects.as_slice());
                        let before_state = target_code_objects
                            .iter()
                            .find(|candidate| {
                                candidate.kind == "state_update"
                                    || candidate.kind == "synapse_state_update"
                            })
                            .is_none_or(|state| {
                                (code.order, code.name.as_str())
                                    < (state.order, state.name.as_str())
                            });
                        Ok(SummedRuntime {
                            program,
                            result,
                            before_state,
                            target_population,
                            target_start: if target_is_pre {
                                def.source_start
                            } else {
                                def.target_start
                            },
                            target_count: if target_is_pre {
                                def.source_count
                            } else {
                                def.target_count
                            },
                            target_state: target_indices[target_state_name],
                            target_is_pre,
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                let mut post_writable: BTreeMap<_, _> = synapse_indices
                    .iter()
                    .map(|(name, &position)| (name.clone(), WriteTarget::Synapse(position)))
                    .collect();
                post_writable.extend(def.post_state_aliases.iter().map(|(alias, state)| {
                    (alias.clone(), WriteTarget::Neuron(target_indices[state]))
                }));
                let pathways = instance
                    .pathways
                    .iter()
                    .enumerate()
                    .map(|(pathway_position, pathway)| -> Result<PathwayRuntime> {
                        let is_pre = pathway.kind == "pre";
                        let code = def
                            .code_objects
                            .iter()
                            .find(|code| {
                                code.pathway_name.as_deref() == Some(pathway.name.as_str())
                            })
                            .ok_or("missing pathway code object")?;
                        let population = if is_pre {
                            def.source_population
                        } else {
                            def.target_population
                        };
                        let population_definition = &d.populations[population];
                        let mut pathway_inputs = inputs.clone();
                        pathway_inputs.insert(
                            "dt".to_owned(),
                            Input::Constant(decode_bits(&population_definition.dt)?, DType::F64),
                        );
                        let program = Program::compile(
                            code,
                            pathway_inputs,
                            if is_pre { &writable } else { &post_writable },
                            model.instance.rng_seed,
                            &d.functions,
                        )?;
                        let (indices, endpoint_count) = if is_pre {
                            (&source, def.source_count)
                        } else {
                            (&target, def.target_count)
                        };
                        Ok(PathwayRuntime {
                            program,
                            topology: PathwayQueue::new(
                                pathway,
                                &generated_delays[pathway_position],
                                indices,
                                endpoint_count,
                                population_definition.steps,
                            ),
                            delay_ticks: generated_delays[pathway_position].clone(),
                            is_pre,
                            event_index: d.populations[if is_pre {
                                def.source_population
                            } else {
                                def.target_population
                            }]
                            .events
                            .iter()
                            .position(|event| event == &pathway.event)
                            .ok_or("pathway event missing from endpoint")?,
                            active_sources: Vec::new(),
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                let monitors = def
                    .state_monitors
                    .iter()
                    .map(|monitor| -> Result<SynapseMonitorRuntime> {
                        let values = monitor
                            .sources
                            .iter()
                            .map(|source| -> Result<SynapseMonitorValue> {
                                Ok(match source {
                                    SynapseMonitorSourceDefinition::SynapseState {
                                        name, ..
                                    } => SynapseMonitorValue::SynapseState(synapse_indices[name]),
                                    SynapseMonitorSourceDefinition::PreState { name, .. } => {
                                        SynapseMonitorValue::PreState(source_indices[name])
                                    }
                                    SynapseMonitorSourceDefinition::PostState { name, .. } => {
                                        SynapseMonitorValue::PostState(target_indices[name])
                                    }
                                    SynapseMonitorSourceDefinition::Linked { name, .. } => {
                                        let linked = def
                                            .linked_variables
                                            .iter()
                                            .find(|linked| &linked.name == name)
                                            .ok_or("synapse monitor linked source missing")?;
                                        SynapseMonitorValue::Linked(synapse_linked_input(
                                            &d.populations,
                                            linked,
                                        )?)
                                    }
                                })
                            })
                            .collect::<Result<Vec<_>>>()?;
                        let mut samples = monitor
                            .sources
                            .iter()
                            .map(|source| {
                                StateArray::empty(match source {
                                    SynapseMonitorSourceDefinition::SynapseState {
                                        dtype, ..
                                    }
                                    | SynapseMonitorSourceDefinition::PreState { dtype, .. }
                                    | SynapseMonitorSourceDefinition::PostState { dtype, .. } => {
                                        *dtype
                                    }
                                    SynapseMonitorSourceDefinition::Linked { dtype, .. } => *dtype,
                                })
                            })
                            .collect::<Result<Vec<_>>>()?;
                        let sample_count =
                            model.run.clocks[monitor.clock].steps * monitor.record.len();
                        for sample in &mut samples {
                            sample.try_reserve_exact(sample_count)?;
                        }
                        Ok(SynapseMonitorRuntime { values, samples })
                    })
                    .collect::<Result<Vec<_>>>()?;
                Ok(SynapseRuntime {
                    source,
                    target,
                    states,
                    subexpression_update,
                    summed,
                    update,
                    regular,
                    pathways,
                    monitors,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        for synapse in &mut synapse_runtimes {
            synapse.bind_links(&state_pointers);
        }

        let clock_ticks: Vec<_> = model
            .run
            .clocks
            .iter()
            .map(|clock| clock.start_tick)
            .collect();
        let clock_end_ticks: Vec<_> = model
            .run
            .clocks
            .iter()
            .map(|clock| clock.start_tick + clock.steps)
            .collect();
        let clock_epsilon = d
            .clocks
            .iter()
            .map(|clock| decode_bits(&clock.dt).unwrap())
            .reduce(f64::min)
            .unwrap()
            * 1e-12;
        Ok(Self {
            model,
            populations,
            synapse_runtimes,
            clock_ticks,
            clock_end_ticks,
            clock_epsilon,
            dispatch,
            failed: false,
        })
    }

    pub(super) fn spike_count(&self) -> usize {
        self.populations
            .iter()
            .map(|population| population.spikes.len())
            .sum()
    }

    pub(super) fn is_finished(&self) -> bool {
        self.clock_ticks
            .iter()
            .zip(&self.clock_end_ticks)
            .all(|(tick, end)| tick >= end)
    }

    pub(super) fn step(&mut self, max_ticks: usize) -> Result<bool> {
        check(max_ticks > 0, "step requires a positive tick budget")?;
        check(!self.failed, "execution failed; create a new executor")?;
        let result = self.advance(max_ticks);
        if result.is_err() {
            self.failed = true;
        }
        result
    }

    fn advance(&mut self, max_ticks: usize) -> Result<bool> {
        let Self {
            model,
            populations,
            synapse_runtimes,
            clock_ticks,
            clock_end_ticks,
            clock_epsilon,
            dispatch,
            ..
        } = self;
        let d = &model.definition;
        let clock_epsilon = *clock_epsilon;
        for _ in 0..max_ticks {
            let next_time = clock_ticks
                .iter()
                .zip(clock_end_ticks.iter())
                .zip(&d.clocks)
                .filter(|((&tick, &end_tick), _)| tick < end_tick)
                .map(|((&tick, _), clock)| tick as f64 * decode_bits(&clock.dt).unwrap())
                .reduce(f64::min);
            let Some(time) = next_time else { break };
            let active_clocks: Vec<bool> = clock_ticks
                .iter()
                .zip(clock_end_ticks.iter())
                .zip(&d.clocks)
                .map(|((&tick, &end_tick), clock)| {
                    if tick >= end_tick {
                        return false;
                    }
                    let local = tick as f64 * decode_bits(&clock.dt).unwrap();
                    (local - time).abs() <= clock_epsilon
                })
                .collect();
            let population_active: Vec<bool> = d
                .populations
                .iter()
                .map(|population| active_clocks[population.clock])
                .collect();
            // Retained only for the unreachable legacy phase fallback below.
            let active = population_active.clone();

            // The validated schedule is the reference executor's semantic source
            // of truth.  AOT backends may later fuse independent nodes, but this
            // path deliberately dispatches the total Brian order one node at a
            // time so ordering changes cannot hide in hand-written phase loops.
            if !d.schedule.nodes.is_empty() {
                let mut recorded_state = vec![false; populations.len()];
                for &node_index in dispatch.iter() {
                    let node = &d.schedule.nodes[node_index];
                    if !active_clocks[node.clock] {
                        continue;
                    }
                    // The scheduler groups clocks within epsilon, but Brian's t
                    // variables retain each owner's exact tick * dt value.
                    let owner_clock = match (node.operation, node.owner_kind) {
                        (_, ScheduleOwnerKind::Population) => d.populations[node.owner_index].clock,
                        (ScheduleOperation::CodeObject, ScheduleOwnerKind::Synapse) => node.clock,
                        (_, ScheduleOwnerKind::Synapse) => node.clock,
                    };
                    let time =
                        clock_ticks[owner_clock] as f64 * decode_bits(&d.clocks[owner_clock].dt)?;
                    match (node.operation, node.owner_kind) {
                        (ScheduleOperation::StateMonitor, ScheduleOwnerKind::Population) => {
                            if !recorded_state[node.owner_index] {
                                record_states(
                                    &mut populations[node.owner_index],
                                    &d.populations[node.owner_index],
                                )?;
                                recorded_state[node.owner_index] = true;
                            }
                        }
                        (ScheduleOperation::StateMonitor, ScheduleOwnerKind::Synapse) => {
                            record_synapse_states(
                                &mut synapse_runtimes[node.owner_index],
                                &d.synapses[node.owner_index],
                                &populations,
                                node.item_index,
                            )?;
                        }
                        (ScheduleOperation::SpikeMonitor, ScheduleOwnerKind::Population) => {
                            record_spikes(
                                &mut populations[node.owner_index],
                                &d.populations[node.owner_index],
                            );
                        }
                        (ScheduleOperation::EventMonitor, ScheduleOwnerKind::Population) => {
                            record_events(&mut populations[node.owner_index], node.item_index)?;
                        }
                        (ScheduleOperation::EventSource, ScheduleOwnerKind::Population) => {
                            emit_spike_generator(
                                &mut populations[node.owner_index],
                                &d.populations[node.owner_index],
                            )?;
                        }
                        (ScheduleOperation::CodeObject, ScheduleOwnerKind::Population) => {
                            let definition = &d.populations[node.owner_index];
                            let code = &definition.code_objects[node.item_index];
                            let runtime = &mut populations[node.owner_index];
                            match code.kind.as_str() {
                                "subexpression_update" => run_program_neurons(
                                    runtime
                                        .subexpression_update
                                        .as_mut()
                                        .ok_or("missing population subexpression program")?,
                                    &mut runtime.states,
                                    &[],
                                    runtime.refractory.as_ref(),
                                    definition.count,
                                    time,
                                    clock_ticks[node.clock],
                                )?,
                                "state_update" => {
                                    let dt = decode_bits(&definition.dt)?;
                                    let fixed_refractory = definition
                                        .refractory
                                        .as_ref()
                                        .is_some_and(|refractory| refractory.mode == "fixed");
                                    if fixed_refractory {
                                        if let (Some(refractory), Some(instance)) = (
                                            &mut runtime.refractory,
                                            &model.instance.populations[node.owner_index]
                                                .refractory,
                                        ) {
                                            for neuron in 0..definition.count {
                                                refractory.not_refractory[neuron] = timestep(
                                                    time - refractory.lastspike[neuron],
                                                    dt,
                                                )? >= instance
                                                    .period_ticks;
                                            }
                                        }
                                    }
                                    if let Some(adaptive) = runtime.adaptive_update.as_mut() {
                                        adaptive.run(
                                            &mut runtime.states,
                                            runtime.refractory.as_ref(),
                                            definition.count,
                                            time,
                                            dt,
                                            clock_ticks[node.clock],
                                        )?;
                                    } else {
                                        run_population_update(
                                            runtime
                                                .update
                                                .as_mut()
                                                .ok_or("missing population state program")?,
                                            &mut runtime.states,
                                            &mut runtime.refractory,
                                            definition.count,
                                            time,
                                            clock_ticks[node.clock],
                                        )?;
                                    }
                                }
                                "spatial_state_update" => {
                                    let dt = decode_bits(&definition.dt)?;
                                    runtime
                                        .spatial
                                        .as_mut()
                                        .ok_or("missing spatial population runtime")?
                                        .run(
                                            &mut runtime.states,
                                            runtime.refractory.as_ref(),
                                            definition.count,
                                            dt,
                                            time,
                                            clock_ticks[node.clock],
                                        )?;
                                }
                                "threshold" => {
                                    let position = definition.code_objects[..node.item_index]
                                        .iter()
                                        .filter(|candidate| candidate.kind == "threshold")
                                        .count();
                                    let event = code
                                        .event_name
                                        .as_deref()
                                        .ok_or("scheduled threshold has no event")?;
                                    let event_index = definition
                                        .events
                                        .iter()
                                        .position(|candidate| candidate == event)
                                        .ok_or("scheduled threshold event is unknown")?;
                                    emit_threshold(
                                        runtime,
                                        definition,
                                        position,
                                        event_index,
                                        event == "spike",
                                        time,
                                    )?;
                                }
                                "poisson_input" => {
                                    let position = definition.code_objects[..node.item_index]
                                        .iter()
                                        .filter(|candidate| candidate.kind == "poisson_input")
                                        .count();
                                    run_program_neurons(
                                        runtime
                                            .poisson_inputs
                                            .get_mut(position)
                                            .ok_or("missing PoissonInput runtime program")?,
                                        &mut runtime.states,
                                        &[],
                                        runtime.refractory.as_ref(),
                                        definition.count,
                                        time,
                                        clock_ticks[node.clock],
                                    )?;
                                }
                                "run_regularly" => {
                                    let position = definition.code_objects[..node.item_index]
                                        .iter()
                                        .filter(|candidate| candidate.kind == "run_regularly")
                                        .count();
                                    run_program_neurons(
                                        runtime
                                            .regular
                                            .get_mut(position)
                                            .ok_or("missing run_regularly runtime program")?,
                                        &mut runtime.states,
                                        &[],
                                        runtime.refractory.as_ref(),
                                        definition.count,
                                        time,
                                        clock_ticks[node.clock],
                                    )?;
                                }
                                "reset" => {
                                    let position = definition.code_objects[..node.item_index]
                                        .iter()
                                        .filter(|candidate| candidate.kind == "reset")
                                        .count();
                                    let reset = runtime
                                        .resets
                                        .get_mut(position)
                                        .ok_or("missing population reset program")?;
                                    let event = code
                                        .event_name
                                        .as_deref()
                                        .ok_or("scheduled reset has no event")?;
                                    let event_index = definition
                                        .events
                                        .iter()
                                        .position(|candidate| candidate == event)
                                        .ok_or("scheduled reset event is unknown")?;
                                    reset.prepare(time, clock_ticks[node.clock])?;
                                    for &neuron in &runtime.fired[event_index] {
                                        reset.run(
                                            &runtime.states,
                                            &runtime.states,
                                            &runtime.states,
                                            &[],
                                            runtime.refractory.as_ref(),
                                            Batch::neurons(neuron, 1),
                                        )?;
                                        reset.commit_neurons(&mut runtime.states, neuron, 1);
                                    }
                                }
                                _ => {
                                    return Err("unsupported scheduled population operation".into())
                                }
                            }
                        }
                        (ScheduleOperation::CodeObject, ScheduleOwnerKind::Synapse) => {
                            let definition = &d.synapses[node.owner_index];
                            let code = &definition.code_objects[node.item_index];
                            if code.kind == "summed_variable"
                                && (definition.source_synapse.is_some()
                                    || definition.target_synapse.is_some())
                            {
                                let position = definition.code_objects[..node.item_index]
                                    .iter()
                                    .filter(|candidate| candidate.kind == "summed_variable")
                                    .count();
                                run_edge_endpoint_summed(
                                    synapse_runtimes,
                                    node.owner_index,
                                    position,
                                    populations,
                                    definition,
                                    time,
                                )?;
                            } else if matches!(code.kind.as_str(), "synapses" | "synapses_post")
                                && definition.target_synapse.is_some()
                            {
                                let pathway_name = code
                                    .pathway_name
                                    .as_deref()
                                    .ok_or("scheduled pathway has no name")?;
                                let position = model.instance.synapses[node.owner_index]
                                    .pathways
                                    .iter()
                                    .position(|pathway| pathway.name == pathway_name)
                                    .ok_or("scheduled pathway has no runtime instance")?;
                                run_edge_target_pathway(
                                    synapse_runtimes,
                                    node.owner_index,
                                    definition.target_synapse.unwrap(),
                                    position,
                                    populations,
                                    definition,
                                    time,
                                )?;
                            } else {
                                let runtime = &mut synapse_runtimes[node.owner_index];
                                match code.kind.as_str() {
                                    "synapse_subexpression_update" => run_synapse_program(
                                        runtime
                                            .subexpression_update
                                            .as_mut()
                                            .ok_or("missing synapse subexpression program")?,
                                        &populations,
                                        definition,
                                        SynapseArrays {
                                            source: &runtime.source,
                                            target: &runtime.target,
                                            states: &mut runtime.states,
                                        },
                                        time,
                                        clock_ticks[node.clock],
                                    )?,
                                    "synapse_state_update" => run_synapse_program(
                                        runtime
                                            .update
                                            .as_mut()
                                            .ok_or("missing synapse state program")?,
                                        &populations,
                                        definition,
                                        SynapseArrays {
                                            source: &runtime.source,
                                            target: &runtime.target,
                                            states: &mut runtime.states,
                                        },
                                        time,
                                        clock_ticks[node.clock],
                                    )?,
                                    "synapse_run_regularly" => {
                                        let position = definition.code_objects[..node.item_index]
                                            .iter()
                                            .filter(|candidate| {
                                                candidate.kind == "synapse_run_regularly"
                                            })
                                            .count();
                                        run_synapse_program(
                                            runtime
                                                .regular
                                                .get_mut(position)
                                                .ok_or("missing synapse run_regularly program")?,
                                            &populations,
                                            definition,
                                            SynapseArrays {
                                                source: &runtime.source,
                                                target: &runtime.target,
                                                states: &mut runtime.states,
                                            },
                                            time,
                                            clock_ticks[node.clock],
                                        )?;
                                    }
                                    "summed_variable" => {
                                        let position = definition.code_objects[..node.item_index]
                                            .iter()
                                            .filter(|candidate| candidate.kind == "summed_variable")
                                            .count();
                                        run_summed_program(
                                            runtime
                                                .summed
                                                .get_mut(position)
                                                .ok_or("missing summed-variable runtime program")?,
                                            populations,
                                            definition,
                                            &runtime.source,
                                            &runtime.target,
                                            &runtime.states,
                                            time,
                                        )?;
                                    }
                                    "synapses" | "synapses_post" => {
                                        let pathway_name = code
                                            .pathway_name
                                            .as_deref()
                                            .ok_or("scheduled pathway has no name")?;
                                        let position = model.instance.synapses[node.owner_index]
                                            .pathways
                                            .iter()
                                            .position(|pathway| pathway.name == pathway_name)
                                            .ok_or("scheduled pathway has no runtime instance")?;
                                        let SynapseRuntime {
                                            source,
                                            target,
                                            states,
                                            pathways,
                                            ..
                                        } = runtime;
                                        run_pathway(
                                            &mut pathways[position],
                                            populations,
                                            definition,
                                            source,
                                            target,
                                            states,
                                            time,
                                        )?;
                                    }
                                    _ => {
                                        return Err("unsupported scheduled synapse operation".into())
                                    }
                                }
                            }
                        }
                        _ => return Err("invalid scheduled operation/owner combination".into()),
                    }
                }
                for (index, runtime) in populations.iter_mut().enumerate() {
                    if population_active[index] {
                        if let Some(spike) = d.populations[index]
                            .events
                            .iter()
                            .position(|event| event == "spike")
                        {
                            runtime.last_fired.clone_from(&runtime.fired[spike]);
                        } else {
                            runtime.last_fired.clear();
                        }
                        runtime.tick = clock_ticks[d.populations[index].clock] + 1;
                    }
                }
                for (clock, active) in clock_ticks.iter_mut().zip(&active_clocks) {
                    *clock += usize::from(*active);
                }
                continue;
            }

            for index in 0..populations.len() {
                if !active[index] {
                    continue;
                }
                let def = &d.populations[index];
                let runtime = &mut populations[index];
                if let Some(update) = &mut runtime.subexpression_update {
                    run_program_neurons(
                        update,
                        &mut runtime.states,
                        &[],
                        runtime.refractory.as_ref(),
                        def.count,
                        time,
                        runtime.tick,
                    )?;
                }
            }

            // Brian schedules stored ``constant over dt`` synaptic
            // subexpressions in before_start, ahead of monitors and all other
            // phases on the source clock.
            for (syn_def, syn) in d.synapses.iter().zip(synapse_runtimes.iter_mut()) {
                if !active[syn_def.source_population] {
                    continue;
                }
                if let Some(update) = &mut syn.subexpression_update {
                    update.prepare(time, populations[syn_def.source_population].tick)?;
                    for edge in 0..syn.source.len() {
                        let source = syn.source[edge];
                        let target = syn.target[edge];
                        let source_states = &populations[syn_def.source_population].states;
                        let target_pop = &populations[syn_def.target_population];
                        update.run(
                            &target_pop.states,
                            source_states,
                            &target_pop.states,
                            &syn.states,
                            target_pop.refractory.as_ref(),
                            Batch {
                                start: 0,
                                len: 1,
                                source,
                                target,
                                source_state: source + syn_def.source_start,
                                target_state: target + syn_def.target_start,
                                edge,
                            },
                        )?;
                        update.commit_synapses(&mut syn.states, edge, 1);
                    }
                }
            }

            for index in 0..populations.len() {
                if !active[index] {
                    continue;
                }
                let def = &d.populations[index];
                let runtime = &mut populations[index];
                if runtime.tick < runtime.end_tick - def.monitor.window_steps {
                    continue;
                }
                for &neuron in &def.monitor.record {
                    for (sample, value) in runtime.samples.iter_mut().zip(&runtime.monitor_values) {
                        sample.push_register(match value {
                            MonitorValue::State(state) => {
                                runtime.states[*state].get_register(neuron)
                            }
                            MonitorValue::Scalar(value) => *value,
                            MonitorValue::Parameter(values) => values[neuron],
                            MonitorValue::Linked(input) => input.linked_value(neuron)?,
                        });
                    }
                }
            }
            // Brian2 schedules summed-variable updaters in the groups slot at
            // order=-1, before neuron and synapse state updaters at order=0.
            // Each updater first clears exactly its endpoint (including subgroup
            // bounds), then accumulates in connection-creation order.
            for (syn_def, syn) in d.synapses.iter().zip(synapse_runtimes.iter_mut()) {
                for summed in &mut syn.summed {
                    if !summed.before_state || !active[summed.target_population] {
                        continue;
                    }
                    populations[summed.target_population].states[summed.target_state].fill_f64(
                        summed.target_start..summed.target_start + summed.target_count,
                        0.0,
                    );
                    let tick = populations[summed.target_population].tick;
                    summed.program.prepare(time, tick)?;
                    for edge in 0..syn.source.len() {
                        let source = syn.source[edge];
                        let target = syn.target[edge];
                        let source_state = source + syn_def.source_start;
                        let target_state = target + syn_def.target_start;
                        {
                            let source_population = &populations[syn_def.source_population];
                            let target_population = &populations[syn_def.target_population];
                            summed.program.run(
                                &target_population.states,
                                &source_population.states,
                                &target_population.states,
                                &syn.states,
                                target_population.refractory.as_ref(),
                                Batch {
                                    start: 0,
                                    len: 1,
                                    source,
                                    target,
                                    source_state,
                                    target_state,
                                    edge,
                                },
                            )?;
                        }
                        let destination = if summed.target_is_pre {
                            source_state
                        } else {
                            target_state
                        };
                        populations[summed.target_population].states[summed.target_state]
                            .add_f64(destination, summed.program.registers[summed.result][0]);
                    }
                }
            }
            for index in 0..populations.len() {
                if !active[index] {
                    continue;
                }
                let def = &d.populations[index];
                let dt = decode_bits(&def.dt)?;
                let runtime = &mut populations[index];
                if let (Some(refractory), Some(instance)) = (
                    &mut runtime.refractory,
                    &model.instance.populations[index].refractory,
                ) {
                    for neuron in 0..def.count {
                        refractory.not_refractory[neuron] =
                            timestep(time - refractory.lastspike[neuron], dt)?
                                >= instance.period_ticks;
                    }
                }
                if let Some(update) = &mut runtime.update {
                    run_program_neurons(
                        update,
                        &mut runtime.states,
                        &[],
                        runtime.refractory.as_ref(),
                        def.count,
                        time,
                        runtime.tick,
                    )?;
                } else if let Some(adaptive) = &mut runtime.adaptive_update {
                    adaptive.run(
                        &mut runtime.states,
                        runtime.refractory.as_ref(),
                        def.count,
                        time,
                        dt,
                        runtime.tick,
                    )?;
                }
            }
            for (syn_def, syn) in d.synapses.iter().zip(synapse_runtimes.iter_mut()) {
                if active[syn_def.source_population] {
                    if let Some(update) = &mut syn.update {
                        update.prepare(time, populations[syn_def.source_population].tick)?;
                        for edge in 0..syn.source.len() {
                            let source = syn.source[edge];
                            let target = syn.target[edge];
                            let source_states = &populations[syn_def.source_population].states;
                            let target_pop = &populations[syn_def.target_population];
                            update.run(
                                &target_pop.states,
                                source_states,
                                &target_pop.states,
                                &syn.states,
                                target_pop.refractory.as_ref(),
                                Batch {
                                    start: 0,
                                    len: 1,
                                    source,
                                    target,
                                    source_state: source + syn_def.source_start,
                                    target_state: target + syn_def.target_start,
                                    edge,
                                },
                            )?;
                            update.commit_synapses(&mut syn.states, edge, 1);
                        }
                    }
                }
            }
            // A subgroup has order=parent.order+1. Its default summed updater can
            // therefore sort after the parent's state updater in the same groups
            // slot; preserve Brian's (order, name) ordering instead of forcing all
            // summed variables into the earlier phase.
            for (syn_def, syn) in d.synapses.iter().zip(synapse_runtimes.iter_mut()) {
                for summed in &mut syn.summed {
                    if summed.before_state || !active[summed.target_population] {
                        continue;
                    }
                    populations[summed.target_population].states[summed.target_state].fill_f64(
                        summed.target_start..summed.target_start + summed.target_count,
                        0.0,
                    );
                    let tick = populations[summed.target_population].tick;
                    summed.program.prepare(time, tick)?;
                    for edge in 0..syn.source.len() {
                        let source = syn.source[edge];
                        let target = syn.target[edge];
                        let source_state = source + syn_def.source_start;
                        let target_state = target + syn_def.target_start;
                        {
                            let source_population = &populations[syn_def.source_population];
                            let target_population = &populations[syn_def.target_population];
                            summed.program.run(
                                &target_population.states,
                                &source_population.states,
                                &target_population.states,
                                &syn.states,
                                target_population.refractory.as_ref(),
                                Batch {
                                    start: 0,
                                    len: 1,
                                    source,
                                    target,
                                    source_state,
                                    target_state,
                                    edge,
                                },
                            )?;
                        }
                        let destination = if summed.target_is_pre {
                            source_state
                        } else {
                            target_state
                        };
                        populations[summed.target_population].states[summed.target_state]
                            .add_f64(destination, summed.program.registers[summed.result][0]);
                    }
                }
            }
            for index in 0..populations.len() {
                if !active[index] {
                    continue;
                }
                let def = &d.populations[index];
                let runtime = &mut populations[index];
                runtime.fired[0].clear();
                if let Some(threshold) = runtime.thresholds.first_mut() {
                    threshold.prepare(time, runtime.tick)?;
                    let condition = threshold.condition.ok_or("missing threshold condition")?;
                    for start in (0..def.count).step_by(LANES) {
                        let len = LANES.min(def.count - start);
                        threshold.run(
                            &runtime.states,
                            &runtime.states,
                            &runtime.states,
                            &[],
                            runtime.refractory.as_ref(),
                            Batch::neurons(start, len),
                        )?;
                        for lane in 0..len {
                            let neuron = start + lane;
                            if threshold.registers[condition][lane] != 0.0
                                && runtime
                                    .refractory
                                    .as_ref()
                                    .is_none_or(|r| r.not_refractory[neuron])
                            {
                                runtime.fired[0].push(neuron);
                                if runtime.tick >= runtime.end_tick - def.monitor.window_steps {
                                    runtime.counts[neuron] += 1;
                                    runtime.spikes.push((runtime.tick, neuron));
                                }
                                if let Some(refractory) = &mut runtime.refractory {
                                    refractory.lastspike[neuron] = time;
                                    refractory.not_refractory[neuron] = false;
                                }
                            }
                        }
                    }
                } else if !runtime.spike_schedule.is_empty() {
                    while runtime.spike_cursor < runtime.spike_schedule.len()
                        && runtime.spike_schedule[runtime.spike_cursor].0 < runtime.tick
                    {
                        runtime.spike_cursor += 1;
                    }
                    while runtime.spike_cursor < runtime.spike_schedule.len()
                        && runtime.spike_schedule[runtime.spike_cursor].0 == runtime.tick
                    {
                        let neuron = runtime.spike_schedule[runtime.spike_cursor].1;
                        runtime.fired[0].push(neuron);
                        if runtime.tick >= runtime.end_tick - def.monitor.window_steps {
                            runtime.counts[neuron] += 1;
                            runtime.spikes.push((runtime.tick, neuron));
                        }
                        runtime.spike_cursor += 1;
                    }
                }
                runtime.last_fired.clone_from(&runtime.fired[0]);
            }
            for (syn_def, syn) in d.synapses.iter().zip(synapse_runtimes.iter_mut()) {
                let SynapseRuntime {
                    source,
                    target,
                    states,
                    pathways,
                    ..
                } = syn;
                for pathway in pathways.iter_mut() {
                    if pathway.is_pre && active[syn_def.source_population] {
                        run_pathway(pathway, populations, syn_def, source, target, states, time)?;
                    }
                }
            }
            for index in 0..populations.len() {
                if !active[index] {
                    continue;
                }
                let def = &d.populations[index];
                let runtime = &mut populations[index];
                for poisson_input in &mut runtime.poisson_inputs {
                    run_program_neurons(
                        poisson_input,
                        &mut runtime.states,
                        &[],
                        runtime.refractory.as_ref(),
                        def.count,
                        time,
                        runtime.tick,
                    )?;
                }
            }
            for (definition, synapse) in d.synapses.iter().zip(synapse_runtimes.iter_mut()) {
                let SynapseRuntime {
                    source,
                    target,
                    states,
                    pathways,
                    ..
                } = synapse;
                for pathway in pathways.iter_mut() {
                    if !pathway.is_pre && active[definition.target_population] {
                        run_pathway(
                            pathway,
                            populations,
                            definition,
                            source,
                            target,
                            states,
                            time,
                        )?;
                    }
                }
            }
            for index in 0..populations.len() {
                if !active[index] {
                    continue;
                }
                let runtime = &mut populations[index];
                if let Some(reset) = runtime.resets.first_mut() {
                    reset.prepare(time, runtime.tick)?;
                    for &neuron in &runtime.fired[0] {
                        reset.run(
                            &runtime.states,
                            &runtime.states,
                            &runtime.states,
                            &[],
                            runtime.refractory.as_ref(),
                            Batch::neurons(neuron, 1),
                        )?;
                        reset.commit_neurons(&mut runtime.states, neuron, 1);
                    }
                }
                runtime.tick += 1;
            }
            for (clock, active) in clock_ticks.iter_mut().zip(&active_clocks) {
                *clock += usize::from(*active);
            }
        }
        Ok(self.is_finished())
    }

    pub(super) fn write_results(
        &self,
        results: impl Write,
        events: impl Write,
    ) -> Result<serde_json::Value> {
        check(
            !self.failed && self.is_finished(),
            "results require a completed execution",
        )?;
        let model = &self.model;
        let populations = &self.populations;
        let synapse_runtimes = &self.synapse_runtimes;
        let dump_bytes = write_dump(model, populations, synapse_runtimes, results)?;
        let event_dump_bytes = write_event_monitor_dump(&model.definition, populations, events)?;
        Ok(serde_json::json!({
            "schema": "b2-result-dump-v3", "dump_bytes": dump_bytes,
            "event_dump_bytes": event_dump_bytes,
            "population_count": populations.len(), "neuron_count": model.instance.neuron_count,
            "spike_count": populations.iter().map(|p| p.spikes.len()).sum::<usize>(),
            "final_time_seconds": decode_bits(&model.run.start)? + decode_bits(&model.run.duration)?,
            "synaptic_events": synapse_runtimes.iter().flat_map(|s| &s.pathways).map(|p| p.topology.delivered).sum::<usize>(),
        }))
    }
}

pub(super) fn execute(mut model: Model, directory: &Path) -> Result<()> {
    model.validate()?;
    model.complete_execution_effects()?;
    let dispatch = (0..model.definition.schedule.nodes.len()).collect();
    let mut runtime = Runtime::new(model, dispatch)?;
    fs::create_dir_all(directory)?;
    let started = Instant::now();
    runtime.step(usize::MAX)?;
    let simulation_seconds = started.elapsed().as_secs_f64();
    let output_started = Instant::now();
    let results = BufWriter::with_capacity(65_536, File::create(directory.join("results.bin"))?);
    // Preserve the native convention: no events.bin when no event streams exist.
    let has_events = runtime
        .model
        .definition
        .populations
        .iter()
        .any(|p| !p.events.is_empty() || !p.event_monitors.is_empty());
    let events: Box<dyn Write> = if has_events {
        Box::new(BufWriter::with_capacity(
            65_536,
            File::create(directory.join("events.bin"))?,
        ))
    } else {
        Box::new(std::io::sink())
    };
    let mut summary = runtime.write_results(results, events)?;
    check(
        File::open(directory.join("results.bin"))?.metadata()?.len()
            == summary["dump_bytes"].as_u64().unwrap(),
        "result dump size mismatch",
    )?;
    if has_events {
        check(
            File::open(directory.join("events.bin"))?.metadata()?.len()
                == summary["event_dump_bytes"].as_u64().unwrap(),
            "event monitor dump size mismatch",
        )?;
    }
    summary["timings"] = serde_json::json!({"simulation_and_recording_seconds": simulation_seconds,
        "dump_write_seconds": output_started.elapsed().as_secs_f64()});
    fs::write(
        directory.join("summary.json"),
        serde_json::to_string_pretty(&summary)? + "\n",
    )?;
    println!(
        "completed {} populations for {} neurons",
        summary["population_count"], summary["neuron_count"]
    );
    Ok(())
}

#[derive(Clone)]
enum Input {
    Constant(f64, DType),
    Time,
    Index(Domain),
    State(usize, Domain, DType),
    PreState(usize, DType),
    PostState(usize, DType),
    SynapseState(usize, DType),
    Linked {
        source_population: usize,
        source_state: usize,
        source_pointer: usize,
        index: LinkedInputIndex,
        dtype: DType,
    },
    Parameter(Vec<f64>, Domain, DType),
    LastSpike,
    Available(Domain),
    Random(u64, Domain),
    Normal(u64, Domain),
}

#[derive(Clone)]
enum LinkedInputIndex {
    Identity,
    Constant(Arc<[usize]>),
    State {
        population: usize,
        state: usize,
        pointer: usize,
    },
}

#[derive(Clone, Copy)]
enum WriteTarget {
    Neuron(usize),
    PreNeuron(usize),
    Synapse(usize),
    Available,
}

impl Input {
    fn scalar(&self) -> bool {
        matches!(self, Self::Constant(..) | Self::Time)
    }

    fn dtype(&self) -> DType {
        match self {
            Self::Constant(_, dtype)
            | Self::State(_, _, dtype)
            | Self::PreState(_, dtype)
            | Self::PostState(_, dtype)
            | Self::SynapseState(_, dtype)
            | Self::Parameter(_, _, dtype) => *dtype,
            Self::Linked { dtype, .. } => *dtype,
            Self::Time | Self::LastSpike | Self::Random(..) | Self::Normal(..) => DType::F64,
            Self::Index(_) => DType::Index,
            Self::Available(_) => DType::Bool,
        }
    }

    fn bind_link(&mut self, state_pointers: &[Vec<usize>]) {
        if let Self::Linked {
            source_population,
            source_state,
            source_pointer,
            index,
            ..
        } = self
        {
            *source_pointer = state_pointers[*source_population][*source_state];
            if let LinkedInputIndex::State {
                population,
                state,
                pointer,
            } = index
            {
                *pointer = state_pointers[*population][*state];
            }
        }
    }

    fn linked_value(&self, neuron: usize) -> Result<f64> {
        let Self::Linked {
            source_pointer,
            index,
            ..
        } = self
        else {
            return Err("monitor value is not linked".into());
        };
        check(*source_pointer != 0, "unbound linked source")?;
        let source_index = match index {
            LinkedInputIndex::Identity => neuron,
            LinkedInputIndex::Constant(values) => values[neuron],
            LinkedInputIndex::State { pointer, .. } => {
                check(*pointer != 0, "unbound linked index state")?;
                // SAFETY: link binding records addresses of fixed StateArray
                // entries after every PopulationRuntime has been allocated.
                unsafe { &*(*pointer as *const StateArray) }.get_index(neuron)?
            }
        };
        // SAFETY: the source StateArray is fixed for the lifetime of execution;
        // schedule dependencies prevent a simultaneous writer/read pair.
        let source = unsafe { &*(*source_pointer as *const StateArray) };
        check(
            source_index < source.len(),
            "linked variable index out of bounds",
        )?;
        Ok(source.get_register(source_index))
    }
}

#[derive(Clone, Copy)]
enum IntegerUnary {
    Neg,
    Abs,
}

#[derive(Clone, Copy)]
enum IntegerBinary {
    Add,
    Sub,
    Mul,
    Mod,
    FloorDiv,
    Gt,
    Ge,
    Lt,
    Le,
}

#[derive(Clone)]
enum Op {
    Neg,
    Not,
    Abs,
    Arccos,
    Arcsin,
    Arctan,
    Ceil,
    Cos,
    Cosh,
    Exp,
    Expm1,
    Exprel,
    Floor,
    Trunc,
    RoundF32,
    Cast(DType, DType),
    IntegerUnary(DType, IntegerUnary),
    IntegerBinary(DType, IntegerBinary),
    Timestep,
    TickOffset(i64),
    Log,
    Log10,
    Log1p,
    Sign,
    Sin,
    Sinh,
    Sqrt,
    Tan,
    Tanh,
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Mod,
    FloorDiv,
    Eq,
    Ne,
    Gt,
    Ge,
    Lt,
    Le,
    And,
    Or,
    Clip,
    Select,
    Binomial(u64, u64, bool, Domain),
    Poisson(u64, Domain),
    TimedArray {
        values: Vec<f64>,
        rows: usize,
        columns: Option<usize>,
        epsilon: f64,
        upsampling: usize,
    },
}

struct Instruction {
    op: Op,
    left: usize,
    right: usize,
    third: Option<usize>,
    guard: Option<usize>,
    output: usize,
}

struct Program {
    registers: Vec<Vec<f64>>,
    symbols: BTreeMap<String, usize>,
    symbol_dtypes: BTreeMap<String, DType>,
    inputs: BTreeMap<String, Input>,
    bindings: Vec<(usize, Input)>,
    scalar_random_bindings: BTreeSet<usize>,
    broadcast: Vec<usize>,
    scalar: Vec<Instruction>,
    vector: Vec<Instruction>,
    writes: Vec<(WriteTarget, usize)>, // state array, final SSA register
    condition: Option<usize>,
    rng_seed: u64,
    tick: u64,
    rng_domain: Domain,
    functions: BTreeMap<String, FunctionDefinition>,
}

impl Program {
    fn compile(
        code: &CodeObjectSpec,
        inputs: BTreeMap<String, Input>,
        writable: &BTreeMap<String, WriteTarget>,
        rng_seed: u64,
        functions: &[FunctionDefinition],
    ) -> Result<Self> {
        let symbol_dtypes = inputs
            .iter()
            .map(|(name, input)| (name.clone(), input.dtype()))
            .collect();
        let mut plan = Self {
            registers: Vec::new(),
            symbols: BTreeMap::new(),
            symbol_dtypes,
            inputs,
            bindings: Vec::new(),
            scalar_random_bindings: BTreeSet::new(),
            broadcast: Vec::new(),
            scalar: Vec::new(),
            vector: Vec::new(),
            writes: Vec::new(),
            condition: None,
            rng_seed,
            tick: 0,
            rng_domain: match code.iteration_domain.as_str() {
                "all_synapses" | "active_synapses" => Domain::Edge,
                _ => Domain::Neuron,
            },
            functions: functions
                .iter()
                .map(|function| (function.name.clone(), function.clone()))
                .collect(),
        };
        plan.scalar = plan.statements(&code.scalar)?;
        plan.scalar_random_bindings = plan
            .bindings
            .iter()
            .filter(|(_, input)| matches!(input, Input::Random(..) | Input::Normal(..)))
            .map(|(slot, _)| *slot)
            .collect();
        // Scalar temporaries are evaluated once per phase and broadcast to the
        // tile. A vector assignment gets a new register, preserving old values.
        plan.broadcast.extend(0..plan.registers.len());
        plan.vector = plan.statements(&code.vector)?;
        plan.broadcast.extend(
            plan.bindings
                .iter()
                .filter(|(_, input)| input.scalar())
                .map(|(slot, _)| *slot),
        );
        plan.broadcast.sort_unstable();
        plan.broadcast.dedup();
        plan.writes = if code.adaptive.is_some() {
            Vec::new()
        } else {
            code.effects
                .writes
                .iter()
                .map(|name| (writable[name], plan.symbols[name]))
                .collect()
        };
        if code.kind == "threshold" {
            plan.condition = Some(plan.symbols["_cond"]);
        }
        Ok(plan)
    }

    fn bind_links(&mut self, state_pointers: &[Vec<usize>]) {
        for input in self.inputs.values_mut() {
            input.bind_link(state_pointers);
        }
        for (_, input) in &mut self.bindings {
            input.bind_link(state_pointers);
        }
    }

    fn register(&mut self) -> Result<usize> {
        // Bound scratch storage independently of the serialized IR budget.
        budget(self.registers.len() + 1, LANES, 4_000_000)?;
        let slot = self.registers.len();
        self.registers.push(vec![0.0; LANES]);
        Ok(slot)
    }

    fn input(&mut self, value: Input) -> Result<usize> {
        let slot = self.register()?;
        self.bindings.push((slot, value));
        Ok(slot)
    }

    fn resolve(&mut self, name: &str) -> Result<usize> {
        if let Some(&slot) = self.symbols.get(name) {
            return Ok(slot);
        }
        let input = self.inputs.get(name).ok_or("missing plan input")?.clone();
        let slot = self.input(input)?;
        self.symbols.insert(name.to_owned(), slot);
        Ok(slot)
    }

    fn expression_dtype(&self, expr: &Expr) -> Result<DType> {
        expr.infer(&self.symbol_dtypes, &mut BTreeSet::new(), &self.functions)
    }

    fn expression(
        &mut self,
        expr: &Expr,
        guard: Option<usize>,
        ops: &mut Vec<Instruction>,
    ) -> Result<usize> {
        use Expr::*;
        let (op, left, right, third) = match expr {
            Literal { bits } => {
                return self.input(Input::Constant(decode_bits(bits)?, DType::F64));
            }
            Boolean { value } => {
                return self.input(Input::Constant(f64::from(*value), DType::Bool));
            }
            Integer { dtype, value } => {
                validate_integer_literal(*dtype, value)?;
                let raw = match dtype {
                    DType::I32 => value.parse::<i32>()? as u32 as u64,
                    DType::I64 => value.parse::<i64>()? as u64,
                    DType::U32 => value.parse::<u32>()? as u64,
                    DType::U64 => value.parse::<u64>()?,
                    _ => return Err("invalid integer literal dtype".into()),
                };
                return self.input(Input::Constant(f64::from_bits(raw), *dtype));
            }
            Cast { dtype, arg } => {
                let source = self.expression_dtype(arg)?;
                let slot = self.expression(arg, guard, ops)?;
                (Op::Cast(source, *dtype), slot, slot, None)
            }
            Rand { stream } => {
                return self.input(Input::Random(*stream, self.rng_domain));
            }
            Randn { stream } => {
                return self.input(Input::Normal(*stream, self.rng_domain));
            }
            Binomial {
                stream,
                n,
                p,
                approximate,
            } => {
                let slot = self.expression(p, guard, ops)?;
                (
                    Op::Binomial(*stream, *n, *approximate, self.rng_domain),
                    slot,
                    slot,
                    None,
                )
            }
            Poisson { stream, lambda } => {
                let slot = self.expression(lambda, guard, ops)?;
                (Op::Poisson(*stream, self.rng_domain), slot, slot, None)
            }
            TimedArray {
                values,
                rows,
                columns,
                epsilon,
                upsampling,
                time,
                index,
                ..
            } => {
                let time_slot = self.expression(time, guard, ops)?;
                let index_slot = index
                    .as_ref()
                    .map(|value| self.expression(value, guard, ops))
                    .transpose()?
                    .unwrap_or(time_slot);
                (
                    Op::TimedArray {
                        values: values
                            .iter()
                            .map(|value| decode_bits(value))
                            .collect::<Result<_>>()?,
                        rows: *rows,
                        columns: *columns,
                        epsilon: decode_bits(epsilon)?,
                        upsampling: *upsampling,
                    },
                    time_slot,
                    index_slot,
                    None,
                )
            }
            Load { name } => return self.resolve(name),
            Call {
                function,
                arguments,
            } => {
                let contract = self
                    .functions
                    .get(function)
                    .cloned()
                    .ok_or("missing Function contract in execution plan")?;
                let slots = arguments
                    .iter()
                    .map(|argument| self.expression(argument, guard, ops))
                    .collect::<Result<Vec<_>>>()?;
                let previous: Vec<_> = contract
                    .arguments
                    .iter()
                    .zip(slots)
                    .map(|(argument, slot)| {
                        (
                            argument.name.clone(),
                            self.symbols.insert(argument.name.clone(), slot),
                        )
                    })
                    .collect();
                let previous_dtypes: Vec<_> = contract
                    .arguments
                    .iter()
                    .map(|argument| {
                        (
                            argument.name.clone(),
                            self.symbol_dtypes
                                .insert(argument.name.clone(), argument.dtype),
                        )
                    })
                    .collect();
                let result = match contract.body.as_ref() {
                    Some(body) => self.expression(body, guard, ops),
                    None => Err("reference executor cannot execute a native-only Function".into()),
                };
                for (name, value) in previous {
                    if let Some(slot) = value {
                        self.symbols.insert(name, slot);
                    } else {
                        self.symbols.remove(&name);
                    }
                }
                for (name, value) in previous_dtypes {
                    if let Some(dtype) = value {
                        self.symbol_dtypes.insert(name, dtype);
                    } else {
                        self.symbol_dtypes.remove(&name);
                    }
                }
                return result;
            }
            // Validated boolean/logical registers have exact f64
            // representations in the reference executor. Conversion nodes
            // remain explicit in B2IR and compile away only after validation.
            BoolToF64 { arg } | IndexToF64 { arg } | TickToF64 { arg } | F32ToF64 { arg } => {
                return self.expression(arg, guard, ops);
            }
            F64ToF32 { arg } => {
                let slot = self.expression(arg, guard, ops)?;
                (Op::RoundF32, slot, slot, None)
            }
            Timestep { time, dt } => (
                Op::Timestep,
                self.expression(time, guard, ops)?,
                self.expression(dt, guard, ops)?,
                None,
            ),
            TickOffset { tick, offset } => {
                let slot = self.expression(tick, guard, ops)?;
                (Op::TickOffset(*offset), slot, slot, None)
            }
            Neg { arg }
            | Not { arg }
            | Abs { arg }
            | Arccos { arg }
            | Arcsin { arg }
            | Arctan { arg }
            | Ceil { arg }
            | Cos { arg }
            | Cosh { arg }
            | Exp { arg }
            | Expm1 { arg }
            | Exprel { arg }
            | Floor { arg }
            | Trunc { arg }
            | Log { arg }
            | Log10 { arg }
            | Log1p { arg }
            | Sign { arg }
            | Sin { arg }
            | Sinh { arg }
            | Sqrt { arg }
            | Tan { arg }
            | Tanh { arg } => {
                let slot = self.expression(arg, guard, ops)?;
                let dtype = self.expression_dtype(arg)?;
                let op = match expr {
                    Neg { .. } if dtype.is_integer() => Op::IntegerUnary(dtype, IntegerUnary::Neg),
                    Neg { .. } => Op::Neg,
                    Not { .. } => Op::Not,
                    Abs { .. } if dtype.is_integer() => Op::IntegerUnary(dtype, IntegerUnary::Abs),
                    Abs { .. } => Op::Abs,
                    Arccos { .. } => Op::Arccos,
                    Arcsin { .. } => Op::Arcsin,
                    Arctan { .. } => Op::Arctan,
                    Ceil { .. } => Op::Ceil,
                    Cos { .. } => Op::Cos,
                    Cosh { .. } => Op::Cosh,
                    Exp { .. } => Op::Exp,
                    Expm1 { .. } => Op::Expm1,
                    Exprel { .. } => Op::Exprel,
                    Floor { .. } => Op::Floor,
                    Trunc { .. } => Op::Trunc,
                    Log { .. } => Op::Log,
                    Log10 { .. } => Op::Log10,
                    Log1p { .. } => Op::Log1p,
                    Sign { .. } => Op::Sign,
                    Sin { .. } => Op::Sin,
                    Sinh { .. } => Op::Sinh,
                    Sqrt { .. } => Op::Sqrt,
                    Tan { .. } => Op::Tan,
                    Tanh { .. } => Op::Tanh,
                    _ => unreachable!(),
                };
                (op, slot, slot, None)
            }
            Add { left, right }
            | Sub { left, right }
            | Mul { left, right }
            | Div { left, right }
            | Pow { left, right }
            | Mod { left, right }
            | FloorDiv { left, right }
            | Eq { left, right }
            | Ne { left, right }
            | Gt { left, right }
            | Ge { left, right }
            | Lt { left, right }
            | Le { left, right }
            | And { left, right }
            | Or { left, right } => {
                let dtype = self.expression_dtype(left)?;
                let op = match expr {
                    Add { .. } if dtype.is_integer() => {
                        Op::IntegerBinary(dtype, IntegerBinary::Add)
                    }
                    Sub { .. } if dtype.is_integer() => {
                        Op::IntegerBinary(dtype, IntegerBinary::Sub)
                    }
                    Mul { .. } if dtype.is_integer() => {
                        Op::IntegerBinary(dtype, IntegerBinary::Mul)
                    }
                    Mod { .. } if dtype.is_integer() => {
                        Op::IntegerBinary(dtype, IntegerBinary::Mod)
                    }
                    FloorDiv { .. } if dtype.is_integer() => {
                        Op::IntegerBinary(dtype, IntegerBinary::FloorDiv)
                    }
                    Gt { .. } if dtype.is_integer() => Op::IntegerBinary(dtype, IntegerBinary::Gt),
                    Ge { .. } if dtype.is_integer() => Op::IntegerBinary(dtype, IntegerBinary::Ge),
                    Lt { .. } if dtype.is_integer() => Op::IntegerBinary(dtype, IntegerBinary::Lt),
                    Le { .. } if dtype.is_integer() => Op::IntegerBinary(dtype, IntegerBinary::Le),
                    Add { .. } => Op::Add,
                    Sub { .. } => Op::Sub,
                    Mul { .. } => Op::Mul,
                    Div { .. } => Op::Div,
                    Pow { .. } => Op::Pow,
                    Mod { .. } => Op::Mod,
                    FloorDiv { .. } => Op::FloorDiv,
                    Eq { .. } => Op::Eq,
                    Ne { .. } => Op::Ne,
                    Gt { .. } => Op::Gt,
                    Ge { .. } => Op::Ge,
                    Lt { .. } => Op::Lt,
                    Le { .. } => Op::Le,
                    And { .. } => Op::And,
                    Or { .. } => Op::Or,
                    _ => unreachable!(),
                };
                (
                    op,
                    self.expression(left, guard, ops)?,
                    self.expression(right, guard, ops)?,
                    None,
                )
            }
            Clip { value, min, max } => (
                Op::Clip,
                self.expression(value, guard, ops)?,
                self.expression(min, guard, ops)?,
                Some(self.expression(max, guard, ops)?),
            ),
        };
        let output = self.register()?;
        ops.push(Instruction {
            op,
            left,
            right,
            third,
            guard,
            output,
        });
        Ok(output)
    }

    fn statements(&mut self, statements: &[Statement]) -> Result<Vec<Instruction>> {
        let mut ops = Vec::new();
        for statement in statements {
            let guard = statement
                .condition
                .as_ref()
                .map(|name| self.resolve(name))
                .transpose()?;
            let old = if guard.is_some() {
                if let Some(&slot) = self.symbols.get(&statement.target) {
                    Some(slot)
                } else if self.inputs.contains_key(&statement.target) {
                    Some(self.resolve(&statement.target)?)
                } else {
                    None
                }
            } else {
                None
            };
            let value = self.expression(&statement.value, guard, &mut ops)?;
            let output = if let Some(old) = old {
                let output = self.register()?;
                ops.push(Instruction {
                    op: Op::Select,
                    left: value,
                    right: old,
                    third: None,
                    guard,
                    output,
                });
                output
            } else {
                value
            };
            self.symbols.insert(statement.target.clone(), output);
            self.symbol_dtypes
                .insert(statement.target.clone(), statement.dtype);
        }
        Ok(ops)
    }

    fn prepare(&mut self, time: f64, tick: usize) -> Result<()> {
        self.tick = tick as u64;
        for (slot, input) in &self.bindings {
            match input {
                Input::Constant(value, _) => self.registers[*slot][0] = *value,
                Input::Time => self.registers[*slot][0] = time,
                Input::Random(stream, _) if self.scalar_random_bindings.contains(slot) => {
                    self.registers[*slot][0] =
                        counter_uniform(self.rng_seed, *stream, self.tick, 0);
                }
                Input::Normal(stream, _) if self.scalar_random_bindings.contains(slot) => {
                    self.registers[*slot][0] = counter_normal(self.rng_seed, *stream, self.tick, 0);
                }
                _ => {}
            }
        }
        evaluate(
            &self.scalar,
            &mut self.registers,
            1,
            Some((self.rng_seed, self.tick, Batch::neurons(0, 1))),
        )?;
        for &slot in &self.broadcast {
            let value = self.registers[slot][0];
            self.registers[slot].fill(value);
        }
        Ok(())
    }

    fn run(
        &mut self,
        states: &[StateArray],
        pre_states: &[StateArray],
        post_states: &[StateArray],
        synapse_states: &[StateArray],
        refractory: Option<&RefractoryRuntime>,
        batch: Batch,
    ) -> Result<()> {
        for (slot, input) in &self.bindings {
            if self.scalar_random_bindings.contains(slot) {
                continue;
            }
            let output = &mut self.registers[*slot][..batch.len];
            match input {
                Input::State(state, domain, _) => {
                    copy_state_input(output, &states[*state], *domain, batch)
                }
                Input::PreState(state, _) => {
                    output.fill(pre_states[*state].get_register(batch.source_state));
                }
                Input::PostState(state, _) => {
                    output.fill(post_states[*state].get_register(batch.target_state));
                }
                Input::SynapseState(state, _) => {
                    copy_state_input(output, &synapse_states[*state], Domain::Edge, batch)
                }
                Input::Linked { .. } => {
                    for (lane, value) in output.iter_mut().enumerate() {
                        *value = input.linked_value(batch.start + lane)?;
                    }
                }
                Input::Parameter(values, domain, _) => copy_input(output, values, *domain, batch),
                Input::Index(domain) => {
                    for (lane, value) in output.iter_mut().enumerate() {
                        *value = batch.index(*domain, lane) as f64;
                    }
                }
                Input::LastSpike => copy_input(
                    output,
                    &refractory.ok_or("missing refractory")?.lastspike,
                    Domain::Neuron,
                    batch,
                ),
                Input::Available(domain) => {
                    let flags = &refractory.ok_or("missing refractory")?.not_refractory;
                    for (lane, value) in output.iter_mut().enumerate() {
                        let index = match domain {
                            Domain::Post => batch.target_state,
                            Domain::Pre => batch.source_state,
                            _ => batch.index(*domain, lane),
                        };
                        *value = f64::from(flags[index]);
                    }
                }
                Input::Random(stream, domain) => {
                    for (lane, value) in output.iter_mut().enumerate() {
                        let index = batch.index(*domain, lane) as u64;
                        *value = counter_uniform(self.rng_seed, *stream, self.tick, index);
                    }
                }
                Input::Normal(stream, domain) => {
                    for (lane, value) in output.iter_mut().enumerate() {
                        let index = batch.index(*domain, lane) as u64;
                        *value = counter_normal(self.rng_seed, *stream, self.tick, index);
                    }
                }
                Input::Constant(..) | Input::Time => {}
            }
        }
        evaluate(
            &self.vector,
            &mut self.registers,
            batch.len,
            Some((self.rng_seed, self.tick, batch)),
        )
    }

    fn commit_neurons(&self, states: &mut [StateArray], start: usize, len: usize) {
        for &(target, register) in &self.writes {
            if let WriteTarget::Neuron(state) = target {
                for lane in 0..len {
                    states[state].set_register(start + lane, self.registers[register][lane]);
                }
            }
        }
    }

    fn commit_available(&self, refractory: &mut RefractoryRuntime, start: usize, len: usize) {
        for &(target, register) in &self.writes {
            if let WriteTarget::Available = target {
                for lane in 0..len {
                    refractory.not_refractory[start + lane] = self.registers[register][lane] != 0.0;
                }
            }
        }
    }

    fn commit_synapses(&self, states: &mut [StateArray], edge: usize, len: usize) {
        for &(target, register) in &self.writes {
            if let WriteTarget::Synapse(state) = target {
                for lane in 0..len {
                    states[state].set_register(edge + lane, self.registers[register][lane]);
                }
            }
        }
    }

    fn commit_event(
        &self,
        pre_neuron_states: &mut [StateArray],
        post_neuron_states: &mut [StateArray],
        synapse_states: &mut [StateArray],
        source_neuron: usize,
        target_neuron: usize,
        edge: usize,
    ) {
        for &(target, register) in &self.writes {
            match target {
                WriteTarget::Neuron(state) => post_neuron_states[state]
                    .set_register(target_neuron, self.registers[register][0]),
                WriteTarget::PreNeuron(state) => pre_neuron_states[state]
                    .set_register(source_neuron, self.registers[register][0]),
                WriteTarget::Synapse(state) => {
                    synapse_states[state].set_register(edge, self.registers[register][0])
                }
                WriteTarget::Available => {}
            }
        }
    }

    fn commit_event_same_population(
        &self,
        neuron_states: &mut [StateArray],
        synapse_states: &mut [StateArray],
        source_neuron: usize,
        target_neuron: usize,
        edge: usize,
    ) {
        for &(target, register) in &self.writes {
            match target {
                WriteTarget::Neuron(state) => {
                    neuron_states[state].set_register(target_neuron, self.registers[register][0])
                }
                WriteTarget::PreNeuron(state) => {
                    neuron_states[state].set_register(source_neuron, self.registers[register][0])
                }
                WriteTarget::Synapse(state) => {
                    synapse_states[state].set_register(edge, self.registers[register][0])
                }
                WriteTarget::Available => {}
            }
        }
    }
}

#[derive(Clone, Copy)]
struct Batch {
    start: usize,
    len: usize,
    source: usize,
    target: usize,
    source_state: usize,
    target_state: usize,
    edge: usize,
}

impl Batch {
    fn neurons(start: usize, len: usize) -> Self {
        Self {
            start,
            len,
            source: 0,
            target: 0,
            source_state: 0,
            target_state: 0,
            edge: 0,
        }
    }
    fn index(self, domain: Domain, lane: usize) -> usize {
        match domain {
            Domain::Neuron => self.start + lane,
            Domain::Pre => self.source,
            Domain::Post => self.target,
            Domain::Edge => self.edge,
        }
    }
}

fn copy_input(output: &mut [f64], input: &[f64], domain: Domain, batch: Batch) {
    if matches!(domain, Domain::Neuron) {
        output.copy_from_slice(&input[batch.start..batch.start + batch.len]);
    } else {
        output.fill(input[batch.index(domain, 0)]);
    }
}

fn copy_state_input(output: &mut [f64], input: &StateArray, domain: Domain, batch: Batch) {
    if matches!(domain, Domain::Neuron) {
        input.copy_registers(output, batch.start);
    } else {
        output.fill(input.get_register(batch.index(domain, 0)));
    }
}

// Dispatch happens once per instruction/tile, not once per expression/neuron.
// Keep every arithmetic intermediate finite, and never evaluate a guarded RHS
// on inactive lanes (e.g. division by zero in a refractory neuron).
fn binary<F: Fn(f64, f64) -> f64>(
    left: &[f64],
    right: &[f64],
    output: &mut [f64],
    guard: Option<&[f64]>,
    operation: F,
) -> Result<()> {
    let mut valid = true;
    if let Some(guard) = guard {
        for (((out, a), b), mask) in output.iter_mut().zip(left).zip(right).zip(guard) {
            if *mask != 0.0 {
                *out = operation(*a, *b);
                valid &= out.is_finite();
            }
        }
    } else {
        for ((out, a), b) in output.iter_mut().zip(left).zip(right) {
            *out = operation(*a, *b);
            valid &= out.is_finite();
        }
    }
    check(valid, "non-finite numeric value")
}

fn ternary<F: Fn(f64, f64, f64) -> f64>(
    first: &[f64],
    second: &[f64],
    third: &[f64],
    output: &mut [f64],
    guard: Option<&[f64]>,
    operation: F,
) -> Result<()> {
    let mut valid = true;
    if let Some(guard) = guard {
        for ((((out, a), b), c), mask) in output
            .iter_mut()
            .zip(first)
            .zip(second)
            .zip(third)
            .zip(guard)
        {
            if *mask != 0.0 {
                *out = operation(*a, *b, *c);
                valid &= out.is_finite();
            }
        }
    } else {
        for (((out, a), b), c) in output.iter_mut().zip(first).zip(second).zip(third) {
            *out = operation(*a, *b, *c);
            valid &= out.is_finite();
        }
    }
    check(valid, "non-finite numeric value")
}

fn cast_register(value: f64, source: DType, target: DType) -> Result<f64> {
    if target == DType::F64 {
        return match source {
            DType::Bool => Ok(f64::from(value != 0.0)),
            DType::F32 | DType::F64 => Ok(value),
            DType::I32 => Ok((value.to_bits() as u32 as i32) as f64),
            DType::I64 => Ok((value.to_bits() as i64) as f64),
            DType::U32 => Ok((value.to_bits() as u32) as f64),
            DType::U64 => Ok(value.to_bits() as f64),
            _ => Err("unsupported cast source".into()),
        };
    }
    let encode_signed = |numeric: i128| match target {
        DType::I32 => Some((numeric as i32) as u32 as u64),
        DType::I64 => Some((numeric as i64) as u64),
        DType::U32 => Some((numeric as u32) as u64),
        DType::U64 => Some(numeric as u64),
        _ => None,
    };
    let encode_unsigned = |numeric: u128| match target {
        DType::I32 => Some((numeric as i32) as u32 as u64),
        DType::I64 => Some((numeric as i64) as u64),
        DType::U32 => Some((numeric as u32) as u64),
        DType::U64 => Some(numeric as u64),
        _ => None,
    };
    let integer_raw = match source {
        DType::I32 => encode_signed((value.to_bits() as u32 as i32) as i128),
        DType::I64 => encode_signed((value.to_bits() as i64) as i128),
        DType::U32 => encode_unsigned((value.to_bits() as u32) as u128),
        DType::U64 => encode_unsigned(value.to_bits() as u128),
        _ => None,
    };
    if let Some(raw) = integer_raw {
        return Ok(f64::from_bits(raw));
    }
    let numeric = match source {
        DType::Bool => f64::from(value != 0.0),
        DType::F32 | DType::F64 => value,
        _ => return Err("unsupported cast source".into()),
    };
    let raw = match target {
        DType::I32 => (numeric as i32) as u32 as u64,
        DType::I64 => (numeric as i64) as u64,
        DType::U32 => (numeric as u32) as u64,
        DType::U64 => numeric as u64,
        _ => return Err("unsupported cast target".into()),
    };
    Ok(f64::from_bits(raw))
}

fn integer_unary_value(dtype: DType, op: IntegerUnary, value: f64) -> f64 {
    let raw = value.to_bits();
    let result = match dtype {
        DType::I32 => match op {
            IntegerUnary::Neg => (raw as u32 as i32).wrapping_neg() as u32 as u64,
            IntegerUnary::Abs => (raw as u32 as i32).wrapping_abs() as u32 as u64,
        },
        DType::I64 => match op {
            IntegerUnary::Neg => (raw as i64).wrapping_neg() as u64,
            IntegerUnary::Abs => (raw as i64).wrapping_abs() as u64,
        },
        _ => unreachable!("validated integer unary dtype"),
    };
    f64::from_bits(result)
}

fn integer_binary_value(dtype: DType, op: IntegerBinary, left: f64, right: f64) -> Result<f64> {
    macro_rules! calculate {
        ($ty:ty, $decode:expr, $encode:expr) => {{
            let a: $ty = $decode(left.to_bits());
            let b: $ty = $decode(right.to_bits());
            let boolean = match op {
                IntegerBinary::Gt => Some(a > b),
                IntegerBinary::Ge => Some(a >= b),
                IntegerBinary::Lt => Some(a < b),
                IntegerBinary::Le => Some(a <= b),
                _ => None,
            };
            if let Some(value) = boolean {
                return Ok(f64::from(value));
            }
            let value: $ty = match op {
                IntegerBinary::Add => a.wrapping_add(b),
                IntegerBinary::Sub => a.wrapping_sub(b),
                IntegerBinary::Mul => a.wrapping_mul(b),
                IntegerBinary::Mod => {
                    check(b != 0, "integer modulo by zero")?;
                    let quotient = a.wrapping_div(b);
                    let remainder = a.wrapping_rem(b);
                    let floor_quotient = if remainder != 0 && (remainder < 0) != (b < 0) {
                        quotient.wrapping_sub(1)
                    } else {
                        quotient
                    };
                    a.wrapping_sub(floor_quotient.wrapping_mul(b))
                }
                IntegerBinary::FloorDiv => {
                    check(b != 0, "integer division by zero")?;
                    let quotient = a.wrapping_div(b);
                    let remainder = a.wrapping_rem(b);
                    if remainder != 0 && (remainder < 0) != (b < 0) {
                        quotient.wrapping_sub(1)
                    } else {
                        quotient
                    }
                }
                _ => unreachable!(),
            };
            $encode(value)
        }};
    }
    let raw = match dtype {
        DType::I32 => calculate!(i32, |v: u64| v as u32 as i32, |v: i32| v as u32 as u64),
        DType::I64 => calculate!(i64, |v: u64| v as i64, |v: i64| v as u64),
        DType::U32 => {
            let a = left.to_bits() as u32;
            let b = right.to_bits() as u32;
            match op {
                IntegerBinary::Gt => return Ok(f64::from(a > b)),
                IntegerBinary::Ge => return Ok(f64::from(a >= b)),
                IntegerBinary::Lt => return Ok(f64::from(a < b)),
                IntegerBinary::Le => return Ok(f64::from(a <= b)),
                IntegerBinary::Add => a.wrapping_add(b) as u64,
                IntegerBinary::Sub => a.wrapping_sub(b) as u64,
                IntegerBinary::Mul => a.wrapping_mul(b) as u64,
                IntegerBinary::Mod => a.checked_rem(b).ok_or("integer modulo by zero")? as u64,
                IntegerBinary::FloorDiv => {
                    a.checked_div(b).ok_or("integer division by zero")? as u64
                }
            }
        }
        DType::U64 => {
            let a = left.to_bits();
            let b = right.to_bits();
            match op {
                IntegerBinary::Gt => return Ok(f64::from(a > b)),
                IntegerBinary::Ge => return Ok(f64::from(a >= b)),
                IntegerBinary::Lt => return Ok(f64::from(a < b)),
                IntegerBinary::Le => return Ok(f64::from(a <= b)),
                IntegerBinary::Add => a.wrapping_add(b),
                IntegerBinary::Sub => a.wrapping_sub(b),
                IntegerBinary::Mul => a.wrapping_mul(b),
                IntegerBinary::Mod => a.checked_rem(b).ok_or("integer modulo by zero")?,
                IntegerBinary::FloorDiv => a.checked_div(b).ok_or("integer division by zero")?,
            }
        }
        _ => return Err("invalid integer operation dtype".into()),
    };
    Ok(f64::from_bits(raw))
}

fn evaluate(
    instructions: &[Instruction],
    registers: &mut [Vec<f64>],
    len: usize,
    random: Option<(u64, u64, Batch)>,
) -> Result<()> {
    for instruction in instructions {
        // SSA guarantees that all inputs precede the output: safe disjoint
        // borrows, with no unsafe indexing or aliasing of input/output arrays.
        let (inputs, outputs) = registers.split_at_mut(instruction.output);
        let left = &inputs[instruction.left][..len];
        let right = &inputs[instruction.right][..len];
        let third = instruction.third.map(|slot| &inputs[slot][..len]);
        let output = &mut outputs[0][..len];
        let guard = instruction.guard.map(|slot| &inputs[slot][..len]);
        match &instruction.op {
            Op::Neg => binary(left, right, output, guard, |a, _| -a)?,
            Op::Not => binary(left, right, output, guard, |a, _| f64::from(a == 0.0))?,
            Op::Abs => binary(left, right, output, guard, |a, _| a.abs())?,
            Op::Arccos => binary(left, right, output, guard, |a, _| a.acos())?,
            Op::Arcsin => binary(left, right, output, guard, |a, _| a.asin())?,
            Op::Arctan => binary(left, right, output, guard, |a, _| a.atan())?,
            Op::Ceil => binary(left, right, output, guard, |a, _| a.ceil())?,
            Op::Cos => binary(left, right, output, guard, |a, _| a.cos())?,
            Op::Cosh => binary(left, right, output, guard, |a, _| a.cosh())?,
            Op::Exp => binary(left, right, output, guard, |a, _| a.exp())?,
            Op::Expm1 => binary(left, right, output, guard, |a, _| a.exp_m1())?,
            Op::Exprel => binary(left, right, output, guard, |a, _| {
                if a.abs() < 1e-16 {
                    1.0
                } else if a > 717.0 {
                    f64::INFINITY
                } else {
                    a.exp_m1() / a
                }
            })?,
            Op::Floor => binary(left, right, output, guard, |a, _| a.floor())?,
            Op::Trunc => binary(left, right, output, guard, |a, _| {
                if a.abs() <= 9_007_199_254_740_992.0 {
                    a.trunc()
                } else {
                    f64::NAN
                }
            })?,
            Op::RoundF32 => binary(left, right, output, guard, |a, _| {
                let value = a as f32;
                if value.is_finite() {
                    value as f64
                } else {
                    f64::NAN
                }
            })?,
            Op::Cast(source, target) => {
                for lane in 0..len {
                    if guard.is_none_or(|mask| mask[lane] != 0.0) {
                        output[lane] = cast_register(left[lane], *source, *target)?;
                    }
                }
            }
            Op::IntegerUnary(dtype, operation) => {
                for lane in 0..len {
                    if guard.is_none_or(|mask| mask[lane] != 0.0) {
                        output[lane] = integer_unary_value(*dtype, *operation, left[lane]);
                    }
                }
            }
            Op::IntegerBinary(dtype, operation) => {
                for lane in 0..len {
                    if guard.is_none_or(|mask| mask[lane] != 0.0) {
                        output[lane] =
                            integer_binary_value(*dtype, *operation, left[lane], right[lane])?;
                    }
                }
            }
            Op::Timestep => {
                for lane in 0..len {
                    if guard.is_none_or(|mask| mask[lane] != 0.0) {
                        let value = timestep(left[lane], right[lane])?;
                        check(
                            value <= 9_007_199_254_740_992,
                            "timestep outside exact f64 execution range",
                        )?;
                        output[lane] = value as f64;
                    }
                }
            }
            Op::TickOffset(offset) => binary(left, right, output, guard, |value, _| {
                let result = value + *offset as f64;
                if result.abs() <= 9_007_199_254_740_992.0 {
                    result
                } else {
                    f64::NAN
                }
            })?,
            Op::Log => binary(left, right, output, guard, |a, _| a.ln())?,
            Op::Log10 => binary(left, right, output, guard, |a, _| a.log10())?,
            Op::Log1p => binary(left, right, output, guard, |a, _| a.ln_1p())?,
            Op::Sign => binary(left, right, output, guard, |a, _| {
                if a > 0.0 {
                    1.0
                } else if a < 0.0 {
                    -1.0
                } else {
                    0.0
                }
            })?,
            Op::Sin => binary(left, right, output, guard, |a, _| a.sin())?,
            Op::Sinh => binary(left, right, output, guard, |a, _| a.sinh())?,
            Op::Sqrt => binary(left, right, output, guard, |a, _| a.sqrt())?,
            Op::Tan => binary(left, right, output, guard, |a, _| a.tan())?,
            Op::Tanh => binary(left, right, output, guard, |a, _| a.tanh())?,
            Op::Add => binary(left, right, output, guard, |a, b| a + b)?,
            Op::Sub => binary(left, right, output, guard, |a, b| a - b)?,
            Op::Mul => binary(left, right, output, guard, |a, b| a * b)?,
            Op::Div => binary(left, right, output, guard, |a, b| a / b)?,
            Op::Pow => binary(left, right, output, guard, |a, b| a.powf(b))?,
            Op::Mod => binary(left, right, output, guard, |a, b| a - b * (a / b).floor())?,
            Op::FloorDiv => binary(left, right, output, guard, |a, b| (a / b).floor())?,
            Op::Eq => binary(left, right, output, guard, |a, b| f64::from(a == b))?,
            Op::Ne => binary(left, right, output, guard, |a, b| f64::from(a != b))?,
            Op::Gt => binary(left, right, output, guard, |a, b| f64::from(a > b))?,
            Op::Ge => binary(left, right, output, guard, |a, b| f64::from(a >= b))?,
            Op::Lt => binary(left, right, output, guard, |a, b| f64::from(a < b))?,
            Op::Le => binary(left, right, output, guard, |a, b| f64::from(a <= b))?,
            Op::And => binary(left, right, output, guard, |a, b| {
                f64::from(a != 0.0 && b != 0.0)
            })?,
            Op::Or => binary(left, right, output, guard, |a, b| {
                f64::from(a != 0.0 || b != 0.0)
            })?,
            Op::Binomial(stream, n, approximate, domain) => {
                let (seed, tick, batch) = random.ok_or("binomial in scalar expression")?;
                let mut valid = true;
                for lane in 0..len {
                    if guard.is_none_or(|mask| mask[lane] != 0.0) {
                        let probability = left[lane];
                        let lane_valid =
                            probability.is_finite() && (0.0..=1.0).contains(&probability);
                        valid &= lane_valid;
                        if lane_valid {
                            output[lane] = counter_binomial(
                                seed,
                                *stream,
                                tick,
                                batch.index(*domain, lane) as u64,
                                *n,
                                probability,
                                *approximate,
                            );
                        }
                    }
                }
                check(valid, "invalid binomial probability")?;
            }
            Op::Poisson(stream, domain) => {
                let (seed, tick, batch) = random.ok_or("poisson in scalar expression")?;
                let mut valid = true;
                for lane in 0..len {
                    if guard.is_none_or(|mask| mask[lane] != 0.0) {
                        output[lane] = counter_poisson(
                            seed,
                            *stream,
                            tick,
                            batch.index(*domain, lane) as u64,
                            left[lane],
                        );
                        valid &= output[lane].is_finite();
                    }
                }
                check(valid, "invalid poisson lambda")?;
            }
            Op::Clip => ternary(
                left,
                right,
                third.ok_or("missing clip maximum")?,
                output,
                guard,
                |value, min, max| value.max(min).min(max),
            )?,
            Op::Select => {
                for (((out, new), old), mask) in output
                    .iter_mut()
                    .zip(left)
                    .zip(right)
                    .zip(guard.ok_or("missing mask")?)
                {
                    *out = if *mask != 0.0 { *new } else { *old };
                }
            }
            Op::TimedArray {
                values,
                rows,
                columns,
                epsilon,
                upsampling,
            } => {
                let width = columns.unwrap_or(1);
                let mut valid = true;
                for lane in 0..len {
                    if guard.is_some_and(|mask| mask[lane] == 0.0) {
                        continue;
                    }
                    let raw_row = ((left[lane] / epsilon + 0.5) / *upsampling as f64).trunc();
                    let row = if raw_row <= 0.0 {
                        0
                    } else {
                        (raw_row as usize).min(rows - 1)
                    };
                    let column = if columns.is_some() {
                        let raw = right[lane];
                        if !raw.is_finite() || raw < 0.0 || raw >= width as f64 {
                            valid = false;
                            continue;
                        }
                        raw as usize
                    } else {
                        0
                    };
                    output[lane] = values[row * width + column];
                    valid &= output[lane].is_finite();
                }
                check(valid, "TimedArray index/value outside supported range")?;
            }
        }
    }
    Ok(())
}

struct PathwayQueue {
    offsets: Vec<usize>,
    edges: Vec<usize>,
    queue: Vec<Vec<usize>>, // uniform: source IDs; heterogeneous: edge IDs
    uniform_delay: Option<usize>,
    delivered: usize,
}

fn edge_csr(indices: &[usize], count: usize) -> (Vec<usize>, Vec<usize>) {
    let mut offsets = vec![0; count + 1];
    for &index in indices {
        offsets[index + 1] += 1;
    }
    for index in 0..count {
        offsets[index + 1] += offsets[index];
    }
    let mut next = offsets[..count].to_vec();
    let mut edges = vec![0; indices.len()];
    for (edge, &index) in indices.iter().enumerate() {
        edges[next[index]] = edge;
        next[index] += 1;
    }
    (offsets, edges)
}

impl PathwayQueue {
    fn new(
        pathway: &PathwayInstance,
        delay_ticks: &[usize],
        endpoint_indices: &[usize],
        endpoint_count: usize,
        steps: usize,
    ) -> Self {
        let (offsets, edges) = edge_csr(endpoint_indices, endpoint_count);
        let uniform_delay = delay_ticks
            .first()
            .copied()
            .filter(|delay| delay_ticks.iter().all(|value| value == delay));
        let max_delay = delay_ticks.iter().copied().max().unwrap_or(0);
        let mut queue = vec![Vec::new(); (max_delay + 1).min(steps)];
        for event in &pathway.pending {
            let slot = event.delivery_tick % queue.len();
            queue[slot].push(event.item);
        }
        Self {
            offsets,
            edges,
            queue,
            uniform_delay,
            delivered: 0,
        }
    }
}

fn add_parameters(
    inputs: &mut BTreeMap<String, Input>,
    symbols: &[Symbol],
    values: &BTreeMap<String, EncodedArray>,
    domain: Domain,
    generated: &BTreeMap<String, Vec<f64>>,
) -> Result<()> {
    for symbol in symbols {
        let array = if let Some(array) = generated.get(&symbol.name) {
            array.clone()
        } else {
            values[&symbol.name]
                .iter()
                .map(|bits| decode_typed_float(bits, symbol.dtype))
                .collect::<Result<Vec<_>>>()?
        };
        inputs.insert(
            symbol.name.clone(),
            if symbol.index_domain == IndexDomain::Scalar {
                Input::Constant(array[0], symbol.dtype)
            } else {
                Input::Parameter(array, domain, symbol.dtype)
            },
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn three_distinct_mut_preserves_requested_order_for_every_permutation() {
        for [first, second, third] in [
            [0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0],
        ] {
            let mut values = [0, 0, 0];
            let (a, b, c) = three_distinct_mut(&mut values, first, second, third).unwrap();
            *a = 11;
            *b = 22;
            *c = 33;
            assert_eq!(values[first], 11);
            assert_eq!(values[second], 22);
            assert_eq!(values[third], 33);
        }
    }
}
