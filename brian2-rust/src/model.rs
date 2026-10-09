// Shared AtlasIR types, validation and native command-line adapter.
use compact_input::EncodedArray;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::time::Instant;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Model {
    schema: String,
    protocol: Protocol,
    definition: Definition,
    instance: Instance,
    run: Run,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Protocol {
    name: String,
    version: ProtocolVersion,
    canonical_encoding: String,
    hash_algorithm: String,
    layers: ProtocolLayers,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ProtocolVersion {
    major: u32,
    minor: u32,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ProtocolLayers {
    definition: String,
    instance: String,
    run: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Definition {
    synapses: Vec<SynapseDefinition>,
    populations: Vec<PopulationDefinition>,
    clocks: Vec<ClockDefinition>,
    functions: Vec<FunctionDefinition>,
    schedule: ScheduleDefinition,
    numeric_profile: String,
    rng_algorithm: String,
}
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct FunctionDefinition {
    name: String,
    semantic_version: String,
    abi: String,
    arguments: Vec<FunctionArgument>,
    return_dtype: DType,
    return_dimensions: Dimensions,
    effects: FunctionEffects,
    body: Option<Expr>,
    implementations: BTreeMap<String, String>,
    backend_implementations: BTreeMap<String, NativeFunctionImplementation>,
}
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct NativeFunctionImplementation {
    abi: String,
    symbol: String,
    source: String,
    source_sha256: String,
}
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct FunctionArgument {
    name: String,
    dtype: DType,
    dimensions: Dimensions,
}
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct FunctionEffects {
    stateful: bool,
    deterministic: bool,
    thread_safe: bool,
    rng: bool,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ScheduleDefinition {
    base_slots: Vec<String>,
    slots: Vec<String>,
    nodes: Vec<ScheduleNode>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ScheduleNode {
    id: String,
    operation: ScheduleOperation,
    owner_kind: ScheduleOwnerKind,
    owner_index: usize,
    item_index: usize,
    clock: usize,
    when: String,
    order: i32,
    name: String,
    effects: ScheduleEffects,
    dependencies: Vec<String>,
}
#[derive(Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum ScheduleOperation {
    CodeObject,
    StateMonitor,
    SpikeMonitor,
    EventMonitor,
    EventSource,
}
#[derive(Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum ScheduleOwnerKind {
    Population,
    Synapse,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ScheduleEffects {
    reads: Vec<String>,
    writes: Vec<String>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PopulationDefinition {
    name: String,
    offset: usize,
    count: usize,
    states: Vec<Symbol>,
    parameters: Vec<Symbol>,
    linked_variables: Vec<LinkedVariableDefinition>,
    #[serde(default)]
    spatial: Option<SpatialDefinition>,
    refractory: Option<RefractoryDefinition>,
    events: Vec<String>,
    code_objects: Vec<CodeObjectSpec>,
    state_monitors: Vec<StateMonitorDefinition>,
    #[serde(default)]
    monitor_expressions: BTreeMap<String, String>,
    #[serde(default)]
    monitor_timed_arrays: BTreeMap<String, MonitorTimedArrayDefinition>,
    event_monitors: Vec<EventMonitorDefinition>,
    spike_monitor: Option<String>,
    #[serde(default)]
    rate_monitors: Vec<RateMonitorDefinition>,
    monitor: Monitor,
    clock: usize,
    dt: String,
    steps: usize,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SpatialDefinition {
    voltage: String,
    membrane_current: String,
    capacitance: String,
    resistivity: String,
    area: String,
    r_length_1: String,
    r_length_2: String,
    starts: Vec<usize>,
    ends: Vec<usize>,
    parents: Vec<usize>,
    child_slots: Vec<usize>,
    children_count: Vec<usize>,
    children: Vec<usize>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct MonitorTimedArrayDefinition {
    values: Vec<String>,
    rows: usize,
    columns: Option<usize>,
    epsilon: String,
    upsampling: usize,
    dimensions: Dimensions,
}

impl MonitorTimedArrayDefinition {
    fn validate(&self) -> Result<()> {
        let width = self.columns.unwrap_or(1);
        check(
            self.rows > 0
                && width > 0
                && self.rows.checked_mul(width) == Some(self.values.len())
                && self.values.len() <= resource_limits::timed_array_values()?
                && decode_bits(&self.epsilon)? > 0.0
                && self.upsampling >= 1
                && valid_dimensions(&self.dimensions),
            "invalid monitor TimedArray layout",
        )?;
        for value in &self.values {
            decode_bits(value)?;
        }
        Ok(())
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct LinkedVariableDefinition {
    name: String,
    dtype: DType,
    dimensions: Dimensions,
    source_population: usize,
    source_state: String,
    index: LinkedIndexDefinition,
}
#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum LinkedIndexDefinition {
    Identity,
    Constant { values: Vec<usize> },
    State { name: String },
    Parameter { name: String },
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StateMonitorDefinition {
    name: String,
    variables: Vec<String>,
    #[serde(default)]
    output_variables: Vec<String>,
    #[serde(default)]
    sources: Vec<SynapseMonitorSourceDefinition>,
    record: Vec<usize>,
    clock: usize,
    #[serde(default = "default_state_monitor_when")]
    when: String,
    #[serde(default)]
    order: i32,
}

#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum SynapseMonitorSourceDefinition {
    SynapseState { name: String, dtype: DType },
    PreState { name: String, dtype: DType },
    PostState { name: String, dtype: DType },
    Linked { name: String, dtype: DType },
}

fn default_state_monitor_when() -> String {
    "start".to_owned()
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RateMonitorDefinition {
    name: String,
    dtype: DType,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct EventMonitorDefinition {
    name: String,
    event: String,
    variables: Vec<String>,
    clock: usize,
    when: String,
    order: i32,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ClockDefinition {
    name: String,
    dt: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Symbol {
    name: String,
    dtype: DType,
    dimensions: [f64; 7],
    index_domain: IndexDomain,
}
#[derive(Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
enum DType {
    F32,
    F64,
    I32,
    I64,
    U32,
    U64,
    Bool,
    Index,
    Tick,
}

impl DType {
    fn is_integer(self) -> bool {
        matches!(self, Self::I32 | Self::I64 | Self::U32 | Self::U64)
    }
}
type Dimensions = [f64; 7];
const DIMENSIONLESS: Dimensions = [0.0; 7];
const TIME_DIMENSIONS: Dimensions = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];

fn valid_dimensions(value: &Dimensions) -> bool {
    value
        .iter()
        .all(|component| component.is_finite() && component.abs() <= 1.0e6)
}

fn same_dimensions(left: &Dimensions, right: &Dimensions) -> bool {
    left.iter()
        .zip(right)
        .all(|(left, right)| (left - right).abs() <= 1.0e-12)
}

fn combine_dimensions(
    left: &Dimensions,
    right: &Dimensions,
    operation: impl Fn(f64, f64) -> f64,
) -> Dimensions {
    std::array::from_fn(|index| operation(left[index], right[index]))
}
#[derive(Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
enum IndexDomain {
    Scalar,
    Neuron,
    Synapse,
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Instance {
    neuron_count: usize,
    populations: Vec<PopulationInstance>,
    synapses: Vec<SynapseInstance>,
    rng_seed: u64,
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct PopulationInstance {
    initial_state: BTreeMap<String, EncodedArray>,
    parameters: BTreeMap<String, EncodedArray>,
    refractory: Option<RefractoryInstance>,
    spike_generator: Option<SpikeGeneratorInstance>,
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct SpikeGeneratorInstance {
    spike_ticks: Vec<usize>,
    spike_indices: Vec<usize>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RefractoryDefinition {
    frozen_states: Vec<String>,
    mode: String,
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct RefractoryInstance {
    period: String,
    period_ticks: u64,
    initial_lastspike: EncodedArray,
    initial_not_refractory: Vec<bool>,
}
struct RefractoryRuntime {
    lastspike: Vec<f64>,
    not_refractory: Vec<bool>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SynapseDefinition {
    name: String,
    states: Vec<Symbol>,
    clock_driven_states: Vec<String>,
    parameters: Vec<Symbol>,
    #[serde(default)]
    linked_variables: Vec<LinkedVariableDefinition>,
    pre_state_aliases: BTreeMap<String, String>,
    post_state_aliases: BTreeMap<String, String>,
    source_population: usize,
    target_population: usize,
    #[serde(default)]
    source_synapse: Option<usize>,
    #[serde(default)]
    target_synapse: Option<usize>,
    source_start: usize,
    target_start: usize,
    source_count: usize,
    target_count: usize,
    code_objects: Vec<CodeObjectSpec>,
    #[serde(default)]
    state_monitors: Vec<StateMonitorDefinition>,
    #[serde(default)]
    monitor_expressions: BTreeMap<String, String>,
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct SynapseInstance {
    source: Vec<usize>,
    target: Vec<usize>,
    topology: TopologyInstance,
    initial_state: BTreeMap<String, EncodedArray>,
    parameters: BTreeMap<String, EncodedArray>,
    pathways: Vec<PathwayInstance>,
}
#[derive(Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum TopologyInstance {
    Explicit,
    BinaryCsr {
        edge_count: usize,
        path: String,
        sha256: String,
        column_count: usize,
        initializers: BTreeMap<String, usize>,
    },
    FixedTotal {
        edge_count: usize,
        seed: u64,
        initializers: BTreeMap<String, Initializer>,
    },
    FixedIndegree {
        edge_count: usize,
        indegree: usize,
        seed: u64,
        initializers: BTreeMap<String, Initializer>,
    },
}

impl SynapseInstance {
    fn edge_count(&self) -> usize {
        match &self.topology {
            TopologyInstance::Explicit => self.source.len(),
            TopologyInstance::FixedTotal { edge_count, .. }
            | TopologyInstance::FixedIndegree { edge_count, .. }
            | TopologyInstance::BinaryCsr { edge_count, .. } => *edge_count,
        }
    }
}
#[derive(Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum Initializer {
    ClippedNormal {
        mean: String,
        std: String,
        minimum: Option<String>,
        maximum: Option<String>,
        stream: u64,
    },
    Uniform {
        minimum: String,
        maximum: String,
        stream: u64,
    },
}
enum InitializerValues {
    ClippedNormal {
        mean: f64,
        std: f64,
        minimum: Option<f64>,
        maximum: Option<f64>,
        stream: u64,
    },
    Uniform {
        minimum: f64,
        maximum: f64,
        stream: u64,
    },
}

impl Initializer {
    fn values(&self) -> Result<InitializerValues> {
        match self {
            Self::ClippedNormal {
                mean,
                std,
                minimum,
                maximum,
                stream,
            } => {
                let mean = decode_bits(mean)?;
                let std = decode_bits(std)?;
                let minimum = minimum.as_deref().map(decode_bits).transpose()?;
                let maximum = maximum.as_deref().map(decode_bits).transpose()?;
                check(
                    std >= 0.0 && minimum.zip(maximum).is_none_or(|(low, high)| low <= high),
                    "invalid clipped-normal initializer",
                )?;
                Ok(InitializerValues::ClippedNormal {
                    mean,
                    std,
                    minimum,
                    maximum,
                    stream: *stream,
                })
            }
            Self::Uniform {
                minimum,
                maximum,
                stream,
            } => {
                let minimum = decode_bits(minimum)?;
                let maximum = decode_bits(maximum)?;
                check(
                    minimum.is_finite() && maximum.is_finite() && minimum <= maximum,
                    "invalid uniform initializer",
                )?;
                Ok(InitializerValues::Uniform {
                    minimum,
                    maximum,
                    stream: *stream,
                })
            }
        }
    }

    fn bounds(&self) -> Result<(Option<f64>, Option<f64>)> {
        Ok(match self.values()? {
            InitializerValues::ClippedNormal {
                minimum, maximum, ..
            } => (minimum, maximum),
            InitializerValues::Uniform {
                minimum, maximum, ..
            } => (Some(minimum), Some(maximum)),
        })
    }
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct PathwayInstance {
    name: String,
    kind: String,
    event: String,
    delay: EncodedArray,
    delay_ticks: Vec<usize>,
    delay_initializer: Option<Initializer>,
    pending: Vec<PendingEvent>,
}
#[derive(Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct PendingEvent {
    delivery_tick: usize,
    item: usize,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Run {
    start: String,
    duration: String,
    clocks: Vec<RunClock>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RunClock {
    start_tick: usize,
    steps: usize,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Monitor {
    variables: Vec<String>,
    record: Vec<usize>,
    window_steps: usize,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CodeObjectSpec {
    name: String,
    kind: String,
    when: String,
    order: i32,
    clock: usize,
    iteration_domain: String,
    scalar: Vec<Statement>,
    vector: Vec<Statement>,
    effects: Effects,
    #[serde(default)]
    summed_target: Option<String>,
    #[serde(default)]
    summed_state: Option<String>,
    #[serde(default)]
    pathway_name: Option<String>,
    #[serde(default)]
    event_name: Option<String>,
    #[serde(default)]
    adaptive: Option<AdaptiveIntegratorDefinition>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct AdaptiveIntegratorDefinition {
    integrator: String,
    states: Vec<String>,
    derivatives: Vec<String>,
    absolute_errors: Vec<String>,
    adaptable_timestep: bool,
    max_steps: usize,
    use_last_timestep: bool,
    last_timestep: Option<String>,
    failed_steps: Option<String>,
    step_count: Option<String>,
    frozen_states: Vec<String>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Effects {
    reads: Vec<String>,
    writes: Vec<String>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Statement {
    target: String,
    dtype: DType,
    dimensions: [f64; 7],
    value: Expr,
    condition: Option<String>,
}
#[derive(Clone, Deserialize, Serialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
enum Expr {
    Literal {
        bits: String,
    },
    Boolean {
        value: bool,
    },
    Integer {
        dtype: DType,
        value: String,
    },
    Cast {
        dtype: DType,
        arg: Box<Expr>,
    },
    Rand {
        stream: u64,
    },
    Randn {
        stream: u64,
    },
    Binomial {
        stream: u64,
        n: u64,
        p: Box<Expr>,
        approximate: bool,
    },
    Poisson {
        stream: u64,
        lambda: Box<Expr>,
    },
    TimedArray {
        values: Vec<String>,
        rows: usize,
        columns: Option<usize>,
        epsilon: String,
        upsampling: usize,
        dimensions: [f64; 7],
        time: Box<Expr>,
        index: Option<Box<Expr>>,
    },
    BoolToF64 {
        arg: Box<Expr>,
    },
    IndexToF64 {
        arg: Box<Expr>,
    },
    TickToF64 {
        arg: Box<Expr>,
    },
    F32ToF64 {
        arg: Box<Expr>,
    },
    F64ToF32 {
        arg: Box<Expr>,
    },
    Timestep {
        time: Box<Expr>,
        dt: Box<Expr>,
    },
    TickOffset {
        tick: Box<Expr>,
        offset: i64,
    },
    Load {
        name: String,
    },
    Call {
        function: String,
        arguments: Vec<Expr>,
    },
    Neg {
        arg: Box<Expr>,
    },
    Not {
        arg: Box<Expr>,
    },
    Abs {
        arg: Box<Expr>,
    },
    Arccos {
        arg: Box<Expr>,
    },
    Arcsin {
        arg: Box<Expr>,
    },
    Arctan {
        arg: Box<Expr>,
    },
    Ceil {
        arg: Box<Expr>,
    },
    Cos {
        arg: Box<Expr>,
    },
    Cosh {
        arg: Box<Expr>,
    },
    Exp {
        arg: Box<Expr>,
    },
    Expm1 {
        arg: Box<Expr>,
    },
    Exprel {
        arg: Box<Expr>,
    },
    Floor {
        arg: Box<Expr>,
    },
    Trunc {
        arg: Box<Expr>,
    },
    Log {
        arg: Box<Expr>,
    },
    Log10 {
        arg: Box<Expr>,
    },
    Log1p {
        arg: Box<Expr>,
    },
    Sign {
        arg: Box<Expr>,
    },
    Sin {
        arg: Box<Expr>,
    },
    Sinh {
        arg: Box<Expr>,
    },
    Sqrt {
        arg: Box<Expr>,
    },
    Tan {
        arg: Box<Expr>,
    },
    Tanh {
        arg: Box<Expr>,
    },
    Add {
        left: Box<Expr>,
        right: Box<Expr>,
    },
    Sub {
        left: Box<Expr>,
        right: Box<Expr>,
    },
    Mul {
        left: Box<Expr>,
        right: Box<Expr>,
    },
    Div {
        left: Box<Expr>,
        right: Box<Expr>,
    },
    Pow {
        left: Box<Expr>,
        right: Box<Expr>,
    },
    Mod {
        left: Box<Expr>,
        right: Box<Expr>,
    },
    FloorDiv {
        left: Box<Expr>,
        right: Box<Expr>,
    },
    Eq {
        left: Box<Expr>,
        right: Box<Expr>,
    },
    Ne {
        left: Box<Expr>,
        right: Box<Expr>,
    },
    Gt {
        left: Box<Expr>,
        right: Box<Expr>,
    },
    Ge {
        left: Box<Expr>,
        right: Box<Expr>,
    },
    Lt {
        left: Box<Expr>,
        right: Box<Expr>,
    },
    Le {
        left: Box<Expr>,
        right: Box<Expr>,
    },
    And {
        left: Box<Expr>,
        right: Box<Expr>,
    },
    Or {
        left: Box<Expr>,
        right: Box<Expr>,
    },
    Clip {
        value: Box<Expr>,
        min: Box<Expr>,
        max: Box<Expr>,
    },
}
fn check(condition: bool, message: &str) -> Result<()> {
    if condition {
        Ok(())
    } else {
        Err(message.into())
    }
}
fn finite(value: f64) -> Result<f64> {
    check(value.is_finite(), "non-finite numeric value")?;
    Ok(value)
}
fn decode_bits(bits: &str) -> Result<f64> {
    check(
        bits.len() == 16
            && bits
                .bytes()
                .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c)),
        "f64 must be 16 lowercase hex digits (IEEE-754 bits)",
    )?;
    finite(f64::from_bits(u64::from_str_radix(bits, 16)?))
}
fn decode_f32_bits(bits: &str) -> Result<f32> {
    check(
        bits.len() == 8
            && bits
                .bytes()
                .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c)),
        "f32 must be 8 lowercase hex digits (IEEE-754 bits)",
    )?;
    let value = f32::from_bits(u32::from_str_radix(bits, 16)?);
    check(value.is_finite(), "non-finite numeric value")?;
    Ok(value)
}

fn decode_typed_float(bits: &str, dtype: DType) -> Result<f64> {
    match dtype {
        DType::F32 => Ok(decode_f32_bits(bits)? as f64),
        DType::F64 => decode_bits(bits),
        DType::Bool => match bits {
            "00" => Ok(0.0),
            "01" => Ok(1.0),
            _ => Err("bool must be encoded as 00 or 01".into()),
        },
        DType::I32 | DType::U32 => {
            check(
                bits.len() == 8
                    && bits
                        .bytes()
                        .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()),
                "32-bit integer must be 8 lowercase hex digits",
            )?;
            Ok(f64::from_bits(u64::from(u32::from_str_radix(bits, 16)?)))
        }
        DType::I64 | DType::U64 => {
            check(
                bits.len() == 16
                    && bits
                        .bytes()
                        .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()),
                "64-bit integer must be 16 lowercase hex digits",
            )?;
            Ok(f64::from_bits(u64::from_str_radix(bits, 16)?))
        }
        _ => Err("numeric array requires a public storage dtype".into()),
    }
}

fn validate_integer_literal(dtype: DType, value: &str) -> Result<()> {
    check(
        dtype.is_integer() && !value.is_empty(),
        "invalid integer literal dtype",
    )?;
    match dtype {
        DType::I32 => {
            value.parse::<i32>()?;
        }
        DType::I64 => {
            value.parse::<i64>()?;
        }
        DType::U32 => {
            value.parse::<u32>()?;
        }
        DType::U64 => {
            value.parse::<u64>()?;
        }
        _ => unreachable!(),
    }
    Ok(())
}
fn valid_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 128
        && name
            .bytes()
            .enumerate()
            .all(|(i, c)| c.is_ascii_alphabetic() || c == b'_' || (i > 0 && c.is_ascii_digit()))
}
fn budget(a: usize, b: usize, limit: usize) -> Result<()> {
    check(
        a.checked_mul(b).is_some_and(|v| v <= limit),
        "probe resource budget exceeded",
    )
}

const MAX_SYNAPSE_TICKS: u128 = 50_000_000_000_000;

// Match brian2.core.clocks.Clock._calc_timestep: exact/near-exact grid
// positions round to the nearest tick, all other times round up so adjacent
// Network.run intervals never execute the same tick twice.
fn clock_timestep(time: f64, dt: f64) -> Result<usize> {
    check(time >= 0.0 && time.is_finite() && dt > 0.0 && dt.is_finite(),
          "invalid Clock time or dt")?;
    let ratio = time / dt;
    let nearest = ratio.round();
    let tick = if (nearest * dt - time).abs() / dt <= 1e-4 {
        nearest
    } else {
        ratio.ceil()
    };
    check(tick >= 0.0 && tick <= usize::MAX as f64,
          "Clock timestep outside usize range")?;
    Ok(tick as usize)
}

fn synapse_budget(a: usize, b: usize) -> Result<()> {
    check(
        (a as u128) * (b as u128) <= MAX_SYNAPSE_TICKS,
        "synapse-tick budget exceeded",
    )
}

// Brian2 timestep(), deliberately distinct from delay's round-to-nearest.
fn timestep(time: f64, dt: f64) -> Result<u64> {
    let steps = (time + 1e-3 * dt) / dt;
    check(
        time >= 0.0
            && dt.is_finite()
            && dt > 0.0
            && steps.is_finite()
            && steps < 9_223_372_036_854_775_808.0,
        "timestep outside Brian2 int64 range",
    )?;
    Ok(steps as u64)
}

impl RefractoryInstance {
    fn validate(&self, n: usize, dt: f64, start: f64, steps: usize) -> Result<()> {
        check(
            self.period_ticks <= 1_000_000
                && timestep(decode_bits(&self.period)?, dt)? == self.period_ticks,
            "invalid refractory period/ticks",
        )?;
        check(
            self.initial_lastspike.len() == n && self.initial_not_refractory.len() == n,
            "invalid refractory array shape",
        )?;
        for value in self.initial_lastspike.unique_iter() {
            let last = decode_bits(value)?;
            check(
                last <= start,
                "initial lastspike must be at or before run start",
            )?;
            timestep(start + steps as f64 * dt - last, dt)?;
        }
        Ok(())
    }
    fn runtime(&self) -> Result<RefractoryRuntime> {
        Ok(RefractoryRuntime {
            lastspike: self
                .initial_lastspike
                .iter()
                .map(|v| decode_bits(v))
                .collect::<Result<_>>()?,
            not_refractory: self.initial_not_refractory.clone(),
        })
    }
}

impl Expr {
    fn random_streams(&self, streams: &mut Vec<u64>) {
        use Expr::*;
        match self {
            Rand { stream } | Randn { stream } => streams.push(*stream),
            Binomial { stream, p, .. } => {
                streams.push(*stream);
                p.random_streams(streams);
            }
            Poisson { stream, lambda } => {
                streams.push(*stream);
                lambda.random_streams(streams);
            }
            TimedArray { time, index, .. } => {
                time.random_streams(streams);
                if let Some(index) = index {
                    index.random_streams(streams);
                }
            }
            Timestep { time, dt } => {
                time.random_streams(streams);
                dt.random_streams(streams);
            }
            TickOffset { tick, .. } => tick.random_streams(streams),
            Call { arguments, .. } => {
                for argument in arguments {
                    argument.random_streams(streams);
                }
            }
            Cast { arg, .. }
            | BoolToF64 { arg }
            | IndexToF64 { arg }
            | TickToF64 { arg }
            | F32ToF64 { arg }
            | F64ToF32 { arg }
            | Neg { arg }
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
            | Tanh { arg } => arg.random_streams(streams),
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
                left.random_streams(streams);
                right.random_streams(streams);
            }
            Clip { value, min, max } => {
                value.random_streams(streams);
                min.random_streams(streams);
                max.random_streams(streams);
            }
            Literal { .. } | Integer { .. } | Boolean { .. } | Load { .. } => {}
        }
    }

    fn has_random(&self) -> bool {
        use Expr::*;
        match self {
            Rand { .. } | Randn { .. } | Binomial { .. } | Poisson { .. } => true,
            TimedArray { time, index, .. } => {
                time.has_random() || index.as_ref().is_some_and(|value| value.has_random())
            }
            Timestep { time, dt } => time.has_random() || dt.has_random(),
            TickOffset { tick, .. } => tick.has_random(),
            Call { arguments, .. } => arguments.iter().any(Expr::has_random),
            Cast { arg, .. }
            | BoolToF64 { arg }
            | IndexToF64 { arg }
            | TickToF64 { arg }
            | F32ToF64 { arg }
            | F64ToF32 { arg }
            | Neg { arg }
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
            | Tanh { arg } => arg.has_random(),
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
            | Or { left, right } => left.has_random() || right.has_random(),
            Clip { value, min, max } => value.has_random() || min.has_random() || max.has_random(),
            Literal { .. } | Integer { .. } | Boolean { .. } | Load { .. } => false,
        }
    }

    fn loads(&self, names: &mut BTreeSet<String>) {
        use Expr::*;
        match self {
            Load { name } => {
                names.insert(name.clone());
            }
            Call { arguments, .. } => {
                for argument in arguments {
                    argument.loads(names);
                }
            }
            TimedArray { time, index, .. } => {
                time.loads(names);
                if let Some(index) = index {
                    index.loads(names);
                }
            }
            Timestep { time, dt } => {
                time.loads(names);
                dt.loads(names);
            }
            TickOffset { tick, .. } => tick.loads(names),
            Binomial { p, .. } => p.loads(names),
            Poisson { lambda, .. } => lambda.loads(names),
            Cast { arg, .. }
            | BoolToF64 { arg }
            | IndexToF64 { arg }
            | TickToF64 { arg }
            | F32ToF64 { arg }
            | F64ToF32 { arg }
            | Neg { arg }
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
            | Tanh { arg } => arg.loads(names),
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
                left.loads(names);
                right.loads(names);
            }
            Clip { value, min, max } => {
                value.loads(names);
                min.loads(names);
                max.loads(names);
            }
            Literal { .. } | Integer { .. } | Boolean { .. } | Rand { .. } | Randn { .. } => {}
        }
    }

    fn infer(
        &self,
        symbols: &BTreeMap<String, DType>,
        reads: &mut BTreeSet<String>,
        functions: &BTreeMap<String, FunctionDefinition>,
    ) -> Result<DType> {
        use Expr::*;
        match self {
            Literal { bits } => {
                decode_bits(bits)?;
                Ok(DType::F64)
            }
            Boolean { .. } => Ok(DType::Bool),
            Integer { dtype, value } => {
                validate_integer_literal(*dtype, value)?;
                Ok(*dtype)
            }
            Cast { dtype, arg } => {
                let source = arg.infer(symbols, reads, functions)?;
                check(
                    (*dtype == DType::F64 || dtype.is_integer())
                        && (matches!(source, DType::Bool | DType::F32 | DType::F64)
                            || source.is_integer()
                            || matches!(source, DType::Index | DType::Tick)),
                    "invalid numeric cast",
                )?;
                Ok(*dtype)
            }
            Rand { stream } | Randn { stream } => {
                check(*stream < 1_000_000, "random stream id exceeds probe budget")?;
                Ok(DType::F64)
            }
            Binomial { stream, n, p, .. } => {
                check(
                    *stream < 1_000_000 && (1..=i32::MAX as u64).contains(n),
                    "invalid binomial expression",
                )?;
                check(
                    p.infer(symbols, reads, functions)? == DType::F64,
                    "binomial probability must be f64",
                )?;
                Ok(DType::F64)
            }
            Poisson { stream, lambda } => {
                check(*stream < 1_000_000, "random stream id exceeds probe budget")?;
                check(
                    lambda.infer(symbols, reads, functions)? == DType::F64,
                    "poisson lambda must be f64",
                )?;
                Ok(DType::F64)
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
                let width = columns.unwrap_or(1);
                check(
                    *rows > 0
                        && width > 0
                        && rows.checked_mul(width) == Some(values.len())
                        && values.len() <= resource_limits::timed_array_values()?
                        && decode_bits(epsilon)? > 0.0
                        && *upsampling >= 1
                        && columns.is_some() == index.is_some(),
                    "invalid TimedArray layout",
                )?;
                for value in values {
                    decode_bits(value)?;
                }
                check(
                    time.infer(symbols, reads, functions)? == DType::F64,
                    "TimedArray time argument must be f64",
                )?;
                if let Some(index) = index {
                    let dtype = index.infer(symbols, reads, functions)?;
                    check(
                        matches!(dtype, DType::F64 | DType::Index),
                        "TimedArray index argument must be numeric",
                    )?;
                }
                Ok(DType::F64)
            }
            BoolToF64 { arg } => {
                check(
                    arg.infer(symbols, reads, functions)? == DType::Bool,
                    "bool_to_f64 requires bool",
                )?;
                Ok(DType::F64)
            }
            IndexToF64 { arg } => {
                check(
                    arg.infer(symbols, reads, functions)? == DType::Index,
                    "index_to_f64 requires index",
                )?;
                Ok(DType::F64)
            }
            TickToF64 { arg } => {
                check(
                    arg.infer(symbols, reads, functions)? == DType::Tick,
                    "tick_to_f64 requires tick",
                )?;
                Ok(DType::F64)
            }
            F32ToF64 { arg } => {
                check(
                    arg.infer(symbols, reads, functions)? == DType::F32,
                    "f32_to_f64 requires f32",
                )?;
                Ok(DType::F64)
            }
            F64ToF32 { arg } => {
                check(
                    arg.infer(symbols, reads, functions)? == DType::F64,
                    "f64_to_f32 requires f64",
                )?;
                Ok(DType::F32)
            }
            Timestep { time, dt } => {
                check(
                    time.infer(symbols, reads, functions)? == DType::F64
                        && dt.infer(symbols, reads, functions)? == DType::F64,
                    "timestep arguments must be f64",
                )?;
                Ok(DType::Tick)
            }
            TickOffset { tick, offset } => {
                check(
                    tick.infer(symbols, reads, functions)? == DType::Tick
                        && offset.unsigned_abs() <= 9_007_199_254_740_992,
                    "invalid tick_offset",
                )?;
                Ok(DType::Tick)
            }
            Load { name } => {
                reads.insert(name.clone());
                symbols
                    .get(name)
                    .copied()
                    .ok_or_else(|| format!("undefined symbol: {name}").into())
            }
            Call {
                function,
                arguments,
            } => {
                let contract = functions
                    .get(function)
                    .ok_or_else(|| format!("undefined function: {function}"))?;
                check(
                    arguments.len() == contract.arguments.len(),
                    "function argument count does not match contract",
                )?;
                for (value, argument) in arguments.iter().zip(&contract.arguments) {
                    check(
                        value.infer(symbols, reads, functions)? == argument.dtype,
                        "function argument dtype does not match contract",
                    )?;
                }
                Ok(contract.return_dtype)
            }
            Neg { arg } => {
                let dtype = arg.infer(symbols, reads, functions)?;
                check(
                    matches!(dtype, DType::F64 | DType::I32 | DType::I64),
                    "neg requires f64 or a signed integer",
                )?;
                Ok(dtype)
            }
            Not { arg } => {
                check(
                    arg.infer(symbols, reads, functions)? == DType::Bool,
                    "not requires bool",
                )?;
                Ok(DType::Bool)
            }
            Abs { arg } => {
                let dtype = arg.infer(symbols, reads, functions)?;
                check(
                    matches!(dtype, DType::F64 | DType::I32 | DType::I64),
                    "abs requires f64 or a signed integer",
                )?;
                Ok(dtype)
            }
            Arccos { arg }
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
                check(
                    arg.infer(symbols, reads, functions)? == DType::F64,
                    "math function requires f64",
                )?;
                Ok(DType::F64)
            }
            Add { left, right }
            | Sub { left, right }
            | Mul { left, right }
            | Pow { left, right }
            | Mod { left, right }
            | FloorDiv { left, right }
            | Gt { left, right }
            | Ge { left, right }
            | Lt { left, right }
            | Le { left, right } => {
                let left_dtype = left.infer(symbols, reads, functions)?;
                let right_dtype = right.infer(symbols, reads, functions)?;
                let relational = matches!(self, Gt { .. } | Ge { .. } | Lt { .. } | Le { .. });
                let integer_operands = left_dtype == right_dtype && left_dtype.is_integer();
                check(
                    left_dtype == DType::F64 && right_dtype == DType::F64
                        || integer_operands
                        || relational
                            && left_dtype == right_dtype
                            && matches!(left_dtype, DType::F32 | DType::Index | DType::Tick),
                    "numeric operands must be f64 or matching logical values",
                )?;
                Ok(if relational {
                    DType::Bool
                } else if integer_operands {
                    left_dtype
                } else {
                    DType::F64
                })
            }
            Div { left, right } => {
                check(
                    left.infer(symbols, reads, functions)? == DType::F64
                        && right.infer(symbols, reads, functions)? == DType::F64,
                    "division operands must be f64",
                )?;
                Ok(DType::F64)
            }
            Eq { left, right } | Ne { left, right } => {
                check(
                    left.infer(symbols, reads, functions)?
                        == right.infer(symbols, reads, functions)?,
                    "equality operands must have the same type",
                )?;
                Ok(DType::Bool)
            }
            And { left, right } | Or { left, right } => {
                check(
                    left.infer(symbols, reads, functions)? == DType::Bool
                        && right.infer(symbols, reads, functions)? == DType::Bool,
                    "boolean operands must be bool",
                )?;
                Ok(DType::Bool)
            }
            Clip { value, min, max } => {
                check(
                    value.infer(symbols, reads, functions)? == DType::F64
                        && min.infer(symbols, reads, functions)? == DType::F64
                        && max.infer(symbols, reads, functions)? == DType::F64,
                    "clip operands must be f64",
                )?;
                Ok(DType::F64)
            }
        }
    }

    fn constant_f64(&self) -> Result<Option<f64>> {
        match self {
            Self::Literal { bits } => Ok(Some(decode_bits(bits)?)),
            Self::Integer { dtype, value } => {
                validate_integer_literal(*dtype, value)?;
                Ok(Some(value.parse::<f64>()?))
            }
            Self::Cast { arg, .. } => arg.constant_f64(),
            Self::Neg { arg } => Ok(arg.constant_f64()?.map(|value| -value)),
            _ => Ok(None),
        }
    }

    fn is_zero_literal(&self) -> Result<bool> {
        Ok(self.constant_f64()?.is_some_and(|value| value == 0.0))
    }

    fn infer_dimensions(
        &self,
        symbols: &BTreeMap<String, Dimensions>,
        functions: &BTreeMap<String, FunctionDefinition>,
    ) -> Result<Dimensions> {
        use Expr::*;
        match self {
            Literal { .. } | Integer { .. } | Boolean { .. } | Rand { .. } | Randn { .. } => {
                Ok(DIMENSIONLESS)
            }
            Binomial { p, .. } => {
                check(
                    same_dimensions(&p.infer_dimensions(symbols, functions)?, &DIMENSIONLESS),
                    "binomial probability must be dimensionless",
                )?;
                Ok(DIMENSIONLESS)
            }
            BoolToF64 { arg } => {
                check(
                    same_dimensions(&arg.infer_dimensions(symbols, functions)?, &DIMENSIONLESS),
                    "bool_to_f64 requires a dimensionless bool",
                )?;
                Ok(DIMENSIONLESS)
            }
            IndexToF64 { arg } | TickToF64 { arg } => {
                check(
                    same_dimensions(&arg.infer_dimensions(symbols, functions)?, &DIMENSIONLESS),
                    "logical-to-f64 conversion requires a dimensionless value",
                )?;
                Ok(DIMENSIONLESS)
            }
            Cast { arg, .. } | F32ToF64 { arg } | F64ToF32 { arg } => {
                arg.infer_dimensions(symbols, functions)
            }
            Timestep { time, dt } => {
                check(
                    same_dimensions(
                        &time.infer_dimensions(symbols, functions)?,
                        &TIME_DIMENSIONS,
                    ) && same_dimensions(
                        &dt.infer_dimensions(symbols, functions)?,
                        &TIME_DIMENSIONS,
                    ),
                    "timestep requires time-valued arguments",
                )?;
                Ok(DIMENSIONLESS)
            }
            TickOffset { tick, .. } => {
                check(
                    same_dimensions(&tick.infer_dimensions(symbols, functions)?, &DIMENSIONLESS),
                    "tick_offset requires a dimensionless tick",
                )?;
                Ok(DIMENSIONLESS)
            }
            Load { name } => symbols
                .get(name)
                .copied()
                .ok_or_else(|| format!("undefined symbol dimensions: {name}").into()),
            Call {
                function,
                arguments,
            } => {
                let contract = functions
                    .get(function)
                    .ok_or_else(|| format!("undefined function: {function}"))?;
                check(
                    arguments.len() == contract.arguments.len(),
                    "function argument count does not match contract",
                )?;
                for (value, argument) in arguments.iter().zip(&contract.arguments) {
                    check(
                        same_dimensions(
                            &value.infer_dimensions(symbols, functions)?,
                            &argument.dimensions,
                        ),
                        "function argument dimensions do not match contract",
                    )?;
                }
                Ok(contract.return_dimensions)
            }
            Poisson { lambda, .. } => {
                check(
                    same_dimensions(
                        &lambda.infer_dimensions(symbols, functions)?,
                        &DIMENSIONLESS,
                    ),
                    "poisson lambda must be dimensionless",
                )?;
                Ok(DIMENSIONLESS)
            }
            TimedArray {
                dimensions,
                time,
                index,
                ..
            } => {
                check(
                    valid_dimensions(dimensions)
                        && same_dimensions(
                            &time.infer_dimensions(symbols, functions)?,
                            &TIME_DIMENSIONS,
                        ),
                    "TimedArray requires a time-valued first argument",
                )?;
                if let Some(index) = index {
                    check(
                        same_dimensions(
                            &index.infer_dimensions(symbols, functions)?,
                            &DIMENSIONLESS,
                        ),
                        "TimedArray index must be dimensionless",
                    )?;
                }
                Ok(*dimensions)
            }
            Neg { arg } | Abs { arg } | Ceil { arg } | Floor { arg } | Trunc { arg } => {
                arg.infer_dimensions(symbols, functions)
            }
            Not { arg } => {
                check(
                    same_dimensions(&arg.infer_dimensions(symbols, functions)?, &DIMENSIONLESS),
                    "boolean not requires a dimensionless operand",
                )?;
                Ok(DIMENSIONLESS)
            }
            Sign { arg } => {
                arg.infer_dimensions(symbols, functions)?;
                Ok(DIMENSIONLESS)
            }
            Arccos { arg }
            | Arcsin { arg }
            | Arctan { arg }
            | Cos { arg }
            | Cosh { arg }
            | Exp { arg }
            | Expm1 { arg }
            | Exprel { arg }
            | Log { arg }
            | Log10 { arg }
            | Log1p { arg }
            | Sin { arg }
            | Sinh { arg }
            | Tan { arg }
            | Tanh { arg } => {
                check(
                    same_dimensions(&arg.infer_dimensions(symbols, functions)?, &DIMENSIONLESS),
                    "transcendental function requires a dimensionless operand",
                )?;
                Ok(DIMENSIONLESS)
            }
            Sqrt { arg } => {
                let value = arg.infer_dimensions(symbols, functions)?;
                Ok(std::array::from_fn(|index| value[index] * 0.5))
            }
            Add { left, right } | Sub { left, right } | Mod { left, right } => {
                let left = left.infer_dimensions(symbols, functions)?;
                let right = right.infer_dimensions(symbols, functions)?;
                if self.left_is_zero()? {
                    return Ok(right);
                }
                if self.right_is_zero()? {
                    return Ok(left);
                }
                check(
                    same_dimensions(&left, &right),
                    "add/sub/mod operands must have identical dimensions",
                )?;
                Ok(left)
            }
            Mul { left, right } => Ok(combine_dimensions(
                &left.infer_dimensions(symbols, functions)?,
                &right.infer_dimensions(symbols, functions)?,
                |left, right| left + right,
            )),
            Div { left, right } => Ok(combine_dimensions(
                &left.infer_dimensions(symbols, functions)?,
                &right.infer_dimensions(symbols, functions)?,
                |left, right| left - right,
            )),
            Pow { left, right } => {
                let base = left.infer_dimensions(symbols, functions)?;
                check(
                    same_dimensions(&right.infer_dimensions(symbols, functions)?, &DIMENSIONLESS),
                    "power exponent must be dimensionless",
                )?;
                if same_dimensions(&base, &DIMENSIONLESS) {
                    return Ok(DIMENSIONLESS);
                }
                let exponent = right
                    .constant_f64()?
                    .ok_or("dimensionful base requires a constant exponent")?;
                let result = std::array::from_fn(|index| base[index] * exponent);
                check(
                    valid_dimensions(&result),
                    "power produces invalid dimensions",
                )?;
                Ok(result)
            }
            FloorDiv { left, right }
            | Eq { left, right }
            | Ne { left, right }
            | Gt { left, right }
            | Ge { left, right }
            | Lt { left, right }
            | Le { left, right } => {
                check(
                    same_dimensions(
                        &left.infer_dimensions(symbols, functions)?,
                        &right.infer_dimensions(symbols, functions)?,
                    ) || left.is_zero_literal()?
                        || right.is_zero_literal()?,
                    "comparison/floor_div operands must have identical dimensions",
                )?;
                Ok(DIMENSIONLESS)
            }
            And { left, right } | Or { left, right } => {
                check(
                    same_dimensions(&left.infer_dimensions(symbols, functions)?, &DIMENSIONLESS)
                        && same_dimensions(
                            &right.infer_dimensions(symbols, functions)?,
                            &DIMENSIONLESS,
                        ),
                    "boolean operands must be dimensionless",
                )?;
                Ok(DIMENSIONLESS)
            }
            Clip { value, min, max } => {
                let value = value.infer_dimensions(symbols, functions)?;
                check(
                    (min.is_zero_literal()?
                        || same_dimensions(&value, &min.infer_dimensions(symbols, functions)?))
                        && (max.is_zero_literal()?
                            || same_dimensions(&value, &max.infer_dimensions(symbols, functions)?)),
                    "clip operands must have identical dimensions",
                )?;
                Ok(value)
            }
        }
    }

    fn left_is_zero(&self) -> Result<bool> {
        match self {
            Self::Add { left, .. } | Self::Sub { left, .. } | Self::Mod { left, .. } => {
                left.is_zero_literal()
            }
            _ => Ok(false),
        }
    }

    fn right_is_zero(&self) -> Result<bool> {
        match self {
            Self::Add { right, .. } | Self::Sub { right, .. } | Self::Mod { right, .. } => {
                right.is_zero_literal()
            }
            _ => Ok(false),
        }
    }
}

impl FunctionDefinition {
    fn validate(&self) -> Result<()> {
        check(
            valid_name(&self.name)
                && self.semantic_version == "1.0.0"
                && self.abi == "b2ir-function-v1"
                && self.arguments.len() <= 32
                && !self.effects.stateful
                && self.effects.deterministic
                && self.effects.thread_safe
                && !self.effects.rng
                && self.body.as_ref().is_none_or(|body| !body.has_random()),
            "invalid Function contract",
        )?;
        let mut symbols = BTreeMap::new();
        let mut dimensions = BTreeMap::new();
        for argument in &self.arguments {
            check(
                valid_name(&argument.name)
                    && matches!(argument.dtype, DType::F64 | DType::I64 | DType::Bool)
                    && valid_dimensions(&argument.dimensions)
                    && symbols
                        .insert(argument.name.clone(), argument.dtype)
                        .is_none()
                    && dimensions
                        .insert(argument.name.clone(), argument.dimensions)
                        .is_none(),
                "invalid or duplicate Function argument",
            )?;
        }
        check(
            valid_dimensions(&self.return_dimensions)
                && matches!(self.return_dtype, DType::F64 | DType::I64 | DType::Bool)
                && (!matches!(self.return_dtype, DType::I64 | DType::Bool)
                    || same_dimensions(&self.return_dimensions, &DIMENSIONLESS)),
            "invalid Function return type or dimensions",
        )?;
        if let Some(body) = &self.body {
            let mut reads = BTreeSet::new();
            let no_functions = BTreeMap::new();
            check(
                body.infer(&symbols, &mut reads, &no_functions)? == self.return_dtype
                    && same_dimensions(
                        &body.infer_dimensions(&dimensions, &no_functions)?,
                        &self.return_dimensions,
                    )
                    && reads.is_subset(&symbols.keys().cloned().collect()),
                "Function body does not satisfy its contract",
            )?;
            let canonical_body = serde_json::to_value(body)?;
            let body_hash = canonical_hash(&canonical_body)?;
            check(
                self.implementations.len() == 1
                    && self.implementations.get("b2ir-expression-v1") == Some(&body_hash),
                "invalid portable Function implementation hash",
            )?;
        } else {
            check(
                self.implementations.is_empty(),
                "Function without a portable body has a portable implementation hash",
            )?;
        }
        for (backend, native) in &self.backend_implementations {
            let expected_abi = match backend.as_str() {
                "cpu" => "b2ir-c-abi-v1",
                "cuda" => "b2ir-cuda-device-v1",
                "metal" => "b2ir-metal-v1",
                "wgsl" => "b2ir-wgsl-v1",
                _ => "",
            };
            check(
                !expected_abi.is_empty()
                    && native.abi == expected_abi
                    && valid_name(&native.symbol)
                    && !native.source.trim().is_empty()
                    && native.source.len() <= 1_048_576
                    && native.source_sha256
                        == format!("{:x}", Sha256::digest(native.source.as_bytes())),
                "invalid native Function implementation",
            )?;
        }
        check(
            self.body.is_some() || !self.backend_implementations.is_empty(),
            "Function has no executable implementation",
        )
    }
}

impl CodeObjectSpec {
    fn validate(
        &self,
        functions: &BTreeMap<String, FunctionDefinition>,
        validation: &ValidationSymbols,
        writable: &BTreeSet<String>,
        conditional_writes: &BTreeMap<String, String>,
    ) -> Result<()> {
        check(
            self.scalar.len() + self.vector.len() <= 128,
            "too many statements",
        )?;
        check(
            self.kind == "run_regularly"
                || self
                    .scalar
                    .iter()
                    .all(|statement| !statement.value.has_random()),
            "scalar random draws require run_regularly",
        )?;
        let mut symbols = validation.scalar.clone();
        let mut symbol_dimensions = validation.scalar_dimensions.clone();
        let mut reads = BTreeSet::new();
        let mut writes = BTreeSet::new();
        // A fresh guarded compiler temporary has no meaningful value on lanes
        // where its guard is false. Track that fact until an unconditional
        // assignment initializes every lane, and reject any differently
        // guarded (or unguarded) use.
        let mut conditional_locals = BTreeMap::<String, String>::new();
        for (scalar, statements) in [(true, &self.scalar), (false, &self.vector)] {
            if !scalar {
                symbols.extend(validation.inputs.clone());
                symbol_dimensions.extend(validation.dimensions.clone());
            }
            for statement in statements {
                let target = &statement.target;
                check(valid_name(target), "invalid assignment target")?;
                if validation.inputs.contains_key(target) {
                    if let Some(condition) = conditional_writes.get(target) {
                        check(
                            statement.condition.as_ref() == Some(condition),
                            "conditional write does not match refractory definition",
                        )?;
                    } else {
                        check(
                            statement.condition.as_ref().is_none_or(|condition| {
                                condition.starts_with("b2_subgroup_mask_")
                            }),
                            "unexpected conditional model-state write",
                        )?;
                    }
                }
                if let Some(condition) = &statement.condition {
                    check(
                        symbols.get(condition) == Some(&DType::Bool),
                        "conditional write requires a bool input",
                    )?;
                    check(
                        !conditional_locals.contains_key(condition),
                        "conditional write guard is not initialized on every lane",
                    )?;
                    reads.insert(condition.clone());
                }
                let mut loads = BTreeSet::new();
                statement.value.loads(&mut loads);
                for name in loads {
                    if let Some(condition) = conditional_locals.get(&name) {
                        check(
                            statement.condition.as_ref() == Some(condition),
                            "conditionally initialized local used without matching guard",
                        )?;
                    }
                }
                let target_was_defined = symbols.contains_key(target);
                let target_was_conditional = conditional_locals.contains_key(target);
                let inferred = statement.value.infer(&symbols, &mut reads, functions)?;
                check(inferred == statement.dtype, "statement dtype mismatch")?;
                let inferred_dimensions = statement
                    .value
                    .infer_dimensions(&symbol_dimensions, functions)?;
                if !(valid_dimensions(&statement.dimensions)
                    && (same_dimensions(&inferred_dimensions, &statement.dimensions)
                        || statement.value.is_zero_literal()?)
                    && (statement.dtype != DType::Bool
                        || same_dimensions(&statement.dimensions, &DIMENSIONLESS)))
                {
                    return Err(format!(
                        "statement dimensions do not match expression in {} ({}) for {target}: declared {:?}, inferred {:?}",
                        self.name, self.kind,
                        statement.dimensions, inferred_dimensions
                    )
                    .into());
                }
                if validation.inputs.contains_key(target) {
                    let scalar_broadcast = scalar && self.kind == "run_regularly";
                    check(
                        (scalar_broadcast || (!scalar && self.kind != "threshold"))
                            && writable.contains(target),
                        "read-only or out-of-domain assignment",
                    )?;
                    writes.insert(target.clone());
                }
                if let Some(previous) = symbols.get(target) {
                    check(*previous == inferred, "assignment changes symbol dtype")?;
                }
                if let Some(previous) = symbol_dimensions.get(target) {
                    check(
                        same_dimensions(previous, &inferred_dimensions)
                            || statement.value.is_zero_literal()?,
                        "assignment changes symbol dimensions",
                    )?;
                }
                // A local cannot shadow a neuron input in the scalar block.
                check(
                    !scalar
                        || !validation.inputs.contains_key(target)
                        || self.kind == "run_regularly",
                    "scalar write to model symbol requires run_regularly",
                )?;
                symbols.insert(target.clone(), inferred);
                symbol_dimensions.insert(target.clone(), inferred_dimensions);
                if !validation.inputs.contains_key(target) {
                    if let Some(condition) = &statement.condition {
                        if !target_was_defined || target_was_conditional {
                            conditional_locals.insert(target.clone(), condition.clone());
                        }
                    } else {
                        conditional_locals.remove(target);
                    }
                }
            }
        }
        if self.kind == "threshold" {
            check(
                symbols.get("_cond") == Some(&DType::Bool),
                "threshold must define bool _cond",
            )?;
        }
        if self.kind == "synapse_state_update" {
            check(
                &writes == writable,
                "synapse state update must write every clock-driven ODE state",
            )?;
        }
        if self.kind == "summed_variable" {
            check(
                symbols.get("_synaptic_var") == Some(&DType::F64) && writes.is_empty(),
                "summed variable must produce one read-only f64 _synaptic_var",
            )?;
        }
        let expected_writes = if let Some(adaptive) = &self.adaptive {
            check(
                self.kind == "state_update"
                    && matches!(
                        adaptive.integrator.as_str(),
                        "rk2" | "rk4" | "rkf45" | "rkck" | "rk8pd"
                    )
                    && !adaptive.states.is_empty()
                    && adaptive.states.len() == adaptive.derivatives.len()
                    && adaptive.states.len() == adaptive.absolute_errors.len()
                    && adaptive.max_steps > 0
                    && adaptive.use_last_timestep == adaptive.last_timestep.is_some(),
                "invalid adaptive integrator definition",
            )?;
            let state_set: BTreeSet<_> = adaptive.states.iter().cloned().collect();
            let derivative_set: BTreeSet<_> = adaptive.derivatives.iter().cloned().collect();
            check(
                state_set.len() == adaptive.states.len()
                    && derivative_set.len() == adaptive.derivatives.len()
                    && adaptive
                        .absolute_errors
                        .iter()
                        .all(|value| decode_bits(value).is_ok_and(|v| v.is_finite() && v > 0.0))
                    && adaptive.states.iter().all(|name| {
                        matches!(validation.inputs.get(name), Some(DType::F32 | DType::F64))
                            && writable.contains(name)
                    })
                    && adaptive.derivatives.iter().all(|name| {
                        symbols
                            .get(name)
                            .is_some_and(|dtype| matches!(dtype, DType::F32 | DType::F64))
                            && !validation.inputs.contains_key(name)
                    })
                    && adaptive
                        .frozen_states
                        .iter()
                        .all(|name| state_set.contains(name)),
                "invalid adaptive states, derivatives, or error scales",
            )?;
            let mut expected = state_set;
            for name in [
                adaptive.last_timestep.as_ref(),
                adaptive.failed_steps.as_ref(),
                adaptive.step_count.as_ref(),
            ]
            .into_iter()
            .flatten()
            {
                check(
                    writable.contains(name) && validation.inputs.contains_key(name),
                    "adaptive diagnostic state is unavailable",
                )?;
                expected.insert(name.clone());
            }
            check(
                writes.is_empty(),
                "adaptive derivative program cannot write state",
            )?;
            expected
        } else {
            writes
        };
        let expected_reads: Vec<_> = reads
            .into_iter()
            .filter(|name| validation.inputs.contains_key(name))
            .collect();
        check(
            self.effects.reads == expected_reads
                && self.effects.writes == expected_writes.into_iter().collect::<Vec<_>>(),
            "declared effects do not match code object",
        )
    }
}

struct ValidationSymbols {
    inputs: BTreeMap<String, DType>,
    scalar: BTreeMap<String, DType>,
    dimensions: BTreeMap<String, Dimensions>,
    scalar_dimensions: BTreeMap<String, Dimensions>,
}

#[derive(Clone, Copy)]
struct PopulationSchemaRef<'a> {
    dtypes: &'a BTreeMap<String, DType>,
    dimensions: &'a BTreeMap<String, Dimensions>,
}

impl SynapseDefinition {
    fn symbols(
        &self,
        instance: &SynapseInstance,
        population_states: (&BTreeSet<String>, &BTreeSet<String>),
        population_schemas: (PopulationSchemaRef<'_>, PopulationSchemaRef<'_>),
        all_populations: &[PopulationDefinition],
        all_population_schemas: &[PopulationSchemaRef<'_>],
        population_clocks: ((f64, usize), (f64, usize)),
        total_values: &mut usize,
        target_has_refractory: bool,
    ) -> Result<ValidationSymbols> {
        let (source_states, target_states) = population_states;
        let (source_schema, target_schema) = population_schemas;
        let source_dtypes = source_schema.dtypes;
        let target_dtypes = target_schema.dtypes;
        let source_dimensions = source_schema.dimensions;
        let target_dimensions = target_schema.dimensions;
        let ((source_dt, source_steps), (target_dt, _target_steps)) = population_clocks;
        let edges = instance.edge_count();
        let topology_valid = match &instance.topology {
            TopologyInstance::Explicit => {
                edges <= resource_limits::explicit_synapses()?
                    && instance.target.len() == edges
                    && instance.source.iter().all(|i| *i < self.source_count)
                    && instance.target.iter().all(|i| *i < self.target_count)
                    && instance
                        .pathways
                        .iter()
                        .all(|pathway| pathway.delay_initializer.is_none())
            }
            TopologyInstance::BinaryCsr {
                edge_count,
                path,
                sha256,
                column_count,
                initializers,
            } => {
                let names: BTreeSet<_> = self
                    .parameters
                    .iter()
                    .filter(|p| p.index_domain == IndexDomain::Synapse)
                    .map(|p| p.name.clone())
                    .collect();
                check(
                    instance.source.is_empty()
                        && instance.target.is_empty()
                        && self.states.is_empty()
                        && self
                            .parameters
                            .iter()
                            .filter(|p| p.index_domain == IndexDomain::Synapse)
                            .all(|p| p.dtype == DType::F64)
                        && names == initializers.keys().cloned().collect()
                        && initializers.values().all(|c| *c < *column_count)
                        && initializers.values().copied().collect::<BTreeSet<_>>()
                            == (0..*column_count).collect()
                        && sha256.len() == 64
                        && sha256.bytes().all(|c| c.is_ascii_hexdigit())
                        && instance.pathways.iter().all(|p| {
                            p.kind == "pre" && p.delay_initializer.is_none() && p.delay.len() == 1
                        }),
                    "invalid binary CSR mapping/pathway",
                )?;
                binary_topology::read_binary_csr(
                    Path::new(path),
                    self.source_count,
                    self.target_count,
                    *edge_count,
                    *column_count,
                    sha256,
                    false,
                )?;
                true
            }
            TopologyInstance::FixedTotal {
                edge_count,
                initializers,
                ..
            }
            | TopologyInstance::FixedIndegree {
                edge_count,
                initializers,
                ..
            } => {
                let parameter_initializers: BTreeSet<_> = self
                    .parameters
                    .iter()
                    .filter(|parameter| parameter.index_domain == IndexDomain::Synapse)
                    .map(|parameter| parameter.name.clone())
                    .collect();
                *edge_count > 0
                    && *edge_count <= u32::MAX as usize
                    && instance.source.is_empty()
                    && instance.target.is_empty()
                    && self.states.is_empty()
                    && initializers.keys().cloned().collect::<BTreeSet<_>>()
                        == parameter_initializers
                    && initializers
                        .values()
                        .all(|initializer| initializer.values().is_ok())
                    && instance.pathways.iter().all(|pathway| {
                        pathway.delay_initializer.is_some() || pathway.delay.len() == 1
                    })
            }
        };
        if let TopologyInstance::FixedIndegree {
            edge_count,
            indegree,
            ..
        } = &instance.topology
        {
            check(
                *indegree > 0
                    && *indegree <= self.source_count
                    && self.target_count.checked_mul(*indegree) == Some(*edge_count),
                "invalid fixed-indegree topology",
            )?;
        }
        check(
            topology_valid && self.source_count > 0 && self.target_count > 0,
            "invalid synaptic topology",
        )?;
        let all_edge_work = self.code_objects.iter().any(|code| {
            matches!(
                code.kind.as_str(),
                "synapse_subexpression_update"
                    | "synapse_state_update"
                    | "summed_variable"
                    | "synapse_run_regularly"
            )
        });
        if all_edge_work {
            synapse_budget(edges, source_steps)?;
        }
        let mut pathway_names = BTreeSet::new();
        for pathway in &instance.pathways {
            check(
                valid_name(&pathway.name)
                    && pathway_names.insert(pathway.name.clone())
                    && matches!(pathway.kind.as_str(), "pre" | "post")
                    && valid_name(&pathway.event)
                    && pathway.delay.len() == pathway.delay_ticks.len()
                    && (pathway.delay_initializer.is_some()
                        || pathway.delay.len() == 1
                        || pathway.delay.len() == edges),
                "invalid synaptic pathway instance",
            )?;
            let dt = if pathway.kind == "pre" {
                source_dt
            } else {
                target_dt
            };
            if let Some(initializer) = &pathway.delay_initializer {
                check(
                    pathway.delay.is_empty() && pathway.delay_ticks.is_empty(),
                    "procedural delay cannot include materialized values",
                )?;
                let (minimum, maximum) = initializer.bounds()?;
                check(
                    minimum.is_none_or(|value| value >= 0.0)
                        && maximum.is_none_or(|value| value / dt < 1_000_001.0),
                    "invalid procedural delay bounds",
                )?;
            }
            for (encoded, &delay_ticks) in pathway.delay.iter().zip(&pathway.delay_ticks) {
                let delay = decode_bits(encoded)?;
                let ticks = (delay / dt + 0.5).floor();
                check(
                    delay >= 0.0
                        && ticks.is_finite()
                        && ticks <= 1_000_000.0
                        && delay_ticks <= 1_000_000
                        && ticks as usize == delay_ticks,
                    "invalid synaptic pathway delay",
                )?;
            }
        }
        let expected_pre = source_states
            .iter()
            .map(|s| (format!("{s}_pre"), s.clone()))
            .collect();
        let expected_post = target_states
            .iter()
            .flat_map(|s| [(s.clone(), s.clone()), (format!("{s}_post"), s.clone())])
            .collect();
        check(
            self.pre_state_aliases == expected_pre && self.post_state_aliases == expected_post,
            "invalid synaptic state aliases",
        )?;
        let mut inputs = BTreeMap::from([
            ("dt".to_owned(), DType::F64),
            ("t".to_owned(), DType::F64),
            ("i".to_owned(), DType::Index),
            ("j".to_owned(), DType::Index),
            ("N".to_owned(), DType::Index),
            ("N_pre".to_owned(), DType::Index),
            ("N_post".to_owned(), DType::Index),
        ]);
        let mut scalar = inputs.clone();
        scalar.remove("i");
        scalar.remove("j");
        let mut input_dimensions: BTreeMap<_, _> = [
            ("dt", TIME_DIMENSIONS),
            ("t", TIME_DIMENSIONS),
            ("i", DIMENSIONLESS),
            ("j", DIMENSIONLESS),
            ("N", DIMENSIONLESS),
            ("N_pre", DIMENSIONLESS),
            ("N_post", DIMENSIONLESS),
        ]
        .into_iter()
        .map(|(name, dimensions)| (name.to_owned(), dimensions))
        .collect();
        let mut scalar_dimensions = input_dimensions.clone();
        scalar_dimensions.remove("i");
        scalar_dimensions.remove("j");
        if target_has_refractory {
            inputs.insert("not_refractory_post".to_owned(), DType::Bool);
            input_dimensions.insert("not_refractory_post".to_owned(), DIMENSIONLESS);
        }
        for (alias, state) in &self.pre_state_aliases {
            check(
                valid_name(alias) && inputs.insert(alias.clone(), source_dtypes[state]).is_none(),
                "colliding synaptic alias",
            )?;
            input_dimensions.insert(alias.clone(), source_dimensions[state]);
        }
        for (alias, state) in &self.post_state_aliases {
            check(
                valid_name(alias) && inputs.insert(alias.clone(), target_dtypes[state]).is_none(),
                "colliding synaptic alias",
            )?;
            input_dimensions.insert(alias.clone(), target_dimensions[state]);
        }
        check(
            self.states.len() <= 32 && self.parameters.len() <= 128,
            "too many synaptic states or parameters",
        )?;
        let state_names: BTreeSet<_> = self.states.iter().map(|s| s.name.clone()).collect();
        check(
            state_names.len() == self.states.len()
                && state_names == instance.initial_state.keys().cloned().collect(),
            "synaptic state arrays do not match definition",
        )?;
        for symbol in &self.states {
            check(
                valid_name(&symbol.name)
                    && symbol.name != "_cond"
                    && matches!(
                        symbol.dtype,
                        DType::F32
                            | DType::F64
                            | DType::I32
                            | DType::I64
                            | DType::U32
                            | DType::U64
                            | DType::Bool
                    )
                    && valid_dimensions(&symbol.dimensions)
                    && symbol.index_domain == IndexDomain::Synapse
                    && inputs.insert(symbol.name.clone(), symbol.dtype).is_none()
                    && input_dimensions
                        .insert(symbol.name.clone(), symbol.dimensions)
                        .is_none(),
                "invalid synaptic state",
            )?;
            let values = &instance.initial_state[&symbol.name];
            check(values.len() == edges, "synaptic state array shape mismatch")?;
            for value in values.unique_iter() {
                decode_typed_float(value, symbol.dtype)?;
            }
            *total_values += values.len();
        }
        let names: BTreeSet<_> = self.parameters.iter().map(|s| s.name.clone()).collect();
        check(
            names.len() == self.parameters.len()
                && names == instance.parameters.keys().cloned().collect(),
            "synaptic arrays do not match definition",
        )?;
        let topology_values = if matches!(&instance.topology, TopologyInstance::Explicit) {
            2usize
                .checked_mul(edges)
                .ok_or("topology value count overflow")?
        } else {
            0
        };
        *total_values += topology_values
            + instance
                .pathways
                .iter()
                .map(|pathway| pathway.delay.len() + pathway.delay_ticks.len())
                .sum::<usize>();
        for symbol in &self.parameters {
            check(
                valid_name(&symbol.name)
                    && symbol.name != "_cond"
                    && matches!(
                        symbol.dtype,
                        DType::F32
                            | DType::F64
                            | DType::I32
                            | DType::I64
                            | DType::U32
                            | DType::U64
                            | DType::Bool
                    )
                    && valid_dimensions(&symbol.dimensions)
                    && matches!(
                        symbol.index_domain,
                        IndexDomain::Scalar | IndexDomain::Synapse
                    )
                    && inputs.insert(symbol.name.clone(), symbol.dtype).is_none()
                    && input_dimensions
                        .insert(symbol.name.clone(), symbol.dimensions)
                        .is_none(),
                "invalid synaptic parameter",
            )?;
            let values = &instance.parameters[&symbol.name];
            let generated = match &instance.topology {
                TopologyInstance::FixedTotal { initializers, .. }
                | TopologyInstance::FixedIndegree { initializers, .. } => {
                    initializers.contains_key(&symbol.name)
                }
                TopologyInstance::BinaryCsr { initializers, .. } => {
                    initializers.contains_key(&symbol.name)
                }
                TopologyInstance::Explicit => false,
            };
            let expected = if generated {
                0
            } else if symbol.index_domain == IndexDomain::Scalar {
                1
            } else {
                edges
            };
            check(values.len() == expected, "synaptic array shape mismatch")?;
            for value in values.unique_iter() {
                decode_typed_float(value, symbol.dtype)?;
            }
            *total_values += values.len();
            if symbol.index_domain == IndexDomain::Scalar {
                scalar.insert(symbol.name.clone(), symbol.dtype);
                scalar_dimensions.insert(symbol.name.clone(), symbol.dimensions);
            }
        }
        let linked_names: BTreeSet<_> = self
            .linked_variables
            .iter()
            .map(|linked| linked.name.clone())
            .collect();
        check(
            self.linked_variables.len() <= 128
                && linked_names.len() == self.linked_variables.len(),
            "invalid synapse linked-variable table",
        )?;
        for linked in &self.linked_variables {
            check(
                valid_name(&linked.name)
                    && linked.name != "_cond"
                    && linked.source_population < all_populations.len()
                    && matches!(
                        linked.dtype,
                        DType::F32
                            | DType::F64
                            | DType::I32
                            | DType::I64
                            | DType::U32
                            | DType::U64
                            | DType::Bool
                    )
                    && valid_dimensions(&linked.dimensions)
                    && inputs.insert(linked.name.clone(), linked.dtype).is_none()
                    && input_dimensions
                        .insert(linked.name.clone(), linked.dimensions)
                        .is_none(),
                "invalid synapse linked variable",
            )?;
            let source = &all_populations[linked.source_population];
            let source_symbol = source
                .states
                .iter()
                .find(|symbol| symbol.name == linked.source_state)
                .ok_or("synapse linked source must be mutable population state")?;
            let source_schema = all_population_schemas[linked.source_population];
            check(
                source_schema.dtypes.get(&linked.source_state) == Some(&linked.dtype)
                    && source_symbol.dtype == linked.dtype
                    && same_dimensions(&source_symbol.dimensions, &linked.dimensions),
                "synapse linked variable dtype or dimensions differ from source",
            )?;
            let LinkedIndexDefinition::Constant { values } = &linked.index else {
                return Err("synapse linked variables require constant mappings".into());
            };
            check(
                values.len() == edges && values.iter().all(|index| *index < source.count),
                "synapse linked mapping is out of source bounds",
            )?;
            *total_values += values.len();
        }
        check(*total_values <= resource_limits::initial_values()?, "probe array budget exceeded")?;
        Ok(ValidationSymbols {
            inputs,
            scalar,
            dimensions: input_dimensions,
            scalar_dimensions,
        })
    }
}

impl Model {
    // The frozen v1 wire graph only records explicit CodeObject accesses and
    // event bindings. Complete it after integrity/semantic validation, without
    // changing the canonical input identity. All execution-side consumers must
    // see the implicit refractory state machine as well.
    fn complete_execution_effects(&mut self) -> Result<()> {
        let effects = self
            .definition
            .schedule
            .nodes
            .iter()
            .map(|node| {
                let (mut reads, mut writes) = self.schedule_node_effects(node)?;
                if node.operation == ScheduleOperation::CodeObject
                    && node.owner_kind == ScheduleOwnerKind::Population
                {
                    let population = &self.definition.populations[node.owner_index];
                    let code = &population.code_objects[node.item_index];
                    if let Some(refractory) = &population.refractory {
                        let lastspike =
                            format!("population/{}/refractory/lastspike", node.owner_index);
                        let available =
                            format!("population/{}/refractory/not_refractory", node.owner_index);
                        if code.kind == "threshold" && code.event_name.as_deref() == Some("spike") {
                            reads.insert(available.clone());
                            writes.extend([lastspike, available]);
                        } else if code.kind == "state_update" && refractory.mode == "fixed" {
                            reads.insert(lastspike);
                            writes.insert(available);
                        }
                    }
                }
                Ok((reads, writes))
            })
            .collect::<Result<Vec<_>>>()?;
        let mut last_writer = BTreeMap::<String, String>::new();
        let mut readers = BTreeMap::<String, BTreeSet<String>>::new();
        for (node, (reads, writes)) in self.definition.schedule.nodes.iter_mut().zip(effects) {
            let mut dependencies = BTreeSet::new();
            for resource in reads.union(&writes) {
                if let Some(writer) = last_writer.get(resource) {
                    dependencies.insert(writer.clone());
                }
            }
            for resource in &writes {
                dependencies.extend(readers.get(resource).into_iter().flatten().cloned());
                last_writer.insert(resource.clone(), node.id.clone());
                readers.insert(resource.clone(), BTreeSet::new());
            }
            for resource in &reads {
                readers
                    .entry(resource.clone())
                    .or_default()
                    .insert(node.id.clone());
            }
            node.effects.reads = reads.into_iter().collect();
            node.effects.writes = writes.into_iter().collect();
            node.dependencies = dependencies.into_iter().collect();
        }
        Ok(())
    }

    fn schedule_node_effects(
        &self,
        node: &ScheduleNode,
    ) -> Result<(BTreeSet<String>, BTreeSet<String>)> {
        let population_resource = |population: usize, category: &str, name: &str| {
            format!("population/{population}/{category}/{name}")
        };
        let event_resource =
            |population: usize, event: &str| population_resource(population, "event", event);
        let mut reads = BTreeSet::new();
        let mut writes = BTreeSet::new();
        match (node.operation, node.owner_kind) {
            (ScheduleOperation::CodeObject, ScheduleOwnerKind::Population) => {
                let population = &self.definition.populations[node.owner_index];
                let code = &population.code_objects[node.item_index];
                let resource = |name: &str| {
                    if population.states.iter().any(|item| item.name == name) {
                        Some(population_resource(node.owner_index, "state", name))
                    } else if population.parameters.iter().any(|item| item.name == name) {
                        Some(population_resource(node.owner_index, "parameter", name))
                    } else if let Some(linked) = population
                        .linked_variables
                        .iter()
                        .find(|linked| linked.name == name)
                    {
                        Some(population_resource(
                            linked.source_population,
                            "state",
                            &linked.source_state,
                        ))
                    } else if matches!(name, "lastspike" | "not_refractory") {
                        Some(population_resource(node.owner_index, "refractory", name))
                    } else {
                        None
                    }
                };
                reads.extend(code.effects.reads.iter().filter_map(|name| resource(name)));
                for name in &code.effects.reads {
                    if let Some(linked) = population
                        .linked_variables
                        .iter()
                        .find(|linked| linked.name == *name)
                    {
                        match &linked.index {
                            LinkedIndexDefinition::State { name } => {
                                reads.insert(population_resource(node.owner_index, "state", name));
                            }
                            LinkedIndexDefinition::Parameter { name } => {
                                reads.insert(population_resource(
                                    node.owner_index,
                                    "parameter",
                                    name,
                                ));
                            }
                            _ => {}
                        }
                    }
                }
                writes.extend(code.effects.writes.iter().filter_map(|name| resource(name)));
                if code.kind == "threshold" {
                    writes.insert(event_resource(
                        node.owner_index,
                        code.event_name
                            .as_deref()
                            .ok_or("threshold event missing")?,
                    ));
                } else if code.kind == "reset" {
                    reads.insert(event_resource(
                        node.owner_index,
                        code.event_name.as_deref().ok_or("reset event missing")?,
                    ));
                } else if code.kind == "spatial_state_update" {
                    let spatial = population
                        .spatial
                        .as_ref()
                        .ok_or("spatial update missing morphology definition")?;
                    for name in [
                        &spatial.voltage,
                        &spatial.capacitance,
                        &spatial.resistivity,
                        &spatial.area,
                        &spatial.r_length_1,
                        &spatial.r_length_2,
                    ] {
                        reads.insert(resource(name).ok_or("spatial input is not stored")?);
                    }
                    writes.insert(
                        resource(&spatial.voltage).ok_or("spatial voltage is not stored")?,
                    );
                    writes.insert(
                        resource(&spatial.membrane_current)
                            .ok_or("spatial membrane current is not stored")?,
                    );
                }
            }
            (ScheduleOperation::CodeObject, ScheduleOwnerKind::Synapse) => {
                let synapse = &self.definition.synapses[node.owner_index];
                let code = &synapse.code_objects[node.item_index];
                let resource = |name: &str| {
                    if synapse.states.iter().any(|item| item.name == name) {
                        Some(format!("synapse/{}/state/{name}", node.owner_index))
                    } else if synapse.parameters.iter().any(|item| item.name == name) {
                        Some(format!("synapse/{}/parameter/{name}", node.owner_index))
                    } else if let Some(state) = synapse.pre_state_aliases.get(name) {
                        Some(if let Some(source) = synapse.source_synapse {
                            format!("synapse/{source}/state/{state}")
                        } else {
                            population_resource(synapse.source_population, "state", state)
                        })
                    } else if let Some(state) = synapse.post_state_aliases.get(name) {
                        Some(if let Some(target) = synapse.target_synapse {
                            format!("synapse/{target}/state/{state}")
                        } else {
                            population_resource(synapse.target_population, "state", state)
                        })
                    } else if let Some(linked) = synapse
                        .linked_variables
                        .iter()
                        .find(|linked| linked.name == name)
                    {
                        Some(population_resource(
                            linked.source_population,
                            "state",
                            &linked.source_state,
                        ))
                    } else if name == "not_refractory_post" {
                        Some(population_resource(
                            synapse.target_population,
                            "refractory",
                            "not_refractory",
                        ))
                    } else {
                        None
                    }
                };
                reads.extend(code.effects.reads.iter().filter_map(|name| resource(name)));
                writes.extend(code.effects.writes.iter().filter_map(|name| resource(name)));
                match code.kind.as_str() {
                    "summed_variable" => {
                        let is_pre = code.summed_target.as_deref() == Some("pre");
                        let endpoint_synapse = if is_pre {
                            synapse.source_synapse
                        } else {
                            synapse.target_synapse
                        };
                        let state = code
                            .summed_state
                            .as_deref()
                            .ok_or("missing summed state")?;
                        if let Some(endpoint) = endpoint_synapse {
                            writes.insert(format!("synapse/{endpoint}/state/{state}"));
                        } else {
                            let population = if is_pre {
                                synapse.source_population
                            } else {
                                synapse.target_population
                            };
                            writes.insert(population_resource(population, "state", state));
                        }
                    }
                    "synapses" => {
                        reads.insert(event_resource(
                            synapse.source_population,
                            code.event_name.as_deref().ok_or("pathway event missing")?,
                        ));
                    }
                    "synapses_post" => {
                        reads.insert(event_resource(
                            synapse.target_population,
                            code.event_name.as_deref().ok_or("pathway event missing")?,
                        ));
                    }
                    _ => {}
                }
            }
            (ScheduleOperation::StateMonitor, ScheduleOwnerKind::Population) => {
                let population = &self.definition.populations[node.owner_index];
                let monitor = &population.state_monitors[node.item_index];
                for name in &monitor.variables {
                    if let Some(linked) = population
                        .linked_variables
                        .iter()
                        .find(|linked| linked.name == *name)
                    {
                        reads.insert(population_resource(
                            linked.source_population,
                            "state",
                            &linked.source_state,
                        ));
                        match &linked.index {
                            LinkedIndexDefinition::State { name } => {
                                reads.insert(population_resource(node.owner_index, "state", name));
                            }
                            LinkedIndexDefinition::Parameter { name } => {
                                reads.insert(population_resource(
                                    node.owner_index,
                                    "parameter",
                                    name,
                                ));
                            }
                            _ => {}
                        }
                    } else {
                        let category = if population.states.iter().any(|item| item.name == *name) {
                            "state"
                        } else {
                            "parameter"
                        };
                        reads.insert(population_resource(node.owner_index, category, name));
                    }
                }
            }
            (ScheduleOperation::StateMonitor, ScheduleOwnerKind::Synapse) => {
                let synapse = &self.definition.synapses[node.owner_index];
                let monitor = &synapse.state_monitors[node.item_index];
                for source in &monitor.sources {
                    match source {
                        SynapseMonitorSourceDefinition::SynapseState { name, .. } => {
                            reads.insert(format!(
                                "synapse/{}/state/{}", node.owner_index, name
                            ));
                        }
                        SynapseMonitorSourceDefinition::PreState { name, .. } => {
                            reads.insert(population_resource(
                                synapse.source_population, "state", name,
                            ));
                        }
                        SynapseMonitorSourceDefinition::PostState { name, .. } => {
                            reads.insert(population_resource(
                                synapse.target_population, "state", name,
                            ));
                        }
                        SynapseMonitorSourceDefinition::Linked { name, .. } => {
                            let linked = synapse
                                .linked_variables
                                .iter()
                                .find(|linked| linked.name == *name)
                                .ok_or("missing synapse linked monitor source")?;
                            reads.insert(population_resource(
                                linked.source_population,
                                "state",
                                &linked.source_state,
                            ));
                        }
                    }
                }
            }
            (ScheduleOperation::SpikeMonitor, ScheduleOwnerKind::Population) => {
                reads.insert(event_resource(node.owner_index, "spike"));
            }
            (ScheduleOperation::EventMonitor, ScheduleOwnerKind::Population) => {
                let population = &self.definition.populations[node.owner_index];
                let monitor = &population.event_monitors[node.item_index];
                reads.insert(event_resource(node.owner_index, &monitor.event));
                for name in &monitor.variables {
                    if let Some(linked) = population
                        .linked_variables
                        .iter()
                        .find(|linked| linked.name == *name)
                    {
                        reads.insert(population_resource(
                            linked.source_population,
                            "state",
                            &linked.source_state,
                        ));
                        match &linked.index {
                            LinkedIndexDefinition::State { name } => {
                                reads.insert(population_resource(node.owner_index, "state", name));
                            }
                            LinkedIndexDefinition::Parameter { name } => {
                                reads.insert(population_resource(
                                    node.owner_index,
                                    "parameter",
                                    name,
                                ));
                            }
                            _ => {}
                        }
                    } else {
                        let category = if population.states.iter().any(|item| item.name == *name) {
                            "state"
                        } else {
                            "parameter"
                        };
                        reads.insert(population_resource(node.owner_index, category, name));
                    }
                }
            }
            (ScheduleOperation::EventSource, ScheduleOwnerKind::Population) => {
                writes.insert(event_resource(node.owner_index, "spike"));
            }
            _ => return Err("invalid schedule operation/owner combination".into()),
        }
        Ok((reads, writes))
    }

    fn validate_schedule(&self) -> Result<()> {
        let schedule = &self.definition.schedule;
        let base_slots: BTreeSet<_> = schedule.base_slots.iter().cloned().collect();
        check(
            !schedule.base_slots.is_empty()
                && schedule.base_slots.len() <= 256
                && base_slots.len() == schedule.base_slots.len()
                && schedule.base_slots.iter().all(|slot| {
                    valid_name(slot) && !slot.starts_with("before_") && !slot.starts_with("after_")
                }),
            "invalid base schedule slots",
        )?;
        let expected_slots: Vec<_> = schedule
            .base_slots
            .iter()
            .flat_map(|slot| {
                [
                    format!("before_{slot}"),
                    slot.clone(),
                    format!("after_{slot}"),
                ]
            })
            .collect();
        check(
            schedule.slots == expected_slots,
            "schedule slots are not the canonical before/base/after expansion",
        )?;
        let slot_positions: BTreeMap<_, _> = schedule
            .slots
            .iter()
            .enumerate()
            .map(|(index, slot)| (slot.as_str(), index))
            .collect();
        let mut expected_ids = BTreeSet::new();
        for (population_index, population) in self.definition.populations.iter().enumerate() {
            for code_index in 0..population.code_objects.len() {
                expected_ids.insert(format!("population/{population_index}/code/{code_index}"));
            }
            for monitor_index in 0..population.state_monitors.len() {
                expected_ids.insert(format!(
                    "population/{population_index}/state_monitor/{monitor_index}"
                ));
            }
            for monitor_index in 0..population.event_monitors.len() {
                expected_ids.insert(format!(
                    "population/{population_index}/event_monitor/{monitor_index}"
                ));
            }
            if population.spike_monitor.is_some() {
                expected_ids.insert(format!("population/{population_index}/spike_monitor/0"));
            }
            if self.instance.populations[population_index]
                .spike_generator
                .is_some()
            {
                expected_ids.insert(format!("population/{population_index}/event_source/0"));
            }
        }
        for (synapse_index, synapse) in self.definition.synapses.iter().enumerate() {
            for code_index in 0..synapse.code_objects.len() {
                expected_ids.insert(format!("synapse/{synapse_index}/code/{code_index}"));
            }
            for monitor_index in 0..synapse.state_monitors.len() {
                expected_ids.insert(format!(
                    "synapse/{synapse_index}/state_monitor/{monitor_index}"
                ));
            }
        }

        let mut seen_ids = BTreeSet::new();
        let mut previous_order: Option<(usize, i32, String, String)> = None;
        let mut last_writer: BTreeMap<String, String> = BTreeMap::new();
        let mut readers: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        for node in &schedule.nodes {
            let slot = *slot_positions
                .get(node.when.as_str())
                .ok_or("schedule node references an unknown slot")?;
            let order = (slot, node.order, node.name.clone(), node.id.clone());
            check(
                previous_order
                    .as_ref()
                    .is_none_or(|previous| previous < &order),
                "schedule nodes are not in canonical (slot, order, name, id) order",
            )?;
            previous_order = Some(order);
            check(
                !node.id.is_empty()
                    && node.id.len() <= 256
                    && valid_name(&node.name)
                    && seen_ids.insert(node.id.clone())
                    && node.clock < self.definition.clocks.len(),
                "invalid schedule node identity",
            )?;
            check(
                node.effects.reads.windows(2).all(|pair| pair[0] < pair[1])
                    && node.effects.writes.windows(2).all(|pair| pair[0] < pair[1])
                    && node.dependencies.windows(2).all(|pair| pair[0] < pair[1])
                    && node
                        .effects
                        .reads
                        .iter()
                        .chain(&node.effects.writes)
                        .all(|resource| !resource.is_empty() && resource.len() <= 512),
                "schedule effects must be sorted, unique and bounded",
            )?;

            let expected_id = match (node.operation, node.owner_kind) {
                (ScheduleOperation::CodeObject, ScheduleOwnerKind::Population) => {
                    let population = self
                        .definition
                        .populations
                        .get(node.owner_index)
                        .ok_or("schedule population owner out of range")?;
                    let code = population
                        .code_objects
                        .get(node.item_index)
                        .ok_or("schedule population code object out of range")?;
                    check(
                        node.clock == code.clock
                            && node.when == code.when
                            && node.order == code.order
                            && node.name == code.name,
                        "schedule population code metadata mismatch",
                    )?;
                    format!("population/{}/code/{}", node.owner_index, node.item_index)
                }
                (ScheduleOperation::CodeObject, ScheduleOwnerKind::Synapse) => {
                    let synapse = self
                        .definition
                        .synapses
                        .get(node.owner_index)
                        .ok_or("schedule synapse owner out of range")?;
                    let code = synapse
                        .code_objects
                        .get(node.item_index)
                        .ok_or("schedule synapse code object out of range")?;
                    check(
                        node.clock == code.clock
                            && node.when == code.when
                            && node.order == code.order
                            && node.name == code.name,
                        "schedule synapse code metadata mismatch",
                    )?;
                    format!("synapse/{}/code/{}", node.owner_index, node.item_index)
                }
                (ScheduleOperation::StateMonitor, ScheduleOwnerKind::Population) => {
                    let population = self
                        .definition
                        .populations
                        .get(node.owner_index)
                        .ok_or("schedule StateMonitor owner out of range")?;
                    let monitor = population
                        .state_monitors
                        .get(node.item_index)
                        .ok_or("schedule StateMonitor out of range")?;
                    check(
                        node.clock == monitor.clock
                            && node.when == monitor.when
                            && node.order == monitor.order
                            && node.name == monitor.name,
                        "schedule StateMonitor metadata mismatch",
                    )?;
                    format!(
                        "population/{}/state_monitor/{}",
                        node.owner_index, node.item_index
                    )
                }
                (ScheduleOperation::StateMonitor, ScheduleOwnerKind::Synapse) => {
                    let synapse = self
                        .definition
                        .synapses
                        .get(node.owner_index)
                        .ok_or("schedule synapse StateMonitor owner out of range")?;
                    let monitor = synapse
                        .state_monitors
                        .get(node.item_index)
                        .ok_or("schedule synapse StateMonitor out of range")?;
                    check(
                        node.clock == monitor.clock
                            && node.when == monitor.when
                            && node.order == monitor.order
                            && node.name == monitor.name,
                        "schedule synapse StateMonitor metadata mismatch",
                    )?;
                    format!(
                        "synapse/{}/state_monitor/{}",
                        node.owner_index, node.item_index
                    )
                }
                (ScheduleOperation::SpikeMonitor, ScheduleOwnerKind::Population) => {
                    let population = self
                        .definition
                        .populations
                        .get(node.owner_index)
                        .ok_or("schedule SpikeMonitor owner out of range")?;
                    let monitor = population
                        .spike_monitor
                        .as_ref()
                        .filter(|_| node.item_index == 0)
                        .ok_or("schedule SpikeMonitor out of range")?;
                    check(
                        node.clock == population.clock
                            && node.when == "thresholds"
                            && node.order == 1
                            && node.name == *monitor,
                        "schedule SpikeMonitor metadata mismatch",
                    )?;
                    format!("population/{}/spike_monitor/0", node.owner_index)
                }
                (ScheduleOperation::EventMonitor, ScheduleOwnerKind::Population) => {
                    let population = self
                        .definition
                        .populations
                        .get(node.owner_index)
                        .ok_or("schedule EventMonitor owner out of range")?;
                    let monitor = population
                        .event_monitors
                        .get(node.item_index)
                        .ok_or("schedule EventMonitor out of range")?;
                    check(
                        node.clock == monitor.clock
                            && node.when == monitor.when
                            && node.order == monitor.order
                            && node.name == monitor.name,
                        "schedule EventMonitor metadata mismatch",
                    )?;
                    format!(
                        "population/{}/event_monitor/{}",
                        node.owner_index, node.item_index
                    )
                }
                (ScheduleOperation::EventSource, ScheduleOwnerKind::Population) => {
                    let population = self
                        .definition
                        .populations
                        .get(node.owner_index)
                        .ok_or("schedule event source owner out of range")?;
                    check(
                        node.item_index == 0
                            && self.instance.populations[node.owner_index]
                                .spike_generator
                                .is_some()
                            && node.clock == population.clock
                            && node.when == "thresholds"
                            && node.order == 0
                            && node.name == population.name,
                        "schedule event source metadata mismatch",
                    )?;
                    format!("population/{}/event_source/0", node.owner_index)
                }
                _ => return Err("invalid schedule operation/owner combination".into()),
            };
            check(
                node.id == expected_id,
                "schedule node id/reference mismatch",
            )?;
            let (expected_reads, expected_writes) = self.schedule_node_effects(node)?;
            check(
                node.effects.reads.iter().cloned().collect::<BTreeSet<_>>() == expected_reads
                    && node.effects.writes.iter().cloned().collect::<BTreeSet<_>>()
                        == expected_writes,
                "schedule node effects do not match referenced operation",
            )?;

            let mut expected_dependencies = BTreeSet::new();
            for resource in &node.effects.reads {
                if let Some(writer) = last_writer.get(resource) {
                    expected_dependencies.insert(writer.clone());
                }
            }
            for resource in &node.effects.writes {
                if let Some(writer) = last_writer.get(resource) {
                    expected_dependencies.insert(writer.clone());
                }
                expected_dependencies.extend(readers.get(resource).into_iter().flatten().cloned());
            }
            check(
                node.dependencies.iter().cloned().collect::<BTreeSet<_>>() == expected_dependencies
                    && node.dependencies.len() == expected_dependencies.len()
                    && node
                        .dependencies
                        .iter()
                        .all(|dependency| seen_ids.contains(dependency)),
                "schedule dependency graph does not match ordered effects",
            )?;
            for resource in &node.effects.writes {
                last_writer.insert(resource.clone(), node.id.clone());
                readers.insert(resource.clone(), BTreeSet::new());
            }
            for resource in &node.effects.reads {
                readers
                    .entry(resource.clone())
                    .or_default()
                    .insert(node.id.clone());
            }
        }
        check(
            seen_ids == expected_ids,
            "schedule must cover every executable object exactly once",
        )?;
        Ok(())
    }

    fn validate(&self) -> Result<()> {
        check(
            self.schema == "b2ir-v1",
            "unsupported schema; regenerate model with current exporter",
        )?;
        check(
            self.protocol.name == "b2ir"
                && self.protocol.version.major == 1
                && self.protocol.version.minor == 0
                && self.protocol.canonical_encoding == "b2ir-canonical-json-v1"
                && self.protocol.hash_algorithm == "sha256",
            "unsupported AtlasIR protocol envelope",
        )?;
        check(
            [
                &self.protocol.layers.definition,
                &self.protocol.layers.instance,
                &self.protocol.layers.run,
            ]
            .into_iter()
            .all(|hash| {
                hash.len() == 64
                    && hash
                        .bytes()
                        .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
            }),
            "invalid canonical layer hash encoding",
        )?;
        let d = &self.definition;
        check(
            d.numeric_profile == "reference-f64",
            "unsupported numeric profile",
        )?;
        check(
            d.rng_algorithm == "splitmix64-counter-v1",
            "unsupported RNG algorithm",
        )?;
        check(
            d.functions.len() <= 128
                && d.functions
                    .windows(2)
                    .all(|pair| pair[0].name < pair[1].name),
            "Function contracts must be unique, sorted and bounded",
        )?;
        let functions: BTreeMap<_, _> = d
            .functions
            .iter()
            .map(|function| {
                function.validate()?;
                Ok((function.name.clone(), function.clone()))
            })
            .collect::<Result<_>>()?;
        for backend in ["cpu", "cuda", "metal", "wgsl"] {
            let native_symbols: Vec<_> = d
                .functions
                .iter()
                .filter_map(|function| function.backend_implementations.get(backend))
                .map(|implementation| implementation.symbol.as_str())
                .collect();
            check(
                native_symbols
                    .iter()
                    .copied()
                    .collect::<BTreeSet<_>>()
                    .len()
                    == native_symbols.len(),
                "native Function entry points must be unique within each backend",
            )?;
        }
        let mut random_streams = Vec::new();
        for code in d
            .populations
            .iter()
            .flat_map(|population| &population.code_objects)
            .chain(d.synapses.iter().flat_map(|synapse| &synapse.code_objects))
        {
            for statement in code.scalar.iter().chain(&code.vector) {
                statement.value.random_streams(&mut random_streams);
            }
        }
        let unique_random_streams: BTreeSet<_> = random_streams.iter().copied().collect();
        check(
            random_streams.len() <= 1_000_000
                && unique_random_streams.len() == random_streams.len(),
            "random draw-site stream ids must be unique and bounded",
        )?;
        let start = decode_bits(&self.run.start)?;
        let duration = decode_bits(&self.run.duration)?;
        check(
            start >= 0.0 && duration >= 0.0 && (start + duration).is_finite(),
            "non-negative start and duration required",
        )?;
        check(
            !d.clocks.is_empty()
                && d.clocks.len() <= 1024
                && d.clocks.len() == self.run.clocks.len()
                && d.clocks.windows(2).all(|pair| pair[0].name < pair[1].name),
            "invalid canonical clock table",
        )?;
        let clock_dts = d
            .clocks
            .iter()
            .zip(&self.run.clocks)
            .map(|(definition, run)| -> Result<f64> {
                let dt = decode_bits(&definition.dt)?;
                let expected_start = clock_timestep(start, dt)?;
                let expected_end = clock_timestep(start + duration, dt)?;
                check(
                    valid_name(&definition.name)
                        && dt > 0.0
                        && run.steps <= 10_000_000
                        && run.start_tick == expected_start
                        && run.start_tick.checked_add(run.steps) == Some(expected_end),
                    "clock run interval does not match start/duration",
                )?;
                Ok(dt)
            })
            .collect::<Result<Vec<_>>>()?;
        check(!d.populations.is_empty(), "invalid population count")?;
        check(
            self.instance.populations.len() == d.populations.len(),
            "population definition/instance mismatch",
        )?;
        check(
            (1..=resource_limits::neurons()?).contains(&self.instance.neuron_count),
            "invalid neuron count",
        )?;

        let mut names = BTreeSet::new();
        let mut state_monitor_names = BTreeSet::new();
        let mut event_monitor_names = BTreeSet::new();
        let mut spike_monitor_names = BTreeSet::new();
        let mut rate_monitor_names = BTreeSet::new();
        let mut next_offset = 0usize;
        let mut total_values = 0usize;
        let mut population_states = Vec::new();
        let mut population_frozen = Vec::new();
        let mut population_dtypes = Vec::new();
        let mut population_dimensions = Vec::new();
        let mut population_dts = Vec::new();
        for (population, instance) in d.populations.iter().zip(&self.instance.populations) {
            check(
                valid_name(&population.name)
                    && names.insert(population.name.clone())
                    && population.offset == next_offset
                    && population.count > 0,
                "invalid population layout",
            )?;
            check(
                population.clock < d.clocks.len(),
                "invalid population clock",
            )?;
            for monitor in &population.state_monitors {
                check(
                    valid_name(&monitor.name)
                        && monitor.clock < d.clocks.len()
                        && state_monitor_names.insert(monitor.name.clone()),
                    "invalid state monitor layout",
                )?;
            }
            for monitor in &population.event_monitors {
                check(
                    valid_name(&monitor.name)
                        && event_monitor_names.insert(monitor.name.clone())
                        && monitor.clock == population.clock
                        && d.schedule.slots.contains(&monitor.when),
                    "invalid event monitor layout",
                )?;
            }
            if let Some(name) = &population.spike_monitor {
                check(
                    valid_name(name) && spike_monitor_names.insert(name.clone()),
                    "invalid spike monitor layout",
                )?;
            }
            for monitor in &population.rate_monitors {
                check(
                    valid_name(&monitor.name)
                        && rate_monitor_names.insert(monitor.name.clone())
                        && matches!(monitor.dtype, DType::F32 | DType::F64)
                        && population.spike_monitor.is_some(),
                    "invalid rate monitor layout",
                )?;
            }
            next_offset += population.count;
            let dt = decode_bits(&population.dt)?;
            let run_clock = &self.run.clocks[population.clock];
            check(
                dt > 0.0
                    && dt.to_bits() == clock_dts[population.clock].to_bits()
                    && population.steps == run_clock.steps
                    && run_clock.start_tick <= usize::MAX - population.steps
                    && population.steps <= resource_limits::population_steps()?,
                "positive population dt and valid Clock interval required",
            )?;
            check(
                population.steps <= resource_limits::population_steps()?
                    && (population.count as u128) * (population.steps as u128) <= 250_000_000_000,
                "probe population duration budget exceeded",
            )?;
            population_dts.push(dt);
            check(
                population.monitor_timed_arrays.len() <= 128
                    && population
                        .monitor_timed_arrays
                        .keys()
                        .all(|name| valid_name(name)),
                "invalid monitor TimedArray table",
            )?;
            for table in population.monitor_timed_arrays.values() {
                table.validate()?;
            }

            let mut inputs = BTreeMap::from([
                ("dt".to_owned(), DType::F64),
                ("t".to_owned(), DType::F64),
                ("i".to_owned(), DType::Index),
                ("N".to_owned(), DType::Index),
            ]);
            let mut scalar_inputs = inputs.clone();
            scalar_inputs.remove("i");
            let mut input_dimensions = BTreeMap::from([
                ("dt".to_owned(), TIME_DIMENSIONS),
                ("t".to_owned(), TIME_DIMENSIONS),
                ("i".to_owned(), DIMENSIONLESS),
                ("N".to_owned(), DIMENSIONLESS),
            ]);
            let mut scalar_input_dimensions = input_dimensions.clone();
            scalar_input_dimensions.remove("i");
            let state_names: BTreeSet<_> = population
                .states
                .iter()
                .map(|symbol| symbol.name.clone())
                .collect();
            check(
                population.states.len() <= 32
                    && state_names.len() == population.states.len()
                    && state_names == instance.initial_state.keys().cloned().collect(),
                "population state arrays do not match definition",
            )?;
            for symbol in &population.states {
                check(
                    valid_name(&symbol.name)
                        && symbol.name != "_cond"
                        && matches!(
                            symbol.dtype,
                            DType::F32
                                | DType::F64
                                | DType::I32
                                | DType::I64
                                | DType::U32
                                | DType::U64
                                | DType::Bool
                        )
                        && valid_dimensions(&symbol.dimensions)
                        && symbol.index_domain == IndexDomain::Neuron
                        && inputs.insert(symbol.name.clone(), symbol.dtype).is_none()
                        && input_dimensions
                            .insert(symbol.name.clone(), symbol.dimensions)
                            .is_none(),
                    "invalid population state",
                )?;
                let values = &instance.initial_state[&symbol.name];
                check(
                    values.len() == population.count,
                    "population state array shape mismatch",
                )?;
                for value in values.unique_iter() {
                    decode_typed_float(value, symbol.dtype)?;
                }
                total_values += values.len();
            }
            let parameter_names: BTreeSet<_> = population
                .parameters
                .iter()
                .map(|symbol| symbol.name.clone())
                .collect();
            check(
                population.parameters.len() <= 128
                    && parameter_names.len() == population.parameters.len()
                    && parameter_names == instance.parameters.keys().cloned().collect(),
                "population parameter arrays do not match definition",
            )?;
            for symbol in &population.parameters {
                check(
                    valid_name(&symbol.name)
                        && symbol.name != "_cond"
                        && matches!(
                            symbol.dtype,
                            DType::F32
                                | DType::F64
                                | DType::I32
                                | DType::I64
                                | DType::U32
                                | DType::U64
                                | DType::Bool
                        )
                        && valid_dimensions(&symbol.dimensions)
                        && matches!(
                            symbol.index_domain,
                            IndexDomain::Scalar | IndexDomain::Neuron
                        )
                        && inputs.insert(symbol.name.clone(), symbol.dtype).is_none()
                        && input_dimensions
                            .insert(symbol.name.clone(), symbol.dimensions)
                            .is_none(),
                    "invalid population parameter",
                )?;
                let expected = if symbol.index_domain == IndexDomain::Scalar {
                    1
                } else {
                    population.count
                };
                let values = &instance.parameters[&symbol.name];
                check(
                    values.len() == expected,
                    "population parameter array shape mismatch",
                )?;
                for value in values.unique_iter() {
                    decode_typed_float(value, symbol.dtype)?;
                }
                total_values += values.len();
                if symbol.index_domain == IndexDomain::Scalar {
                    scalar_inputs.insert(symbol.name.clone(), symbol.dtype);
                    scalar_input_dimensions.insert(symbol.name.clone(), symbol.dimensions);
                }
            }
            let linked_names: BTreeSet<_> = population
                .linked_variables
                .iter()
                .map(|linked| linked.name.clone())
                .collect();
            check(
                population.linked_variables.len() <= 128
                    && linked_names.len() == population.linked_variables.len(),
                "invalid linked-variable table",
            )?;
            for linked in &population.linked_variables {
                check(
                    valid_name(&linked.name)
                        && linked.name != "_cond"
                        && linked.source_population < d.populations.len()
                        && matches!(
                            linked.dtype,
                            DType::F32
                                | DType::F64
                                | DType::I32
                                | DType::I64
                                | DType::U32
                                | DType::U64
                                | DType::Bool
                        )
                        && valid_dimensions(&linked.dimensions)
                        && inputs.insert(linked.name.clone(), linked.dtype).is_none()
                        && input_dimensions
                            .insert(linked.name.clone(), linked.dimensions)
                            .is_none(),
                    "invalid linked variable",
                )?;
                let source = &d.populations[linked.source_population];
                let source_symbol = source
                    .states
                    .iter()
                    .find(|symbol| symbol.name == linked.source_state)
                    .ok_or("linked variable source must be mutable population state")?;
                check(
                    source_symbol.dtype == linked.dtype
                        && same_dimensions(&source_symbol.dimensions, &linked.dimensions),
                    "linked variable dtype or dimensions differ from source",
                )?;
                match &linked.index {
                    LinkedIndexDefinition::Identity => check(
                        population.count == source.count,
                        "identity linked mapping requires equal population sizes",
                    )?,
                    LinkedIndexDefinition::Constant { values } => {
                        check(
                            values.len() == population.count
                                && values.iter().all(|index| *index < source.count),
                            "constant linked mapping is out of source bounds",
                        )?;
                        total_values += values.len();
                    }
                    LinkedIndexDefinition::State { name } => check(
                        state_names.contains(name)
                            && population
                                .states
                                .iter()
                                .find(|symbol| &symbol.name == name)
                                .is_some_and(|symbol| symbol.dtype.is_integer()),
                        "linked index state must be a local integer state",
                    )?,
                    LinkedIndexDefinition::Parameter { name } => check(
                        parameter_names.contains(name)
                            && population
                                .parameters
                                .iter()
                                .find(|symbol| &symbol.name == name)
                                .is_some_and(|symbol| {
                                    symbol.dtype.is_integer()
                                        && symbol.index_domain == IndexDomain::Neuron
                                }),
                        "linked index parameter must be a local per-neuron integer parameter",
                    )?,
                }
            }
            if let Some(spatial) = &population.spatial {
                let sections = spatial.starts.len();
                let state = |name: &str| {
                    population.states.iter().find(|symbol| symbol.name == name)
                };
                let parameter = |name: &str| {
                    population
                        .parameters
                        .iter()
                        .find(|symbol| symbol.name == name)
                };
                check(
                    sections > 0
                        && spatial.ends.len() == sections
                        && spatial.parents.len() == sections
                        && spatial.child_slots.len() == sections
                        && spatial.children_count.len() == sections + 1
                        && spatial.children.len() % (sections + 1) == 0,
                    "invalid spatial morphology table shape",
                )?;
                let child_width = spatial.children.len() / (sections + 1);
                check(
                    child_width > 0
                        && spatial.starts[0] == 0
                        && spatial.ends[sections - 1] == population.count
                        && spatial
                            .starts
                            .iter()
                            .zip(&spatial.ends)
                            .enumerate()
                            .all(|(section, (&start, &end))| {
                                start < end
                                    && end <= population.count
                                    && (section == 0
                                        || start == spatial.ends[section - 1])
                            })
                        && spatial.parents[0] == 0
                        && spatial.children_count.iter().sum::<usize>() == sections
                        && spatial
                            .children_count
                            .iter()
                            .all(|&count| count <= child_width)
                        && (0..sections).all(|section| {
                            let parent = spatial.parents[section];
                            let slot = spatial.child_slots[section];
                            parent <= section
                                && slot < spatial.children_count[parent]
                                && spatial.children[parent * child_width + slot] == section + 1
                        }),
                    "invalid spatial morphology tree",
                )?;
                check(
                    state(&spatial.voltage).is_some_and(|symbol| {
                        symbol.dtype == DType::F64
                            && symbol.index_domain == IndexDomain::Neuron
                    })
                        && state(&spatial.membrane_current).is_some_and(|symbol| {
                            symbol.dtype == DType::F64
                                && symbol.index_domain == IndexDomain::Neuron
                        })
                        && parameter(&spatial.capacitance).is_some_and(|symbol| {
                            symbol.dtype == DType::F64
                                && symbol.index_domain == IndexDomain::Neuron
                        })
                        && parameter(&spatial.resistivity).is_some_and(|symbol| {
                            symbol.dtype == DType::F64
                                && symbol.index_domain == IndexDomain::Scalar
                        })
                        && [&spatial.area, &spatial.r_length_1, &spatial.r_length_2]
                            .iter()
                            .all(|name| {
                                parameter(name).is_some_and(|symbol| {
                                    symbol.dtype == DType::F64
                                        && symbol.index_domain == IndexDomain::Neuron
                                })
                            }),
                    "spatial solver fields require f64 neuron arrays and scalar Ri",
                )?;
            }
            check(
                population.refractory.is_some() == instance.refractory.is_some(),
                "population refractory definition/instance mismatch",
            )?;
            let mut frozen = BTreeSet::new();
            if let (Some(def), Some(refractory)) = (&population.refractory, &instance.refractory) {
                frozen = def.frozen_states.iter().cloned().collect();
                check(
                    frozen.len() == def.frozen_states.len()
                        && frozen.is_subset(&state_names)
                        && matches!(def.mode.as_str(), "fixed" | "expression"),
                    "invalid refractory frozen states",
                )?;
                refractory.validate(population.count, dt, start, population.steps)?;
                check(
                    def.mode == "fixed"
                        || (decode_bits(&refractory.period)?.to_bits() == 0
                            && refractory.period_ticks == 0),
                    "expression refractory cannot declare a fixed period",
                )?;
                inputs.insert("lastspike".to_owned(), DType::F64);
                inputs.insert("not_refractory".to_owned(), DType::Bool);
                input_dimensions.insert("lastspike".to_owned(), TIME_DIMENSIONS);
                input_dimensions.insert("not_refractory".to_owned(), DIMENSIONLESS);
                total_values += 2 * population.count;
            }
            let event_names: BTreeSet<_> = population.events.iter().cloned().collect();
            check(
                population.events.len() <= 128
                    && event_names.len() == population.events.len()
                    && population.events.windows(2).all(|pair| pair[0] < pair[1])
                    && population.events.iter().all(|name| valid_name(name)),
                "invalid canonical population event table",
            )?;
            for monitor in &population.event_monitors {
                let variable_names: BTreeSet<_> = monitor.variables.iter().cloned().collect();
                check(
                    event_names.contains(&monitor.event)
                        && variable_names.len() == monitor.variables.len()
                        && monitor.variables.windows(2).all(|pair| pair[0] < pair[1])
                        && variable_names.is_subset(
                            &state_names
                                .union(&parameter_names)
                                .cloned()
                                .chain(linked_names.iter().cloned())
                                .collect(),
                        ),
                    "invalid EventMonitor event or variables",
                )?;
            }
            let has_update = population
                .code_objects
                .iter()
                .any(|code| code.kind == "state_update");
            let has_subexpression_update = population
                .code_objects
                .iter()
                .any(|code| code.kind == "subexpression_update");
            let spatial_update_count = population
                .code_objects
                .iter()
                .filter(|code| code.kind == "spatial_state_update")
                .count();
            let poisson_input_count = population
                .code_objects
                .iter()
                .filter(|code| code.kind == "poisson_input")
                .count();
            let run_regularly_count = population
                .code_objects
                .iter()
                .filter(|code| code.kind == "run_regularly")
                .count();
            let has_spike_generator = instance.spike_generator.is_some();
            if let Some(generator) = &instance.spike_generator {
                let first_tick = run_clock.start_tick;
                let end_tick = first_tick + population.steps;
                check(
                    generator.spike_ticks.len() == generator.spike_indices.len()
                        && generator.spike_ticks.len() <= 10_000_000
                        && generator
                            .spike_ticks
                            .iter()
                            .zip(&generator.spike_indices)
                            .all(|(&tick, &index)| {
                                (first_tick..end_tick).contains(&tick) && index < population.count
                            })
                        && generator
                            .spike_ticks
                            .iter()
                            .zip(&generator.spike_indices)
                            .zip(
                                generator
                                    .spike_ticks
                                    .iter()
                                    .zip(&generator.spike_indices)
                                    .skip(1),
                            )
                            .all(|(left, right)| left < right),
                    "invalid SpikeGeneratorGroup schedule",
                )?;
                total_values += 2 * generator.spike_ticks.len();
                check(
                    population.states.is_empty()
                        && population.parameters.is_empty()
                        && population.refractory.is_none()
                        && population.state_monitors.is_empty()
                        && population.code_objects.is_empty(),
                    "SpikeGeneratorGroup must be a stateless event source",
                )?;
            }
            check(
                population.spike_monitor.is_none() || event_names.contains("spike"),
                "population spike monitor requires an event source",
            )?;
            check(
                population.refractory.is_none() || event_names.contains("spike"),
                "population refractory requires a spike event",
            )?;
            let has_dynamic_storage =
                !population.states.is_empty() || population.refractory.is_some();
            check(
                (has_update || run_regularly_count > 0) == has_dynamic_storage,
                "stateful or refractory populations require a state update or run_regularly; stateless populations do not",
            )?;
            check(
                has_update
                    || population.refractory.is_none()
                    || has_spike_generator,
                "stateless population with refractory state requires an update",
            )?;
            check(
                poisson_input_count <= 32 && (has_update || poisson_input_count == 0),
                "PoissonInput requires a stateful target population",
            )?;
            check(
                run_regularly_count <= 128
                    && (!population.states.is_empty() || run_regularly_count == 0),
                "run_regularly requires a stateful target population",
            )?;
            let validation_symbols = ValidationSymbols {
                inputs: inputs.clone(),
                scalar: scalar_inputs.clone(),
                dimensions: input_dimensions.clone(),
                scalar_dimensions: scalar_input_dimensions.clone(),
            };
            let mut code_names = BTreeSet::new();
            let mut threshold_events = BTreeSet::new();
            let mut reset_events = BTreeSet::new();
            for code in &population.code_objects {
                check(code.clock < d.clocks.len(), "invalid population code clock")?;
                let common = valid_name(&code.name)
                    && code_names.insert(code.name.clone())
                    && d.schedule.slots.contains(&code.when);
                let shape = match code.kind.as_str() {
                    "run_regularly" => code.iteration_domain == "all_neurons",
                    "subexpression_update" => {
                        code.clock == population.clock
                            && code.when == "before_start"
                            && code.order == 0
                            && code.iteration_domain == "all_neurons"
                            && code.event_name.is_none()
                    }
                    "state_update" => {
                        code.clock == population.clock
                            && code.when == "groups"
                            && code.order == 0
                            && code.iteration_domain == "all_neurons"
                            && code.event_name.is_none()
                    }
                    "spatial_state_update" => {
                        population.spatial.is_some()
                            && code.clock == population.clock
                            && code.when == "groups"
                            && code.order == 1
                            && code.iteration_domain == "all_neurons"
                            && code.event_name.is_none()
                    }
                    "poisson_input" => {
                        code.clock == population.clock
                            && code.when == "synapses"
                            && code.order == 0
                            && code.iteration_domain == "all_neurons"
                            && code.event_name.is_none()
                    }
                    "threshold" => {
                        code.clock == population.clock
                            && code.iteration_domain == "all_neurons"
                            && code.event_name.as_ref().is_some_and(|event| {
                                event_names.contains(event)
                                    && threshold_events.insert(event.clone())
                            })
                    }
                    "reset" => {
                        code.clock == population.clock
                            && code.iteration_domain == "spiking_neurons"
                            && code.event_name.as_ref().is_some_and(|event| {
                                event_names.contains(event) && reset_events.insert(event.clone())
                            })
                    }
                    _ => false,
                };
                check(
                    common && shape,
                    "unsupported population code-object schedule",
                )?;
                let conditions = if !matches!(code.kind.as_str(), "threshold" | "reset") {
                    frozen
                        .iter()
                        .map(|name| (name.clone(), "not_refractory".to_owned()))
                        .collect()
                } else {
                    BTreeMap::new()
                };
                let mut writable = if code.kind == "spatial_state_update" {
                    BTreeSet::new()
                } else {
                    state_names.clone()
                };
                if code.kind == "state_update"
                    && population
                        .refractory
                        .as_ref()
                        .is_some_and(|refractory| refractory.mode == "expression")
                {
                    writable.insert("not_refractory".to_owned());
                }
                code.validate(&functions, &validation_symbols, &writable, &conditions)?;
                if code.kind == "poisson_input" {
                    check(
                        code.effects.writes.len() == 1,
                        "PoissonInput must write exactly one population state",
                    )?;
                }
            }
            check(
                usize::from(has_subexpression_update)
                    == population
                        .code_objects
                        .iter()
                        .filter(|code| code.kind == "subexpression_update")
                        .count()
                    && usize::from(has_update)
                        == population
                            .code_objects
                            .iter()
                            .filter(|code| code.kind == "state_update")
                            .count()
                    && spatial_update_count == usize::from(population.spatial.is_some())
                    && threshold_events.len()
                        == population
                            .code_objects
                            .iter()
                            .filter(|code| code.kind == "threshold")
                            .count()
                    && reset_events.len()
                        == population
                            .code_objects
                            .iter()
                            .filter(|code| code.kind == "reset")
                            .count()
                    && reset_events.is_subset(&threshold_events)
                    && if has_spike_generator {
                        event_names == BTreeSet::from(["spike".to_owned()])
                            && threshold_events.is_empty()
                    } else {
                        threshold_events == event_names
                    },
                "population event sources and reset consumers are inconsistent",
            )?;
            let monitor = &population.monitor;
            let parameter_names: BTreeSet<_> = population
                .parameters
                .iter()
                .map(|symbol| symbol.name.clone())
                .collect();
            let mut expected_record = BTreeSet::new();
            let mut expected_variables = BTreeSet::new();
            for state_monitor in &population.state_monitors {
                check(
                    !state_monitor.variables.is_empty()
                        && state_monitor.variables.iter().all(|name| {
                            state_names.contains(name)
                                || parameter_names.contains(name)
                                || linked_names.contains(name)
                        })
                        && state_monitor
                            .variables
                            .iter()
                            .collect::<BTreeSet<_>>()
                            .len()
                            == state_monitor.variables.len()
                        && state_monitor
                            .record
                            .iter()
                            .all(|index| *index < population.count),
                    "invalid StateMonitor definition",
                )?;
                check(
                    state_monitor.output_variables.is_empty()
                        || state_monitor.output_variables.iter().all(|name| {
                            state_names.contains(name)
                                || parameter_names.contains(name)
                                || linked_names.contains(name)
                                || population.monitor_expressions.contains_key(name)
                                || (population.refractory.is_some()
                                    && (name == "lastspike" || name == "not_refractory"))
                        }),
                    "invalid StateMonitor output variable",
                )?;
                if !state_monitor.record.is_empty() {
                    expected_record.extend(state_monitor.record.iter().copied());
                    expected_variables.extend(state_monitor.variables.iter().cloned());
                }
            }
            check(
                monitor.record.iter().all(|index| *index < population.count)
                    && monitor
                        .record
                        .iter()
                        .copied()
                        .collect::<BTreeSet<_>>()
                        .len()
                        == monitor.record.len()
                    && monitor.variables.iter().all(|name| {
                        state_names.contains(name)
                            || parameter_names.contains(name)
                            || linked_names.contains(name)
                    })
                    && monitor.variables.iter().collect::<BTreeSet<_>>().len()
                        == monitor.variables.len()
                    && ((population.steps == 0 && monitor.window_steps == 0)
                        || (population.steps > 0
                            && monitor.window_steps >= 1))
                    && monitor.window_steps <= population.steps
                    && monitor.record.iter().copied().collect::<BTreeSet<_>>() == expected_record
                    && monitor.variables.iter().cloned().collect::<BTreeSet<_>>()
                        == expected_variables,
                "physical monitor plan does not match StateMonitor definitions",
            )?;
            let monitor_sample_steps = if population.state_monitors.is_empty() {
                0
            } else if monitor.window_steps < population.steps {
                monitor.window_steps
            } else {
                population
                    .state_monitors
                    .iter()
                    .map(|state_monitor| state_monitor.clock)
                    .collect::<BTreeSet<_>>()
                    .into_iter()
                    .map(|clock| self.run.clocks[clock].steps)
                    .sum()
            };
            budget(
                monitor.record.len() * monitor.variables.len(),
                monitor_sample_steps,
                100_000_000,
            )?;
            population_states.push(state_names);
            population_frozen.push(frozen);
            population_dtypes.push(
                population
                    .states
                    .iter()
                    .chain(&population.parameters)
                    .map(|symbol| (symbol.name.clone(), symbol.dtype))
                    .chain(
                        population
                            .linked_variables
                            .iter()
                            .map(|linked| (linked.name.clone(), linked.dtype)),
                    )
                    .collect::<BTreeMap<_, _>>(),
            );
            population_dimensions.push(
                population
                    .states
                    .iter()
                    .chain(&population.parameters)
                    .map(|symbol| (symbol.name.clone(), symbol.dimensions))
                    .chain(
                        population
                            .linked_variables
                            .iter()
                            .map(|linked| (linked.name.clone(), linked.dimensions)),
                    )
                    .collect::<BTreeMap<_, _>>(),
            );
            check(total_values <= resource_limits::initial_values()?, "probe array budget exceeded")?;
        }
        check(
            next_offset == self.instance.neuron_count,
            "population layout does not cover neuron count",
        )?;
        check(
            d.synapses.len() == self.instance.synapses.len(),
            "synaptic definition/instance mismatch",
        )?;
        let synapse_states = d
            .synapses
            .iter()
            .map(|definition| {
                definition
                    .states
                    .iter()
                    .map(|state| state.name.clone())
                    .collect::<BTreeSet<_>>()
            })
            .collect::<Vec<_>>();
        let synapse_dtypes = d
            .synapses
            .iter()
            .map(|definition| {
                definition
                    .states
                    .iter()
                    .map(|state| (state.name.clone(), state.dtype))
                    .collect::<BTreeMap<_, _>>()
            })
            .collect::<Vec<_>>();
        let synapse_dimensions = d
            .synapses
            .iter()
            .map(|definition| {
                definition
                    .states
                    .iter()
                    .map(|state| (state.name.clone(), state.dimensions))
                    .collect::<BTreeMap<_, _>>()
            })
            .collect::<Vec<_>>();
        let mut synapse_names = BTreeSet::new();
        let mut total_synapse_ticks = 0u128;
        for (synapse_index, (def, instance)) in d
            .synapses
            .iter()
            .zip(&self.instance.synapses)
            .enumerate()
        {
            let source_layout_valid = if let Some(source) = def.source_synapse {
                source < d.synapses.len()
                    && source != synapse_index
                    && d.synapses[source].source_synapse.is_none()
                    && d.synapses[source].target_synapse.is_none()
                    && def.source_start == 0
                    && def.source_count == self.instance.synapses[source].edge_count()
            } else {
                def.source_start
                    .checked_add(def.source_count)
                    .is_some_and(|end| end <= d.populations[def.source_population].count)
            };
            let target_layout_valid = if let Some(target) = def.target_synapse {
                target < d.synapses.len()
                    && target != synapse_index
                    && d.synapses[target].source_synapse.is_none()
                    && d.synapses[target].target_synapse.is_none()
                    && def.target_start == 0
                    && def.target_count == self.instance.synapses[target].edge_count()
            } else {
                def.target_start
                    .checked_add(def.target_count)
                    .is_some_and(|end| end <= d.populations[def.target_population].count)
            };
            check(
                valid_name(&def.name)
                    && synapse_names.insert(def.name.clone())
                    && def.source_population < d.populations.len()
                    && def.target_population < d.populations.len()
                    && def.source_count > 0
                    && def.target_count > 0
                    && source_layout_valid
                    && target_layout_valid,
                "invalid synaptic population reference",
            )?;
            for monitor in &def.state_monitors {
                let valid_sources = monitor.sources.iter().all(|source| match source {
                    SynapseMonitorSourceDefinition::SynapseState { name, dtype } => def
                        .states
                        .iter()
                        .any(|symbol| symbol.name == *name && symbol.dtype == *dtype),
                    SynapseMonitorSourceDefinition::PreState { name, dtype } => {
                        if let Some(source) = def.source_synapse {
                            d.synapses[source]
                                .states
                                .iter()
                                .any(|symbol| symbol.name == *name && symbol.dtype == *dtype)
                        } else {
                            d.populations[def.source_population]
                                .states
                                .iter()
                                .any(|symbol| symbol.name == *name && symbol.dtype == *dtype)
                        }
                    }
                    SynapseMonitorSourceDefinition::PostState { name, dtype } => {
                        if let Some(target) = def.target_synapse {
                            d.synapses[target]
                                .states
                                .iter()
                                .any(|symbol| symbol.name == *name && symbol.dtype == *dtype)
                        } else {
                            d.populations[def.target_population]
                                .states
                                .iter()
                                .any(|symbol| symbol.name == *name && symbol.dtype == *dtype)
                        }
                    }
                    SynapseMonitorSourceDefinition::Linked { name, dtype } => def
                        .linked_variables
                        .iter()
                        .any(|linked| linked.name == *name && linked.dtype == *dtype),
                });
                check(
                    valid_name(&monitor.name)
                        && state_monitor_names.insert(monitor.name.clone())
                        && monitor.clock < d.clocks.len()
                        && d.schedule.slots.contains(&monitor.when)
                        && !monitor.variables.is_empty()
                        && monitor.variables.len() == monitor.sources.len()
                        && monitor.variables.iter().collect::<BTreeSet<_>>().len()
                            == monitor.variables.len()
                        && (monitor.output_variables.is_empty()
                            || monitor.output_variables.iter().all(|name| {
                                monitor.variables.contains(name)
                                    || def.monitor_expressions.contains_key(name)
                            }))
                        && valid_sources
                        && monitor.record.iter().all(|&edge| edge < instance.edge_count()),
                    "invalid synapse StateMonitor definition",
                )?;
            }
            let all_edge_work = def.code_objects.iter().any(|code| {
                matches!(
                    code.kind.as_str(),
                    "synapse_subexpression_update"
                        | "synapse_state_update"
                        | "summed_variable"
                        | "synapse_run_regularly"
                )
            });
            if all_edge_work {
                total_synapse_ticks += (instance.edge_count() as u128)
                    * (d.populations[def.source_population].steps as u128);
            }
            check(
                total_synapse_ticks <= MAX_SYNAPSE_TICKS,
                "synapse-tick budget exceeded",
            )?;
            let all_population_schemas = population_dtypes
                .iter()
                .zip(&population_dimensions)
                .map(|(dtypes, dimensions)| PopulationSchemaRef { dtypes, dimensions })
                .collect::<Vec<_>>();
            let source_states = def
                .source_synapse
                .map(|source| &synapse_states[source])
                .unwrap_or(&population_states[def.source_population]);
            let source_schema = if let Some(source) = def.source_synapse {
                PopulationSchemaRef {
                    dtypes: &synapse_dtypes[source],
                    dimensions: &synapse_dimensions[source],
                }
            } else {
                PopulationSchemaRef {
                    dtypes: &population_dtypes[def.source_population],
                    dimensions: &population_dimensions[def.source_population],
                }
            };
            let target_states = def
                .target_synapse
                .map(|target| &synapse_states[target])
                .unwrap_or(&population_states[def.target_population]);
            let target_schema = if let Some(target) = def.target_synapse {
                PopulationSchemaRef {
                    dtypes: &synapse_dtypes[target],
                    dimensions: &synapse_dimensions[target],
                }
            } else {
                PopulationSchemaRef {
                    dtypes: &population_dtypes[def.target_population],
                    dimensions: &population_dimensions[def.target_population],
                }
            };
            let symbols = def.symbols(
                instance,
                (
                    source_states,
                    target_states,
                ),
                (
                    source_schema,
                    target_schema,
                ),
                &d.populations,
                &all_population_schemas,
                (
                    (
                        population_dts[def.source_population],
                        d.populations[def.source_population].steps,
                    ),
                    (
                        population_dts[def.target_population],
                        d.populations[def.target_population].steps,
                    ),
                ),
                &mut total_values,
                def.target_synapse.is_none()
                    && d.populations[def.target_population].refractory.is_some(),
            )?;
            check(
                instance.pathways.windows(2).all(|pair| {
                    let left_kind = usize::from(pair[0].kind == "post");
                    let right_kind = usize::from(pair[1].kind == "post");
                    left_kind < right_kind
                        || (left_kind == right_kind && pair[0].name < pair[1].name)
                }),
                "synaptic pathways must be ordered by kind and name",
            )?;
            for pathway in &instance.pathways {
                let (population, endpoint_count) = if pathway.kind == "pre" {
                    (def.source_population, def.source_count)
                } else {
                    (def.target_population, def.target_count)
                };
                let dt = population_dts[population];
                let start_tick = (start / dt).round() as usize;
                let end_tick = start_tick
                    .checked_add(d.populations[population].steps)
                    .ok_or("population tick range overflow")?;
                let uniform_delay = pathway
                    .delay_ticks
                    .first()
                    .filter(|delay| pathway.delay_ticks.iter().all(|value| value == *delay));
                check(
                    pathway.pending.len() <= 10_000_000
                        && pathway
                            .pending
                            .windows(2)
                            .all(|pair| pair[0].delivery_tick <= pair[1].delivery_tick)
                        && pathway.pending.iter().all(|event| {
                            event.delivery_tick >= start_tick
                                && event.delivery_tick < end_tick
                                && if uniform_delay.is_some() {
                                    event.item < endpoint_count
                                } else {
                                    event.item < instance.edge_count()
                                }
                        }),
                    "invalid pending synaptic pathway event",
                )?;
            }
            let synapse_state_names: BTreeSet<_> =
                def.states.iter().map(|state| state.name.clone()).collect();
            let clock_driven: BTreeSet<_> = def.clock_driven_states.iter().cloned().collect();
            check(
                clock_driven.len() == def.clock_driven_states.len()
                    && clock_driven.is_subset(&synapse_state_names),
                "invalid clock-driven synaptic state list",
            )?;
            let has_state_update = !clock_driven.is_empty();
            let subexpression_count = def
                .code_objects
                .iter()
                .filter(|code| code.kind == "synapse_subexpression_update")
                .count();
            let regular_count = def
                .code_objects
                .iter()
                .filter(|code| code.kind == "synapse_run_regularly")
                .count();
            let summed_count = def
                .code_objects
                .iter()
                .filter(|code| code.kind == "summed_variable")
                .count();
            let pre_count = instance
                .pathways
                .iter()
                .filter(|pathway| pathway.kind == "pre")
                .count();
            let post_count = instance.pathways.len() - pre_count;
            check(
                def.code_objects.len()
                    == subexpression_count
                        + summed_count
                        + regular_count
                        + usize::from(has_state_update)
                        + pre_count
                        + post_count
                    && subexpression_count <= 1
                    && (pre_count > 0 || post_count > 0 || summed_count > 0
                        || has_state_update),
                "expected optional subexpression/state updates, summed variables and pathways",
            )?;
            let mut expected = Vec::new();
            expected.extend(std::iter::repeat_n(
                (
                    "synapse_subexpression_update",
                    "before_start",
                    0,
                    "all_synapses",
                    None,
                ),
                subexpression_count,
            ));
            for code in def
                .code_objects
                .iter()
                .filter(|code| code.kind == "summed_variable")
            {
                let (start, count, full_count) = if code.summed_target.as_deref() == Some("pre") {
                    (
                        def.source_start,
                        def.source_count,
                        def.source_synapse
                            .map(|source| self.instance.synapses[source].edge_count())
                            .unwrap_or(d.populations[def.source_population].count),
                    )
                } else {
                    (
                        def.target_start,
                        def.target_count,
                        def.target_synapse
                            .map(|target| self.instance.synapses[target].edge_count())
                            .unwrap_or(d.populations[def.target_population].count),
                    )
                };
                let expected_order = if start == 0 && count == full_count {
                    -1
                } else {
                    0
                };
                expected.push((
                    "summed_variable",
                    code.when.as_str(),
                    expected_order,
                    "all_synapses",
                    None,
                ));
            }
            if has_state_update {
                expected.push(("synapse_state_update", "groups", 0, "all_synapses", None));
            }
            for code in def
                .code_objects
                .iter()
                .filter(|code| code.kind == "synapse_run_regularly")
            {
                expected.push((
                    "synapse_run_regularly",
                    code.when.as_str(),
                    code.order,
                    "all_synapses",
                    None,
                ));
            }
            for pathway in &instance.pathways {
                expected.push(if pathway.kind == "pre" {
                    ("synapses", "synapses", -1, "active_synapses", Some(pathway))
                } else {
                    (
                        "synapses_post",
                        "synapses",
                        1,
                        "active_synapses",
                        Some(pathway),
                    )
                });
            }
            let mut code_names = BTreeSet::new();
            for (code, (kind, slot, order, domain, pathway)) in
                def.code_objects.iter().zip(expected)
            {
                let clock_population = if kind == "summed_variable" {
                    if code.summed_target.as_deref() == Some("pre") {
                        def.source_population
                    } else {
                        def.target_population
                    }
                } else if kind == "synapses_post" {
                    def.target_population
                } else {
                    def.source_population
                };
                check(
                    valid_name(&code.name)
                        && code_names.insert(code.name.clone())
                        && code.clock < d.clocks.len()
                        && (matches!(kind, "synapse_run_regularly" | "summed_variable")
                            || code.clock == d.populations[clock_population].clock)
                        && code.kind == kind
                        && code.when == slot
                        && code.order == order
                        && code.iteration_domain == domain
                        && code.pathway_name.as_deref()
                            == pathway.map(|instance| instance.name.as_str())
                        && code.event_name.as_deref()
                            == pathway.map(|instance| instance.event.as_str()),
                    "unsupported synaptic code-object schedule",
                )?;
                if kind == "summed_variable" {
                    check(
                        code.pathway_name.is_none()
                            && matches!(code.when.as_str(), "groups" | "after_groups"),
                        "pathway metadata on summed code object",
                    )?;
                    check(
                        matches!(code.summed_target.as_deref(), Some("pre" | "post")),
                        "invalid summed variable endpoint",
                    )?;
                    let target_state = code
                        .summed_state
                        .as_ref()
                        .ok_or("missing summed variable target state")?;
                    check(
                        (def.source_synapse.is_none() && def.target_synapse.is_none())
                            || code.summed_target.as_deref() == Some("post"),
                        "edge-domain summed variables must target post",
                    )?;
                    check(
                        if code.summed_target.as_deref() == Some("pre") {
                            source_states.contains(target_state)
                        } else {
                            target_states.contains(target_state)
                        },
                        "summed variable target must be mutable population state",
                    )?;
                    code.validate(&functions, &symbols, &BTreeSet::new(), &BTreeMap::new())?;
                } else if matches!(
                    kind,
                    "synapse_state_update"
                        | "synapse_subexpression_update"
                        | "synapse_run_regularly"
                ) {
                    check(
                        code.summed_target.is_none()
                            && code.summed_state.is_none()
                            && code.pathway_name.is_none(),
                        "summed metadata on non-summed code object",
                    )?;
                    let writable = if kind == "synapse_state_update" {
                        clock_driven.clone()
                    } else {
                        synapse_state_names.clone()
                    };
                    code.validate(&functions, &symbols, &writable, &BTreeMap::new())?;
                } else if matches!(kind, "synapses" | "synapses_post") {
                    check(
                        code.summed_target.is_none() && code.summed_state.is_none(),
                        "summed metadata on non-summed code object",
                    )?;
                    let mut writable: BTreeSet<String> = def
                        .post_state_aliases
                        .keys()
                        .cloned()
                        .chain(def.states.iter().map(|s| s.name.clone()))
                        .collect();
                    if kind == "synapses" {
                        writable.extend(def.pre_state_aliases.iter().filter_map(
                            |(alias, state)| {
                                (def.source_synapse.is_some()
                                    || !population_frozen[def.source_population].contains(state))
                                    .then_some(alias.clone())
                            },
                        ));
                    }
                    let conditions = def
                        .post_state_aliases
                        .iter()
                        .filter(|(_, state)| {
                        def.target_synapse.is_none()
                            && population_frozen[def.target_population].contains(*state)
                        })
                        .map(|(alias, _)| (alias.clone(), "not_refractory_post".to_owned()))
                        .collect();
                    code.validate(&functions, &symbols, &writable, &conditions)?;
                } else {
                    check(
                        code.summed_target.is_none() && code.summed_state.is_none(),
                        "summed metadata on non-summed code object",
                    )?;
                    let writable = def.states.iter().map(|s| s.name.clone()).collect();
                    code.validate(&functions, &symbols, &writable, &BTreeMap::new())?;
                }
                if let Some(pathway) = pathway {
                    check(
                        (pathway.kind != "post" || def.target_synapse.is_none())
                            && (pathway.kind != "pre" || def.source_synapse.is_none()),
                        "a Synapses endpoint cannot generate events",
                    )?;
                    let endpoint = if pathway.kind == "pre" {
                        def.source_population
                    } else {
                        def.target_population
                    };
                    check(
                        d.populations[endpoint].events.contains(&pathway.event),
                        "synaptic pathway references an unknown endpoint event",
                    )?;
                }
            }
        }
        self.validate_schedule()?;
        check(total_values <= resource_limits::initial_values()?, "probe array budget exceeded")?;
        Ok(())
    }
}
pub fn run_cli() -> Result<()> {
    let args: Vec<_> = std::env::args_os().collect();
    let initialization = args.get(1).is_some_and(|arg| arg == "--gpu-initialization");
    check(
        args.len() == if initialization { 5 } else { 3 },
        "usage: b2-runner MODEL.json OUTPUT_DIRECTORY | b2-runner --validate MODEL.json | b2-runner --canonical-hashes MODEL.json | b2-runner --gpu-initialization MODEL.json OUTPUT_DIRECTORY MAX_BYTES",
    )?;
    let validate_only = args[1] == "--validate";
    let hash_only = args[1] == "--canonical-hashes";
    let max_ir_bytes = resource_limits::ir_bytes()? as u64;
    resource_limits::initial_values()?;
    resource_limits::neurons()?;
    resource_limits::population_steps()?;
    // Large current-schema documents can be checked without a JSON string tree.
    // A mismatch (including omitted optional fields or an older wire schema)
    // uses the original loader, preserving public acceptance and diagnostics.
    if validate_only {
        let input = File::open(&args[2])?;
        let size = input.metadata()?.len();
        check(size <= max_ir_bytes, "IR exceeds the configured byte budget")?;
        if size >= 8 * 1024 * 1024 {
            let mut reader = BufReader::with_capacity(64 * 1024, input.take(max_ir_bytes + 1));
            if compact_input::validate(&mut reader).is_ok() && reader.get_ref().limit() > 0 {
                println!("validated AtlasIR");
                return Ok(());
            }
        }
    }
    check(
        validate_only
            || hash_only
            || initialization
            || !args[1].to_string_lossy().starts_with("--"),
        "unknown b2-runner option",
    )?;
    let directory = Path::new(&args[if initialization { 3 } else { 2 }]);
    check(
        validate_only || hash_only || !directory.exists(),
        "output directory already exists; choose a new directory",
    )?;
    let mut data = Vec::new();
    File::open(
        &args[if validate_only || hash_only || initialization {
            2
        } else {
            1
        }],
    )?
    .take(max_ir_bytes + 1)
    .read_to_end(&mut data)?;
    check(
        data.len() as u64 <= max_ir_bytes,
        "IR exceeds the configured byte budget",
    )?;
    let mut value: serde_json::Value = serde_json::from_slice(&data)?;
    drop(data);
    prepare_protocol(&mut value)?;
    // Keep only the hash envelope before consuming the potentially large tree.
    let expected = expected_protocol(&value)?;
    let actual = value.get("protocol").cloned();
    let model: Model = serde_json::from_value(value)?;
    model.validate()?;
    check(
        actual.as_ref() == Some(&expected),
        "AtlasIR canonical layer hash mismatch",
    )?;
    if initialization {
        let budget: usize = args[4]
            .to_str()
            .ok_or("invalid initialization budget")?
            .parse()?;
        gpu_initialization::export(&model, directory, budget)
    } else if hash_only {
        println!("{}", serde_json::to_string(&expected["layers"])?);
        Ok(())
    } else if validate_only {
        println!("validated AtlasIR");
        Ok(())
    } else {
        executor::execute(model, directory)
    }
}

fn canonical_hash(value: &serde_json::Value) -> Result<String> {
    struct HashWriter(Sha256);
    impl Write for HashWriter {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.0.update(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let mut hash = HashWriter(Sha256::new());
    {
        let mut writer = BufWriter::with_capacity(64 * 1024, &mut hash);
        canonical::write(value, &mut writer)?;
        writer.flush()?;
    }
    Ok(format!("{:x}", hash.0.finalize()))
}

fn protocol_layers(value: &serde_json::Value) -> Result<serde_json::Value> {
    let object = value.as_object().ok_or("AtlasIR root must be an object")?;
    Ok(serde_json::json!({
        "definition": canonical_hash(object.get("definition").ok_or("missing definition")?)?,
        "instance": canonical_hash(object.get("instance").ok_or("missing instance")?)?,
        "run": canonical_hash(object.get("run").ok_or("missing run")?)?,
    }))
}

fn expected_protocol(value: &serde_json::Value) -> Result<serde_json::Value> {
    Ok(serde_json::json!({
        "name": "b2ir",
        "version": {"major": 1, "minor": 0},
        "canonical_encoding": "b2ir-canonical-json-v1",
        "hash_algorithm": "sha256",
        "layers": protocol_layers(value)?,
    }))
}

fn prepare_protocol(value: &mut serde_json::Value) -> Result<()> {
    let schema = value
        .get("schema")
        .and_then(serde_json::Value::as_str)
        .ok_or("missing AtlasIR schema")?
        .to_owned();
    if matches!(
        schema.as_str(),
        "b2ir-gate0-probe-v34"
            | "b2ir-gate0-probe-v35"
            | "b2ir-gate0-probe-v36"
            | "b2ir-gate0-probe-v37"
    ) {
        if schema != "b2ir-gate0-probe-v34" {
            verify_protocol(value)?;
        } else {
            check(
                value.get("protocol").is_none(),
                "legacy AtlasIR has a protocol envelope",
            )?;
        }
        let populations = value
            .get_mut("definition")
            .and_then(|definition| definition.get_mut("populations"))
            .and_then(serde_json::Value::as_array_mut)
            .ok_or("legacy AtlasIR has no population table")?;
        for population in populations {
            population
                .as_object_mut()
                .ok_or("invalid legacy population definition")?
                .entry("linked_variables")
                .or_insert_with(|| serde_json::Value::Array(Vec::new()));
        }
        let functions = value
            .get_mut("definition")
            .and_then(|definition| definition.get_mut("functions"))
            .and_then(serde_json::Value::as_array_mut)
            .ok_or("legacy AtlasIR has no Function table")?;
        for function in functions {
            let function = function
                .as_object_mut()
                .ok_or("invalid legacy Function definition")?;
            function.insert(
                "abi".into(),
                serde_json::Value::String("b2ir-function-v1".into()),
            );
            function
                .entry("backend_implementations")
                .or_insert_with(|| serde_json::Value::Object(serde_json::Map::new()));
        }
        value["schema"] = serde_json::Value::String("b2ir-v1".into());
        let protocol = expected_protocol(value)?;
        value
            .as_object_mut()
            .ok_or("AtlasIR root must be an object")?
            .insert("protocol".into(), protocol);
        return Ok(());
    }
    check(schema == "b2ir-v1", "unsupported AtlasIR schema")?;
    Ok(())
}

fn verify_protocol(value: &serde_json::Value) -> Result<()> {
    let expected = expected_protocol(value)?;
    check(
        value
            .get("protocol")
            .is_some_and(|actual| actual == &expected),
        "AtlasIR canonical layer hash mismatch",
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_integer_cast_accepts_tick_and_index_domains() {
        let symbols = BTreeMap::from([
            ("tick".to_string(), DType::Tick),
            ("index".to_string(), DType::Index),
        ]);
        let functions = BTreeMap::new();
        for name in ["tick", "index"] {
            let expression = Expr::Cast {
                dtype: DType::I64,
                arg: Box::new(Expr::Load {
                    name: name.to_string(),
                }),
            };
            assert!(
                expression
                    .infer(&symbols, &mut BTreeSet::new(), &functions)
                    .unwrap()
                    == DType::I64
            );
        }
    }

    #[test]
    fn complete_refractory_effects_only_after_validating_frozen_wire_graph() {
        let value: serde_json::Value =
            serde_json::from_str(include_str!("../tests/golden/b2ir-v1/minimal-v1.json")).unwrap();
        verify_protocol(&value).unwrap();
        let mut model: Model = serde_json::from_value(value.clone()).unwrap();
        model.validate().unwrap();
        model.complete_execution_effects().unwrap();
        let nodes = &model.definition.schedule.nodes;
        let updater = nodes
            .iter()
            .find(|node| node.id == "population/0/code/0")
            .unwrap();
        let threshold = nodes
            .iter()
            .find(|node| node.id == "population/0/code/1")
            .unwrap();
        assert!(updater
            .effects
            .reads
            .contains(&"population/0/refractory/lastspike".into()));
        assert!(updater
            .effects
            .writes
            .contains(&"population/0/refractory/not_refractory".into()));
        assert!(threshold
            .effects
            .reads
            .contains(&"population/0/refractory/not_refractory".into()));
        for resource in ["lastspike", "not_refractory"] {
            assert!(threshold
                .effects
                .writes
                .contains(&format!("population/0/refractory/{resource}")));
        }
        assert!(threshold.dependencies.contains(&updater.id));
        assert_eq!(
            protocol_layers(&value).unwrap(),
            value["protocol"]["layers"]
        );
    }
}
