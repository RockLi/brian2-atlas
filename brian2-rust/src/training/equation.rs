//! Native reverse-mode differentiation of bounded scalar SSA update programs.
use super::*;
use super::math::Function;

#[derive(Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IntegerBinary { Add, Sub, Mul, Min, Max, FloorDiv, Mod, BitAnd, BitOr, BitXor, LeftShift, RightShift }
#[derive(Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Comparison { Gt, Ge, Lt, Le, Eq, Ne }
impl Comparison {
    fn test(self, a: f64, b: f64) -> bool {
        match self { Self::Gt => a>b, Self::Ge => a>=b, Self::Lt => a<b,
            Self::Le => a<=b, Self::Eq => a==b, Self::Ne => a!=b }
    }
}
pub(super) fn int32(value: f64) -> Result<i32> {
    ensure(value.is_finite() && value == value.trunc() && value >= i32::MIN as f64 && value <= i32::MAX as f64,
        "discrete integer value is outside int32")?;
    Ok(value as i32)
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
pub enum Node {
    EagerBooleanAnd { left: usize, right: usize },
    EagerBooleanOr { left: usize, right: usize },
    Sequence { left: usize, right: usize, #[serde(default)] boolean: bool },
    IntegerSequence { left: usize, right: usize },
    Poisson { rate: usize, stream: usize },
    Math { arg: usize, kind: Function },
    IntegerConstant { value: i32 },
    IntegerState { index: usize },
    IntegerParameter { bank: usize, index: usize },
    IntegerNeuronParameter { bank: usize, index: usize },
    IntegerMappedParameter { bank: usize, mapping: usize },
    IntegerParameterGather { bank: usize, index: usize },
    IntegerCast { arg: usize },
    IntegerFloat { arg: usize },
    IntegerBinary { left: usize, right: usize, kind: IntegerBinary },
    IntegerCompare { left: usize, right: usize, kind: Comparison },
    IntegerSelect { condition: usize, yes: usize, no: usize },
    DiscreteCompare { left: usize, right: usize, kind: Comparison },
    BooleanCast { arg: usize },
    IntegerNeg { arg: usize },
    SurrogateStep { arg: usize, slope: usize, scale: usize, inclusive: bool },
    BooleanAnd { left: usize, right: usize },
    BooleanOr { left: usize, right: usize },
    BooleanNot { arg: usize },
    Equality { left: usize, right: usize, unequal: bool },
    Select { condition: usize, yes: usize, no: usize,
        #[serde(default, skip_serializing_if="false_flag")] boolean: bool },
    Time,
    TimeWord { word: u32, #[serde(default)] float32: bool },
    ElapsedCompare { low: usize, high: usize, right: usize, kind: Comparison, constant: Option<f64> },
    ClockTime { index: usize },
    Noise { stream: usize },
    UniformNoise { stream: usize },
    Voltage,
    State { index: usize },
    // A detached discrete predicate; only legal for a refractory counter in
    // v4 update programs. Plan validation enforces that narrower contract.
    RefractoryActive { index: usize },
    Constant { value: f64 },
    Parameter { bank: usize, index: usize },
    NeuronParameter { bank: usize, index: usize },
    MappedParameter { bank: usize, mapping: usize },
    ParameterGather { bank: usize, index: usize },
    TimedParameter { bank: usize, rows: usize, columns: usize, epsilon: f64, k: u64, time: usize, index: usize },
    Min { left: usize, right: usize },
    Max { left: usize, right: usize },
    Add { left: usize, right: usize },
    Sub { left: usize, right: usize },
    Mul { left: usize, right: usize },
    Div { left: usize, right: usize },
    FloorDiv { left: usize, right: usize },
    Modulo { left: usize, right: usize },
    Neg { arg: usize },
    Exp { arg: usize },
    Log { arg: usize },
    Tanh { arg: usize },
    Sqrt { arg: usize },
    Sin { arg: usize },
    Cos { arg: usize },
    Pow { arg: usize, value: f64 },
}
pub type Program = Vec<Node>;
fn false_flag(value: &bool) -> bool { !*value }
// Shared GPU ABI: opcode, local operands, global parameter slot; the scalar
// constant is carried separately as float32. v1/v2 metadata stays unchanged.
pub fn gpu_node(node: &Node, offsets: &[usize]) -> ([u64; 4], f64) {
    let (op, a, b, param, value) = match *node {
        Node::EagerBooleanAnd {left,right} => (56,left,right,0,0.0),
        Node::EagerBooleanOr {left,right} => (57,left,right,0,0.0),
        Node::Sequence { left, right, boolean } => (54, left, right, boolean as usize, 0.0),
        Node::IntegerSequence { left, right } => (55, left, right, 0, 0.0),
        Node::Poisson { rate, stream } => (53, rate, stream, 0, 0.0),
        Node::Math { arg, kind } => (52, arg, 0, kind as usize, 0.0),
        Node::IntegerConstant { value } => (33, value as u32 as usize, 0, 0, 0.0),
        Node::IntegerState { index } => (34, index, 0, 0, 0.0),
        Node::IntegerParameter { bank, index } => (35, 0, 0, offsets[bank]+index, 0.0),
        Node::IntegerNeuronParameter { bank, index } => (36, 0, 0, offsets[bank]+index, 0.0),
        Node::IntegerMappedParameter { bank, mapping } => (37, mapping, 0, offsets[bank], 0.0),
        Node::IntegerParameterGather { bank, index } => (49, index, offsets[bank+1]-offsets[bank], offsets[bank], 0.0),
        Node::ParameterGather { bank, index } => (48, index, offsets[bank+1]-offsets[bank], offsets[bank], 0.0),
        Node::IntegerCast { arg } => (38, arg, 0, 0, 0.0),
        Node::IntegerFloat { arg } => (39, arg, 0, 0, 0.0),
        Node::IntegerBinary { left, right, kind } => (40, left, right, kind as usize, 0.0),
        Node::IntegerCompare { left, right, kind } => (41, left, right, kind as usize, 0.0),
        Node::IntegerSelect { condition, yes, no } => (42, condition, yes, no, 0.0),
        Node::DiscreteCompare { left, right, kind } => (43, left, right, kind as usize, 0.0),
        Node::BooleanCast { arg } => (44, arg, 0, 0, 0.0),
        Node::IntegerNeg { arg } => (45, arg, 0, 0, 0.0),
        Node::FloorDiv { left, right } => (46, left, right, 0, 0.0),
        Node::Modulo { left, right } => (47, left, right, 0, 0.0),
        Node::SurrogateStep { arg, slope, scale, inclusive } => (if inclusive { 27 } else { 26 }, arg, slope, scale, 0.0),
        Node::BooleanAnd { left, right } => (28, left, right, 0, 0.0),
        Node::BooleanOr { left, right } => (29, left, right, 0, 0.0),
        Node::BooleanNot { arg } => (30, arg, 0, 0, 0.0),
        Node::Equality { left, right, unequal } => (if unequal { 32 } else { 31 }, left, right, 0, 0.0),
        Node::Select { condition, yes, no, .. } => (22, condition, yes, no, 0.0),
        Node::Time => (18, 0, 0, 0, 0.0),
        Node::TimeWord { word, float32 } => (50, word as usize, float32 as usize, 0, 0.0),
        Node::ElapsedCompare { low, high, .. } => (51, low, high, 0, 0.0),
        Node::ClockTime { index } => (23, index, 0, 0, 0.0),
        Node::Noise { stream } | Node::UniformNoise { stream } => (19, stream, 0, 0, 0.0),
        Node::Voltage => (0, 0, 0, 0, 0.0),
        Node::State { index } => (15, index, 0, 0, 0.0),
        Node::RefractoryActive { index } => (16, index, 0, 0, 0.0),
        Node::Constant { value } => (1, 0, 0, 0, value),
        Node::Parameter { bank, index } => (2, 0, 0, offsets[bank] + index, 0.0),
        Node::NeuronParameter { bank, index } => (17, 0, 0, offsets[bank] + index, 0.0),
        Node::MappedParameter { bank, mapping } => (24, mapping, 0, offsets[bank], 0.0),
        Node::TimedParameter { bank, time, index, epsilon, .. } => (25, time, index, offsets[bank], epsilon),
        Node::Min { left, right } => (20, left, right, 0, 0.0),
        Node::Max { left, right } => (21, left, right, 0, 0.0),
        Node::Add { left, right } => (3, left, right, 0, 0.0),
        Node::Sub { left, right } => (4, left, right, 0, 0.0),
        Node::Mul { left, right } => (5, left, right, 0, 0.0),
        Node::Div { left, right } => (6, left, right, 0, 0.0),
        Node::Neg { arg } => (7, arg, 0, 0, 0.0),
        Node::Exp { arg } => (8, arg, 0, 0, 0.0),
        Node::Log { arg } => (9, arg, 0, 0, 0.0),
        Node::Tanh { arg } => (10, arg, 0, 0, 0.0),
        Node::Sqrt { arg } => (11, arg, 0, 0, 0.0),
        Node::Sin { arg } => (12, arg, 0, 0, 0.0),
        Node::Cos { arg } => (13, arg, 0, 0, 0.0),
        Node::Pow { arg, value } => (14, arg, 0, 0, value),
    };
    ([op, a as u64, b as u64, param as u64], value)
}
pub fn validate(program: &Program, plan: &Plan) -> Result<()> {
    validate_states(program, plan, 0)
}
pub fn validate_states(program: &Program, plan: &Plan, states: usize) -> Result<()> {
    ensure(
        !program.is_empty() && program.len() <= 128,
        "equation requires 1..128 SSA nodes",
    )?;
    for (i, node) in program.iter().enumerate() {
        let valid = match *node {
            Node::Sequence { left, right, boolean } => left < i && right < i
                && !integer_node(&program[right]) && (!boolean || boolean_node(&program[right])),
            Node::IntegerSequence { left, right } => states > 0 && left < i && right < i
                && integer_node(&program[right]),
            Node::Poisson { rate, stream } => (plan.dynamic.is_some() || plan.state_equations.is_some() || plan.equations.is_some())
                && states > 0 && stream < 16 && plan.noise_streams.is_some() && rate < i && !integer_node(&program[rate]),
            Node::IntegerConstant { .. } => states > 0,
            Node::IntegerState { index } => plan.dynamic.is_some() && index < states,
            Node::IntegerParameter { bank, index } | Node::IntegerNeuronParameter { bank, index } =>
                plan.dynamic.is_some() && bank < plan.masks.len() && index < plan.masks[bank].len(),
            Node::IntegerMappedParameter { bank, mapping } => bank < plan.masks.len() && plan.dynamic.as_ref()
                .is_some_and(|d| mapping < d.parameter_maps.len() && d.parameter_maps[mapping].iter().all(|&k| k < plan.masks[bank].len())),
            Node::ParameterGather { bank, index } | Node::IntegerParameterGather { bank, index } =>
                plan.dynamic.is_some() && bank < plan.masks.len() && !plan.masks[bank].is_empty()
                && index < i && integer_node(&program[index]),
            Node::IntegerCast { arg } | Node::BooleanCast { arg } => states > 0 && arg < i && !integer_node(&program[arg]),
            Node::IntegerFloat { arg } | Node::IntegerNeg { arg } => states > 0 && arg < i && integer_node(&program[arg]),
            Node::IntegerBinary { left, right, .. } | Node::IntegerCompare { left, right, .. } =>
                states > 0 && left < i && right < i && integer_node(&program[left]) && integer_node(&program[right]),
            Node::DiscreteCompare { left, right, .. } => states > 0 && left < i && right < i
                && !integer_node(&program[left]) && !integer_node(&program[right]),
            Node::IntegerSelect { condition, yes, no } => states > 0 && condition < i && yes < i && no < i
                && !integer_node(&program[condition]) && integer_node(&program[yes]) && integer_node(&program[no]),
            Node::SurrogateStep { arg, slope, scale, .. } => plan.dynamic.is_some() && arg < i && slope < i && scale < i
                && matches!(program[slope], Node::Constant { value } if value == plan.surrogate.slope)
                && matches!(program[scale], Node::Constant { value } if value == plan.surrogate.scale),
            Node::EagerBooleanAnd {left,right} | Node::EagerBooleanOr {left,right}
            | Node::BooleanAnd { left, right } | Node::BooleanOr { left, right } => states > 0
                && left < i && right < i && boolean_node(&program[left]) && boolean_node(&program[right]),
            Node::BooleanNot { arg } => states > 0 && arg < i && boolean_node(&program[arg]),
            Node::Equality { left, right, .. } => states > 0 && left < i && right < i,
            Node::Select { condition, yes, no, boolean } => states > 0 && condition < i && yes < i && no < i
                && (!boolean || boolean_node(&program[yes]) && boolean_node(&program[no])),
            Node::Time => states > 0 && plan.clock.is_some(),
            Node::TimeWord { word, .. } => plan.dynamic.is_some() && plan.clock.is_some() && word < 2,
            Node::ElapsedCompare { low, high, right, constant, .. } => plan.dynamic.is_some() && plan.clock.is_some()
                && low < i && high < i && right < i && integer_node(&program[low]) && integer_node(&program[high])
                && !integer_node(&program[right]) && constant.is_none_or(|v| v.is_finite()),
            Node::ClockTime { index } => plan.dynamic.as_ref().and_then(|d| d.clocks.as_ref())
                .is_some_and(|clocks| index < clocks.dts.len()),
            Node::Noise { stream } | Node::UniformNoise { stream } => states > 0 && stream < 16 && plan.noise_streams.is_some(),
            Node::Voltage => true,
            Node::State { index } => index < states,
            Node::RefractoryActive { index } => index < states,
            Node::Constant { value } => value.is_finite(),
            Node::Parameter { bank, index } => {
                bank < plan.masks.len() && index < plan.masks[bank].len()
            }
            Node::NeuronParameter { bank, index } => {
                states > 0 && bank < plan.masks.len() && index < plan.masks[bank].len()
            }
            Node::MappedParameter { bank, mapping } => {
                states > 0 && bank < plan.masks.len() && plan.dynamic.as_ref()
                    .and_then(|d| d.parameter_maps.get(mapping)).is_some_and(|indices|
                        !indices.is_empty() && indices.iter().all(|&i| i < plan.masks[bank].len()))
            }
            Node::TimedParameter { bank, rows, columns, epsilon, k, time, index } => {
                (plan.dynamic.is_some() || plan.state_equations.is_some() || plan.equations.is_some()) && states > 0 && time < i && index < i
                    && rows > 0 && columns > 0 && bank < plan.masks.len()
                    && rows.checked_mul(columns).is_some_and(|n| n == plan.masks[bank].len())
                    && epsilon.is_finite() && epsilon > 0.0 && k > 0 && k <= (1u64 << 53) && k.is_power_of_two()
            }
            Node::FloorDiv { left, right } | Node::Modulo { left, right }
            | Node::Min { left, right } | Node::Max { left, right } => states > 0 && left < i && right < i,
            Node::Add { left, right }
            | Node::Sub { left, right }
            | Node::Mul { left, right }
            | Node::Div { left, right } => left < i && right < i,
            Node::Pow { arg, value } => arg < i && value.is_finite(),
            Node::Neg { arg }
            | Node::Exp { arg }
            | Node::Log { arg }
            | Node::Tanh { arg }
            | Node::Sqrt { arg }
            | Node::Sin { arg }
            | Node::Cos { arg } | Node::Math { arg, .. } => arg < i,
        };
        ensure(valid, "invalid equation SSA reference or constant")?;
        let floating = match *node {
            Node::Select { condition, yes, no, .. } => vec![condition,yes,no],
            Node::SurrogateStep { arg, slope, scale, .. } => vec![arg,slope,scale],
            Node::TimedParameter { time, index, .. } => vec![time,index],
            Node::Add { left, right } | Node::Sub { left, right } | Node::Mul { left, right }
            | Node::FloorDiv { left, right } | Node::Modulo { left, right }
            | Node::Div { left, right } | Node::Min { left, right } | Node::Max { left, right }
            | Node::Equality { left, right, .. } => vec![left,right],
            Node::Neg { arg } | Node::Exp { arg } | Node::Log { arg } | Node::Tanh { arg }
            | Node::Sqrt { arg } | Node::Sin { arg } | Node::Cos { arg } | Node::Pow { arg, .. } | Node::Math { arg, .. } => vec![arg],
            _ => vec![],
        };
        ensure(floating.iter().all(|&k| !integer_node(&program[k])), "integer operands require explicit numeric conversion")?;
    }
    Ok(())
}
pub(super) fn boolean_node(node: &Node) -> bool {
    matches!(node, Node::EagerBooleanAnd {..} | Node::EagerBooleanOr {..} | Node::Select { boolean: true, .. } | Node::Sequence { boolean: true, .. } | Node::SurrogateStep { .. } | Node::BooleanAnd { .. } | Node::BooleanOr { .. }
        | Node::BooleanNot { .. } | Node::Equality { .. } | Node::IntegerCompare { .. }
        | Node::DiscreteCompare { .. } | Node::ElapsedCompare { .. } | Node::BooleanCast { .. }) || matches!(node, Node::Constant { value } if *value == 0.0 || *value == 1.0)
}
pub(super) fn integer_node(node: &Node) -> bool {
    matches!(node, Node::IntegerSequence { .. } | Node::Poisson { .. } | Node::IntegerConstant { .. } | Node::IntegerState { .. } | Node::IntegerParameter { .. }
        | Node::IntegerNeuronParameter { .. } | Node::IntegerMappedParameter { .. } | Node::IntegerParameterGather { .. } | Node::IntegerCast { .. }
        | Node::IntegerBinary { .. } | Node::IntegerSelect { .. } | Node::IntegerNeg { .. } | Node::TimeWord { .. })
}
fn values(program: &Program, states: &[f64], weights: &[Vec<f64>], neuron: usize, time: f64, noise: &[f64], clock_times: &[f64], parameter_maps: &[Vec<usize>]) -> Result<[f64; 128]> {
    Ok(evaluated(program, states, weights, neuron, time, noise, clock_times, parameter_maps)?.0)
}
fn evaluated(program: &Program, states: &[f64], weights: &[Vec<f64>], neuron: usize, time: f64, noise: &[f64], clock_times: &[f64], parameter_maps: &[Vec<usize>]) -> Result<([f64; 128], [bool; 128])> {
    let mut x = [0.0_f64; 128];
    let mut visited = [false; 128];
    if program.iter().any(|node| matches!(node, Node::Poisson { .. } | Node::Select { .. } | Node::IntegerSelect { .. } | Node::BooleanAnd { .. } | Node::BooleanOr { .. })) {
        visit_value(program.len() - 1, program, states, weights, neuron, time, noise, clock_times, parameter_maps, &mut x, &mut visited)?;
        return Ok((x, visited));
    }
    for (i, node) in program.iter().enumerate() {
        x[i] = node_value(node, &x, states, weights, neuron, time, noise, clock_times, parameter_maps)?;
        visited[i] = true;
    }
    Ok((x, visited))
}
fn node_value(node: &Node, x: &[f64; 128], states: &[f64], weights: &[Vec<f64>], neuron: usize, time: f64, noise: &[f64], clock_times: &[f64], parameter_maps: &[Vec<usize>]) -> Result<f64> {
    let value = match *node {
            Node::EagerBooleanAnd {left,right} => x[left]*x[right],
            Node::EagerBooleanOr {left,right} => x[left]+x[right]-x[left]*x[right],
            Node::Sequence { right, .. } | Node::IntegerSequence { right, .. } => x[right],
            Node::Poisson { rate, stream } => {
                let key=poisson_ir::decode_key(noise,stream)?;
                poisson_cache::sample(stream,x[rate],key,noise.get(80+stream).is_some_and(|&v|v==1.))? as f64
            }
            Node::TimeWord { word, float32 } => {
                let timestamp = if float32 { (time as f32) as f64 } else { time };
                ensure(timestamp.is_finite(), "nonfinite precise timestamp")?;
                ((timestamp.to_bits() >> (32*word)) as u32 as i32) as f64
            }
            Node::ElapsedCompare { low, high, right, kind, constant } => {
                let bits=(int32(x[low])? as u32 as u64) | ((int32(x[high])? as u32 as u64)<<32);
                let timestamp=f64::from_bits(bits);let elapsed=time-timestamp;
                ensure(timestamp.is_finite() && elapsed.is_finite(), "nonfinite precise timestamp difference")?;
                if kind.test(elapsed,constant.unwrap_or(x[right])) {1.0} else {0.0}
            }
            Node::IntegerConstant { value } => value as f64,
            Node::IntegerState { index } => int32(states[index])? as f64,
            Node::IntegerParameter { bank, index } => int32(weights[bank][index])? as f64,
            Node::IntegerNeuronParameter { bank, index } => int32(weights[bank][index+neuron])? as f64,
            Node::IntegerMappedParameter { bank, mapping } => int32(weights[bank][parameter_maps[mapping][neuron]])? as f64,
            Node::IntegerParameterGather { bank, index } => int32(weights[bank][gather_index(x[index],weights[bank].len())?])? as f64,
            Node::ParameterGather { bank, index } => weights[bank][gather_index(x[index],weights[bank].len())?],
            Node::IntegerCast { arg } => int32(x[arg].trunc())? as f64,
            Node::IntegerFloat { arg } => x[arg],
            Node::BooleanCast { arg } => if x[arg] != 0.0 { 1.0 } else { 0.0 },
            Node::IntegerCompare { left, right, kind } | Node::DiscreteCompare { left, right, kind } =>
                if kind.test(x[left], x[right]) { 1.0 } else { 0.0 },
            Node::IntegerNeg { arg } => int32(x[arg])?.wrapping_neg() as f64,
            Node::IntegerBinary { left, right, kind } => {
                let a=int32(x[left])?; let b=int32(x[right])?;
                (match kind { IntegerBinary::Add=>a.wrapping_add(b), IntegerBinary::Sub=>a.wrapping_sub(b),
                    IntegerBinary::Mul=>a.wrapping_mul(b), IntegerBinary::Min=>a.min(b), IntegerBinary::Max=>a.max(b),
                    IntegerBinary::BitAnd=>a&b, IntegerBinary::BitOr=>a|b, IntegerBinary::BitXor=>a^b,
                    IntegerBinary::LeftShift | IntegerBinary::RightShift => {
                        ensure((0..32).contains(&b),"integer shift requires count 0..31")?;
                        if matches!(kind,IntegerBinary::LeftShift) {a.wrapping_shl(b as u32)} else {a>>b}
                    }
                    IntegerBinary::FloorDiv | IntegerBinary::Mod => {
                        ensure(b!=0,"integer division by zero")?;
                        // Widen before division: MIN/-1 has a defined wrapping
                        // quotient and zero remainder instead of C/C++ UB.
                        let mut q=(a as i64)/(b as i64); let mut r=(a as i64)%(b as i64);
                        if r!=0 && (r<0)!=(b<0) {q-=1;r+=b as i64;}
                        if matches!(kind,IntegerBinary::FloorDiv) {q as i32} else {r as i32}
                    }
                }) as f64
            }
            Node::IntegerSelect { condition, yes, no } => {
                ensure(x[condition] == 0.0 || x[condition] == 1.0, "integer select condition must be binary")?;
                x[if x[condition] == 1.0 { yes } else { no }]
            }
            Node::SurrogateStep { arg, inclusive, .. } => if if inclusive { x[arg] >= 0.0 } else { x[arg] > 0.0 } { 1.0 } else { 0.0 },
            Node::BooleanAnd { left, right } => if x[left] == 0.0 { 0.0 } else { x[right] },
            Node::BooleanOr { left, right } => if x[left] == 1.0 { 1.0 } else { x[right] },
            Node::BooleanNot { arg } => 1.0 - x[arg],
            Node::Equality { left, right, unequal } => if (x[left] == x[right]) != unequal { 1.0 } else { 0.0 },
            Node::Select { condition, yes, no, .. } => {
                ensure(x[condition] == 0.0 || x[condition] == 1.0, "equation select condition must be binary")?;
                x[if x[condition] == 1.0 { yes } else { no }]
            }
            Node::Time => time,
            Node::ClockTime { index } => *clock_times.get(index).ok_or("missing dynamic clock time")?,
            Node::Noise { stream } | Node::UniformNoise { stream } => noise[stream],
            Node::Voltage => states[0],
            Node::State { index } => states[index],
            Node::RefractoryActive { index } => if states[index] == 0.0 { 1.0 } else { 0.0 },
            Node::Constant { value } => value,
            Node::Parameter { bank, index } => weights[bank][index],
            Node::NeuronParameter { bank, index } => weights[bank][index + neuron],
            Node::MappedParameter { bank, mapping } => weights[bank][parameter_maps[mapping][neuron]],
            Node::TimedParameter { bank, rows, columns, epsilon, k, time, index } => {
                weights[bank][timed_index(rows, columns, epsilon, k, x[time], x[index])?]
            }
            Node::Min { left, right } => x[left].min(x[right]),
            Node::Max { left, right } => x[left].max(x[right]),
            Node::Add { left, right } => x[left] + x[right],
            Node::Sub { left, right } => x[left] - x[right],
            Node::Mul { left, right } => x[left] * x[right],
            Node::Div { left, right } => x[left] / x[right],
            Node::FloorDiv { left, right } => {ensure(x[right]!=0.0,"floor division by zero")?;(x[left]/x[right]).floor()},
            Node::Modulo { left, right } => {
                ensure(x[right]!=0.0,"modulo by zero")?;
                let r=x[left]%x[right];r + if r!=0.0 && (r<0.0)!=(x[right]<0.0) {x[right]} else {0.0*x[right]}
            },
            Node::Neg { arg } => -x[arg],
            Node::Exp { arg } => x[arg].exp(),
            Node::Log { arg } => x[arg].ln(),
            Node::Math { arg, kind } => kind.value(x[arg]),
            Node::Tanh { arg } => x[arg].tanh(),
            Node::Sqrt { arg } => x[arg].sqrt(),
            Node::Sin { arg } => x[arg].sin(),
            Node::Cos { arg } => x[arg].cos(),
            Node::Pow { arg, value } => x[arg].powf(value),
        };
    ensure(value.is_finite(), "nonfinite equation value/domain error")?;
    Ok(value)
}
fn gather_index(value: f64, length: usize) -> Result<usize> {
    let index = int32(value)?;
    ensure(index >= 0 && (index as usize) < length, "parameter gather index outside bank")?;
    Ok(index as usize)
}
fn visit_value(i: usize, program: &Program, states: &[f64], weights: &[Vec<f64>], neuron: usize, time: f64, noise: &[f64], clock_times: &[f64], parameter_maps: &[Vec<usize>], x: &mut [f64; 128], visited: &mut [bool; 128]) -> Result<()> {
    if visited[i] { return Ok(()); }
    let mut dependencies = [0; 2];
    let count = match program[i] {
        Node::ElapsedCompare { low, high, right, constant, .. } => {
            if constant.is_none() {visit_value(right, program, states, weights, neuron, time, noise, clock_times, parameter_maps, x, visited)?;}
            dependencies=[low,high];2
        }
        Node::SurrogateStep { arg, slope, scale, .. } => {
            visit_value(slope, program, states, weights, neuron, time, noise, clock_times, parameter_maps, x, visited)?;
            visit_value(scale, program, states, weights, neuron, time, noise, clock_times, parameter_maps, x, visited)?;
            dependencies[0] = arg; 1
        }
        Node::BooleanAnd { left, right } | Node::BooleanOr { left, right } => {
            visit_value(left, program, states, weights, neuron, time, noise, clock_times, parameter_maps, x, visited)?;
            let skip = if matches!(program[i], Node::BooleanAnd { .. }) { x[left] == 0.0 } else { x[left] == 1.0 };
            dependencies[0] = right; if skip { 0 } else { 1 }
        }
        Node::Select { condition, yes, no, .. } | Node::IntegerSelect { condition, yes, no } => {
            visit_value(condition, program, states, weights, neuron, time, noise, clock_times, parameter_maps, x, visited)?;
            ensure(x[condition] == 0.0 || x[condition] == 1.0, "equation select condition must be binary")?;
            dependencies[0] = if x[condition] == 1.0 { yes } else { no }; 1
        }
        Node::Poisson { rate, stream } => {dependencies[0]=rate;usize::from(poisson_cache::needs_rate(stream))}
        Node::TimedParameter { time, index, .. } => { dependencies = [time, index]; 2 }
        Node::ParameterGather { index, .. } | Node::IntegerParameterGather { index, .. } => { dependencies[0] = index; 1 }
        Node::IntegerBinary { left, right, .. } | Node::IntegerCompare { left, right, .. } | Node::DiscreteCompare { left, right, .. }
        | Node::Equality { left, right, .. } | Node::Min { left, right } | Node::Max { left, right } | Node::Add { left, right }
        | Node::Sequence { left, right, .. } | Node::IntegerSequence { left, right }
        | Node::EagerBooleanAnd {left,right} | Node::EagerBooleanOr {left,right}
        | Node::FloorDiv { left, right } | Node::Modulo { left, right }
        | Node::Sub { left, right } | Node::Mul { left, right } | Node::Div { left, right } => {
            dependencies = [left, right]; 2
        }
        Node::IntegerCast { arg } | Node::IntegerFloat { arg } | Node::IntegerNeg { arg } | Node::BooleanCast { arg }
        | Node::BooleanNot { arg } | Node::Neg { arg } | Node::Exp { arg } | Node::Log { arg } | Node::Tanh { arg }
        | Node::Sqrt { arg } | Node::Sin { arg } | Node::Cos { arg } | Node::Pow { arg, .. } | Node::Math { arg, .. } => {
            dependencies[0] = arg; 1
        }
        _ => 0,
    };
    for &dependency in &dependencies[..count] {
        visit_value(dependency, program, states, weights, neuron, time, noise, clock_times, parameter_maps, x, visited)?;
    }
    x[i] = node_value(&program[i], x, states, weights, neuron, time, noise, clock_times, parameter_maps)?;
    visited[i] = true;
    Ok(())
}
pub fn forward(program: &Program, v: f64, weights: &[Vec<f64>]) -> Result<f64> {
    forward_states(program, &[v], weights, 0, 0.0, &[])
}

pub(super) fn bitwise_used(program: &Program) -> bool {
    program.iter().any(|n|matches!(n,Node::IntegerBinary{kind:IntegerBinary::BitAnd
        |IntegerBinary::BitOr|IntegerBinary::BitXor|IntegerBinary::LeftShift|IntegerBinary::RightShift,..}))
}
pub(super) fn sequence_used(program: &Program) -> bool {
    program.iter().any(|n|matches!(n,Node::Sequence{..}|Node::IntegerSequence{..}))
}
pub(super) fn eager_boolean_used(program: &Program) -> bool {
    program.iter().any(|n|matches!(n,Node::EagerBooleanAnd{..}|Node::EagerBooleanOr{..}))
}
pub fn forward_states(program: &Program, states: &[f64], weights: &[Vec<f64>], neuron: usize, time: f64, noise: &[f64]) -> Result<f64> {
    forward_clocked(program, states, weights, neuron, time, noise, &[], &[])
}
pub fn forward_clocked(program: &Program, states: &[f64], weights: &[Vec<f64>], neuron: usize, time: f64, noise: &[f64], clock_times: &[f64], parameter_maps: &[Vec<usize>]) -> Result<f64> {
    Ok(values(program, states, weights, neuron, time, noise, clock_times, parameter_maps)?[program.len() - 1])
}
pub fn backward(
    program: &Program,
    v: f64,
    weights: &[Vec<f64>],
    seed: f64,
    gradients: &mut [Vec<f64>],
    masks: &[Vec<f64>],
) -> Result<f64> {
    let mut adjoints = [0.0];
    backward_states(
        program,
        &[v],
        weights,
        seed,
        gradients,
        masks,
        &mut adjoints,
        0,
        0.0,
        &[],
    )?;
    Ok(adjoints[0])
}
pub fn backward_states(
    program: &Program,
    states: &[f64],
    weights: &[Vec<f64>],
    seed: f64,
    gradients: &mut [Vec<f64>],
    masks: &[Vec<f64>],
    adjoints: &mut [f64],
    neuron: usize,
    time: f64,
    noise: &[f64],
) -> Result<()> {
    backward_clocked(program, states, weights, seed, gradients, masks, adjoints, neuron, time, noise, &[], &[])
}
pub fn backward_clocked(
    program: &Program, states: &[f64], weights: &[Vec<f64>], seed: f64,
    gradients: &mut [Vec<f64>], masks: &[Vec<f64>], adjoints: &mut [f64],
    neuron: usize, time: f64, noise: &[f64], clock_times: &[f64], parameter_maps: &[Vec<usize>],
) -> Result<()> {
    backward_clocked_activity(program,states,weights,seed,gradients,masks,adjoints,neuron,time,noise,
        clock_times,parameter_maps,None)
}
pub(super) fn backward_clocked_addresses(
    program: &Program, states: &[f64], weights: &[Vec<f64>], seed: f64,
    gradients: &mut [Vec<f64>], masks: &[Vec<f64>], adjoints: &mut [f64],
    neuron: usize, time: f64, noise: &[f64], clock_times: &[f64], parameter_maps: &[Vec<usize>],
    cells: &[usize], detached: &[bool],
) -> Result<()> {
    backward_clocked_activity(program,states,weights,seed,gradients,masks,adjoints,neuron,time,noise,
        clock_times,parameter_maps,Some((cells,detached)))
}
fn backward_clocked_activity(
    program: &Program, states: &[f64], weights: &[Vec<f64>], seed: f64,
    gradients: &mut [Vec<f64>], masks: &[Vec<f64>], adjoints: &mut [f64],
    neuron: usize, time: f64, noise: &[f64], clock_times: &[f64], parameter_maps: &[Vec<usize>],
    addresses: Option<(&[usize],&[bool])>,
) -> Result<()> {
    let (x,visited)=evaluated(program,states,weights,neuron,time,noise,clock_times,parameter_maps)?;
    let active=vjp_activity(program,&x,&visited,masks,addresses,neuron,parameter_maps)?;
    let mut d=[0.;128];d[program.len()-1]=seed;
    backward_values_active(program,&x,d,gradients,masks,adjoints,neuron,parameter_maps,Some(&active))
}
fn backward_values_active(program: &Program, x: &[f64; 128], mut d: [f64;128],
    gradients: &mut [Vec<f64>], masks: &[Vec<f64>], adjoints: &mut [f64],
    neuron: usize, parameter_maps: &[Vec<usize>], active: Option<&[bool;128]>) -> Result<()> {
    for (i, node) in program.iter().enumerate().rev() {
        // Discard derivatives of unrequested leaves before singular local
        // derivatives are evaluated. Active singularities remain errors.
        if active.is_some_and(|a|!a[i]) {d[i] = 0.;continue;}
        let g = d[i];
        if g == 0.0 {
            continue;
        }
        match *node {
            Node::EagerBooleanAnd {left,right} => {d[left]+=g*x[right];d[right]+=g*x[left];}
            Node::EagerBooleanOr {left,right} => {d[left]+=g*(1.-x[right]);d[right]+=g*(1.-x[left]);}
            Node::Sequence { right, .. } => d[right] += g,
            Node::IntegerSequence { .. } => {},
            Node::Poisson { .. } | Node::IntegerConstant { .. } | Node::IntegerState { .. } | Node::IntegerParameter { .. }
            | Node::IntegerNeuronParameter { .. } | Node::IntegerMappedParameter { .. } | Node::IntegerParameterGather { .. } | Node::IntegerCast { .. }
            | Node::IntegerFloat { .. } | Node::IntegerBinary { .. } | Node::IntegerCompare { .. }
            | Node::IntegerSelect { .. } | Node::DiscreteCompare { .. } | Node::BooleanCast { .. } | Node::IntegerNeg { .. } => {}
            Node::SurrogateStep { arg, slope, scale, .. } => {
                let z = 1.0 + x[slope] * x[arg].abs(); d[arg] += g * x[scale] / (z * z);
            }
            Node::BooleanAnd { left, right } => {
                if x[left] == 0.0 { d[left] += g; }
                else { d[left] += g * x[right]; d[right] += g; }
            }
            Node::BooleanOr { left, right } => {
                if x[left] == 1.0 { d[left] += g; }
                else { d[left] += g * (1.0 - x[right]); d[right] += g; }
            }
            Node::BooleanNot { arg } => d[arg] -= g,
            Node::Equality { .. } => {}
            Node::Select { condition, yes, no, .. } => d[if x[condition] == 1.0 { yes } else { no }] += g,
            Node::Time | Node::TimeWord { .. } | Node::ElapsedCompare { .. } | Node::ClockTime { .. } | Node::Noise { .. } | Node::UniformNoise { .. } => {}
            Node::Voltage => adjoints[0] += g,
            Node::State { index } => adjoints[index] += g,
            Node::Constant { .. } => {}
            Node::RefractoryActive { .. } => {}
            Node::Parameter { bank, index } => gradients[bank][index] += g * masks[bank][index],
            Node::NeuronParameter { bank, index } => gradients[bank][index + neuron] += g * masks[bank][index + neuron],
            Node::MappedParameter { bank, mapping } => {
                let index = parameter_maps[mapping][neuron]; gradients[bank][index] += g * masks[bank][index];
            }
            Node::ParameterGather { bank, index } => {
                // x is reconstructed from the taped pre-action context, never
                // from mutable live state after subsequent writes.
                let slot = gather_index(x[index], masks[bank].len())?;
                gradients[bank][slot] += g * masks[bank][slot];
            }
            Node::TimedParameter { bank, rows, columns, epsilon, k, time, index } => {
                let slot = timed_index(rows, columns, epsilon, k, x[time], x[index])?;
                gradients[bank][slot] += g * masks[bank][slot];
                // Piecewise-constant sampling has no time/index derivative.
            }
            Node::Min { left, right } => d[if x[left] <= x[right] { left } else { right }] += g,
            Node::Max { left, right } => d[if x[left] >= x[right] { left } else { right }] += g,
            Node::Add { left, right } => {
                d[left] += g;
                d[right] += g;
            }
            Node::Sub { left, right } => {
                d[left] += g;
                d[right] -= g;
            }
            Node::Mul { left, right } => {
                d[left] += g * x[right];
                d[right] += g * x[left];
            }
            Node::FloorDiv { .. } => {}
            Node::Modulo { left, right } => {
                d[left] += g;
                // Reconstruct the selected integer quotient from the actual
                // remainder, including division rounded onto a boundary.
                d[right] -= g * ((x[left]-x[i])/x[right]).round();
            }
            Node::Div { left, right } => {
                d[left] += g / x[right];
                d[right] -= g * x[left] / (x[right] * x[right]);
            }
            Node::Neg { arg } => d[arg] -= g,
            Node::Exp { arg } => d[arg] += g * x[i],
            Node::Log { arg } => d[arg] += g / x[arg],
            Node::Math { arg, kind } => d[arg] += g * kind.derivative(x[arg]),
            Node::Tanh { arg } => d[arg] += g * (1.0 - x[i] * x[i]),
            Node::Sqrt { arg } => d[arg] += g / (2.0 * x[i]),
            Node::Sin { arg } => d[arg] += g * x[arg].cos(),
            Node::Cos { arg } => d[arg] -= g * x[arg].sin(),
            Node::Pow { arg, value } => {
                if value != 0.0 {
                    d[arg] += g * value * x[arg].powf(value - 1.0);
                }
            }
        }
    }
    ensure(
        adjoints.iter().chain(d.iter()).all(|v| v.is_finite()),
        "nonfinite equation derivative",
    )?;
    Ok(())
}

fn timed_index(rows: usize, columns: usize, epsilon: f64, k: u64, time: f64, index: f64) -> Result<usize> {
    ensure(time.is_finite() && index.is_finite() && index.fract() == 0.0
        && index >= 0.0 && index < columns as f64, "TimedArray column index out of range")?;
    let row = ((time / epsilon + 0.5) / k as f64).max(0.0).min((rows - 1) as f64) as usize;
    Ok(row * columns + index as usize)
}

/// One stream has one distribution within an execution context. Repeated nodes
/// reuse a draw; accidentally assigning two distributions to it is rejected.
pub(super) fn noise_masks<'a>(programs: impl Iterator<Item = &'a Program>) -> (u16, u16) {
    let mut normal = 0; let mut uniform = 0;
    for node in programs.flatten() {
        match node {
            Node::Noise { stream } if *stream < 16 => normal |= 1u16 << stream,
            Node::UniformNoise { stream } if *stream < 16 => uniform |= 1u16 << stream,
            _ => (),
        }
    }
    (normal, uniform)
}

/// Score every reached random site once, independently of output pathwise
/// adjoints and detached writes. The stopped loss includes batch normalization.
pub(super) fn backward_poisson<'a>(programs: impl IntoIterator<Item=&'a Program>, states: &[f64], weights: &[Vec<f64>], loss: f64, boundary: &[f64; 16],
    gradients: &mut [Vec<f64>], masks: &[Vec<f64>], adjoints: &mut [f64],
    neuron: usize, time: f64, noise: &[f64], clock_times: &[f64], parameter_maps: &[Vec<usize>],
    cells: &[usize], detached: &[bool]) -> Result<()> {
    let mut seen = 0u16;
    for program in programs {
        if !program.iter().any(|n| matches!(n, Node::Poisson { .. })) {continue;}
        let (x, visited) = evaluated(program, states, weights, neuron, time, noise, clock_times, parameter_maps)?;
        let mut d = [0.; 128];
        for (i, node) in program.iter().enumerate() {
            if let Node::Poisson { rate, stream } = *node {
                if !visited[i] || seen & (1 << stream) != 0 || !poisson_cache::owns_score(stream) {continue;}
                seen |= 1 << stream;
                // Constant/frozen integer controls have no continuous rate VJP.
                if !poisson_ir::rate_differentiable(program, rate) {continue;}
                if x[rate] == 0. {
                    ensure(boundary[stream].is_finite(), "missing Poisson boundary replay")?;
                    d[rate] += boundary[stream];
                    continue;
                }
                let score = poisson::score(x[rate], int32(x[i])?)
                    .map_err(|e| format!("Poisson likelihood derivative: {e:?}"))?;
                ensure(score.is_finite(), "nonfinite Poisson likelihood derivative")?;
                d[rate] += loss * score;
            }
        }
        let active=vjp_activity(program,&x,&visited,masks,Some((cells,detached)),neuron,parameter_maps)?;
        backward_values_active(program,&x,d,gradients,masks,adjoints,neuron,parameter_maps,Some(&active))?;
    }
    Ok(())
}

/// Actual visited-branch reachability to requested continuous leaves. The
/// fixed array adds no heap allocation; masks and recorded physical addresses
/// determine activity without changing the forward model's domain checks.
fn vjp_activity(program: &Program, x: &[f64;128], visited: &[bool;128], masks: &[Vec<f64>],
    addresses: Option<(&[usize],&[bool])>, neuron: usize, maps: &[Vec<usize>]) -> Result<[bool;128]> {
    let state_active=|index:usize| -> Result<bool> {
        if let Some((cells,detached))=addresses {
            ensure(index<cells.len()&&cells[index]<detached.len(),"invalid equation state address")?;
            Ok(!detached[cells[index]])
        }else{Ok(true)}
    };
    let mut active=[false;128];
    for (i,node) in program.iter().enumerate() {
        if !visited[i]{continue;}
        active[i]=match *node {
            Node::EagerBooleanAnd{left,right}|Node::EagerBooleanOr{left,right}=>active[left]||active[right],
            Node::Sequence { right, .. } => active[right],
            Node::Voltage=>state_active(0)?,
            Node::State{index}=>state_active(index)?,
            Node::Parameter{bank,index}=>masks[bank][index]!=0.,
            Node::NeuronParameter{bank,index}=>masks[bank][index+neuron]!=0.,
            Node::MappedParameter{bank,mapping}=>masks[bank][maps[mapping][neuron]]!=0.,
            Node::ParameterGather{bank,index}=>masks[bank][gather_index(x[index],masks[bank].len())?]!=0.,
            Node::TimedParameter{bank,rows,columns,epsilon,k,time,index}=>
                masks[bank][timed_index(rows,columns,epsilon,k,x[time],x[index])?]!=0.,
            Node::Select{condition,yes,no,..}=>active[if x[condition]==1. {yes}else{no}],
            Node::Min{left,right}=>active[if x[left]<=x[right] {left}else{right}],
            Node::Max{left,right}=>active[if x[left]>=x[right] {left}else{right}],
            Node::SurrogateStep{arg,..}=>active[arg],
            Node::BooleanAnd{left,right}=>active[left]||x[left]!=0.&&active[right],
            Node::BooleanOr{left,right}=>active[left]||x[left]!=1.&&active[right],
            Node::Add{left,right}|Node::Sub{left,right}|Node::Mul{left,right}|Node::Div{left,right}
                |Node::Modulo{left,right}=>active[left]||active[right],
            Node::Math{arg,..}|Node::Neg{arg}|Node::Exp{arg}|Node::Log{arg}|Node::Tanh{arg}
                |Node::Sqrt{arg}|Node::Sin{arg}|Node::Cos{arg}|Node::Pow{arg,..}|Node::BooleanNot{arg}=>active[arg],
            _=>false,
        };
    }
    Ok(active)
}

/// Test the actual baseline rate VJP after parameter aliases/masks and physical
/// state aliases/detaches have been applied. Structural dependency is only an
/// admission hint: cancellation and a selected constant branch can have zero VJP.
fn rate_has_vjp(program: &Program, x: &[f64;128], visited: &[bool;128], rate: usize, masks: &[Vec<f64>],
    cells: &[usize], detached: &[bool], neuron: usize, maps: &[Vec<usize>]) -> Result<bool> {
    let active=vjp_activity(program,x,visited,masks,Some((cells,detached)),neuron,maps)?;
    if !active[rate] {return Ok(false);}
    let mut gradients: Vec<Vec<f64>> = masks.iter().map(|row| vec![0.;row.len()]).collect();
    let mut adjoints = vec![0.;cells.len()];
    let mut seed = [0.;128];seed[rate] = 1.;
    backward_values_active(program,x,seed,&mut gradients,masks,&mut adjoints,neuron,maps,Some(&active))?;
    ensure(gradients.iter().flatten().all(|v|v.is_finite()), "nonfinite Poisson rate VJP")?;
    let mut states = std::collections::BTreeMap::<usize,f64>::new();
    for (&cell,gradient) in cells.iter().zip(adjoints) {
        ensure(cell < detached.len(), "invalid Poisson rate state address")?;
        if !detached[cell] {*states.entry(cell).or_default() += gradient;}
    }
    ensure(states.values().all(|v|v.is_finite()), "nonfinite Poisson rate state VJP")?;
    Ok(gradients.iter().flatten().chain(states.values()).any(|&v|v != 0.))
}

/// Return reached structural zero-rate sites and the subset requiring replay.
/// The visited mask is essential: invalid rates in unselected branches are inert.
/// The caller explicitly records zero coefficients for cancelled/frozen VJPs.
pub(super) fn zero_poisson_sites<'a>(programs: impl IntoIterator<Item=&'a Program>, states: &[f64], weights: &[Vec<f64>],
    neuron: usize, time: f64, noise: &[f64], clocks: &[f64], maps: &[Vec<usize>],
    masks: &[Vec<f64>], cells: &[usize], detached: &[bool]) -> Result<(u16,u16)> {
    let mut zero = 0; let mut replay = 0;
    for program in programs {
        if !program.iter().any(|node| matches!(node, Node::Poisson {..})) {continue;}
        let (x, visited) = evaluated(program,states,weights,neuron,time,noise,clocks,maps)?;
        for (i,node) in program.iter().enumerate() {
            if let Node::Poisson {rate,stream} = *node {
                if visited[i] && poisson_cache::owns_score(stream) && x[rate] == 0. && zero & (1 << stream) == 0
                    && poisson_ir::rate_differentiable(program,rate) {
                    zero |= 1 << stream;
                    if rate_has_vjp(program,&x,&visited,rate,masks,cells,detached,neuron,maps)? {replay |= 1 << stream;}
                }
            }
        }
    }
    Ok((zero,replay))
}

#[cfg(test)]
mod bitwise_tests {
    use super::*;
    #[test]
    fn exact_signed_edges_and_shift_admission() {
        use IntegerBinary::*;
        for (a,b,kind,expected) in [
            (i32::MIN,31,RightShift,-1), (i32::MAX,1,LeftShift,-2),
            (1,31,LeftShift,i32::MIN), (-1,0,RightShift,-1),
            (i32::MIN,i32::MAX,BitAnd,0), (i32::MIN,i32::MAX,BitOr,-1),
            (-1,i32::MIN,BitXor,i32::MAX),
        ] {
            let program=vec![Node::IntegerConstant{value:a},Node::IntegerConstant{value:b},
                Node::IntegerBinary{left:0,right:1,kind}];
            assert_eq!(forward(&program,0.,&[]).unwrap(),expected as f64);
            assert_eq!(backward(&program,0.,&[],1.,&mut[],&[]).unwrap(),0.);
        }
        for b in [-1,32,i32::MAX] {
            let program=vec![Node::IntegerConstant{value:1},Node::IntegerConstant{value:b},
                Node::IntegerBinary{left:0,right:1,kind:LeftShift}];
            assert!(forward(&program,0.,&[]).unwrap_err().to_string().contains("count 0..31"));
        }
    }
}

#[cfg(test)]
mod sequence_tests {
    use super::*;
    #[test]
    fn discarded_primal_is_checked_without_a_singular_vjp() {
        let program=vec![Node::Voltage,Node::Sqrt{arg:0},Node::Constant{value:2.},
            Node::Sequence{left:1,right:2,boolean:false}];
        assert_eq!(forward(&program,0.,&[]).unwrap(),2.);
        assert_eq!(backward(&program,0.,&[],1.,&mut[],&[]).unwrap(),0.);
        assert!(forward(&program,-1.,&[]).is_err());
        assert!(!poisson_ir::rate_differentiable(&program,3));
    }
    #[test]
    fn sequencing_is_eager_inside_only_the_selected_branch() {
        let program=vec![Node::Constant{value:0.},Node::Constant{value:1.},
            Node::Div{left:1,right:0},Node::Voltage,
            Node::Sequence{left:2,right:3,boolean:false},
            Node::Select{condition:0,yes:4,no:3,boolean:false}];
        assert_eq!(forward(&program,0.25,&[]).unwrap(),0.25);
        assert_eq!(backward(&program,0.25,&[],1.,&mut[],&[]).unwrap(),1.);
        assert!(forward(&program[..5].to_vec(),0.25,&[]).is_err());
    }
    #[test]
    fn discarded_integer_and_exact_integer_result_are_detached() {
        let program=vec![Node::Voltage,Node::IntegerConstant{value:i32::MAX},
            Node::IntegerSequence{left:0,right:1}];
        assert_eq!(forward(&program,0.3,&[]).unwrap(),i32::MAX as f64);
        assert_eq!(backward(&program,0.3,&[],1.,&mut[],&[]).unwrap(),0.);
        assert!(integer_node(&program[2]));
    }
    #[test]
    fn boolean_select_keeps_selected_surrogate_vjp_and_legacy_wire_shape() {
        let program=vec![Node::Constant{value:1.},Node::Constant{value:2.},
            Node::Constant{value:1.},Node::Voltage,
            Node::SurrogateStep{arg:3,slope:1,scale:2,inclusive:false},Node::Constant{value:0.},
            Node::Select{condition:0,yes:4,no:5,boolean:true}];
        assert_eq!(forward(&program,0.3,&[]).unwrap(),1.);
        assert!((backward(&program,0.3,&[],1.,&mut[],&[]).unwrap()-1./(1.6*1.6)).abs()<1e-14);
        assert!(boolean_node(&program[6]));
        let old=serde_json::json!({"op":"select","condition":0,"yes":1,"no":2});
        let node:Node=serde_json::from_value(old.clone()).unwrap();
        assert_eq!(serde_json::to_value(node).unwrap(),old);
    }
    #[test]
    fn eager_boolean_products_use_both_known_gate_values() {
        for (left,right) in [(0.,0.),(0.,1.),(1.,0.),(1.,1.)] {
            let prefix=vec![Node::Voltage,Node::Constant{value:2.},Node::Constant{value:1.},
                Node::SurrogateStep{arg:0,slope:1,scale:2,inclusive:false},Node::Constant{value:right}];
            for is_and in [true,false] {
                let mut program=prefix.clone();program.push(if is_and{Node::EagerBooleanAnd{left:3,right:4}}else{Node::EagerBooleanOr{left:3,right:4}});
                let v=if left==0. {-0.5}else{0.5};let coefficient=if is_and{right}else{1.-right};
                assert_eq!(backward(&program,v,&[],1.,&mut[],&[]).unwrap(),coefficient/4.);
            }
        }
    }
}
