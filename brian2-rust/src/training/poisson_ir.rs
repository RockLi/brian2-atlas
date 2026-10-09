//! Poisson SSA admission and lossless counter-key transport. Sampling stays in
//! the expression evaluator so rates may depend on the current runtime state.
use super::*;
use equation::{Node, Program};
use std::collections::HashMap;

pub(super) fn encode_keys(out: &mut [f64], plan: &Plan, action: &dynamic::Action,
    sequence: u64, batch: usize, tick: u64) {
    let event = action.event_noise.as_ref().map_or(poisson::Event::Clock, |a|
        a.pending.map_or(poisson::Event::Emission {delay:a.delay}, |id| poisson::Event::Pending {id}));
    for stream in 0..action.noise_streams {
        let key = poisson::key(plan.seed, sequence, batch as u64, action.noise_domain,
            action.noise_entity, tick, stream as u64, event);
        // Four 16-bit limbs survive both f64 and the future f32 device ABI.
        for word in 0..4 {out[16+4*stream+word] = ((key>>(16*word))&0xffff) as f64;}
    }
}
pub(super) fn decode_key(noise: &[f64], stream: usize) -> Result<u64> {
    let start=16+4*stream;
    let limbs=noise.get(start..start+4).ok_or("missing Poisson counter-key context")?;
    let mut key=0;
    for (word,&value) in limbs.iter().enumerate() {
        ensure(value.is_finite() && value>=0. && value<=65535. && value.fract()==0., "invalid Poisson key limb")?;
        key |= (value as u64) << (16*word);
    }
    Ok(key)
}

pub(super) fn operands(node: &Node) -> &'static [&'static str] {
    match node {
        Node::Sequence {..} | Node::IntegerSequence {..} => &["left","right"],
        Node::EagerBooleanAnd{..} | Node::EagerBooleanOr{..} => &["left","right"],
        Node::Poisson {..} => &["rate"],
        Node::Select {..} | Node::IntegerSelect {..} => &["condition","yes","no"],
        Node::SurrogateStep {..} => &["arg","slope","scale"],
        Node::ElapsedCompare {..} => &["low","high","right"],
        Node::TimedParameter {..} => &["time","index"],
        Node::ParameterGather {..} | Node::IntegerParameterGather {..} => &["index"],
        Node::IntegerBinary {..} | Node::IntegerCompare {..} | Node::DiscreteCompare {..}
        | Node::BooleanAnd {..} | Node::BooleanOr {..} | Node::Equality {..}
        | Node::Add {..} | Node::Sub {..} | Node::Mul {..} | Node::Div {..}
        | Node::FloorDiv {..} | Node::Modulo {..} | Node::Min {..} | Node::Max {..} => &["left","right"],
        Node::IntegerCast {..} | Node::IntegerFloat {..} | Node::IntegerNeg {..} | Node::BooleanCast {..}
        | Node::BooleanNot {..} | Node::Neg {..} | Node::Exp {..} | Node::Log {..} | Node::Tanh {..}
        | Node::Sqrt {..} | Node::Sin {..} | Node::Cos {..} | Node::Pow {..} | Node::Math {..} => &["arg"],
        _ => &[],
    }
}

pub(super) fn validate(programs: &[Program], action: &dynamic::Action, _plan: &Plan) -> Result<()> {
    if !programs.iter().flatten().any(|n|matches!(n,Node::Poisson{..})) {return Ok(());}
    let (normal,uniform)=equation::noise_masks(programs.iter());
    let mut sites=HashMap::new(); let mut intern=HashMap::new();
    // Intern canonical DAG nodes across every output. Reindex operands first;
    // unrelated prefixes and common-subexpression layout cannot change identity.
    // This is linear in SSA size, with exact string equality (no hash-only test).
    for program in programs {
        let mut ids=Vec::<usize>::new();
        for node in program {
            let mut value=serde_json::to_value(node)?;
            for &field in operands(node) {
                let index=value[field].as_u64().ok_or("invalid Poisson SSA operand")? as usize;
                value[field]=serde_json::json!(ids.get(index).ok_or("invalid Poisson SSA dependency")?);
            }
            let key=serde_json::to_string(&value)?;let next=intern.len();
            let id=*intern.entry(key).or_insert(next);
            if let Node::Poisson{rate,stream}=*node {
                ensure(stream<action.noise_streams && stream<16,"Poisson stream outside action")?;
                ensure((normal|uniform)&(1<<stream)==0,"dynamic noise stream mixes distributions")?;
                let signature=ids[rate];
                ensure(sites.insert(stream,signature).is_none_or(|old|old==signature),
                    "Poisson stream reused with different rate expressions")?;
            }
            ids.push(id);
        }
    }
    Ok(())
}

/// Conservative structural check for a continuous rate path. Discrete choices,
/// integer samples and fixed RNG/time values contribute no pathwise derivative.
pub(super) fn rate_differentiable(program: &Program, root: usize) -> bool {
    let mut active=[false;128];
    for (i,node) in program.iter().enumerate().take(root+1) {
        active[i]=match node {
            Node::Sequence {right,..} => active[*right],
            Node::Voltage | Node::State{..} | Node::Parameter{..} | Node::NeuronParameter{..}
            | Node::MappedParameter{..} | Node::ParameterGather{..} | Node::TimedParameter{..} => true,
            Node::IntegerFloat{..} | Node::FloorDiv{..} | Node::IntegerCompare{..}
            | Node::DiscreteCompare{..} | Node::BooleanCast{..} | Node::Equality{..} | Node::ElapsedCompare{..} => false,
            n if equation::integer_node(n) => false,
            _ => {
                let value=serde_json::to_value(node).expect("SSA serializes");
                operands(node).iter().any(|&field|active[value[field].as_u64().unwrap() as usize])
            }
        };
    }
    active[root]
}

/// Normalize rate dependencies to physical cells and canonical parameter slots.
/// A repeated stochastic address denotes one draw, whose first actual visit
/// supplies the rate. Different rate definitions cannot silently share it.
pub(super) fn validate_actions(spec: &dynamic::Spec) -> Result<()> {
    let mut sites=HashMap::new();let mut intern=HashMap::new();
    for action in &spec.actions {
        let Some(programs)=action.program_set.and_then(|i|spec.program_sets.get(i)) else{continue;};
        if !programs.iter().flatten().any(|n|matches!(n,Node::Poisson{..})){continue;}
        for program in programs {
            let mut ids=Vec::new();
            for node in program {
                let mut value=serde_json::to_value(node)?;
                for &field in operands(node) {
                    let index=value[field].as_u64().ok_or("invalid shared Poisson operand")? as usize;
                    value[field]=serde_json::json!(ids.get(index).ok_or("invalid shared Poisson dependency")?);
                }
                match *node {
                    Node::Voltage|Node::State{..}|Node::IntegerState{..}=>{
                        let index=match *node{Node::Voltage=>0,Node::State{index}|Node::IntegerState{index}=>index,_=>unreachable!()};
                        let address=if let Some(read)=action.indirect.as_ref().and_then(|a|a.reads.get(&index)){
                            serde_json::to_value(read)?
                        }else{serde_json::json!(action.reads.get(index).ok_or("shared Poisson state outside context")?)};
                        value=serde_json::json!({"op":if matches!(node,Node::IntegerState{..}){"integer_state"}else{"state"},"address":address});
                    }
                    Node::NeuronParameter{bank,index}|Node::IntegerNeuronParameter{bank,index}=>{
                        value=serde_json::json!({"op":if matches!(node,Node::IntegerNeuronParameter{..}){"integer_parameter"}else{"parameter"},"bank":bank,
                            "index":index.checked_add(action.parameter_index).ok_or("shared Poisson parameter overflow")?});
                    }
                    Node::MappedParameter{bank,mapping}|Node::IntegerMappedParameter{bank,mapping}=>{
                        let index=spec.parameter_maps.get(mapping).and_then(|r|r.get(action.parameter_index)).ok_or("shared Poisson map outside context")?;
                        value=serde_json::json!({"op":if matches!(node,Node::IntegerMappedParameter{..}){"integer_parameter"}else{"parameter"},"bank":bank,"index":index});
                    }
                    Node::Time|Node::TimeWord{..}=>{value["clock"]=serde_json::json!(action.clock.unwrap_or(0));}
                    Node::Noise{stream}|Node::UniformNoise{stream}|Node::Poisson{stream,..}=>{
                        value["site"]=serde_json::to_value(poisson_cache::Site::new(action,stream))?;
                    }
                    _=>(),
                }
                let key=serde_json::to_string(&value)?;let next=intern.len();let id=*intern.entry(key).or_insert(next);
                if let Node::Poisson{rate,stream}=*node {
                    let signature=*ids.get(rate).ok_or("shared Poisson rate outside context")?;
                    ensure(sites.insert(poisson_cache::Site::new(action,stream),signature).is_none_or(|old|old==signature),
                        "Poisson site reused across actions with incompatible rate definitions")?;
                }
                ids.push(id);
            }
        }
    }
    Ok(())
}
pub(super) fn shared_actions(spec:&dynamic::Spec)->bool {
    let mut sites=std::collections::HashSet::new();
    for action in &spec.actions {
        let Some(programs)=action.program_set.and_then(|i|spec.program_sets.get(i)) else{continue;};
        let mut streams=0u16;
        for node in programs.iter().flatten(){if let Node::Poisson{stream,..}=node{if *stream<16{streams|=1<<stream;}}}
        for stream in 0..16 {if streams&(1<<stream)!=0&&!sites.insert(poisson_cache::Site::new(action,stream)){return true;}}
    }
    false
}

/// Whether a CPU execution may need a zero-rate counterfactual trajectory.
pub(super) fn boundary_possible(spec: &dynamic::Spec) -> bool {
    spec.program_sets.iter().flatten().any(|program| program.iter().any(|node|
        matches!(*node, Node::Poisson {rate,..} if rate_differentiable(program,rate))))
}
