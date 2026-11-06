//! Native sequence-boundary replacement of external TimedArray values.
use super::*;

pub(super) fn static_used(plan:&Plan)->bool {
    plan.dynamic.is_none() && (0..plan.sizes.len()-1).any(|l|
        static_poisson::programs(plan,l).flatten().any(|node|matches!(node,equation::Node::TimedParameter{..})))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Update {
    pub bank: usize,
    pub values: Vec<f64>,
}

pub(super) fn execute(plan: Plan, mut state: State, live: Option<Vec<Vec<f64>>>,
    update: Update, tick: u64, sequence: u64, cache:Option<&poisson_cache::Checkpoint>) -> Result<Output> {
    let spec = plan.dynamic.as_ref();
    ensure(spec.is_some() || plan.state_equations.is_some() || plan.equations.is_some(), "TimedArray update requires an equation plan")?;
    validate_state(&plan, &state)?;
    if let Some(rows) = &live { validate_live(&plan, rows)?; }
    ensure(tick <= (1u64 << 53) && plan.time_at(tick).is_finite(), "invalid input update clock")?;
    let reads = |node:&equation::Node| matches!(node, equation::Node::TimedParameter { bank, .. } if *bank == update.bank);
    ensure(spec.map_or_else(|| (0..plan.sizes.len()-1).any(|l|static_poisson::programs(&plan,l).flatten().any(reads)),
        |spec| spec.program_sets.iter().flatten().flatten().any(reads)),
        "input update bank must be read by a TimedArray node")?;
    ensure(!plan.trainable[update.bank] && update.values.len() == state.weights[update.bank].len()
        && update.values.iter().all(|v| v.is_finite()), "input update requires a frozen bank and finite matching values")?;
    ensure(plan.backend == "cpu" || update.values.iter().all(|&v| (v as f32).is_finite()),
        "GPU timed input values must be representable in float32")?;
    let execution_bytes = if let Some(spec)=spec {
        spec.memory_bytes(live.as_ref().map_or(0, Vec::len), 0, plan.mpi_ranks.unwrap_or(1))
    } else {
        // Retained vector state and its voltage-only result copy, plus bounded
        // program/shape scratch. No trajectory tape or GPU dispatch is needed
        // for an explicit metadata update boundary.
        let nodes=(0..plan.sizes.len()-1).try_fold(0usize, |n,l|
            static_poisson::programs(&plan,l).try_fold(n, |n,p| n.checked_add(p.len())));
        nodes.and_then(|n|n.checked_mul(64)).and_then(|n|
            plan.state_width().checked_mul(live.as_ref().map_or(0,Vec::len))?
                .checked_mul(24)?.checked_add(n)?.checked_add(4096))
    };
    let bytes = execution_bytes
        .and_then(|n|cache.map_or(0,|c|c.entries.len()).checked_mul(768)?.checked_add(n))
        .and_then(|n| state.weights.iter().try_fold(n, |s,w| s.checked_add(w.len().checked_mul(24)?)))
        .and_then(|n| n.checked_add(update.values.len().checked_mul(8)?)).ok_or("input update memory overflow")?;
    ensure(bytes <= plan.max_tape_bytes, "input update exceeds memory budget")?;
    state.weights[update.bank] = update.values;
    validate_state(&plan, &state)?;
    let voltage = if let Some(spec)=spec {spec.voltage.clone()} else if plan.equations.is_some() {
        (0..plan.state_width()).collect()
    } else {
        let mut indices=Vec::new();let mut offset=0;
        for (layer,programs) in plan.state_equations.as_ref().unwrap().iter().enumerate() {
            indices.extend(offset..offset+plan.sizes[layer+1]);
            offset+=programs.len()*plan.sizes[layer+1];
        }
        indices
    };
    let membrane = live.as_ref().map_or_else(Vec::new, |rows|
        rows.iter().map(|row| voltage.iter().map(|&k| row[k]).collect()).collect());
    Ok(Output { poisson_state: None, event_visits: None, clock_state: None, updated_dynamic: None,
        schema: "b2-lif-training-result-v1", state, final_state: if plan.equations.is_some(){None}else{live}, final_membrane: membrane,
        final_tick: Some(tick), noise_sequence: plan.noise_streams.as_ref().map(|_| sequence),
        backend: "cpu", numeric_profile: "native-timed-input-update-f64", gpu_dispatches: 0,
        loss: 0.0, gradients: vec![], initial_gradients: vec![], initial_state_gradients: None,
        spikes: vec![], logits: vec![], tape_bytes: bytes, gradient_scope: "input-boundary-no-gradient",
    })
}
