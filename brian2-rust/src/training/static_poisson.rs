//! Static scalar/vector plans retain their original update, analog projection
//! and reset schedules. Only the stochastic observation scope is shared
//! with v5; no dynamic-event conversion or backend fallback is used.
use super::*;

pub(super) fn programs(plan: &Plan, layer: usize) -> impl Iterator<Item=&equation::Program> {
    plan.equations.iter().map(move |p| &p[layer])
        .chain(plan.state_equations.iter().flat_map(move |p| &p[layer]))
        .chain(plan.state_resets.iter().flat_map(move |p| &p[layer]))
}

pub(super) fn used(plan: &Plan) -> bool {
    plan.dynamic.is_none() && (0..plan.sizes.len()-1).any(|l|
        programs(plan,l).flatten().any(|n| matches!(n, equation::Node::Poisson { .. })))
}

pub(super) fn boundary_possible(plan: &Plan) -> bool {
    used(plan) && (0..plan.sizes.len()-1).any(|l| programs(plan,l)
        .any(|p| p.iter().any(|n| matches!(*n, equation::Node::Poisson { rate, .. }
            if poisson_ir::rate_differentiable(p, rate)))))
}

pub(super) fn validate(plan: &Plan) -> Result<()> {
    if !used(plan) { return Ok(()); }
    let counts = plan.noise_streams.as_ref().ok_or("static Poisson requires declared noise streams")?;
    let nodes = (0..plan.sizes.len()-1).try_fold(0usize, |total, l|
        programs(plan,l).try_fold(total, |n, p| n.checked_add(p.len())))
        .ok_or("static Poisson metadata overflow")?;
    ensure(nodes.checked_mul(1024).is_some_and(|n| n <= plan.max_tape_bytes),
        "static Poisson metadata budget exceeded")?;
    for (layer, &count) in counts.iter().enumerate() {
        let programs = programs(plan,layer).cloned().collect::<Vec<_>>();
        let a = dynamic::Action { noise_streams: count, ..Default::default() };
        poisson_ir::validate(&programs, &a, plan)?;
    }
    Ok(())
}

pub(super) fn action(plan: &Plan, layer: usize, neuron: usize) -> dynamic::Action {
    dynamic::Action { noise_domain: layer as u64, noise_entity: neuron as u64,
        noise_streams: plan.noise_streams.as_ref().map_or(0, |r| r[layer]),
        parameter_index: neuron, ..Default::default() }
}

pub(super) fn samples(plan: &Plan, uniform: u16, sequence: u64, batch: usize,
    layer: usize, neuron: usize, tick: u64, enabled: bool) -> [f64; 96] {
    let mut values = [0.; 96];
    values[..16].copy_from_slice(&plan.noise_at(uniform, sequence, batch, layer, neuron, tick));
    if enabled {
        poisson_ir::encode_keys(&mut values, plan, &action(plan, layer, neuron), sequence, batch, tick);
    }
    values
}

pub(super) fn reconcile<T>(result: Result<T>, mpi: Option<&mpi::Context>, enabled: bool) -> Result<T> {
    if enabled {
        if let Some(m) = mpi {
            let mut failed = [f64::from(result.is_err())]; m.sum(&mut failed)?;
            if failed[0] != 0. {
                return Err(result.err().unwrap_or_else(|| PeerFailure.into()));
            }
        }
    }
    result
}

pub(super) fn boundary<'a>(programs: impl IntoIterator<Item=&'a equation::Program>,
    plan: &Plan, weights: &[Vec<f64>], values: &[f64], cells: &[usize], detached: &[bool],
    action: &dynamic::Action, batch: usize, tick: u64, timestamp: f64, noise: &[f64],
    owns: bool, enabled: bool, mpi: Option<&mpi::Context>, sample_loss: f64, total_batch: usize,
    mut replay: impl FnMut(poisson_cache::Identity) -> Result<f64>) -> Result<[f64; 16]> {
    let mut coefficients = [f64::NAN; 16];
    if !enabled { return Ok(coefficients); }
    let query = if owns {
        equation::zero_poisson_sites(programs, values, weights, action.parameter_index,
            timestamp, noise, &[], &[], &plan.masks, cells, detached)
    } else { Ok((0, 0)) };
    let (zero, sites) = reconcile(query, mpi, true)?;
    let mut masks = [zero as f64, sites as f64];
    if let Some(m) = mpi { m.sum(&mut masks)?; }
    let (zero, sites) = (masks[0] as u16, masks[1] as u16);
    for stream in 0..16 {
        if zero & (1 << stream) == 0 { continue; }
        coefficients[stream] = if sites & (1 << stream) == 0 { 0. } else {
            replay(poisson_cache::identity(action, stream, batch, tick))? / total_batch as f64 - sample_loss
        };
    }
    Ok(coefficients)
}

pub(super) fn memory_bytes(plan: &Plan, batch: usize, time: usize,
    state: Option<&poisson_cache::Checkpoint>) -> Option<usize> {
    if !used(plan) { return Some(0); }
    let mut sites = 0usize;
    for l in 0..plan.sizes.len()-1 {
        let mut streams = 0u16;
        for n in programs(plan,l).flatten() {
            if let equation::Node::Poisson { stream, .. } = n { streams |= 1 << stream; }
        }
        sites = sites.checked_add(plan.sizes[l + 1].checked_mul(streams.count_ones() as usize)?)?;
    }
    state.map_or(0, |s| s.entries.len()).checked_add(batch.checked_mul(time)?.checked_mul(sites)?)?
        .checked_mul(768)?.checked_add(plan.state_width().checked_mul(32)?)?.checked_add(4096)
}
