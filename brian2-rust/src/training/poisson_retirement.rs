//! Explicit run-boundary retirement. Continuation certificates prevent a
//! discarded first-observation rate from silently being sampled again.
use super::*;
use poisson_cache::Site;
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::io::{self, Write};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Continuation {
    pub tick: u64,
    pub sha256: String,
}

struct HashWriter(Sha256);
impl Write for HashWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.0.update(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> { Ok(()) }
}

impl Continuation {
    pub(super) fn validate(&self) -> Result<()> {
        ensure(self.tick <= 1u64 << 53 && self.sha256.len() == 64
            && self.sha256.bytes().all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c)),
            "invalid Poisson retirement continuation certificate")
    }
    pub(super) fn new(plan: &Plan, live: &[Vec<f64>], tick: u64,
        clock: Option<&clock_state::Checkpoint>) -> Result<Self> {
        let mut writer = HashWriter(Sha256::new());
        writer.write_all(b"b2-poisson-continuation-v1")?;
        // Serialize directly into the digest; do not allocate another model or
        // physical-state copy. Optimizer settings, masks and partition count
        // cannot change RNG identities and are allowed to change at boundaries.
        serde_json::to_writer(&mut writer, &(&plan.schema, &plan.backend, &plan.sizes,
            &plan.projections, &plan.equations, &plan.state_equations, &plan.state_resets,
            &plan.refractory, &plan.clock, &plan.noise_streams, plan.seed, &plan.dynamic,
            &plan.threshold_parameters, &plan.threshold_per_neuron))?;
        serde_json::to_writer(&mut writer, &clock)?;
        writer.write_all(&tick.to_le_bytes())?;
        writer.write_all(&(live.len() as u64).to_le_bytes())?;
        for row in live {
            writer.write_all(&(row.len() as u64).to_le_bytes())?;
            for value in row { writer.write_all(&value.to_bits().to_le_bytes())?; }
        }
        Ok(Self { tick, sha256: format!("{:x}", writer.0.finalize()) })
    }
    pub(super) fn verify(&self, plan: &Plan, live: &[Vec<f64>], tick: u64,
        clock: Option<&clock_state::Checkpoint>) -> Result<()> {
        self.validate()?;
        ensure(*self == Self::new(plan, live, tick, clock)?,
            "retired Poisson history requires its exact continuation state/clock; restore an earlier checkpoint to rewind")
    }
}

/// Admit the concurrent boundary workspace before cloning the address plan or
/// authenticating GPU samples. The cache reserve also covers the bounded site
/// sets/maps; those maps never grow for sites absent from the incoming cache.
pub(super) fn admission(plan: &Plan, cache: &poisson_cache::Checkpoint, live: &[Vec<f64>]) -> Result<usize> {
    let metadata = if let Some(spec) = &plan.dynamic {
        spec.memory_bytes(0, 0, 0)
    } else {
        let nodes = (0..plan.sizes.len()-1).try_fold(0usize, |sum, layer|
            static_poisson::programs(plan, layer).try_fold(sum, |n, p| n.checked_add(p.len())));
        let topology = plan.projections.as_ref().map_or(Some(0), |ps| ps.iter().try_fold(0usize, |n, p|
            p.sources.len().checked_add(p.targets.len())?.checked_add(p.parameter_ids.len())?
                .checked_mul(8)?.checked_add(192)?.checked_add(n)));
        nodes.and_then(|n| n.checked_mul(128)).and_then(|n| n.checked_add(topology?))
    }.ok_or("Poisson retirement metadata overflow")?;
    let parameters = plan.masks.iter().try_fold(0usize, |n, row| n.checked_add(row.len()));
    let cells = live.iter().try_fold(0usize, |n, row| n.checked_add(row.len()));
    let bytes = cache.entries.len().checked_mul(768)
        .and_then(|n| n.checked_add(metadata.checked_mul(2)?))
        .and_then(|n| n.checked_add(parameters?.checked_mul(48)?))
        .and_then(|n| n.checked_add(cells?.checked_mul(16)?))
        .and_then(|n| n.checked_add(4096)).ok_or("Poisson retirement memory overflow")?;
    ensure(bytes <= plan.max_tape_bytes, "Poisson retirement workspace budget exceeded")?;
    Ok(bytes)
}

/// Keep every identity that any admitted action can still revisit. Unknown
/// imported sites remain untouched. Emission bounds are signed because the RNG
/// represents emissions before tick zero by wrapping u64 subtraction.
pub(super) fn retire(plan: &Plan, cache: &mut poisson_cache::Checkpoint,
    live: &[Vec<f64>], tick: u64, clock: Option<&clock_state::Checkpoint>) -> Result<()> {
    ensure(plan.clock.is_some(), "Poisson retirement requires a clock")?;
    let present: HashSet<_> = cache.entries.iter().map(|e| e.identity.site).collect();
    let mut bounds = HashMap::<Site, i128>::new();
    let mut pending = HashMap::<Site, bool>::new();
    if let Some(spec) = &plan.dynamic {
        let calls = if spec.clocks.is_some() {
            &clock.ok_or("Poisson retirement requires the committed dynamic clock state")?.calls[..]
        } else { &[] };
        let mut exhausted = HashMap::new();
        if let Some(layout) = &spec.delay_layout {
            for path in &layout.paths { for entry in &path.pending {
                exhausted.insert(entry.event, !entry.states.is_empty()
                    && live.iter().all(|row| entry.states.iter().all(|&k| row[k] == 0.)));
            } }
        }
        for (index, action) in spec.actions.iter().enumerate() {
            let count = if calls.is_empty() { tick } else { calls[action.clock.unwrap_or(0)] };
            let programs = action.program_set.map(|i| &spec.program_sets[i]);
            let mut streams = 0u16;
            for node in programs.into_iter().flatten().flatten() {
                if let equation::Node::Poisson { stream, .. } = node { streams |= 1 << stream; }
            }
            for stream in 0..16 { if streams & (1 << stream) != 0 {
                let site = Site::new(action, stream);
                if !present.contains(&site) { continue; }
                if site.kind == 2 {
                    let safe = exhausted.get(&index).copied().unwrap_or(false);
                    pending.entry(site).and_modify(|old| *old &= safe).or_insert(safe);
                } else {
                    let bound = count as i128 - action.event_noise.as_ref().map_or(0, |a| a.delay) as i128;
                    bounds.entry(site).and_modify(|old| *old = (*old).min(bound)).or_insert(bound);
                }
            } }
        }
    } else {
        for layer in 0..plan.sizes.len()-1 {
            let mut streams = 0u16;
            for node in static_poisson::programs(plan, layer).flatten() {
                if let equation::Node::Poisson { stream, .. } = node { streams |= 1 << stream; }
            }
            for neuron in 0..plan.sizes[layer+1] { for stream in 0..16 {
                if streams & (1 << stream) != 0 {
                    let site = Site::new(&static_poisson::action(plan, layer, neuron), stream);
                    if present.contains(&site) { bounds.insert(site, tick as i128); }
                }
            } }
        }
    }
    cache.entries.retain(|entry| {
        let id = entry.identity;
        if id.site.kind == 2 { return !pending.get(&id.site).copied().unwrap_or(false); }
        let instant = if id.site.kind == 1 {
            // Only the supported negative emission window is interpreted as
            // wrapped time. Unrelated future/orphan identities stay retained.
            if id.instant >= u64::MAX - 1_000_000 { (id.instant as i64) as i128 }
            else { id.instant as i128 }
        } else { id.instant as i128 };
        bounds.get(&id.site).is_none_or(|&bound| instant >= bound)
    });
    cache.continuation = Some(Continuation::new(plan, live, tick, clock)?);
    Ok(())
}
