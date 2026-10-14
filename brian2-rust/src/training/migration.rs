//! Atomic structural-boundary state migration. No time step or RNG is consumed.
use super::*;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Restart {
    Initial,
    Queue,
    Time { clock: usize },
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Cell {
    pub index: usize,
    pub owners: Vec<[usize; 2]>,
    pub restart: Restart,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Layout {
    pub controlled_masks: Vec<[usize; 2]>,
    pub cells: Vec<Cell>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MaskUpdate {
    pub masks: Vec<Vec<f64>>,
    pub growth_weight: f64,
}
impl Layout {
    pub(super) fn memory_bytes(&self) -> usize {
        self.controlled_masks.len() * 32 + self.cells.iter().map(|c| 64 + c.owners.len() * 16).sum::<usize>()
    }
    pub(super) fn validate(&self, plan: &Plan, spec: &dynamic::Spec, neuron_width: usize) -> Result<()> {
        let actual: BTreeSet<_> = spec.actions.iter().filter_map(|a| a.mask).collect();
        let controlled: BTreeSet<_> = self.controlled_masks.iter().copied().collect();
        ensure(actual == controlled && controlled.len() == self.controlled_masks.len(),
            "migration controlled masks must match the action masks")?;
        ensure(self.cells.len() <= spec.initial.len(), "migration state count exceeds layout")?;
        let mut owner_count = 0usize;
        for cell in &self.cells {
            owner_count = owner_count.checked_add(cell.owners.len()).ok_or("migration owner budget overflow")?;
            ensure(owner_count <= 1_000_000, "migration owner budget exceeded")?;
        }
        ensure(self.memory_bytes() <= plan.max_tape_bytes, "migration layout exceeds memory budget")?;
        let binary: BTreeSet<_> = spec.binary_states.iter().copied().collect();
        let integers: BTreeSet<_> = spec.integer_states.iter().copied().collect();
        let protected: BTreeSet<_> = spec.actions.iter().filter(|a| a.threshold.is_some())
            .flat_map(|a| a.reads.iter().copied()).collect();
        let permanent: BTreeSet<_> = spec.actions.iter().filter(|a| a.mask.is_none())
            .flat_map(|a| a.possible_writes()).collect();
        let mut declared = BTreeMap::new();
        for cell in &self.cells {
            ensure(!permanent.contains(&cell.index), "migration-owned state has an unmasked writer")?;
            ensure(cell.index >= neuron_width && cell.index < spec.initial.len() && !protected.contains(&cell.index)
                && declared.insert(cell.index, cell.owners.iter().copied().collect::<BTreeSet<_>>()).is_none(),
                "migration cannot own neuron/activity state or duplicate cells")?;
            let owners = &declared[&cell.index];
            ensure(!owners.is_empty() && owners.len() == cell.owners.len() && owners.is_subset(&controlled),
                "invalid migration cell owners")?;
            let valid = match cell.restart {
                Restart::Initial => (!binary.contains(&cell.index) && !spec.detached[cell.index])
                    || integers.contains(&cell.index) || (binary.contains(&cell.index) && spec.detached[cell.index]),
                Restart::Queue => binary.contains(&cell.index) && spec.initial_parameters[cell.index].is_none(),
                Restart::Time { clock } => !integers.contains(&cell.index) && !binary.contains(&cell.index) && spec.detached[cell.index]
                    && spec.clocks.as_ref().is_some_and(|c| clock < c.dts.len()),
            };
            ensure(valid, "migration restart rule does not match the state kind")?;
        }
        let mut writers: BTreeMap<usize, BTreeSet<[usize; 2]>> = BTreeMap::new();
        for action in &spec.actions {
            for k in action.possible_writes() {
                if k < neuron_width || protected.contains(&k) || permanent.contains(&k) { continue; }
                if let Some(mask) = action.mask { writers.entry(k).or_default().insert(mask); }
                else { ensure(!declared.contains_key(&k), "migration-owned state has an unmasked writer")?; }
            }
        }
        ensure(writers == declared, "migration ownership must cover exactly all masked synaptic/history writers")?;
        Ok(())
    }
}

pub(super) fn execute(mut plan: Plan, mut state: State, mut live: Vec<Vec<f64>>, update: MaskUpdate,
    tick: u64, sequence: u64, boundary_times: Option<Vec<f64>>) -> Result<Output> {
    let spec = plan.dynamic.as_ref().ok_or("native mask migration requires a dynamic plan")?;
    let layout = spec.migration.as_ref().ok_or("dynamic mask migration requires an ownership layout")?;
    validate_state(&plan, &state)?;
    validate_live(&plan, &live)?;
    ensure(tick <= (1u64 << 53) && plan.time_at(tick).is_finite(), "invalid migration clock tick")?;
    ensure(update.growth_weight.is_finite() && update.masks.len() == plan.masks.len()
        && update.masks.iter().zip(&plan.masks).all(|(new, old)| new.len() == old.len()
            && new.iter().all(|&v| v == 0.0 || v == 1.0)), "invalid migration masks or growth weight")?;
    let controlled: BTreeSet<_> = layout.controlled_masks.iter().copied().collect();
    let mut changed = BTreeSet::new();
    for (b, (old, new)) in plan.masks.iter().zip(&update.masks).enumerate() {
        for (i, (&a, &c)) in old.iter().zip(new).enumerate() {
            if a != c {
                ensure(controlled.contains(&[b, i]), "dynamic migration may change declared edge masks only")?;
                changed.insert([b, i]);
            }
        }
    }
    let bytes = live.len().checked_mul(spec.initial.len()).and_then(|n| n.checked_mul(16))
        .and_then(|n| state.weights.iter().try_fold(n, |s, w| s.checked_add(w.len().checked_mul(24)?)))
        .and_then(|n| n.checked_add(layout.memory_bytes()))
        .and_then(|n| n.checked_add(spec.clocks.as_ref().map_or(0, |c| c.dts.len() * 48)))
        .ok_or("migration memory overflow")?;
    ensure(bytes <= plan.max_tape_bytes, "migration state exceeds memory budget")?;
    for &[b, i] in &changed {
        state.weights[b][i] = if update.masks[b][i] == 1.0 { update.growth_weight } else { 0.0 };
        state.first_moment[b][i] = 0.0; state.second_moment[b][i] = 0.0;
    }
    let mut clock_times = boundary_times;
    for cell in &layout.cells {
        if !cell.owners.iter().any(|owner| changed.contains(owner)) { continue; }
        // An owner present on both sides keeps shared state alive. Disjoint
        // old/new owner sets mean a new generation, even if neither set is empty.
        if cell.owners.iter().any(|&[b, i]| plan.masks[b][i] == 1.0 && update.masks[b][i] == 1.0) { continue; }
        let active = cell.owners.iter().any(|&[b, i]| update.masks[b][i] == 1.0);
        let value = if !active { 0.0 } else { match cell.restart {
            Restart::Initial => spec.initial_parameters[cell.index].map_or(spec.initial[cell.index], |[b, i]| state.weights[b][i]),
            Restart::Queue => 0.0,
            Restart::Time { clock } => {
                if clock_times.is_none() { clock_times = Some(spec.clocks.as_ref().unwrap().boundary(tick)?); }
                clock_times.as_ref().unwrap()[clock]
            }
        }};
        for row in &mut live { row[cell.index] = value; }
    }
    plan.masks = update.masks;
    validate_state(&plan, &state)?;
    validate_live(&plan, &live)?;
    let voltage = &plan.dynamic.as_ref().unwrap().voltage;
    let membrane = live.iter().map(|row| voltage.iter().map(|&k| row[k]).collect()).collect();
    Ok(Output { poisson_state: None, event_visits: None, clock_state: None, updated_dynamic: None,
        schema: "b2-lif-training-result-v1", state, final_state: Some(live), final_membrane: membrane,
        final_tick: Some(tick), noise_sequence: plan.noise_streams.as_ref().map(|_| sequence),
        backend: "cpu", numeric_profile: "native-structural-state-migration-f64", gpu_dispatches: 0,
        loss: 0.0, gradients: vec![], initial_gradients: vec![], initial_state_gradients: None,
        spikes: vec![], logits: vec![], tape_bytes: bytes, gradient_scope: "structural-boundary-no-gradient",
    })
}
