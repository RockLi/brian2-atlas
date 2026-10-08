//! Runtime state addressing. Integer choices are detached; float VJPs use the
//! addresses actually read and the final writer at each resolved destination.
use super::*;
use dynamic::{Action, Spec};
use std::collections::{BTreeMap, HashSet};

#[derive(Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Read {
    pub index: usize,
    pub tables: Vec<Vec<usize>>,
}
#[derive(Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Index {
    Read { slot: usize },
    Output { slot: usize },
}
#[derive(Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Write {
    pub index: Index,
    pub tables: Vec<Vec<usize>>,
}
#[derive(Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Access {
    #[serde(default)]
    pub reads: BTreeMap<usize, Read>,
    #[serde(default)]
    pub writes: BTreeMap<usize, Write>,
}
impl Access {
    pub(super) fn memory_bytes(&self) -> usize {
        128 + self
            .reads
            .values()
            .map(|r| 128 + table_bytes(&r.tables))
            .sum::<usize>()
            + self
                .writes
                .values()
                .map(|r| 128 + table_bytes(&r.tables))
                .sum::<usize>()
    }
    pub(super) fn validate(&self, action: &Action, spec: &Spec, plan: &Plan) -> Result<()> {
        ensure(
            action.threshold.is_none()
                && (!self.reads.is_empty() || !self.writes.is_empty())
                && self.reads.len() <= action.reads.len()
                && self.writes.len() <= action.writes.len()
                && self.memory_bytes() <= plan.max_tape_bytes,
            "invalid indirect action shape/budget",
        )?;
        let integers: HashSet<_> = spec.integer_states.iter().copied().collect();
        let binary: HashSet<_> = spec.binary_states.iter().copied().collect();
        let width = spec.initial.len();
        let tables = |tables: &[Vec<usize>], placeholder: usize| -> Result<()> {
            ensure(
                !tables.is_empty() && tables.len() <= 16,
                "invalid indirect index depth",
            )?;
            for (depth, row) in tables.iter().enumerate() {
                ensure(
                    !row.is_empty() && row.len() <= 1_000_000 && row.iter().all(|&k| k < width),
                    "invalid indirect address table",
                )?;
                if depth + 1 < tables.len() {
                    ensure(
                        row.iter().all(|k| integers.contains(k)),
                        "indirect intermediate table requires integer state",
                    )?;
                } else {
                    ensure(
                        row.iter().all(|k| {
                            integers.contains(k) == integers.contains(&placeholder)
                                && binary.contains(k) == binary.contains(&placeholder)
                        }),
                        "indirect target storage type mismatch",
                    )?;
                }
            }
            Ok(())
        };
        for (&slot, r) in &self.reads {
            ensure(
                slot < action.reads.len() && r.index < width && integers.contains(&r.index),
                "indirect read requires a valid integer index cell",
            )?;
            ensure(
                !action
                    .trigger
                    .as_ref()
                    .is_some_and(|g| g.state && action.reads[slot] == g.index),
                "indirect reads cannot replace an event gate",
            )?;
            tables(&r.tables, action.reads[slot])?;
        }
        for (&slot, r) in &self.writes {
            ensure(
                slot < action.writes.len(),
                "indirect write outside output domain",
            )?;
            match r.index {
                Index::Read { slot } => ensure(
                    slot < action.reads.len() && integers.contains(&action.reads[slot]),
                    "indirect write index requires an integer context slot",
                )?,
                Index::Output { slot } => {
                    let programs = &spec.program_sets[action
                        .program_set
                        .ok_or("indirect writes require programs")?];
                    ensure(
                        slot < programs.len()
                            && equation::integer_node(programs[slot].last().unwrap()),
                        "indirect write index requires an integer output",
                    )?;
                }
            }
            tables(&r.tables, action.writes[slot])?;
        }
        Ok(())
    }
    pub(super) fn referenced_cells(&self) -> impl Iterator<Item = usize> + '_ {
        self.reads
            .values()
            .flat_map(|r| std::iter::once(r.index).chain(r.tables.iter().flatten().copied()))
            .chain(
                self.writes
                    .values()
                    .flat_map(|r| r.tables.iter().flatten().copied()),
            )
    }
}
fn table_bytes(tables: &[Vec<usize>]) -> usize {
    tables.iter().map(|r| 24 + r.len() * 8).sum()
}
impl Action {
    pub(super) fn possible_writes(&self) -> impl Iterator<Item = usize> + '_ {
        self.writes.iter().enumerate().flat_map(|(slot, _)| {
            self.indirect
                .as_ref()
                .and_then(|r| r.writes.get(&slot))
                .map_or_else(
                    || std::slice::from_ref(&self.writes[slot]),
                    |r| r.tables.last().unwrap().as_slice(),
                )
                .iter()
                .copied()
        })
    }
}
fn address(mut index: f64, tables: &[Vec<usize>], live: &[f64]) -> Result<usize> {
    let mut target = 0;
    for (depth, row) in tables.iter().enumerate() {
        let k = equation::int32(index)?;
        ensure(
            k >= 0 && (k as usize) < row.len(),
            "runtime index outside address table",
        )?;
        target = row[k as usize];
        if depth + 1 < tables.len() {
            index = live[target];
        }
    }
    Ok(target)
}

pub(super) struct Resolved {
    reads: Vec<usize>,
    writes: Vec<usize>,
    old: Vec<f64>,
}
impl Resolved {
    pub(super) fn read_cells(&self) -> &[usize] { &self.reads }
}
pub(super) struct Evaluation<'a> {
    pub plan: &'a Plan,
    pub spec: &'a Spec,
    pub action: &'a Action,
    pub weights: &'a [Vec<f64>],
    pub timestamp: f64,
    pub noise: &'a [f64],
    pub clocks: &'a [f64],
}
impl Evaluation<'_> {
    fn forward(&self, output: usize, context: &[f64]) -> Result<f64> {
        equation::forward_clocked(
            &self.spec.program_sets[self.action.program_set.unwrap()][output],
            context,
            self.weights,
            self.action.parameter_index,
            self.timestamp,
            self.noise,
            self.clocks,
            &self.spec.parameter_maps,
        )
    }
    pub(super) fn run(
        &self,
        context: &mut [f64],
        live: &mut [f64],
        gate: f64,
        owns: bool,
        mpi: Option<&mpi::Context>,
    ) -> Result<Resolved> {
        let access = self.action.indirect.as_ref().unwrap();
        let mut record = Resolved {
            reads: self.action.reads.clone(),
            writes: self.action.writes.clone(),
            old: vec![0.; self.action.writes.len()],
        };
        for (&slot, r) in &access.reads {
            match address(live[r.index], &r.tables, live) {
                Ok(k) => {
                    record.reads[slot] = k;
                    context[slot] = live[k];
                }
                Err(error) => {
                    if gate != 0. {
                        return Err(error);
                    }
                    record.reads[slot] = usize::MAX;
                    context[slot] = 0.;
                }
            }
        }
        let mut values = [0.; 64];
        if gate != 0. {
            if owns {
                for (s, value) in values[..self.action.writes.len()].iter_mut().enumerate() {
                    *value = self.forward(s, context)?;
                }
            }
            if let Some(m) = mpi {
                m.sum(&mut values[..self.action.writes.len()])?;
            }
        }
        for (&slot, r) in &access.writes {
            let index = match r.index {
                Index::Read { slot } => Ok(context[slot]),
                Index::Output { slot } => {
                    if gate != 0. {
                        Ok(values[slot])
                    } else {
                        self.forward(slot, context)
                    }
                }
            };
            let target = index.and_then(|v| address(v, &r.tables, live));
            match target {
                Ok(k) => record.writes[slot] = k,
                Err(error) => {
                    if gate != 0. {
                        return Err(error);
                    }
                    record.writes[slot] = usize::MAX;
                }
            }
        }
        for (slot, &k) in record.writes.iter().enumerate() {
            if k != usize::MAX {
                record.old[slot] = live[k];
            }
        }
        if gate != 0. {
            for (slot, &k) in record.writes.iter().enumerate() {
                let value = values[slot];
                ensure(
                    !self.spec.binary_states.contains(&k) || value == 0. || value == 1.,
                    "indirect binary state update must be binary",
                )?;
                ensure(
                    !self.spec.integer_states.contains(&k) || equation::int32(value).is_ok(),
                    "indirect integer update is outside int32",
                )?;
                live[k] = value;
            }
        }
        Ok(record)
    }
    pub(super) fn backward(
        &self,
        record: &Resolved,
        context: &[f64],
        adjoints: &mut [f64],
        gradients: &mut [Vec<f64>],
        gate: f64,
        differentiate_gate: bool,
        sample_loss: f64,
        boundary: &[f64; 16],
        owns: bool,
        mpi: Option<&mpi::Context>,
    ) -> Result<f64> {
        if gate == 0. && !differentiate_gate {
            return Ok(0.);
        }
        // An inactive hard event need not have a valid counterfactual address.
        // Only surrogate differentiation asks for that counterfactual transform.
        if record.writes.contains(&usize::MAX) {
            ensure(
                !adjoints.iter().any(|&v| v != 0.),
                "invalid runtime index in surrogate counterfactual",
            )?;
            return Ok(0.);
        }
        let c = context.len();
        let w = record.writes.len();
        let mut delta = [0.; 129];
        let mut seen = HashSet::new();
        let winners: Vec<_> = record
            .writes
            .iter()
            .enumerate()
            .rev()
            .filter_map(|(s, &k)| seen.insert(k).then_some(s))
            .collect();
        if owns {
            if gate != 0. {
                equation::backward_poisson(&self.spec.program_sets[self.action.program_set.unwrap()], context, self.weights,
                    sample_loss, boundary, gradients, &self.plan.masks, &mut delta[..c], self.action.parameter_index,
                    self.timestamp, self.noise, self.clocks, &self.spec.parameter_maps, &record.reads, &self.spec.detached)?;
            }
            for &s in &winners {
                let target = record.writes[s];
                if self.spec.detached[target] {
                    continue;
                }
                let g = adjoints[target];
                if g == 0. {
                    continue;
                }
                ensure(
                    !record.reads.contains(&usize::MAX),
                    "invalid runtime index in surrogate counterfactual",
                )?;
                delta[c + s] = g * (1. - gate);
                if gate != 0. {
                    equation::backward_clocked_addresses(
                        &self.spec.program_sets[self.action.program_set.unwrap()][s],
                        context,
                        self.weights,
                        g * gate,
                        gradients,
                        &self.plan.masks,
                        &mut delta[..c],
                        self.action.parameter_index,
                        self.timestamp,
                        self.noise,
                        self.clocks,
                        &self.spec.parameter_maps,
                        &record.reads,
                        &self.spec.detached,
                    )?;
                }
                if differentiate_gate {
                    delta[c + w] += g * (self.forward(s, context)? - record.old[s]);
                }
            }
        }
        if let Some(m) = mpi {
            m.sum(&mut delta[..c + w + 1])?;
        }
        for &s in &winners {
            let k = record.writes[s];
            if !self.spec.detached[k] {
                adjoints[k] = delta[c + s];
            }
        }
        for (slot, &k) in record.reads.iter().enumerate() {
            if k != usize::MAX && !self.spec.detached[k] {
                adjoints[k] += delta[slot];
            }
        }
        Ok(delta[c + w])
    }
}
