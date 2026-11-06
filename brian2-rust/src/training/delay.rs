//! Brian run-boundary delay changes: old arrivals survive, new emissions use
//! the new delays. Rebuild the bounded action graph without moving model cells.
use super::*;
use dynamic::{Action, Trigger};
use equation::Node;
use std::collections::{BTreeSet, HashMap, VecDeque};

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Edge {
    pub event: usize,
    pub source: Trigger,
    pub states: Vec<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub delay_state: Option<usize>,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Route {
    pub edge: usize,
    pub event: usize,
    pub states: Vec<usize>,
    pub selection: usize,
    pub zero_gate: Option<usize>,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Pending {
    pub edge: usize,
    pub event: usize,
    pub states: Vec<usize>,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Path {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub clock: Option<usize>,
    pub name: String,
    pub start: usize,
    pub end: usize,
    pub pending: Vec<Pending>,
    pub edges: Vec<Edge>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub routes: Option<Vec<Route>>,
    #[serde(default, skip_serializing_if = "is_false")]
    pub shared_delay: bool,
    #[serde(default, skip_serializing_if = "is_false")]
    pub captured_shared_delay: bool,
}
impl Path {
    fn dt(&self, plan: &Plan, spec: &dynamic::Spec) -> Result<f64> {
        match self.clock {
            None => Ok(plan.clock.as_ref().unwrap().dt),
            Some(k) => spec.clocks.as_ref().and_then(|c| c.dts.get(k)).copied()
                .filter(|_| k>0).ok_or_else(|| "invalid delay pathway clock".into()),
        }
    }
}
fn is_false(value: &bool) -> bool {
    !value
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Layout {
    pub paths: Vec<Path>,
    pub free_cells: Vec<usize>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Update {
    pub pathways: std::collections::BTreeMap<String, Vec<f64>>,
}

fn plain(spec: &dynamic::Spec, event: usize, _history: &[usize]) -> Action {
    let mut a = spec.actions[event].clone();
    if a.trigger.as_ref().is_some_and(|g| g.state) {
        a.reads.pop();
    }
    a.trigger = None;
    a
}
fn advance(
    states: &[usize],
    owner: usize,
    mask: Option<[usize; 2]>,
    source: Option<&Trigger>,
    selection: Option<usize>,
    clock: Option<usize>,
) -> Vec<(Action, Vec<equation::Program>)> {
    let mut actions = Vec::new();
    if states.is_empty() {
        return actions;
    }
    for start in (0..states.len() - 1).step_by(63) {
        let reads = states[start..(start + 64).min(states.len())].to_vec();
        let programs = (1..reads.len())
            .map(|index| vec![Node::State { index }])
            .collect();
        actions.push((
            Action {
                clock,
                owner,
                mask,
                writes: reads[..reads.len() - 1].to_vec(),
                reads,
                ..Default::default()
            },
            programs,
        ));
    }
    let last = *states.last().unwrap();
    let action = Action {
        clock,
        owner,
        mask,
        reads: vec![last],
        writes: vec![last],
        ..Default::default()
    };
    actions.push((action.clone(), vec![vec![Node::Constant { value: 0.0 }]]));
    if let Some(trigger) = source {
        let mut record = action;
        let value = if let Some(index) = selection {
            record.reads.push(index);
            Node::State { index: 1 }
        } else {
            Node::Constant { value: 1.0 }
        };
        if trigger.state {record.reads.push(trigger.index);}
        actions.push((
            Action {
                trigger: Some(trigger.clone()),
                ..record
            },
            vec![vec![value]],
        ));
    }
    actions
}
fn gate(mut a: Action, states: &[usize], source: Option<&Trigger>) -> Action {
    a.trigger = if states.is_empty() {
        if let Some(g)=source {if g.state {a.reads.push(g.index);}}
        source.cloned()
    } else {
        a.reads.push(states[0]);
        Some(Trigger {
            external: false,
            state: true,
            index: states[0],
        })
    };
    a
}
mod layout;
mod rebuild;

pub(super) struct Prepared {
    pub plan: Plan,
    pub live: Vec<Vec<f64>>,
    pub mapping: Vec<Option<usize>>,
    pub input_width: usize,
    pub bytes: usize,
}

fn quantize(value: f64, dt: f64) -> Result<usize> {
    let ticks = value / dt + 0.5;
    ensure(
        value.is_finite() && value >= 0.0 && ticks.is_finite() && ticks <= 1_000_000.0,
        "delay must be finite, nonnegative and within history budget",
    )?;
    Ok(ticks.floor() as usize)
}

pub(super) fn prepare(plan: &Plan, live: &[Vec<f64>]) -> Result<Option<Prepared>> {
    rebuild::prepare(plan, live)
}

pub(super) fn execute(
    plan: Plan,
    state: State,
    live: Vec<Vec<f64>>,
    update: Update,
    tick: u64,
    sequence: u64,
) -> Result<Output> {
    rebuild::execute(plan, state, live, update, tick, sequence)
}
