//! Reconfigure arrival generations and per-batch emission routes atomically.
use super::*;

struct Emission<'a> {
    edge: usize,
    event: usize,
    states: &'a Vec<usize>,
    selection: Option<usize>,
}
fn emissions(path: &Path) -> Vec<Emission<'_>> {
    if let Some(routes) = &path.routes {
        routes
            .iter()
            .map(|r| Emission {
                edge: r.edge,
                event: r.event,
                states: &r.states,
                selection: Some(r.selection),
            })
            .collect()
    } else {
        let mut entries: Vec<_> = path
            .edges
            .iter()
            .enumerate()
            .map(|(edge, e)| Emission {
                edge,
                event: e.event,
                states: &e.states,
                selection: None,
            })
            .collect();
        entries.sort_by_key(|e| e.event);
        entries
    }
}
struct Rebuild<'a> {
    old: &'a Path,
    pending: Vec<(usize, usize, Vec<usize>)>,
    choices: Vec<Vec<usize>>,
    initial_choices: Vec<usize>,
    variants: Vec<(usize, usize)>,
    changed: bool,
    routed: bool,
}
fn choice(row: &[usize], batch: usize) -> usize {
    row[if row.len() == 1 { 0 } else { batch }]
}

pub(super) fn prepare(plan: &Plan, live: &[Vec<f64>]) -> Result<Option<Prepared>> {
    if !plan
        .dynamic
        .as_ref()
        .and_then(|s| s.delay_layout.as_ref())
        .is_some_and(Layout::runtime)
    {
        return Ok(None);
    }
    rebuild(plan, live, None)
}

pub(super) fn execute(
    plan: Plan,
    state: State,
    live: Vec<Vec<f64>>,
    update: Update,
    tick: u64,
    sequence: u64,
) -> Result<Output> {
    validate_state(&plan, &state)?;
    validate_live(&plan, &live)?;
    ensure(
        tick <= (1u64 << 53) && plan.time_at(tick).is_finite(),
        "invalid delay update clock",
    )?;
    let prepared = rebuild(&plan, &live, Some(&update))?.unwrap();
    let mut plan = prepared.plan;
    let spec = plan.dynamic.take().unwrap();
    let membrane = prepared
        .live
        .iter()
        .map(|r| spec.voltage.iter().map(|&k| r[k]).collect())
        .collect();
    Ok(Output { poisson_state: None, event_visits: None,
        clock_state: None,
        updated_dynamic: Some(spec),
        schema: "b2-lif-training-result-v1",
        state,
        final_state: Some(prepared.live),
        final_membrane: membrane,
        final_tick: Some(tick),
        noise_sequence: plan.noise_streams.as_ref().map(|_| sequence),
        backend: "cpu",
        numeric_profile: "native-delay-boundary-migration-f64",
        gpu_dispatches: 0,
        loss: 0.0,
        gradients: vec![],
        initial_gradients: vec![],
        initial_state_gradients: None,
        spikes: vec![],
        logits: vec![],
        tape_bytes: prepared.bytes,
        gradient_scope: "delay-boundary-no-gradient",
    })
}

fn rebuild(plan: &Plan, live: &[Vec<f64>], update: Option<&Update>) -> Result<Option<Prepared>> {
    let old = plan
        .dynamic
        .as_ref()
        .ok_or("delay update requires a dynamic plan")?;
    let layout = old
        .delay_layout
        .as_ref()
        .ok_or("delay update requires a verified pathway layout")?;
    if let Some(update) = update {
        ensure(
            !update.pathways.is_empty()
                && update
                    .pathways
                    .keys()
                    .all(|n| layout.paths.iter().any(|p| &p.name == n)),
            "unknown or empty delay update",
        )?;
    }
    let old_bytes = old
        .memory_bytes(live.len(), 0, plan.mpi_ranks.unwrap_or(1))
        .ok_or("delay migration memory overflow")?;
    ensure(
        old_bytes <= plan.max_tape_bytes,
        "delay migration exceeds memory budget",
    )?;
    let mut rebuilt = Vec::new();
    let mut new_cells = 0usize;
    let mut old_cells: BTreeSet<_> = layout.free_cells.iter().copied().collect();
    let mut new_actions = old.actions.len();
    let mut route_count = 0usize;
    let mut physical_updates = Vec::new();
    let mut any_changed = update.is_some();
    for path in &layout.paths {
        let dt=path.dt(plan,old)?;
        let values = update.and_then(|u| u.pathways.get(&path.name));
        if let Some(values) = values {
            ensure(
                values.len() == 1 || values.len() == path.edges.len(),
                "delay update shape mismatch",
            )?;
            for &x in values {
                quantize(x, dt)?;
            }
            ensure(
                !path.shared_delay || values.windows(2).all(|w| w[0] == w[1]),
                "shared pathway delay requires one scalar value",
            )?;
        }
        let routed = path.edges.iter().any(|e| e.delay_state.is_some());
        let mut choices = Vec::with_capacity(path.edges.len());
        let mut initial_choices = Vec::with_capacity(path.edges.len());
        let mut variants = Vec::new();
        for (edge, e) in path.edges.iter().enumerate() {
            let (row, initial) = if let Some(values) = values {
                let value = values[if values.len() == 1 { 0 } else { edge }];
                let ticks = quantize(value, dt)?;
                if let Some(k) = e.delay_state {
                    physical_updates.push((k, value));
                }
                (vec![ticks], ticks)
            } else if let Some(k) = e.delay_state {
                (
                    live.iter()
                        .map(|r| quantize(r[k], dt))
                        .collect::<Result<Vec<_>>>()?,
                    quantize(old.initial[k], dt)?,
                )
            } else {
                (vec![e.states.len()], e.states.len())
            };
            let unique: BTreeSet<_> = row
                .iter()
                .copied()
                .chain(std::iter::once(initial))
                .collect();
            route_count = route_count
                .checked_add(unique.len())
                .ok_or("delay route overflow")?;
            ensure(
                route_count <= 1_000_000
                    && route_count
                        .checked_mul(256)
                        .and_then(|n| n.checked_add(old_bytes))
                        .is_some_and(|n| n <= plan.max_tape_bytes),
                "delay route budget exceeded",
            )?;
            variants.extend(unique.into_iter().map(|d| (edge, d)));
            choices.push(row);
            initial_choices.push(initial);
        }
        variants.sort_by_key(|&(e, d)| (std::cmp::Reverse(d), path.edges[e].source.index, e));
        let previous = emissions(path);
        let changed = if routed {
            path.routes.is_none()
                || previous.len() != variants.len()
                || previous.iter().zip(&variants).any(|(p, &(e, d))| {
                    p.edge != e
                        || p.states.len() != d
                        || p.selection.is_none_or(|k| {
                            old.initial[k] != f64::from(initial_choices[e] == d)
                                || live
                                    .iter()
                                    .enumerate()
                                    .any(|(b, r)| r[k] != f64::from(choice(&choices[e], b) == d))
                        })
                })
        } else {
            path.edges
                .iter()
                .enumerate()
                .any(|(e, p)| p.states.len() != choices[e][0])
        };
        any_changed |= changed;
        let mut pending = Vec::new();
        for (edge, event, states) in path.pending.iter().map(|p| (p.edge, p.event, &p.states)).chain(
            previous
                .iter()
                .filter(|_| changed)
                .map(|p| (p.edge, p.event, p.states)),
        ) {
            if let Some(last) = states
                .iter()
                .rposition(|&k| old.initial[k] != 0.0 || live.iter().any(|r| r[k] != 0.0))
            {
                pending.push((edge, event, states[..=last].to_vec()));
            }
        }
        for p in &path.pending {
            old_cells.extend(&p.states);
        }
        for p in previous {
            old_cells.extend(p.states);
            if let Some(k) = p.selection {
                old_cells.insert(k);
            }
        }
        if let Some(routes) = &path.routes {
            for r in routes {
                if let Some(k) = r.zero_gate {
                    old_cells.insert(k);
                }
            }
        }
        let history = pending.iter().map(|(_, _, s)| s.len()).sum::<usize>()
            + variants.iter().map(|&(_, d)| d).sum::<usize>();
        let zero = if routed {
            variants.iter().filter(|&&(_, d)| d == 0).count()
        } else {
            0
        };
        let extra = history
            .checked_add(if routed { variants.len() + zero } else { 0 })
            .ok_or("delay state overflow")?;
        new_cells = new_cells
            .checked_add(extra)
            .ok_or("delay state size overflow")?;
        new_actions = new_actions
            .checked_sub(path.end - path.start)
            .and_then(|n| n.checked_add(variants.len() + pending.len() + history * 2 + zero * 2))
            .ok_or("delay action overflow")?;
        rebuilt.push(Rebuild {
            old: path,
            pending,
            choices,
            initial_choices,
            variants,
            changed,
            routed,
        });
    }
    if !any_changed {
        return Ok(None);
    }
    let width = old
        .initial
        .len()
        .checked_add(new_cells.saturating_sub(old_cells.len()))
        .ok_or("delay state overflow")?;
    ensure(
        width <= 1_000_000 && new_actions <= 1_000_000,
        "delay layout exceeds state/action limit",
    )?;
    let indirect_bytes = old.actions.iter().filter_map(|a| a.indirect.as_ref()).map(|a| a.memory_bytes()).max().unwrap_or(0);
    let action_bytes = 1024usize.checked_add(indirect_bytes).ok_or("delay action metadata overflow")?;
    let bytes = width
        .checked_mul(live.len() + 2)
        .and_then(|n| n.checked_mul(64))
        .and_then(|n| n.checked_add(new_actions.checked_mul(action_bytes)?))
        .and_then(|n| n.checked_add(old_bytes.checked_mul(3)?))
        .and_then(|n| {
            plan.masks
                .iter()
                .try_fold(n, |n, r| n.checked_add(r.len().checked_mul(48)?))
        })
        .ok_or("delay migration memory overflow")?;
    ensure(
        bytes <= plan.max_tape_bytes,
        "delay migration exceeds memory budget",
    )?;
    let mut spec = old.clone();
    spec.actions = Vec::with_capacity(new_actions);
    let mut next = live.to_vec();
    for r in &mut next {
        r.resize(width, 0.0);
    }
    spec.initial.resize(width, 0.0);
    spec.initial_parameters.resize(width, None);
    spec.detached.resize(width, true);
    let mut mapping: Vec<_> = (0..old.initial.len())
        .map(Some)
        .chain(std::iter::repeat_n(None, width - old.initial.len()))
        .collect();
    spec.binary_states.retain(|k| !old_cells.contains(k));
    if let Some(m) = &mut spec.migration {
        m.cells.retain(|c| !old_cells.contains(&c.index));
    }
    for &k in &old_cells {
        spec.initial[k] = 0.0;
        spec.initial_parameters[k] = None;
        spec.detached[k] = true;
        mapping[k] = None;
        for row in &mut next {
            row[k] = 0.0;
        }
    }
    for (k, value) in physical_updates {
        spec.initial[k] = value;
        for row in &mut next {
            row[k] = value;
        }
    }
    let mut free: VecDeque<_> = old_cells
        .into_iter()
        .chain(old.initial.len()..width)
        .collect();
    let mut keys: HashMap<_, _> = spec
        .program_sets
        .iter()
        .enumerate()
        .map(|(i, p)| (serde_json::to_string(p).unwrap(), i))
        .collect();
    let mut new_layout = Layout {
        paths: Vec::new(),
        free_cells: Vec::new(),
    };
    let mut cursor = 0;
    for r in rebuilt {
        spec.actions
            .extend_from_slice(&old.actions[cursor..r.old.start]);
        cursor = r.old.end;
        let templates: Vec<_> = r
            .old
            .edges
            .iter()
            .map(|e| plain(old, e.event, &e.states))
            .collect();
        let previous: HashMap<_, _> = emissions(r.old)
            .into_iter()
            .map(|p| ((p.edge, p.states.len()), p.states))
            .collect();
        let mut path = Path {
            clock: r.old.clock,
            name: r.old.name.clone(),
            start: spec.actions.len(),
            end: 0,
            pending: Vec::new(),
            edges: Vec::new(),
            routes: None,
            shared_delay: r.old.shared_delay,
            captured_shared_delay: r.old.captured_shared_delay,
        };
        let mut allocated = Vec::new();
        for &(edge, d) in &r.variants {
            let prior = if r.changed {
                None
            } else {
                previous.get(&(edge, d)).copied()
            };
            let states = allocate(
                &mut spec,
                &mut next,
                &mut mapping,
                &mut free,
                d,
                prior,
                old,
                live,
                templates[edge].mask,
            );
            let selection = if r.routed {
                let k = free.pop_front().unwrap();
                spec.binary_states.push(k);
                spec.detached[k] = true;
                spec.initial[k] = f64::from(r.initial_choices[edge] == d);
                for (b, row) in next.iter_mut().enumerate() {
                    row[k] = f64::from(choice(&r.choices[edge], b) == d);
                }
                Some(k)
            } else {
                None
            };
            let zero_gate = if r.routed && d == 0 {
                let k = allocate(
                    &mut spec,
                    &mut next,
                    &mut mapping,
                    &mut free,
                    1,
                    None,
                    old,
                    live,
                    templates[edge].mask,
                )[0];
                for (a, p) in advance(
                    &[k],
                    templates[edge].owner,
                    templates[edge].mask,
                    Some(&r.old.edges[edge].source),
                    selection,
                    r.old.clock,
                ) {
                    append(&mut spec, &mut keys, a, p)?;
                }
                Some(k)
            } else {
                None
            };
            allocated.push((edge, states, selection, zero_gate));
        }
        let mut histories = Vec::new();
        for (edge, old_event, prior) in r.pending {
            let states = allocate(
                &mut spec,
                &mut next,
                &mut mapping,
                &mut free,
                prior.len(),
                Some(&prior),
                old,
                live,
                templates[edge].mask,
            );
            path.pending.push(Pending {
                edge,
                event: spec.actions.len(),
                states: states.clone(),
            });
            spec.actions
                .push(gate(plain(old, old_event, &prior), &states, None));
            histories.push((
                states,
                templates[edge].owner,
                templates[edge].mask,
                None,
                None,
            ));
        }
        let mut edges = vec![None; r.old.edges.len()];
        let mut routes = Vec::new();
        for (edge, states, selection, zero_gate) in allocated {
            let e = &r.old.edges[edge];
            let event = spec.actions.len();
            if edges[edge].is_none() {
                edges[edge] = Some(Edge {
                    event,
                    source: e.source.clone(),
                    states: states.clone(),
                    delay_state: e.delay_state,
                });
            }
            let trigger_states = if let Some(k) = zero_gate {
                vec![k]
            } else {
                states.clone()
            };
            let mut template = templates[edge].clone();
            if let Some(address) = &mut template.event_noise {
                address.delay = states.len() as u64;
                address.pending = None;
            }
            spec.actions.push(gate(
                template,
                &trigger_states,
                Some(&e.source),
            ));
            if let Some(k) = selection {
                routes.push(Route {
                    edge,
                    event,
                    states: states.clone(),
                    selection: k,
                    zero_gate,
                });
            }
            if !states.is_empty() {
                histories.push((
                    states,
                    templates[edge].owner,
                    templates[edge].mask,
                    Some(e.source.clone()),
                    selection,
                ));
            }
        }
        path.edges = edges.into_iter().map(Option::unwrap).collect();
        if r.routed {
            path.routes = Some(routes);
        }
        for (states, owner, mask, source, selection) in histories {
            for (a, p) in advance(&states, owner, mask, source.as_ref(), selection, path.clock) {
                append(&mut spec, &mut keys, a, p)?;
            }
        }
        path.end = spec.actions.len();
        new_layout.paths.push(path);
    }
    spec.actions.extend_from_slice(&old.actions[cursor..]);
    let mut unused: BTreeSet<_> = free.into_iter().collect();
    let mut last = width;
    while last > 0 && unused.remove(&(last - 1)) {
        last -= 1;
    }
    new_layout.free_cells = unused.into_iter().collect();
    spec.initial.truncate(last);
    spec.initial_parameters.truncate(last);
    spec.detached.truncate(last);
    mapping.truncate(last);
    for row in &mut next {
        row.truncate(last);
    }
    let used: BTreeSet<_> = spec.actions.iter().filter_map(|a| a.program_set).collect();
    let ids: HashMap<_, _> = used.iter().enumerate().map(|(i, &old)| (old, i)).collect();
    spec.program_sets = used.iter().map(|&i| spec.program_sets[i].clone()).collect();
    for a in &mut spec.actions {
        if let Some(i) = a.program_set {
            a.program_set = Some(ids[&i]);
        }
    }
    spec.binary_states.sort_unstable();
    if let Some(m) = &mut spec.migration {
        m.cells.sort_by_key(|c| c.index);
    }
    spec.delay_layout = Some(new_layout);
    let mut result = plan.clone();
    result.dynamic = Some(spec);
    result.validate()?;
    validate_live(&result, &next)?;
    Ok(Some(Prepared {
        plan: result,
        live: next,
        mapping,
        input_width: old.initial.len(),
        bytes,
    }))
}

fn append(
    spec: &mut dynamic::Spec,
    keys: &mut HashMap<String, usize>,
    mut action: Action,
    programs: Vec<equation::Program>,
) -> Result<()> {
    let key = serde_json::to_string(&programs)?;
    let id = if let Some(&id) = keys.get(&key) {
        id
    } else {
        let id = spec.program_sets.len();
        keys.insert(key, id);
        spec.program_sets.push(programs);
        id
    };
    action.program_set = Some(id);
    spec.actions.push(action);
    Ok(())
}

fn allocate(
    spec: &mut dynamic::Spec,
    next: &mut [Vec<f64>],
    mapping: &mut [Option<usize>],
    free: &mut VecDeque<usize>,
    len: usize,
    previous: Option<&Vec<usize>>,
    old: &dynamic::Spec,
    live: &[Vec<f64>],
    mask: Option<[usize; 2]>,
) -> Vec<usize> {
    let states: Vec<_> = (0..len).map(|_| free.pop_front().unwrap()).collect();
    for (j, &k) in states.iter().enumerate() {
        spec.detached[k] = false;
        spec.binary_states.push(k);
        if let Some(previous) = previous {
            spec.initial[k] = old.initial[previous[j]];
            mapping[k] = Some(previous[j]);
            for (row, before) in next.iter_mut().zip(live) {
                row[k] = before[previous[j]];
            }
        }
        if let (Some(m), Some(owner)) = (&mut spec.migration, mask) {
            m.cells.push(migration::Cell {
                index: k,
                owners: vec![owner],
                restart: migration::Restart::Queue,
            });
        }
    }
    states
}
