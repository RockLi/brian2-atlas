//! Verify queue routing against the actual action graph, including batch selectors.
use super::*;

impl Layout {
    pub(in crate::training) fn memory_bytes(&self) -> usize {
        self.free_cells.len() * 16
            + self
                .paths
                .iter()
                .map(|p| {
                    256 + p.name.len()
                        + p.edges
                            .iter()
                            .map(|e| 112 + e.states.len() * 16)
                            .sum::<usize>()
                        + p.pending
                            .iter()
                            .map(|e| 80 + e.states.len() * 16)
                            .sum::<usize>()
                        + p.routes.as_ref().map_or(0, |rs| {
                            rs.iter().map(|r| 128 + r.states.len() * 16).sum::<usize>()
                        })
                })
                .sum::<usize>()
    }

    pub(in crate::training) fn runtime(&self) -> bool {
        self.paths
            .iter()
            .any(|p| p.edges.iter().any(|e| e.delay_state.is_some()))
    }

    pub(in crate::training) fn validate_live(&self, spec: &dynamic::Spec, rows: &[Vec<f64>]) -> Result<()> {
        for p in &self.paths {
            for pending in &p.pending {
                if spec.actions[pending.event].event_noise.as_ref().is_some_and(|a| a.pending.is_some()) {
                    ensure(rows.iter().all(|row| pending.states.iter().filter(|&&k| row[k] != 0.).count() <= 1),
                        "imported event noise identity cannot deliver multiple queued events")?;
                }
            }
            if let Some(routes) = &p.routes {
                for row in rows {
                    let mut chosen = vec![0usize; p.edges.len()];
                    for r in routes {
                        chosen[r.edge] += usize::from(row[r.selection] == 1.0);
                    }
                    ensure(
                        chosen.iter().all(|&n| n == 1),
                        "runtime delay route selection must be one-hot",
                    )?;
                }
            }
        }
        Ok(())
    }

    pub(in crate::training) fn validate(
        &self,
        plan: &Plan,
        spec: &dynamic::Spec,
        neuron_width: usize,
    ) -> Result<()> {
        ensure(
            self.memory_bytes() <= plan.max_tape_bytes,
            "delay layout exceeds budget",
        )?;
        let mut names = BTreeSet::new();
        let mut cells = BTreeSet::new();
        let mut physical = BTreeSet::new();
        let mut blocks = BTreeSet::new();
        let binary: BTreeSet<_> = spec.binary_states.iter().copied().collect();
        let integers: BTreeSet<_> = spec.integer_states.iter().copied().collect();
        let source_clocks:HashMap<_,_>=spec.actions.iter().filter_map(|a|a.threshold
            .and_then(|n|spec.spike_buffers.get(n).map(|&k|(k,a.clock.unwrap_or(0))))).collect();
        let mut end = 0;
        for (path_index, path) in self.paths.iter().enumerate() {
            // NumPy scatter/re-read stages execute the same arrival batch.
            // Their routes are latched together at the run boundary, so only
            // verified stages may share the original physical delay cells.
            let staged_origin = if let Some((origin, suffix)) = path.name.rsplit_once("::numpy-stage:") {
                let stage = suffix.parse::<usize>().ok().filter(|&n| n > 0)
                    .ok_or("invalid NumPy delay stage name")?;
                let original = self.paths[..path_index].iter().find(|p| p.name == origin)
                    .ok_or("missing NumPy delay stage origin")?;
                let preceding = if stage == 1 { origin.to_string() }
                    else { format!("{origin}::numpy-stage:{}", stage - 1) };
                ensure(path_index > 0 && self.paths[path_index - 1].name == preceding
                    && path.clock == original.clock && path.shared_delay == original.shared_delay
                    && path.captured_shared_delay == original.captured_shared_delay
                    && path.edges.len() == original.edges.len()
                    && path.edges.iter().zip(&original.edges).all(|(a, b)|
                        a.source == b.source && a.delay_state == b.delay_state),
                    "inconsistent NumPy delay stages")?;
                Some(original)
            } else { None };
            ensure(
                !path.name.is_empty()
                    && names.insert(&path.name)
                    && path.start >= end
                    && path.start <= path.end
                    && path.end <= spec.actions.len()
                    && path.edges.len() <= 1_000_000,
                "invalid delay pathway range/name",
            )?;
            let dt=path.dt(plan,spec)?;
            ensure(spec.actions[path.start..path.end].iter().all(|a|a.clock==path.clock),
                "delay pathway actions use different clocks")?;
            end = path.end;
            blocks.extend(path.start..path.end);
            let mut local_physical = BTreeSet::new();
            for e in &path.edges {
                ensure(
                    e.event >= path.start
                        && e.event < path.end
                        && if e.source.state {
                            !e.source.external && source_clocks.get(&e.source.index)==Some(&path.clock.unwrap_or(0))
                        } else {
                            path.clock.is_none() && e.source.index < if e.source.external {plan.sizes[0]} else {spec.voltage.len()}
                        },
                    "invalid delay source",
                )?;
                if let Some(k) = e.delay_state {
                    ensure(
                        k >= neuron_width
                            && k < spec.initial.len()
                            && if local_physical.insert(k) {
                                physical.insert(k) || staged_origin.is_some()
                            } else {
                                path.shared_delay
                            }
                            && !spec.detached[k]
                            && !binary.contains(&k)
                            && !integers.contains(&k)
                            && spec.initial_parameters[k].is_none(),
                        "invalid pathway delay storage",
                    )?;
                    quantize(spec.initial[k], dt)?;
                }
            }
            ensure(!path.captured_shared_delay || path.shared_delay,
                "captured shared delay requires scalar pathway storage")?;
            if path.shared_delay && !path.edges.is_empty() {
                ensure(
                    local_physical.len() == 1 && path.edges.iter().all(|e| e.delay_state.is_some()),
                    "shared pathway delay requires one physical cell",
                )?;
                let k = *local_physical.first().unwrap();
                ensure(
                    path.captured_shared_delay || spec.actions.iter().all(|a| !a.possible_writes().any(|v| v == k)),
                    "scalar pathway delay is read-only in event code",
                )?;
            }
            ensure(
                path.edges.iter().all(|e| e.delay_state.is_some())
                    || path.edges.iter().all(|e| e.delay_state.is_none()),
                "incomplete pathway delay storage",
            )?;
            let mut cell = |k: usize, detached: bool| -> Result<()> {
                ensure(
                    k >= neuron_width
                        && k < spec.initial.len()
                        && cells.insert(k)
                        && binary.contains(&k)
                        && spec.detached[k] == detached
                        && spec.initial_parameters[k].is_none(),
                    "invalid or shared delay history/selector cell",
                )
            };
            let mut cursor = path.start;
            let mut events = Vec::new();
            if let Some(routes) = &path.routes {
                ensure(
                    !path.edges.is_empty()
                        && path.edges.iter().all(|e| e.delay_state.is_some())
                        && routes.len() >= path.edges.len()
                        && routes.len() <= 1_000_000,
                    "invalid runtime delay routes",
                )?;
                let mut first = vec![None; path.edges.len()];
                let mut pairs = BTreeSet::new();
                let mut previous = None;
                for r in routes {
                    ensure(
                        r.edge < path.edges.len()
                            && r.states.len() <= 1_000_000
                            && pairs.insert((r.edge, r.states.len())),
                        "invalid or duplicate delay route",
                    )?;
                    let edge = &path.edges[r.edge];
                    let key = (std::cmp::Reverse(r.states.len()), edge.source.index, r.edge);
                    ensure(
                        previous.is_none_or(|p| p < key),
                        "invalid delay route order",
                    )?;
                    previous = Some(key);
                    if first[r.edge].is_none() {
                        first[r.edge] = Some(r);
                    }
                    cell(r.selection, true)?;
                    let chosen = usize::from(
                        quantize(
                            spec.initial[edge.delay_state.unwrap()],
                            dt,
                        )? == r.states.len(),
                    );
                    ensure(
                        spec.initial[r.selection] == chosen as f64,
                        "initial delay selector mismatch",
                    )?;
                    if let Some(gate) = r.zero_gate {
                        ensure(r.states.is_empty(), "zero delay gate has a history")?;
                        cell(gate, false)?;
                        ensure(
                            spec.initial[gate] == 0.0,
                            "zero delay gate must start clear",
                        )?;
                        let template = &spec.actions[edge.event];
                        for (action, programs) in advance(
                            &[gate],
                            template.owner,
                            template.mask,
                            Some(&edge.source),
                            Some(r.selection),
                            path.clock,
                        ) {
                            check_action(spec, path.end, &mut cursor, action, programs)?;
                        }
                    } else {
                        ensure(!r.states.is_empty(), "zero delay route requires a gate")?;
                    }
                }
                for (e, r) in path.edges.iter().zip(first) {
                    let r = r.ok_or("missing delay route")?;
                    ensure(
                        e.event == r.event && e.states == r.states,
                        "canonical delay route mismatch",
                    )?;
                }
                for r in routes {
                    events.push((r.event, &r.states, r.edge, Some(r.selection), r.zero_gate));
                }
            } else {
                let mut order: Vec<_> = (0..path.edges.len()).collect();
                order.sort_by_key(|&e| {
                    (
                        std::cmp::Reverse(path.edges[e].states.len()),
                        path.edges[e].source.index,
                        e,
                    )
                });
                for e in order {
                    events.push((path.edges[e].event, &path.edges[e].states, e, None, None));
                }
            }
            let mut histories = Vec::new();
            let mut pending_noise_ids = BTreeSet::new();
            for (event, states, edge, pending, selection, zero_gate) in path
                .pending
                .iter()
                .map(|p| (p.event, &p.states, p.edge, true, None, None))
                .chain(events.iter().map(|&(a, b, c, d, e)| (a, b, c, false, d, e)))
            {
                ensure(
                    edge < path.edges.len() && event == cursor && cursor < path.end,
                    "invalid delay event order",
                )?;
                cursor += 1;
                let e = &path.edges[edge];
                let a = &spec.actions[event];
                ensure(
                    a.threshold.is_none()
                        && (!pending || !states.is_empty())
                        && states.len() <= 1_000_000,
                    "invalid delay event/history",
                )?;
                let trigger_cell = states.first().copied().or(zero_gate).or_else(|| e.source.state.then_some(e.source.index));
                if let Some(k) = trigger_cell {
                    ensure(
                        a.reads.last() == Some(&k)
                            && a.trigger.as_ref()
                                == Some(&Trigger {
                                    external: false,
                                    state: true,
                                    index: k,
                                }),
                        "delay gate mismatch",
                    )?;
                    let programs = &spec.program_sets[a.program_set.unwrap()];
                    ensure(!programs.iter().flatten().any(|node| matches!(node,
                        Node::State{index}|Node::IntegerState{index}|Node::RefractoryActive{index} if *index==a.reads.len()-1)),
                        "delay gate cannot be read as model state")?;
                } else {
                    ensure(
                        a.trigger.as_ref() == Some(&e.source),
                        "delay event source mismatch",
                    )?;
                }
                for &k in states {
                    cell(k, false)?;
                }
                if !states.is_empty() {
                    histories.push((
                        states,
                        a.owner,
                        a.mask,
                        if pending { None } else { Some(&e.source) },
                        selection,
                    ));
                }
                let mut actual = plain(spec, event, states);
                let mut template = plain(spec, e.event, &e.states);
                ensure(actual.event_noise.is_some() == template.event_noise.is_some(),
                    "event noise declaration differs from its pathway")?;
                if let Some(address) = &actual.event_noise {
                    if !pending {
                        ensure(address.pending.is_none() && address.delay == states.len() as u64,
                            "emission noise delay differs from its route")?;
                    } else if let Some(id) = address.pending {
                        ensure(pending_noise_ids.insert((edge, id))
                            && states.iter().filter(|&&k| spec.initial[k] != 0.).count() <= 1,
                            "imported event noise identity must be unique")?;
                    }
                }
                actual.event_noise = None;
                template.event_noise = None;
                ensure(
                    actual == template,
                    "event differs from its pathway edge",
                )?;
            }
            for (states, owner, mask, source, selection) in histories {
                for (action, programs) in advance(states, owner, mask, source, selection, path.clock) {
                    check_action(spec, path.end, &mut cursor, action, programs)?;
                }
            }
            ensure(cursor == path.end, "extra action in delay pathway")?;
        }
        for action in &spec.actions {
            if let Some(access) = &action.indirect {
                ensure(!access.referenced_cells().any(|k| cells.contains(&k) || self.free_cells.contains(&k)),
                    "runtime index mappings cannot access delay queue storage")?;
            }
        }
        let used = cells.clone();
        for &k in &self.free_cells {
            ensure(
                k >= neuron_width
                    && k < spec.initial.len()
                    && cells.insert(k)
                    && !binary.contains(&k)
                    && !integers.contains(&k)
                    && spec.initial[k] == 0.0
                    && spec.detached[k]
                    && spec.initial_parameters[k].is_none(),
                "invalid free delay cell",
            )?;
        }
        ensure(
            cells.is_disjoint(&physical),
            "pathway delay storage aliases queue cells",
        )?;
        for (i, a) in spec.actions.iter().enumerate() {
            if !blocks.contains(&i) {
                ensure(
                    !a.reads.iter().chain(&a.writes).any(|k| cells.contains(k)),
                    "binary delay cell used outside its pathway",
                )?;
            } else {
                ensure(
                    !a.reads
                        .iter()
                        .chain(&a.writes)
                        .any(|k| cells.contains(k) && !used.contains(k)),
                    "free delay cell is in use",
                )?;
            }
        }
        for path in &self.paths {
            for e in &path.edges {
                let a = plain(spec, e.event, &e.states);
                ensure(
                    !a.reads.iter().chain(&a.writes).any(|k| cells.contains(k)),
                    "pathway model reads/writes delay storage",
                )?;
            }
        }
        self.validate_live(spec, std::slice::from_ref(&spec.initial))?;
        Ok(())
    }
}

fn check_action(
    spec: &dynamic::Spec,
    end: usize,
    cursor: &mut usize,
    mut expected: Action,
    programs: Vec<equation::Program>,
) -> Result<()> {
    ensure(*cursor < end, "missing delay shift action")?;
    let actual = &spec.actions[*cursor];
    expected.program_set = actual.program_set;
    ensure(
        &expected == actual
            && actual
                .program_set
                .is_some_and(|p| spec.program_sets[p] == programs),
        "delay shift action mismatch",
    )?;
    *cursor += 1;
    Ok(())
}
