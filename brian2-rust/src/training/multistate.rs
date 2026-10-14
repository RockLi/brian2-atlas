//! Coupled state-vector BPTT. Every update reads the previous tick's complete
//! vector. Reset programs are simultaneous expressions for the final values;
//! the frontend composes sequential Brian assignments before lowering them.
use super::*;

pub(super) fn execute(
    plan: &Plan, state: State, inputs: &[Vec<Vec<f64>>], labels: &[usize], live: Vec<Vec<f64>>,
    bytes: usize, operation: &str, mpi: Option<&mpi::Context>, start_tick: u64, noise_sequence: u64,
    cache: Option<&poisson_cache::Checkpoint>,
) -> Result<Output> {
    execute_inner(plan,state,inputs,labels,live,bytes,operation,mpi,start_tick,noise_sequence,
        cache,None,0,inputs.len())
}

fn execute_inner(
    plan: &Plan,
    mut state: State,
    inputs: &[Vec<Vec<f64>>],
    labels: &[usize],
    mut live: Vec<Vec<f64>>,
    bytes: usize,
    operation: &str,
    mpi: Option<&mpi::Context>,
    start_tick: u64,
    noise_sequence: u64,
    cache: Option<&poisson_cache::Checkpoint>,
    forced: Option<poisson_cache::Identity>,
    batch_origin: usize,
    total_batch: usize,
) -> Result<Output> {
    let (offsets, n) = plan.validate()?;
    let width = plan.state_width();
    let batch = inputs.len();
    let time = inputs[0].len();
    let has_poisson = static_poisson::used(plan);
    let reconcile_errors = has_poisson || timed_input::static_used(plan);
    let _draw_scope = poisson_cache::install_static(plan,noise_sequence,total_batch,cache,forced)?;
    let boundary_possible = operation != "evaluate" && static_poisson::boundary_possible(plan);
    let original = if boundary_possible { live.clone() } else { Vec::new() };
    let layers = plan.sizes.len() - 1;
    let updates = plan.state_equations.as_ref().unwrap();
    let resets = plan.state_resets.as_ref().unwrap();
    let uniform_masks: Vec<u16> = updates.iter().zip(resets).map(|(u,r)| equation::noise_masks(u.iter().chain(r)).1).collect();
    let projections = plan.projections.as_ref().unwrap();
    let mut state_offsets = vec![0];
    for l in 0..layers {
        state_offsets.push(state_offsets[l] + updates[l].len() * plan.sizes[l + 1]);
    }
    let index = |l: usize, s: usize, j: usize| state_offsets[l] + s * plan.sizes[l + 1] + j;
    let samples = |b:usize,l:usize,j:usize,t:usize| static_poisson::samples(plan,uniform_masks[l],
        noise_sequence,b+batch_origin,l,j,start_tick+t as u64,has_poisson);
    let mut detached = vec![false; width];
    for l in 0..layers {
        if plan.refractory.as_ref().is_some_and(|r| r[l].is_some()) {
            for j in 0..plan.sizes[l + 1] { detached[index(l,updates[l].len()-1,j)] = true; }
        }
    }
    let refractory = |l: usize| plan.refractory.as_ref().and_then(|r| r[l].as_ref());
    let active = |old: &[f64], at: usize, l: usize, j: usize| {
        refractory(l).is_none() || old[at + index(l, updates[l].len() - 1, j)] == 0.0
    };
    let owns = |k| mpi.is_none_or(|m| m.owns(k, n));
    let theta = (0..layers)
        .map(|l| {
            (0..plan.sizes[l + 1]).map(|j| plan.threshold_reference(l, j)
                .map_or(plan.threshold[l], |[b, i]| state.weights[b][i])).collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let mut old = vec![0.0; batch * time * width];
    let mut before_reset = old.clone();
    let mut pre_voltage = vec![0.0; batch * time * n];
    let mut spikes = pre_voltage.clone();
    for b in 0..batch {
        for t in 0..time {
            let at = (b * time + t) * width;
            let nt = (b * time + t) * n;
            old[at..at + width].copy_from_slice(&live[b]);
            live[b].fill(0.0);
            for l in 0..layers {
                let count = updates[l].len();
                for j in 0..plan.sizes[l + 1] {
                    let k = offsets[l] + j;
                    let action = static_poisson::action(plan,l,j);
                    poisson_cache::begin(&action,b+batch_origin,t,k,start_tick+t as u64,true,owns(k));
                    let result = (|| -> Result<()> {
                    if !owns(k) { return Ok(()); }
                    let mut values = [0.0; 16];
                    for s in 0..count {
                        values[s] = old[at + index(l, s, j)];
                    }
                    for s in 0..count {
                        if let Some(spec) = refractory(l) {
                            if s == count - 1 {
                                live[b][index(l, s, j)] = (values[s] - 1.0).max(0.0);
                                continue;
                            }
                            if !active(&old, at, l, j) && spec.clamp.contains(&s) {
                                live[b][index(l, s, j)] = values[s];
                                continue;
                            }
                        }
                        live[b][index(l, s, j)] = equation::forward_states(
                            &updates[l][s],
                            &values[..count],
                            &state.weights,
                            j,
                            plan.time_at(start_tick + t as u64),
                            &samples(b,l,j,t),
                        )?;
                    }
                    let v = live[b][index(l, 0, j)];
                    pre_voltage[nt + k] = v;
                    spikes[nt + k] = if active(&old, at, l, j) && v > theta[l][j] { 1.0 } else { 0.0 };
                    Ok(())
                    })();
                    static_poisson::reconcile(result,mpi,reconcile_errors)?;
                    poisson_cache::sync(mpi)?;
                }
            }
            if let Some(m) = mpi {
                m.sum(&mut spikes[nt..nt + n])?;
            }
            for (q, p) in projections.iter().enumerate() {
                for e in 0..p.sources.len() {
                    let l = p.target_layer - 1;
                    let j = p.targets[e];
                    if !owns(offsets[l] + j) {
                        continue;
                    }
                    if refractory(l).is_some_and(|r| r.clamp.contains(&0)) && (!active(&old, at, l, j) || spikes[nt + offsets[l] + j] != 0.0) { continue; }
                    let x = if p.source_layer == 0 {
                        inputs[b][t][p.sources[e]]
                    } else {
                        spikes[nt + offsets[p.source_layer - 1] + p.sources[e]]
                    };
                    live[b][index(l, 0, j)] += state.weights[q][p.parameter_ids[e]] * x;
                }
            }
            static_poisson::reconcile(ensure(
                live[b].iter().all(|x| x.is_finite()),
                "nonfinite multi-state synaptic update",
            ),mpi,reconcile_errors)?;
            before_reset[at..at + width].copy_from_slice(&live[b]);
            for l in 0..layers {
                let count = updates[l].len();
                for j in 0..plan.sizes[l + 1] {
                    let k = offsets[l]+j;
                    let action = static_poisson::action(plan,l,j);
                    poisson_cache::begin(&action,b+batch_origin,t,n+k,start_tick+t as u64,true,owns(k)&&spikes[nt+k]!=0.);
                    let result = (|| -> Result<()> {
                    if !owns(k) { return Ok(()); }
                    let mut values = [0.0; 16];
                    for s in 0..count {
                        values[s] = before_reset[at + index(l, s, j)];
                    }
                    if spikes[nt + offsets[l] + j] != 0.0 {
                        for s in 0..count {
                            if let Some(spec) = refractory(l) {
                                if s == count - 1 {
                                    live[b][index(l, s, j)] = spec.steps.saturating_sub(1) as f64;
                                    continue;
                                }
                            }
                            live[b][index(l, s, j)] = equation::forward_states(
                                &resets[l][s],
                                &values[..count],
                                &state.weights,
                                j,
                                plan.time_at(start_tick + t as u64),
                                &samples(b,l,j,t),
                            )?;
                        }
                    }
                    Ok(())
                    })();
                    static_poisson::reconcile(result,mpi,reconcile_errors)?;
                    poisson_cache::sync(mpi)?;
                }
            }
            if let Some(m) = mpi {
                m.sum(&mut live[b])?;
            }
        }
    }
    let classes = plan.sizes[layers];
    let mut logits = vec![vec![0.0; classes]; batch];
    let mut seeds = logits.clone();
    let mut loss = 0.0;
    let mut sample_losses = vec![0.;batch];
    for b in 0..batch {
        for t in 0..time {
            for j in 0..classes {
                logits[b][j] += spikes[(b * time + t) * n + offsets[layers - 1] + j]
                    * plan.logit_scale
                    / time as f64;
            }
        }
        let maximum = logits[b].iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let sum = logits[b].iter().map(|v| (v - maximum).exp()).sum::<f64>();
        sample_losses[b] = (maximum + sum.ln() - logits[b][labels[b]]) / batch as f64;
        loss += sample_losses[b];
        for j in 0..classes {
            seeds[b][j] = ((logits[b][j] - maximum).exp() / sum
                - if j == labels[b] { 1.0 } else { 0.0 })
                / batch as f64
                * plan.logit_scale
                / time as f64;
        }
    }
    let mut gradients = state
        .weights
        .iter()
        .map(|w| vec![0.0; w.len()])
        .collect::<Vec<_>>();
    let mut carry = vec![vec![0.0; width]; batch];
    if operation != "evaluate" {
        for b in 0..batch {
            for t in (0..time).rev() {
                let at = (b * time + t) * width;
                let nt = (b * time + t) * n;
                let mut dz = vec![0.0; width];
                let mut ds = vec![0.0; n];
                let mut previous = vec![0.0; width];
                if mpi.is_none_or(|m| m.rank == 0) {
                    ds[offsets[layers - 1]..offsets[layers]].copy_from_slice(&seeds[b]);
                }
                // Reset VJP precedes edge VJP: auxiliary reset expressions
                // may depend on voltage after synaptic accumulation.
                for l in 0..layers {
                    let count = updates[l].len();
                    for j in 0..plan.sizes[l + 1] {
                        let k = offsets[l] + j;
                        let spike = spikes[nt + k];
                        let mut values = [0.0; 16];
                        let mut adjoints = [0.0; 16];
                        for s in 0..count {
                            values[s] = before_reset[at + index(l, s, j)];
                        }
                        let action = static_poisson::action(plan,l,j);
                        poisson_cache::begin(&action,b+batch_origin,t,n+k,start_tick+t as u64,false,false);
                        let cells = (0..count).map(|s| index(l,s,j)).collect::<Vec<_>>();
                        let boundary = static_poisson::boundary(resets[l].iter(),plan,&state.weights,&values[..count],
                            &cells,&detached,&action,b+batch_origin,start_tick+t as u64,plan.time_at(start_tick+t as u64),
                            &samples(b,l,j,t),owns(k),boundary_possible&&spike!=0.,mpi,sample_losses[b],batch,
                            |force| execute_inner(plan,state.clone(),&inputs[b..b+1],&labels[b..b+1],vec![original[b].clone()],
                                bytes,"evaluate",mpi,start_tick,noise_sequence,cache,Some(force),b+batch_origin,total_batch).map(|r|r.loss))?;
                        let result = (|| -> Result<()> {
                        if !owns(k) { return Ok(()); }
                        if has_poisson && spike != 0. {
                            equation::backward_poisson(resets[l].iter(),&values[..count],&state.weights,sample_losses[b],&boundary,
                                &mut gradients,&plan.masks,&mut adjoints[..count],j,plan.time_at(start_tick+t as u64),
                                &samples(b,l,j,t),&[],&[],&cells,&detached)?;
                        }
                        for s in 0..count {
                            if refractory(l).is_some() && s == count - 1 { continue; }
                            let g = carry[b][index(l, s, j)];
                            adjoints[s] += g * (1.0 - spike);
                            if spike != 0.0 && g != 0.0 {
                                equation::backward_states(
                                    &resets[l][s],
                                    &values[..count],
                                    &state.weights,
                                    g,
                                    &mut gradients,
                                    &plan.masks,
                                    &mut adjoints[..count],
                                    j,
                                    plan.time_at(start_tick + t as u64),
                                    &samples(b,l,j,t),
                                )?;
                            }
                            if !plan.detach_reset && g != 0.0 && active(&old, at, l, j) {
                                let reset = equation::forward_states(
                                    &resets[l][s],
                                    &values[..count],
                                    &state.weights,
                                    j,
                                    plan.time_at(start_tick + t as u64),
                                    &samples(b,l,j,t),
                                )?;
                                ds[k] += g * (reset - values[s]);
                            }
                        }
                        for s in 0..count {
                            dz[index(l, s, j)] = adjoints[s];
                        }
                        Ok(())
                        })();
                        static_poisson::reconcile(result,mpi,reconcile_errors)?;
                    }
                }
                for (q, p) in projections.iter().enumerate() {
                    for e in 0..p.sources.len() {
                        if !owns(offsets[p.target_layer - 1] + p.targets[e]) {
                            continue;
                        }
                        if refractory(p.target_layer - 1).is_some_and(|r| r.clamp.contains(&0))
                            && (!active(&old, at, p.target_layer - 1, p.targets[e]) || spikes[nt + offsets[p.target_layer - 1] + p.targets[e]] != 0.0) { continue; }
                        let g = dz[index(p.target_layer - 1, 0, p.targets[e])];
                        let id = p.parameter_ids[e];
                        let x = if p.source_layer == 0 {
                            inputs[b][t][p.sources[e]]
                        } else {
                            spikes[nt + offsets[p.source_layer - 1] + p.sources[e]]
                        };
                        gradients[q][id] += g * x * plan.masks[q][id];
                        if p.source_layer > 0 {
                            ds[offsets[p.source_layer - 1] + p.sources[e]] +=
                                g * state.weights[q][id];
                        }
                    }
                }
                if let Some(m) = mpi {
                    m.sum(&mut ds)?;
                }
                for l in 0..layers {
                    let count = updates[l].len();
                    for j in 0..plan.sizes[l + 1] {
                        let k = offsets[l] + j;
                        let mut values = [0.0;16];
                        for s in 0..count { values[s] = old[at+index(l,s,j)]; }
                        let action = static_poisson::action(plan,l,j);
                        poisson_cache::begin(&action,b+batch_origin,t,k,start_tick+t as u64,false,false);
                        let cells = (0..count).map(|s| index(l,s,j)).collect::<Vec<_>>();
                        let enabled = |s:usize| refractory(l).is_none_or(|r| s!=count-1
                            && (active(&old,at,l,j) || !r.clamp.contains(&s)));
                        let programs = || updates[l].iter().enumerate().filter(|(s,_)| enabled(*s)).map(|(_,p)|p);
                        let boundary = static_poisson::boundary(programs(),plan,&state.weights,&values[..count],
                            &cells,&detached,&action,b+batch_origin,start_tick+t as u64,plan.time_at(start_tick+t as u64),
                            &samples(b,l,j,t),owns(k),boundary_possible,mpi,sample_losses[b],batch,
                            |force| execute_inner(plan,state.clone(),&inputs[b..b+1],&labels[b..b+1],vec![original[b].clone()],
                                bytes,"evaluate",mpi,start_tick,noise_sequence,cache,Some(force),b+batch_origin,total_batch).map(|r|r.loss))?;
                        let result = (|| -> Result<()> {
                        if !owns(k) { return Ok(()); }
                        let spike_vjp = if active(&old, at, l, j) {
                            ds[k] * plan.surrogate.derivative(pre_voltage[nt + k] - theta[l][j])
                        } else { 0.0 };
                        dz[index(l, 0, j)] += spike_vjp;
                        if let Some([bank, i]) =
                            plan.threshold_reference(l, j)
                        {
                            gradients[bank][i] -= spike_vjp * plan.masks[bank][i];
                        }
                        let mut adjoints = [0.0; 16];
                        if has_poisson {
                            equation::backward_poisson(programs(),&values[..count],&state.weights,sample_losses[b],&boundary,
                                &mut gradients,&plan.masks,&mut adjoints[..count],j,plan.time_at(start_tick+t as u64),
                                &samples(b,l,j,t),&[],&[],&cells,&detached)?;
                        }
                        for s in 0..count {
                            if let Some(spec) = refractory(l) {
                                if s == count - 1 { continue; }
                                if !active(&old, at, l, j) && spec.clamp.contains(&s) {
                                    adjoints[s] += dz[index(l, s, j)];
                                    continue;
                                }
                            }
                            equation::backward_states(
                                &updates[l][s],
                                &values[..count],
                                &state.weights,
                                dz[index(l, s, j)],
                                &mut gradients,
                                &plan.masks,
                                &mut adjoints[..count],
                                j,
                                plan.time_at(start_tick + t as u64),
                                &samples(b,l,j,t),
                            )?;
                        }
                        for s in 0..count {
                            previous[index(l, s, j)] = adjoints[s];
                        }
                        Ok(())
                        })();
                        static_poisson::reconcile(result,mpi,reconcile_errors)?;
                    }
                }
                if plan.tbptt_window.is_some_and(|w| t > 0 && t % w == 0) {
                    previous.fill(0.0);
                }
                if let Some(m) = mpi {
                    m.sum(&mut previous)?;
                }
                carry[b] = previous;
            }
        }
    }
    if let Some(m) = mpi {
        for row in &mut gradients {
            m.sum(row)?;
        }
    }
    ensure(
        loss.is_finite()
            && gradients
                .iter()
                .flatten()
                .chain(carry.iter().flatten())
                .chain(live.iter().flatten())
                .all(|v| v.is_finite()),
        "nonfinite multi-state result",
    )?;
    if operation == "train" {
        apply_optimizer_distributed(plan, &mut state, &gradients, mpi)?;
    }
    let voltage_rows = |values: &[Vec<f64>]| {
        values
            .iter()
            .map(|row| {
                (0..layers)
                    .flat_map(|l| {
                        row[state_offsets[l]..state_offsets[l] + plan.sizes[l + 1]]
                            .iter()
                            .copied()
                    })
                    .collect()
            })
            .collect()
    };
    Ok(Output { poisson_state: poisson_cache::checkpoint(), event_visits: None, clock_state: None, updated_dynamic: None,
        final_tick: None,
        noise_sequence: None,
        schema: "b2-lif-training-result-v1",
        state,
        loss,
        backend: "cpu",
        numeric_profile: if mpi.is_some() {
            "native-mpi-multistate-target-owned-f64-ordered-rank-reduction"
        } else {
            "native-cpu-multistate-f64"
        },
        gpu_dispatches: 0,
        gradients,
        initial_gradients: voltage_rows(&carry),
        final_membrane: voltage_rows(&live),
        initial_state_gradients: Some(carry),
        final_state: Some(live),
        logits,
        tape_bytes: bytes,
        spikes: spikes
            .chunks_exact(time * n)
            .map(|s| s.chunks_exact(n).map(|r| r.to_vec()).collect())
            .collect(),
        gradient_scope: if plan.tbptt_window.is_some_and(|w| w < time) {
            "tbptt-detach-boundaries"
        } else {
            "full-bptt"
        },
    })
}
