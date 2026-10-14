//! Scalar native BPTT with direct clock and stochastic observation contexts.
//! Subtract reset remains before ordered projections; zero reset remains after.
use super::*;

pub(super) fn execute(
    plan: &Plan,
    state: State,
    inputs: &[Vec<Vec<f64>>],
    labels: &[usize],
    membrane: Vec<Vec<f64>>,
    bytes: usize,
    operation: &str,
    mpi: Option<&mpi::Context>,
    start_tick: u64,
    noise_sequence: u64,
    cache: Option<&poisson_cache::Checkpoint>,
) -> Result<Output> {
    execute_inner(
        plan,
        state,
        inputs,
        labels,
        membrane,
        bytes,
        operation,
        mpi,
        start_tick,
        noise_sequence,
        cache,
        None,
        0,
        inputs.len(),
    )
}

fn execute_inner(
    plan: &Plan,
    mut state: State,
    inputs: &[Vec<Vec<f64>>],
    labels: &[usize],
    mut membrane: Vec<Vec<f64>>,
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
    let thresholds = validate_state(plan, &state)?;
    let layers = plan.sizes.len() - 1;
    let batch = inputs.len();
    let time = inputs[0].len();
    let cells = batch * time * n;
    let owns = |k| mpi.is_none_or(|m| m.owns(k, n));
    let has_poisson = static_poisson::used(plan);
    let reconcile_errors = plan.equations.is_some();
    let _draw_scope =
        poisson_cache::install_static(plan, noise_sequence, total_batch, cache, forced)?;
    let boundary_possible = operation != "evaluate" && static_poisson::boundary_possible(plan);
    let original = if boundary_possible {
        membrane.clone()
    } else {
        Vec::new()
    };
    let detached = vec![false; n];
    let uniform_masks = (0..layers)
        .map(|l| equation::noise_masks(static_poisson::programs(plan, l)).1)
        .collect::<Vec<_>>();
    let samples = |b: usize, l: usize, j: usize, t: usize| {
        static_poisson::samples(
            plan,
            uniform_masks[l],
            noise_sequence,
            b + batch_origin,
            l,
            j,
            start_tick + t as u64,
            has_poisson,
        )
    };
    let mut before_update = if plan.equations.is_some() {
        vec![0.0; cells]
    } else {
        Vec::new()
    };
    let mut pre = vec![0.0; cells];
    let mut spikes = vec![0.0; cells];
    let mut pre_reset = if plan.projections.is_some() {
        vec![0.0; cells]
    } else {
        Vec::new()
    };
    for b in 0..batch {
        for t in 0..time {
            let at = (b * time + t) * n;
            // All thresholds precede all synaptic writes, including feedback.
            for l in 0..layers {
                for j in 0..plan.sizes[l + 1] {
                    let k = offsets[l] + j;
                    let action = static_poisson::action(plan, l, j);
                    poisson_cache::begin(
                        &action,
                        b + batch_origin,
                        t,
                        k,
                        start_tick + t as u64,
                        true,
                        owns(k),
                    );
                    let result = (|| -> Result<()> {
                        if !owns(k) {
                            membrane[b][k] = 0.;
                            return Ok(());
                        }
                        let u = if let Some(programs) = &plan.equations {
                            before_update[at + k] = membrane[b][k];
                            equation::forward_states(
                                &programs[l],
                                &[membrane[b][k]],
                                &state.weights,
                                j,
                                plan.time_at(start_tick + t as u64),
                                &samples(b, l, j, t),
                            )?
                        } else {
                            plan.beta[l] * membrane[b][k]
                        };
                        pre[at + k] = u;
                        let s = if u > thresholds[l] { 1.0 } else { 0.0 };
                        spikes[at + k] = s;
                        membrane[b][k] = if plan.reset == "zero" {
                            u
                        } else {
                            u - thresholds[l] * s
                        };
                        Ok(())
                    })();
                    static_poisson::reconcile(result, mpi, reconcile_errors)?;
                    poisson_cache::sync(mpi)?;
                }
            }
            if let Some(m) = mpi {
                m.sum(&mut pre[at..at + n])?;
                m.sum(&mut spikes[at..at + n])?;
            }
            if let Some(projections) = &plan.projections {
                for (q, p) in projections.iter().enumerate() {
                    for e in 0..p.sources.len() {
                        if !owns(offsets[p.target_layer - 1] + p.targets[e]) {
                            continue;
                        }
                        let x = if p.source_layer == 0 {
                            inputs[b][t][p.sources[e]]
                        } else {
                            spikes[at + offsets[p.source_layer - 1] + p.sources[e]]
                        };
                        membrane[b][offsets[p.target_layer - 1] + p.targets[e]] +=
                            state.weights[q][p.parameter_ids[e]] * x;
                    }
                }
                // Zero-reset VJP needs the sum from every incoming projection.
                pre_reset[at..at + n].copy_from_slice(&membrane[b]);
                static_poisson::reconcile(
                    ensure(
                        membrane[b].iter().all(|v| v.is_finite()),
                        "nonfinite forward membrane",
                    ),
                    mpi,
                    reconcile_errors,
                )?;
            } else {
                for l in 0..layers {
                    for j in 0..plan.sizes[l + 1] {
                        if !owns(offsets[l] + j) {
                            continue;
                        }
                        for i in 0..plan.sizes[l] {
                            let x = if l == 0 {
                                inputs[b][t][i]
                            } else {
                                spikes[at + offsets[l - 1] + i]
                            };
                            membrane[b][offsets[l] + j] +=
                                state.weights[l][i * plan.sizes[l + 1] + j] * x;
                        }
                        ensure(
                            membrane[b][offsets[l] + j].is_finite(),
                            "nonfinite forward membrane",
                        )?;
                    }
                }
            }
            if plan.reset == "zero" {
                for k in 0..n {
                    membrane[b][k] *= 1.0 - spikes[at + k];
                }
            }
            if let Some(m) = mpi {
                m.sum(&mut membrane[b])?;
            }
        }
    }
    let classes = plan.sizes[layers];
    let mut logits = vec![vec![0.0; classes]; batch];
    let mut dlogits = logits.clone();
    let mut loss = 0.0;
    let mut sample_losses = vec![0.; batch];
    for b in 0..batch {
        for t in 0..time {
            for j in 0..classes {
                logits[b][j] += spikes[(b * time + t) * n + offsets[layers - 1] + j]
                    * plan.logit_scale
                    / time as f64;
            }
        }
        let maximum = logits[b].iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let sum = logits[b].iter().map(|&v| (v - maximum).exp()).sum::<f64>();
        sample_losses[b] = (maximum + sum.ln() - logits[b][labels[b]]) / batch as f64;
        loss += sample_losses[b];
        for j in 0..classes {
            dlogits[b][j] = ((logits[b][j] - maximum).exp() / sum
                - if j == labels[b] { 1.0 } else { 0.0 })
                / batch as f64;
        }
    }
    let mut gradients = state
        .weights
        .iter()
        .map(|w| vec![0.0; w.len()])
        .collect::<Vec<_>>();
    let mut carry = vec![vec![0.0; n]; batch];
    if operation != "evaluate" {
        for b in 0..batch {
            for t in (0..time).rev() {
                let at = (b * time + t) * n;
                let mut ds = vec![0.0; n];
                let mut previous = vec![0.0; n];
                for j in 0..classes {
                    if mpi.is_none_or(|m| m.rank == 0) {
                        ds[offsets[layers - 1] + j] =
                            dlogits[b][j] * plan.logit_scale / time as f64;
                    }
                }
                if let Some(projections) = &plan.projections {
                    // All edge VJPs precede all neuron VJPs. Layer ordering
                    // cannot substitute for this barrier when feedback exists.
                    for (q, p) in projections.iter().enumerate() {
                        for e in 0..p.sources.len() {
                            let k = offsets[p.target_layer - 1] + p.targets[e];
                            if !owns(k) {
                                continue;
                            }
                            let id = p.parameter_ids[e];
                            let g = carry[b][k]
                                * if plan.reset == "zero" {
                                    1.0 - spikes[at + k]
                                } else {
                                    1.0
                                };
                            let x = if p.source_layer == 0 {
                                inputs[b][t][p.sources[e]]
                            } else {
                                spikes[at + offsets[p.source_layer - 1] + p.sources[e]]
                            };
                            gradients[q][id] += g * x * plan.masks[q][id];
                            if p.source_layer > 0 {
                                ds[offsets[p.source_layer - 1] + p.sources[e]] +=
                                    g * state.weights[q][id];
                            }
                        }
                    }
                } else {
                    for l in (0..layers).rev() {
                        for i in 0..plan.sizes[l] {
                            for j in 0..plan.sizes[l + 1] {
                                let e = i * plan.sizes[l + 1] + j;
                                if !owns(offsets[l] + j) {
                                    continue;
                                }
                                let g = carry[b][offsets[l] + j]
                                    * if plan.reset == "zero" {
                                        1.0 - spikes[at + offsets[l] + j]
                                    } else {
                                        1.0
                                    };
                                let x = if l == 0 {
                                    inputs[b][t][i]
                                } else {
                                    spikes[at + offsets[l - 1] + i]
                                };
                                gradients[l][e] += g * x * plan.masks[l][e];
                                if l > 0 {
                                    ds[offsets[l - 1] + i] += g * state.weights[l][e];
                                }
                            }
                        }
                    }
                }
                if let Some(m) = mpi {
                    m.sum(&mut ds)?;
                }
                for l in (0..layers).rev() {
                    for j in 0..plan.sizes[l + 1] {
                        let k = offsets[l] + j;
                        let action = static_poisson::action(plan, l, j);
                        poisson_cache::begin(
                            &action,
                            b + batch_origin,
                            t,
                            k,
                            start_tick + t as u64,
                            false,
                            false,
                        );
                        let values = [before_update.get(at + k).copied().unwrap_or(0.)];
                        let programs = || plan.equations.iter().map(|p| &p[l]);
                        let boundary = static_poisson::boundary(
                            programs(),
                            plan,
                            &state.weights,
                            &values,
                            &[k],
                            &detached,
                            &action,
                            b + batch_origin,
                            start_tick + t as u64,
                            plan.time_at(start_tick + t as u64),
                            &samples(b, l, j, t),
                            owns(k),
                            boundary_possible,
                            mpi,
                            sample_losses[b],
                            batch,
                            |force| {
                                execute_inner(
                                    plan,
                                    state.clone(),
                                    &inputs[b..b + 1],
                                    &labels[b..b + 1],
                                    vec![original[b].clone()],
                                    bytes,
                                    "evaluate",
                                    mpi,
                                    start_tick,
                                    noise_sequence,
                                    cache,
                                    Some(force),
                                    b + batch_origin,
                                    total_batch,
                                )
                                .map(|r| r.loss)
                            },
                        )?;
                        let result = (|| -> Result<()> {
                            if !owns(k) {
                                return Ok(());
                            }
                            let u = pre[at + k];
                            let s = spikes[at + k];
                            let phi = plan.surrogate.derivative(u - thresholds[l]);
                            let mut before_reset = u;
                            if plan.projections.is_some() {
                                before_reset = pre_reset[at + k];
                            } else if plan.reset == "zero" && !plan.detach_reset {
                                for i in 0..plan.sizes[l] {
                                    let x = if l == 0 {
                                        inputs[b][t][i]
                                    } else {
                                        spikes[at + offsets[l - 1] + i]
                                    };
                                    before_reset += state.weights[l][i * plan.sizes[l + 1] + j] * x;
                                }
                            }
                            let reset_derivative = if plan.reset == "zero" {
                                1.0 - s
                                    - if plan.detach_reset {
                                        0.0
                                    } else {
                                        before_reset * phi
                                    }
                            } else {
                                1.0 - if plan.detach_reset {
                                    0.0
                                } else {
                                    thresholds[l] * phi
                                }
                            };
                            if let Some([bank, index]) =
                                plan.threshold_parameters.as_ref().and_then(|p| p[l])
                            {
                                let reset_theta = if plan.reset == "zero" {
                                    if plan.detach_reset {
                                        0.0
                                    } else {
                                        before_reset * phi
                                    }
                                } else {
                                    -s + if plan.detach_reset {
                                        0.0
                                    } else {
                                        thresholds[l] * phi
                                    }
                                };
                                gradients[bank][index] += (carry[b][k] * reset_theta - ds[k] * phi)
                                    * plan.masks[bank][index];
                            }
                            let du = carry[b][k] * reset_derivative + ds[k] * phi;
                            previous[k] = if let Some(programs) = &plan.equations {
                                let mut adjoints = [0.];
                                if has_poisson {
                                    equation::backward_poisson(
                                        std::iter::once(&programs[l]),
                                        &values,
                                        &state.weights,
                                        sample_losses[b],
                                        &boundary,
                                        &mut gradients,
                                        &plan.masks,
                                        &mut adjoints,
                                        j,
                                        plan.time_at(start_tick + t as u64),
                                        &samples(b, l, j, t),
                                        &[],
                                        &[],
                                        &[k],
                                        &detached,
                                    )?;
                                }
                                equation::backward_states(
                                    &programs[l],
                                    &values,
                                    &state.weights,
                                    du,
                                    &mut gradients,
                                    &plan.masks,
                                    &mut adjoints,
                                    j,
                                    plan.time_at(start_tick + t as u64),
                                    &samples(b, l, j, t),
                                )?;
                                adjoints[0]
                            } else {
                                plan.beta[l] * du
                            };
                            Ok(())
                        })();
                        static_poisson::reconcile(result, mpi, reconcile_errors)?;
                    }
                }
                if plan
                    .tbptt_window
                    .is_some_and(|window| t > 0 && t % window == 0)
                {
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
                .all(|v| v.is_finite()),
        "nonfinite backward result",
    )?;
    if operation == "train" {
        apply_optimizer_distributed(plan, &mut state, &gradients, mpi)?;
    }
    let shaped = (0..batch)
        .map(|b| {
            (0..time)
                .map(|t| spikes[(b * time + t) * n..(b * time + t + 1) * n].to_vec())
                .collect()
        })
        .collect();
    Ok(Output {
        poisson_state: poisson_cache::checkpoint(),
        event_visits: None,
        clock_state: None,
        updated_dynamic: None,
        noise_sequence: plan.noise_streams.as_ref().map(|_| noise_sequence),
        final_tick: plan.clock.as_ref().map(|_| start_tick + time as u64),
        schema: "b2-lif-training-result-v1",
        state,
        loss,
        backend: "cpu",
        numeric_profile: if mpi.is_some() {
            "native-mpi-target-owned-f64-ordered-rank-reduction"
        } else {
            "native-cpu-f64"
        },
        gpu_dispatches: 0,
        gradients,
        initial_gradients: carry,
        final_membrane: membrane,
        final_state: None,
        initial_state_gradients: None,
        spikes: shaped,
        logits,
        tape_bytes: bytes,
        gradient_scope: if plan.tbptt_window.is_some_and(|w| w < time) {
            "tbptt-detach-boundaries"
        } else {
            "full-bptt"
        },
    })
}
