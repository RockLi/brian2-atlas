// Shared canonical event delivery for exactly target_state += immutable_weight.
// FIFO queues, source/edge traversal order and scalar f64 additions are unchanged.
enum MpiDelays<'a> { Uniform(usize), Edges(&'a [usize]) }
enum MpiWeights<'a> { Scalar(f64), Edges(&'a [f64]) }
impl MpiWeights<'_> {
    fn value(&self, edge: usize) -> f64 {
        match self { Self::Scalar(value)=>*value, Self::Edges(values)=>values[edge] }
    }
}
#[inline(never)]
fn mpi_additive_projection(fired: &[usize], source_start: usize, source_count: usize,
                           tick: usize, end_tick: usize, offsets: &[usize], targets: &[u32],
                           delays: MpiDelays<'_>, weights: MpiWeights<'_>, queue: &mut [Vec<usize>],
                           state: &mut [f64], target_start: usize, owned_start: usize) -> usize {
    if targets.is_empty() { return 0; }
    match delays {
        MpiDelays::Uniform(delay) => {
            let delivery=tick+delay;
            if delivery<end_tick {
                let slot=delivery%queue.len();
                for &neuron in fired {
                    if neuron<source_start || neuron>=source_start+source_count { continue; }
                    let source=neuron-source_start;
                    if offsets[source]!=offsets[source+1] { queue[slot].push(source); }
                }
            }
            let slot=tick%queue.len(); let mut active=std::mem::take(&mut queue[slot]);
            let mut delivered=0;
            for &source in &active {
                for edge in offsets[source]..offsets[source+1] {
                    let target=targets[edge] as usize+target_start-owned_start;
                    state[target]+=weights.value(edge); delivered+=1;
                }
            }
            active.clear(); queue[slot]=active; delivered
        },
        MpiDelays::Edges(delays) => {
            let current_slot = tick % queue.len();
            let remaining = queue.len() - current_slot;
            for &neuron in fired {
                if neuron<source_start || neuron>=source_start+source_count { continue; }
                let source=neuron-source_start;
                for edge in offsets[source]..offsets[source+1] {
                    let delivery=tick+delays[edge];
                    if delivery<end_tick {
                        // Preserve the original modulo for out-of-range delays and
                        // release-mode tick overflow. The common path cannot overflow.
                        let delay = delays[edge];
                        let slot = if delivery >= tick && delay < queue.len() {
                            // A mask avoids an unpredictable branch at the ring boundary.
                            // Wrapping arithmetic also handles lengths above usize::MAX/2.
                            let wrap_mask = 0usize.wrapping_sub(usize::from(delay >= remaining));
                            current_slot.wrapping_add(delay).wrapping_sub(queue.len() & wrap_mask)
                        } else { delivery % queue.len() };
                        queue[slot].push(edge);
                    }
                }
            }
            let slot=tick%queue.len(); let mut active=std::mem::take(&mut queue[slot]);
            for &edge in &active {
                let target=targets[edge] as usize+target_start-owned_start;
                state[target]+=weights.value(edge);
            }
            let delivered=active.len(); active.clear(); queue[slot]=active; delivered
        },
    }
}
