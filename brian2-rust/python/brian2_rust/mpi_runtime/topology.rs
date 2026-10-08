// Distributed fixed-total construction. Draw chunks are disjoint; exclusive
// source-count prefixes recover the reference's source-major edge identities.
extern "C" {
    fn b2mpi_topology_counts(local: *const u64, total: *mut u64, prefix: *mut u64, count: i32) -> i32;
    fn b2mpi_route_edges(send: *const u64, counts: *const i32, receive: *mut u64, capacity: i32, received: *mut i32) -> i32;
}
#[derive(Clone, Copy)]
struct MpiBuildEdge { source: u32, target: u32, global: usize }
const MPI_BUILD_BATCH: usize = 16384;
fn mpi_fixed_total(mpi: &MpiWorld, sources: usize, targets: usize,
                   target_start: usize, population_size: usize, population: usize, edges: usize,
                   seed: u64) -> Result<(Vec<usize>, Vec<u32>, Vec<usize>)> {
    check(sources>0 && targets>0 && edges>0 && edges<=u32::MAX as usize,
          "invalid MPI fixed-total recipe")?;
    check(target_start.checked_add(targets).is_some_and(|end| end<=population_size),
          "invalid MPI target subgroup")?;
    let limit = match std::env::var("B2_MPI_MAX_LOCAL_EDGES") {
        Ok(value) => value.parse::<usize>()?,
        Err(std::env::VarError::NotPresent) => 10_000_000,
        Err(error) => return Err(error.into()),
    };
    check(edges <= limit.saturating_mul(mpi.size), "MPI procedural edge budget exceeded")?;
    let assigned = MPI_POPULATION_OWNERS[population];
    if assigned >= 0 {
        if assigned as usize != mpi.rank { return Ok((Vec::new(),Vec::new(),Vec::new())); }
        let cumulative = mpi.constructed_edges.get().checked_add(edges).ok_or("MPI edge budget overflow")?;
        check(cumulative <= limit, "MPI procedural edge budget exceeded")?;
        mpi.constructed_edges.set(cumulative);
        return mpi_owned_fixed_total(sources,targets,edges,seed);
    }
    let start = edges*mpi.rank/mpi.size;
    let stop = edges*(mpi.rank+1)/mpi.size;
    let owner = |target: usize| if assigned < 0 { ((target+target_start+1)*mpi.size-1)/population_size } else { assigned as usize };
    let mut counts = vec![0u64; sources+mpi.size];
    for draw in start..stop {
        counts[topology_bounded(seed,0,draw,sources)] += 1;
        counts[sources+owner(topology_bounded(seed,1,draw,targets))] += 1;
    }
    let mut total = vec![0u64; counts.len()];
    let mut prefix = vec![0u64; counts.len()];
    mpi_check(unsafe { b2mpi_topology_counts(counts.as_ptr(),total.as_mut_ptr(),prefix.as_mut_ptr(),i32::try_from(counts.len())?) })?;
    check(total[..sources].iter().sum::<u64>()==edges as u64 &&
          total[sources..].iter().sum::<u64>()==edges as u64, "MPI topology counts disagree")?;
    let local_count = usize::try_from(total[sources+mpi.rank])?;
    let cumulative = mpi.constructed_edges.get().checked_add(local_count).ok_or("MPI edge budget overflow")?;
    check(cumulative<=limit, "MPI procedural edge budget exceeded")?;
    mpi.constructed_edges.set(cumulative);
    let mut global_offset = 0u64;
    for source in 0..sources {
        prefix[source] += global_offset;
        global_offset += total[source];
    }
    drop(counts); drop(total);
    let mut local = Vec::<MpiBuildEdge>::new();
    local.try_reserve_exact(local_count)?;
    let mut local_offsets = vec![0usize; if local_count == 0 { 0 } else { sources+1 }];
    let capacity = MPI_BUILD_BATCH.checked_mul(mpi.size).and_then(|v|v.checked_mul(3)).ok_or("MPI batch overflow")?;
    let mut receive = vec![0u64; capacity];
    let mut outgoing: Vec<Vec<u64>> = (0..mpi.size).map(|_| Vec::new()).collect();
    let mut send = Vec::with_capacity(MPI_BUILD_BATCH*3);
    let mut send_counts = vec![0i32; mpi.size];
    let max_draws = (edges+mpi.size-1)/mpi.size;
    for offset in (0..max_draws).step_by(MPI_BUILD_BATCH) {
        for items in &mut outgoing { items.clear(); }
        for draw in (start+offset).min(stop)..(start+offset+MPI_BUILD_BATCH).min(stop) {
            let source = topology_bounded(seed,0,draw,sources);
            let target = topology_bounded(seed,1,draw,targets);
            let global = prefix[source]; prefix[source] += 1;
            outgoing[owner(target)].extend_from_slice(&[source as u64,target as u64,global]);
        }
        send.clear();
        for (rank,items) in outgoing.iter().enumerate() {
            send_counts[rank] = i32::try_from(items.len())?;
            send.extend_from_slice(items);
        }
        let mut got = 0;
        mpi_check(unsafe { b2mpi_route_edges(send.as_ptr(),send_counts.as_ptr(),receive.as_mut_ptr(),i32::try_from(capacity)?,&mut got) })?;
        check(got>=0 && got as usize<=capacity && got%3==0, "invalid MPI construction batch")?;
        check(local.len()+(got as usize)/3<=local_count, "MPI construction exceeded counted edges")?;
        for triple in receive[..got as usize].chunks_exact(3) {
            let source = usize::try_from(triple[0])?;
            let target = usize::try_from(triple[1])?;
            let global = usize::try_from(triple[2])?;
            check(source<sources && target<targets && global<edges && owner(target)==mpi.rank,
                  "invalid/non-owned MPI constructed edge")?;
            local_offsets[source+1] += 1;
            local.push(MpiBuildEdge { source: source as u32, target: target as u32, global });
        }
    }
    check(local.len()==local_count, "MPI construction omitted edges")?;
    local.sort_unstable_by_key(|edge|edge.global);
    check(local.windows(2).all(|pair|pair[0].global<pair[1].global && pair[0].source<=pair[1].source),
          "duplicate/out-of-order MPI global edge identities")?;
    for i in 1..local_offsets.len() { local_offsets[i] += local_offsets[i-1]; }
    let targets = local.iter().map(|edge|edge.target).collect();
    let identities = local.iter().map(|edge|edge.global).collect();
    Ok((local_offsets,targets,identities))
}

// A complete target population has exactly one owner. Different owners build
// their projections concurrently, with no per-projection MPI collective. Stable
// counting placement reproduces reference source-major identities in O(E+N).
#[inline(never)]
fn mpi_owned_fixed_total(sources: usize, targets: usize, edges: usize, seed: u64)
    -> Result<(Vec<usize>,Vec<u32>,Vec<usize>)> {
    let mut offsets=vec![0usize;sources+1];
    for draw in 0..edges { offsets[topology_bounded(seed,0,draw,sources)+1]+=1; }
    for source in 0..sources { offsets[source+1]+=offsets[source]; }
    let mut cursor=offsets[..sources].to_vec();
    let mut target_index=Vec::new(); target_index.try_reserve_exact(edges)?;
    target_index.resize(edges,0u32);
    for draw in 0..edges {
        let source=topology_bounded(seed,0,draw,sources);
        target_index[cursor[source]]=topology_bounded(seed,1,draw,targets) as u32;
        cursor[source]+=1;
    }
    check(cursor.iter().zip(&offsets[1..]).all(|(a,b)|a==b), "MPI owner CSR mismatch")?;
    drop(cursor);
    let mut identities=Vec::new(); identities.try_reserve_exact(edges)?;
    identities.extend(0..edges);
    Ok((offsets,target_index,identities))
}

// Shared non-generic initialization kernels keep LLVM work independent of the
// number of projections. Global edge identities and f64 arithmetic are unchanged.
#[derive(Clone, Copy)]
enum MpiInitializer {
    ClippedNormal(u64, f64, f64, Option<f64>, Option<f64>),
    Uniform(u64, f64, f64),
}
impl MpiInitializer {
    fn value(self, seed: u64, edge: usize) -> Result<f64> {
        Ok(match self {
            Self::ClippedNormal(stream, mean, std, minimum, maximum) =>
                clipped_normal_value(seed,stream,edge,mean,std,minimum,maximum)
                    .ok_or("clipped-normal rejection limit exceeded")?,
            Self::Uniform(stream, low, high) => low + (high-low) *
                (((topology_draw(seed,stream,edge,0)>>11) as f64+0.5)*(1.0/9007199254740992.0)),
        })
    }
}
#[inline(never)]
fn mpi_parameter_values(identities: &[usize], seed: u64, recipe: MpiInitializer) -> Result<Vec<f64>> {
    let mut values=Vec::new(); values.try_reserve_exact(identities.len())?;
    for &edge in identities { values.push(recipe.value(seed,edge)?); }
    Ok(values)
}
#[inline(never)]
fn mpi_delay_values(identities: &[usize], seed: u64, recipe: MpiInitializer, dt: f64) -> Result<Vec<usize>> {
    let mut values=Vec::new(); values.try_reserve_exact(identities.len())?;
    for &edge in identities { values.push((recipe.value(seed,edge)?/dt+0.5).floor() as usize); }
    Ok(values)
}
