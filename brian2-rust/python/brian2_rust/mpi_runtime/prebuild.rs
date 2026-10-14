// Optional early shared topology construction. Cache admission covers retained
// vectors and slot metadata; existing service/edge guards bound construction
// temporaries separately. Canonical readers and initializers keep their order.
type MpiTopology = (Vec<usize>, Vec<u32>, Vec<usize>);
struct MpiPreparedTopology { arrays: MpiTopology, seconds: f64, bytes: usize }
#[derive(Clone, Copy, PartialEq)]
struct MpiPrebuildRecipe {
    ordinal: usize, projection: usize, sources: usize, targets: usize,
    target_start: usize, population_size: usize, population: usize,
    edges: usize, seed: u64,
}
#[derive(Default)]
struct MpiPrebuilt {
    entries: Vec<Option<MpiPreparedTopology>>,
    next_fixed: usize, next_shared: usize,
    limit: usize, resident: usize, peak: usize, metadata: usize,
    pending: usize, last_seconds: f64, build_seconds: f64, verify_seconds: f64,
    preparing: bool,
}
fn mpi_prebuilt_verify(mpi: &MpiWorld, path: &Path, expected: [u8;32]) -> Result<()> {
    let started = Instant::now();
    let mut file = File::open(path)?;
    let mut buffer = [0u8;65536];
    let mut hash = Sha256::new();
    loop {
        let n = std::io::Read::read(&mut file, &mut buffer)?;
        if n == 0 { break; }
        hash.update(&buffer[..n]);
    }
    check(hash.finish() == expected, "MPI instance differs from compiled plan; rebuild required")?;
    mpi.prebuilt.borrow_mut().verify_seconds = started.elapsed().as_secs_f64();
    Ok(())
}
fn mpi_prebuilt_bytes(sources: usize, edges: usize) -> Result<usize> {
    let csr = if edges == 0 { 0 } else {
        sources.checked_add(1).and_then(|n|n.checked_mul(std::mem::size_of::<usize>()))
            .ok_or("MPI prebuilt topology byte overflow")?
    };
    edges.checked_mul(std::mem::size_of::<u32>()+std::mem::size_of::<usize>())
        .and_then(|n|n.checked_add(csr)).ok_or_else(|| "MPI prebuilt topology byte overflow".into())
}
fn mpi_prebuilt_admit(mpi: &MpiWorld, sources: usize, edges: usize) -> Result<()> {
    let bytes = mpi_prebuilt_bytes(sources, edges)?;
    let mut state = mpi.prebuilt.borrow_mut();
    check(state.preparing && state.pending == 0, "MPI prebuilt topology admission order")?;
    let total = state.resident.checked_add(bytes).ok_or("MPI prebuilt topology byte overflow")?;
    check(total <= state.limit, "MPI prebuilt topology cache budget exceeded")?;
    state.pending = bytes; state.resident = total; state.peak = state.peak.max(total);
    Ok(())
}
fn mpi_prebuilt_prepare(mpi: &MpiWorld) -> Result<()> {
    let limit = match std::env::var("B2_MPI_MAX_PREBUILT_TOPOLOGY_BYTES") {
        Ok(value) => value.parse::<usize>()?,
        Err(std::env::VarError::NotPresent) => 128*1024*1024,
        Err(error) => return Err(error.into()),
    };
    let metadata = MPI_PREBUILD_RECIPES.len().checked_mul(std::mem::size_of::<Option<MpiPreparedTopology>>())
        .ok_or("MPI prebuilt topology byte overflow")?;
    check(metadata <= limit, "MPI prebuilt topology cache budget exceeded")?;
    {
        let mut state = mpi.prebuilt.borrow_mut();
        check(state.entries.is_empty() && state.next_fixed == 0, "MPI prebuilt topology initialized twice")?;
        state.entries.try_reserve_exact(MPI_PREBUILD_RECIPES.len())?;
        check(state.entries.capacity() == MPI_PREBUILD_RECIPES.len(), "MPI prebuilt topology metadata allocation")?;
        state.limit = limit; state.metadata = metadata; state.resident = metadata;
        state.peak = metadata; state.preparing = true;
    }
    for recipe in &MPI_PREBUILD_RECIPES {
        check(MPI_POPULATION_OWNERS[recipe.population] < 0, "MPI prebuild requires shared target")?;
        let started = Instant::now();
        let arrays = mpi_fixed_total_uncached(mpi, recipe.sources, recipe.targets,
            recipe.target_start, recipe.population_size, recipe.population, recipe.edges, recipe.seed)?;
        let seconds = started.elapsed().as_secs_f64();
        let bytes = arrays.0.capacity().checked_mul(std::mem::size_of::<usize>())
            .and_then(|n|arrays.1.capacity().checked_mul(std::mem::size_of::<u32>()).and_then(|v|n.checked_add(v)))
            .and_then(|n|arrays.2.capacity().checked_mul(std::mem::size_of::<usize>()).and_then(|v|n.checked_add(v)))
            .ok_or("MPI prebuilt topology byte overflow")?;
        let mut state = mpi.prebuilt.borrow_mut();
        check(bytes <= state.pending, "MPI prebuilt topology exceeded admitted storage")?;
        state.resident -= state.pending-bytes; state.pending = 0;
        state.build_seconds += seconds;
        state.entries.push(Some(MpiPreparedTopology { arrays, seconds, bytes }));
    }
    mpi.prebuilt.borrow_mut().preparing = false;
    Ok(())
}
fn mpi_fixed_total(mpi: &MpiWorld, sources: usize, targets: usize,
    target_start: usize, population_size: usize, population: usize, edges: usize,
    seed: u64) -> Result<MpiTopology> {
    let mut state = mpi.prebuilt.borrow_mut();
    check(!state.preparing && state.next_fixed < MPI_PREBUILD_FIXED_COUNT,
        "MPI prebuilt topology consumption order")?;
    let ordinal = state.next_fixed; state.next_fixed += 1; state.last_seconds = 0.0;
    if MPI_POPULATION_OWNERS[population] >= 0 {
        drop(state);
        return mpi_fixed_total_uncached(mpi,sources,targets,target_start,population_size,population,edges,seed);
    }
    let index = state.next_shared;
    let expected = MPI_PREBUILD_RECIPES.get(index).ok_or("MPI prebuilt topology omitted recipe")?;
    let actual = MpiPrebuildRecipe { ordinal, projection: expected.projection,
        sources, targets, target_start, population_size, population, edges, seed };
    check(*expected == actual, "MPI prebuilt topology recipe/order mismatch")?;
    let entry = state.entries.get_mut(index).and_then(Option::take)
        .ok_or("MPI prebuilt topology already consumed")?;
    state.next_shared += 1; state.resident -= entry.bytes; state.last_seconds = entry.seconds;
    Ok(entry.arrays)
}
fn mpi_prebuilt_seconds(mpi: &MpiWorld) -> f64 {
    std::mem::replace(&mut mpi.prebuilt.borrow_mut().last_seconds, 0.0)
}
fn mpi_prebuilt_finish(mpi: &MpiWorld) -> Result<()> {
    let mut state = mpi.prebuilt.borrow_mut();
    check(state.next_fixed == MPI_PREBUILD_FIXED_COUNT && state.next_shared == MPI_PREBUILD_RECIPES.len()
        && state.entries.iter().all(Option::is_none) && state.resident == state.metadata
        && state.pending == 0 && state.last_seconds == 0.0, "MPI prebuilt topology not fully consumed")?;
    state.entries = Vec::new(); state.resident = 0;
    Ok(())
}
fn mpi_prebuilt_records(mpi: &MpiWorld) -> Result<Vec<u64>> {
    let s = mpi.prebuilt.borrow();
    mpi.gather(&[s.next_shared as u64,s.limit as u64,s.metadata as u64,s.peak as u64,
        s.resident as u64,s.next_fixed as u64,s.build_seconds.to_bits(),s.verify_seconds.to_bits()])
}
fn mpi_prebuilt_json(records: &[u64]) -> String {
    format!(r#","prebuilt_topology":{{"columns":["shared_projections","cache_limit_bytes","metadata_bytes","peak_retained_bytes","remaining_bytes","fixed_projections","build_seconds_bits","verify_seconds_bits"],"rank_records":{:?}}}"#,records)
}
