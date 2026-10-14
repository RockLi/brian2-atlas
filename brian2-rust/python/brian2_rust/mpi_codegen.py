"""MPI hooks for canonical CPU AOT emission. The reference executor is untouched."""
import json
from pathlib import Path
from .codegen_sums import usize_sum


MPI_RUNTIME = r'''
extern "C" {
    fn b2mpi_init(rank: *mut i32, size: *mut i32) -> i32;
    fn b2mpi_local_rank() -> i32;
    fn b2mpi_finalize() -> i32;
    fn b2mpi_abort();
    fn b2mpi_barrier() -> i32;
    fn b2mpi_peak_rss_bytes() -> u64;
    fn b2mpi_spikes(send: *const u64, count: i32, recv: *mut u64, capacity: i32, received: *mut i32) -> i32;
    fn b2mpi_collect(send: *const u64, count: i32, recv: *mut u64, total: i32, owner: i32) -> i32;
    fn b2mpi_collect_indexed(ids: *const u64, send: *const u64, count: i32, recv: *mut u64, total: i32) -> i32;
    fn b2mpi_merge(bits: *mut u64, count: i32) -> i32;
    fn b2mpi_gather(send: *const u64, count: i32, recv: *mut u64) -> i32;
    fn b2mpi_host(bytes: *mut u64, capacity: i32) -> i32;
}
struct MpiWorld {
    rank: usize,
    size: usize,
    exchanges: std::cell::Cell<usize>,
    emitted: std::cell::Cell<usize>,
    communication_seconds: std::cell::Cell<f64>,
    buffers: std::cell::RefCell<(Vec<u64>, Vec<u64>)>,
    constructed_edges: std::cell::Cell<usize>,
}
fn mpi_check(code: i32) -> Result<()> { check(code == 0, "MPI operation failed") }
impl MpiWorld {
    fn init() -> Result<Self> {
        let (mut rank, mut size) = (0, 0);
        mpi_check(unsafe { b2mpi_init(&mut rank, &mut size) })?;
        std::panic::set_hook(Box::new(|info| {
            eprintln!("b2-mpi panic: {info}");
            unsafe { b2mpi_abort() };
        }));
        check(size > 0 && rank >= 0 && rank < size, "invalid MPI communicator")?;
        Ok(Self { rank: rank as usize, size: size as usize,
            exchanges: 0.into(), emitted: 0.into(), communication_seconds: 0.0.into(),
            buffers: std::cell::RefCell::new((Vec::new(), Vec::new())), constructed_edges: 0.into() })
    }
    fn range(&self, count: usize, population: usize) -> (usize, usize) {
        let owner = MPI_POPULATION_OWNERS[population];
        if owner < 0 { (count*self.rank/self.size, count*(self.rank+1)/self.size) }
        else if owner as usize == self.rank { (0, count) } else { (0, 0) }
    }
    fn owns(&self, index: usize, count: usize, population: usize) -> bool {
        let (start, stop) = self.range(count, population);
        index >= start && index < stop
    }
    fn exchange_spike_batch(&self, batch: &mut [(&mut Vec<usize>, usize, usize)]) -> Result<()> {
        let started = Instant::now();
        let capacity = batch.iter().try_fold(0usize, |sum, (_, count, _)|
            sum.checked_add(*count).ok_or("MPI spike batch capacity overflow"))?;
        let capacity_i32 = i32::try_from(capacity)?;
        let mut buffers = self.buffers.borrow_mut();
        let (send, receive) = &mut *buffers;
        send.clear();
        let mut offset = 0usize;
        for (fired, count, population) in batch.iter() {
            check(fired.iter().all(|&i| i < *count && self.owns(i, *count, *population)), "non-owned spike")?;
            send.extend(fired.iter().map(|&i| (offset+i) as u64));
            offset += count;
        }
        self.emitted.set(self.emitted.get() + send.len());
        self.exchanges.set(self.exchanges.get() + 1);
        if self.size == 1 { return Ok(()); }
        if receive.len() < capacity { receive.resize(capacity, 0); }
        let mut received = 0i32;
        mpi_check(unsafe { b2mpi_spikes(send.as_ptr(), i32::try_from(send.len())?,
            receive.as_mut_ptr(), capacity_i32, &mut received) })?;
        check(received >= 0 && received as usize <= capacity, "invalid MPI spike count")?;
        let receive = &mut receive[..received as usize];
        receive.sort_unstable();
        check(receive.iter().all(|&i| i < capacity as u64)
            && receive.windows(2).all(|pair| pair[0] < pair[1]), "duplicate/invalid MPI spike")?;
        let (mut offset, mut cursor) = (0usize, 0usize);
        for (fired, count, _) in batch.iter_mut() {
            fired.clear();
            let end = offset + *count;
            while cursor < receive.len() && receive[cursor] < end as u64 {
                fired.push(receive[cursor] as usize-offset);
                cursor += 1;
            }
            offset = end;
        }
        check(cursor == receive.len(), "unconsumed MPI spike batch")?;
        self.communication_seconds.set(self.communication_seconds.get() + started.elapsed().as_secs_f64());
        Ok(())
    }
    fn collect_bits(&self, local: &[u64], total: usize, population: usize) -> Result<Vec<u64>> {
        let mut result = vec![0; if self.rank == 0 { total } else { 0 }];
        mpi_check(unsafe { b2mpi_collect(local.as_ptr(), i32::try_from(local.len())?,
            result.as_mut_ptr(), i32::try_from(total)?, MPI_POPULATION_OWNERS[population]) })?;
        Ok(result)
    }
    fn collect_f64(&self, values: Vec<f64>, total: usize, population: usize) -> Result<Vec<f64>> {
        let bits: Vec<u64> = values.into_iter().map(f64::to_bits).collect();
        Ok(self.collect_bits(&bits, total, population)?.into_iter().map(f64::from_bits).collect())
    }
    fn collect_flags(&self, values: Vec<u8>, total: usize, population: usize) -> Result<Vec<u8>> {
        let bits: Vec<u64> = values.into_iter().map(u64::from).collect();
        self.collect_bits(&bits, total, population)?.into_iter().map(|x| u8::try_from(x).map_err(Into::into)).collect()
    }
    fn collect_f32(&self, values: Vec<f32>, total: usize, population: usize) -> Result<Vec<f32>> {
        let bits: Vec<u64> = values.into_iter().map(|v| u64::from(v.to_bits())).collect();
        self.collect_bits(&bits, total, population)?.into_iter().map(|v| Ok(f32::from_bits(u32::try_from(v)?))).collect()
    }
    fn collect_i32(&self, values: Vec<i32>, total: usize, population: usize) -> Result<Vec<i32>> {
        let bits: Vec<u64> = values.into_iter().map(|v| u64::from(v as u32)).collect();
        self.collect_bits(&bits, total, population)?.into_iter().map(|v| Ok(u32::try_from(v)? as i32)).collect()
    }
    fn collect_i64(&self, values: Vec<i64>, total: usize, population: usize) -> Result<Vec<i64>> {
        let bits: Vec<u64> = values.into_iter().map(|v| v as u64).collect();
        Ok(self.collect_bits(&bits, total, population)?.into_iter().map(|v| v as i64).collect())
    }
    fn collect_u32(&self, values: Vec<u32>, total: usize, population: usize) -> Result<Vec<u32>> {
        let bits: Vec<u64> = values.into_iter().map(u64::from).collect();
        self.collect_bits(&bits, total, population)?.into_iter().map(|v| u32::try_from(v).map_err(Into::into)).collect()
    }
    fn collect_u64(&self, values: Vec<u64>, total: usize, population: usize) -> Result<Vec<u64>> {
        self.collect_bits(&values, total, population)
    }
    fn collect_indexed_bits(&self, ids: &[usize], bits: Vec<u64>, total: usize) -> Result<Vec<u64>> {
        check(ids.len() == bits.len(), "indexed MPI result length mismatch")?;
        let ids: Vec<u64> = ids.iter().copied().map(u64::try_from).collect::<std::result::Result<_,_>>()?;
        let mut result = vec![0; if self.rank == 0 { total } else { 0 }];
        mpi_check(unsafe { b2mpi_collect_indexed(ids.as_ptr(), bits.as_ptr(),
            i32::try_from(bits.len())?, result.as_mut_ptr(), i32::try_from(total)?) })?;
        Ok(result)
    }
    fn merge_f64(&self, values: &mut [f64], owner: impl Fn(usize) -> bool) -> Result<()> {
        let mut bits: Vec<u64> = values.iter().enumerate().map(|(i,v)| if owner(i) { v.to_bits() } else { 0 }).collect();
        mpi_check(unsafe { b2mpi_merge(bits.as_mut_ptr(), i32::try_from(bits.len())?) })?;
        for (v, b) in values.iter_mut().zip(bits) { *v = f64::from_bits(b); }
        Ok(())
    }
    fn merge_f32(&self, values: &mut [f32], owner: impl Fn(usize) -> bool) -> Result<()> {
        let mut bits: Vec<u64> = values.iter().enumerate().map(|(i,v)| if owner(i) { u64::from(v.to_bits()) } else { 0 }).collect();
        mpi_check(unsafe { b2mpi_merge(bits.as_mut_ptr(), i32::try_from(bits.len())?) })?;
        for (v, b) in values.iter_mut().zip(bits) { *v = f32::from_bits(u32::try_from(b)?); } Ok(())
    }
    fn merge_i32(&self, values: &mut [i32], owner: impl Fn(usize) -> bool) -> Result<()> {
        let mut bits: Vec<u64> = values.iter().enumerate().map(|(i,v)| if owner(i) { u64::from(*v as u32) } else { 0 }).collect();
        mpi_check(unsafe { b2mpi_merge(bits.as_mut_ptr(), i32::try_from(bits.len())?) })?;
        for (v, b) in values.iter_mut().zip(bits) { *v = u32::try_from(b)? as i32; } Ok(())
    }
    fn merge_i64(&self, values: &mut [i64], owner: impl Fn(usize) -> bool) -> Result<()> {
        let mut bits: Vec<u64> = values.iter().enumerate().map(|(i,v)| if owner(i) { *v as u64 } else { 0 }).collect();
        mpi_check(unsafe { b2mpi_merge(bits.as_mut_ptr(), i32::try_from(bits.len())?) })?;
        for (v, b) in values.iter_mut().zip(bits) { *v = b as i64; } Ok(())
    }
    fn merge_u32(&self, values: &mut [u32], owner: impl Fn(usize) -> bool) -> Result<()> {
        let mut bits: Vec<u64> = values.iter().enumerate().map(|(i,v)| if owner(i) { u64::from(*v) } else { 0 }).collect();
        mpi_check(unsafe { b2mpi_merge(bits.as_mut_ptr(), i32::try_from(bits.len())?) })?;
        for (v, b) in values.iter_mut().zip(bits) { *v = u32::try_from(b)?; } Ok(())
    }
    fn merge_u64(&self, values: &mut [u64], owner: impl Fn(usize) -> bool) -> Result<()> {
        let mut bits: Vec<u64> = values.iter().enumerate().map(|(i,v)| if owner(i) { *v } else { 0 }).collect();
        mpi_check(unsafe { b2mpi_merge(bits.as_mut_ptr(), i32::try_from(bits.len())?) })?;
        values.copy_from_slice(&bits); Ok(())
    }
    fn merge_flags(&self, values: &mut [u8], owner: impl Fn(usize) -> bool) -> Result<()> {
        let mut bits: Vec<u64> = values.iter().enumerate().map(|(i,v)| if owner(i) { u64::from(*v) } else { 0 }).collect();
        mpi_check(unsafe { b2mpi_merge(bits.as_mut_ptr(), i32::try_from(bits.len())?) })?;
        for (v, b) in values.iter_mut().zip(bits) { *v = u8::try_from(b)?; } Ok(())
    }
    fn gather(&self, values: &[u64]) -> Result<Vec<u64>> {
        let mut result = vec![0; values.len().checked_mul(self.size).ok_or("MPI gather size overflow")?];
        mpi_check(unsafe { b2mpi_gather(values.as_ptr(), i32::try_from(values.len())?, result.as_mut_ptr()) })?;
        Ok(result)
    }
    fn sum(&self, value: usize) -> Result<usize> {
        self.gather(&[u64::try_from(value)?])?.into_iter().try_fold(0usize, |sum, item| {
            sum.checked_add(usize::try_from(item)?).ok_or_else(|| "MPI count overflow".into())
        })
    }
    fn hosts(&self) -> Result<Vec<String>> {
        let mut local = vec![0u64; 1024];
        mpi_check(unsafe { b2mpi_host(local.as_mut_ptr(), i32::try_from(local.len())?) })?;
        let result = self.gather(&local)?;
        Ok(result.chunks(local.len()).map(|host| host.iter().take_while(|&&v| v != 0)
            .map(|&v| { let c = char::from(v as u8); if c.is_ascii_alphanumeric() || "-._".contains(c) { c } else { '_' } }).collect()).collect())
    }
}
fn main() {
    let mpi = match MpiWorld::init() {
        Ok(world) => world,
        Err(error) => { eprintln!("b2-mpi init: {error}"); unsafe { b2mpi_abort() }; std::process::exit(1); }
    };
    if let Err(error) = run_mpi(&mpi) {
        eprintln!("b2-mpi rank {}: {error}", mpi.rank);
        unsafe { b2mpi_abort() };
        std::process::exit(1);
    }
    if unsafe { b2mpi_finalize() } != 0 { std::process::exit(1); }
}
fn run_mpi(mpi: &MpiWorld) -> Result<()> {
    let initialization_started = Instant::now();
    check(mpi.size == MPI_EXPECTED_RANKS, "MPI world size differs from compiled plan")?;
    let identities = mpi.gather(&MPI_PLAN_WORDS)?;
    check(identities.chunks(4).all(|words| words == MPI_PLAN_WORDS), "MPI ranks have different compiled plans")?;
    let args: Vec<_> = std::env::args_os().collect();
    check(args.len() == 3, "usage: b2-mpi INSTANCE.bin OUTPUT_DIRECTORY")?;
    let output = Path::new(&args[2]);
    check(!output.exists(), "output exists")?;
    let index = Path::new(&args[1]);
    check(fs::read(index)?.as_slice() == include_bytes!("instance.bin"), "MPI instance differs from compiled plan; rebuild required")?;
    let shard = index.parent().ok_or("missing MPI instance directory")?.join(format!("instance.rank-{}.bin", mpi.rank));
    let reader = Reader::open_bound(&shard, mpi_instance_identity(mpi.rank))?;
    mpi_check(unsafe { b2mpi_barrier() })?;
    execute(reader, output, initialization_started, mpi)
}
'''


def runtime_source(runtime, plan, *, streamed=False):
    # The standard runtime ends in its CLI. Preserve all arithmetic/IO helpers.
    marker = '\nfn main(){'
    if runtime.count(marker) != 1:
        raise ValueError("CPU runtime main boundary changed")
    words = [int(plan.sha256[i:i+16], 16) for i in range(0, 64, 16)]
    base = runtime[:runtime.index(marker)]
    # The ordinary CPU runtime picks the first allowed core. Under MPI each
    # node-local rank must choose a different core from the same allowed set.
    base = base.replace("affinity_cpus.truncate(threads);",
        "let lane=unsafe{b2mpi_local_rank()} as usize; let offset=lane%affinity_cpus.len(); affinity_cpus.rotate_left(offset); affinity_cpus.truncate(threads);")
    base = base.replace("struct Reader { bytes: Vec<u8>, at: usize }",
        "struct Reader { input: std::io::BufReader<File>, expected: [u8;32], length: usize, hash: Sha256, at: usize }")
    reader_start = base.index(" fn open(path: &Path)")
    reader_end = base.index(" fn u64(&mut self)", reader_start)
    base = base[:reader_start] + r''' fn open_bound(path: &Path, identity: (usize, [u8;32])) -> Result<Self> {
        let (length, expected) = identity;
        let file = File::open(path)?;
        check(file.metadata()?.len() == length as u64, "MPI instance differs from compiled plan; rebuild required")?;
        let mut reader = Self { input: std::io::BufReader::with_capacity(65536, file), expected, length, hash: Sha256::new(), at: 0 };
        check(&reader.take::<8>()? == b"B2AOT001", "invalid MPI shard")?;
        Ok(reader)
    }
    fn take<const S:usize>(&mut self)->Result<[u8;S]> {
        let end = self.at.checked_add(S).ok_or("instance overflow")?;
        let mut value = [0u8; S]; self.input.read_exact(&mut value)?;
        check(end <= self.length, "truncated MPI instance")?;
        self.hash.update(&value);
        self.at=end; Ok(value)
    }
''' + base[reader_end:]
    base = base.replace('check(self.at==self.bytes.len(),"trailing instance data")',
        'check(self.at==self.length && self.hash.finish()==self.expected,"MPI instance differs from compiled plan; rebuild required")')
    digest = (Path(__file__).with_name("mpi_runtime") / "sha256.rs").read_text()
    topology = (Path(__file__).with_name("mpi_runtime") / "topology.rs").read_text()
    additive = (Path(__file__).with_name("mpi_runtime") / "additive.rs").read_text()
    # The closure borrows the same arrays and scalars as the inline node.
    # No task/thread is created; the call completes before the next plan node.
    population_step = '\n#[inline(never)]\nfn mpi_population_step<F: FnOnce()>(step: F) { step(); }\n'
    gpu = ""
    if plan.rank_backends:
        from .mpi_gpu import rust_runtime
        gpu = rust_runtime(plan.rank_backends)
    return (base + gpu + digest + topology + additive + population_step + MPI_RUNTIME
            + '\ninclude!("instance-identities.rs");\n'
            + f"\nconst MPI_EXPECTED_RANKS: usize = {plan.ranks};\n"
            + f"const MPI_POPULATION_OWNERS: [i32; {len(plan.population_owners)}] = {[o if o is not None else -1 for o in plan.population_owners]!r};\n"
            + f"const MPI_PLAN_WORDS: [u64; 4] = {words!r};\n")



def initialization_timing():
    """Keep late initialization out of the first spike exchange's clock."""
    return [
        "    let mpi_local_initialization_seconds = initialization_started.elapsed().as_secs_f64();",
        "    let mpi_initialization_sync_started = Instant::now();",
        "    mpi_check(unsafe { b2mpi_barrier() })?;",
        "    let mpi_initialization_wait_seconds = mpi_initialization_sync_started.elapsed().as_secs_f64();",
    ]


def simulation_timing():
    """Finish simulation on all ranks before starting result collection."""
    return [
        "    let mpi_local_simulation_seconds = started.elapsed().as_secs_f64();",
        "    let mpi_simulation_sync_started = Instant::now();",
        "    mpi_check(unsafe { b2mpi_barrier() })?;",
        "    let mpi_simulation_wait_seconds = mpi_simulation_sync_started.elapsed().as_secs_f64();",
        "    let mpi_collection_started = Instant::now();",
    ]


def collect_results(model, plan):
    """Collect owner values once after simulation, then let only rank zero dump."""
    lines = []
    for p, pop in enumerate(model["definition"]["populations"]):
        n = pop["count"]
        methods = {"bool": "flags", "f32": "f32", "f64": "f64", "i32": "i32",
                   "i64": "i64", "u32": "u32", "u64": "u64"}
        for i, state in enumerate(pop["states"]):
            if (p, state["name"]) not in plan.readonly_pre_states:
                lines.append(f"    let p{p}_state_{i} = mpi.collect_{methods[state['dtype']]}(p{p}_state_{i}, {n}, {p})?;")
        if pop["refractory"] is not None:
            lines += [f"    let p{p}_lastspike = mpi.collect_f64(p{p}_lastspike, {n}, {p})?;",
                      f"    let p{p}_not_refractory = mpi.collect_flags(p{p}_not_refractory, {n}, {p})?;"]
        record = pop["monitor"]["record"]
        if record:
            lines.append(f"    let p{p}_record: [usize; {len(record)}] = {record!r};")
            public = {symbol["name"]: symbol for symbol in
                      pop["states"] + pop["parameters"] + pop.get("linked_variables", [])}
            for column, name in enumerate(pop["monitor"]["variables"]):
                lines.append(f"    mpi.merge_{methods[public[name]['dtype']]}(&mut p{p}_samples_{column}, |i| mpi.owns(p{p}_record[i % {len(record)}], {n}, {p}))?;")
    for q, (synapse, values) in enumerate(zip(model["definition"]["synapses"],
                                               model["instance"]["synapses"], strict=True)):
        if synapse["states"]:
            total = values["topology"].get("edge_count", len(values.get("source", ())))
            for i, state in enumerate(synapse["states"]):
                dtype = state['dtype']
                to_bits = {'f64': 'v.to_bits()', 'f32': 'v.to_bits() as u64',
                           'i32': 'v as u32 as u64'}.get(dtype, 'v as u64')
                from_bits = {'f64': 'f64::from_bits(v)', 'f32': 'f32::from_bits(v as u32)',
                             'bool': 'v as u8'}.get(dtype, f'v as {dtype}')
                lines.append(f"    let s{q}_state_{i} = mpi.collect_indexed_bits(&s{q}_original_edges, s{q}_state_{i}.into_iter().map(|v| {to_bits}).collect(), {total})?.into_iter().map(|v| {from_bits}).collect::<Vec<_>>();")
    delivered = usize_sum(f"s{q}_delivered + s{q}_post_delivered" if any(
        path['kind'] == 'post' for path in model['instance']['synapses'][q]['pathways']) else f's{q}_delivered'
        for q in range(len(model["definition"]["synapses"]))) or "0usize"
    owned = [sum(end-start for start, end in shard.population_ranges) for shard in plan.shards]
    lines += [f"    let owned: [u64; {plan.ranks}] = {owned!r};",
              f"    let rank_work = mpi.gather(&[owned[mpi.rank], mpi.emitted.get() as u64, ({delivered}) as u64])?;",
              "    let hosts = mpi.hosts()?;",
              "    let rank_cpu_ids: Vec<i64> = mpi.gather(&[parallel.affinity_cpus.first().copied().map(|v|v as u64).unwrap_or(u64::MAX)])?.into_iter().map(|v|v as i64).collect();",
              "    let rank_exchange_seconds: Vec<f64> = mpi.gather(&[mpi.communication_seconds.get().to_bits()])?.into_iter().map(f64::from_bits).collect();"]
    # Queue slots retain capacity after delivery. Report the actual retained
    # storage once, without scanning queues in the simulation's hot loop.
    queues = [f'&plan_s{q}p{r}_queue' for q, syn in enumerate(model["instance"]["synapses"])
              for r in range(len(syn["pathways"]))]
    lines += [f"    let queues: &[&[Vec<usize>]] = &[{', '.join(queues)}];",
              '    let queue_capacity_items = queues.iter().flat_map(|queue| queue.iter()).try_fold(0usize, |total, slot| total.checked_add(slot.capacity()).ok_or("MPI queue capacity overflow"))?;',
              '    let queue_capacity_bytes = queue_capacity_items.checked_mul(std::mem::size_of::<usize>()).ok_or("MPI queue byte count overflow")?;',
              '    let rank_queue_capacity_bytes = mpi.gather(&[u64::try_from(queue_capacity_bytes)?])?;']
    for q in range(len(model["definition"]["synapses"])):
        lines.append(f"    s{q}_delivered = mpi.sum(s{q}_delivered)?;")
        if any(path['kind'] == 'post' for path in model['instance']['synapses'][q]['pathways']):
            lines.append(f"    s{q}_post_delivered = mpi.sum(s{q}_post_delivered)?;")
    lines.append('    let mut topology_report = String::from("[");')
    for position, q in enumerate(plan.procedural_projections):
        edge_count = model["instance"]["synapses"][q]["topology"]["edge_count"]
        owner = plan.population_owners[model["definition"]["synapses"][q]["target_population"]]
        draws = (f'{edge_count}usize*(mpi.rank+1)/mpi.size-{edge_count}usize*mpi.rank/mpi.size'
                 if owner is None else f'if mpi.rank == {owner} {{ {edge_count}usize }} else {{ 0usize }}')
        header = ("," if position else "") + json.dumps({"projection": q, "global_edges": edge_count,
            "construction": "distributed-draw-ranges" if owner is None else "target-owner-local"})[:-1] + ',"rank_stats":'
        # Each report is appended immediately; its gathered vectors need not
        # remain live across every later projection and result-writing error.
        lines += ['    {', f'    let topology_stats = mpi.gather(&[s{q}_local_edge_count as u64, ({draws}) as u64])?;',
                  f'    let topology_seconds: Vec<f64> = mpi.gather(&[s{q}_build_seconds.to_bits()])?.into_iter().map(f64::from_bits).collect();',
                  f'    let topology_csr_bytes = mpi.gather(&[(s{q}_offsets.capacity()*std::mem::size_of::<usize>()) as u64])?;',
                  f'    topology_report.push_str({json.dumps(header)});',
                  '    topology_report.push_str(&format!("{:?},\\\"build_seconds\\\":{:?},\\\"rank_csr_offset_bytes\\\":{:?}}}", topology_stats, topology_seconds, topology_csr_bytes));', '    }']
    lines.append('    topology_report.push_str("]");')
    lines += [
        "    let mpi_local_collection_seconds = mpi_collection_started.elapsed().as_secs_f64();",
        "    let rank_peak_rss_bytes = mpi.gather(&[unsafe { b2mpi_peak_rss_bytes() }])?;",
        "    let rank_stage_seconds: Vec<f64> = mpi.gather(&[mpi_local_initialization_seconds.to_bits(), mpi_initialization_wait_seconds.to_bits(), mpi_local_simulation_seconds.to_bits(), mpi_simulation_wait_seconds.to_bits(), mpi_local_collection_seconds.to_bits()])?.into_iter().map(f64::from_bits).collect();",
    ]
    if plan.rank_backends:
        lines.append("    let rank_gpu_dispatches = mpi.gather(&[mpi_gpu.dispatches])?;")
    lines += ["    if mpi.rank != 0 { return Ok(()); }",
              "    fs::create_dir_all(output)?;"]
    base = {"schema": "b2-mpi-runtime-v0", "plan_sha256": plan.sha256,
            "ranks": plan.ranks, "transport": plan.transport, "storage": plan.storage,
            "numeric_profile": plan.numeric_profile,
            "stage_timing_schema": "b2-mpi-stage-timing-v1",
            "stage_timing_scope": "Local elapsed clocks with barriers after initialization and simulation; collection includes result gathering and topology reporting, excludes final timing gather, JSON serialization and disk output",
            "rank_peak_rss_scope": "Process lifetime through distributed result collection; sampled before rank-zero final result serialization and disk output",
            "rank_stage_columns": ["initialization_local", "initialization_wait",
                                   "simulation_local", "simulation_wait",
                                   "result_collection_and_reporting"],
            "spike_exchange_strategy": "consecutive-producers-same-clock",
            "rank_work_columns": ["owned_neurons", "emitted_spikes", "delivered_edges"]}
    if plan.rank_backends:
        base["rank_backends"] = plan.rank_backends
        base["gpu_scope"] = "population-state-update; host threshold/reset/synapses/MPI"
    # Write the static JSON prefix and runtime fields using debug arrays of
    # finite numbers/sanitized hostnames; these are also valid JSON arrays.
    prefix = json.dumps(base)[:-1] + ',"rank_work":'
    lines += [f"    let mut mpi_report = String::from({json.dumps(prefix)});",
              '    mpi_report.push_str(&format!("{:?},\\\"processor_names\\\":{:?},\\\"spike_exchange_seconds\\\":{:?},\\\"exchange_calls_per_rank\\\":{}}}\\n", rank_work, hosts, rank_exchange_seconds, mpi.exchanges.get()));',
              '    mpi_report = mpi_report.trim_end().trim_end_matches("}").to_owned() + &format!(",\\\"rank_cpu_ids\\\":{:?}}}\\n", rank_cpu_ids);',
              '    mpi_report = mpi_report.trim_end().trim_end_matches("}").to_owned() + &format!(",\\\"procedural_topology\\\":{}}}\\n", topology_report);' ,
              '    mpi_report = mpi_report.trim_end().trim_end_matches("}").to_owned() + &format!(",\\\"rank_queue_capacity_bytes\\\":{:?}}}\\n", rank_queue_capacity_bytes);',
              '    fs::write(output.join("mpi-runtime.json"), &mpi_report)?;']
    lines.insert(-1, '    mpi_report = mpi_report.trim_end().trim_end_matches("}").to_owned() + &format!(",\\\"rank_peak_rss_bytes\\\":{:?}}}\\n", rank_peak_rss_bytes);')
    lines.insert(-1, '    mpi_report = mpi_report.trim_end().trim_end_matches("}").to_owned() + &format!(",\\\"rank_stage_seconds\\\":{:?}}}\\n", rank_stage_seconds);')
    if plan.rank_backends:
        lines.insert(-1, '    mpi_report = mpi_report.trim_end().trim_end_matches("}").to_owned() + &format!(",\\\"rank_gpu_dispatches\\\":{:?}}}\\n", rank_gpu_dispatches);')
    return lines
