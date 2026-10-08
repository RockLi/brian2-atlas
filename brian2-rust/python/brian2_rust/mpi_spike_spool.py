"""Opt-in bounded disk history; public compact outputs retain exact bytes."""
from pathlib import Path
from .mpi_spike_history import _once

SPOOL_HELPER = (Path(__file__).with_name('mpi_runtime')/'spike_spool.rs').read_text()


def spool_spike_history_source(model, source, maximum_bytes, maximum_population_bytes=None):
    if type(maximum_bytes) is not int or not 1 <= maximum_bytes <= 512*2**30:
        raise ValueError('spike_spool_bytes must be an integer in 1..512 GiB')
    if maximum_population_bytes is None:
        maximum_population_bytes = maximum_bytes
    if type(maximum_population_bytes) is not int or not 1 <= maximum_population_bytes <= maximum_bytes:
        raise ValueError('spike_spool_population_bytes must be an integer in 1..spike_spool_bytes')
    populations = model['definition']['populations']
    if len(populations)>4096:
        raise ValueError('MPI spool supports at most 4096 populations')
    gate = '    mpi_check(unsafe { b2mpi_barrier() })?;\n    execute(reader, output, initialization_started, mpi)'
    source = _once(source, gate, gate)
    # execute follows the collective input gate, so rank zero may now create
    # its output directory without racing other ranks' output-exists checks.
    anchor = '    let (p0_start, p0_stop) = mpi.range('
    if source.count(anchor)!=1:
        raise ValueError('MPI spool population initialization source differs')
    setup = f'''    let mpi_spool_budget = MpiSpoolBudget::with_population_limit({maximum_bytes}u64,{maximum_population_bytes}u64)?;
    let mpi_spool_directory = output.join("spike-spool");
    if mpi.rank == 0 {{ fs::create_dir_all(output)?; fs::create_dir(&mpi_spool_directory)?; }}
'''
    source = source.replace(anchor,setup+anchor,1)
    names = []
    released = set()
    for p,pop in enumerate(populations):
        if pop.get('events',[]) not in ([],['spike']) or pop.get('event_monitors'):
            raise ValueError('MPI spool requires ordinary spikes without EventMonitor')
        current = [f'p{p}_spikes']
        if pop.get('events') and pop.get('spike_monitor') is None:
            current.append(f'p{p}_event_history_0')
        for name in current:
            source = _once(source,f'let mut {name}: Vec<(u32,u32)> = Vec::new();',
                f'let mut {name} = MpiSpikeSpool::new(mpi_spool_directory.join("{name}.bin"), mpi_spool_budget.clone());')
            names.append(name)
        if pop.get('spike_monitor') is not None:
            source = _once(source,f'p{p}_spikes.push(mpi_spike_record(p{p}_tick,i));',
                f'p{p}_spikes.push(mpi_spike_record(p{p}_tick,i))?;')
        if len(current)==2:
            old=f'p{p}_event_history_0.extend(p{p}_fired.iter().map(|&i| mpi_spike_record(p{p}_tick, i)));'
            source = _once(source,old,old[:-1]+'?;')
        source = _once(source,f'dump_spikes_compact(&mut dump, &p{p}_spikes)?;',
            f'p{p}_spikes.copy_to(&mut dump)?;')
        if pop.get('events'):
            source = _once(source,f'dump_spikes_compact(&mut events_dump, &{current[-1]})?;',
                f'{current[-1]}.copy_final(&mut events_dump,output)?;')
            released.add(current[-1])
    anchor = '    let dump_write_seconds = output_started.elapsed().as_secs_f64();'
    cleanup=''.join(f'    {n}.remove()?;\n' for n in names if n not in released)+'    fs::remove_dir(&mpi_spool_directory)?;\n'
    source = _once(source,anchor,'    File::open(output)?.sync_all()?;\n'+cleanup+anchor)
    anchor = '    fs::write(output.join("mpi-runtime.json"), &mpi_report)?;'
    report = r'''    mpi_report = mpi_report.trim_end().trim_end_matches("}").to_owned() + &format!(",\"spike_history_storage\":\"bounded-disk-spool\",\"spike_spool_bytes\":{},\"spike_spool_maximum_bytes\":{},\"spike_spool_maximum_population_bytes\":{}}}\n", mpi_spool_budget.used.get(), mpi_spool_budget.maximum, mpi_spool_budget.maximum_population);
'''
    source = _once(source,anchor,report+anchor)
    source = _once(source, ')->Result<BufWriter<File>>{let mut w=BufWriter::with_capacity(65536,File::create(output.join("results.bin"))?);',
        ')->Result<MpiBoundedOutput>{let mut w=MpiBoundedOutput::new(File::create(output.join("results.bin"))?,size);')
    source = _once(source, 'fn dump_finish(mut w:BufWriter<File>,', 'fn dump_finish(mut w:MpiBoundedOutput,')
    if any(pop.get('events') for pop in populations):
        source = _once(source, 'let mut events_dump = BufWriter::with_capacity(65536, File::create(output.join("events.bin"))?);',
            'let mut events_dump = MpiBoundedOutput::new(File::create(output.join("events.bin"))?,event_dump_bytes);')
    return source+SPOOL_HELPER, dict(storage='bounded-disk-spool', maximum_bytes=maximum_bytes,
        maximum_population_bytes=maximum_population_bytes,
        maximum_spike_file_overlap_bytes=2*maximum_bytes+maximum_population_bytes,
        disk_overlap_excludes='non-spike output bytes, filesystem allocation/metadata, inputs and reserve',
        buffer_bytes_per_history=65536, dirty_bytes_per_history=1048576,
        final_output_cache_release_bytes=64*2**20,
        maximum_histories=len(names), output_format='unchanged-v4',
        cleanup='each-final-event-prefix-synced-and-directory-persisted',
        failure='durable-completed-prefixes-and-remaining-spools-no-success-summary',
        platform='Linux64', temporary_disk_bytes='8 per recorded spike shared by both outputs')
