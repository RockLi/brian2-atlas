"""Checked, shared u32 spike history; the two public binary files stay unchanged.

Applied before projection/population aggregation. Only ordinary spike histories
are supported: named events and EventMonitor samples retain their own semantics
and are rejected by this opt-in transform. Output size remains uncompressed.
"""
import re


def _once(source, old, new):
    if source.count(old) != 1:
        raise ValueError('MPI spike history: generated source differs at ' + old[:80])
    return source.replace(old, new, 1)


RECORD_HELPER = r'''
#[inline]
fn mpi_spike_record(tick: usize, neuron: usize) -> (u32,u32) {
    (u32::try_from(tick).expect("MPI spike history tick exceeds u32"),
     u32::try_from(neuron).expect("MPI spike history neuron exceeds u32"))
}
'''


def compact_spike_history_source(model, source):
    populations = model['definition']['populations']
    for pop in populations:
        if pop.get('events', []) not in ([], ['spike']) or pop.get('event_monitors'):
            raise ValueError('MPI spike history requires ordinary spike events without EventMonitor')
        clock = model['run']['clocks'][pop['clock']]
        end = clock['start_tick'] + clock['steps']
        if not 0 <= clock['start_tick'] <= 2**32 - 1 or end > 2**32:
            raise ValueError('MPI spike history tick exceeds u32')
        if not 0 <= pop['count'] <= 2**32:
            raise ValueError('MPI spike history neuron exceeds u32')
    histories = []
    shared = 0
    # Emission drift is an error, never a partially narrowed recorder.
    for p, pop in enumerate(populations):
        source = _once(source,
            f'let mut p{p}_spikes: Vec<(usize,usize)> = Vec::new();',
            f'let mut p{p}_spikes: Vec<(u32,u32)> = Vec::new();')
        if pop.get('spike_monitor') is not None:
            source = _once(source, f'p{p}_spikes.push((p{p}_tick,i));',
                f'p{p}_spikes.push(mpi_spike_record(p{p}_tick,i));')
        if pop.get('events') and pop.get('spike_monitor') is not None:
            # Both histories use the same frozen recording-window predicate.
            window = pop['monitor']['window_steps']
            gate = f'if mpi.rank == 0 && p{p}_tick >= p{p}_end_tick - {window} {{'
            if source.count(gate) != 2:
                raise ValueError('MPI spike history: recording windows differ')
            source = _once(source,
                f'    let mut p{p}_event_history_0: Vec<(usize,usize)> = Vec::new();\n', '')
            pattern = (re.escape(gate) + r'\s*' + re.escape(
                f'p{p}_event_history_0.extend(p{p}_fired.iter().map(|&i| (p{p}_tick, i)));')
                + r'\s*\}')
            source, removed = re.subn(pattern, '', source)
            if removed != 1 or source.count(f'p{p}_event_history_0') != 3:
                # Three remaining references: size, count header, dump call.
                raise ValueError('MPI spike history: event stream layout differs')
            source = source.replace(f'p{p}_event_history_0', f'p{p}_spikes')
            histories.append(f'p{p}_spikes')
            shared += 1
        elif pop.get('events'):
            # No SpikeMonitor: the result spike vector is intentionally empty,
            # while the named EventStream still records spikes. Do not alias it.
            source = _once(source,
                f'let mut p{p}_event_history_0: Vec<(usize,usize)> = Vec::new();',
                f'let mut p{p}_event_history_0: Vec<(u32,u32)> = Vec::new();')
            source = _once(source,
                f'p{p}_event_history_0.extend(p{p}_fired.iter().map(|&i| (p{p}_tick, i)));',
                f'p{p}_event_history_0.extend(p{p}_fired.iter().map(|&i| mpi_spike_record(p{p}_tick, i)));')
            histories.append(f'p{p}_event_history_0')
        old = f'dump_spikes_usize(&mut dump, &p{p}_spikes)?;'
        source = _once(source, old, old.replace('dump_spikes_usize', 'dump_spikes_u32'))
        if pop.get('events'):
            old = f'dump_spikes_usize(&mut events_dump, &{histories[-1]})?;'
            source = _once(source, old, old.replace('dump_spikes_usize', 'dump_spikes_u32'))
    # Rank zero owns history; report its actual retained allocation once after
    # simulation. Public output bytes still use two signed 64-bit words/event.
    capacities = ', '.join(f'{name}.capacity()' for name in histories) or '0usize'
    lengths = ', '.join(f'{name}.len()' for name in histories) or '0usize'
    reporting = f'''
    let spike_history_capacity_bytes = [{capacities}].iter().try_fold(0usize, |total, &n| n.checked_mul(std::mem::size_of::<(u32,u32)>()).and_then(|bytes| total.checked_add(bytes)).ok_or("MPI spike history capacity overflow"))?;
    let spike_history_records = [{lengths}].iter().try_fold(0usize, |total, &n| total.checked_add(n).ok_or("MPI spike history count overflow"))?;
''' + r'''    mpi_report = mpi_report.trim_end().trim_end_matches("}").to_owned() + &format!(",\"spike_history_capacity_bytes\":{},\"spike_history_records\":{},\"spike_history_index_bits\":32,\"spike_history_copies\":1}}\n", spike_history_capacity_bytes, spike_history_records);
'''
    anchor = '    fs::write(output.join("mpi-runtime.json"), &mpi_report)?;'
    source = _once(source, anchor, reporting + anchor)
    source += RECORD_HELPER
    return source, dict(compacted=True, index_bits=32, retained_copies=1,
        populations=len(populations), shared_populations=shared,
        stream_only_populations=len(histories)-shared,
        overflow_policy='reject-before-narrowing',
        output_format='unchanged', event_order='unchanged')
