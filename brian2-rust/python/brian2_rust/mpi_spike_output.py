"""Opt-in result v4: checked u32 spike pairs on disk, unchanged logical data."""
from .mpi_spike_history import _once

OUTPUT_HELPER = r'''
fn dump_spikes_compact<W:Write>(w:&mut W,values:&[(u32,u32)])->Result<()> {
    let mut bytes=[0u8;8192];
    for chunk in values.chunks(1024) {
        for (slot,&(tick,index)) in bytes.chunks_exact_mut(8).zip(chunk) {
            slot[..4].copy_from_slice(&tick.to_le_bytes());
            slot[4..].copy_from_slice(&index.to_le_bytes());
        }
        w.write_all(&bytes[..chunk.len()*8])?;
    }
    Ok(())
}
'''


def compact_spike_output_source(model, source):
    """Apply after history32/projection folding, before population aggregation."""
    populations = model['definition']['populations']
    for p, pop in enumerate(populations):
        if pop.get('events', []) not in ([], ['spike']) or pop.get('event_monitors'):
            raise ValueError('compact spike output requires ordinary events without EventMonitor')
        clock = model['run']['clocks'][pop['clock']]
        if not 0 <= clock['start_tick'] < 2**32 or clock['start_tick'] + clock['steps'] > 2**32 or not 0 <= pop['count'] <= 2**32:
            raise ValueError('compact spike output exceeds u32 domain')
        if f'let mut p{p}_spikes: Vec<(u32,u32)> = Vec::new();' not in source:
            raise ValueError('compact spike output requires checked history32 source')
        names = [f'p{p}_spikes']
        if pop.get('events'):
            names.append(f'p{p}_spikes' if pop.get('spike_monitor') is not None else f'p{p}_event_history_0')
        for name in set(names):
            old = name + '.len()*16'
            if source.count(old) != names.count(name):
                raise ValueError('compact spike output size expression differs')
            source = source.replace(old, name + '.len()*8')
        source = _once(source, f'dump_spikes_u32(&mut dump, &p{p}_spikes)?;',
                       f'dump_spikes_compact(&mut dump, &p{p}_spikes)?;')
        if pop.get('events'):
            source = _once(source, f'dump_spikes_u32(&mut events_dump, &{names[-1]})?;',
                           f'dump_spikes_compact(&mut events_dump, &{names[-1]})?;')
    source = _once(source, 'w.write_all(&3u32.to_le_bytes())?;', 'w.write_all(&4u32.to_le_bytes())?;')
    source = _once(source, 'b2-result-dump-v3', 'b2-result-dump-v4')
    if any(pop.get('events') for pop in populations):
        source = _once(source, 'B2EVT001', 'B2EVT002')
    source += OUTPUT_HELPER
    return source, dict(compacted=True, result_version=4, event_magic='B2EVT002',
                        index_bits=32, bytes_per_record=8,
                        logical_records='unchanged', overflow_policy='reject-before-narrowing')
