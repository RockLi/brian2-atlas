"""Opt-in u32 FIFO storage for exclusively canonical additive MPI pathways.

Only queue entries narrow: ticks, offsets, loop indices and f64 arithmetic keep
their existing types and order. Runtime domain checks precede narrowing; a
model requiring wider rank-local edge indices fails instead of truncating.
The default emitter is unchanged. Source drift is an explicit error.
"""
from pathlib import Path
import re

from .mpi_additive import shared_additive


def _once(source, old, new):
    if source.count(old) != 1:
        raise ValueError('MPI queue compaction: generated source differs at ' + old[:80])
    return source.replace(old, new, 1)


def compact_additive_kernel(source):
    """Transform the shared kernel, separately testable against its wide version."""
    source = _once(source, 'queue: &mut [Vec<usize>]', 'queue: &mut [Vec<u32>]')
    source = _once(source, '    if targets.is_empty()',
                   '    mpi_queue_index_bounds(source_count, targets.len());\n    if targets.is_empty()')
    for name in ('source', 'edge'):
        source = _once(source, f'queue[slot].push({name});',
                       f'queue[slot].push({name} as u32);')
        source = _once(source, f'for &{name} in &active {{',
                       f'for &{name} in &active {{\n                let {name} = {name} as usize;')
    return '''// Domain sizes include index zero, so 2^32 entries are representable.
#[inline]
fn mpi_queue_index_bounds(sources: usize, edges: usize) {
    assert!(sources.saturating_sub(1) <= u32::MAX as usize,
            "MPI compact queue source index exceeds u32");
    assert!(edges.saturating_sub(1) <= u32::MAX as usize,
            "MPI compact queue edge index exceeds u32");
}
''' + source


def compact_queue_source(model, source):
    count = 0
    for q, (syn, inst) in enumerate(zip(model['definition']['synapses'],
                                       model['instance']['synapses'], strict=True)):
        if syn['source_count'] > 2**32:
            raise ValueError('MPI compact queue source index exceeds u32')
        codes = [c for c in syn['code_objects'] if c['kind'] in ('synapses', 'synapses_post')]
        for path in inst['pathways']:
            matches = [c for c in codes if c['pathway_name'] == path['name']]
            if len(matches) != 1 or shared_additive(model, q, matches[0]) is None:
                raise ValueError('MPI compact queues require canonical additive pathways')
            count += 1
    kernel = (Path(__file__).with_name('mpi_runtime')/'additive.rs').read_text()
    source = _once(source, kernel, compact_additive_kernel(kernel))
    # Match declarations only; never rewrite ticks, topology or model literals.
    source, ordinary = re.subn(r'(?m)^(    let mut plan_s\d+p\d+_queue: )Vec<Vec<usize>>',
                               r'\1Vec<Vec<u32>>', source)
    bank = '    let mut mpi_pathway_queues: Vec<Vec<Vec<usize>>>'
    if bank in source:
        if ordinary or not count:
            raise ValueError('MPI queue compaction: inconsistent queue declarations')
        source = _once(source, bank, bank.replace('usize', 'u32'))
    else:
        if ordinary != count:
            raise ValueError('MPI queue compaction: missing queue declarations')
        source = _once(source, 'let queues: &[&[Vec<usize>]]', 'let queues: &[&[Vec<u32>]]')
    source = _once(source,
        'queue_capacity_items.checked_mul(std::mem::size_of::<usize>())',
        'queue_capacity_items.checked_mul(std::mem::size_of::<u32>())')
    # Restored pending items enter the same checked source/local-edge domain
    # as the shared additive kernel's ordinary enqueue operation.
    source = re.sub(r'((?:plan_s\d+p\d+_queue|mpi_pathway_queues\[\d+\])\[slot\]\.push\()(item|edge)(\);)',
                    r'\1\2 as u32\3', source)
    return source, {'compacted': True, 'index_bits': 32, 'pathways': count,
                    'overflow_policy': 'reject-before-narrowing', 'fifo_order': 'unchanged'}
