"""Rank-local instance emission without changing logical neuron/edge identities."""
import json
import re

import numpy as np

from .binary_topology import csr_arrays, file_hash, inspect_csr

CHUNK = 65536


def random_edges(synapse):
    # Validated expression trees carry stochastic operation names explicitly.
    def walk(node):
        if isinstance(node, dict):
            return node.get('op') in {'rand', 'randn', 'poisson', 'binomial'} or any(walk(v) for v in node.values())
        return isinstance(node, list) and any(walk(v) for v in node)
    return walk(synapse['code_objects'])


def write_shards(model, directory, plan):
    """Bounded edge slices; no Python object or complete selection array per edge."""
    d, inst = model['definition'], model['instance']
    readonly = set(plan.readonly_pre_states)
    # Explicit endpoints are immutable. Convert and partition them once instead
    # of rebuilding and scanning the complete topology for every rank. Each
    # rank's IDs remain in stable source-row/creation order, so shard bytes and
    # global edge identities retain the original contract.
    explicit_cache = {}
    for q, (synapse, state) in enumerate(zip(d['synapses'], inst['synapses'], strict=True)):
        if state['topology']['kind'] != 'explicit':
            continue
        targets = np.asarray(state['target'], dtype='<u4')
        sources = np.asarray(state['source'], dtype='<u4')
        total = len(targets)
        if len(sources) != total:
            raise ValueError('explicit source and target arrays differ in length')
        identity_order = total < 2 or bool(np.all(sources[:-1] <= sources[1:]))
        order = None if identity_order else np.argsort(sources, kind='stable')
        id_dtype = '<u4' if total <= np.iinfo(np.uint32).max else '<u8'
        ordered_ids = np.arange(total, dtype=id_dtype) if order is None else order.astype(id_dtype, copy=False)
        target_owner = plan.population_owners[synapse['target_population']]
        if target_owner is None:
            target_count = d['populations'][synapse['target_population']]['count']
            rank_for_edge = ((targets.astype(np.uint64) + synapse['target_start'] + 1)
                             * plan.ranks - 1) // target_count
            ordered_ranks = rank_for_edge if order is None else rank_for_edge[order]
            rank_ids = tuple(ordered_ids[ordered_ranks == rank] for rank in range(plan.ranks))
        else:
            empty = np.empty(0, dtype=id_dtype)
            rank_ids = tuple(ordered_ids if rank == target_owner else empty for rank in range(plan.ranks))
        explicit_cache[q] = (sources, targets, rank_ids)
    hashes = {}
    for shard in plan.shards:
        path = directory / f'instance.rank-{shard.rank}.bin'
        with path.open('wb') as out:
            def integers(items, dtype='<u8'):
                for first in range(0, len(items), CHUNK):
                    out.write(np.asarray(items[first:first+CHUNK], dtype=dtype).tobytes())
            def bits(items, dtype='<u8'):
                for first in range(0, len(items), CHUNK):
                    integers([int(v, 16) for v in items[first:first+CHUNK]], dtype)
            def symbol_bits(items, symbol):
                dtype = symbol['dtype']
                storage = {'bool': 'u1', 'f32': '<u4', 'i32': '<u4', 'u32': '<u4',
                           'f64': '<u8', 'i64': '<u8', 'u64': '<u8'}[dtype]
                bits(items, storage)
            out.write(b'B2AOT001')
            integers([len(d['populations']), inst['rng_seed']])
            for p, (pop, state) in enumerate(zip(d['populations'], inst['populations'], strict=True)):
                start, stop = shard.population_ranges[p]
                integers([int(pop['dt'], 16), pop['steps'], pop['count']])
                for symbol in pop['states']:
                    values = state['initial_state'][symbol['name']]
                    symbol_bits(values if (p, symbol['name']) in readonly else values[start:stop], symbol)
                for symbol in pop['parameters']:
                    values = state['parameters'][symbol['name']]
                    symbol_bits(values if symbol['index_domain'] == 'scalar' else values[start:stop], symbol)
                if state['refractory'] is not None:
                    ref = state['refractory']
                    integers([ref['period_ticks']])
                    bits(ref['initial_lastspike'][start:stop])
                    integers(ref['initial_not_refractory'][start:stop], 'u1')
                generator = state.get('spike_generator')
                if generator is not None:
                    # Only owned external sources are advanced by this process.
                    ids = np.asarray(generator['spike_indices'])
                    take = (ids >= start) & (ids < stop)
                    integers([int(take.sum())])
                    integers(np.asarray(generator['spike_ticks'])[take])
                    integers(ids[take])
            for q, (synapse, state) in enumerate(zip(d['synapses'], inst['synapses'], strict=True)):
                topology = state['topology']
                if topology['kind'] == 'fixed_total':
                    # Only the immutable recipe/scalars cross the Python boundary.
                    integers([topology['edge_count'], topology['seed']])
                    for symbol in synapse['parameters']:
                        if symbol['index_domain'] == 'scalar':
                            symbol_bits(state['parameters'][symbol['name']], symbol)
                    for pathway in state['pathways']:
                        if pathway.get('delay_initializer') is None:
                            integers(pathway['delay_ticks'])
                        integers([0])
                    continue
                if topology['kind'] == 'binary_csr':
                    info = inspect_csr(topology['path'])
                    if file_hash(info['path']) != topology['sha256']:
                        raise ValueError('binary CSR changed after export')
                    offsets, targets, columns = csr_arrays(info)
                    total = info['edge_count']
                    explicit = False
                else:
                    # Explicit AtlasIR already contains endpoint lists; stable sort
                    # retains original creation order within each source row.
                    sources, targets, rank_ids = explicit_cache[q]
                    total = len(targets)
                    explicit = True
                start, stop = shard.population_ranges[synapse['target_population']]
                counts = np.zeros(synapse['source_count'] + 1, dtype='<u8')
                def slices():
                    if explicit:
                        local_ids = rank_ids[shard.rank]
                        for first in range(0, len(local_ids), CHUNK):
                            ids = local_ids[first:first+CHUNK]
                            yield ids, targets[ids]
                        return
                    for first in range(0, total, CHUNK):
                        ids = np.arange(first, min(first+CHUNK, total), dtype=np.int64)
                        target = targets[ids]
                        take = (target.astype(np.int64) + synapse['target_start'] >= start) & (target.astype(np.int64) + synapse['target_start'] < stop)
                        yield ids[take], target[take]
                for ids, _ in slices():
                    source = sources[ids] if explicit else np.searchsorted(offsets, ids, side='right') - 1
                    np.add.at(counts, source + 1, 1)
                np.cumsum(counts, out=counts)
                local = int(counts[-1])
                if local != shard.incoming_edges[q]:
                    raise ValueError('rank topology disagrees with compiled ownership plan')
                integers([local]); integers(counts)
                for _, target in slices():
                    integers(target, '<u4')
                needs_original_edges = random_edges(synapse) or bool(synapse['states']) or any(path['kind'] == 'post' for path in state['pathways']) or any(
                    path['pending'] and len(set(path['delay_ticks'])) != 1 for path in state['pathways'])
                if needs_original_edges:
                    for ids, _ in slices():
                        integers(ids)
                for symbol in synapse['states']:
                    name = symbol['name']
                    for ids, _ in slices():
                        symbol_bits([state['initial_state'][name][int(i)] for i in ids], symbol)
                for symbol in synapse['parameters']:
                    name = symbol['name']
                    if symbol['index_domain'] == 'scalar':
                        symbol_bits(state['parameters'][name], symbol)
                    elif explicit:
                        for ids, _ in slices():
                            symbol_bits([state['parameters'][name][int(i)] for i in ids], symbol)
                    else:
                        values = columns[topology['initializers'][name]]
                        for ids, _ in slices():
                            out.write(np.asarray(values[ids], dtype='<f8').tobytes())
                for pathway in state['pathways']:
                    delays = pathway['delay_ticks']
                    uniform = bool(delays) and all(x == delays[0] for x in delays)
                    if uniform:
                        integers([delays[0]])
                    else:
                        for ids, _ in slices():
                            integers([delays[int(i)] for i in ids])
                    # Endpoint batches are expanded through this rank's CSR.
                    # Per-edge pending items must instead be mapped from the
                    # stable global creation identity to compact local storage.
                    pending = pathway['pending']
                    if not uniform:
                        local_map = {int(edge): local for local, edge in enumerate(
                            edge for ids, _ in slices() for edge in ids)}
                        pending = [dict(event, item=local_map[event['item']])
                                   for event in pending if event['item'] in local_map]
                    integers([len(pending)])
                    integers([event['delivery_tick'] for event in pending])
                    integers([event['item'] for event in pending])
            integers([clock['start_tick'] for clock in model['run']['clocks']])
            import struct
            from .native import _number
            out.write(struct.pack('<d', _number(model['run']['start']) + _number(model['run']['duration'])))
        hashes[path.name] = file_hash(path)
    # The CLI argument remains an immutable index. Each process opens its own
    # shard, even when all files are available on a shared filesystem.
    (directory/'instance.bin').write_text(json.dumps({'schema': 'b2-mpi-shards-v1',
        'plan_sha256': plan.sha256, 'files': hashes}, sort_keys=True)+'\n')
    identities = []
    for rank in range(plan.ranks):
        name = f'instance.rank-{rank}.bin'
        digest = list(bytes.fromhex(hashes[name]))
        identities.append(f'{rank} => ({(directory/name).stat().st_size}, {digest!r}),')
    (directory/'instance-identities.rs').write_text(
        'fn mpi_instance_identity(rank: usize) -> (usize, [u8;32]) { match rank {\n'
        + '\n'.join(identities) + '\n_ => unreachable!(), }}\n')
    return hashes


def population_ranges(pop, p):
    n = pop['count']
    return [f'    let (p{p}_start, p{p}_stop) = mpi.range({n}, {p});']


def synapse_loading(model, q, plan):
    syn = model['definition']['synapses'][q]
    values = model['instance']['synapses'][q]
    if values['topology']['kind'] == 'fixed_total':
        from .mpi_procedural import fixed_total_loading
        return fixed_total_loading(model, q)
    global_count = sum(s.incoming_edges[q] for s in plan.shards)
    expected = [s.incoming_edges[q] for s in plan.shards]
    n = syn['source_count']
    s = f's{q}_'
    lines = [f'    let {s}edge_count = {global_count}usize;',
             f'    let {s}local_edge_count = data.usize()?;',
             f'    check({s}local_edge_count == {expected!r}[mpi.rank], "rank edge count mismatch")?;',
             f'    let {s}offsets = data.usize_vec({n+1})?;',
             f'    check({s}offsets[0] == 0 && {s}offsets[{n}] == {s}local_edge_count && {s}offsets.windows(2).all(|w| w[0]<=w[1]), "invalid rank CSR")?;',
             f'    let {s}target_index = data.u32_vec({s}local_edge_count)?;',
             f'    check({s}target_index.iter().all(|&i| (i as usize) < {syn["target_count"]} && mpi.owns(i as usize+{syn["target_start"]}, {model["definition"]["populations"][syn["target_population"]]["count"]}, {syn["target_population"]})), "non-owned rank edge")?;']
    if random_edges(syn) or syn['states'] or any(path['kind'] == 'post' for path in values['pathways']) or any(
            path['pending'] and len(set(path['delay_ticks'])) != 1 for path in values['pathways']):
        lines += [f'    let {s}original_edges = data.usize_vec({s}local_edge_count)?;',
                  f'    check({s}original_edges.iter().all(|&e| e < {s}edge_count), "invalid global edge identity")?;']
    for i, symbol in enumerate(syn['states']):
        from .native import _rust_dtype
        lines.append(f'    let mut {s}state_{i} = data.{_rust_dtype(symbol)}_vec({s}local_edge_count)?;')
    for i, symbol in enumerate(syn['parameters']):
        from .native import _rust_dtype
        dtype = _rust_dtype(symbol)
        read = f'data.{dtype}()?' if symbol['index_domain'] == 'scalar' else f'data.{dtype}_vec({s}local_edge_count)?'
        lines.append(f'    let {s}parameter_{i} = {read};')
    for r, path in enumerate(values['pathways']):
        prefix = s if r == 0 else f's{q}p{r}_'
        delays = path['delay_ticks']
        uniform = bool(delays) and all(x == delays[0] for x in delays)
        count = '1' if uniform else f'{s}local_edge_count'
        lines += [f'    let {prefix}delay_values = data.usize_vec({count})?;',
                  f'    let {prefix}pending_count = data.usize()?;',
                  f'    let {prefix}pending_ticks = data.usize_vec({prefix}pending_count)?;',
                  f'    let {prefix}pending_items = data.usize_vec({prefix}pending_count)?;']
        if uniform:
            lines += [f'    check({prefix}delay_values[0] == {delays[0]}, "delay mismatch")?;',
                      f'    let {prefix}delay_ticks = {delays[0]}usize;']
        else:
            lines.append(f'    let {prefix}delay_ticks = {prefix}delay_values;')
    if any(path['kind'] == 'post' for path in values['pathways']):
        lines.append(f'    let ({s}target_offsets, mut {s}target_edges) = source_csr_u32(&{s}target_index, {syn["target_count"]});')
        lines.append(f'    for target in 0..{syn["target_count"]} {{ {s}target_edges[{s}target_offsets[target]..{s}target_offsets[target+1]].sort_unstable_by_key(|&edge| {s}original_edges[edge as usize]); }}')
    lines += [f'    let mut {s}delivered = 0usize;', f'    let mut {s}post_delivered = 0usize;']
    return lines


def localize_accesses(lines, model, plan):
    """Keep global loop/RNG indices; subtract the offset only at storage access."""
    source = '\n'.join(lines)
    readonly = set(plan.readonly_pre_states)
    for p, pop in enumerate(model['definition']['populations']):
        names = [f'p{p}_state_{i}' for i, symbol in enumerate(pop['states']) if (p, symbol['name']) not in readonly]
        names += [f'p{p}_parameter_{i}' for i, symbol in enumerate(pop['parameters']) if symbol['index_domain'] != 'scalar']
        if pop['refractory'] is not None:
            names += [f'p{p}_{name}' for name in ('lastspike', 'not_refractory', 'refractory_until')]
        for name in names:
            source = re.sub(r'\b'+name+r'\[([^\[\]]+)\]', lambda m: f'{name}[({m[1]})-p{p}_start]', source)
            source = re.sub(r'\b'+name+r'\.get_unchecked(_mut)?\(([^()]*)\)', lambda m: f'{name}.get_unchecked{m[1] or ""}(({m[2]})-p{p}_start)', source)
    return source.splitlines()
