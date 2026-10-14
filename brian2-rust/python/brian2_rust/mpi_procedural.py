"""Distributed fixed-total recipes: global identities, local storage/initializers."""
from .native import _initializer_call


def _initializer(initializer):
    function, arguments = _initializer_call(initializer)
    kind = 'ClippedNormal' if function == 'materialize_clipped_normal' else 'Uniform'
    return f'MpiInitializer::{kind}({arguments})'


def fixed_total_loading(model, q):
    syn = model['definition']['synapses'][q]
    values = model['instance']['synapses'][q]
    topology = values['topology']
    edges, seed = topology['edge_count'], topology['seed']
    target_pop = model['definition']['populations'][syn['target_population']]
    p = f's{q}_'
    lines = [f'    let {p}edge_count = {edges}usize;',
             f'    check(data.usize()? == {edges} && data.u64()? == {seed}u64, "MPI topology recipe mismatch")?;',
             f'    let {p}build_started = Instant::now();',
             f'    let ({p}offsets, {p}target_index, {p}original_edges) = mpi_fixed_total(&mpi, {syn["source_count"]}, {syn["target_count"]}, {syn["target_start"]}, {target_pop["count"]}, {syn["target_population"]}, {edges}, {seed}u64)?;',
             f'    let {p}local_edge_count = {p}target_index.len();']
    for i, symbol in enumerate(syn['parameters']):
        if symbol['index_domain'] == 'scalar':
            expression = 'data.f64()?'
        else:
            value = _initializer(topology['initializers'][symbol['name']])
            expression = f'mpi_parameter_values(&{p}original_edges, {seed}u64, {value})?'
        lines.append(f'    let {p}parameter_{i} = {expression};')
    for r, path in enumerate(values['pathways']):
        prefix = p if r == 0 else f's{q}p{r}_'
        initializer = path.get('delay_initializer')
        if initializer is None:
            delay = path['delay_ticks'][0]
            lines += [f'    check(data.usize()? == {delay}, "MPI delay mismatch")?;',
                      f'    let {prefix}delay_ticks = {delay}usize;']
        else:
            dt = f'f64::from_bits(0x{target_pop["dt"]}u64)'
            value = _initializer(initializer)
            lines.append(f'    let {prefix}delay_ticks = mpi_delay_values(&{p}original_edges, {seed}u64, {value}, {dt})?;')
        lines.append('    check(data.usize()? == 0, "MPI pending events unsupported")?;')
    from .mpi_partition import random_edges
    if not random_edges(syn):
        lines.append(f'    drop({p}original_edges);')
    lines += [f'    let {p}build_seconds = {p}build_started.elapsed().as_secs_f64();',
              f'    let mut {p}delivered = 0usize;', f'    let mut {p}post_delivered = 0usize;']
    return lines
