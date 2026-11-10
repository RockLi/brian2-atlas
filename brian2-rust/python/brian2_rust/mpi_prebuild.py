"""Opt-in early shared topology, after compaction, without changing the plan.

All fixed-total constructors are visited in canonical projection order by every
rank. An ordinal distinguishes even identical recipes; other topology kinds do
not consume ordinals. Checked source anchors keep this physical pass atomic.
"""
from pathlib import Path
import re


def prebuild_shared_source(model, plan, source):
    recipes = []
    count = 0
    for q, (syn, inst) in enumerate(zip(model['definition']['synapses'], model['instance']['synapses'], strict=True)):
        topology = inst.get('topology', {})
        if topology.get('kind') != 'fixed_total':
            continue
        if plan.population_owners[syn['target_population']] is None:
            fields = dict(ordinal=count, projection=q, sources=syn['source_count'],
                targets=syn['target_count'], target_start=syn['target_start'],
                population_size=model['definition']['populations'][syn['target_population']]['count'],
                population=syn['target_population'], edges=topology['edge_count'], seed=topology['seed'])
            recipes.append('MpiPrebuildRecipe { '+', '.join(f'{k}: {v}'+('u64' if k == 'seed' else '') for k,v in fields.items())+' }')
        count += 1
    if not recipes:
        return source, dict(enabled=False, shared_projections=0, fixed_projections=count)

    def replace(old, new):
        nonlocal source
        if source.count(old) != 1:
            raise ValueError('MPI topology prebuild: generated source differs at '+old[:80])
        source = source.replace(old, new)

    replace('fn mpi_fixed_total(', 'fn mpi_fixed_total_uncached(')
    replace('    constructed_edges: std::cell::Cell<usize>,',
            '    constructed_edges: std::cell::Cell<usize>,\n    prebuilt: std::cell::RefCell<MpiPrebuilt>,')
    replace('constructed_edges: 0.into() })', 'constructed_edges: 0.into(), prebuilt: Default::default() })')
    anchor = '    let local_count = usize::try_from(total[sources+mpi.rank])?;'
    replace(anchor, anchor+'\n    mpi_prebuilt_admit(mpi,sources,local_count)?;')
    anchor = '    let reader = Reader::open_bound(&shard, mpi_instance_identity(mpi.rank))?;'
    replace(anchor, anchor+'\n    mpi_prebuilt_verify(mpi,&shard,mpi_instance_identity(mpi.rank).1)?;')
    anchor = '    let parallel_threads = parallel.threads();'
    replace(anchor, anchor+'\n    mpi_prebuilt_prepare(mpi)?;')
    anchor = '    let mpi_local_initialization_seconds = initialization_started.elapsed().as_secs_f64();'
    replace(anchor, '    mpi_prebuilt_finish(mpi)?;\n'+anchor)
    source, timings = re.subn(r'(    let (?:s\d+_)?build_seconds = (?:s\d+_)?build_started\.elapsed\(\)\.as_secs_f64\(\));',
                            r'\1 + mpi_prebuilt_seconds(mpi);', source)
    if not timings:
        raise ValueError('MPI topology prebuild: generated source differs at build timing')
    anchor = '    if mpi.rank != 0 { return Ok(()); }'
    replace(anchor, '    let mpi_prebuilt_stats = mpi_prebuilt_records(mpi)?;\n'+anchor)
    anchor = '    fs::write(output.join("mpi-runtime.json"), &mpi_report)?;'
    replace(anchor, '    mpi_report = mpi_report.trim_end().trim_end_matches("}").to_owned() + &mpi_prebuilt_json(&mpi_prebuilt_stats) + "}\\n";\n'+anchor)
    runtime = Path(__file__).with_name('mpi_runtime').joinpath('prebuild.rs').read_text()
    source += '\n'+runtime+f'\nconst MPI_PREBUILD_FIXED_COUNT: usize = {count};\n'
    source += f'static MPI_PREBUILD_RECIPES: [MpiPrebuildRecipe; {len(recipes)}] = [\n'+',\n'.join(recipes)+'\n];\n'
    return source, dict(enabled=True, shared_projections=len(recipes), fixed_projections=count,
        cache_limit_environment='B2_MPI_MAX_PREBUILT_TOPOLOGY_BYTES', default_cache_limit_bytes=128*2**20,
        cache_scope='Retained topology vectors and slot metadata; construction temporaries remain under service guards',
        source_bytes=len(source.encode()))
