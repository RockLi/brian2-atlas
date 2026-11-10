"""Opt-in source compaction for homogeneous MPI fixed-total projections.

This physical emission pass retains canonical operation order, random identities,
shards, scalar kernels and Result error propagation. Projection recipes and
queues use tables; adjacent compatible additive nodes share an ordered loop.
It never combines floating-point reductions or crosses another schedule node.

The pass consumes only this package's generated Rust. Expected source fragments
are checked explicitly, including under Python -O: emitter drift is an error,
not a partially rewritten executable. Other projection layouts retain the
ordinary emitter. This can bound compiler front-end cost; it does not guarantee
bounded LLVM optimization time for every model.
"""

import re
from .mpi_procedural import fixed_total_loading, _initializer
from .mpi_partition import random_edges
from .native import _dtype_size, _synapse_edge_count
from .codegen_sums import usize_sum
from .mpi_report_compact import PROJECTION_REPORT_BATCH_SIZE, projection_report_source

TOKENS = re.compile(
    'r(?P<hashes>\\#*)".*?"(?P=hashes)|"(?:\\\\.|[^"\\\\])*"|//[^\\n]*|/\\*.*?\\*/|\\bs\\d+(?:p\\d+)?_[A-Za-z0-9_]+\\b',
    re.S,
)


def _structured(model, source):
    """Share recipe loaders by field layout; keep Reader consumption in order."""
    replacements = {}
    helpers = []
    shapes = {}
    changes = []
    cursor = 0
    parts = []
    for q, (syn, inst) in enumerate(
        zip(model["definition"]["synapses"], model["instance"]["synapses"])
    ):
        if inst.get("topology", {}).get("kind") != "fixed_total":
            continue
        if syn["states"]:
            raise ValueError(
                "MPI projection compaction: generated source differs at not syn['states']"
            )
        p = f"s{q}_"
        fields = [
            (p + "edge_count", "edge_count", "usize"),
            (p + "offsets", "offsets", "Vec<usize>"),
            (p + "target_index", "target_index", "Vec<u32>"),
            (p + "local_edge_count", "local_edge_count", "usize"),
        ]
        if random_edges(syn):
            fields.append((p + "original_edges", "original_edges", "Vec<usize>"))
        for i, symbol in enumerate(syn["parameters"]):
            fields.append(
                (
                    p + f"parameter_{i}",
                    f"parameter_{i}",
                    "f64" if symbol["index_domain"] == "scalar" else "Vec<f64>",
                )
            )
        for r, path in enumerate(inst["pathways"]):
            prefix = p if r == 0 else f"s{q}p{r}_"
            fields.append(
                (
                    prefix + "delay_ticks",
                    f"delay_{r}",
                    "Vec<usize>"
                    if path.get("delay_initializer") is not None
                    else "usize",
                )
            )
        fields.extend(
            (
                (p + name, name, dtype)
                for name, dtype in [
                    ("build_seconds", "f64"),
                    ("delivered", "usize"),
                    ("post_delivered", "usize"),
                ]
            )
        )
        shape = tuple(((name, dtype) for _, name, dtype in fields))
        fresh = shape not in shapes
        if fresh:
            shapes[shape] = f"MpiProjection{len(shapes)}"
        typename = shapes[shape]
        loader = "mpi_load_" + typename
        pop = model["definition"]["populations"][syn["target_population"]]
        recipe = inst["topology"]
        declarations = [
            "data: &mut Reader",
            "mpi: &MpiWorld",
            "sources: usize",
            "targets: usize",
            "target_start: usize",
            "target_count: usize",
            "target_population: usize",
            "edges: usize",
            "seed: u64",
            "dt: f64",
        ]
        arguments = [
            "&mut data",
            "mpi",
            str(syn["source_count"]),
            str(syn["target_count"]),
            str(syn["target_start"]),
            str(pop["count"]),
            str(syn["target_population"]),
            str(recipe["edge_count"]),
            str(recipe["seed"]) + "u64",
            f"f64::from_bits(0x{pop['dt']}u64)",
        ]
        body = [
            '    check(data.usize()? == edges && data.u64()? == seed, "MPI topology recipe mismatch")?;',
            "    let build_started = Instant::now();",
            "    let (offsets,target_index,original_edges) = mpi_fixed_total(mpi,sources,targets,target_start,target_count,target_population,edges,seed)?;",
            "    let local_edge_count = target_index.len();",
        ]
        for i, symbol in enumerate(syn["parameters"]):
            if symbol["index_domain"] == "scalar":
                expression = "data.f64()?"
            else:
                declarations.append(f"initializer_parameter_{i}: MpiInitializer")
                arguments.append(_initializer(recipe["initializers"][symbol["name"]]))
                expression = f"mpi_parameter_values(&original_edges,seed,initializer_parameter_{i})?"
            body.append(f"    let parameter_{i} = {expression};")
        for r, path in enumerate(inst["pathways"]):
            if path.get("delay_initializer") is None:
                declarations.append(f"delay_{r}: usize")
                arguments.append(str(path["delay_ticks"][0]))
                body.append(
                    f'    check(data.usize()? == delay_{r}, "MPI delay mismatch")?;'
                )
            else:
                declarations.append(f"initializer_delay_{r}: MpiInitializer")
                arguments.append(_initializer(path["delay_initializer"]))
                body.append(
                    f"    let delay_{r} = mpi_delay_values(&original_edges,seed,initializer_delay_{r},dt)?;"
                )
            body.append(
                '    check(data.usize()? == 0, "MPI pending events unsupported")?;'
            )
        if not random_edges(syn):
            body.append("    drop(original_edges);")
        body += [
            "    let edge_count = edges;",
            "    let build_seconds = build_started.elapsed().as_secs_f64();",
            "    let delivered = 0usize;",
            "    let post_delivered = 0usize;",
        ]
        body.append(
            "    Ok("
            + typename
            + " { "
            + ", ".join((name for _, name, _ in fields))
            + " })"
        )
        if fresh:
            helpers.append(
                "#[inline(never)]\nfn "
                + loader
                + "("
                + ", ".join(declarations)
                + ") -> Result<"
                + typename
                + "> {\n"
                + "\n".join(body)
                + "\n}"
            )
        old = "\n".join(fixed_total_loading(model, q))
        at = source.find(old, cursor)
        if at < cursor:
            raise ValueError(
                "MPI projection compaction: generated source differs at at >= cursor"
            )
        parts.extend(
            [
                source[cursor:at],
                f"    let mut s{q}: {typename} = "
                + loader
                + "("
                + ", ".join(arguments)
                + ")?;",
            ]
        )
        cursor = at + len(old)
        changes.append(q)
        replacements.update(((var, f"s{q}.{name}") for var, name, _ in fields))
    parts.append(source[cursor:])
    source = "".join(parts)
    boundary = source.index("\nfn execute(") + 1
    source = source[:boundary] + TOKENS.sub(
        lambda m: replacements.get(m.group(), m.group()), source[boundary:]
    )
    declarations = [
        "struct "
        + name
        + " { "
        + ", ".join((f"{field}: {dtype}" for field, dtype in shape))
        + " }"
        for shape, name in shapes.items()
    ]
    return (
        source + "\n" + "\n".join(declarations + helpers) + "\n",
        dict(projections=len(changes), shapes=len(shapes), fields=len(replacements)),
    )


def _fold_dump(model, source):
    d = model["definition"]
    inst = model["instance"]
    static_bytes = 64 + sum(
        (
            24
            + sum((_dtype_size(s) for s in syn["states"])) * _synapse_edge_count(state)
            for syn, state in zip(d["synapses"], inst["synapses"])
        )
    )
    terms = [f"{static_bytes}usize"]
    for p, pop in enumerate(d["populations"]):
        terms += ["64usize"]
        terms += [
            f"p{p}_samples_{column}.len()*{_dtype_size(symbol)}usize"
            for column, name in enumerate(pop["monitor"]["variables"])
            for symbol in pop["states"]
            + pop["parameters"]
            + pop.get("linked_variables", [])
            if symbol["name"] == name
        ]
        terms += [
            f"p{p}_spikes.len()*16usize",
            f"p{p}_counts.len()*8usize",
            f"p{p}_last_fired.len()*8usize",
            f"{sum((_dtype_size(s) for s in pop['states'])) * pop['count']}usize",
        ]
        if pop["refractory"] is not None:
            terms += [f"p{p}_lastspike.len()*8usize", f"p{p}_not_refractory.len()"]
    new = "    let dump_bytes = " + usize_sum(terms) + ";"
    source, count = re.subn(
        "^    let dump_bytes = .*;$", lambda m: new, source, flags=re.M
    )
    if count != 1:
        raise ValueError(
            "MPI projection compaction: generated source differs at count == 1"
        )
    return (source, dict(dump_terms=len(terms), folded_static_bytes=static_bytes))


LEX = re.compile(
    TOKENS.pattern + "|\\bs\\d+\\.[A-Za-z0-9_]+|\\bplan_s\\d+p\\d+_queue\\b", re.S
)


def _banked(model, source):
    """Bank homogeneous storage and replace accounting/output repetition."""
    source, stats = _structured(model, source)
    count = len(model["definition"]["synapses"])
    if not count or stats["projections"] != count or stats["shapes"] != 1:
        return (source, stats)
    boundary = source.index("\nfn execute(") + 1
    prefix = source[:boundary]
    body = source[boundary:]
    pattern = "^    let mut s(\\d+): MpiProjection0 = (.*);$"
    loads = list(re.finditer(pattern, body, re.M))
    if len(loads) != count:
        raise ValueError(
            "MPI projection compaction: generated source differs at len(loads) == count"
        )
    body = re.sub(
        pattern, lambda m: "    mpi_projections.push(" + m[2] + ");", body, flags=re.M
    )
    first = body.index("    mpi_projections.push(")
    body = (
        body[:first]
        + f"    let mut mpi_projections: Vec<MpiProjection0> = Vec::with_capacity({count});\n"
        + body[first:]
    )
    queues = list(
        re.finditer(
            "^    let mut (plan_s\\d+p\\d+_queue): Vec<Vec<usize>> = (.*);$", body, re.M
        )
    )
    names = {m[1]: f"mpi_pathway_queues[{i}]" for i, m in enumerate(queues)}
    body = re.sub(
        "^    let mut (plan_s\\d+p\\d+_queue): Vec<Vec<usize>> = (.*);$",
        lambda m: "    mpi_pathway_queues.push(" + m[2] + ");",
        body,
        flags=re.M,
    )
    if queues:
        first = body.index("    mpi_pathway_queues.push(")
        body = (
            body[:first]
            + f"    let mut mpi_pathway_queues: Vec<Vec<Vec<usize>>> = Vec::with_capacity({len(queues)});\n"
            + body[first:]
        )

    def rewrite(m):
        text = m.group()
        if text in names:
            return names[text]
        match = re.fullmatch("s(\\d+)\\.([A-Za-z0-9_]+)", text)
        return f"mpi_projections[{match[1]}].{match[2]}" if match else text

    body = LEX.sub(rewrite, body)
    old = "\n".join(
        (
            f"    mpi_projections[{q}].delivered = mpi.sum(mpi_projections[{q}].delivered)?;"
            for q in range(count)
        )
    )
    if body.count(old) != 1:
        raise ValueError(
            "MPI projection compaction: generated source differs at body.count(old) == 1"
        )
    body = body.replace(
        old,
        "    // Delivery sums are checked in the bounded topology report batches below.",
    )
    if queues:
        body, n = re.subn(
            "^    let queues: &\\[&\\[Vec<usize>\\]\\] = &\\[.*\\];$",
            "    let queues = &mpi_pathway_queues;",
            body,
            flags=re.M,
        )
        if n != 1:
            raise ValueError(
                "MPI projection compaction: generated source differs at n == 1"
            )
    start = body.index('    let mut topology_report = String::from("[");')
    end = body.index('    topology_report.push_str("]");', start) + len(
        '    topology_report.push_str("]");'
    )
    report = projection_report_source()
    body = body[:start] + report + body[end:]
    old = "\n".join(
        (
            line
            for q, syn in enumerate(model["instance"]["synapses"])
            for line in [
                "    dump_u64(&mut dump, 0)?;",
                f"    dump_u64(&mut dump, {syn['topology']['edge_count']})?;",
                f"    dump_u64(&mut dump, mpi_projections[{q}].delivered + mpi_projections[{q}].post_delivered)?;",
            ]
        )
    )
    if body.count(old) != 1:
        raise ValueError(
            "MPI projection compaction: generated source differs at body.count(old) == 1"
        )
    body = body.replace(
        old,
        "    for projection in &mpi_projections { dump_u64(&mut dump, 0)?; dump_u64(&mut dump, projection.edge_count)?; dump_u64(&mut dump, projection.delivered + projection.post_delivered)?; }",
    )
    report_write = '    fs::write(output.join("mpi-runtime.json"), &mpi_report)?;'
    if body.count(report_write) != 1:
        raise ValueError("MPI projection compaction: generated report writer differs")
    report_metrics = r'''    mpi_report = mpi_report.trim_end().trim_end_matches("}").to_owned() + &format!(",\"projection_report_collectives\":{},\"projection_report_packet_values\":{}}}\n", mpi_projection_report_collectives, mpi_projection_report_packet_values);'''
    body = body.replace(report_write, report_metrics + "\n" + report_write)
    source = (
        prefix
        + body
        + "\n"
        + f"const MPI_PROJECTION_TARGETS: [usize; {count}] = "
        + repr([s["target_population"] for s in model["definition"]["synapses"]])
        + ";\n"
    )
    source, fold = _fold_dump(model, source)
    stats.update(
        fold, banked=True, queues=len(queues),
        projection_report_batch_size=PROJECTION_REPORT_BATCH_SIZE,
        projection_report_collectives=(count+PROJECTION_REPORT_BATCH_SIZE-1)//PROJECTION_REPORT_BATCH_SIZE,
        source_bytes=len(source.encode())
    )
    return (source, stats)


def split_arguments(text):
    parts = []
    start = 0
    depth = 0
    for i, char in enumerate(text):
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == "," and depth == 0:
            parts.append(text[start:i].strip())
            start = i + 1
    parts.append(text[start:].strip())
    if depth != 0:
        raise ValueError(
            "MPI projection compaction: generated source differs at depth == 0"
        )
    return parts


def _tabled(model, source):
    """Move static recipe operands and queue dimensions into constant tables."""
    source, stats = _banked(model, source)
    if not stats.get("banked"):
        return (source, stats)
    count = stats["projections"]
    signature = re.search(
        "fn mpi_load_MpiProjection0\\((.*?)\\) -> Result<MpiProjection0>", source
    ).group(1)
    declarations = split_arguments(signature)[2:]
    fields = [d.split(":", 1)[0] for d in declarations]
    calls = list(
        re.finditer(
            "^    mpi_projections.push\\(mpi_load_MpiProjection0\\((.*)\\)\\?\\);$",
            source,
            re.M,
        )
    )
    if len(calls) != count:
        raise ValueError(
            "MPI projection compaction: generated source differs at len(calls) == count"
        )
    recipes = []
    for call in calls:
        args = split_arguments(call[1])[2:]
        if len(args) != len(fields):
            raise ValueError(
                "MPI projection compaction: generated source differs at len(args) == len(fields)"
            )
        recipes.append(
            "MpiRecipe { "
            + ", ".join((f"{name}: {arg}" for name, arg in zip(fields, args)))
            + " }"
        )
    old = "\n".join((c.group() for c in calls))
    if source.count(old) != 1:
        raise ValueError(
            "MPI projection compaction: generated source differs at source.count(old) == 1"
        )
    loop = (
        "    for recipe in &MPI_RECIPES { mpi_projections.push(mpi_load_MpiProjection0(&mut data, mpi, "
        + ", ".join(("recipe." + f for f in fields))
        + ")?); }"
    )
    source = source.replace(old, loop)
    source += (
        "\nstruct MpiRecipe { "
        + ", ".join(declarations)
        + " }\nstatic MPI_RECIPES: [MpiRecipe; "
        + str(count)
        + "] = [\n"
        + ",\n".join(recipes)
        + "];\n"
    )
    queues = list(re.finditer("^    mpi_pathway_queues.push\\(.*\\);$", source, re.M))
    paths = model["instance"]["synapses"][0]["pathways"]
    width = len(paths)
    if len(queues) != count * width:
        raise ValueError(
            "MPI projection compaction: generated source differs at len(queues) == count * width"
        )
    old = "\n".join((c.group() for c in queues))
    if source.count(old) != 1:
        raise ValueError(
            "MPI projection compaction: generated source differs at source.count(old) == 1"
        )
    queue_steps = []
    for definition, inst in zip(
        model["definition"]["synapses"], model["instance"]["synapses"]
    ):
        queue_steps.append(
            [
                model["definition"]["populations"][
                    definition[
                        "source_population"
                        if path["kind"] == "pre"
                        else "target_population"
                    ]
                ]["steps"]
                for path in inst["pathways"]
            ]
        )
    lines = ["    for (q, projection) in mpi_projections.iter().enumerate() {"]
    for r, path in enumerate(paths):
        maximum = (
            f"projection.delay_{r}"
            if path.get("delay_initializer") is None
            else f"projection.delay_{r}.iter().copied().max().unwrap_or(0)"
        )
        lines.append(
            f"        mpi_pathway_queues.push(vec![Vec::new(); (({maximum})+1).min(MPI_QUEUE_STEPS[q][{r}].max(1))]);"
        )
    lines.append("    }")
    source = source.replace(old, "\n".join(lines))
    source += (
        f"const MPI_QUEUE_STEPS: [[usize; {width}]; {count}] = "
        + repr(queue_steps)
        + ";\n"
    )
    stats.update(tabled=True, source_bytes=len(source.encode()))
    return (source, stats)


PATTERN = re.compile(
    "        // canonical node ([^\\n]+)\\n        if c(\\d+)_active \\{\\n            mpi_projections\\[(\\d+)\\]\\.delivered \\+= mpi_additive_projection\\((.*?)\\);\\n        \\}",
    re.S,
)


def _batch(model, source):
    """Combine only adjacent canonical additive nodes with identical layouts."""
    source, stats = _tabled(model, source)
    if not stats.get("tabled"):
        return (source, stats)
    count = stats["projections"]
    if count > 1:
        for term, iterator in [
            (
                lambda q: f"mpi_projections[{q}].delivered",
                "mpi_projections.iter().map(|projection| projection.delivered).sum::<usize>()",
            ),
            (
                lambda q: f"mpi_projections[{q}].delivered + mpi_projections[{q}].post_delivered",
                "mpi_projections.iter().map(|projection| projection.delivered+projection.post_delivered).sum::<usize>()",
            ),
        ]:
            old = usize_sum((term(q) for q in range(count)))
            if old not in source:
                raise ValueError(
                    "MPI projection compaction: generated source differs at old in source"
                )
            source = source.replace(old, iterator)
    groups = []
    for m in PATTERN.finditer(source):
        q = int(m[3])
        args = split_arguments(m[4])
        if len(args) != 13:
            raise ValueError(
                "MPI projection compaction: generated source differs at len(args) == 13"
            )
        src = int(re.fullmatch("&p(\\d+)_fired", args[0])[1])
        target = re.fullmatch("&mut p(\\d+)_state_(\\d+)", args[10])
        tgt, col = map(int, target.groups())
        if args[3:7] != [
            f"p{src}_tick",
            f"p{src}_end_tick",
            f"&mpi_projections[{q}].offsets",
            f"&mpi_projections[{q}].target_index",
        ]:
            raise ValueError(
                "MPI projection compaction: generated source differs at args[3:7] == [f'p{src}_tick', f'p{src}_end_tick', f'&mpi_projections[{q}].offsets', f'&mpi_projections[{q}].target_index']"
            )
        if args[12] != f"p{tgt}_start":
            raise ValueError(
                "MPI projection compaction: generated source differs at args[12] == f'p{tgt}_start'"
            )
        queue = int(re.fullmatch("&mut mpi_pathway_queues\\[(\\d+)\\]", args[9])[1])
        delay = args[7].replace(f"mpi_projections[{q}]", "projection")
        weight = args[8].replace(f"mpi_projections[{q}]", "projection")
        key = (m[2], delay, weight)
        item = dict(
            match=m,
            q=q,
            source=src,
            target=(tgt, col),
            queue=queue,
            start=int(args[1]),
            count=int(args[2]),
            target_start=int(args[11]),
            key=key,
        )
        if (
            groups
            and groups[-1][-1]["key"] == key
            and (not source[groups[-1][-1]["match"].end() : m.start()].strip())
        ):
            groups[-1].append(item)
        else:
            groups.append([item])
    groups = [g for g in groups if len(g) > 1]
    parts = []
    cursor = 0
    constants = []
    total = 0
    for index, group in enumerate(groups):
        sources = list(dict.fromkeys((i["source"] for i in group)))
        targets = list(dict.fromkeys((i["target"] for i in group)))
        ops = [
            (
                i["q"],
                sources.index(i["source"]),
                targets.index(i["target"]),
                i["start"],
                i["count"],
                i["target_start"],
                i["queue"],
            )
            for i in group
        ]
        clock, delay, weight = group[0]["key"]
        lines = [
            f"        // consecutive canonical additive nodes: {len(group)}",
            f"        if c{clock}_active {{",
            f"            let sources: [&[usize]; {len(sources)}] = ["
            + ", ".join((f"&p{p}_fired" for p in sources))
            + "];",
            f"            let ticks: [(usize,usize); {len(sources)}] = ["
            + ", ".join((f"(p{p}_tick,p{p}_end_tick)" for p in sources))
            + "];",
            f"            let mut states: [&mut [f64]; {len(targets)}] = ["
            + ", ".join((f"&mut p{p}_state_{c}" for p, c in targets))
            + "];",
            f"            let starts: [usize; {len(targets)}] = "
            + "["
            + ", ".join((f"p{p}_start" for p, c in targets))
            + "];",
            f"            for &(q,source,target,source_start,source_count,target_start,queue) in &MPI_ADDITIVE_OPS_{index} {{",
            "                let projection = &mut mpi_projections[q];",
            "                projection.delivered += mpi_additive_projection(sources[source], source_start, source_count, ticks[source].0, ticks[source].1,",
            f"                    &projection.offsets, &projection.target_index, {delay}, {weight}, &mut mpi_pathway_queues[queue],",
            "                    &mut *states[target], target_start, starts[target]);",
            "            }",
            "        }",
        ]
        parts.extend([source[cursor : group[0]["match"].start()], "\n".join(lines)])
        cursor = group[-1]["match"].end()
        total += len(group)
        constants.append(
            f"const MPI_ADDITIVE_OPS_{index}: [(usize,usize,usize,usize,usize,usize,usize); {len(ops)}] = "
            + repr(ops)
            + ";"
        )
    parts.append(source[cursor:])
    source = "".join(parts) + "\n" + "\n".join(constants) + "\n"
    stats.update(
        additive_groups=len(groups),
        additive_nodes=total,
        source_bytes=len(source.encode()),
    )
    return (source, stats)


def compact_source(model, source):
    """Return (Rust source, metadata), preserving the source on unsupported layouts."""
    synapses = model["definition"]["synapses"]
    instances = model["instance"]["synapses"]
    if not synapses or any(
        synapse["states"] or instance.get("topology", {}).get("kind") != "fixed_total"
        for synapse, instance in zip(synapses, instances, strict=True)
    ):
        return source, {"compacted": False}
    candidate, details = _batch(model, source)
    if not details.get("tabled"):
        return source, {"compacted": False}
    return candidate, {"compacted": True, **details}
