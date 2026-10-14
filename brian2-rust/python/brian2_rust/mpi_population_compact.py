"""Aggregate MPI population storage without changing field types or arithmetic.

The caller first applies homogeneous projection compaction. Each population's
initialization stays in its original order, returning one inferred aggregate.
Final collection moves vectors through empty values so error paths never leave
partially moved aggregates. Expected emitter fragments fail closed on drift.
"""

import re


_TOKENS = re.compile(
    r'r(?P<hashes>\#*)".*?"(?P=hashes)|"(?:\\.|[^"\\])*"|'
    r'//[^\n]*|/\*.*?\*/|\bp\d+_[A-Za-z0-9_]+\b',
    re.S,
)
_COLLECTION = re.compile(
    r'^    let (p\d+_(?:state_\d+|lastspike|not_refractory)) = '
    r'mpi\.(collect_f64|collect_flags)\(\1, (.*?)\)\?;$',
    re.M,
)


def aggregate_population_source(model, source):
    """Return aggregate-storage Rust and its generation metadata."""
    populations = model["definition"]["populations"]
    parts = []
    cursor = 0
    shapes = {}
    replacements = {}
    for p in range(len(populations)):
        start = source.index(
            f"    let (p{p}_start, p{p}_stop) = mpi.range(", cursor
        )
        marker = (
            f"    let (p{p+1}_start, p{p+1}_stop) = mpi.range("
            if p + 1 < len(populations) else "    let c0_dt = "
        )
        end = source.index(marker, start)
        block = source[start:end]
        bindings = [f"p{p}_start", f"p{p}_stop"]
        bindings.extend(
            match[1] for match in re.finditer(
                rf"^    let (?:mut )?(p{p}_[A-Za-z0-9_]+)(?=[: =])",
                block, re.M,
            )
        )
        if len(set(bindings)) != len(bindings):
            raise ValueError(f"MPI population compaction: duplicate binding in {p}")
        fields = tuple(name[len(f"p{p}_"):] for name in bindings)
        if fields not in shapes:
            shapes[fields] = f"MpiPopulation{len(shapes)}"
        typename = shapes[fields]
        values = ", ".join(
            f"{field}: {name}" for field, name in zip(fields, bindings, strict=True)
        )
        wrapper = (
            f"    let mut mpi_population_{p} = mpi_initialization_scope(|| {{\n"
            + block + "        Ok(" + typename + " { " + values
            + " })\n    })?;\n"
        )
        parts.extend([source[cursor:start], wrapper])
        cursor = end
        replacements.update(
            (name, f"mpi_population_{p}.{field}")
            for field, name in zip(fields, bindings, strict=True)
        )

    # Keep initialization's local declarations untouched. Only subsequent uses
    # refer to aggregates; helper kernels before execute retain their arguments.
    suffix = source[cursor:]
    suffix, collections = _COLLECTION.subn(
        lambda match: (
            f"    {match[1]} = mpi.{match[2]}(std::mem::take(&mut {match[1]}), "
            f"{match[3]})?;"
        ),
        suffix,
    )
    for match in re.finditer(r"\blet (?:mut )?(p\d+_[A-Za-z0-9_]+)", suffix):
        if match[1] in replacements:
            raise ValueError(
                f"MPI population compaction: unexpected rebinding {match[1]}"
            )
    suffix = _TOKENS.sub(
        lambda match: replacements.get(match.group(), match.group()), suffix
    )
    parts.append(suffix)
    source = "".join(parts)
    for fields, typename in shapes.items():
        types = ", ".join(f"T{i}" for i in range(len(fields)))
        members = ", ".join(f"{name}: T{i}" for i, name in enumerate(fields))
        source += f"\nstruct {typename}<{types}> {{ " + members + " }\n"
    source += (
        "\n#[inline(never)]\n"
        "fn mpi_initialization_scope<T, F: FnOnce() -> Result<T>>"
        "(initialize: F) -> Result<T> { initialize() }\n"
    )
    return source, {
        "compacted": True,
        "population_structs": len(populations),
        "population_shapes": len(shapes),
        "population_fields": len(replacements),
        "collection_moves": collections,
        "source_bytes": len(source.encode()),
    }
