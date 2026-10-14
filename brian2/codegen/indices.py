"""Ordering of array reads with indirect index dependencies."""


def order_by_index(variable_names, variable_indices, *, allow_self_index=False):
    """Include transitive indices and place them before the variables they index.

    Root order is preserved where dependencies allow it. Index cycles fail
    before code generation, rather than producing uninitialised array accesses.
    The explicit stack also handles long chains without Python recursion.
    """
    ordered = []
    done = set()
    active = set()
    for root in variable_names:
        stack = [(root, False)]
        while stack:
            name, expanded = stack.pop()
            if name in done:
                continue
            if expanded:
                active.remove(name)
                done.add(name)
                ordered.append(name)
                continue
            if name in active:
                raise ValueError(f"Cyclic array index dependencies involving '{name}'")
            active.add(name)
            stack.append((name, True))
            index = variable_indices[name]
            if index not in ("_idx", "0") and not (allow_self_index and index == name):
                stack.append((index, False))
    return ordered
