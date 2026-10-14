"""Shallow expressions for side-effect-free, nonnegative integer counters."""


def usize_sum(terms):
    """Keep Rust AST depth logarithmic even with thousands of projections.

    Reassociation is only used for usize lengths/counters, never floats or
    simulation state updates. Their total and overflow outcome are unchanged.
    """
    level = list(terms)
    if not level:
        return "0usize"
    while len(level) > 1:
        level = [f"({level[i]} + {level[i+1]})" if i+1 < len(level) else level[i]
                 for i in range(0, len(level), 2)]
    return level[0]
