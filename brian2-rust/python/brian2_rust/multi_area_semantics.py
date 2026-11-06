"""Explicit, opt-in time-grid translation for the MAM NEST reference.

NEST iaf_psc_exp stamps a spike at the right edge of its integration step,
then skips t_ref/dt complete updates. Brian stamps that same spike at the
left edge and resumes when (tick-lastspike) >= refractory/dt. Consequently
the corresponding Brian duration is t_ref + dt. This is an adapter setting,
not a change to Brian's general refractory behavior or the biological t_ref.
Audited against NEST v2.8.0 and v3.10 iaf_psc_exp.cpp update().
"""

import math


def nest_grid_refractory_ms(t_ref_ms, dt_ms):
    """Return Brian duration for a nonnegative, grid-aligned NEST t_ref."""
    if not math.isfinite(t_ref_ms) or t_ref_ms < 0:
        raise ValueError('NEST refractory duration must be finite and nonnegative')
    if not math.isfinite(dt_ms) or dt_ms <= 0:
        raise ValueError('NEST clock step must be finite and positive')
    ticks = t_ref_ms / dt_ms
    if not math.isfinite(ticks) or not math.isclose(ticks, round(ticks), rel_tol=0, abs_tol=1e-9):
        raise ValueError('NEST refractory translation requires an integer number of clock steps')
    return (round(ticks) + 1) * dt_ms


def nest_poisson_gate_tick(delay_ticks):
    """Brian synapses-slot first gate for NEST poisson_generator start=0.

    NEST tests is_active at the left edge (strictly greater than zero), then
    emits at its right edge. Its first possible emission is physical tick 2.
    Current therefore arrives at physical tick delay+2, visible in Brian's
    start monitor one tick after the synapses-slot gate delay+1. Audited on
    NEST v2.8.0/v3.10 source and measured on v3.10, not arbitrary generators,
    nonzero start/stop times or continuation/checkpoint inputs.
    """
    if type(delay_ticks) is not int or delay_ticks < 1:
        raise ValueError('NEST Poisson delay must be a positive integer number of ticks')
    return delay_ticks + 1
