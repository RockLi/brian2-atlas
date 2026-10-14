"""Host scheduling of native GPU dispatches using the Rust clock contract.

Clock decisions remain f64 even when device state arithmetic is explicit f32.
Coincident clocks execute nodes in the validated global Brian schedule order.
"""
from pathlib import Path

from .metal import number


def clock_arrays(clocks):
    return ([c.start_tick for c in clocks],
            [c.start_tick+c.steps for c in clocks],
            [number(c.dt) for c in clocks])


def dispatch_ticks(clocks, stage_clocks):
    ticks, ends, dts = clock_arrays(clocks)
    epsilon = min(dts)*1e-12
    while True:
        times = [float(tick)*dt for tick, dt in zip(ticks, dts, strict=True)]
        live = [times[c] for c in range(len(clocks)) if ticks[c]<ends[c]]
        if not live:
            return
        time = min(live)
        active = [ticks[c]<ends[c] and abs(times[c]-time)<=epsilon for c in range(len(clocks))]
        for stage, clock in enumerate(stage_clocks):
            if active[clock]:
                yield stage, ticks[clock]
        for c in range(len(clocks)):
            ticks[c] += active[c]


def native_scheduler_source():
    return (Path(__file__).parent/'metal_runtime/clocks.h').read_text()


def owner_tick_expression(logical, dispatch_clock, owner_clock):
    """Map a scheduled tick to Brian's next owner tick, with a checked grid.

    GPU kernels lack f64 clock arithmetic. Certify a small rational grid against
    the host scheduler's epsilon over this activation before emitting integers.
    Reject uncertifiable grids instead of rounding time to a different clock.
    """
    if dispatch_clock == owner_clock:
        return 'tick'
    import math
    from fractions import Fraction
    from .plan import PlanValidationError

    source, owner = logical.clocks[dispatch_clock], logical.clocks[owner_clock]
    source_dt, owner_dt = number(source.dt), number(owner.dt)
    exact = Fraction(source_dt) / Fraction(owner_dt)
    ratio = exact.limit_denominator(1_000_000)
    numerator, denominator = ratio.numerator, ratio.denominator
    maximum = source.start_tick + source.steps
    owner_maximum = owner.start_tick + owner.steps
    epsilon = min(number(clock.dt) for clock in logical.clocks) * 1e-12

    def rounding_bound(dt, ticks):
        # Multiplication by an exact binary grid is exact while the integer
        # significand fits f64; otherwise use a conservative full-ulp bound.
        fraction = Fraction(dt)
        if abs(fraction.numerator) * ticks <= 2**53:
            return 0.0
        return math.ulp(float(ticks) * dt)

    # Coalescence is a global decision: certify every participating clock on
    # the same lattice, including third clocks near a source/owner tie.
    common_denominator = 1
    error = Fraction(0)
    epsilon = Fraction(epsilon)
    for clock in logical.clocks:
        dt = number(clock.dt)
        ticks = clock.start_tick + clock.steps
        clock_exact = Fraction(dt) / Fraction(owner_dt)
        clock_ratio = clock_exact.limit_denominator(1_000_000)
        common_denominator = math.lcm(common_denominator, clock_ratio.denominator)
        source_rounding = rounding_bound(dt, ticks)
        owner_rounding = rounding_bound(owner_dt, max(
            owner_maximum, ticks*clock_ratio.numerator//clock_ratio.denominator+1))
        if not math.isfinite(source_rounding+owner_rounding):
            error = math.inf
            break
        bound = (abs(clock_exact-clock_ratio) * ticks * Fraction(owner_dt)
                 + Fraction(source_rounding) + Fraction(owner_rounding))
        error = max(error, bound)
    if (error > epsilon/4
            or epsilon+2*error >= Fraction(owner_dt)/common_denominator
            or maximum*numerator+denominator-1 >= 2**63):
        raise PlanValidationError(
            'GPU owner-clock time cannot be certified over this activation; '
            'use a shorter activation or an exact commensurate clock grid')
    raw = f'((tick*{numerator}L+{denominator-1}L)/{denominator}L)'
    return (f'({raw} < {owner.start_tick}L ? {owner.start_tick}L : '
            f'({raw} > {owner_maximum}L ? {owner_maximum}L : {raw}))')
