//! Bounded Brian clock visits, independent of batches, devices and MPI ranks.
//! A visit includes every pending clock value, but only `active` clocks execute.
//! Exact ties use the supplied Brian clock iteration order: epsilon coalescing
//! is not transitive when clocks have different dt values.
type Result<T> = std::result::Result<T, &'static str>;
const MAX_TICK: u64 = 1 << 53;
pub(super) const MAX_WORK: u64 = 10_000_000;

#[derive(Clone, Copy)]
pub(super) struct Schedule<'a> {
    pub start: f64,
    pub dts: &'a [f64],
    pub epsilon: f64,
    /// Empty means the historical array order, for old serialized plans.
    pub order: &'a [usize],
}
#[derive(Clone, Debug, PartialEq)]
pub(super) struct State {
    pub start: f64,
    pub initial_calls: Vec<u64>,
    pub calls: Vec<u64>,
    pub ticks: Vec<u64>,
    pub visits: u64,
}
#[derive(Clone, Copy, Debug)]
pub(super) struct Visit<'a> {
    pub time: f64,
    pub ticks: &'a [u64],
    pub calls: &'a [u64],
    pub times: &'a [f64],
    pub active: &'a [usize],
}
pub(super) struct Cursor<'a> {
    schedule: Schedule<'a>,
    state: State,
    times: Vec<f64>,
    active: Vec<usize>,
    first: usize,
}
impl<'a> Schedule<'a> {
    pub fn validate(&self) -> Result<()> {
        if !self.start.is_finite() || self.start < 0.0 || !self.epsilon.is_finite()
            || self.epsilon <= 0.0 || self.epsilon >= 0.5 || self.dts.is_empty()
            || self.dts.len() > 256 || self.dts.iter().any(|dt| !dt.is_finite() || *dt <= 0.0) {
            return Err("invalid dynamic clock schedule");
        }
        if !self.order.is_empty() {
            let mut seen = vec![false; self.dts.len()];
            if self.order.len() != seen.len() { return Err("invalid dynamic clock tie order"); }
            for &i in self.order {
                if i >= seen.len() || seen[i] { return Err("invalid dynamic clock tie order"); }
                seen[i] = true;
            }
        }
        for &dt in self.dts { self.tick_at(self.start, dt)?; }
        Ok(())
    }
    pub fn tick_at(&self, target: f64, dt: f64) -> Result<u64> {
        let q = target / dt;
        if !dt.is_finite() || dt <= 0.0 || !target.is_finite() || target < 0.0 || !q.is_finite() || q >= MAX_TICK as f64 {
            return Err("dynamic clock tick exceeds precision budget");
        }
        let near = q.round_ties_even();
        let tick = if (near * dt - target).abs() / dt <= self.epsilon { near } else { q.ceil() };
        if tick >= MAX_TICK as f64 || !(tick * dt).is_finite() { return Err("dynamic clock tick exceeds precision budget"); }
        Ok(tick as u64)
    }
    pub fn cursor(self) -> Result<Cursor<'a>> {
        self.validate()?;
        let ticks = self.dts.iter().map(|&dt| self.tick_at(self.start, dt)).collect::<Result<Vec<_>>>()?;
        let times = ticks.iter().zip(self.dts).map(|(&t, &dt)| t as f64 * dt).collect();
        let mut cursor = Cursor { schedule: self, state: State { start: self.start, initial_calls: vec![0; ticks.len()], calls: vec![0; ticks.len()], ticks, visits: 0 }, times, active: Vec::new(), first: 0 };
        cursor.select();
        Ok(cursor)
    }
    /// Validate external checkpoint ticks by bounded deterministic replay.
    pub fn restore(self, state: &State) -> Result<Cursor<'a>> {
        if state.visits > MAX_WORK / self.dts.len().max(1) as u64 {
            return Err("dynamic clock replay exceeds work budget");
        }
        let mut cursor = self.cursor()?;
        if state.initial_calls.len() != self.dts.len() { return Err("dynamic clock checkpoint width mismatch"); }
        cursor.state.calls.clone_from(&state.initial_calls);
        cursor.restart(state.start)?;
        for _ in 0..state.visits { cursor.advance()?; }
        if cursor.state != *state { return Err("dynamic clock checkpoint differs from schedule"); }
        Ok(cursor)
    }
    pub fn limits(&self, end: f64) -> Result<Vec<u64>> {
        self.dts.iter().map(|&dt| self.tick_at(end, dt)).collect()
    }
}
impl Cursor<'_> {
    fn select(&mut self) {
        self.first = self.schedule.order.first().copied().unwrap_or(0);
        if self.schedule.order.is_empty() {
            for i in 0..self.times.len() { if self.times[i] < self.times[self.first] { self.first = i; } }
        } else {
            for &i in self.schedule.order { if self.times[i] < self.times[self.first] { self.first = i; } }
        }
        let time = self.times[self.first];
        let dt = self.schedule.dts[self.first];
        self.active.clear();
        for i in 0..self.times.len() {
            if (self.times[i] - time).abs() / self.schedule.dts[i].min(dt) < self.schedule.epsilon {
                self.active.push(i);
            }
        }
    }
    pub fn visit(&self) -> Visit<'_> {
        Visit { time: self.times[self.first], ticks: &self.state.ticks, calls: &self.state.calls, times: &self.times, active: &self.active }
    }
    pub fn snapshot(&self) -> State { self.state.clone() }
    /// Realign pending ticks as Network.run does at a new interval. Keep
    /// execution counts separate from clock ticks, including for explicit
    /// realignment. Interval admission still rejects out-of-range coalescing.
    pub fn restart(&mut self, start: f64) -> Result<()> {
        if start < self.schedule.start { return Err("dynamic clock restart precedes snapshot"); }
        let ticks = self.schedule.dts.iter().map(|&dt| self.schedule.tick_at(start, dt)).collect::<Result<Vec<_>>>()?;
        let times = ticks.iter().zip(self.schedule.dts).map(|(&t, &dt)| t as f64 * dt).collect();
        self.state = State { start, initial_calls: self.state.calls.clone(), calls: self.state.calls.clone(), ticks, visits: 0 };
        self.times = times; self.select();
        Ok(())
    }
    pub fn before(&self, limits: &[u64]) -> Result<bool> {
        if limits.len() != self.times.len() { return Err("dynamic clock interval width mismatch"); }
        // Brian tests only the minimum clock's interval end, not every active
        // clock independently. The latter drops near-boundary coalesced visits.
        Ok(self.state.ticks[self.first] < limits[self.first])
    }
    pub fn advance(&mut self) -> Result<()> {
        let visits = self.state.visits.checked_add(1).ok_or("dynamic clock visit overflow")?;
        if visits > MAX_WORK / self.times.len() as u64 { return Err("dynamic clock replay exceeds work budget"); }
        // Check every active clock before committing any part of this visit.
        for &i in &self.active {
            let tick = self.state.ticks[i] + 1;
            let next = tick as f64 * self.schedule.dts[i];
            if tick >= MAX_TICK || !next.is_finite() || next <= self.times[i] || self.state.calls[i] == u64::MAX {
                return Err("dynamic clock time overflow or insufficient precision");
            }
        }
        for &i in &self.active {
            self.state.ticks[i] += 1;
            self.state.calls[i] += 1;
            self.times[i] = self.state.ticks[i] as f64 * self.schedule.dts[i];
        }
        self.state.visits = visits;
        self.select();
        Ok(())
    }
    /// Stream complete visits without allocating a tape proportional to all
    /// clock ticks. Consumers admit their own action/tape memory before use.
    pub fn run_until(&mut self, end: f64, visit: impl FnMut(Visit<'_>) -> Result<()>) -> Result<()> {
        if end < self.state.start { return Err("dynamic clock interval ends before its start"); }
        let limits = self.schedule.limits(end)?;
        self.run_to_limits(&limits, visit)
    }
    pub fn run_to_limits(&mut self, limits: &[u64], mut visit: impl FnMut(Visit<'_>) -> Result<()>) -> Result<()> {
        while self.before(limits)? {
            // Network selects by the earliest clock, but Clock.advance rejects
            // every active clock that would pass its own interval limit. Admit
            // the visit before invoking actions, so a caller can fail safely.
            if self.active.iter().any(|&i| self.state.ticks[i] >= limits[i]) {
                return Err("dynamic clock coalescing exceeds interval end");
            }
            visit(self.visit())?;
            self.advance()?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn schedule(dts: &[f64]) -> Schedule<'_> { Schedule { start: 0.0, dts, epsilon: 1e-4, order: &[] } }
    #[test]
    fn fast_slow_visits_and_restore() {
        let s = schedule(&[0.2, 0.3, 0.1]);
        let mut c = s.cursor().unwrap(); let mut counts = [0; 3];
        c.run_until(0.6, |v| { for &i in v.active { counts[i] += 1; } Ok(()) }).unwrap();
        assert_eq!(counts, [3, 2, 6]);
        let state = c.snapshot(); let mut restored = s.restore(&state).unwrap();
        c.run_until(1.2, |_| Ok(())).unwrap(); restored.run_until(1.2, |_| Ok(())).unwrap();
        assert_eq!(c.snapshot(), restored.snapshot());
        let mut bad = state; bad.ticks[1] += 1; assert!(s.restore(&bad).is_err());
    }
    #[test]
    fn exact_tie_priority_changes_nontransitive_coalescing() {
        let mut s = schedule(&[0.2, 0.4, 0.40003]); s.start = 0.4;
        let a = s.cursor().unwrap(); assert_eq!(a.visit().active, [0, 1]);
        s.order = &[1, 0, 2]; let b = s.cursor().unwrap(); assert_eq!(b.visit().active, [0, 1, 2]);
    }
    #[test]
    fn invalid_orders_and_clock_precision() {
        let mut s = schedule(&[0.2, 0.4]); s.order = &[0, 0]; assert!(s.cursor().is_err());
        s.order = &[1]; assert!(s.cursor().is_err());
        s.order = &[]; s.start = f64::NAN; assert!(s.cursor().is_err());
        s.start = (MAX_TICK - 1) as f64 * 0.2;
        let mut c = s.cursor().unwrap(); let old = c.snapshot();
        assert!(c.advance().is_err()); assert_eq!(old, c.snapshot());
    }
    #[test]
    fn restart_preserves_counts_when_tolerance_repeats_a_tick() {
        let s = schedule(&[0.0002, 0.00099995, 0.00049996]); let mut c = s.cursor().unwrap();
        // An unbounded stream can be explicitly realigned; a Brian interval
        // ending at this boundary must instead reject the overrun below.
        while c.visit().time < 0.001 { c.advance().unwrap(); }
        let before = c.snapshot(); assert_eq!(before.ticks[1], 2);
        c.restart(0.001).unwrap(); let at = c.snapshot();
        assert_eq!(at.ticks[1], 1); assert_eq!(at.calls, before.calls);
        c.advance().unwrap(); assert_eq!(c.snapshot().calls[1], before.calls[1] + 1);
        assert_eq!(s.restore(&c.snapshot()).unwrap().snapshot(), c.snapshot());
    }
    #[test]
    fn coalescing_past_a_clock_interval_is_rejected() {
        let s = schedule(&[0.0002, 0.00099995, 0.00049996]);
        let mut c = s.cursor().unwrap();
        assert_eq!(c.run_until(0.001, |_| Ok(())), Err("dynamic clock coalescing exceeds interval end"));
        // The rejected visit did not increment the exhausted clock.
        assert_eq!(c.snapshot().ticks[1], 1);
    }
    #[test]
    fn work_budget_and_counter_failure_are_atomic() {
        let s = schedule(&[0.2]); let mut c = s.cursor().unwrap();
        c.state.visits = MAX_WORK; let old = c.snapshot();
        assert!(c.advance().is_err()); assert_eq!(old, c.snapshot());
        c.state.visits = 0; c.state.calls[0] = u64::MAX; let old = c.snapshot();
        assert!(c.advance().is_err()); assert_eq!(old, c.snapshot());
        assert!(schedule(&[1e308]).tick_at(1.7e308, 1e308).is_err());
    }
    #[test]
    fn strict_coalescing_and_inclusive_rounding_differ() {
        let mut s = schedule(&[1.0, 2.0]); s.start = 1.0; s.epsilon = 0.25;
        assert_eq!(s.tick_at(1.25, 1.0).unwrap(), 1);
        // Rounding uses <=; coalescing uses <. Use binary-exact values.
        s.dts = &[1.0, 1.25]; assert_eq!(s.cursor().unwrap().visit().active, [0]);
    }
}
