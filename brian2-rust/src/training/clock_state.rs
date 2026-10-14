//! Serializable run-boundary cursor. Device kernels consume the same restored
//! host clock rows as CPU; checkpoints never reconstruct a supplied history.
use super::*;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Checkpoint {
    pub next_tick: u64,
    pub start: f64,
    pub initial_calls: Vec<u64>,
    pub calls: Vec<u64>,
    pub ticks: Vec<u64>,
    pub visits: u64,
}
impl Checkpoint {
    fn state(&self) -> clock::State {
        clock::State { start: self.start, initial_calls: self.initial_calls.clone(),
            calls: self.calls.clone(), ticks: self.ticks.clone(), visits: self.visits }
    }
    pub(super) fn from_state(next_tick: u64, state: clock::State) -> Self {
        Self { next_tick, start: state.start, initial_calls: state.initial_calls,
            calls: state.calls, ticks: state.ticks, visits: state.visits }
    }
}

impl dynamic::ClockSchedule {
    pub(super) fn restored(&self, tick: u64, saved: &Checkpoint) -> Result<clock::Cursor<'_>> {
        let s = self.schedule();
        s.validate()?;
        let n = self.dts.len();
        ensure(saved.next_tick == tick && saved.calls.len() == n && saved.ticks.len() == n
            && saved.initial_calls.len() == n && saved.calls[0] == tick,
            "dynamic clock checkpoint tick/width mismatch")?;
        let base = s.tick_at(self.start, self.dts[0])?;
        let end_main = base.checked_add(tick).filter(|&v| v < (1u64 << 53))
            .ok_or("dynamic clock checkpoint tick overflow")?;
        ensure(saved.ticks[0] == end_main
            && s.tick_at(saved.start, self.dts[0])?.checked_sub(base) == Some(saved.initial_calls[0])
            && saved.initial_calls.iter().zip(&saved.calls).all(|(&a,&b)| a <= b),
            "dynamic clock checkpoint counters differ from boundary")?;
        let cursor = s.restore(&saved.state())?;
        let limits = s.limits(end_main as f64 * self.dts[0])?;
        ensure(!cursor.before(&limits)?, "dynamic clock checkpoint is not an interval boundary")?;
        // A completed run cannot have executed beyond any clock's limit.
        ensure(saved.ticks.iter().zip(&limits).all(|(a,b)| a <= b),
            "dynamic clock checkpoint exceeds interval boundary")?;
        Ok(cursor)
    }

    pub(super) fn checkpoint_times(&self, tick: u64, saved: &Checkpoint) -> Result<Vec<f64>> {
        Ok(self.restored(tick, saved)?.visit().times.to_vec())
    }

    /// Without a checkpoint, retain historical continuous-prefix sampling.
    /// With one, start a new Brian run at the committed primary boundary,
    /// realigning pending ticks while retaining every clock's execution count.
    pub(super) fn run_samples(&self, tick: u64, time: usize, saved: Option<&Checkpoint>)
        -> Result<(Vec<f64>, Checkpoint)> {
        let s = self.schedule();
        let mut cursor = if let Some(saved) = saved { self.restored(tick, saved)? } else { s.cursor()? };
        let base = s.tick_at(self.start, self.dts[0])?;
        let end = tick.checked_add(time as u64).ok_or("dynamic clock tick overflow")?;
        let end_main = base.checked_add(end).filter(|&v| v < (1u64 << 53))
            .ok_or("dynamic clock tick overflow")?;
        if saved.is_some() && tick != 0 { cursor.restart((base + tick) as f64 * self.dts[0])?; }
        let limits = s.limits(end_main as f64 * self.dts[0])?;
        let mut rows = Vec::with_capacity(time.checked_mul(self.dts.len()).ok_or("clock table overflow")?);
        cursor.run_to_limits(&limits, |v| {
            if v.active.contains(&0) && v.calls[0] >= tick { rows.extend_from_slice(v.times); }
            Ok(())
        })?;
        let state = cursor.snapshot();
        ensure(state.calls[0] == end && rows.len() == time * self.dts.len(),
            "dynamic clock interval does not contain requested main ticks")?;
        Ok((rows, Checkpoint::from_state(end, state)))
    }
}

pub(super) fn boundary(plan: &Plan, tick: u64, saved: Option<&Checkpoint>) -> Result<Option<Vec<f64>>> {
    if let Some(saved) = saved {
        let clocks = plan.dynamic.as_ref().and_then(|s| s.clocks.as_ref())
            .ok_or("clock_state requires a dynamic clock schedule")?;
        ensure(clocks.dts.len() * 256 <= plan.max_tape_bytes, "clock checkpoint memory budget exceeded")?;
        Ok(Some(clocks.checkpoint_times(tick, saved)?))
    } else { Ok(None) }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn clocks(start: f64) -> dynamic::ClockSchedule {
        dynamic::ClockSchedule { start, dts: vec![0.0002,0.00015,0.00033], epsilon: 1e-4, order: vec![0,1,2] }
    }
    #[test]
    fn cold_warm_restart_preserves_execution_counts() {
        for start in [0.0,0.0003,0.000650001] {
            let s = clocks(start);
            let (whole, end) = s.run_samples(0, 6, None).unwrap();
            let (head, first) = s.run_samples(0, 2, None).unwrap();
            let (tail, last) = s.run_samples(2, 4, Some(&first)).unwrap();
            assert_eq!([head,tail].concat(), whole);
            assert_eq!(last.calls, end.calls);
            assert_eq!(last.ticks, end.ticks);
            assert_eq!(last.initial_calls, first.calls);
            assert_eq!(s.restored(6, &last).unwrap().snapshot(), last.state());
        }
    }
    #[test]
    fn checkpoint_must_be_at_the_requested_boundary() {
        let s = clocks(0.0); let (_, original) = s.run_samples(0, 4, None).unwrap();
        assert!(s.restored(3, &original).is_err());
        let mut bad = original.clone(); bad.ticks[1] += 1;
        assert!(s.restored(4, &bad).is_err());
        let mut bad = original.clone(); bad.initial_calls[0] += 1;
        assert!(s.restored(4, &bad).is_err());
        let mut bad = original; bad.visits = clock::MAX_WORK + 1;
        assert!(s.restored(4, &bad).is_err());
    }
    #[test]
    fn overrun_interval_does_not_produce_a_checkpoint() {
        let s = dynamic::ClockSchedule { start:0.0, dts:vec![0.0002,0.00099995,0.00049996], epsilon:1e-4, order:vec![] };
        let (_, saved) = s.run_samples(0, 2, None).unwrap();
        let before = saved.clone();
        assert!(s.run_samples(2, 3, Some(&saved)).is_err());
        assert_eq!(before, saved);
    }
}
