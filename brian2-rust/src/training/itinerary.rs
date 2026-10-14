//! Admitted, replayable action visits. Output frames remain primary-clock ticks.
use super::*;

#[derive(Serialize)]
pub struct Events {
    pub clock_times: Vec<Vec<f64>>,
    pub neuron_clocks: Vec<usize>,
    pub frame_indices: Vec<usize>,
    pub spikes: Vec<Vec<Vec<f64>>>,
}

/// Count every actual emission. A leading visit before the first primary
/// clock belongs to frame zero; later visits belong to the preceding frame.
pub(super) fn outputs(spec: &dynamic::Spec, tape: &Tape, flat: &[f64], batch: usize, frames: usize)
    -> (Vec<Vec<Vec<f64>>>, Option<Events>) {
    let neurons=spec.voltage.len(); let mut frame=0;
    let indices:Vec<_>=tape.frames.iter().map(|f| {if let Some(f)=f {frame=*f;} frame}).collect();
    let mut counts=vec![vec![vec![0.0;neurons];frames];batch];
    for b in 0..batch {for (v,&f) in indices.iter().enumerate() {for n in 0..neurons {
        counts[b][f][n]+=flat[(b*tape.len()+v)*neurons+n];
    }}}
    let events=if spec.spike_buffers.is_empty() {None} else {
        let mut neuron_clocks=vec![0;neurons];
        for a in &spec.actions {if let Some(n)=a.threshold {neuron_clocks[n]=a.clock.unwrap_or(0);}}
        Some(Events {clock_times:tape.times.chunks_exact(tape.width).map(|r|r.to_vec()).collect(),
            neuron_clocks,frame_indices:indices,
            spikes:flat.chunks_exact(tape.len()*neurons).map(|s|s.chunks_exact(neurons).map(|r|r.to_vec()).collect()).collect()})
    };
    (counts,events)
}

pub(super) struct Tape {
    pub width: usize,
    pub times: Vec<f64>,
    pub calls: Vec<u64>,
    pub active: Vec<bool>,
    pub frames: Vec<Option<usize>>,
    pub checkpoint: Option<clock_state::Checkpoint>,
}
impl Tape {
    pub fn len(&self) -> usize { self.frames.len() }
    pub fn enabled(&self, visit: usize, action: &dynamic::Action) -> bool {
        self.active[visit*self.width+action.clock.unwrap_or(0)]
    }
    pub fn row(&self, visit: usize) -> &[f64] { &self.times[visit*self.width..(visit+1)*self.width] }
    pub fn tick(&self, visit: usize, action: &dynamic::Action) -> u64 {
        self.calls[visit*self.width+action.clock.unwrap_or(0)]
    }
    pub fn bytes(width: usize, visits: usize) -> Option<usize> {
        width.checked_mul(24)?.checked_add(16)?.checked_mul(visits)?.checked_add(width.checked_mul(512)?)
    }
}

fn asynchronous(spec: &dynamic::Spec) -> bool { spec.actions.iter().any(|a| a.clock.unwrap_or(0)!=0) }

/// Stream and check the whole run before allocating its tape. A manual async
/// start without saved history reconstructs one canonical preceding run.
fn walk(spec: &dynamic::Spec, tick: u64, time: usize, saved: Option<&clock_state::Checkpoint>,
    mut accept: impl FnMut(clock::Visit<'_>) -> std::result::Result<(), &'static str>)
    -> Result<clock_state::Checkpoint> {
    let clocks = spec.clocks.as_ref().ok_or("async actions require clock schedule")?;
    let schedule = clocks.schedule();
    let base = schedule.tick_at(clocks.start, clocks.dts[0])?;
    let end = tick.checked_add(time as u64).ok_or("async clock tick overflow")?;
    let end_main = base.checked_add(end).filter(|&v| v < (1u64<<53)).ok_or("async clock tick overflow")?;
    let mut cursor = if let Some(saved) = saved { clocks.restored(tick,saved)? } else { schedule.cursor()? };
    if tick!=0 {
        let boundary = (base+tick) as f64*clocks.dts[0];
        if saved.is_none() { cursor.run_to_limits(&schedule.limits(boundary)?, |_| Ok(()))?; }
        cursor.restart(boundary)?;
    }
    let mut used = vec![false;clocks.dts.len()]; used[0]=true;
    for action in &spec.actions { used[action.clock.unwrap_or(0)]=true; }
    cursor.run_to_limits(&schedule.limits(end_main as f64*clocks.dts[0])?, |v| {
        if v.active.iter().any(|&k|used[k]) { accept(v)?; }
        Ok(())
    })?;
    let state=cursor.snapshot();
    ensure(state.calls[0]==end, "async interval main tick count mismatch")?;
    Ok(clock_state::Checkpoint::from_state(end,state))
}

pub(super) fn count(spec: &dynamic::Spec, tick:u64,time:usize,saved:Option<&clock_state::Checkpoint>) -> Result<usize> {
    if !asynchronous(spec) { return Ok(time); }
    let mut count=0;
    walk(spec,tick,time,saved, |_|{count+=1;Ok(())})?;
    Ok(count)
}

pub(super) fn build(plan:&Plan,tick:u64,time:usize,saved:Option<&clock_state::Checkpoint>,visits:usize) -> Result<Tape> {
    let spec=plan.dynamic.as_ref().unwrap();let width=spec.clocks.as_ref().map_or(1,|s|s.dts.len());
    let mut tape=Tape { width,times:Vec::with_capacity(visits*width),calls:Vec::with_capacity(visits*width),
        active:Vec::with_capacity(visits*width),frames:Vec::with_capacity(visits),checkpoint:None };
    if asynchronous(spec) {
        tape.checkpoint=Some(walk(spec,tick,time,saved, |v| {
            if tape.frames.len()>=visits {return Err("async visit admission mismatch");}
            tape.frames.push(if v.active.contains(&0) {Some((v.calls[0]-tick) as usize)} else {None});
            tape.times.extend_from_slice(v.times);tape.calls.extend_from_slice(v.calls);
            for k in 0..width {tape.active.push(v.active.contains(&k));}
            Ok(())
        })?);
    } else {
        if let Some(clocks)=&spec.clocks {
            let (rows,end)=clocks.run_samples(tick,time,saved)?;tape.times=rows;tape.checkpoint=Some(end);
        } else {tape.times.extend((0..time).map(|i|plan.time_at(tick+i as u64)));}
        for i in 0..time {tape.frames.push(Some(i));for k in 0..width {tape.active.push(k==0);tape.calls.push(tick+i as u64);}}
    }
    ensure(tape.len()==visits && tape.frames.iter().flatten().copied().eq(0..time), "async frame/visit admission mismatch")?;
    Ok(tape)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn spec() -> dynamic::Spec {
        serde_json::from_value(serde_json::json!({
            "clocks":{"start":0.0,"dts":[0.0002,0.00015,0.00007],"epsilon":0.0001},
            "initial":[0.0],"initial_parameters":[null],"detached":[false],"voltage":[0],"program_sets":[],
            "actions":[{"clock":1,"owner":0,"reads":[0],"writes":[],"program_set":null,"threshold":null,"trigger":null}]
        })).unwrap()
    }
    #[test]
    fn visits_include_intermediate_actions_and_ignore_passive_only_clocks() {
        let s=spec();let mut rows=Vec::new();
        let end=walk(&s,0,4,None,|v| {rows.push((v.calls[0],v.active.to_vec()));Ok(())}).unwrap();
        assert_eq!(rows.len(),8);assert_eq!(count(&s,0,4,None).unwrap(),8);
        assert_eq!(&end.calls[..2], &[4,6]);
        assert_eq!(rows.iter().enumerate().filter(|(_,r)|r.1.contains(&0)).map(|(k,_)|k).collect::<Vec<_>>(),vec![0,2,4,6]);
    }
    #[test]
    fn async_restart_and_manual_prefix_keep_clock_draw_counts() {
        let s=spec();let first=walk(&s,0,4,None,|_|Ok(())).unwrap();
        let mut resumed=Vec::new();let mut manual=Vec::new();
        let a=walk(&s,4,2,Some(&first),|v| {resumed.push((v.calls.to_vec(),v.times.to_vec(),v.active.to_vec()));Ok(())}).unwrap();
        let c=walk(&s,4,2,None,|v| {manual.push((v.calls.to_vec(),v.times.to_vec(),v.active.to_vec()));Ok(())}).unwrap();
        assert_eq!(resumed,manual);assert_eq!(a,c);assert_eq!(&a.calls[..2],&[6,8]);
    }
    #[test]
    fn visit_tape_budget_uses_checked_arithmetic() {
        assert!(Tape::bytes(256,usize::MAX).is_none());
        assert_eq!(Tape::bytes(2,8),Some(1536));
    }
    #[test]
    fn event_outputs_keep_leading_visits_and_multiple_emissions() {
        let mut s=spec();s.spike_buffers=vec![1];s.actions[0].threshold=Some(0);
        let tape=Tape {width:2,times:vec![0.0;10],calls:vec![0;10],active:vec![true;10],
            frames:vec![None,Some(0),None,Some(1),None],checkpoint:None};
        let (counts,events)=outputs(&s,&tape,&[1.,0.,1.,0.,1.],1,2);
        assert_eq!(counts,vec![vec![vec![2.],vec![1.]]]);
        let events=events.unwrap();assert_eq!(events.neuron_clocks,vec![1]);
        assert_eq!(events.frame_indices,vec![0,0,0,1,1]);
        assert_eq!(events.spikes[0].iter().flatten().sum::<f64>(),3.);
    }
}
