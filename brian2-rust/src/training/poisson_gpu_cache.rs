//! Device cache layout. Host interns identities, never samples counts.
use super::*;
use std::collections::HashMap;
pub(super) struct Layout {
    ids:HashMap<poisson_cache::Identity,usize>,
    lanes:Vec<Vec<poisson_cache::Identity>>,
    pub ordinary:usize, pub per_visit:usize, pub width:usize, pub host_bytes:usize,
}
impl Layout {
    pub fn build(plan:&Plan,itinerary:&itinerary::Tape,ordinary:usize,batch:usize,state:Option<&poisson_cache::Checkpoint>)->Result<Self>{
        let spec=plan.dynamic.as_ref().unwrap();let mut streams=Vec::with_capacity(spec.actions.len());
        let mut per_step=0usize;
        for a in &spec.actions {
            let mut mask=0u16;
            if let Some(programs)=a.program_set.map(|i|&spec.program_sets[i]) {
                for node in programs.iter().flatten(){if let equation::Node::Poisson{stream,..}=node{mask|=1<<stream;}}
            }
            per_step=per_step.checked_add(mask.count_ones() as usize).ok_or("GPU Poisson cache size overflow")?;streams.push(mask);
        }
        let maximum=itinerary.len().checked_mul(per_step).and_then(|n|n.checked_mul(batch))
            .and_then(|n|n.checked_add(state.map_or(0,|s|s.entries.len()))).ok_or("GPU Poisson cache size overflow")?;
        let host_bytes=maximum.checked_mul(768).and_then(|x|x.checked_add(streams.len()*2+batch*24+4096)).ok_or("GPU Poisson cache memory overflow")?;
        ensure(host_bytes<=plan.max_tape_bytes,"GPU Poisson identity cache budget exceeded")?;
        let mut ids=HashMap::new();let mut lanes=vec![Vec::new();batch];
        if let Some(state)=state {for entry in &state.entries {
            let lane=&mut lanes[entry.identity.batch];ids.insert(entry.identity,lane.len());lane.push(entry.identity);
        }}
        for (batch,lane) in lanes.iter_mut().enumerate(){for visit in 0..itinerary.len(){for (a,&mask) in spec.actions.iter().zip(&streams){
            if !itinerary.enabled(visit,a){continue;}
            for stream in 0..16 {if mask&(1<<stream)!=0 {
                let id=poisson_cache::identity(a,stream,batch,itinerary.tick(visit,a));
                if let std::collections::hash_map::Entry::Vacant(entry)=ids.entry(id){entry.insert(lane.len());lane.push(id);}
            }}
        }}}
        let per_visit=lanes.iter().map(Vec::len).max().unwrap_or(0).div_ceil(itinerary.len());
        let width=ordinary.checked_add(per_visit.checked_mul(4).ok_or("GPU Poisson cache width overflow")?).ok_or("GPU Poisson cache width overflow")?;
        Ok(Self{ids,lanes,ordinary,per_visit,width,host_bytes})
    }
    pub fn identity_offset(&self,id:poisson_cache::Identity,visits:usize)->Result<u32>{
        let Some(&slot)=self.ids.get(&id) else{return Ok(0)};
        let offset=id.batch.checked_mul(visits).and_then(|x|x.checked_mul(self.width))
            .and_then(|x|x.checked_add((slot/self.per_visit)*self.width+self.ordinary+4*(slot%self.per_visit)))
            .ok_or("GPU Poisson cache offset overflow")?;
        u32::try_from(offset).map_err(|_|"GPU Poisson cache offset outside uint32".into())
    }
    pub fn offset(&self,a:&dynamic::Action,stream:usize,visit:usize,batch:usize,itinerary:&itinerary::Tape)->Result<u32>{
        self.identity_offset(poisson_cache::identity(a,stream,batch,itinerary.tick(visit,a)),itinerary.len())
    }
    pub fn checkpoint(&self,plan:&Plan,sequence:u64,tape:&[f32],visits:usize)->Result<poisson_cache::Checkpoint>{
        let mut entries=Vec::new();
        for &id in self.lanes.iter().flatten(){
            let at=self.identity_offset(id,visits)? as usize;
            ensure(tape[at]==0.||tape[at]==1.,"invalid GPU Poisson cache valid word")?;
            if tape[at]==0.{continue;}
            let count=tape[at+1].to_bits() as i32;let rate=tape[at+2] as f64;
            ensure(count>=0&&rate.is_finite()&&rate>=0.&&rate<2147483648.,"invalid GPU Poisson cache output")?;
            entries.push(poisson_cache::Entry{identity:id,count,rate});
        }
        entries.sort_by_key(|e|e.identity);
        Ok(poisson_cache::Checkpoint{schema:"b2-poisson-draw-state-v2".into(),numeric_profile:Some(poisson_gpu_validate::profile(plan).into()),continuation:None,
            seed:plan.seed,sequence,batch:self.lanes.len(),entries})
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn tape()->itinerary::Tape {itinerary::Tape{width:1,times:vec![0.;3],calls:vec![0,1,2],active:vec![true;3],frames:vec![Some(0),Some(1),Some(2)],checkpoint:None}}
    fn a(delay:u64,pending:Option<u64>)->dynamic::Action {
        dynamic::Action{noise_domain:71,noise_streams:1,event_noise:Some(noise::EventAddress{delay,pending}),..Default::default()}
    }
    #[test]
    fn emission_aliases_use_tail_slots_and_separate_batch_lanes(){
        let t=tape();let now=a(0,None);let old=a(1,None);
        let ids=(0..2).flat_map(|b|[(poisson_cache::identity(&now,0,b,0),0),(poisson_cache::identity(&old,0,b,0),1),
            (poisson_cache::identity(&now,0,b,1),2),(poisson_cache::identity(&now,0,b,2),3)]).collect();
        let c=Layout{ids,lanes:vec![],ordinary:10,per_visit:2,width:18,host_bytes:0};
        assert_eq!(c.offset(&now,0,0,0,&t).unwrap(),c.offset(&old,0,1,0,&t).unwrap());
        assert_eq!(c.offset(&now,0,0,1,&t).unwrap(),54+c.offset(&now,0,0,0,&t).unwrap());
        for visit in 0..3 {let at=c.offset(&now,0,visit,0,&t).unwrap() as usize;assert!(at%18>=10&&at%18+4<=18);}
    }
    #[test]
    fn pending_slots_recur_without_clock_or_stream_alias(){
        let t=tape();let pending=a(0,Some(73));let other=a(0,None);
        let ids=[(poisson_cache::identity(&pending,0,0,0),0),(poisson_cache::identity(&other,0,0,0),1)].into_iter().collect();
        let c=Layout{ids,lanes:vec![],ordinary:10,per_visit:1,width:14,host_bytes:0};
        assert_eq!(c.offset(&pending,0,0,0,&t).unwrap(),c.offset(&pending,0,2,0,&t).unwrap());
        assert_ne!(c.offset(&pending,0,0,0,&t).unwrap(),c.offset(&other,0,0,0,&t).unwrap());
        assert_eq!(c.offset(&pending,1,0,0,&t).unwrap(),0);
    }
    #[test]
    fn bit_packed_offsets_reject_uint32_overflow(){
        let t=tape();let pending=a(0,Some(73));let ids=[(poisson_cache::identity(&pending,0,u32::MAX as usize,0),0)].into_iter().collect();
        let c=Layout{ids,lanes:vec![],ordinary:10,per_visit:1,width:14,host_bytes:0};
        assert!(c.offset(&pending,0,0,u32::MAX as usize,&t).is_err());
    }
}
