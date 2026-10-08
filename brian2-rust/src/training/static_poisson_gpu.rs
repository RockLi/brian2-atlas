//! Static vector device observation layout. The host stages identities, not draws.
use super::*;
use std::collections::HashMap;

pub(super) struct Layout {
    ids:HashMap<poisson_cache::Identity,usize>,
    lanes:Vec<Vec<poisson_cache::Identity>>,
    pub base:usize, pub lane_words:usize, pub words:usize, pub host_bytes:usize,
}
impl Layout {
    pub fn build(plan:&Plan,start:u64,time:usize,batch:usize,base:usize,cache:Option<&poisson_cache::Checkpoint>)->Result<Self>{
        let masks=(0..plan.sizes.len()-1)
            .map(|l|static_poisson::programs(plan,l).flatten().fold(0u16,|mask,n|if let equation::Node::Poisson{stream,..}=n{mask|1<<stream}else{mask}))
            .collect::<Vec<_>>();
        let sites=masks.iter().zip(&plan.sizes[1..]).try_fold(0usize,|sum,(&mask,&n)|
            sum.checked_add(n.checked_mul(mask.count_ones() as usize)?)).ok_or("static GPU Poisson site overflow")?;
        let maximum=time.checked_mul(batch).and_then(|n|n.checked_mul(sites))
            .and_then(|n|n.checked_add(cache.map_or(0,|c|c.entries.len()))).ok_or("static GPU Poisson identity overflow")?;
        let lane_bytes=batch.checked_mul(24).and_then(|n|n.checked_add(4096)).ok_or("static GPU Poisson memory overflow")?;
        let host_bytes=maximum.checked_mul(768).and_then(|n|n.checked_add(lane_bytes)).ok_or("static GPU Poisson memory overflow")?;
        ensure(host_bytes<=plan.max_tape_bytes,"static GPU Poisson identity budget exceeded")?;
        let mut ids=HashMap::new();let mut lanes=vec![Vec::new();batch];
        if let Some(cache)=cache {for entry in &cache.entries {
            let lane=&mut lanes[entry.identity.batch];ids.insert(entry.identity,lane.len());lane.push(entry.identity);
        }}
        for (bi,lane) in lanes.iter_mut().enumerate(){for tick in 0..time {for (layer,&mask) in masks.iter().enumerate(){
            for j in 0..plan.sizes[layer+1] {for stream in 0..16 {if mask&(1<<stream)!=0 {
                let tick=start.checked_add(tick as u64).ok_or("static GPU Poisson clock overflow")?;
                let id=poisson_cache::identity(&static_poisson::action(plan,layer,j),stream,bi,tick);
                if let std::collections::hash_map::Entry::Vacant(entry)=ids.entry(id){entry.insert(lane.len());lane.push(id);}
            }}}
        }}}
        let lane_words=lanes.iter().map(Vec::len).max().unwrap_or(0).checked_mul(4).ok_or("static GPU Poisson cache overflow")?;
        let words=lane_words.checked_mul(batch).ok_or("static GPU Poisson cache overflow")?;
        ensure(base.checked_add(words).is_some_and(|n|n<=u32::MAX as usize),"static GPU Poisson uint32 cache address exceeded")?;
        Ok(Self{ids,lanes,base,lane_words,words,host_bytes})
    }
    pub fn offset(&self,id:poisson_cache::Identity)->u32 {
        self.ids.get(&id).map_or(0,|&slot|(self.base+id.batch*self.lane_words+slot*4) as u32)
    }
    pub fn checkpoint(&self,plan:&Plan,sequence:u64,tape:&[f32],mpi:Option<&mpi::Context>)->Result<poisson_cache::Checkpoint>{
        let mut values=Vec::with_capacity(self.lanes.iter().map(Vec::len).sum::<usize>()*3);
        let local=(||->Result<()>{for &id in self.lanes.iter().flatten(){
            let at=self.offset(id) as usize;let valid=tape[at];
            ensure(valid==0.||valid==1.,"invalid static GPU Poisson cache valid word")?;
            let count=tape[at+1].to_bits() as i32;let rate=tape[at+2];
            if valid==1. {ensure(count>=0&&rate.is_finite()&&rate>=0.&&rate<2147483648.,"invalid static GPU Poisson record")?;}
            values.extend_from_slice(&[valid as f64,if valid==1.{count as f64}else{0.},if valid==1.{rate.to_bits() as f64}else{0.}]);
        }Ok(())})();
        static_poisson::reconcile(local,mpi,true)?;
        if let Some(m)=mpi {m.sum(&mut values)?;}
        let mut entries=Vec::new();
        for (&id,row) in self.lanes.iter().flatten().zip(values.chunks_exact(3)){
            if row[0]==0.{continue;}
            // New records have one owner; imported detached records are present
            // on every rank. Average integer count/rate-bit payloads, never
            // resample or numerically sum rates: that would erase negative zero.
            ensure(row[0]==1.||row[0]==mpi.map_or(1,|m|m.size) as f64,"invalid static GPU Poisson ownership")?;
            let count=row[1]/row[0];let bits=row[2]/row[0];
            ensure(bits>=0.&&bits<=u32::MAX as f64&&bits==bits.floor(),"invalid static GPU Poisson rate bits")?;
            let rate=f32::from_bits(bits as u32) as f64;
            ensure(count>=0.&&count<=i32::MAX as f64&&count==count.floor()&&rate.is_finite()&&rate>=0.&&rate<2147483648.,
                "inconsistent static GPU Poisson cache")?;
            entries.push(poisson_cache::Entry{identity:id,count:count as i32,rate});
        }
        entries.sort_by_key(|e|e.identity);
        Ok(poisson_cache::Checkpoint{schema:"b2-poisson-draw-state-v2".into(),numeric_profile:Some(poisson_gpu_validate::profile(plan).into()),continuation:None,
            seed:plan.seed,sequence,batch:self.lanes.len(),entries})
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn plan()->Plan {
        serde_json::from_value(serde_json::json!({
            "schema":"b2-state-training-plan-v4","sizes":[1,1,2],"beta":[0.,0.],"threshold":[0.6,0.6],
            "reset":"subtract","detach_reset":false,"surrogate":{"kind":"fast_sigmoid","slope":1.,"scale":1.},
            "optimizer":{"kind":"adam","learning_rate":0.01,"beta1":0.9,"beta2":0.999,"epsilon":1e-8},
            "trainable":[],"masks":[],"seed":731,"logit_scale":1.7,"max_tape_bytes":1000000,"tbptt_window":null,
            "noise_streams":[1,1],"backend":"metal",
            "state_equations":[[[{"op":"constant","value":1.},{"op":"poisson","rate":0,"stream":0}]],
                               [[{"op":"constant","value":1.},{"op":"poisson","rate":0,"stream":0}]]],
            "state_resets":[[[{"op":"constant","value":1.},{"op":"poisson","rate":0,"stream":0}]],
                            [[{"op":"constant","value":1.},{"op":"poisson","rate":0,"stream":0}]]]
        })).unwrap()
    }
    fn id(p:&Plan,batch:usize,layer:usize,neuron:usize,tick:u64)->poisson_cache::Identity {
        poisson_cache::identity(&static_poisson::action(p,layer,neuron),0,batch,tick)
    }
    #[test]
    fn shared_phases_distinct_lanes_and_lossless_integer_payload() {
        let p=plan();let layout=Layout::build(&p,5,2,2,73,None).unwrap();
        assert_eq!(layout.lane_words,24);assert_eq!(layout.words,48);
        assert_eq!(layout.ids.len(),12); // update and reset share each identity
        assert_ne!(layout.offset(id(&p,0,0,0,5)),layout.offset(id(&p,1,0,0,5)));
        assert_eq!(layout.offset(id(&p,0,0,0,4)),0);
        let mut tape=vec![0.;layout.base+layout.words];let at=layout.offset(id(&p,1,1,1,6)) as usize;
        tape[at]=1.;tape[at+1]=f32::from_bits(1000000001);tape[at+2]=1e9;
        let state=layout.checkpoint(&p,9,&tape,None).unwrap();
        assert_eq!(state.entries.len(),1);assert_eq!(state.entries[0].count,1000000001);
        assert_eq!(state.numeric_profile.as_deref(),Some("native-metal-poisson-f32-v1"));
    }
    #[test]
    fn imported_history_is_preserved_and_lane_padding_is_separate() {
        let p=plan();let old=id(&p,1,0,0,3);
        let cache=poisson_cache::Checkpoint{schema:"b2-poisson-draw-state-v2".into(),numeric_profile:None,continuation:None,seed:731,sequence:9,batch:2,
            entries:vec![poisson_cache::Entry{identity:old,count:2,rate:1.}]};
        let layout=Layout::build(&p,5,1,2,32,Some(&cache)).unwrap();
        assert_eq!(layout.lanes[0].len(),3);assert_eq!(layout.lanes[1].len(),4);
        assert_eq!(layout.lane_words,16);assert_eq!(layout.offset(old),48);
        let mut tape=vec![0.;64];tape[48]=1.;tape[49]=f32::from_bits(2);tape[50]=1.;
        assert_eq!(layout.checkpoint(&p,9,&tape,None).unwrap().entries[0].identity,old);
    }
    #[test]
    fn exported_rate_keeps_ieee_negative_zero() {
        let p=plan();let layout=Layout::build(&p,0,1,1,17,None).unwrap();
        let mut tape=vec![0.;layout.base+layout.words];let at=layout.offset(id(&p,0,0,0,0)) as usize;
        tape[at]=1.;tape[at+1]=f32::from_bits(0);tape[at+2]=-0.;
        let cache=layout.checkpoint(&p,9,&tape,None).unwrap();
        assert_eq!(cache.entries[0].rate.to_bits(),(-0.0f64).to_bits());
    }
    #[test]
    fn budgets_addresses_and_clock_are_checked_before_dispatch() {
        let mut p=plan();p.max_tape_bytes=1;
        assert!(Layout::build(&p,0,1,2,32,None).is_err());p.max_tape_bytes=1000000;
        assert!(Layout::build(&p,0,1,1,u32::MAX as usize,None).is_err());
        assert!(Layout::build(&p,u64::MAX,2,1,32,None).is_err());
        assert!(Layout::build(&p,0,usize::MAX,2,32,None).is_err());
    }
}
