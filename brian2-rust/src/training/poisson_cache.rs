//! Scoped CPU draw identities. Cache hits are detached consumers; only the
//! first actual forward visit owns the saved rate and its likelihood VJP.
use super::*;
use std::cell::RefCell;
use std::collections::{BTreeSet, HashMap};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Site { pub domain:u64, pub entity:u64, pub stream:usize, pub kind:u8, pub pending:u64 }
impl Site {
    pub(super) fn new(a:&dynamic::Action,stream:usize)->Self {
        let (kind,pending)=a.event_noise.as_ref().map_or((0,0),|e|e.pending.map_or((1,0),|id|(2,id)));
        Self{domain:a.noise_domain,entity:a.noise_entity,stream,kind,pending}
    }
    pub(super) fn key(self,seed:u64,sequence:u64,batch:usize,instant:u64)->u64 {
        let event=match self.kind {0=>poisson::Event::Clock,1=>poisson::Event::Emission{delay:0},_=>poisson::Event::Pending{id:self.pending}};
        poisson::key(seed,sequence,batch as u64,self.domain,self.entity,instant,self.stream as u64,event)
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Identity { pub site:Site, pub batch:usize, pub instant:u64 }
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Entry { pub identity:Identity, pub count:i32, pub rate:f64 }
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Checkpoint {
    pub schema:String,
    #[serde(default,skip_serializing_if="Option::is_none")]
    pub numeric_profile:Option<String>,
    #[serde(default,skip_serializing_if="Option::is_none")]
    pub continuation:Option<super::poisson_retirement::Continuation>,
    pub seed:u64, pub sequence:u64, pub batch:usize, pub entries:Vec<Entry>
}
impl Checkpoint {
    pub(super) fn validate(&self,plan:&Plan,sequence:u64,batch:usize)->Result<usize> {
        let cpu=plan.backend=="cpu";
        let profile=if cpu {self.schema=="b2-poisson-draw-state-v1" && self.numeric_profile.is_none()}
            else {self.schema=="b2-poisson-draw-state-v2" && self.numeric_profile.as_deref()==Some(poisson_gpu_validate::profile(plan))};
        ensure(profile,"invalid Poisson draw checkpoint numerical profile")?;
        ensure(self.seed==plan.seed && self.sequence==sequence
            && self.batch==batch && batch>0 && batch<=4096,"invalid Poisson draw checkpoint header")?;
        if let Some(boundary)=&self.continuation {boundary.validate()?;}
        ensure(self.entries.len().checked_mul(768).is_some_and(|n|n<=plan.max_tape_bytes),"Poisson draw checkpoint budget exceeded")?;
        let mut seen=BTreeSet::new();let mut work=0u64;
        for e in &self.entries {
            let id=e.identity;
            ensure(id.batch<batch && id.site.stream<16 && id.site.kind<=2
                && (id.site.kind==2 || id.site.pending==0)
                && (id.site.kind!=2 || id.instant==id.site.pending)
                && e.count>=0 && e.rate.is_finite() && e.rate>=0. && e.rate<=i32::MAX as f64
                && seen.insert(id),"invalid or duplicated Poisson draw checkpoint entry")?;
            if !cpu {
                ensure(((e.rate as f32) as f64).to_bits()==e.rate.to_bits() && e.rate<2147483648.,"GPU Poisson checkpoint rate is not exact float32")?;
                continue;
            }
            let draw=poisson::sample(e.rate,id.site.key(self.seed,sequence,id.batch,id.instant))
                .map_err(|_|"invalid Poisson draw checkpoint sample")?;
            work=work.checked_add(draw.draws as u64).ok_or("Poisson draw checkpoint work overflow")?;
            ensure(work<=clock::MAX_WORK && draw.count==e.count,"Poisson draw checkpoint count mismatch or work budget")?;
        }
        if cpu {Ok(0)} else {poisson_gpu_validate::validate(plan,self)}
    }
}
#[derive(Clone, Copy, Debug, PartialEq)]
struct Point { batch:usize, visit:usize, action:usize }
#[derive(Clone)]
struct Draw { count:i32, rate:f64, origin:Option<Point> }
struct Runtime {
    seed:u64, sequence:u64, batch:usize, values:HashMap<Identity,Draw>,
    point:Point, action:dynamic::Action, tick:u64, forward:bool, store_new:bool,
    dirty:[bool;16], forced:Option<Identity>,
}
thread_local! { static RUNTIME:RefCell<Option<Runtime>>=const{RefCell::new(None)}; }
pub(super) struct Scope(Option<Runtime>);
impl Drop for Scope {fn drop(&mut self){RUNTIME.with(|r|{*r.borrow_mut()=self.0.take();});}}
pub(super) fn identity(a:&dynamic::Action,stream:usize,batch:usize,tick:u64)->Identity {
    let site=Site::new(a,stream);
    let instant=a.event_noise.as_ref().map_or(tick,|e|e.pending.unwrap_or_else(||tick.wrapping_sub(e.delay)));
    Identity{site,batch,instant}
}
pub(super) fn install(plan:&Plan,sequence:u64,batch:usize,state:Option<&Checkpoint>,forced:Option<Identity>)->Result<Scope> {
    let spec=plan.dynamic.as_ref().unwrap();
    let used=spec.program_sets.iter().flatten().flatten().any(|n|matches!(n,equation::Node::Poisson{..}));
    install_used(plan,sequence,batch,state,forced,used)
}
pub(super) fn install_static(plan:&Plan,sequence:u64,batch:usize,state:Option<&Checkpoint>,forced:Option<Identity>)->Result<Scope> {
    install_used(plan,sequence,batch,state,forced,static_poisson::used(plan))
}
fn install_used(plan:&Plan,sequence:u64,batch:usize,state:Option<&Checkpoint>,forced:Option<Identity>,used:bool)->Result<Scope> {
    let runtime=if used {
        let mut values=HashMap::new();
        if let Some(state)=state {for e in &state.entries {values.insert(e.identity,Draw{count:e.count,rate:e.rate,origin:None});}}
        Some(Runtime{seed:plan.seed,sequence,batch:state.map_or(batch,|s|s.batch),values,point:Point{batch:0,visit:0,action:0},
            action:dynamic::Action::default(),tick:0,forward:true,store_new:false,dirty:[false;16],forced})
    }else{None};
    Ok(Scope(RUNTIME.with(|r|r.replace(runtime))))
}
pub(super) fn begin(a:&dynamic::Action,batch:usize,visit:usize,index:usize,tick:u64,forward:bool,store_new:bool){
    RUNTIME.with(|r|{if let Some(r)=r.borrow_mut().as_mut(){
        r.point=Point{batch,visit,action:index};r.action=a.clone();r.tick=tick;
        r.forward=forward;r.store_new=store_new;r.dirty.fill(false);
    }});
}
fn current(r:&Runtime,stream:usize)->Identity {identity(&r.action,stream,r.point.batch,r.tick)}
pub(super) fn needs_rate(stream:usize)->bool {
    RUNTIME.with(|r|r.borrow().as_ref().is_none_or(|r|r.values.get(&current(r,stream))
        .is_none_or(|d| !r.forward && d.origin==Some(r.point))))
}
pub(super) fn owns_score(stream:usize)->bool {
    RUNTIME.with(|r|r.borrow().as_ref().is_none_or(|r|r.values.get(&current(r,stream))
        .is_some_and(|d|d.origin==Some(r.point))))
}
pub(super) fn sample(stream:usize,rate:f64,key:u64,forced:bool)->Result<i32> {
    RUNTIME.with(|cell|{
        let mut value=cell.borrow_mut();
        if let Some(r)=value.as_mut(){
            let id=current(r,stream);
            ensure(key==id.site.key(r.seed,r.sequence,id.batch,id.instant),"Poisson draw cache counter mismatch")?;
            if let Some(d)=r.values.get(&id){
                if !r.forward && d.origin==Some(r.point){ensure(rate==d.rate,"Poisson origin rate replay mismatch")?;}
                return Ok(d.count);
            }
            let count=draw(rate,key,forced||r.forced==Some(id))?;
            if r.forward&&r.store_new {r.values.insert(id,Draw{count,rate,origin:Some(r.point)});r.dirty[stream]=true;}
            return Ok(count);
        }
        draw(rate,key,forced)
    })
}
fn draw(rate:f64,key:u64,forced:bool)->Result<i32>{
    if forced {ensure(rate==0.,"Poisson boundary replay rate changed before forced site")?;Ok(1)}
    else {Ok(poisson::sample(rate,key).map_err(|e|format!("Poisson sample: {e:?}"))?.count)}
}
// Only the owner creates real draw records. Broadcast its fixed per-stream
// delta after the action, so subsequent owners see the same first observation.
pub(super) fn sync(mpi:Option<&mpi::Context>)->Result<()> {
    let Some(m)=mpi else{return Ok(())};
    let active=RUNTIME.with(|r|r.borrow().is_some());if !active{return Ok(())}
    let mut wire=[0.;48];
    RUNTIME.with(|cell|{let r=cell.borrow();let r=r.as_ref().unwrap();
        for s in 0..16 {if r.dirty[s]{let d=&r.values[&current(r,s)];wire[3*s]=1.;wire[3*s+1]=d.count as f64;wire[3*s+2]=d.rate;}}
    });
    m.sum(&mut wire)?;
    RUNTIME.with(|cell|->Result<()>{let mut cell=cell.borrow_mut();let r=cell.as_mut().unwrap();
        for s in 0..16 {if wire[3*s]!=0. {
            ensure(wire[3*s]==1.,"Poisson draw cache has multiple owners")?;
            let id=current(r,s);let d=Draw{count:equation::int32(wire[3*s+1])?,rate:wire[3*s+2],origin:Some(r.point)};
            if let Some(old)=r.values.get(&id){ensure(old.count==d.count&&old.rate==d.rate&&old.origin==d.origin,"Poisson owner cache mismatch")?;}
            else{r.values.insert(id,d);}
        }}Ok(())
    })
}
pub(super) fn checkpoint()->Option<Checkpoint>{
    RUNTIME.with(|cell|cell.borrow().as_ref().map(|r|{
        let mut entries=r.values.iter().map(|(&identity,d)|Entry{identity,count:d.count,rate:d.rate}).collect::<Vec<_>>();
        entries.sort_by_key(|e|e.identity);
        Checkpoint{schema:"b2-poisson-draw-state-v1".into(),numeric_profile:None,continuation:None,seed:r.seed,sequence:r.sequence,batch:r.batch,entries}
    }))
}
pub(super) fn memory_bytes(spec:&dynamic::Spec,batch:usize,visits:usize,state:Option<&Checkpoint>)->Option<usize>{
    let mut sites=0usize;
    for a in &spec.actions {if let Some(programs)=a.program_set.and_then(|i|spec.program_sets.get(i)){
        let mut streams=0u16;for n in programs.iter().flatten(){if let equation::Node::Poisson{stream,..}=n{streams|=1<<stream;}}
        sites=sites.checked_add(streams.count_ones() as usize)?;
    }}
    if sites==0{return Some(0)}
    state.map_or(0,|s|s.entries.len()).checked_add(batch.checked_mul(visits)?.checked_mul(sites)?)?
        .checked_mul(768)?.checked_add(1024)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn scope(forced:Option<Identity>)->Scope {
        Scope(RUNTIME.with(|cell|cell.replace(Some(Runtime {
            seed:13, sequence:9, batch:1, values:HashMap::new(),
            point:Point{batch:0,visit:0,action:0},action:dynamic::Action::default(),
            tick:0, forward:true, store_new:true, dirty:[false;16], forced,
        }))))
    }
    fn action()->dynamic::Action {dynamic::Action{noise_domain:71,noise_streams:1,..Default::default()}}
    fn key(a:&dynamic::Action,tick:u64)->u64 {
        let id=identity(a,0,0,tick);id.site.key(13,9,0,id.instant)
    }
    #[test]
    fn aliases_skip_rate_but_reverse_origin_checks_it() {
        let _scope=scope(None);let a=action();let k=key(&a,0);
        begin(&a,0,0,0,0,true,true);let count=sample(0,1.3,k,false).unwrap();
        begin(&a,0,0,1,0,true,true);assert!(!needs_rate(0));assert!(!owns_score(0));
        assert_eq!(sample(0,-1.,k,false).unwrap(),count);
        begin(&a,0,0,0,0,false,false);assert!(needs_rate(0));assert!(owns_score(0));
        assert!(sample(0,1.4,k,false).is_err());assert_eq!(sample(0,1.3,k,false).unwrap(),count);
    }
    #[test]
    fn nested_counterfactual_restores_baseline_after_error() {
        let a=action();let _outer=scope(None);begin(&a,0,0,0,0,true,true);
        assert_eq!(sample(0,0.,key(&a,0),false).unwrap(),0);
        let baseline=checkpoint();
        {
            let _inner=scope(Some(identity(&a,0,0,0)));begin(&a,0,0,0,0,true,true);
            assert_eq!(sample(0,0.,key(&a,0),false).unwrap(),1);
            begin(&a,0,0,1,0,true,true);assert_eq!(sample(0,-1.,key(&a,0),false).unwrap(),1);
            assert!(sample(0,0.,key(&a,0)^1,false).is_err());
        }
        assert_eq!(checkpoint(),baseline);
    }
    #[test]
    fn delayed_and_pending_identities_do_not_alias_rng_hashes() {
        let a=action();let mut delayed=a.clone();delayed.event_noise=Some(noise::EventAddress{delay:2,pending:None});
        assert_eq!(identity(&delayed,0,0,1).instant,u64::MAX);
        assert_eq!(identity(&delayed,0,0,2).site,identity(&delayed,0,0,3).site);
        assert_ne!(identity(&delayed,0,0,2),identity(&a,0,0,0));
        delayed.event_noise.as_mut().unwrap().pending=Some(73);
        assert_eq!(identity(&delayed,0,0,1),identity(&delayed,0,0,19));
        assert_ne!(identity(&delayed,0,0,1),identity(&delayed,0,1,1));
    }
    #[test]
    fn restored_draw_has_no_current_score_owner() {
        let _scope=scope(None);let a=action();let id=identity(&a,0,0,0);
        RUNTIME.with(|cell|cell.borrow_mut().as_mut().unwrap().values.insert(id,Draw{count:2,rate:1.3,origin:None}));
        begin(&a,0,0,0,0,false,false);assert!(!needs_rate(0));assert!(!owns_score(0));
        assert_eq!(sample(0,-1.,key(&a,0),false).unwrap(),2);
    }
}
