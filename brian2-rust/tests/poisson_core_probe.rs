//! Test-only C ABI for the actual native sampler, not a second implementation.
#[path="../src/training/poisson.rs"] mod poisson;
fn error(e:poisson::Error)->u32 {match e {poisson::Error::Rate=>1,poisson::Error::Uniform=>2,
    poisson::Error::Iterations=>3,poisson::Error::Count=>4,poisson::Error::ScoreBoundary=>5}}
#[no_mangle]
pub unsafe extern "C" fn poisson_batch(n:usize,rates:*const f64,keys:*const u64,
    counts:*mut i32,draws:*mut u32,errors:*mut u32,scores:*mut f64,logps:*mut f64) {
    for i in 0..n {
        match poisson::sample(*rates.add(i),*keys.add(i)) {
            Ok(s)=>{*counts.add(i)=s.count;*draws.add(i)=s.draws;*errors.add(i)=0;
                *scores.add(i)=poisson::score(*rates.add(i),s.count).unwrap_or(f64::NAN);
                *logps.add(i)=poisson::log_probability(s.count as f64,*rates.add(i));}
            Err(e)=>{*counts.add(i)=0;*draws.add(i)=0;*errors.add(i)=error(e);*scores.add(i)=f64::NAN;*logps.add(i)=f64::NAN;}
        }
    }
}
#[no_mangle]
pub extern "C" fn poisson_uniform(key:u64,draw:u64)->f64 {poisson::uniform(key,draw)}
#[no_mangle]
pub extern "C" fn poisson_logp(count:f64,rate:f64)->f64 {poisson::log_probability(count,rate)}
