//! Version-one counter-addressed Poisson sampling and likelihood scores.
//! This numerical core is independent of SSA evaluation and its gradient policy.

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Error { Rate, Uniform, Iterations, Count, ScoreBoundary }
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sample { pub count: i32, pub draws: u32 }
#[derive(Clone, Copy)]
pub enum Event { Clock, Emission { delay: u64 }, Pending { id: u64 } }
pub const MAX_DRAWS: u32 = 100_000;

fn mix(mut x: u64) -> u64 {
    x=(x^(x>>30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x=(x^(x>>27)).wrapping_mul(0x94d049bb133111eb);
    x^(x>>31)
}

/// Event identity follows emission time or the persistent imported queue ID.
pub fn key(seed:u64, sequence:u64, batch:u64, domain:u64, entity:u64,
           tick:u64, stream:u64, event:Event) -> u64 {
    let (kind,instant)=match event {
        Event::Clock=>(0,tick), Event::Emission{delay}=>(1,tick.wrapping_sub(delay)),
        Event::Pending{id}=>(2,id),
    };
    let mut value=mix(seed^0x4232504f49533031);
    for x in [sequence,batch,domain,entity,kind,instant,stream] {
        value=mix(value^mix(x.wrapping_add(0x9e3779b97f4a7c15)));
    }
    value
}

/// An open uniform from 52 random bits. The odd numerator is exactly
/// representable in f64, including the largest value strictly below one.
pub fn uniform(key:u64, draw:u64) -> f64 {
    let bits=mix(key^mix(draw.wrapping_add(0x9e3779b97f4a7c15)));
    let odd=((bits>>12)<<1)|1;
    odd as f64*(1.0/9007199254740992.0)
}

/// Stable log probability, avoiding cancellation of three O(rate*log(rate))
/// terms near a large mean. Stirling error plus the Poisson deviance.
pub fn log_probability(count:f64, rate:f64) -> f64 {
    if count==0.0 {return -rate;}
    if count<16.0 {
        let factorial=(2..=count as u32).map(|k|(k as f64).ln()).sum::<f64>();
        return count*rate.ln()-rate-factorial;
    }
    let inv=1.0/count;let inv2=inv*inv;
    let stirling=inv*(1.0/12.0-inv2*(1.0/360.0-inv2*(1.0/1260.0-inv2*(1.0/1680.0-inv2/1188.0))));
    let delta=count-rate;
    let deviance=if delta.abs()<0.1*(count+rate) {
        let v=delta/(count+rate);let v2=v*v;
        let mut sum=delta*v;let mut term=2.0*count*v;
        for j in 1..100 {term*=v2;let next=sum+term/(2*j+1) as f64;if next==sum {break;}sum=next;}
        sum
    } else {count*(count/rate).ln()+rate-count};
    -stirling-deviance-0.5*(std::f64::consts::TAU*count).ln()
}

fn checked(count:f64, draws:u32) -> Result<Sample,Error> {
    if count<0.0 || count>i32::MAX as f64 {return Err(Error::Count);}
    Ok(Sample{count:count as i32,draws})
}

/// Exponential waiting times below ten, transformed rejection (PTRS) above.
/// Waiting times preserve tiny positive rates without rounding exp(-rate) to 1.
/// A failed draw budget or accepted count overflow is an error, never a biased
/// capped count, zero substitute, or rejection of an otherwise accepted sample.
pub fn sample_with(rate:f64, mut source:impl FnMut(u32)->f64) -> Result<Sample,Error> {
    if !rate.is_finite() || rate<0.0 || rate>i32::MAX as f64 {return Err(Error::Rate);}
    if rate==0.0 {return Ok(Sample{count:0,draws:0});}
    let mut draws=0;
    let mut next=|| {
        if draws==MAX_DRAWS {return Err(Error::Iterations);}
        let value=source(draws);draws+=1;
        if !value.is_finite() || value<=0.0 || value>=1.0 {return Err(Error::Uniform);}
        Ok(value)
    };
    if rate<10.0 {
        let mut arrival=0.0;let mut count=0;
        loop {arrival-=(-next()?).ln_1p();if arrival>=rate {return Ok(Sample{count,draws});}count+=1;}
    }
    let b=0.931+2.53*rate.sqrt();let a=-0.059+0.02483*b;
    let alpha=1.1239+1.1328/(b-3.4);let vr=0.9277-3.6224/(b-2.0);
    loop {
        let u=next()?-0.5;let v=next()?;let us=0.5-u.abs();
        let count=((2.0*a/us+b)*u+rate+0.43).floor();
        if us>=0.07 && v<=vr {return checked(count,draws);}
        if count<0.0 || us<0.013 && v>us {continue;}
        if v.ln()+alpha.ln()-(a/(us*us)+b).ln()<=log_probability(count,rate) {
            return checked(count,draws);
        }
    }
}

pub fn sample(rate:f64, key:u64) -> Result<Sample,Error> {
    sample_with(rate,|draw|uniform(key,draw as u64))
}

/// d log P(K|rate)/d rate; this is NOT a pathwise sample derivative.
/// At zero the distribution exists, but this score formula is not defined.
pub fn score(rate:f64, count:i32) -> Result<f64,Error> {
    if !rate.is_finite() || rate<0.0 || rate>i32::MAX as f64 || count<0 {return Err(Error::Rate);}
    if rate==0.0 {return Err(Error::ScoreBoundary);}
    Ok((count as f64-rate)/rate)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn event_addresses_survive_routing_and_replay() {
        let k=|t,event|key(19,7,3,5,11,t,13,event);
        assert_eq!(k(10,Event::Emission{delay:3}),k(12,Event::Emission{delay:5}));
        assert_eq!(k(0,Event::Pending{id:4}),k(u64::MAX,Event::Pending{id:4}));
        assert_ne!(k(0,Event::Pending{id:4}),k(0,Event::Pending{id:5}));
        assert_ne!(k(7,Event::Clock),k(7,Event::Emission{delay:0}));
        assert_ne!(k(0,Event::Emission{delay:3}),k(1<<53,Event::Emission{delay:3}));
        assert_ne!(k(0,Event::Clock),key(19,7,3,5,11,0,14,Event::Clock));
    }
    #[test]
    fn zero_and_errors_have_explicit_semantics() {
        assert_eq!(sample_with(0.0,|_|panic!("zero must not draw")),Ok(Sample{count:0,draws:0}));
        for rate in [-1.0,f64::NAN,f64::INFINITY,i32::MAX as f64+1.0] {assert_eq!(sample(rate,1),Err(Error::Rate));}
        for u in [0.0,1.0,f64::NAN] {assert_eq!(sample_with(2.0,|_|u),Err(Error::Uniform));}
        assert_eq!(sample_with(1.0,|_|1e-100),Err(Error::Iterations));
        assert_eq!(sample_with(i32::MAX as f64,|j|if j%2==0 {0.6} else {0.1}),Err(Error::Count));
        assert_eq!(score(0.0,0),Err(Error::ScoreBoundary));
        assert_eq!(score(2.0,3),Ok(0.5));
    }
    #[test]
    fn tiny_rates_do_not_round_the_probability_to_zero() {
        assert_eq!(sample_with(1e-12,|j|if j==0 {5e-13} else {0.5}),Ok(Sample{count:1,draws:2}));
    }
    #[test]
    fn moments_scores_and_large_integer_resolution() {
        for rate in [0.05,1.0,9.99,10.0,20.0,1000.0,1e9] {
            let n=50_000;let mut sum=0.0;let mut square=0.0;let mut odds=0;
            for i in 0..n {
                let k=key(7123,0,0,1,i,0,0,Event::Clock);
                let count=sample(rate,k).unwrap().count;
                let delta=count as f64-rate;sum+=delta;square+=delta*delta;odds+=count&1;
            }
            assert!((sum/n as f64).abs()<6.0*(rate/n as f64).sqrt(),"mean at {rate}");
            assert!((square/n as f64-rate).abs()<7.0*((rate+2.0*rate*rate)/n as f64).sqrt(),"variance at {rate}");
            if rate>1e8 {assert!((odds as f64/n as f64-0.5).abs()<0.02,"lost integer low bits");}
        }
    }
}
