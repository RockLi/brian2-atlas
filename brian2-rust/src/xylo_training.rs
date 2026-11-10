//! CPU quantization-aware training with the admitted Xylo Audio 2 transition.
//! Integer forward; declared surrogate/straight-through backward; Adam in f64.
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use crate::training::{Optimizer, State, Surrogate};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
fn require(ok: bool, message: &str) -> Result<()> { if ok {Ok(())} else {Err(message.into())} }
const LO: i32 = -32768;
const HI: i32 = 32767;
fn clip(x: i32) -> i32 { x.clamp(LO, HI) }
fn decay(x: i32) -> i32 { (x >> 1) + i32::from(x == 1) }
fn interior(x: i32) -> f64 { if (LO..=HI).contains(&x) {1.0} else {0.0} }

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Plan {
    pub schema: String,
    pub backend: String,
    pub sizes: Vec<usize>,
    pub threshold: Vec<i32>,
    pub weight_scale: f64,
    pub synaptic_shift: u8,
    pub membrane_shift: u8,
    pub detach_reset: bool,
    pub surrogate: Surrogate,
    pub optimizer: Optimizer,
    pub logit_scale: f64,
    pub max_tape_bytes: usize,
    pub seed: u64,
    #[serde(default, skip_serializing_if="Option::is_none")]
    pub recurrent: Option<bool>,
    #[serde(default, skip_serializing_if="Option::is_none")]
    pub readout_start: Option<usize>,
    /// Optional fixed binary class codebook; neural output counts remain unchanged.
    #[serde(default, skip_serializing_if="Option::is_none")]
    pub class_codes: Option<Vec<Vec<i32>>>,
    #[serde(default, skip_serializing_if="Option::is_none")]
    pub input_max: Option<u8>,
    /// Training objective only: mean squared output-rate error to 0.1/0.9 code bits.
    #[serde(default, skip_serializing_if="Option::is_none")]
    pub code_rate_weight: Option<f64>,
    #[serde(default, skip_serializing_if="Option::is_none")]
    pub readout_source: Option<String>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub plan: Plan,
    pub state: Option<State>,
    pub operation: String,
    pub inputs: Vec<Vec<Vec<f64>>>,
    pub labels: Vec<usize>,
    pub initial: Option<Continuation>,
}
/// Explicit evaluation state, including pending spikes and readout history.
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Continuation {
    pub schema: String,
    pub model_sha256: String,
    pub elapsed_ticks: usize,
    pub currents: Vec<Vec<i32>>,
    pub membranes: Vec<Vec<i32>>,
    pub spikes: Vec<Vec<i32>>,
    pub counts: Vec<Vec<i32>>,
    #[serde(default, skip_serializing_if="Option::is_none")]
    pub current_sums: Option<Vec<Vec<i32>>>,
}
#[derive(Serialize)]
pub struct Output {
    pub schema: &'static str,
    pub backend: &'static str,
    pub numeric_profile: &'static str,
    pub state: State,
    pub loss: f64,
    pub gradients: Vec<Vec<f64>>,
    pub final_membrane: Vec<Vec<f64>>,
    pub spikes: Vec<Vec<Vec<f64>>>,
    pub logits: Vec<Vec<f64>>,
    #[serde(skip_serializing_if="Option::is_none")]
    pub currents: Option<Vec<Vec<Vec<f64>>>>,
    #[serde(skip_serializing_if="Option::is_none")]
    pub membranes: Option<Vec<Vec<Vec<f64>>>>,
    #[serde(skip_serializing_if="Option::is_none")]
    pub continuation: Option<Continuation>,
    pub tape_bytes: usize,
    pub gradient_scope: &'static str,
}
impl Plan {
    fn validate(&self) -> Result<()> {
        let v2=self.schema=="atlas-xylo-training-plan-v2";
        require((self.schema=="atlas-xylo-training-plan-v1" || v2) && self.backend=="cpu",
                "Xylo training requires its CPU v1/v2 plan")?;
        require(if v2 {self.recurrent==Some(true) && self.readout_start.is_some_and(|n|n<128)}
                else {self.recurrent.is_none() && self.readout_start.is_none()},
                "Xylo v2 requires recurrence and bounded readout_start; v1 rejects these fields")?;
        require(self.sizes.len()==3 && (1..=16).contains(&self.sizes[0]) &&
                (1..=if v2 {128} else {64}).contains(&self.sizes[1]) && (1..=8).contains(&self.sizes[2]),
                "Xylo training requires 1..16 inputs, 1..64 hidden (v1) or 1..128 (v2), 1..8 outputs")?;
        require(self.input_max.is_none_or(|n| (1..=15).contains(&n)), "input_max must be 1..15")?;
        if let Some(codes)=&self.class_codes {
            require(v2 && (2..=32).contains(&codes.len()) && codes.iter().all(|row|
                row.len()==self.sizes[2] && row.iter().all(|v| *v==0 || *v==1)
                && row.iter().any(|v|*v==1)), "class codes require v2 and 2..32 nonzero binary rows")?;
            let weight=codes[0].iter().sum::<i32>();
            require(codes.iter().all(|row|row.iter().sum::<i32>()==weight)
                && (0..codes.len()).all(|i| !codes[..i].contains(&codes[i])),
                "class codes must be distinct and have equal Hamming weight")?;
        }
        if let Some(source)=&self.readout_source {
            require(v2 && source=="synaptic_current" && self.code_rate_weight.is_none(),
                    "readout_source requires recurrent synaptic_current without a code-rate loss")?;
        }
        if let Some(weight)=self.code_rate_weight {
            require(v2 && self.class_codes.is_some() && weight.is_finite() && (0.0..=100.0).contains(&weight),
                    "code_rate_weight requires recurrent class codes and finite weight in 0..100")?;
        }
        require(self.threshold.len()==2 && self.threshold.iter().all(|v| (1..=32767).contains(v)),
                "Xylo thresholds must be positive int16")?;
        require(self.synaptic_shift==1 && self.membrane_shift==1 && self.detach_reset,
                "Xylo training v1 requires shifts=1 and detached reset")?;
        require(self.weight_scale.is_finite() && self.weight_scale>=1.0 && self.weight_scale<=128.0
                && self.weight_scale.log2().fract()==0.0,"invalid Xylo weight scale")?;
        require(self.logit_scale.is_finite() && self.logit_scale>0.0 && self.logit_scale<=100.0,
                "invalid Xylo logit scale")?;
        require(self.surrogate.kind=="fast_sigmoid" && self.surrogate.slope.is_finite()
                && self.surrogate.slope>0.0 && self.surrogate.scale.is_finite() && self.surrogate.scale>0.0,
                "invalid Xylo surrogate")?;
        let o=&self.optimizer;
        require(matches!(o.kind.as_str(),"adam"|"sgd") && o.learning_rate.is_finite() && o.learning_rate>0.0
                && o.beta1.is_finite() && (0.0..1.0).contains(&o.beta1)
                && o.beta2.is_finite() && (0.0..1.0).contains(&o.beta2)
                && o.epsilon.is_finite() && o.epsilon>0.0,"invalid Xylo optimizer")?;
        Ok(())
    }
}

pub fn execute(request: Request) -> Result<Output> {
    let Request{plan:p,state,operation,inputs:x,labels,initial}=request;
    p.validate()?;
    require(matches!(operation.as_str(),"train"|"evaluate"|"gradients"),"unknown Xylo training operation")?;
    let recurrent=p.schema=="atlas-xylo-training-plan-v2";
    require(initial.is_none() || (recurrent && operation=="evaluate"),
            "Xylo continuation is evaluation-only in v2; training requires fresh state")?;
    let readout_start=p.readout_start.unwrap_or(0);
    let current_readout=p.readout_source.is_some();
    let [ni,nh,no]=[p.sizes[0],p.sizes[1],p.sizes[2]];
    let classes=p.class_codes.as_ref().map_or(no,|c|c.len());
    let cells=nh+no;let batch=x.len();
    require((1..=1024).contains(&batch) && labels.len()==batch,"invalid Xylo batch/labels")?;
    let steps=x[0].len();
    require((1..=128).contains(&steps) && x.iter().all(|s| s.len()==steps && s.iter().all(|r|
        r.len()==ni && r.iter().all(|v| v.is_finite() && *v>=0.0 && *v<=p.input_max.unwrap_or(1) as f64 && v.fract()==0.0))) && labels.iter().all(|y| *y<classes),
        "Xylo inputs must be bounded integer counts, batch x 1..128 ticks x input; labels in range")?;
    let lengths=if recurrent {vec![ni*nh,nh*no,nh*nh]} else {vec![ni*nh,nh*no]};
    let mut state=state.ok_or("Xylo training requires explicit initial weights")?;
    require(state.weights.len()==lengths.len() && state.first_moment.len()==lengths.len() && state.second_moment.len()==lengths.len(),
        "Xylo optimizer bank mismatch")?;
    for l in 0..lengths.len() {
        require(state.weights[l].len()==lengths[l] && state.first_moment[l].len()==lengths[l]
            && state.second_moment[l].len()==lengths[l] &&
            state.weights[l].iter().all(|v| v.is_finite() && (-128.0..=127.0).contains(&(v*p.weight_scale)))
            && state.first_moment[l].iter().all(|v| v.is_finite())
            && state.second_moment[l].iter().all(|v| v.is_finite() && *v>=0.0),"invalid Xylo optimizer state")?;
    }
    // Bound all persistent forward/backward buffers plus nested result storage.
    let count=batch.checked_mul(steps).and_then(|v|v.checked_mul(cells)).ok_or("Xylo tape overflow")?;
    let bytes=count.checked_mul(128).and_then(|v|v.checked_add(lengths.iter().sum::<usize>()*64))
        .ok_or("Xylo tape overflow")?;
    require(bytes<=p.max_tape_bytes,"Xylo training tape budget exceeded")?;
    let weights:Vec<Vec<i32>>=state.weights.iter().map(|bank|bank.iter()
        .map(|v|(v*p.weight_scale).round_ties_even() as i32).collect()).collect();
    let identity=format!("{:x}",Sha256::digest(serde_json::to_vec(&(&p,&weights))?));
    let initial=initial.unwrap_or_else(||Continuation{schema:"atlas.xylo-evaluation-state.v1".into(),
        model_sha256:identity.clone(),elapsed_ticks:0,currents:vec![vec![0;cells];batch],
        membranes:vec![vec![0;cells];batch],spikes:vec![vec![0;cells];batch],counts:vec![vec![0;no];batch],
        current_sums:if current_readout {Some(vec![vec![0;no];batch])} else {None}});
    require(initial.schema=="atlas.xylo-evaluation-state.v1" && initial.model_sha256==identity,
            "Xylo continuation model identity mismatch")?;
    require(initial.elapsed_ticks<=128 && steps<=128-initial.elapsed_ticks,
            "Xylo continuation exceeds 128-tick horizon")?;
    let elapsed=initial.elapsed_ticks;let read_ticks=(elapsed+steps).saturating_sub(readout_start);
    let old_read_ticks=elapsed.saturating_sub(readout_start);
    require(operation=="evaluate" || read_ticks>0,"training requires a nonempty readout window")?;
    for bank in [&initial.currents,&initial.membranes,&initial.spikes] {
        require(bank.len()==batch && bank.iter().all(|r|r.len()==cells),"Xylo continuation state shape mismatch")?;
    }
    require(initial.currents.iter().flatten().chain(initial.membranes.iter().flatten()).all(|v|(LO..=HI).contains(v))
        && initial.spikes.iter().all(|r|r.iter().enumerate().all(|(k,q)|*q>=0 && *q<=if k<nh {31}else{1}))
        && initial.counts.len()==batch && initial.counts.iter().all(|r|r.len()==no && r.iter().all(|v|*v>=0 && *v<=old_read_ticks as i32)),
        "Xylo continuation state out of range")?;
    require(if current_readout {initial.current_sums.as_ref().is_some_and(|s|
        s.len()==batch && s.iter().all(|r|r.len()==no && r.iter().all(|v|
            (LO*old_read_ticks as i32..=HI*old_read_ticks as i32).contains(v))))}
        else {initial.current_sums.is_none()}, "Xylo continuation readout history mismatch")?;
    let mut current=vec![0;count];let mut voltage=vec![0;count];let mut pre=vec![0;count];
    let mut spike=vec![0;count];
    let at=|b:usize,t:usize,k:usize| (b*steps+t)*cells+k;
    for b in 0..batch {for t in 0..steps {for l in 0..2 {
        let (fan,n,offset,cap)=if l==0 {(ni,nh,0,31)} else {(nh,no,nh,1)};
        for j in 0..n {
            let idx=at(b,t,offset+j);
            let mut i=if t==0 {initial.currents[b][offset+j]} else {current[idx-cells]};
            for src in 0..fan {
                let input=if l==0 {x[b][t][src] as i32} else if t==0 {initial.spikes[b][src]} else {spike[at(b,t-1,src)]};
                i=clip(i+weights[l][src*n+j]*input);
            }
            if recurrent && l==0 {
                for src in 0..nh {
                    let input=if t==0 {initial.spikes[b][src]} else {spike[at(b,t-1,src)]};
                    i=clip(i+weights[2][src*nh+j]*input);
                }
            }
            i-=decay(i);current[idx]=i;
            let old=if t==0 {initial.membranes[b][offset+j]} else {voltage[idx-cells]};
            let u=clip(old+clip(-decay(old)+i));pre[idx]=u;
            let q=(u.div_euclid(p.threshold[l])).clamp(0,cap);spike[idx]=q;
            voltage[idx]=u-q*p.threshold[l];
        }
    }}}
    let mut logits=vec![vec![0.0;classes];batch];let mut dlogits=vec![vec![0.0;no];batch];let mut loss=0.0;
    let mut counts=initial.counts.clone();
    let mut current_sums=initial.current_sums.clone();
    for b in 0..batch {
        for j in 0..no {
            counts[b][j]+=(0..steps).filter(|t|elapsed+t>=readout_start).map(|t|spike[at(b,t,nh+j)]).sum::<i32>();
        }
        if let Some(sums)=&mut current_sums {
            for j in 0..no {
                sums[b][j]+=(0..steps).filter(|t|elapsed+t>=readout_start).map(|t|current[at(b,t,nh+j)]).sum::<i32>();
            }
        }
        let values=if let Some(sums)=&current_sums {&sums[b]} else {&counts[b]};
        for c in 0..classes {
            let score=if let Some(codes)=&p.class_codes {
                (0..no).map(|j|values[j]*codes[c][j]).sum::<i32>()
            } else {values[c]};
            logits[b][c]=score as f64*p.logit_scale/read_ticks.max(1) as f64;
            if current_readout {logits[b][c]/=p.threshold[1] as f64;}
        }
        let max=logits[b].iter().copied().fold(f64::NEG_INFINITY,f64::max);
        let z=logits[b].iter().map(|v|(v-max).exp()).sum::<f64>();
        loss+=(z.ln()+max-logits[b][labels[b]])/batch as f64;
        for c in 0..classes {
            let grad=((logits[b][c]-max).exp()/z-f64::from(c==labels[b]))/batch as f64;
            if let Some(codes)=&p.class_codes {
                for j in 0..no {dlogits[b][j]+=grad*codes[c][j] as f64;}
            } else {dlogits[b][c]=grad;}
        }
        if let Some(weight)=p.code_rate_weight {
            let codes=p.class_codes.as_ref().unwrap(); // validated above
            for j in 0..no {
                let target=0.1+0.8*codes[labels[b]][j] as f64;
                let error=counts[b][j] as f64/read_ticks.max(1) as f64-target;
                loss+=weight*error*error/(batch*no) as f64;
                // dlogits is scaled by logit_scale/read_ticks in the reverse pass.
                dlogits[b][j]+=2.0*weight*error/((batch*no) as f64*p.logit_scale);
            }
        }
    }
    let mut gradient:Vec<Vec<f64>>=lengths.iter().map(|n|vec![0.0;*n]).collect();
    if operation!="evaluate" {
        let mut di=vec![0.0;count];let mut dv=di.clone();let mut dq=di.clone();
        for b in 0..batch { for t in 0..steps { for j in 0..no {
            if t>=readout_start {
                let g=dlogits[b][j]*p.logit_scale/read_ticks as f64;
                if current_readout {di[at(b,t,nh+j)]=g/p.threshold[1] as f64;}
                else {dq[at(b,t,nh+j)]=g;}
            }
        }}}
        for b in 0..batch {for t in (0..steps).rev() {for l in (0..2).rev() {
            let (fan,n,offset,cap)=if l==0 {(ni,nh,0,31)} else {(nh,no,nh,1)};
            let theta=p.threshold[l] as f64;
            for j in 0..n {
                let idx=at(b,t,offset+j);let old_v=if t==0 {0} else {voltage[idx-cells]};
                // Sum the declared unit-spike surrogate at every count boundary.
                let derivative=(1..=cap).map(|k|p.surrogate.derivative(pre[idx] as f64/theta-k as f64)/theta).sum::<f64>();
                let du=dv[idx]+dq[idx]*derivative; // reset subtraction is detached
                let inner=-decay(old_v)+current[idx];
                let g=du*interior(old_v+clip(inner));
                let gi=g*interior(inner);
                if t>0 {dv[idx-cells]+=g-0.5*gi;}
                // STE through minimum-decay/rounding, with nominal shift derivative.
                let mut gsum=(di[idx]+gi)*0.5;
                let mut sum=if t==0 {0} else {current[idx-cells]};
                let mut before=vec![0;fan];
                for src in 0..fan {
                    let input=if l==0 {x[b][t][src] as i32} else if t==0 {initial.spikes[b][src]} else {spike[at(b,t-1,src)]};
                    before[src]=sum;sum=clip(sum+weights[l][src*n+j]*input);
                }
                // Recurrent input follows external input; reverse it first.
                if recurrent && l==0 {
                    let mut rec_before=vec![0;nh];
                    for src in 0..nh {
                        let input=if t==0 {0} else {spike[at(b,t-1,src)]};
                        rec_before[src]=sum;sum=clip(sum+weights[2][src*nh+j]*input);
                    }
                    for src in (0..nh).rev() {
                        let input=if t==0 {0} else {spike[at(b,t-1,src)]};let w=weights[2][src*nh+j];
                        gsum*=interior(rec_before[src]+w*input);
                        gradient[2][src*nh+j]+=gsum*input as f64*p.weight_scale;
                        if t>0 {dq[at(b,t-1,src)]+=gsum*w as f64;}
                    }
                }
                // Reverse each saturating accumulation in its source-major order.
                for src in (0..fan).rev() {
                    let input=if l==0 {x[b][t][src] as i32} else if t==0 {initial.spikes[b][src]} else {spike[at(b,t-1,src)]};
                    let w=weights[l][src*n+j];gsum*=interior(before[src]+w*input);
                    gradient[l][src*n+j]+=gsum*input as f64*p.weight_scale;
                    if l==1 && t>0 {dq[at(b,t-1,src)]+=gsum*w as f64;}
                }
                if t>0 {di[idx-cells]+=gsum;}
            }
        }}}
        require(gradient.iter().flatten().all(|v|v.is_finite()),"nonfinite Xylo gradient")?;
    }
    if operation=="train" {
        state.step=state.step.checked_add(1).ok_or("Xylo optimizer step overflow")?;
        let o=&p.optimizer;
        for l in 0..lengths.len() {for e in 0..lengths[l] {
            let g=gradient[l][e];
            let update=if o.kind=="sgd" {g} else {
                state.first_moment[l][e]=o.beta1*state.first_moment[l][e]+(1.0-o.beta1)*g;
                state.second_moment[l][e]=o.beta2*state.second_moment[l][e]+(1.0-o.beta2)*g*g;
                let m=state.first_moment[l][e]/(1.0-o.beta1.powf(state.step as f64));
                let v=state.second_moment[l][e]/(1.0-o.beta2.powf(state.step as f64));m/(v.sqrt()+o.epsilon)
            };
            require(update.is_finite() && state.second_moment[l][e].is_finite(),"nonfinite Xylo optimizer update")?;
            state.weights[l][e]=(state.weights[l][e]-o.learning_rate*update).clamp(-128.0/p.weight_scale,127.0/p.weight_scale);
        }}
    }
    let nested=|data:&[i32]|(0..batch).map(|b|(0..steps).map(|t|
        data[at(b,t,0)..at(b,t,0)+cells].iter().map(|v|*v as f64).collect()).collect()).collect();
    let final_membrane=(0..batch).map(|b|voltage[at(b,steps-1,0)..at(b,steps-1,0)+cells].iter().map(|v|*v as f64).collect()).collect();
    let continuation=if recurrent && operation=="evaluate" {
        let last=|data:&[i32]|(0..batch).map(|b|data[at(b,steps-1,0)..at(b,steps-1,0)+cells].to_vec()).collect();
        Some(Continuation{schema:initial.schema,model_sha256:identity,elapsed_ticks:elapsed+steps,
            currents:last(&current),membranes:last(&voltage),spikes:last(&spike),counts,current_sums})
    } else {None};
    Ok(Output{schema:"b2-lif-training-result-v1",backend:"cpu",numeric_profile:"xylo-int16-forward-f64-surrogate",
        state,loss,gradients:gradient,final_membrane,spikes:nested(&spike),logits,
        currents:if operation=="evaluate" {Some(nested(&current))} else {None},
        membranes:if operation=="evaluate" {Some(nested(&voltage))} else {None},
        continuation,tape_bytes:bytes,gradient_scope:"fresh-sequence-surrogate-bptt-quantization-ste"})
}
