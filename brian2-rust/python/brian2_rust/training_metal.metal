#include <metal_stdlib>
using namespace metal;

// Metal has no expm1/log1p intrinsic. Taylor below .5 avoids cancellation;
// the compensated logarithm recovers the bits lost when forming 1+x.
float atlas_dynamic_expm1(float x){
 if(abs(x)<.5f)return x*(1+x*(.5f+x*(1.0f/6+x*(1.0f/24+x*(1.0f/120+x*(1.0f/720+x*(1.0f/5040+x*(1.0f/40320+x*(1.0f/362880+x/3628800.0f)))))))));
 return exp(x)-1;
}
float atlas_dynamic_log1p(float x){
 if(x< -1)return NAN;if(x== -1)return -INFINITY;
 float u=1+x;if(u==1)return x;
 return log(u)*(x/(u-1));
}

// Math extension ABI 1: stable scalar values and pathwise derivatives.
float atlas_dynamic_math(ulong kind,float x){
 switch(kind){
  case 0:return tan(x);case 1:return cosh(x);case 2:return sinh(x);
  case 3:return log10(x);case 4:return atlas_dynamic_expm1(x);case 5:return atlas_dynamic_log1p(x);
  case 6:return x==0?1.0f:atlas_dynamic_expm1(x)/x;
  case 7:return acos(x);case 8:return asin(x);case 9:return atan(x);
  case 13:return floor(x);case 10:return ceil(x);case 11:return abs(x);case 12:return float(x>0)-float(x<0);
  default:return NAN;
 }
}
float atlas_dynamic_math_grad(ulong kind,float x){
 switch(kind){
  case 0:{float c=cos(x);return 1/(c*c);}case 1:return sinh(x);case 2:return cosh(x);
  case 3:return x>1?(1/x)/2.302585092994046f:1/(x*2.302585092994046f);
  case 4:return exp(x);case 5:return 1/(1+x);
  case 6:
   if(abs(x)<.1f)return .5f+x*(1.0f/3+x*(1.0f/8+x*(1.0f/30+x*(1.0f/144+x*(1.0f/840+x*(1.0f/5760+x/45360.0f))))));
   if(x< -20){float inv=1/x;return inv*inv;}
   if(x>20)return (atlas_dynamic_expm1(x)*(1-1/x)+1)/x;
   return ((x-1)*atlas_dynamic_expm1(x)+x)/(x*x);
  case 7:return -1/sqrt((1-x)*(1+x));case 8:return 1/sqrt((1-x)*(1+x));
  case 9:if(abs(x)>1){float inv=1/x;return inv*inv/(1+inv*inv);}return 1/(1+x*x);
  case 10:case 12:case 13:return 0;case 11:return float(x>0)-float(x<0);
  default:return NAN;
 }
}

// One sample per GPU lane, batch-private VJPs; no atomic floating-point sum.
// This correctness-first kernel makes no performance/scalability claim.
kernel void atlas_lif_bptt(
 device const ulong* m [[buffer(0)]], device const float* p [[buffer(1)]],
 device const float* x [[buffer(2)]], device const float* w [[buffer(3)]],
 device const float* initial [[buffer(4)]], device const ulong* labels [[buffer(5)]],
 device float* pre [[buffer(6)]], device float* spike [[buffer(7)]],
 device float* membrane [[buffer(8)]], device float* grad [[buffer(9)]],
 device float* initial_grad [[buffer(10)]], device float* logits [[buffer(11)]],
 device float* losses [[buffer(12)]], device float* carry [[buffer(13)]],
 device float* ds [[buffer(14)]], device float* previous [[buffer(15)]], uint b [[thread_position_in_grid]]) {
 ulong B=m[0],T=m[1],L=m[2],I=m[3],N=m[4],E=m[5],C=m[16+L];
 if(b>=B)return;
 ulong base=ulong(b)*N;
 for(ulong k=0;k<N;k++){membrane[base+k]=initial[base+k];carry[base+k]=0;}
 for(ulong e=0;e<E;e++)grad[ulong(b)*E+e]=0;
 for(ulong t=0;t<T;t++){
  ulong at=(ulong(b)*T+t)*N;
  for(ulong l=0;l<L;l++)for(ulong j=0;j<m[17+l];j++){
   ulong k=m[40+l]+j;float u=p[l]*membrane[base+k];float s=float(u>p[L+l]);
   pre[at+k]=u;spike[at+k]=s;
   membrane[base+k]=m[6]?u:u-p[L+l]*s;
  }
  for(ulong l=0;l<L;l++)for(ulong j=0;j<m[17+l];j++){
   ulong k=m[40+l]+j;
   for(ulong i=0;i<m[16+l];i++){
    float input=l?spike[at+m[39+l]+i]:x[(ulong(b)*T+t)*I+i];
    membrane[base+k]+=w[m[64+l]+i*m[17+l]+j]*input;
   }
  }
  if(m[6])for(ulong k=0;k<N;k++)membrane[base+k]*=1-spike[at+k];
 }
 float scale=p[2*L+2];
 for(ulong j=0;j<C;j++){
  float sum=0;for(ulong t=0;t<T;t++)sum+=spike[(ulong(b)*T+t)*N+m[39+L]+j]*scale/float(T);
  logits[ulong(b)*C+j]=sum;
 }
 float maxlog=-INFINITY;for(ulong j=0;j<C;j++)maxlog=max(maxlog,logits[ulong(b)*C+j]);
 float denom=0;for(ulong j=0;j<C;j++)denom+=exp(logits[ulong(b)*C+j]-maxlog);
 losses[b]=maxlog+log(denom)-logits[ulong(b)*C+labels[b]];
 if(m[9])for(ulong t=T;t-->0;){
  ulong at=(ulong(b)*T+t)*N;
  for(ulong k=0;k<N;k++){ds[base+k]=0;previous[base+k]=0;}
  for(ulong j=0;j<C;j++){
   float d=(exp(logits[ulong(b)*C+j]-maxlog)/denom-float(j==labels[b]))/float(B);
   ds[base+m[39+L]+j]=d*scale/float(T);
  }
  for(ulong l=L;l-->0;){
   for(ulong i=0;i<m[16+l];i++)for(ulong j=0;j<m[17+l];j++){
    ulong k=m[40+l]+j,e=m[64+l]+i*m[17+l]+j;
    float g=carry[base+k]*(m[6]?1-spike[at+k]:1);
    float input=l?spike[at+m[39+l]+i]:x[(ulong(b)*T+t)*I+i];
    grad[ulong(b)*E+e]+=g*input;
    if(l)ds[base+m[39+l]+i]+=g*w[e];
   }
   for(ulong j=0;j<m[17+l];j++){
    ulong k=m[40+l]+j;float u=pre[at+k],s=spike[at+k];
    float q=1+p[2*L]*abs(u-p[L+l]);float phi=p[2*L+1]/(q*q);
    float before_reset=u;
    if(m[6]&&!m[7])for(ulong i=0;i<m[16+l];i++){
     float input=l?spike[at+m[39+l]+i]:x[(ulong(b)*T+t)*I+i];
     before_reset+=w[m[64+l]+i*m[17+l]+j]*input;
    }
    float derivative=m[6]?(1-s-(m[7]?0:before_reset*phi)):(1-(m[7]?0:p[L+l]*phi));
    previous[base+k]=p[l]*(carry[base+k]*derivative+ds[base+k]*phi);
   }
  }
  for(ulong k=0;k<N;k++)carry[base+k]=(t>0&&t%m[8]==0)?0:previous[base+k];
 }
 for(ulong k=0;k<N;k++)initial_grad[base+k]=carry[base+k];
}

// Bounded scalar SSA interpreter and reverse-mode VJP; all arithmetic executes
// on the selected GPU. Parameter nodes accumulate into the shared bank slots.
float atlas_equation(device const ulong* m,device const float* p,
 device const float* w,ulong layer,float voltage,device float* grad,
 ulong gradient_base,float seed,bool backward) {
 ulong header=m[11]+3*layer,count=m[header],start=m[header+1],constants=m[header+2];
 float value[128],adjoint[128];bool active[128];bool activity=(m[9]&2)!=0;
 for(ulong i=0;i<count;i++){
  ulong q=start+4*i,op=m[q],a=m[q+1],b=m[q+2],param=m[q+3];float c=p[constants+i],z=0;
  switch(op){
   case 0:z=voltage;break;case 1:z=c;break;case 2:z=w[param];break;
   case 3:z=value[a]+value[b];break;case 4:z=value[a]-value[b];break;
   case 5:z=value[a]*value[b];break;case 6:z=value[a]/value[b];break;
   case 7:z=-value[a];break;case 8:z=exp(value[a]);break;case 9:z=log(value[a]);break;
   case 10:z=tanh(value[a]);break;case 11:z=sqrt(value[a]);break;
   case 12:z=sin(value[a]);break;case 13:z=cos(value[a]);break;
   case 52:z=atlas_dynamic_math(param,value[a]);break;
   case 14:z=c==0?1:pow(abs(value[a]),c);if(value[a]<0){if(floor(c)!=c)return NAN;if(fmod(abs(c),2.0f)==1)z=-z;}break;
  }
  if(!isfinite(z))return NAN;value[i]=z;adjoint[i]=0;
  active[i]=false;
  if(activity)switch(op){
   case 0:case 15:active[i]=true;break;
   case 2:active[i]=p[m[13]-m[5]+param]!=0;break;
   case 3:case 4:case 5:case 6:active[i]=active[a]||active[b];break;
   case 7:case 8:case 9:case 10:case 11:case 12:case 13:case 14:case 52:active[i]=active[a];break;
  }
 }
 if(!backward)return value[count-1];
 adjoint[count-1]=seed;float dv=0;
 for(ulong i=count;i-->0;){
  ulong q=start+4*i,op=m[q],a=m[q+1],b=m[q+2],param=m[q+3];float c=p[constants+i],g=adjoint[i];
  if(activity&&!active[i]){adjoint[i]=0;continue;}
  if(!isfinite(g))return NAN;if(g==0)continue;
  switch(op){
   case 0:dv+=g;break;case 1:break;case 2:grad[gradient_base+param]+=g;break;
   case 3:adjoint[a]+=g;adjoint[b]+=g;break;case 4:adjoint[a]+=g;adjoint[b]-=g;break;
   case 5:adjoint[a]+=g*value[b];adjoint[b]+=g*value[a];break;
   case 6:adjoint[a]+=g/value[b];adjoint[b]-=g*value[a]/(value[b]*value[b]);break;
   case 7:adjoint[a]-=g;break;case 8:adjoint[a]+=g*value[i];break;
   case 9:adjoint[a]+=g/value[a];break;case 10:adjoint[a]+=g*(1-value[i]*value[i]);break;
   case 11:adjoint[a]+=g/(2*value[i]);break;case 12:adjoint[a]+=g*cos(value[a]);break;
   case 13:adjoint[a]-=g*sin(value[a]);break;
   case 52:adjoint[a]+=g*atlas_dynamic_math_grad(param,value[a]);break;
   case 14:if(c!=0){float z=pow(abs(value[a]),c-1);if(value[a]<0&&fmod(abs(c-1),2.0f)==1)z=-z;adjoint[a]+=g*c*z;}break;
  }
 }
 return dv;
}

// The contextual scalar ABI uses the shared native interpreter with one state,
// while retaining the scalar schedule and public tape/result layout.
float atlas_state_equation(device const ulong* m,device const float* p,
 device const float* w,ulong header,ulong neuron,float time,ulong noise_base,thread const float* states,device float* grad,
 ulong gradient_base,float seed,bool backward,thread float* state_adjoint,device float* cache,bool cache_store);
float atlas_dynamic_equation_scored(device const ulong* m,device const float* p,device const float* weights,
 ulong header,ulong parameter_index,float time,ulong exact_time,ulong clocks,ulong noise,
 thread const float* state,device float* gradient,ulong gradient_base,float seed,bool backward,
 thread float* state_gradient,thread bool& valid,float score_loss,bool score_enabled,
 thread uint& scored,thread uint& needed_mask,thread const int* sources,ulong context_count,
 uint activity,device float* cache,bool cache_store);
int atlas_dynamic_int(float x);
float atlas_scalar_equation(device const ulong* m,device const float* p,device const float* w,
 ulong layer,ulong neuron,ulong batch,ulong tick,float voltage,device float* grad,
 ulong gradient_base,float seed,bool backward,device float* cache) {
 if(!m[79])return atlas_equation(m,p,w,layer,voltage,grad,gradient_base,seed,backward);
 float states[1]={voltage},adjoints[1]={0};
 float time=m[82]?p[m[82]+tick]:0;
 ulong noise=m[81]?m[m[81]]+(batch*m[1]+tick)*m[m[81]+1]+m[m[81]+3+2*layer]+neuron*m[m[81]+2+2*layer]:0;
 float result=atlas_state_equation(m,p,w,m[11]+3*layer,neuron,time,noise,states,
     grad,gradient_base,seed,backward,adjoints,cache,!backward);
 return backward?(isfinite(result)?adjoints[0]:NAN):result;
}

// v2: explicit ordered projection graph with tied parameter indices. Thresholds
// precede all edges; every edge VJP precedes every neuron VJP, including cycles.
kernel void atlas_graph_bptt(
 device const ulong* m [[buffer(0)]], device const float* p [[buffer(1)]],
 device const float* x [[buffer(2)]], device const float* w [[buffer(3)]],
 device const float* initial [[buffer(4)]], device const ulong* labels [[buffer(5)]],
 device float* pre [[buffer(6)]], device float* spike [[buffer(7)]],
 device float* membrane [[buffer(8)]], device float* grad [[buffer(9)]],
 device float* initial_grad [[buffer(10)]], device float* logits [[buffer(11)]],
 device float* losses [[buffer(12)]], device float* carry [[buffer(13)]],
 device float* ds [[buffer(14)]], device float* previous [[buffer(15)]], uint b [[thread_position_in_grid]]) {
 ulong B=m[0],T=m[1],L=m[2],I=m[3],N=m[4],E=m[5],C=m[16+L],G=m[10];
 if(b>=B)return;
 ulong base=ulong(b)*N,cells=B*T*N;
 for(ulong k=0;k<N;k++){membrane[base+k]=initial[base+k];carry[base+k]=0;}
 for(ulong e=0;e<E;e++)grad[ulong(b)*E+e]=0;
 for(ulong t=0;t<T;t++){
  ulong at=(ulong(b)*T+t)*N;
  for(ulong l=0;l<L;l++)for(ulong j=0;j<m[17+l];j++){
   ulong k=m[40+l]+j;
   if(m[11])pre[2*cells+at+k]=membrane[base+k];
   float u=m[11]?atlas_scalar_equation(m,p,w,l,j,b,t,membrane[base+k],grad,ulong(b)*E,0,false,pre):p[l]*membrane[base+k];
   ulong threshold_id=m[14]?m[m[14]+l]:0;float theta=threshold_id?w[threshold_id-1]:p[L+l];
   float s=float(u>theta);
   pre[at+k]=u;spike[at+k]=s;membrane[base+k]=m[6]?u:u-theta*s;
  }
  for(ulong e=0;e<G;e++){
   ulong q=84+4*e,source=m[q+1],target=m[q+2],param=m[q+3];
   float input=m[q]?x[(ulong(b)*T+t)*I+source]:spike[at+source];
   membrane[base+target]+=w[param]*input;
  }
  for(ulong k=0;k<N;k++){
   pre[cells+at+k]=membrane[base+k];
   if(m[6])membrane[base+k]*=1-spike[at+k];
  }
 }
 float scale=p[2*L+2];
 for(ulong j=0;j<C;j++){
  float sum=0;for(ulong t=0;t<T;t++)sum+=spike[(ulong(b)*T+t)*N+m[39+L]+j]*scale/float(T);
  logits[ulong(b)*C+j]=sum;
 }
 float maxlog=-INFINITY;for(ulong j=0;j<C;j++)maxlog=max(maxlog,logits[ulong(b)*C+j]);
 float denom=0;for(ulong j=0;j<C;j++)denom+=exp(logits[ulong(b)*C+j]-maxlog);
 losses[b]=maxlog+log(denom)-logits[ulong(b)*C+labels[b]];
 if(m[9])for(ulong t=T;t-->0;){
  ulong at=(ulong(b)*T+t)*N;
  for(ulong k=0;k<N;k++){ds[base+k]=0;previous[base+k]=0;}
  for(ulong j=0;j<C;j++){
   float d=(exp(logits[ulong(b)*C+j]-maxlog)/denom-float(j==labels[b]))/float(B);
   ds[base+m[39+L]+j]=d*scale/float(T);
  }
  for(ulong e=0;e<G;e++){
   ulong q=84+4*e,source=m[q+1],target=m[q+2],param=m[q+3];
   float g=carry[base+target]*(m[6]?1-spike[at+target]:1);
   float input=m[q]?x[(ulong(b)*T+t)*I+source]:spike[at+source];
   grad[ulong(b)*E+param]+=g*input;
   if(!m[q])ds[base+source]+=g*w[param];
  }
  for(ulong l=0;l<L;l++)for(ulong j=0;j<m[17+l];j++){
   ulong k=m[40+l]+j;float u=pre[at+k],s=spike[at+k];
   ulong threshold_id=m[14]?m[m[14]+l]:0;float theta=threshold_id?w[threshold_id-1]:p[L+l];
   float q=1+p[2*L]*abs(u-theta),phi=p[2*L+1]/(q*q);
   float derivative=m[6]?(1-s-(m[7]?0:pre[cells+at+k]*phi)):(1-(m[7]?0:theta*phi));
   if(threshold_id){
    float reset_theta=m[6]?(m[7]?0:pre[cells+at+k]*phi):(-s+(m[7]?0:theta*phi));
    grad[ulong(b)*E+threshold_id-1]+=carry[base+k]*reset_theta-ds[base+k]*phi;
   }
   float du=carry[base+k]*derivative+ds[base+k]*phi;
   previous[base+k]=m[11]?atlas_scalar_equation(m,p,w,l,j,b,t,pre[2*cells+at+k],grad,ulong(b)*E,du,true,pre):p[l]*du;
  }
  for(ulong k=0;k<N;k++)carry[base+k]=(t>0&&t%m[8]==0)?0:previous[base+k];
 }
 for(ulong k=0;k<N;k++)initial_grad[base+k]=carry[base+k];
}
