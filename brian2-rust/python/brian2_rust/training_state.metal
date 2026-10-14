int atlas_dynamic_int(float x);
float atlas_dynamic_equation_scored(device const ulong* m,device const float* p,device const float* weights,
 ulong header,ulong parameter_index,float time,ulong exact_time,ulong clocks,ulong noise,
 thread const float* state,device float* gradient,ulong gradient_base,float seed,bool backward,
 thread float* state_gradient,thread bool& valid,float score_loss,bool score_enabled,
 thread uint& scored,thread uint& needed_mask,thread const int* sources,ulong context_count,
 uint activity,device float* cache,bool cache_store);
// v4 typed locals share v5's exact integer payload and lazy expression evaluator.
float atlas_dynamic_equation(device const ulong* m,device const float* p,device const float* weights,
 ulong header,ulong parameter_index,float time,ulong exact_time,ulong clocks,ulong noise,
 thread const float* state,device float* gradient,ulong gradient_base,
 float seed,bool backward,thread float* state_gradient,thread bool& valid);
// Coupled state-vector SSA and phased GPU BPTT (Metal/CUDA shared source).
float atlas_state_equation(device const ulong* m,device const float* p,
 device const float* w,ulong header,ulong neuron,float time,ulong noise_base,thread const float* states,device float* grad,
 ulong gradient_base,float seed,bool backward,thread float* state_adjoint,device float* cache,bool cache_store) {
 ulong count=m[header]&255,start=m[header+1],constants=m[header+2];
 if(m[header]&256){
  bool valid=false;uint scored=0,needed=0;int sources[1]={0};
  float result=atlas_dynamic_equation_scored(m,p,w,header,neuron,time,0,0,noise_base,states,grad,
   gradient_base,seed,backward,state_adjoint,valid,0,false,scored,needed,sources,0,2,cache,cache_store);
  return valid?result:NAN;
 }
 for(ulong i=0;i<count;i++)if(m[start+4*i]>=20){
  bool valid=false;
  float result=atlas_dynamic_equation(m,p,w,header,neuron,time,0,0,noise_base,states,grad,
                                     gradient_base,seed,backward,state_adjoint,valid);
  return valid?result:NAN;
 }
 float value[128],adjoint[128];bool active[128];bool activity=(m[9]&2)!=0;
 for(ulong i=0;i<count;i++){
  ulong q=start+4*i,op=m[q],a=m[q+1],b=m[q+2],param=m[q+3];float c=p[constants+i],z=0;
  switch(op){
   case 0:z=states[0];break;case 15:z=states[a];break;case 1:z=c;break;case 2:z=w[param];break;
   case 19:z=p[noise_base+a];break;case 18:z=time;break;case 16:z=float(states[a]==0);break;case 17:z=w[param+neuron];break;
   case 3:z=value[a]+value[b];break;case 4:z=value[a]-value[b];break;
   case 5:z=value[a]*value[b];break;case 6:z=value[a]/value[b];break;
   case 7:z=-value[a];break;case 8:z=exp(value[a]);break;case 9:z=log(value[a]);break;
   case 10:z=tanh(value[a]);break;case 11:z=sqrt(value[a]);break;
   case 12:z=sin(value[a]);break;case 13:z=cos(value[a]);break;
   case 14:z=c==0?1:pow(abs(value[a]),c);if(value[a]<0){if(floor(c)!=c)return NAN;if(fmod(abs(c),2.0f)==1)z=-z;}break;
  }
  if(!isfinite(z))return NAN;value[i]=z;adjoint[i]=0;
  active[i]=false;
  if(activity)switch(op){
   case 0:case 15:active[i]=true;break;
   case 2:active[i]=p[m[13]-m[5]+param]!=0;break;
   case 17:active[i]=p[m[13]-m[5]+param+neuron]!=0;break;
   case 3:case 4:case 5:case 6:active[i]=active[a]||active[b];break;
   case 7:case 8:case 9:case 10:case 11:case 12:case 13:case 14:case 52:active[i]=active[a];break;
  }
 }
 if(!backward)return value[count-1];
 adjoint[count-1]=seed;
 for(ulong i=count;i-->0;){
  ulong q=start+4*i,op=m[q],a=m[q+1],b=m[q+2],param=m[q+3];float c=p[constants+i],g=adjoint[i];
  if(activity&&!active[i]){adjoint[i]=0;continue;}
  if(!isfinite(g))return NAN;if(g==0)continue;
  switch(op){
   case 0:state_adjoint[0]+=g;break;case 15:state_adjoint[a]+=g;break;case 1:break;case 2:grad[gradient_base+param]+=g;break;
   case 19:break;case 18:break;case 16:break; // refractory activity is a detached discrete gate
   case 17:grad[gradient_base+param+neuron]+=g;break;
   case 3:adjoint[a]+=g;adjoint[b]+=g;break;case 4:adjoint[a]+=g;adjoint[b]-=g;break;
   case 5:adjoint[a]+=g*value[b];adjoint[b]+=g*value[a];break;
   case 6:adjoint[a]+=g/value[b];adjoint[b]-=g*value[a]/(value[b]*value[b]);break;
   case 7:adjoint[a]-=g;break;case 8:adjoint[a]+=g*value[i];break;
   case 9:adjoint[a]+=g/value[a];break;case 10:adjoint[a]+=g*(1-value[i]*value[i]);break;
   case 11:adjoint[a]+=g/(2*value[i]);break;case 12:adjoint[a]+=g*cos(value[a]);break;
   case 13:adjoint[a]-=g*sin(value[a]);break;
   case 14:if(c!=0){float z=pow(abs(value[a]),c-1);if(value[a]<0&&fmod(abs(c-1),2.0f)==1)z=-z;adjoint[a]+=g*c*z;}break;
  }
 }
 return value[count-1];
}

kernel void atlas_state_phase(
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
 ulong H=m[83],M=m[H],base=ulong(b)*M,cells=B*T*M;
 ulong ranks=m[m[15]],rank=m[m[15]+1],phase=m[m[15]+2],t=m[m[15]+3];
 float time=m[82]?p[m[82]+t]:0;
 ulong at=(ulong(b)*T+t)*M,nt=(ulong(b)*T+t)*N;
 if(phase==0){
  if(m[80]){
   ulong h=m[80],begin=m[h+1]+ulong(b)*m[h+2],end=begin+m[h+2];
   for(ulong k=begin;k<end;k++)pre[k]=0;
   for(ulong j=0;j<m[h+4];j++){
    ulong q=m[h+3]+3*j,slot=ulong(uint(atlas_dynamic_int(p[q])));
    if(slot>=begin&&slot<end){pre[slot]=1;pre[slot+1]=p[q+1];pre[slot+2]=p[q+2];pre[slot+3]=0;}
   }
   if(m[h+6])for(ulong k=0;k<T*N*2;k++)pre[m[h+5]+ulong(b)*T*N*2+k]=0;
  }
  for(ulong k=0;k<M;k++){membrane[base+k]=initial[base+k];carry[base+k]=0;}
  for(ulong e=0;e<E;e++)grad[ulong(b)*E+e]=0;
  losses[b]=0;return;
 }
 if(phase==1){
  for(ulong l=0;l<L;l++){
   ulong h=H+1+6*l,S=m[h],offset=m[h+1],update=m[h+2],size=m[17+l];
   for(ulong j=0;j<size;j++){
    ulong noise_base=m[81]?m[m[81]]+(ulong(b)*T+t)*m[m[81]+1]+m[m[81]+3+2*l]+j*m[m[81]+2+2*l]:0;
    ulong k=m[40+l]+j;
    if(k*ranks/N!=rank){
     spike[nt+k]=0;for(ulong s=0;s<S;s++)membrane[base+offset+s*size+j]=0;continue;
    }
    float values[16],adjoints[16];
    for(ulong s=0;s<S;s++){
     ulong idx=offset+s*size+j;values[s]=membrane[base+idx];pre[at+idx]=values[s];adjoints[s]=0;
    }
    bool active=m[h+4]==0||values[S-1]==0;
    for(ulong s=0;s<S;s++){
     if(m[h+4]&&s==S-1){membrane[base+offset+s*size+j]=max(values[s]-1,0.0f);continue;}
     if(!active&&(m[h+5]&(1ul<<s))){membrane[base+offset+s*size+j]=values[s];continue;}
     float value=atlas_state_equation(m,p,w,update+3*s,j,time,noise_base,values,grad,ulong(b)*E,0,false,adjoints,pre,true);
     if(!isfinite(value))losses[b]=NAN;
     membrane[base+offset+s*size+j]=value;
    }
    float v=membrane[base+offset+j];
    ulong tid=m[14]?m[m[14]+k]:0;float theta=tid?w[tid-1]:p[L+l];
    pre[2*cells+nt+k]=v;spike[nt+k]=float(active&&v>theta);
   }
  }
  return;
 }
 if(phase==2){
  for(ulong e=0;e<G;e++){
   ulong q=84+4*e,source=m[q+1],target=m[q+2],param=m[q+3];
   if(target*ranks/N!=rank)continue;
   ulong l=0;while(target>=m[41+l])l++;
   ulong dst=m[H+2+6*l]+target-m[40+l];
   ulong h=H+1+6*l;
   if(m[h+4]&&(m[h+5]&1)&&(pre[at+m[h+1]+(m[h]-1)*m[17+l]+target-m[40+l]]!=0||spike[nt+target]!=0))continue;
   float input=m[q]?x[(ulong(b)*T+t)*I+source]:spike[nt+source];
   membrane[base+dst]+=w[param]*input;
  }
  for(ulong l=0;l<L;l++){
   ulong h=H+1+6*l,S=m[h],offset=m[h+1],reset=m[h+3],size=m[17+l];
   for(ulong j=0;j<size;j++){
    ulong noise_base=m[81]?m[m[81]]+(ulong(b)*T+t)*m[m[81]+1]+m[m[81]+3+2*l]+j*m[m[81]+2+2*l]:0;
    if(m[80])noise_base+=113;
    ulong k=m[40+l]+j;if(k*ranks/N!=rank)continue;
    float values[16],adjoints[16];
    for(ulong s=0;s<S;s++){
     ulong idx=offset+s*size+j;values[s]=membrane[base+idx];pre[cells+at+idx]=values[s];adjoints[s]=0;
     if(!isfinite(values[s]))losses[b]=NAN;
    }
    if(spike[nt+k]!=0)for(ulong s=0;s<S;s++){
     if(m[h+4]&&s==S-1){membrane[base+offset+s*size+j]=float(m[h+4]>1?m[h+4]-2:0);continue;}
     float value=atlas_state_equation(m,p,w,reset+3*s,j,time,noise_base,values,grad,ulong(b)*E,0,false,adjoints,pre,true);
     if(!isfinite(value))losses[b]=NAN;
     membrane[base+offset+s*size+j]=value;
    }
   }
  }
  return;
 }
 if(phase==3){
  float scale=p[2*L+2];
  for(ulong j=0;j<C;j++){
   float sum=0;for(ulong tick=0;tick<T;tick++)sum+=spike[(ulong(b)*T+tick)*N+m[39+L]+j]*scale/float(T);
   logits[ulong(b)*C+j]=sum;
  }
  float maxlog=-INFINITY;for(ulong j=0;j<C;j++)maxlog=max(maxlog,logits[ulong(b)*C+j]);
  float denom=0;for(ulong j=0;j<C;j++)denom+=exp(logits[ulong(b)*C+j]-maxlog);
  if(isfinite(losses[b]))losses[b]=maxlog+log(denom)-logits[ulong(b)*C+labels[b]];
  return;
 }
 if(phase==4){
  for(ulong k=0;k<M;k++){ds[base+k]=0;previous[base+k]=0;}
  float maxlog=-INFINITY;for(ulong j=0;j<C;j++)maxlog=max(maxlog,logits[ulong(b)*C+j]);
  float denom=0;for(ulong j=0;j<C;j++)denom+=exp(logits[ulong(b)*C+j]-maxlog);
  if(rank==0)for(ulong j=0;j<C;j++){
   ds[base+m[39+L]+j]=(exp(logits[ulong(b)*C+j]-maxlog)/denom-float(j==labels[b]))/float(B)*p[2*L+2]/float(T);
  }
  for(ulong l=0;l<L;l++){
   ulong h=H+1+6*l,S=m[h],offset=m[h+1],reset=m[h+3],size=m[17+l];
   for(ulong j=0;j<size;j++){
    ulong noise_base=m[81]?m[m[81]]+(ulong(b)*T+t)*m[m[81]+1]+m[m[81]+3+2*l]+j*m[m[81]+2+2*l]:0;
    if(m[80])noise_base+=113;
    ulong k=m[40+l]+j;if(k*ranks/N!=rank)continue;
    float values[16],adjoints[16];float sp=spike[nt+k];
    for(ulong s=0;s<S;s++){values[s]=pre[cells+at+offset+s*size+j];adjoints[s]=0;}
    bool active=m[h+4]==0||pre[at+offset+(S-1)*size+j]==0;
    if(m[80]&&sp!=0){
     uint scored=0,needed=0;int sources[16];for(ulong s=0;s<S;s++)sources[s]=int(offset+s*size+j);
     for(ulong s=0;s<S;s++){
      if((m[h+4]&&s==S-1)||!(m[reset+3*s]&256))continue;
      bool valid=false;atlas_dynamic_equation_scored(m,p,w,reset+3*s,j,time,0,0,noise_base,values,grad,
       ulong(b)*E,0,true,adjoints,valid,losses[b]/float(B),true,scored,needed,sources,S,2,pre,false);
      if(!valid)losses[b]=NAN;
     }
     if(m[m[80]+7])pre[m[m[80]+5]+((ulong(b)*T+t)*N+k)*2+1]=float(needed);
    }
    for(ulong s=0;s<S;s++){
     if(m[h+4]&&s==S-1)continue;
     float g=carry[base+offset+s*size+j];adjoints[s]+=g*(1-sp);
     if(g!=0&&(sp!=0||(!m[7]&&active))){
      float value=atlas_state_equation(m,p,w,reset+3*s,j,time,noise_base,values,grad,ulong(b)*E,g,sp!=0,adjoints,pre,false);
      if(!isfinite(value))losses[b]=NAN;
      if(!m[7])ds[base+k]+=g*(value-values[s]);
     }
    }
    for(ulong s=0;s<S;s++){
     if(!isfinite(adjoints[s]))losses[b]=NAN;
     previous[base+offset+s*size+j]=adjoints[s];
    }
   }
  }
  for(ulong e=0;e<G;e++){
   ulong q=84+4*e,source=m[q+1],target=m[q+2],param=m[q+3];
   if(target*ranks/N!=rank)continue;
   ulong l=0;while(target>=m[41+l])l++;
   ulong dst=m[H+2+6*l]+target-m[40+l];
   ulong h=H+1+6*l;
   if(m[h+4]&&(m[h+5]&1)&&(pre[at+m[h+1]+(m[h]-1)*m[17+l]+target-m[40+l]]!=0||spike[nt+target]!=0))continue;
   float g=previous[base+dst];
   float input=m[q]?x[(ulong(b)*T+t)*I+source]:spike[nt+source];
   grad[ulong(b)*E+param]+=g*input;
   if(!m[q])ds[base+source]+=g*w[param];
  }
  return;
 }
 if(phase==5){
  for(ulong k=0;k<M;k++)carry[base+k]=0;
  for(ulong l=0;l<L;l++){
   ulong h=H+1+6*l,S=m[h],offset=m[h+1],update=m[h+2],size=m[17+l];
   for(ulong j=0;j<size;j++){
    ulong noise_base=m[81]?m[m[81]]+(ulong(b)*T+t)*m[m[81]+1]+m[m[81]+3+2*l]+j*m[m[81]+2+2*l]:0;
    ulong k=m[40+l]+j;if(k*ranks/N!=rank)continue;
    ulong tid=m[14]?m[m[14]+k]:0;float theta=tid?w[tid-1]:p[L+l];
    float q=1+p[2*L]*abs(pre[2*cells+nt+k]-theta),phi=p[2*L+1]/(q*q),g=ds[base+k]*phi;
    bool active=m[h+4]==0||pre[at+offset+(S-1)*size+j]==0;
    if(!active)g=0;
    previous[base+offset+j]+=g;if(tid)grad[ulong(b)*E+tid-1]-=g;
    float values[16],adjoints[16];
    for(ulong s=0;s<S;s++){values[s]=pre[at+offset+s*size+j];adjoints[s]=0;}
    if(m[80]){
     uint scored=0,needed=0;int sources[16];for(ulong s=0;s<S;s++)sources[s]=int(offset+s*size+j);
     for(ulong s=0;s<S;s++){
      if((m[h+4]&&(s==S-1||(!active&&(m[h+5]&(1ul<<s)))))||!(m[update+3*s]&256))continue;
      bool valid=false;atlas_dynamic_equation_scored(m,p,w,update+3*s,j,time,0,0,noise_base,values,grad,
       ulong(b)*E,0,true,adjoints,valid,losses[b]/float(B),true,scored,needed,sources,S,2,pre,false);
      if(!valid)losses[b]=NAN;
     }
     if(m[m[80]+7])pre[m[m[80]+5]+((ulong(b)*T+t)*N+k)*2]=float(needed);
    }
    for(ulong s=0;s<S;s++){
     if(m[h+4]&&s==S-1)continue;
     if(!active&&(m[h+5]&(1ul<<s))){adjoints[s]+=previous[base+offset+s*size+j];continue;}
     float value=atlas_state_equation(m,p,w,update+3*s,j,time,noise_base,values,grad,ulong(b)*E,previous[base+offset+s*size+j],true,adjoints,pre,false);
     if(!isfinite(value))losses[b]=NAN;
    }
    for(ulong s=0;s<S;s++){
     if(!isfinite(adjoints[s]))losses[b]=NAN;
     carry[base+offset+s*size+j]=(t>0&&t%m[8]==0)?0:adjoints[s];
    }
   }
  }
  return;
 }
 if(phase==6)for(ulong k=0;k<M;k++)initial_grad[base+k]=carry[base+k];
}
